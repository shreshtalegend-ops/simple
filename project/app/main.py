import asyncio
import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import databases
from sqlalchemy import select, desc 
from database import AsyncSessionLocal, Prediction, Trade, get_db
from decimal import Decimal
from manager import manager
from sqlalchemy.ext.asyncio import AsyncSession
from inference import load_all_models, predict_next_price
from fastapi.staticfiles import StaticFiles




# --- CONFIGURATION ---
DB_USER = "root"
DB_PASS = "Hello%402Legend"
DB_HOST = "localhost"
DB_NAME = "lstm_trading_bot"
DATABASE_URL = f"mysql+aiomysql://{DB_USER}:Hello%402Legend@{DB_HOST}/{DB_NAME}"

# Initialize Async Database Client
database = databases.Database(DATABASE_URL)

# Global Simulation State
simulation_running = False
SIMULATED_PORTFOLIO_BALANCE = 100000.00 # Starting with $100k
TICK_RATE_SECONDS = 1.0 # How fast the replay engine moves

# Load CSV (Loading one ticker, e.g., AAPL, to demonstrate the loop)
print("Loading CSV data into memory...")
df_aapl = pd.read_csv("app/AAPL_1min_historical.csv")
# Sort by time just in case, and reset index
df_aapl = df_aapl.sort_values("timestamp").reset_index(drop=True)


# # --- MOCK MODEL INFERENCE ---
# def lstm_predict(window_data):
#     """
#     Replace this with your actual PyTorch/Keras model.predict() and scaler.transform().
#     For now, it returns a dummy prediction slightly above or below the last close.
#     """
#     last_close = window_data['close'].iloc[-1]
#     # Simulating a random prediction between -0.5% and +0.5%
#     noise = np.random.uniform(-0.005, 0.005)
#     return last_close * (1 + noise)


# --- THE REPLAY ENGINE ---
async def run_replay_engine(ticker: str):
    global simulation_running, SIMULATED_PORTFOLIO_BALANCE
    window_size = 60

    for i in range(window_size, len(df_aapl)):
        if not simulation_running:
            break

        window = df_aapl.iloc[i - window_size:i]
        current_row = df_aapl.iloc[i]
        current_time = pd.to_datetime(current_row['timestamp'])
        current_close = Decimal(str(round(current_row['close'], 2)))

        # LSTM Inference & Signal
        predicted_val = Decimal(str(round(float(predict_next_price(ticker, window)), 2)))
        
        action_signal = 'HOLD'
        if predicted_val > current_close * Decimal("1.002"):
            action_signal = 'BUY'
        elif predicted_val < current_close * Decimal("0.998"):
            action_signal = 'SELL'

        # Write to MySQL via SQLAlchemy AsyncSession
        async with AsyncSessionLocal() as session:
            # 1. Add Prediction Record
            pred_record = Prediction(
                timestamp=current_time,
                ticker=ticker,
                predicted_price=predicted_val,
                actual_price=current_close,
                action_signal=action_signal
            )
            session.add(pred_record)

            # 2. Add Trade Record (if applicable)
            if action_signal in ("BUY", "SELL"):
                qty = 10
                trade_cost = Decimal(qty) * current_close

                if action_signal == "BUY":
                    SIMULATED_PORTFOLIO_BALANCE -= float(trade_cost)
                else:
                    SIMULATED_PORTFOLIO_BALANCE += float(trade_cost)

                trade_record = Trade(
                    timestamp=current_time,
                    ticker=ticker,
                    action=action_signal,
                    quantity=qty,
                    execution_price=current_close,
                    portfolio_balance=Decimal(str(round(SIMULATED_PORTFOLIO_BALANCE, 2)))
                )
                session.add(trade_record)

            # Commit the transaction
            await session.commit()
            await manager.broadcast({
                "type": "tick",
                "timestamp": str(current_time),
                "ticker": ticker,
                "actual_price": float(current_close),
                "predicted_price": float(predicted_val),
                "action": action_signal,
                "portfolio_balance": float(SIMULATED_PORTFOLIO_BALANCE)
            })

        await asyncio.sleep(TICK_RATE_SECONDS)

# --- FASTAPI LIFECYCLE & ENDPOINTS ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to DB on startup
    await database.connect()
    load_all_models()
    yield
    # Disconnect on shutdown
    await database.disconnect()

app = FastAPI(lifespan=lifespan)


@app.post("/start")
async def start_simulation(background_tasks: BackgroundTasks, ticker: str = "AAPL"):
    global simulation_running, SIMULATED_PORTFOLIO_BALANCE
    if not simulation_running:
        simulation_running = True
        
        # Reset balance for a new run
        SIMULATED_PORTFOLIO_BALANCE = 100000.00 
        
        # Pass the selected ticker into the background task
        background_tasks.add_task(run_replay_engine, ticker)
        return {"status": f"Simulation started for {ticker}"}
    return {"status": "Simulation already running"}

@app.post("/stop")
async def stop_simulation():
    global simulation_running
    simulation_running = False
    return {"status": "Simulation stopped"}

@app.get("/status")
async def get_status():
    return {
        "running": simulation_running,
        "portfolio_balance": SIMULATED_PORTFOLIO_BALANCE
    }

@app.get("/api/recent_predictions")
async def get_recent_predictions(limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Fetch the latest predictions to populate the frontend table on load."""
    query = select(Prediction).order_by(desc(Prediction.timestamp)).limit(limit)
    result = await db.execute(query)
    predictions = result.scalars().all()
    
    return [
        {
            "timestamp": p.timestamp,
            "ticker": p.ticker,
            "predicted_price": p.predicted_price,
            "actual_price": p.actual_price,
            "action": p.action_signal
        }
        for p in predictions
    ]

@app.get("/api/portfolio_history")
async def get_portfolio_history(db: AsyncSession = Depends(get_db)):
    """Fetch the equity curve data for the frontend chart."""
    query = select(Trade.timestamp, Trade.portfolio_balance).order_by(Trade.timestamp)
    result = await db.execute(query)
    trades = result.all()
    
    return [{"timestamp": t.timestamp, "balance": t.portfolio_balance} for t in trades]


# --- WEBSOCKET ENDPOINT ---

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """The frontend connects here to receive live replay ticks."""
    await manager.connect(websocket)
    try:
        while True:
            # We must wait for data, even if we only plan to send.
            # This keeps the connection open and listens for client disconnects.
            data = await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected from WebSocket stream.")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)