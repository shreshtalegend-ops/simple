import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


# A dictionary to act as our Model Registry
MODEL_REGISTRY = {}

def create_model(look_back_steps=60):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def load_all_models():
    """Called once when FastAPI starts up to cache models and scalers."""
    tickers = ["AAPL", "MSFT", "TSLA", "AMZN"]
    print("Loading AI Models into memory...")
    
    for ticker in tickers:
        try:
            # Load weights and the specific scaler for this stock
            model=create_model()
            model.load_weights(f"app/models/{ticker}_model.weights.h5")
            scaler = joblib.load(f"app/models/{ticker}_scaler.pkl")
            
            MODEL_REGISTRY[ticker] = {
                "model": model,
                "scaler": scaler
            }
            print(f"✅ {ticker} model loaded.")
        except Exception as e:
            print(f"❌ Failed to load {ticker}: {e}")

def predict_next_price(ticker: str, window_df: pd.DataFrame) -> float:
    """Takes the last 60 rows, scales them, runs inference, and un-scales the result."""
    if ticker not in MODEL_REGISTRY:
        raise ValueError(f"Model for {ticker} not loaded.")
        
    registry = MODEL_REGISTRY[ticker]
    model = registry["model"]
    scaler = registry["scaler"]
    
    # 1. Extract the raw close prices (or whatever features you trained on)
    raw_data = window_df['close'].values.reshape(-1, 1)
    
    # 2. Scale data using the original training scaler
    scaled_data = scaler.transform(raw_data)
    
    # 3. Reshape for LSTM (Batch Size, Time Steps, Features) -> (1, 60, 1)
    model_input = scaled_data.reshape(1, window_df.shape[0], 1)
    
    # 4. Predict
    scaled_prediction = model.predict(model_input, verbose=0)
    
    # 5. Inverse transform to get real-world dollar value
    real_dollar_prediction = scaler.inverse_transform(scaled_prediction)[0][0]
    
    return float(real_dollar_prediction)