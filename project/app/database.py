from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import String, Integer, Numeric, DateTime, Enum as SQLEnum, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# Async connection string using aiomysql
DB_USER = "root"
DB_PASS = "Hello%402Legend"
DB_HOST = "localhost"
DB_NAME = "lstm_trading_bot"
DATABASE_URL = f"mysql+aiomysql://{DB_USER}:Hello%402Legend@{DB_HOST}/{DB_NAME}"
engine = create_async_engine(DATABASE_URL, echo=False, pool_recycle=3600)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    predicted_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    actual_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2), nullable=True)
    action_signal: Mapped[str] = mapped_column(
        SQLEnum("BUY", "SELL", "HOLD", name="action_signal_enum"), 
        nullable=False
    )

    __table_args__ = (
        Index("idx_ticker_time", "ticker", "timestamp"),
    )

class Trade(Base):
    __tablename__ = "trades"

    trade_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    action: Mapped[str] = mapped_column(SQLEnum("BUY", "SELL", name="trade_action_enum"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    execution_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    portfolio_balance: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)

    __table_args__ = (
        Index("idx_portfolio_time", "timestamp"),
    )

# Dependency for FastAPI route endpoints
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session