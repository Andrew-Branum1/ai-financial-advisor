# data_collector_enhanced.py
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import ta
import logging
import numpy as np

# Import settings from the expanded universe config files
from config import AGENT_TICKERS, BENCHMARK_TICKER

# ========== Logging Configuration ==========
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========== CONFIGURATION ==========
TICKERS_TO_FETCH = sorted(list(set(AGENT_TICKERS + [BENCHMARK_TICKER])))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(PROJECT_ROOT, "data")
DB_FILENAME = "market_data.db"
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)
DB_URI = f"sqlite:///{DB_PATH}"

os.makedirs(DB_DIR, exist_ok=True)
engine = create_engine(DB_URI)


def create_table_if_not_exists():
    """Create the comprehensive price data table with all technical indicators."""
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    daily_return REAL,
                    sma_10 REAL,
                    sma_20 REAL,
                    sma_50 REAL,
                    ema_12 REAL,
                    ema_26 REAL,
                    close_vs_sma_10 REAL,
                    close_vs_sma_20 REAL,
                    close_vs_sma_50 REAL,
                    close_vs_ema_12 REAL,
                    close_vs_ema_26 REAL,
                    volatility_5 REAL,
                    volatility_10 REAL,
                    volatility_20 REAL,
                    volatility_60 REAL,
                    momentum_5 REAL,
                    momentum_10 REAL,
                    momentum_20 REAL,
                    momentum_60 REAL,
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bollinger_upper REAL,
                    bollinger_lower REAL,
                    bollinger_width REAL,
                    bollinger_position REAL,
                    volume_sma_10 REAL,
                    volume_sma_20 REAL,
                    volume_ratio REAL,
                    obv REAL,
                    atr REAL,
                    stoch_k REAL,
                    stoch_d REAL,
                    williams_r REAL,
                    cci REAL,
                    mfi REAL,
                    adx REAL,
                    trend_strength REAL,
                    volatility_ratio REAL,
                    roc_10 REAL,
                    roc_20 REAL,
                    volume_roc REAL,
                    price_efficiency REAL,
                    mean_reversion_signal REAL,
                    UNIQUE(ticker, date)
                );
            """
                )
            )
            conn.commit()
        logging.info("Comprehensive table 'price_data' created successfully.")
    except Exception as e:
        logging.error(f"Error creating table 'price_data': {e}")
        raise


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators for the given DataFrame."""
    if df.empty:
        return df

    # Ensure required columns exist
    required_cols = ["high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Missing required columns: {required_cols}")
        return df

    # Basic features
    df["daily_return"] = df["close"].pct_change()

    # Moving Averages
    df["sma_10"] = ta.trend.sma_indicator(df.close, window=10)
    df["sma_20"] = ta.trend.sma_indicator(df.close, window=20)
    df["sma_50"] = ta.trend.sma_indicator(df.close, window=50)
    df["ema_12"] = ta.trend.ema_indicator(df.close, window=12)
    df["ema_26"] = ta.trend.ema_indicator(df.close, window=26)

    # Price vs Moving Averages
    df["close_vs_sma_10"] = (df["close"] / df["sma_10"]) - 1
    df["close_vs_sma_20"] = (df["close"] / df["sma_20"]) - 1
    df["close_vs_sma_50"] = (df["close"] / df["sma_50"]) - 1
    df["close_vs_ema_12"] = (df["close"] / df["ema_12"]) - 1
    df["close_vs_ema_26"] = (df["close"] / df["ema_26"]) - 1

    # Volatility Measures
    df["volatility_5"] = df["daily_return"].rolling(window=5).std() * np.sqrt(252)
    df["volatility_10"] = df["daily_return"].rolling(window=10).std() * np.sqrt(252)
    df["volatility_20"] = df["daily_return"].rolling(window=20).std() * np.sqrt(252)
    df["volatility_60"] = df["daily_return"].rolling(window=60).std() * np.sqrt(252)

    # Momentum Indicators
    df["momentum_5"] = df.close.diff(5)
    df["momentum_10"] = df.close.diff(10)
    df["momentum_20"] = df.close.diff(20)
    df["momentum_60"] = df.close.diff(60)

    # RSI
    df["rsi"] = ta.momentum.rsi(df.close, window=14)

    # MACD
    macd = ta.trend.MACD(df.close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_histogram"] = macd.macd_diff()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df.close, window=20, window_dev=2)
    df["bollinger_upper"] = bollinger.bollinger_hband()
    df["bollinger_lower"] = bollinger.bollinger_lband()
    df["bollinger_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / df[
        "close"
    ]
    df["bollinger_position"] = (df["close"] - df["bollinger_lower"]) / (
        df["bollinger_upper"] - df["bollinger_lower"]
    )

    # Volume Indicators
    df["volume_sma_10"] = df.volume.rolling(window=10).mean()
    df["volume_sma_20"] = df.volume.rolling(window=20).mean()
    df["volume_ratio"] = df.volume / df["volume_sma_20"]
    df["obv"] = ta.volume.on_balance_volume(close=df["close"], volume=df["volume"])

    # ATR
    df["atr"] = ta.volatility.average_true_range(df.high, df.low, df.close, window=14)

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df.high, df.low, df.close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Williams %R
    df["williams_r"] = ta.momentum.williams_r(df.high, df.low, df.close)

    # CCI
    df["cci"] = ta.trend.cci(df.high, df.low, df.close)

    # MFI
    df["mfi"] = ta.volume.money_flow_index(df.high, df.low, df.close, df.volume)

    # ADX
    df["adx"] = ta.trend.adx(df.high, df.low, df.close)

    # Trend Strength
    df["trend_strength"] = np.abs(df["close_vs_sma_50"])

    # Volatility Ratio
    df["volatility_ratio"] = df["volatility_20"] / (df["volatility_60"] + 1e-8)

    # Rate of Change
    df["roc_10"] = (df["close"] / df["close"].shift(10) - 1) * 100
    df["roc_20"] = (df["close"] / df["close"].shift(20) - 1) * 100

    # Volume Rate of Change
    df["volume_roc"] = (df["volume"] / df["volume"].shift(10) - 1) * 100

    # Price Efficiency Ratio
    df["price_efficiency"] = np.abs(df["close"] - df["close"].shift(20)) / df[
        "close"
    ].rolling(window=20).apply(lambda x: np.sum(np.abs(x.diff().dropna())))

    # Mean Reversion Signal
    df["mean_reversion_signal"] = (df["close"] - df["sma_50"]) / (df["atr"] + 1e-8)

    return df


def fetch_and_store_ticker_data(
    ticker_symbol: str, start_date_str: str, end_date_str: str
):
    """Fetch and store comprehensive data for a ticker."""
    logging.info(f"Fetching comprehensive data for {ticker_symbol}...")
    df = yf.download(
        ticker_symbol,
        start=start_date_str,
        end=end_date_str,
        progress=False,
        auto_adjust=True,
    )

    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning(f"No data returned for {ticker_symbol}. Skipping.")
        return

    df.reset_index(inplace=True)

    # Handle tuple-based column names from yfinance
    new_columns = []
    for col in df.columns:
        col_name = str(col[0]) if isinstance(col, tuple) else str(col)
        new_columns.append(col_name.lower().replace(" ", "_"))
    df.columns = new_columns

    # Ensure required columns exist
    if not all(c in df.columns for c in ["high", "low", "close", "volume"]):
        logging.error(
            f"Required columns not found for {ticker_symbol}. Cannot proceed."
        )
        return

    # Calculate all technical indicators
    df = calculate_all_indicators(df)

    # Add ticker and date columns
    df["ticker"] = ticker_symbol
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Handle missing values
    df = df.dropna()  # Drop all rows with any NaNs

    try:
        df.to_sql("price_data", con=engine, if_exists="append", index=False)
        logging.info(f"Comprehensive data for {ticker_symbol} stored successfully.")
    except Exception as e:
        logging.error(f"Error storing data for {ticker_symbol}: {e}")


def run_data_collection_job():
    """Run the comprehensive data collection job for all tickers."""
    create_table_if_not_exists()
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=40 * 365)  # 40 years of data
    logging.info(f"Starting comprehensive data collection for tickers: {TICKERS_TO_FETCH}")
    print("Tickers to fetch:", TICKERS_TO_FETCH)
    successful = []
    failed = []
    for ticker in TICKERS_TO_FETCH:
        try:
            fetch_and_store_ticker_data(
                ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            # Check if ticker was added to DB
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM price_data WHERE ticker = :ticker"), {"ticker": ticker})
                count = result.scalar()
            if count > 0:
                successful.append(ticker)
            else:
                failed.append(ticker)
        except Exception as e:
            logging.error(f"Exception for {ticker}: {e}")
            failed.append(ticker)
    logging.info("Comprehensive data collection job finished.")
    print(f"\nSummary: {len(successful)} tickers succeeded, {len(failed)} failed.")
    print("Successful tickers:", successful)
    print("Failed tickers:", failed)


if __name__ == "__main__":
    run_data_collection_job()
