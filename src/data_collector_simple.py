#!/usr/bin/env python3
"""
Simple and Reliable Data Collector
Collects data for the extended universe of stocks using a straightforward approach.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import ta
import logging
import numpy as np
import time

# Import extended universe configuration
from config_extended_universe import SECTOR_MAPPING

# ========== Logging Configuration ==========
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(PROJECT_ROOT, "data")
DB_FILENAME = "market_data.db"
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)

os.makedirs(DB_DIR, exist_ok=True)

# Rate limiting to avoid API issues
RATE_LIMIT_DELAY = 0.1  # seconds between requests


def create_database():
    """Create the database and table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table with all necessary columns
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            volume INTEGER,
            daily_return REAL,
            close_vs_sma_10 REAL,
            close_vs_sma_20 REAL,
            close_vs_sma_50 REAL,
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
            bollinger_width REAL,
            bollinger_position REAL,
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
        )
    """
    )

    conn.commit()
    conn.close()
    logger.info("Database and table created successfully.")


def calculate_indicators(df):
    """Calculate technical indicators for a DataFrame."""
    if df.empty:
        return df

    try:
        # Basic indicators
        df["daily_return"] = df["Close"].pct_change()

        # Moving averages
        df["sma_10"] = df["Close"].rolling(window=10).mean()
        df["sma_20"] = df["Close"].rolling(window=20).mean()
        df["sma_50"] = df["Close"].rolling(window=50).mean()

        # Exponential moving averages
        df["ema_12"] = df["Close"].ewm(span=12).mean()
        df["ema_26"] = df["Close"].ewm(span=26).mean()

        # Price vs moving averages
        df["close_vs_sma_10"] = df["Close"] / df["sma_10"] - 1
        df["close_vs_sma_20"] = df["Close"] / df["sma_20"] - 1
        df["close_vs_sma_50"] = df["Close"] / df["sma_50"] - 1
        df["close_vs_ema_12"] = df["Close"] / df["ema_12"] - 1
        df["close_vs_ema_26"] = df["Close"] / df["ema_26"] - 1

        # Volatility measures
        for window in [5, 10, 20, 60]:
            df[f"volatility_{window}"] = df["daily_return"].rolling(window=window).std()

        # Momentum indicators
        for window in [5, 10, 20, 60]:
            df[f"momentum_{window}"] = df["Close"] / df["Close"].shift(window) - 1

        # RSI
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

        # MACD
        macd = ta.trend.MACD(df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["Close"])
        df["bollinger_upper"] = bb.bollinger_hband()
        df["bollinger_lower"] = bb.bollinger_lband()
        df["bollinger_width"] = (
            bb.bollinger_hband() - bb.bollinger_lband()
        ) / bb.bollinger_mavg()
        df["bollinger_position"] = (df["Close"] - bb.bollinger_lband()) / (
            bb.bollinger_hband() - bb.bollinger_lband()
        )

        # Volume indicators
        df["volume_sma_10"] = df["Volume"].rolling(window=10).mean()
        df["volume_sma_20"] = df["Volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(window=20).mean()
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(
            df["Close"], df["Volume"]
        ).on_balance_volume()

        # ATR
        df["atr"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"]
        ).average_true_range()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Williams %R
        df["williams_r"] = ta.momentum.WilliamsRIndicator(
            df["High"], df["Low"], df["Close"]
        ).williams_r()

        # CCI
        df["cci"] = ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"]).cci()

        # Money Flow Index
        df["mfi"] = ta.volume.MFIIndicator(
            df["High"], df["Low"], df["Close"], df["Volume"]
        ).money_flow_index()

        # ADX
        df["adx"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()

        # Custom indicators
        df["trend_strength"] = abs(df["close_vs_sma_20"]) * np.sign(df["momentum_20"])
        df["volatility_ratio"] = df["volatility_10"] / df["volatility_60"]
        df["roc_10"] = ta.momentum.ROCIndicator(df["Close"], window=10).roc()
        df["roc_20"] = ta.momentum.ROCIndicator(df["Close"], window=20).roc()
        df["volume_roc"] = ta.momentum.ROCIndicator(df["Volume"], window=10).roc()
        df["price_efficiency"] = df["momentum_20"] / (df["volatility_20"] * np.sqrt(20))
        df["mean_reversion_signal"] = -df["close_vs_sma_50"] * df["rsi"] / 100

        # Clean up intermediate columns
        df = df.drop(
            ["sma_10", "sma_20", "sma_50", "ema_12", "ema_26"], axis=1, errors="ignore"
        )

        return df

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df


def fetch_and_store_ticker(ticker, start_date, end_date):
    """Fetch and store data for a single ticker."""
    try:
        logger.info(f"Fetching data for {ticker}")

        # Fetch data from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return False

        # Calculate indicators
        df = calculate_indicators(df)

        # Reset index to get date as a column
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

        # Add ticker column
        df["ticker"] = ticker

        # Select columns to store
        columns_to_store = [
            "ticker",
            "Date",
            "Close",
            "Volume",
            "daily_return",
            "close_vs_sma_10",
            "close_vs_sma_20",
            "close_vs_sma_50",
            "volatility_5",
            "volatility_10",
            "volatility_20",
            "volatility_60",
            "momentum_5",
            "momentum_10",
            "momentum_20",
            "momentum_60",
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bollinger_width",
            "bollinger_position",
            "volume_ratio",
            "obv",
            "atr",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "cci",
            "mfi",
            "adx",
            "trend_strength",
            "volatility_ratio",
            "roc_10",
            "roc_20",
            "volume_roc",
            "price_efficiency",
            "mean_reversion_signal",
        ]

        # Filter available columns
        available_columns = [col for col in columns_to_store if col in df.columns]
        df_to_store = df[available_columns].copy()

        # Handle NaN values
        df_to_store = df_to_store.ffill().bfill().fillna(0)

        # Store in database using simple SQLite approach
        conn = sqlite3.connect(DB_PATH)

        # Delete existing data for this ticker
        conn.execute("DELETE FROM price_data WHERE ticker = ?", (ticker,))

        # Insert new data
        df_to_store.to_sql("price_data", conn, if_exists="append", index=False)

        conn.commit()
        conn.close()

        logger.info(f"Successfully stored {len(df_to_store)} records for {ticker}")
        return True

    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return False


def collect_data_for_universe(universe_size="medium", strategy="diversified"):
    """Collect data for a specific universe."""
    from config_extended_universe import get_ticker_universe

    # Get tickers
    tickers = get_ticker_universe(universe_size, strategy)

    logger.info(f"Collecting data for {len(tickers)} tickers using {strategy} strategy")
    logger.info(f"Tickers: {tickers}")

    # Create database
    create_database()

    # Set date range
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=10 * 365)  # 10 years of data

    # Process tickers
    successful_count = 0
    failed_count = 0

    for i, ticker in enumerate(tickers):
        try:
            logger.info(f"Processing {ticker} ({i+1}/{len(tickers)})")

            success = fetch_and_store_ticker(
                ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )

            if success:
                successful_count += 1
            else:
                failed_count += 1

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            failed_count += 1

    logger.info(f"Data collection completed!")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(
        f"Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%"
    )


def collect_data_for_sectors(sectors=None):
    """Collect data for specific sectors."""
    if sectors is None:
        sectors = list(SECTOR_MAPPING.keys())

    all_tickers = []
    for sector in sectors:
        if sector in SECTOR_MAPPING:
            all_tickers.extend(SECTOR_MAPPING[sector])

    # Remove duplicates
    all_tickers = list(set(all_tickers))

    logger.info(
        f"Collecting data for {len(all_tickers)} tickers from sectors: {sectors}"
    )

    # Create database
    create_database()

    # Set date range
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=10 * 365)  # 10 years of data

    # Process tickers
    successful_count = 0
    failed_count = 0

    for i, ticker in enumerate(all_tickers):
        try:
            logger.info(f"Processing {ticker} ({i+1}/{len(all_tickers)})")

            success = fetch_and_store_ticker(
                ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )

            if success:
                successful_count += 1
            else:
                failed_count += 1

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            failed_count += 1

    logger.info(f"Data collection completed!")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(
        f"Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%"
    )


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple data collector for extended universe"
    )
    parser.add_argument(
        "--universe-size",
        choices=["small", "medium", "large", "full"],
        default="medium",
        help="Size of the universe to collect",
    )
    parser.add_argument(
        "--strategy",
        choices=["diversified", "momentum", "value", "balanced"],
        default="diversified",
        help="Selection strategy",
    )
    parser.add_argument(
        "--sectors", nargs="+", help="Specific sectors to collect data for"
    )

    args = parser.parse_args()

    if args.sectors:
        collect_data_for_sectors(args.sectors)
    else:
        collect_data_for_universe(args.universe_size, args.strategy)


if __name__ == "__main__":
    main()
