#!/usr/bin/env python3
"""
Extended Data Collector
Collects data for the extended universe of stocks.
"""

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
from typing import List
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
DB_URI = f"sqlite:///{DB_PATH}"

os.makedirs(DB_DIR, exist_ok=True)
engine = create_engine(DB_URI)

# Rate limiting to avoid API issues
RATE_LIMIT_DELAY = 0.1  # seconds between requests
BATCH_SIZE = 10  # number of tickers to process in a batch


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
                );
            """
                )
            )
            conn.commit()
        logger.info(
            "Table 'price_data' created successfully with comprehensive columns."
        )
    except Exception as e:
        logger.error(f"Error creating table 'price_data': {e}")
        raise


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive technical indicators for a stock.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with technical indicators added
    """
    if df.empty:
        return df

    try:
        # Ensure we have the required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(
                f"Missing required columns. Available: {df.columns.tolist()}"
            )
            return df

        # Basic indicators
        df["daily_return"] = df["Close"].pct_change()

        # Moving averages
        df["sma_10"] = df["Close"].rolling(window=10).mean()
        df["sma_20"] = df["Close"].rolling(window=20).mean()
        df["sma_50"] = df["Close"].rolling(window=50).mean()

        # Price vs moving averages
        df["close_vs_sma_10"] = df["Close"] / df["sma_10"] - 1
        df["close_vs_sma_20"] = df["Close"] / df["sma_20"] - 1
        df["close_vs_sma_50"] = df["Close"] / df["sma_50"] - 1

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
        df["bollinger_width"] = (
            bb.bollinger_hband() - bb.bollinger_lband()
        ) / bb.bollinger_mavg()
        df["bollinger_position"] = (df["Close"] - bb.bollinger_lband()) / (
            bb.bollinger_hband() - bb.bollinger_lband()
        )

        # Volume indicators
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

        # Trend strength (custom)
        df["trend_strength"] = abs(df["close_vs_sma_20"]) * np.sign(df["momentum_20"])

        # Volatility ratio
        df["volatility_ratio"] = df["volatility_10"] / df["volatility_60"]

        # Rate of Change
        df["roc_10"] = ta.momentum.ROCIndicator(df["Close"], window=10).roc()
        df["roc_20"] = ta.momentum.ROCIndicator(df["Close"], window=20).roc()

        # Volume ROC
        df["volume_roc"] = ta.momentum.ROCIndicator(df["Volume"], window=10).roc()

        # Price efficiency (custom)
        df["price_efficiency"] = df["momentum_20"] / (df["volatility_20"] * np.sqrt(20))

        # Mean reversion signal (custom)
        df["mean_reversion_signal"] = -df["close_vs_sma_50"] * df["rsi"] / 100

        # Clean up intermediate columns
        df = df.drop(["sma_10", "sma_20", "sma_50"], axis=1, errors="ignore")

        return df

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return df


def fetch_and_store_ticker_data(ticker: str, start_date: str, end_date: str) -> bool:
    """
    Fetch and store comprehensive data for a single ticker.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")

        # Fetch data from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return False

        # Calculate technical indicators
        df = calculate_technical_indicators(df)

        # Reset index to get date as a column
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

        # Prepare data for database
        df["ticker"] = ticker

        # Select only the columns we want to store
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

        # Filter columns that exist in the dataframe
        available_columns = [col for col in columns_to_store if col in df.columns]
        df_to_store = df[available_columns].copy()

        # Handle NaN values - Fixed deprecated method
        df_to_store = df_to_store.ffill().bfill().fillna(0)

        # Store in database - Fixed insertion method
        with engine.connect() as conn:
            # Delete existing data for this ticker in the date range
            conn.execute(
                text(
                    """
                DELETE FROM price_data 
                WHERE ticker = ? AND date BETWEEN ? AND ?
            """
                ),
                (ticker, start_date, end_date),
            )

            # Insert new data row by row to avoid SQL issues
            for _, row in df_to_store.iterrows():
                # Convert row to dictionary and handle NaN values
                row_dict = row.to_dict()
                row_dict = {
                    k: (v if pd.notna(v) else None) for k, v in row_dict.items()
                }

                # Create column names and values for INSERT
                columns = list(row_dict.keys())
                placeholders = ", ".join(["?" for _ in columns])
                values = list(row_dict.values())

                # Build and execute INSERT statement
                insert_sql = f"""
                    INSERT INTO price_data ({', '.join(columns)})
                    VALUES ({placeholders})
                """
                conn.execute(text(insert_sql), values)

            conn.commit()

        logger.info(f"Successfully stored {len(df_to_store)} records for {ticker}")
        return True

    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return False


def collect_data_for_universe(
    universe_size: str = "medium", strategy: str = "diversified"
) -> None:
    """
    Collect data for a specific universe of stocks.

    Args:
        universe_size: 'small', 'medium', 'large', or 'full'
        strategy: 'diversified', 'momentum', 'value', or 'balanced'
    """
    from config_extended_universe import get_ticker_universe

    # Get the ticker universe
    tickers = get_ticker_universe(universe_size, strategy)

    logger.info(f"Collecting data for {len(tickers)} tickers using {strategy} strategy")
    logger.info(f"Tickers: {tickers}")

    # Create table if it doesn't exist
    create_table_if_not_exists()

    # Set date range
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=40 * 365)  # 40 years of data

    # Process tickers in batches
    successful_count = 0
    failed_count = 0

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        logger.info(
            f"Processing batch {i//BATCH_SIZE + 1}/{(len(tickers) + BATCH_SIZE - 1)//BATCH_SIZE}"
        )

        for ticker in batch:
            try:
                success = fetch_and_store_ticker_data(
                    ticker,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
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

        # Longer delay between batches
        time.sleep(RATE_LIMIT_DELAY * 2)

    logger.info(f"Data collection completed!")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(
        f"Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%"
    )


def collect_data_for_sectors(sectors: List[str] = None) -> None:
    """
    Collect data for specific sectors.

    Args:
        sectors: List of sector names to collect data for
    """
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

    # Create table if it doesn't exist
    create_table_if_not_exists()

    # Set date range
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=40 * 365)  # 40 years of data

    # Process tickers
    successful_count = 0
    failed_count = 0

    for i, ticker in enumerate(all_tickers):
        try:
            logger.info(f"Processing {ticker} ({i+1}/{len(all_tickers)})")

            success = fetch_and_store_ticker_data(
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
    """Main function to run data collection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect market data for extended universe"
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
