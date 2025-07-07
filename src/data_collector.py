# data_collector.py
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import logging
import os

# --- MODIFICATION: Import settings from the central config file ---
from config import AGENT_TICKERS, BENCHMARK_TICKER

# ========== Logging Configuration ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== CONFIGURATION ==========
# --- MODIFICATION: Tickers are now sourced from the config file ---
TICKERS_TO_FETCH = sorted(list(set(AGENT_TICKERS + [BENCHMARK_TICKER])))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(PROJECT_ROOT, 'data')
DB_FILENAME = 'market_data.db'
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)
DB_URI = f'sqlite:///{DB_PATH}'

os.makedirs(DB_DIR, exist_ok=True)
engine = create_engine(DB_URI)

def create_table_if_not_exists():
    """Create the basic price data table with only essential columns."""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    close REAL,
                    volume INTEGER,
                    daily_return REAL,
                    UNIQUE(ticker, date)
                );
            """))
            conn.commit()
        logging.info("Table 'price_data' created successfully with basic columns.")
    except Exception as e:
        logging.error(f"Error creating table 'price_data': {e}")
        raise

def fetch_and_store_ticker_data(ticker_symbol: str, start_date_str: str, end_date_str: str):
    """Fetch and store basic price data for a ticker."""
    logging.info(f"Fetching data for {ticker_symbol}...")
    df = yf.download(ticker_symbol, start=start_date_str, end=end_date_str, progress=False, auto_adjust=True)

    # General robustness check: Ensure we have a DataFrame
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning(f"No data returned or invalid format for {ticker_symbol}. Skipping.")
        return

    df.reset_index(inplace=True)

    # --- INTEGRATED FIX: Handle tuple-based column names from yfinance ---
    new_columns = []
    for col in df.columns:
        # If a column name is a tuple, take the first element. Otherwise, use the name as is.
        col_name = str(col[0]) if isinstance(col, tuple) else str(col)
        new_columns.append(col_name.lower().replace(' ', '_'))
    df.columns = new_columns

    # Ensure required columns exist
    if not all(c in df.columns for c in ['close', 'volume']):
        logging.error(f"Required columns (close, volume) not found for {ticker_symbol}. Cannot proceed.")
        return

    # Calculate daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Add ticker and date columns
    df['ticker'] = ticker_symbol
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Select only the columns we need
    final_columns = ['ticker', 'date', 'close', 'volume', 'daily_return']
    df = df[final_columns]

    # Handle missing values
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    try:
        df.to_sql('price_data', con=engine, if_exists='append', index=False)
        logging.info(f"Basic data for {ticker_symbol} stored successfully.")
    except Exception as e:
        logging.error(f"Error storing data for {ticker_symbol}: {e}")

def run_data_collection_job():
    """Run the data collection job for all tickers."""
    create_table_if_not_exists()
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=25 * 365)  # 25 years of data
    logging.info(f"Starting basic data collection for tickers: {TICKERS_TO_FETCH}")
    for ticker in TICKERS_TO_FETCH:
        fetch_and_store_ticker_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    logging.info("Basic data collection job finished.")

if __name__ == "__main__":
    run_data_collection_job()

