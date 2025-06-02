# data_collector.py
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import ta
import logging
import os

# ========== Logging Configuration ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== CONFIGURATION ==========
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY'] # Added SPY for benchmark
# Database path relative to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(PROJECT_ROOT, 'data')
DB_FILENAME = 'market_data.db'
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)
DB_URI = f'sqlite:///{DB_PATH}'

# Ensure the data directory exists
os.makedirs(DB_DIR, exist_ok=True)
logging.info(f"Database will be located at: {DB_PATH}")
engine = create_engine(DB_URI)

# ========== CREATE TABLE IF NOT EXISTS ==========
def create_table_if_not_exists():
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    sma_10 REAL,
                    sma_50 REAL,
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    daily_return REAL,
                    volatility_5 REAL,
                    volatility_20 REAL,
                    momentum_10 REAL,
                    avg_volume_10 REAL,
                    UNIQUE(ticker, date)
                );
            """))
            conn.commit()
        logging.info("Table 'price_data' checked/created successfully.")
    except Exception as e:
        logging.error(f"Error creating/checking table 'price_data': {e}")
        raise

# ========== FUNCTION TO FETCH AND PROCESS ==========
def fetch_and_store_ticker_data(ticker_symbol: str, start_date_str: str, end_date_str: str):
    logging.info(f"Fetching data for {ticker_symbol} from {start_date_str} to {end_date_str}...")
    try:
        df = yf.download(ticker_symbol, start=start_date_str, end=end_date_str, progress=False)
    except Exception as e: # More specific exceptions could be caught
        logging.error(f"Failed to download data for {ticker_symbol}: {e}")
        return

    if df.empty:
        logging.warning(f"No data downloaded for {ticker_symbol} for the range {start_date_str} to {end_date_str}.")
        return

    df.reset_index(inplace=True) # Date becomes a column

    # Normalize column names
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            # If it's a tuple, take the first element, make it lowercase, and replace spaces
            # This is a common case if yfinance returns a MultiIndex, e.g., ('Close', '')
            col_name = str(col[0]) if col[0] else '' # Ensure it's a string
        else:
            col_name = str(col) # Ensure it's a string
        new_columns.append(col_name.lower().replace(' ', '_'))
    df.columns = new_columns
    if 'adj_close' in df.columns: # Prefer adjusted close
        df['close'] = df['adj_close']

    df['ticker'] = ticker_symbol
    # Ensure 'date' column is datetime.date objects for SQLite compatibility
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Technical Indicators (apply fillna after all calculations or per indicator)
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd_obj = ta.trend.MACD(df['close'])
    df['macd'] = macd_obj.macd()
    df['macd_signal'] = macd_obj.macd_signal()

    df['daily_return'] = df['close'].pct_change() # First value will be NaN
    df['volatility_5'] = df['daily_return'].rolling(window=5).std()
    df['volatility_20'] = df['daily_return'].rolling(window=20).std()
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['avg_volume_10'] = df['volume'].rolling(window=10).mean()

    # Handle NaNs created by TA functions (especially at the beginning of the series)
    # Option 1: Fill with 0 (can be problematic for some indicators)
    # df.fillna(0, inplace=True)
    # Option 2: Backward fill then forward fill
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    # Option 3: Drop rows with any NaNs if critical, but this reduces data
    # df.dropna(inplace=True)


    # Select and order columns to match table schema
    final_columns = [
        'ticker', 'date', 'open', 'high', 'low', 'close', 'volume',
        'sma_10', 'sma_50', 'rsi', 'macd', 'macd_signal',
        'daily_return', 'volatility_5', 'volatility_20',
        'momentum_10', 'avg_volume_10'
    ]
    # Ensure all columns exist, add missing ones with default (e.g., if yf.download didn't return one)
    for col in final_columns:
        if col not in df.columns:
            df[col] = 0.0 # Or None, depending on desired default
            logging.warning(f"Column '{col}' was missing for {ticker_symbol}, added with default value.")
    df = df[final_columns]


    try:
        # Using 'append' relies on the UNIQUE constraint to prevent exact duplicates.
        # It will not update existing rows if data for a (ticker, date) changes.
        # For full refresh of a period, consider deleting data for that period first.
        df.to_sql('price_data', con=engine, if_exists='append', index=False)
        logging.info(f"Data for {ticker_symbol} ( {len(df)} rows) appended/skipped due to UNIQUE constraint.")
    except Exception as e:
        logging.error(f"Error inserting data for {ticker_symbol} into database: {e}")
        logging.debug(f"Sample of problematic data for {ticker_symbol}:\n{df.head()}")

# ========== MAIN JOB EXECUTION ==========
def run_data_collection_job():
    create_table_if_not_exists()

    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=5*365) # Fetch approx. 5 years of data

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    logging.info(f"Starting data collection job for tickers: {TICKERS}")
    for ticker_symbol in TICKERS:
        fetch_and_store_ticker_data(ticker_symbol, start_date_str, end_date_str)
    logging.info("Data collection job finished.")

if __name__ == "__main__":
    run_data_collection_job()