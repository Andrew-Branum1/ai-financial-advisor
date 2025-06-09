# data_collector.py
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import ta # Technical Analysis library
import logging
import os

# ========== Logging Configuration ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== CONFIGURATION ==========
TICKERS = ['MSFT', 'GOOGL', 'AMZN', 'SPY']
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(PROJECT_ROOT, 'data')
DB_FILENAME = 'market_data.db'
DB_PATH = os.path.join(DB_DIR, DB_FILENAME)
DB_URI = f'sqlite:///{DB_PATH}'

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
                    bollinger_hband REAL,      -- Bollinger High Band
                    bollinger_lband REAL,      -- Bollinger Low Band
                    bollinger_mavg REAL,       -- Bollinger Moving Average
                    atr REAL,                  -- Average True Range (ATR)
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
    except Exception as e:
        logging.error(f"Failed to download data for {ticker_symbol}: {e}")
        return

    if df.empty:
        logging.warning(f"No data downloaded for {ticker_symbol} for the range {start_date_str} to {end_date_str}.")
        return

    df.reset_index(inplace=True)

    new_columns = []
    for col in df.columns:
        col_name = str(col[0]) if isinstance(col, tuple) and col[0] else str(col)
        new_columns.append(col_name.lower().replace(' ', '_'))
    df.columns = new_columns
    if 'adj_close' in df.columns:
        df['close'] = df['adj_close']

    df['ticker'] = ticker_symbol
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Ensure required columns for TA-Lib exist
    required_ta_cols = {'high', 'low', 'close', 'volume'}
    if not required_ta_cols.issubset(df.columns):
        missing_ta_cols = required_ta_cols - set(df.columns)
        logging.error(f"Missing required columns for TA indicators for {ticker_symbol}: {missing_ta_cols}. Skipping TA.")
        return # Or handle by filling with defaults if appropriate

    # Technical Indicators
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd_obj = ta.trend.MACD(df['close'])
    df['macd'] = macd_obj.macd()
    df['macd_signal'] = macd_obj.macd_signal()

    df['daily_return'] = df['close'].pct_change()
    df['volatility_5'] = df['daily_return'].rolling(window=5).std()
    df['volatility_20'] = df['daily_return'].rolling(window=20).std()
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['avg_volume_10'] = df['volume'].rolling(window=10).mean()

    # New Indicators: Bollinger Bands and ATR
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bollinger_hband'] = bollinger.bollinger_hband()
    df['bollinger_lband'] = bollinger.bollinger_lband()
    df['bollinger_mavg'] = bollinger.bollinger_mavg()
    df['atr'] = ta.volatility.average_true_range(high=df['high'], low=df['low'], close=df['close'], window=14)

    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    final_columns = [
        'ticker', 'date', 'open', 'high', 'low', 'close', 'volume',
        'sma_10', 'sma_50', 'rsi', 'macd', 'macd_signal',
        'daily_return', 'volatility_5', 'volatility_20',
        'momentum_10', 'avg_volume_10',
        'bollinger_hband', 'bollinger_lband', 'bollinger_mavg', 'atr' # Added new features
    ]
    for col in final_columns:
        if col not in df.columns:
            df[col] = 0.0 # Default for missing columns (e.g. if yfinance didn't return one)
            logging.warning(f"Column '{col}' was missing for {ticker_symbol}, added with default value 0.0.")
    df = df[final_columns]

    try:
        df.to_sql('price_data', con=engine, if_exists='append', index=False)
        logging.info(f"Data for {ticker_symbol} ({len(df)} rows) appended/skipped.")
    except Exception as e:
        logging.error(f"Error inserting data for {ticker_symbol} into database: {e}")
        logging.debug(f"Sample of problematic data for {ticker_symbol}:\n{df.head()}")

# ========== MAIN JOB EXECUTION ==========
def run_data_collection_job():
    create_table_if_not_exists() # Ensures table schema is up-to-date if run first

    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=25*365) # Approx. 15 years

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    logging.info(f"Starting data collection job for tickers: {TICKERS}")
    for ticker_symbol in TICKERS:
        fetch_and_store_ticker_data(ticker_symbol, start_date_str, end_date_str)
    logging.info("Data collection job finished.")

if __name__ == "__main__":
    run_data_collection_job()