# data_collector.py
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import ta
import logging
import os

# --- MODIFICATION: Import settings from the central config file ---
from config import AGENT_TICKERS, BENCHMARK_TICKER, FEATURES_TO_CALCULATE

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
    # This function remains the same, it correctly uses FEATURES_TO_CALCULATE
    # to build the table schema from the config.
    try:
        with engine.connect() as conn:
            # Dynamically build the columns string from the config
            columns_sql = ",\n".join([f"                    {col} REAL" for col in FEATURES_TO_CALCULATE])
            
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    volume INTEGER,
                    {columns_sql},
                    UNIQUE(ticker, date)
                );
            """))
            conn.commit()
        logging.info("Table 'price_data' checked/created successfully.")
    except Exception as e:
        logging.error(f"Error creating/checking table 'price_data': {e}")
        raise

# In src/data_collector.py

def fetch_and_store_ticker_data(ticker_symbol: str, start_date_str: str, end_date_str: str):
    logging.info(f"Fetching data for {ticker_symbol}...")
    df = yf.download(ticker_symbol, start=start_date_str, end=end_date_str, progress=False, auto_adjust=True)

    # General robustness check: Ensure we have a DataFrame
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning(f"No data returned or invalid format for {ticker_symbol}. Skipping.")
        return

    df.reset_index(inplace=True)

    # --- INTEGRATED FIX: Handle tuple-based column names from yfinance ---
    # This logic comes from your older, working script.
    new_columns = []
    for col in df.columns:
        # If a column name is a tuple, take the first element. Otherwise, use the name as is.
        col_name = str(col[0]) if isinstance(col, tuple) else str(col)
        new_columns.append(col_name.lower().replace(' ', '_'))
    df.columns = new_columns
    # --- END INTEGRATED FIX ---

    # The yfinance 'auto_adjust=True' argument now handles 'adj_close' automatically,
    # so the manual 'close' column adjustment is no longer needed.
    df['ticker'] = ticker_symbol
    df['date'] = pd.to_datetime(df['date']).dt.date

    # --- Feature Calculation ---
    # Ensure required columns exist before calculating technical indicators
    if not all(c in df.columns for c in ['high', 'low', 'close', 'volume']):
        logging.error(f"Base columns (high, low, close, volume) not found for {ticker_symbol}. Cannot calculate features.")
        return

    df['daily_return'] = ta.others.daily_return(df.close)
    df['sma_10'] = ta.trend.sma_indicator(df.close, window=10)
    df['sma_50'] = ta.trend.sma_indicator(df.close, window=50)
    df['rsi'] = ta.momentum.rsi(df.close, window=14)
    macd = ta.trend.MACD(df.close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['volatility_5'] = df['daily_return'].rolling(window=5).std()
    df['volatility_20'] = df['daily_return'].rolling(window=20).std()
    df['momentum_10'] = df.close.diff(10)
    df['avg_volume_10'] = df.volume.rolling(window=10).mean()
    bollinger = ta.volatility.BollingerBands(df.close, window=20, window_dev=2)
    df['bollinger_hband'] = bollinger.bollinger_hband()
    df['bollinger_lband'] = bollinger.bollinger_lband()
    df['bollinger_mavg'] = bollinger.bollinger_mavg()
    df['atr'] = ta.volatility.average_true_range(df.high, df.low, df.close, window=14)
    
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    # Dynamically select only the columns that are in the config AND were successfully calculated
    final_columns = ['ticker', 'date', 'open', 'high', 'low', 'volume'] + FEATURES_TO_CALCULATE
    available_columns = [col for col in final_columns if col in df.columns]
    df = df[available_columns]
    
    try:
        df.to_sql('price_data', con=engine, if_exists='append', index=False)
        logging.info(f"Data for {ticker_symbol} stored successfully.")
    except Exception as e:
        logging.error(f"Error storing data for {ticker_symbol}: {e}")

def run_data_collection_job():
    create_table_if_not_exists()
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=25 * 365)
    logging.info(f"Starting data collection for tickers: {TICKERS_TO_FETCH}")
    for ticker in TICKERS_TO_FETCH:
        fetch_and_store_ticker_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    logging.info("Data collection job finished.")

if __name__ == "__main__":
    run_data_collection_job()

