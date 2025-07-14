# src/data_manager.py
import sys
import os
# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
import yfinance as yf
import numpy as np
# CORRECTED: Import names now match your config.py file
from config import MODEL_CONFIGS, ALL_FEATURES, BENCHMARK_TICKER, ALL_TICKERS

# --- Configuration ---
DB_PATH = os.path.join("data", "market_data.db")
RAW_TABLE_NAME = "raw_market_data"
FEATURES_TABLE_NAME = "features_market_data"

def get_all_tickers():
    """Gets a unique, sorted list of all tickers from the global config."""
    # This now correctly uses the ALL_TICKERS list from your config
    return sorted(list(set(ALL_TICKERS)))

def initialize_database():
    """Creates the database and tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    print(f"Database created/connected at {DB_PATH}")
    conn.close()

def fetch_and_store_raw_data(tickers, start_date="2000-01-01", end_date="2024-12-31"):
    """
    Downloads raw OHLCV data from Yahoo Finance and stores it in the database.
    """
    print(f"Fetching raw data for {len(tickers)} tickers...")
    data = yf.download(
        " ".join(tickers),
        start=start_date,
        end=end_date,
        group_by='ticker'
    )
    
    all_dfs = []
    for ticker in tickers:
        # Handle cases where yfinance returns a multi-level column index or a single-level one
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data and not data[ticker].empty:
                ticker_df = data[ticker].copy()
                ticker_df['ticker'] = ticker
                all_dfs.append(ticker_df)
        else: # Single ticker download
             if not data.empty:
                data['ticker'] = ticker
                all_dfs.append(data)
                break # Exit loop since it was a single download

    if not all_dfs:
        print("Could not fetch any raw data. Exiting.")
        return

    combined_df = pd.concat(all_dfs)
    combined_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Adj Close": "adj_close"}, inplace=True)
    combined_df.dropna(subset=['close'], inplace=True)
    
    conn = sqlite3.connect(DB_PATH)
    try:
        combined_df.to_sql(RAW_TABLE_NAME, conn, if_exists='replace', index=True)
        print(f"Successfully stored raw data for {len(tickers)} tickers in table '{RAW_TABLE_NAME}'.")
    finally:
        conn.close()

def calculate_and_store_features():
    """
    Loads raw data, calculates all features manually using pandas to match the
    original training configuration, and stores them.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        raw_df = pd.read_sql(f"SELECT * FROM {RAW_TABLE_NAME}", conn, index_col='Date', parse_dates=True)
        
        print("Calculating all financial features to match training setup...")
        
        all_features_list = []
        tickers = raw_df['ticker'].unique()

        for ticker in tickers:
            df = raw_df[raw_df['ticker'] == ticker].copy().sort_index()
            
            # Skip tickers with no data
            if df.empty:
                continue
            
            # --- Manual Feature Calculation using only Pandas ---

            # 1. RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean() # com=13 for period=14
            loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
            rs = gain / (loss + 1e-9) 
            df['rsi'] = 100 - (100 / (1 + rs))

            # 2. MACD (Moving Average Convergence Divergence)
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # 3. Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            df['bollinger_high'] = sma_20 + (std_20 * 2)
            df['bollinger_low'] = sma_20 - (std_20 * 2)
            df['bollinger_width'] = (df['bollinger_high'] - df['bollinger_low']) / (sma_20 + 1e-9)
            df['bollinger_position'] = (df['close'] - df['bollinger_low']) / ((df['bollinger_high'] - df['bollinger_low']) + 1e-9)

            # 4. OBV (On-Balance Volume)
            obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv'] = obv

            # 5. ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
            df['atr'] = tr.ewm(com=13, adjust=False).mean()

            # 6. Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / ((high_14 - low_14) + 1e-9))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # 7. MFI (Money Flow Index)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            mf_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
            signed_mf = money_flow * mf_sign
            positive_mf = pd.Series(np.where(signed_mf > 0, signed_mf, 0), index=df.index)
            negative_mf = pd.Series(np.where(signed_mf < 0, -signed_mf, 0), index=df.index)
            mf_ratio = positive_mf.rolling(14).sum() / (negative_mf.rolling(14).sum() + 1e-9)
            df['mfi'] = 100 - (100 / (1 + mf_ratio))

            # 8. Custom features from ALL_FEATURES list
            df['daily_return'] = df['close'].pct_change()
            df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
            df['close_vs_sma_50'] = df['close'] / (df['close'].rolling(50).mean() + 1e-9)
            df['volatility_20'] = df['daily_return'].rolling(20).std()

            all_features_list.append(df)

        combined_df = pd.concat(all_features_list)
        print("Feature calculation complete.")
        
        # Pivot to wide format for the RL environment
        pivot_df = combined_df.pivot(columns='ticker')
        # Create clear column names like 'AAPL_close', 'MSFT_rsi'
        pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
        
        # Forward-fill any remaining NaNs after pivoting
        pivot_df.ffill(inplace=True)

        pivot_df.to_sql(FEATURES_TABLE_NAME, conn, if_exists='replace', index=True)
        print(f"Successfully stored all features in table '{FEATURES_TABLE_NAME}'.")
        
    finally:
        conn.close()

def load_market_data_from_db():
    """Loads the final, feature-rich data from the database."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Please run 'python src/data_manager.py' first.")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(f"SELECT * FROM {FEATURES_TABLE_NAME}", conn, index_col='Date', parse_dates=True)
        print("Feature data loaded successfully from DB.")
        return df
    finally:
        conn.close()

if __name__ == "__main__":
    print("--- Starting Data Management ---")
    initialize_database()
    tickers = get_all_tickers()
    fetch_and_store_raw_data(tickers)
    calculate_and_store_features()
    print("\n--- Data Management Complete ---")
    df = load_market_data_from_db()
    print("\nTest Load:")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
