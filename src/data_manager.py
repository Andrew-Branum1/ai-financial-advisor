import os
import sys
import sqlite3
import pandas as pd
import yfinance as yf
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALL_TICKERS

DB_PATH = os.path.join("data", "market_data.db")
RAW_TABLE = "raw_market_data"
FEATURES_TABLE = "features_market_data"


def fetch_raw_data(tickers, start="2000-01-01", end="2024-12-31"):
    print(f"Fetching raw data for {len(tickers)} tickers...")
    
    data = yf.download(" ".join(tickers), start=start, end=end, group_by='ticker')
    
    all_dfs = []
    for ticker in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data and not data[ticker].empty:
                ticker_df = data[ticker].copy()
                ticker_df['ticker'] = ticker
                all_dfs.append(ticker_df)
        elif not data.empty: 
            data['ticker'] = ticker
            all_dfs.append(data)
            break 
            
    if not all_dfs:
        print("Could not fetch any raw data. Exiting.")
        return

    # Clean up and store the raw data
    combined = pd.concat(all_dfs)
    combined.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Adj Close": "adj_close"}, inplace=True)
    combined.dropna(subset=['close'], inplace=True)
    
    with sqlite3.connect(DB_PATH) as conn:
        combined.to_sql(RAW_TABLE, conn, if_exists='replace', index=True)
        print(f"Stored raw data in table '{RAW_TABLE}'.")


def calculate_features():
    with sqlite3.connect(DB_PATH) as conn:
        raw_df = pd.read_sql(f"SELECT * FROM {RAW_TABLE}", conn, index_col='Date', parse_dates=True)
        
    #print("Calculating financial features...")
    
    all_features_list = []
    for ticker in raw_df['ticker'].unique():
        df = raw_df[raw_df['ticker'] == ticker].copy().sort_index()
        if df.empty:
            continue
            
        # Features
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        bollinger_high = sma_20 + (std_20 * 2)
        bollinger_low = sma_20 - (std_20 * 2)
        df['bollinger_position'] = (df['close'] - bollinger_low) / ((bollinger_high - bollinger_low) + 1e-9)

        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['daily_return'] = df['close'].pct_change()
        df['close_vs_sma_50'] = df['close'] / (df['close'].rolling(50).mean() + 1e-9)
        df['volatility_20'] = df['daily_return'].rolling(20).std()
        df['close_vs_sma_20'] = df['close'] / (sma_20 + 1e-9)
        df['momentum_20'] = df['close'].pct_change(20)

        all_features_list.append(df)

    # Pivot 
    features_df = pd.concat(all_features_list)
    pivot_df = features_df.pivot(columns='ticker')
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df.ffill(inplace=True)

    with sqlite3.connect(DB_PATH) as conn:
        pivot_df.to_sql(FEATURES_TABLE, conn, if_exists='replace', index=True)
        print(f"Stored all features in table '{FEATURES_TABLE}'.")
        

def load_market_data():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}.")
    
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {FEATURES_TABLE}", conn, index_col='Date', parse_dates=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    tickers = sorted(list(set(ALL_TICKERS)))
    fetch_raw_data(tickers)
    calculate_features()