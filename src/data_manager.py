import os
import sys
import sqlite3
import pandas as pd
import yfinance as yf
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALL_TICKERS

DB = os.path.join("data", "market_data.db")
RAW = "raw_market_data"
FEAT = "features_market_data"


def fetch_raw_data(tickers, start="2000-01-01", end="2024-12-31"):
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

    # Clean up and store the raw data
    combined = pd.concat(all_dfs)
    combined.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Adj Close": "adj_close"
        }, inplace=True)
    combined.dropna(subset=['close'], inplace=True)
    
    with sqlite3.connect(DB) as conn:
        combined.to_sql(RAW, conn, if_exists='replace', index=True)
        #print("Stored raw data")


def calculate_features():
    with sqlite3.connect(DB) as conn:
        raw_df = pd.read_sql(f"SELECT * FROM {RAW}", conn, index_col='Date', parse_dates=True)
        
    #print("Calculating financial features...")
    
    features = []
    for ticker in raw_df['ticker'].unique():
        df = raw_df[raw_df['ticker'] == ticker].copy().sort_index()
        if df.empty:
            continue
            
        # Features
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        bollinger_high = sma_20 + (std_20 * 2)
        bollinger_low = sma_20 - (std_20 * 2)
        bollinger_width = bollinger_high - bollinger_low
        df['bollinger_position'] = (df['close'] - bollinger_low) / (bollinger_width + 1e-9)

        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['daily_return'] = df['close'].pct_change()
        df['close_vs_sma_50'] = df['close'] / (df['close'].rolling(50).mean() + 1e-9)
        df['volatility_20'] = df['daily_return'].rolling(20).std()
        df['close_vs_sma_20'] = df['close'] / (sma_20 + 1e-9)
        df['momentum_20'] = df['close'].pct_change(20)

        features.append(df)

    # Pivot 
    features_df = pd.concat(features)
    pivot_df = features_df.pivot(columns='ticker')
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df.ffill(inplace=True)

    with sqlite3.connect(DB) as conn:
        pivot_df.to_sql(FEAT, conn, if_exists='replace', index=True)
        #print("Stored all features")
        

def load_market_data():
    with sqlite3.connect(DB) as conn:
        df = pd.read_sql(f"SELECT * FROM {FEAT}", conn, index_col='Date', parse_dates=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    tickers = sorted(list(set(ALL_TICKERS)))
    fetch_raw_data(tickers)
    calculate_features()