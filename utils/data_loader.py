# utils/data_loader.py
import pandas as pd
import sqlite3

def load_market_data(db_path="data/market_data.db", table="market_data"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()

    # Fix multi-index columns (e.g., ('SPY', 'SPY')) → 'SPY'
    if isinstance(df.columns[0], tuple):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = df.drop(columns=['Date'], errors='ignore')
    return df

