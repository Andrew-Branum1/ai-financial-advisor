#!/usr/bin/env python3
"""
Simple script to check database contents
"""

import sqlite3
import pandas as pd
import os

def check_database():
    db_path = 'data/market_data.db'
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Check tables
        print("=== DATABASE TABLES ===")
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        print(tables)
        
        # Check price_data structure
        print("\n=== PRICE_DATA COLUMNS ===")
        columns = pd.read_sql_query("PRAGMA table_info(price_data)", conn)
        print(columns)
        
        # Check data counts
        print("\n=== DATA COUNTS BY TICKER ===")
        counts = pd.read_sql_query("SELECT ticker, COUNT(*) as count FROM price_data GROUP BY ticker", conn)
        print(counts)
        
        # Check date range
        print("\n=== DATE RANGE ===")
        date_range = pd.read_sql_query("SELECT MIN(date) as min_date, MAX(date) as max_date FROM price_data", conn)
        print(date_range)
        
        # Sample data
        print("\n=== SAMPLE DATA (first 5 rows) ===")
        sample = pd.read_sql_query("SELECT * FROM price_data LIMIT 5", conn)
        print(sample)
        
        # Check for specific tickers
        print("\n=== DATA FOR TARGET TICKERS ===")
        target_tickers = ['AMZN', 'JNJ', 'JPM', 'MSFT', 'NVDA', 'SPY']
        for ticker in target_tickers:
            count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM price_data WHERE ticker = '{ticker}'", conn)
            print(f"{ticker}: {count.iloc[0]['count']} rows")
        
        conn.close()
        
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    check_database() 