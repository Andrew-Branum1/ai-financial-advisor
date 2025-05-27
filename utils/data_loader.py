# utils/data_loader.py
import pandas as pd
import sqlite3

def load_long_market_data(
    db_path="data/market_data.db",
    tickers=None,
    indicators=None,
    start_date=None,
    end_date=None
):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM price_data"
    filters = []

    if tickers:
        tickers_str = ",".join(f"'{t}'" for t in tickers)
        filters.append(f"ticker IN ({tickers_str})")

    if start_date:
        filters.append(f"date >= '{start_date}'")
    if end_date:
        filters.append(f"date <= '{end_date}'")

    if filters:
        query += " WHERE " + " AND ".join(filters)

    df = pd.read_sql(query, conn, parse_dates=["date"])
    conn.close()

    if indicators:
        columns = ['ticker', 'date'] + indicators
        df = df[columns]

    return df


