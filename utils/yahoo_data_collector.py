import yfinance as yf
import sqlite3
import pandas as pd

ASSETS = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD"]
START = "2015-01-01"
END = "2024-01-01"

def fetch_data(assets=ASSETS, start=START, end=END):
    all_data = []

    for asset in assets:
        df = yf.download(asset, start=start, end=end, auto_adjust=True)

        if df.empty:
            print(f"❌ No data for {asset}, skipping.")
            continue

        # Use 'Close' explicitly
        if "Close" not in df.columns:
            print(f"❌ No 'Close' price for {asset}, skipping.")
            continue

        df = df[["Close"]].rename(columns={"Close": asset})
        all_data.append(df)

    if not all_data:
        raise ValueError("No valid data pulled.")

    # Merge on index (Date)
    merged = pd.concat(all_data, axis=1).dropna()
    return merged


def save_to_db(df, db_path="data/market_data.db", table_name="market_data"):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

if __name__ == "__main__":
    df = fetch_data()
    save_to_db(df)
    print("✅ Data pulled and stored successfully.")
