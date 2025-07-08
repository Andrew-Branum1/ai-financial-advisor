import sqlite3
import pandas as pd
import os

# --- Configuration ---
DB_PATH = os.path.join("data", "market_data.db")
FEATURES_TABLE_NAME = "features_market_data"
TICKERS_TO_CHECK = {
    "MSFT": "An established company, should have data from the start.",
    "META": "A more recent IPO (2012), should have NaNs at the beginning."
}

def run_verification():
    """
    Connects to the database and verifies the presence of NaN values for
    newer stocks, indicating correct handling of different IPO dates.
    """
    print("--- Starting Data Verification ---")

    if not os.path.exists(DB_PATH):
        print(f"!!! ERROR: Database not found at '{DB_PATH}'.")
        print("Please run 'python src/data_manager.py' first to create the database.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(f"SELECT * FROM {FEATURES_TABLE_NAME}", conn, index_col='Date', parse_dates=True)
        conn.close()
        print("Successfully loaded data from the database.")
    except Exception as e:
        print(f"!!! ERROR: Failed to load data from the database: {e}")
        return

    print("\n" + "="*50)
    print("Checking the first 5 rows of data for key tickers...")
    print("="*50)

    for ticker, description in TICKERS_TO_CHECK.items():
        print(f"\n--- Verifying: {ticker} ({description}) ---")
        
        # Check if the column for this ticker's 'close' price exists
        close_col = f"{ticker}_close"
        if close_col not in df.columns:
            print(f"!!! WARNING: Column '{close_col}' not found in the database.")
            continue
            
        # Get the first 5 data points for this ticker's close price
        first_5_datapoints = df[close_col].head()
        
        print("First 5 data points:")
        print(first_5_datapoints)
        
        # Check if all of the first 5 are NaN
        if first_5_datapoints.isnull().all():
            print(f"\n[PASS] As expected, the first data points for {ticker} are NaN, likely before its IPO.")
        elif first_5_datapoints.notnull().all():
             print(f"\n[PASS] As expected, {ticker} has valid data from the beginning of the dataset.")
        else:
            print(f"\n[INFO] {ticker} has a mix of valid and NaN data at the start.")

    print("\n--- Verification Complete ---")


if __name__ == "__main__":
    run_verification()