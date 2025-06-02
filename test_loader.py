# test_loader.py
import logging
from src.utils import load_market_data_from_db # Make sure src.utils is correct

logging.basicConfig(level=logging.INFO)

test_tickers = ['AAPL', 'MSFT'] # Choose a subset of your collected tickers
df = load_market_data_from_db(
    tickers_list=test_tickers,
    value_column='close', # Ensure this column exists in your DB
    min_data_points=50
)

if not df.empty:
    print("\n--- Data Loaded by load_market_data_from_db ---")
    print("Head:\n", df.head())
    print("\nInfo:")
    df.info()
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Index name: {df.index.name}")

    # Basic Assertions
    assert all(ticker in df.columns for ticker in test_tickers), "Not all requested tickers are columns."
    assert df.index.name == 'date', "DataFrame index is not 'date'."
    assert not df.isnull().values.any(), "NaNs found in the DataFrame after loading."
    print("\n✅ Basic assertions passed for loaded data.")
else:
    print("\n❌ Failed to load data or DataFrame is empty.")