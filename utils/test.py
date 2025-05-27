from utils.data_loader import load_long_market_data

# Test 1: Load all data
print("=== All Data (Head) ===")
df_all = load_long_market_data()
print(df_all.head())

# Test 2: Filter by tickers and indicators
print("\n=== AAPL + SPY with MACD/RSI ===")
df_filtered = load_long_market_data(
    tickers=["AAPL", "SPY"],
    indicators=["close", "rsi", "macd"]
)
print(df_filtered.head())

# Test 3: Add date filtering
print("\n=== AAPL from 2022 ===")
df_date = load_long_market_data(
    tickers=["AAPL"],
    start_date="2022-01-01",
    end_date="2022-12-31"
)
print(df_date.head())
