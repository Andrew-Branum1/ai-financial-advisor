from utils.data_loader import load_market_data
df = load_market_data()
print(df.isna().sum())
print(df.head(10))
