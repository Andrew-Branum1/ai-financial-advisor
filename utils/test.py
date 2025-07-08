from collector import MarketDataCollector

if __name__ == "__main__":
    collector = MarketDataCollector(
        tickers=["AAPL", "MSFT", "SPY"], 
        start="2022-01-01", 
        end="2023-01-01"
    )
    collector.fetch()
