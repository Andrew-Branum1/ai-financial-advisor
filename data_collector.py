import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import ta

# ========== CONFIGURATION ==========
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
DB_PATH = 'sqlite:////app/data/market_data.db'  # 4 slashes for absolute path
engine = create_engine(DB_PATH)

# ========== CREATE TABLE IF NOT EXISTS ==========
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            sma_10 REAL,
            sma_50 REAL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            daily_return REAL,
            volatility_5 REAL,
            volatility_20 REAL,
            momentum_10 REAL,
            avg_volume_10 REAL,
            UNIQUE(ticker, date)
        );
    """))

# ========== FUNCTION TO FETCH AND PROCESS ==========
def fetch_and_store(ticker):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=60)  # more days for indicators

    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        print(f"No data for {ticker}")
        return

    df.reset_index(inplace=True)

    # Normalize and flatten column names
    df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]

    # Add the ticker column
    df['ticker'] = ticker

    # ========== Add Technical Indicators ==========
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd_indicator = ta.trend.MACD(df['close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()


    # ========== Add Rolling Statistical Features ==========
    df['daily_return'] = df['close'].pct_change()
    df['volatility_5'] = df['daily_return'].rolling(window=5).std()
    df['volatility_20'] = df['daily_return'].rolling(window=20).std()
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['avg_volume_10'] = df['volume'].rolling(window=10).mean()

    # ========== Finalize Columns ==========
    df = df[[
        'ticker', 'date', 'open', 'high', 'low', 'close', 'volume',
        'sma_10', 'sma_50', 'rsi', 'macd', 'macd_signal',
        'daily_return', 'volatility_5', 'volatility_20',
        'momentum_10', 'avg_volume_10'
    ]]

    try:
        df.to_sql('price_data', con=engine, if_exists='append', index=False)
        print(f"✅ Stored data for {ticker}")
    except Exception as e:
        print(f"❌ Error inserting {ticker}: {e}")

# ========== RUN JOB ==========
def run_job():
    for ticker in TICKERS:
        fetch_and_store(ticker)

if __name__ == "__main__":
    run_job()
