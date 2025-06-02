# src/utils.py
import pandas as pd
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.db') # Assumes /data/market_data.db from project root

def load_market_data_from_db(
    tickers_list: list = None,
    value_column: str = 'close',
    start_date: str = None,
    end_date: str = None,
    db_path: str = DEFAULT_DB_PATH,
    min_data_points: int = 60 # Minimum rows after processing
):
    """
    Loads market data from the SQLite database for specified tickers,
    selects the 'value_column' (e.g., 'close'), and pivots it into a
    wide format suitable for PortfolioEnv.

    Args:
        tickers_list (list): List of ticker symbols to load. Defaults to ['AAPL', 'MSFT', 'GOOGL'].
        value_column (str): The column to use as asset values (e.g., 'close', 'open'). Defaults to 'close'.
        start_date (str, optional): Start date for data filtering (YYYY-MM-DD).
        end_date (str, optional): End date for data filtering (YYYY-MM-DD).
        db_path (str): Path to the SQLite database file.
        min_data_points (int): Minimum number of data points required after processing.

    Returns:
        pd.DataFrame: A DataFrame with dates as index, tickers as columns,
                      and the 'value_column' as values. Empty if data loading fails.
    """
    if tickers_list is None:
        tickers_list = ['AAPL', 'MSFT', 'GOOGL'] # Default tickers

    logging.info(f"Attempting to load data for tickers: {tickers_list} from DB: {db_path}")

    if not os.path.exists(db_path):
        logging.error(f"Database file not found at {db_path}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT date, ticker, {value_column} FROM price_data"
        filters = []
        params = []

        if tickers_list:
            placeholders = ','.join('?' for _ in tickers_list)
            filters.append(f"ticker IN ({placeholders})")
            params.extend(tickers_list)

        if start_date:
            filters.append("date >= ?")
            params.append(start_date)
        if end_date:
            filters.append("date <= ?")
            params.append(end_date)

        if filters:
            query += " WHERE " + " AND ".join(filters)
        
        query += " ORDER BY date ASC" # Ensure data is sorted by date

        long_df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    if long_df.empty:
        logging.warning(f"No data retrieved for tickers {tickers_list} with column '{value_column}'.")
        return pd.DataFrame()

    # Pivot the table
    try:
        market_df = long_df.pivot(index='date', columns='ticker', values=value_column)
    except Exception as e:
        logging.error(f"Error pivoting DataFrame: {e}. Long data head:\n{long_df.head()}")
        return pd.DataFrame()

    # Handle missing values
    market_df.fillna(method='ffill', inplace=True)
    market_df.fillna(method='bfill', inplace=True)
    market_df.dropna(how='any', inplace=True) # Drop rows with any remaining NaNs

    if market_df.empty:
        logging.warning("DataFrame is empty after pivoting and NaN handling.")
        return pd.DataFrame()

    if len(market_df) < min_data_points:
        logging.warning(f"Loaded data has {len(market_df)} rows, less than minimum {min_data_points}.")
        # Depending on strictness, you might return pd.DataFrame() or just warn.

    # Ensure all requested tickers are present as columns, if not, it might indicate missing data in DB
    for ticker in tickers_list:
        if ticker not in market_df.columns:
            logging.warning(f"Ticker '{ticker}' not found in the final DataFrame columns after processing.")


    logging.info(f"Market data loaded successfully. Shape: {market_df.shape}")
    return market_df