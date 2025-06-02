# src/utils.py
import pandas as pd
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.db') # Assumes /data/market_data.db from project root

def load_market_data_from_db(
    tickers_list: list = None,
    feature_columns: list = None, # Changed from value_column
    start_date: str = None,
    end_date: str = None,
    db_path: str = DEFAULT_DB_PATH,
    min_data_points: int = 60
):
    """
    Loads specified market data features for given tickers from the SQLite database,
    and pivots it into a wide format suitable for PortfolioEnv.
    Each column will be named TICKER_feature (e.g., AAPL_close, AAPL_rsi).

    Args:
        tickers_list (list): List of ticker symbols to load. Defaults to ['AAPL', 'MSFT', 'GOOGL'].
        feature_columns (list): List of feature column names to load (e.g., ['close', 'rsi', 'volatility_20']). Defaults to ['close'].
        start_date (str, optional): Start date for data filtering (YYYY-MM-DD).
        end_date (str, optional): End date for data filtering (YYYY-MM-DD).
        db_path (str): Path to the SQLite database file.
        min_data_points (int): Minimum number of data points required after processing.

    Returns:
        pd.DataFrame: DataFrame with dates as index, and TICKER_feature as columns. Empty if fails.
    """
    if tickers_list is None:
        tickers_list = ['AAPL', 'MSFT', 'GOOGL']
    if feature_columns is None:
        feature_columns = ['close']

    logging.info(f"Attempting to load features: {feature_columns} for tickers: {tickers_list} from DB: {db_path}")

    if not os.path.exists(db_path):
        logging.error(f"Database file not found at {db_path}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        # Ensure 'date' and 'ticker' are always selected along with feature_columns
        columns_to_select_str = ", ".join(feature_columns)
        query = f"SELECT date, ticker, {columns_to_select_str} FROM price_data"
        
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
        
        query += " ORDER BY date ASC, ticker ASC" # Ensure consistent ordering for processing

        long_df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    if long_df.empty:
        logging.warning(f"No data retrieved for tickers {tickers_list} with features: {feature_columns}.")
        return pd.DataFrame()

    # Pivot the table to create feature columns for each ticker
    try:
        # Set index for pivoting correctly
        long_df = long_df.set_index(['date', 'ticker'])
        market_df = long_df.unstack(level='ticker') # Unstack ticker to create Ticker1_featureA, Ticker2_featureA columns etc.
        
        # Flatten MultiIndex columns: from (feature, ticker) to ticker_feature
        market_df.columns = [f"{ticker}_{feature}" for feature, ticker in market_df.columns]
        
        # Reorder columns: Group by ticker, then by feature (optional but good for consistency)
        # e.g., AAPL_close, AAPL_rsi, ..., MSFT_close, MSFT_rsi, ...
        sorted_columns = []
        processed_tickers = sorted(list(set(col.split('_')[0] for col in market_df.columns if '_' in col))) # Get unique tickers from columns
        processed_features = sorted(list(set(col.split('_')[1] for col in market_df.columns if '_' in col)))# Get unique features

        # Check if feature_columns match processed_features
        if not all(fc in processed_features for fc in feature_columns):
            logging.warning(f"Not all requested feature_columns {feature_columns} were found in the processed data features {processed_features}. Some might be missing from DB for the selected tickers/dates.")
            # Update processed_features to only those actually loaded and pivoted
            processed_features = [f for f in feature_columns if any(f_tick.endswith(f"_{f}") for f_tick in market_df.columns)]


        for ticker in processed_tickers:
            for feature in processed_features: # Use the originally requested features for ordering
                col_name = f"{ticker}_{feature}"
                if col_name in market_df.columns:
                    sorted_columns.append(col_name)
        
        if not sorted_columns: # Handle case where no columns could be formed
            logging.error("Could not form sorted columns. Check data and feature names.")
            return pd.DataFrame()

        market_df = market_df[sorted_columns]

    except Exception as e:
        logging.error(f"Error pivoting or reformatting DataFrame: {e}. Long data head:\n{long_df.head(20)}", exc_info=True)
        return pd.DataFrame()

    # Handle missing values that might have occurred during pivot/unstack
    market_df.ffill(inplace=True) # Use .ffill() instead of fillna(method='ffill')
    market_df.bfill(inplace=True) # Use .bfill() instead of fillna(method='bfill')
    market_df.dropna(how='any', inplace=True) # Drop rows with any remaining NaNs

    if market_df.empty:
        logging.warning("DataFrame is empty after pivoting and NaN handling.")
        return pd.DataFrame()

    if len(market_df) < min_data_points:
        logging.warning(f"Loaded data has {len(market_df)} rows, less than minimum {min_data_points}.")

    logging.info(f"Market data loaded successfully. Shape: {market_df.shape}. Columns: {market_df.columns.tolist()[:10]}...") # Log first 10 cols
    return market_df