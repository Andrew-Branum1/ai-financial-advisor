# src/utils.py
import pandas as pd
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.db')

def load_market_data_from_db(
    tickers_list: list = None,
    feature_columns: list = None,
    start_date: str = None,
    end_date: str = None,
    db_path: str = DEFAULT_DB_PATH,
    min_data_points: int = 60
):
    """
    Loads specified market data features for given tickers from the SQLite database,
    and pivots it into a wide format suitable for PortfolioEnv.
    Each column will be named TICKER_feature (e.g., AAPL_close, AAPL_rsi, MSFT_bollinger_hband).

    Args:
        tickers_list (list): List of ticker symbols to load. Defaults to ['AAPL', 'MSFT', 'GOOGL'].
        feature_columns (list): List of feature column names to load from the DB
                                (e.g., ['close', 'rsi', 'volatility_20', 'bollinger_hband', 'atr']).
                                Defaults to ['close'].
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
        feature_columns = ['close'] # Default to only 'close' if not specified

    logging.info(f"Attempting to load features: {feature_columns} for tickers: {tickers_list} from DB: {db_path}")

    if not os.path.exists(db_path):
        logging.error(f"Database file not found at {db_path}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        # Ensure 'date' and 'ticker' are always selected along with requested feature_columns
        columns_to_select_str = ", ".join(sorted(list(set(feature_columns)))) # Use set to avoid duplicates if any
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

        query += " ORDER BY date ASC, ticker ASC"

        long_df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
    except Exception as e:
        logging.error(f"Error querying database (features: {feature_columns}): {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    if long_df.empty:
        logging.warning(f"No data retrieved for tickers {tickers_list} with features: {feature_columns}.")
        return pd.DataFrame()

    # Verify all requested feature_columns are present in the loaded long_df before pivot
    missing_in_df = [fc for fc in feature_columns if fc not in long_df.columns]
    if missing_in_df:
        logging.warning(f"The following requested feature_columns were not found in the data fetched from DB: {missing_in_df}. They might be missing from the table for the selected period/tickers or there might be a typo.")
        # Filter out missing columns to prevent errors in pivot/unstack
        feature_columns = [fc for fc in feature_columns if fc in long_df.columns]
        if not feature_columns:
            logging.error("No valid feature columns remaining after checking DB results. Cannot proceed with pivot.")
            return pd.DataFrame()
        logging.info(f"Proceeding with pivot using available features: {feature_columns}")


    try:
        long_df = long_df.set_index(['date', 'ticker'])
        # Pivot only the actual feature columns loaded (not 'date' or 'ticker' which are index)
        market_df = long_df[feature_columns].unstack(level='ticker')

        # Flatten MultiIndex columns: from (feature, ticker) to ticker_feature
        market_df.columns = [f"{ticker}_{feature}" for feature, ticker in market_df.columns]

        # Reorder columns: Group by ticker, then by feature (for consistency)
        sorted_columns = []
        # Use the tickers_list provided by the user for the primary order of tickers
        # and feature_columns for the order of features within each ticker.
        processed_tickers_ordered = [t for t in tickers_list if t in {col.split('_')[0] for col in market_df.columns}]


        for ticker in processed_tickers_ordered:
            for feature in feature_columns: # Use the (potentially filtered) feature_columns
                col_name = f"{ticker}_{feature}"
                if col_name in market_df.columns:
                    sorted_columns.append(col_name)
        
        if not sorted_columns:
            logging.error("Could not form sorted columns. Check data, ticker names, and feature names.")
            return pd.DataFrame()

        market_df = market_df[sorted_columns]

    except Exception as e:
        logging.error(f"Error pivoting or reformatting DataFrame: {e}. Long data head:\n{long_df.head(20)}", exc_info=True)
        return pd.DataFrame()

    market_df.ffill(inplace=True)
    market_df.bfill(inplace=True)
    market_df.dropna(how='any', inplace=True)

    if market_df.empty:
        logging.warning("DataFrame is empty after pivoting and NaN handling.")
        return pd.DataFrame()

    if len(market_df) < min_data_points:
        logging.warning(f"Loaded data has {len(market_df)} rows, less than minimum {min_data_points}.")

    logging.info(f"Market data loaded successfully. Shape: {market_df.shape}. Columns (first 10 of {len(market_df.columns)}): {market_df.columns.tolist()[:10]}...")
    return market_df