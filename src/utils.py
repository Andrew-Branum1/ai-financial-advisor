# src/utils.py
import pandas as pd
import sqlite3
import os
import logging
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
import gym
from config import AGENT_TICKERS, FEATURES_TO_USE_IN_MODEL


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.db')

def load_market_data_for_universal_env(
    tickers_list: list,
    feature_columns: list,
    start_date: str,
    end_date: str,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """
    Loads market data in a long format suitable for the universal environment.
    It does NOT pivot the data, keeping it scalable.

    Returns:
        pd.DataFrame: A DataFrame with columns ['date', 'ticker', 'feature1', 'feature2', ...],
                      indexed by date.
    """
    if not os.path.exists(db_path):
        logging.error(f"Database file not found at {db_path}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        columns_to_select_str = ", ".join(sorted(list(set(feature_columns))))
        
        query = f"""
            SELECT date, ticker, {columns_to_select_str} 
            FROM price_data
            WHERE ticker IN ({','.join('?' for _ in tickers_list)})
            AND date BETWEEN ? AND ?
            ORDER BY date ASC, ticker ASC
        """
        
        params = tickers_list + [start_date, end_date]
        long_df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
        
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    if long_df.empty:
        logging.warning("No data returned from the database for the given parameters.")
        return pd.DataFrame()
        
    # Forward-fill and back-fill missing values per ticker
    long_df = long_df.groupby('ticker').apply(lambda group: group.ffill().bfill())
    long_df.dropna(inplace=True) # Drop any remaining NaNs

    logging.info(f"Loaded long-format data with shape: {long_df.shape}")
    return long_df.set_index('date')


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

# Add this new code to the end of your src/utils.py file

def load_and_split_data(start_date: str, end_date: str, **kwargs):
    """
    A wrapper around load_market_data_from_db to provide a consistent
    interface for loading data for specific date ranges.
    """
    logging.info(f"Loading data for date range: {start_date} to {end_date}")
    
    # Use kwargs to allow overriding, otherwise use defaults from config.py
    tickers = kwargs.get('tickers_list', AGENT_TICKERS)
    features = kwargs.get('feature_columns', FEATURES_TO_USE_IN_MODEL)

    df = load_market_data_from_db(
        tickers_list=tickers,
        feature_columns=features,
        start_date=start_date,
        end_date=end_date,
        db_path=DEFAULT_DB_PATH
    )
    return df

def calculate_evaluation_kpis(model: BaseAlgorithm, eval_env: gym.Env) -> dict:
    """
    Evaluates the agent on the evaluation environment and calculates a dictionary of KPIs.
    """
    logging.info("Starting evaluation for KPI calculation...")
    obs, info = eval_env.reset()
    terminated, truncated = False, False
    
    portfolio_values = [eval_env.get_wrapper_attr('initial_balance')]
    turnover_history = []
    
    # Store the weights at each step
    weights_history = [eval_env.get_wrapper_attr('weights').copy()]
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = eval_env.step(action)
        
        # Calculate turnover for this step
        prev_weights = weights_history[-1]
        current_weights = info.get('weights', prev_weights)
        turnover = np.sum(np.abs(np.array(current_weights) - np.array(prev_weights)))
        
        portfolio_values.append(info.get('portfolio_value'))
        turnover_history.append(turnover)
        weights_history.append(current_weights)

    logging.info(f"Evaluation finished. Portfolio value records captured: {len(portfolio_values)}")

    if len(portfolio_values) < 2:
        return {"sharpe_ratio": -1.0} # Return a default if something went wrong

    portfolio_df = pd.DataFrame({'value': portfolio_values})
    daily_returns = portfolio_df['value'].pct_change().dropna()

    kpis = {}
    if daily_returns.empty or daily_returns.std() == 0:
        kpis['sharpe_ratio'] = -1.0
    else:
        # Calculate all KPIs
        kpis['sharpe_ratio'] = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        kpis['cumulative_return'] = (portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0]) - 1
        
        # Calculate Max Drawdown
        roll_max = portfolio_df['value'].cummax()
        drawdown = (portfolio_df['value'] - roll_max) / roll_max
        kpis['max_drawdown'] = drawdown.min()

    kpis['average_daily_turnover'] = np.mean(turnover_history) if turnover_history else 0.0
    
    logging.info(f"Calculated KPIs: {kpis}")
    return kpis