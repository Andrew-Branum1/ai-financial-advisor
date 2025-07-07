# src/utils.py
import pandas as pd
import sqlite3
import os
import logging
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
import gym
from config import AGENT_TICKERS, FEATURES_TO_USE_IN_MODEL
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'market_data.db')

@dataclass
class UserProfile:
    name: str
    age: int
    income: float
    investment_amount: float
    time_horizon: str  # e.g., 'short_term', 'long_term'
    risk_tolerance: str  # e.g., 'conservative', 'moderate', 'aggressive'
    goal: str  # e.g., 'growth', 'balanced', 'preservation'


def map_user_profile_to_env_params(user: UserProfile) -> Dict:
    """
    Map user profile attributes to RL environment parameters.
    This can be expanded with more sophisticated logic as needed.
    """
    # Example mapping logic
    params = {
        'initial_balance': user.investment_amount,
        'transaction_cost_pct': 0.001,
    }
    # Risk tolerance mapping
    if user.risk_tolerance == 'conservative':
        params.update({
            'max_drawdown_limit': 0.10,
            'max_concentration_per_asset': 0.20,
            'turnover_penalty_weight': 0.02,
            'window_size': 60 if user.time_horizon == 'long_term' else 30,
            'min_holding_period': 30 if user.time_horizon == 'long_term' else 5,
        })
    elif user.risk_tolerance == 'moderate':
        params.update({
            'max_drawdown_limit': 0.20,
            'max_concentration_per_asset': 0.35,
            'turnover_penalty_weight': 0.01,
            'window_size': 60 if user.time_horizon == 'long_term' else 30,
            'min_holding_period': 20 if user.time_horizon == 'long_term' else 3,
        })
    else:  # aggressive
        params.update({
            'max_drawdown_limit': 0.30,
            'max_concentration_per_asset': 0.50,
            'turnover_penalty_weight': 0.005,
            'window_size': 60 if user.time_horizon == 'long_term' else 30,
            'min_holding_period': 10 if user.time_horizon == 'long_term' else 1,
        })
    # Goal-based adjustments (optional)
    if user.goal == 'preservation':
        params['volatility_target'] = 0.08
    elif user.goal == 'balanced':
        params['volatility_target'] = 0.12
    else:  # growth
        params['volatility_target'] = 0.18
    return params


def load_market_data_for_universal_env(
    tickers_list: List[str],
    feature_columns: List[str],
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
    tickers_list: Optional[List[str]] = None,
    feature_columns: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
    min_data_points: int = 60
) -> pd.DataFrame:
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

    # Ensure market_df is a DataFrame
    if not isinstance(market_df, pd.DataFrame):
        market_df = pd.DataFrame(market_df)
    
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


def load_and_split_data(start_date: str, end_date: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Load and split data into training and validation sets.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        **kwargs: Additional arguments for load_market_data_from_db
    
    Returns:
        Dictionary with 'train' and 'val' DataFrames
    """
    # Load full dataset
    full_data = load_market_data_from_db(start_date=start_date, end_date=end_date, **kwargs)
    
    if full_data.empty:
        return {'train': pd.DataFrame(), 'val': pd.DataFrame()}
    
    # Split data (80% train, 20% validation)
    split_idx = int(len(full_data) * 0.8)
    train_data = full_data.iloc[:split_idx]
    val_data = full_data.iloc[split_idx:]
    
    return {'train': train_data, 'val': val_data}


def calculate_evaluation_kpis(model: BaseAlgorithm, eval_env: gym.Env) -> Dict[str, float]:
    """
    Calculate key performance indicators for model evaluation.
    
    Args:
        model: Trained model
        eval_env: Evaluation environment
    
    Returns:
        Dictionary of KPIs
    """
    obs, _ = eval_env.reset()
    total_reward = 0.0
    steps = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += float(reward)
        steps += 1
        
        if terminated or truncated:
            break
    
    kpis = {
        'total_reward': total_reward,
        'steps': steps,
        'avg_reward_per_step': total_reward / steps if steps > 0 else 0.0
    }
    
    logging.info(f"Calculated KPIs: {kpis}")
    return kpis


def calculate_technical_indicators(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Calculate comprehensive technical indicators from basic price data.
    
    Args:
        df: DataFrame with basic price data (close, volume, daily_return)
        ticker: Ticker symbol for column naming
    
    Returns:
        DataFrame with calculated technical indicators
    """
    result = df.copy()
    
    # Ensure we have the required columns
    required_cols = [f'{ticker}_close', f'{ticker}_volume', f'{ticker}_daily_return']
    if not all(col in result.columns for col in required_cols):
        logging.warning(f"Missing required columns for {ticker}: {required_cols}")
        return result
    
    close_col = f'{ticker}_close'
    volume_col = f'{ticker}_volume'
    return_col = f'{ticker}_daily_return'
    
    # Basic price data
    close_prices = result[close_col]
    volumes = result[volume_col]
    returns = result[return_col]
    
    # 1. Moving Averages
    result[f'{ticker}_sma_10'] = close_prices.rolling(window=10).mean()
    result[f'{ticker}_sma_20'] = close_prices.rolling(window=20).mean()
    result[f'{ticker}_sma_50'] = close_prices.rolling(window=50).mean()
    result[f'{ticker}_ema_12'] = close_prices.ewm(span=12).mean()
    result[f'{ticker}_ema_26'] = close_prices.ewm(span=26).mean()
    
    # 2. Price vs Moving Averages
    result[f'{ticker}_close_vs_sma_10'] = close_prices / result[f'{ticker}_sma_10'] - 1
    result[f'{ticker}_close_vs_sma_20'] = close_prices / result[f'{ticker}_sma_20'] - 1
    result[f'{ticker}_close_vs_sma_50'] = close_prices / result[f'{ticker}_sma_50'] - 1
    result[f'{ticker}_close_vs_ema_12'] = close_prices / result[f'{ticker}_ema_12'] - 1
    result[f'{ticker}_close_vs_ema_26'] = close_prices / result[f'{ticker}_ema_26'] - 1
    
    # 3. Volatility Measures
    result[f'{ticker}_volatility_5'] = returns.rolling(window=5).std() * np.sqrt(252)
    result[f'{ticker}_volatility_10'] = returns.rolling(window=10).std() * np.sqrt(252)
    result[f'{ticker}_volatility_20'] = returns.rolling(window=20).std() * np.sqrt(252)
    result[f'{ticker}_volatility_60'] = returns.rolling(window=60).std() * np.sqrt(252)
    
    # 4. Momentum Indicators
    result[f'{ticker}_momentum_5'] = close_prices / close_prices.shift(5) - 1
    result[f'{ticker}_momentum_10'] = close_prices / close_prices.shift(10) - 1
    result[f'{ticker}_momentum_20'] = close_prices / close_prices.shift(20) - 1
    result[f'{ticker}_momentum_60'] = close_prices / close_prices.shift(60) - 1
    
    # 5. RSI (Relative Strength Index)
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, delta)).rolling(window=14).mean()
    rs = gain / loss
    result[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
    
    # 6. MACD
    ema_12 = close_prices.ewm(span=12).mean()
    ema_26 = close_prices.ewm(span=26).mean()
    result[f'{ticker}_macd'] = ema_12 - ema_26
    result[f'{ticker}_macd_signal'] = result[f'{ticker}_macd'].ewm(span=9).mean()
    result[f'{ticker}_macd_histogram'] = result[f'{ticker}_macd'] - result[f'{ticker}_macd_signal']
    
    # 7. Bollinger Bands
    sma_20 = close_prices.rolling(window=20).mean()
    std_20 = close_prices.rolling(window=20).std()
    result[f'{ticker}_bollinger_upper'] = sma_20 + (std_20 * 2)
    result[f'{ticker}_bollinger_lower'] = sma_20 - (std_20 * 2)
    result[f'{ticker}_bollinger_width'] = (result[f'{ticker}_bollinger_upper'] - result[f'{ticker}_bollinger_lower']) / sma_20
    result[f'{ticker}_bollinger_position'] = (close_prices - result[f'{ticker}_bollinger_lower']) / (result[f'{ticker}_bollinger_upper'] - result[f'{ticker}_bollinger_lower'])
    
    # 8. Volume Indicators
    result[f'{ticker}_volume_sma_10'] = volumes.rolling(window=10).mean()
    result[f'{ticker}_volume_sma_20'] = volumes.rolling(window=20).mean()
    result[f'{ticker}_volume_ratio'] = volumes / result[f'{ticker}_volume_sma_20']
    result[f'{ticker}_obv'] = (volumes * np.sign(returns)).cumsum()
    
    # 9. ATR (Average True Range) - simplified version
    high_low = close_prices.rolling(window=1).max() - close_prices.rolling(window=1).min()
    high_close = np.abs(close_prices.rolling(window=1).max() - close_prices.shift(1))
    low_close = np.abs(close_prices.rolling(window=1).min() - close_prices.shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    result[f'{ticker}_atr'] = true_range.rolling(window=14).mean()
    
    # 10. Stochastic Oscillator
    low_14 = close_prices.rolling(window=14).min()
    high_14 = close_prices.rolling(window=14).max()
    result[f'{ticker}_stoch_k'] = 100 * (close_prices - low_14) / (high_14 - low_14)
    result[f'{ticker}_stoch_d'] = result[f'{ticker}_stoch_k'].rolling(window=3).mean()
    
    # 11. Williams %R
    result[f'{ticker}_williams_r'] = -100 * (high_14 - close_prices) / (high_14 - low_14)
    
    # 12. CCI (Commodity Channel Index)
    typical_price = close_prices  # Simplified - using close price
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    result[f'{ticker}_cci'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # 13. MFI (Money Flow Index)
    typical_price = close_prices  # Simplified
    money_flow = typical_price * volumes
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    money_ratio = positive_flow / negative_flow
    result[f'{ticker}_mfi'] = 100 - (100 / (1 + money_ratio))
    
    # 14. ADX (Average Directional Index) - simplified
    plus_dm = delta.where(delta > 0, delta).rolling(window=14).mean()
    minus_dm = (-delta.where(delta < 0, delta)).rolling(window=14).mean()
    tr = true_range.rolling(window=14).mean()
    plus_di = 100 * plus_dm / tr
    minus_di = 100 * minus_dm / tr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    result[f'{ticker}_adx'] = pd.Series(dx).rolling(window=14).mean()
    
    # 15. Trend Strength
    result[f'{ticker}_trend_strength'] = np.abs(result[f'{ticker}_close_vs_sma_50'])
    
    # 16. Volatility Ratio
    result[f'{ticker}_volatility_ratio'] = result[f'{ticker}_volatility_20'] / result[f'{ticker}_volatility_60']
    
    # 17. Price Rate of Change
    result[f'{ticker}_roc_10'] = (close_prices / close_prices.shift(10) - 1) * 100
    result[f'{ticker}_roc_20'] = (close_prices / close_prices.shift(20) - 1) * 100
    
    # 18. Volume Rate of Change
    result[f'{ticker}_volume_roc'] = (volumes / volumes.shift(10) - 1) * 100
    
    # 19. Price Efficiency Ratio
    result[f'{ticker}_price_efficiency'] = np.abs(close_prices - close_prices.shift(20)) / close_prices.rolling(window=20).apply(lambda x: np.sum(np.abs(x.diff().dropna())))
    
    # 20. Mean Reversion Signal
    result[f'{ticker}_mean_reversion_signal'] = (close_prices - result[f'{ticker}_sma_50']) / (result[f'{ticker}_atr'] + 1e-8)
    
    return result


def enhance_market_data_with_indicators(df: pd.DataFrame, tickers_list: List[str]) -> pd.DataFrame:
    """
    Enhance market data with technical indicators for all tickers.
    
    Args:
        df: DataFrame with basic market data
        tickers_list: List of ticker symbols
    
    Returns:
        Enhanced DataFrame with technical indicators
    """
    enhanced_df = df.copy()
    
    for ticker in tickers_list:
        logging.info(f"Calculating technical indicators for {ticker}")
        enhanced_df = calculate_technical_indicators(enhanced_df, ticker)
    
    # Handle NaN values more gracefully
    if isinstance(enhanced_df, pd.DataFrame):
        # Replace infinite values with NaN
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        
        # Forward-fill and back-fill NaN values for technical indicators
        # This preserves the original data while filling gaps in calculated indicators
        enhanced_df = enhanced_df.ffill().bfill()
        
        # Only drop rows that still have NaN values after forward/back-filling
        # This should be minimal since we're filling gaps
        rows_before = len(enhanced_df)
        enhanced_df = enhanced_df.dropna()
        rows_after = len(enhanced_df)
        
        if rows_before != rows_after:
            logging.info(f"Dropped {rows_before - rows_after} rows with persistent NaN values")
    
    logging.info(f"Enhanced data shape: {enhanced_df.shape}")
    return enhanced_df


def load_market_data_with_indicators(
    tickers_list: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
    min_data_points: int = 60
) -> pd.DataFrame:
    """
    Load market data with technical indicators directly from the database.
    
    Args:
        tickers_list: List of ticker symbols to load
        start_date: Start date for data filtering (YYYY-MM-DD)
        end_date: End date for data filtering (YYYY-MM-DD)
        db_path: Path to the SQLite database file
        min_data_points: Minimum number of data points required after processing
    
    Returns:
        DataFrame with dates as index and TICKER_feature as columns, including technical indicators
    """
    # Define all available features in the database
    all_features = [
        'close', 'volume', 'daily_return',
        'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'close_vs_sma_10', 'close_vs_sma_20', 'close_vs_sma_50', 'close_vs_ema_12', 'close_vs_ema_26',
        'volatility_5', 'volatility_10', 'volatility_20', 'volatility_60',
        'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60',
        'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bollinger_upper', 'bollinger_lower', 'bollinger_width', 'bollinger_position',
        'volume_sma_10', 'volume_sma_20', 'volume_ratio', 'obv',
        'atr', 'stoch_k', 'stoch_d', 'williams_r',
        'cci', 'mfi', 'adx',
        'trend_strength', 'volatility_ratio',
        'roc_10', 'roc_20', 'volume_roc',
        'price_efficiency', 'mean_reversion_signal'
    ]
    
    # Load all available features from the database
    df = load_market_data_from_db(
        tickers_list=tickers_list,
        feature_columns=all_features,
        start_date=start_date,
        end_date=end_date,
        db_path=db_path,
        min_data_points=min_data_points
    )
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.error("Failed to load market data with indicators from database")
        return pd.DataFrame()
    
    # Handle any remaining NaN values gracefully
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    
    # Only drop rows that still have NaN values after forward/back-filling
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    
    if rows_before != rows_after:
        logging.info(f"Dropped {rows_before - rows_after} rows with persistent NaN values")
    
    logging.info(f"Market data with indicators loaded successfully. Shape: {df.shape}")
    return df