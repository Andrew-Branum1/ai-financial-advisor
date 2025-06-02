# rl/portfolio_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

class PortfolioEnv(gym.Env):
    """
    A portfolio management environment using Gymnasium.
    Observations include multiple features per asset (e.g., close, rsi, volatility).
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df: pd.DataFrame, 
                 feature_columns_ordered: list = None, # e.g., ['close', 'rsi', 'volatility_20']
                 window_size: int = 30, 
                 initial_balance: float = 10000.0):
        super(PortfolioEnv, self).__init__()

        if df.empty:
            raise ValueError("Input DataFrame `df` cannot be empty.")
        
        # Derive tickers and features from DataFrame columns (e.g., AAPL_close, MSFT_close, AAPL_rsi)
        # Assumes columns are named TICKER_FEATURE
        self.tickers = sorted(list(set(col.split('_')[0] for col in df.columns if '_' in col)))
        if not self.tickers:
            raise ValueError("Could not derive tickers from DataFrame columns. Ensure columns are named 'TICKER_FEATURE'.")
        
        self.asset_dim = len(self.tickers) # Number of assets to manage

        if feature_columns_ordered is None:
            # Attempt to derive features if not provided, taking the suffix after the first underscore
            # This assumes all tickers have the same set of features and in the same order implicitly if not provided
            example_ticker_cols = [col for col in df.columns if col.startswith(self.tickers[0] + '_')]
            self.feature_names = sorted(list(set(col.split('_',1)[1] for col in example_ticker_cols)))
            logging.warning(f"feature_columns_ordered not provided, derived features: {self.feature_names}. Ensure this order is consistent.")

        else:
            self.feature_names = feature_columns_ordered
        
        self.num_features = len(self.feature_names)
        if self.num_features == 0:
            raise ValueError("Number of features cannot be zero.")

        # Verify all expected columns exist: TICKER_FEATURE for all tickers and features
        expected_df_cols = []
        for ticker in self.tickers:
            for feature in self.feature_names:
                expected_df_cols.append(f"{ticker}_{feature}")
        
        missing_cols = [col for col in expected_df_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing expected columns: {missing_cols}. Ensure data loader provides all ticker_feature combinations.")

        # Reorder df columns to ensure consistency: Ticker1_Feat1, Ticker1_Feat2, ..., TickerN_FeatM
        # This order is critical for how observations are constructed and interpreted.
        self.df = df[expected_df_cols].reset_index(drop=True)


        if self.df.shape[0] <= window_size:
            raise ValueError(f"DataFrame length ({self.df.shape[0]}) must be greater than window_size ({window_size}).")

        self.window_size = window_size
        self.initial_balance = float(initial_balance)

        # Action space: normalized weights for each asset (one weight per ticker)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.asset_dim,), dtype=np.float32)

        # Observation space: (window_size, num_assets * num_features)
        # Data is flattened: [Asset1_Feat1, Asset1_Feat2, ..., AssetN_FeatM] for each step in window
        self.observation_space_shape = (self.window_size, self.asset_dim * self.num_features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_space_shape, dtype=np.float32
        )

        self.current_step = 0
        self.portfolio_value = 0.0
        self.weights = np.zeros(self.asset_dim, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.weights = np.full(self.asset_dim, 1.0 / self.asset_dim, dtype=np.float32)
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray):
        action = np.clip(action, 0, None).astype(np.float32)
        action_sum = np.sum(action)
        if action_sum > 1e-6:
            self.weights = action / action_sum
        else:
            self.weights = np.full(self.asset_dim, 1.0 / self.asset_dim, dtype=np.float32)

        # Get 'close' prices for return calculation
        # Assumes 'close' is one of the features and columns are Ticker1_close, Ticker2_close ...
        prev_close_prices = []
        current_close_prices = []

        # Find the index of the 'close' feature
        try:
            close_feature_idx = self.feature_names.index('close')
        except ValueError:
            raise ValueError("The feature 'close' must be in feature_columns_ordered to calculate returns.")

        for i in range(self.asset_dim):
            # Columns for asset 'i' are grouped together: Ticker_i_Feat1, Ticker_i_Feat2, ...
            # The 'close' feature for asset 'i' is at base_col_idx + close_feature_idx
            base_col_idx_for_asset_i = i * self.num_features
            
            prev_close_prices.append(self.df.iloc[self.current_step - 1, base_col_idx_for_asset_i + close_feature_idx])
            current_close_prices.append(self.df.iloc[self.current_step, base_col_idx_for_asset_i + close_feature_idx])
        
        prev_close_prices = np.array(prev_close_prices, dtype=np.float32)
        current_close_prices = np.array(current_close_prices, dtype=np.float32)

        if not (np.all(np.isfinite(prev_close_prices)) and np.all(np.isfinite(current_close_prices)) and np.all(prev_close_prices > 0)):
            daily_asset_returns = np.zeros(self.asset_dim, dtype=np.float32)
            logging.warning(f"Step {self.current_step}: Invalid close price data for return calculation.")
        else:
            daily_asset_returns = (current_close_prices / prev_close_prices) - 1
        
        portfolio_daily_return = np.dot(self.weights, daily_asset_returns)
        self.portfolio_value *= (1 + portfolio_daily_return)
        reward = portfolio_daily_return

        self.current_step += 1
        terminated = False
        truncated = False
        if self.current_step >= len(self.df) -1 : # -1 because current_step is next state index
            truncated = True
        
        observation = self._get_observation()
        info = self_get_info()
        return observation, float(reward), terminated, truncated, info

    def _get_observation(self):
        obs_start_idx = self.current_step - self.window_size
        obs_end_idx = self.current_step
        
        # Raw observation window: (window_size, num_assets * num_features)
        raw_obs_window = self.df.iloc[obs_start_idx:obs_end_idx].values.astype(np.float32)

        # Apply normalization
        # Reshape to (window_size, num_assets, num_features) for easier feature-wise normalization
        # This assumes columns in self.df are ordered: T1_F1, T1_F2, T2_F1, T2_F2 ...
        # If they are T1_F1, T2_F1, T1_F2, T2_F2, then reshaping logic needs adjustment.
        # Based on utils.py, they should be T1_F1, T1_F2, ..., TN_F1, TN_F2 ...
        
        normalized_obs_window = np.zeros_like(raw_obs_window)

        for i in range(self.asset_dim): # Iterate through each asset
            for j, feature_name in enumerate(self.feature_names): # Iterate through each feature for that asset
                # Calculate column index in the raw_obs_window
                # This is Ticker_i_Feature_j
                col_idx = i * self.num_features + j
                feature_window = raw_obs_window[:, col_idx].copy() # Get window for this specific feature of this asset

                if feature_name == 'close':
                    if feature_window[0] > 1e-6: # Avoid division by zero or tiny numbers
                        normalized_obs_window[:, col_idx] = (feature_window / feature_window[0]) - 1.0
                    else: # If first price is zero/small, keep relative changes zero or use other fill
                        normalized_obs_window[:, col_idx] = 0.0 
                elif feature_name == 'rsi':
                    normalized_obs_window[:, col_idx] = feature_window / 100.0 # Scale RSI to 0-1
                elif feature_name == 'volatility_20' or 'volatility' in feature_name : # Example for volatility
                    normalized_obs_window[:, col_idx] = np.log1p(feature_window) # Log transform for skewed, non-negative data
                else:
                    # For other features, you might need specific normalization
                    # For now, let's do a simple relative normalization if first value is non-zero
                    # This is a basic approach; more robust scaling per feature type is better
                    if feature_window[0] != 0:
                         normalized_obs_window[:, col_idx] = feature_window / feature_window[0] -1.0 if feature_window[0] > 1e-6 else 0.0
                    else: # if first element is 0, relative change is problematic
                         normalized_obs_window[:, col_idx] = feature_window 


        if not np.all(np.isfinite(normalized_obs_window)):
            logging.warning(f"Step {self.current_step}: NaN/Inf found in normalized_obs_window, replacing with 0.")
            normalized_obs_window = np.nan_to_num(normalized_obs_window, nan=0.0, posinf=0.0, neginf=0.0)
            
        return normalized_obs_window.astype(np.float32)


    def _get_info(self):
        return {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.tolist(),
            "current_step": self.current_step
        }

    def render(self, mode='human'):
        if mode == 'human':
            actual_trading_step = self.current_step - self.window_size
            print(f"Trading Step: {actual_trading_step}, Portfolio Value: ${self.portfolio_value:,.2f}")
        else:
            super().render(mode=mode)

    def close(self):
        pass