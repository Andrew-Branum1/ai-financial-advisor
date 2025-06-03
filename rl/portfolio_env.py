# rl/portfolio_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import collections # For deque

class PortfolioEnv(gym.Env):
    """
    A portfolio management environment using Gymnasium.
    Observations include multiple features per asset.
    Includes transaction costs and advanced reward shaping for drawdown reduction.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df: pd.DataFrame,
                 feature_columns_ordered: list = None,
                 window_size: int = 30,
                 initial_balance: float = 10000.0,
                 transaction_cost_pct: float = 0.001,
                 volatility_penalty_weight: float = 0.05, # HYPERPARAMETER for reward shaping
                 loss_aversion_factor: float = 1.5,      # HYPERPARAMETER for reward shaping
                 rolling_volatility_window: int = 20):   # Window for rolling volatility calculation
        super(PortfolioEnv, self).__init__()

        if df.empty:
            raise ValueError("Input DataFrame `df` cannot be empty.")

        self.tickers = sorted(list(set(col.split('_')[0] for col in df.columns if '_' in col)))
        if not self.tickers:
            raise ValueError("Could not derive tickers from DataFrame columns. Ensure columns are named 'TICKER_FEATURE'.")

        self.asset_dim = len(self.tickers)

        if feature_columns_ordered is None:
            example_ticker_cols = [col for col in df.columns if col.startswith(self.tickers[0] + '_')]
            self.feature_names = sorted(list(set(col.split('_',1)[1] for col in example_ticker_cols)))
            logging.warning(f"feature_columns_ordered not provided, derived features: {self.feature_names}. Ensure this order is consistent.")
        else:
            self.feature_names = feature_columns_ordered

        self.num_features = len(self.feature_names)
        if self.num_features == 0:
            raise ValueError("Number of features cannot be zero.")

        expected_df_cols = []
        for ticker in self.tickers:
            for feature in self.feature_names:
                expected_df_cols.append(f"{ticker}_{feature}")

        missing_cols = [col for col in expected_df_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing expected columns: {missing_cols}. Ensure data loader provides all ticker_feature combinations.")

        self.df = df[expected_df_cols].reset_index(drop=True)

        if self.df.shape[0] <= window_size:
            raise ValueError(f"DataFrame length ({self.df.shape[0]}) must be greater than window_size ({window_size}).")

        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.volatility_penalty_weight = float(volatility_penalty_weight)
        self.loss_aversion_factor = float(loss_aversion_factor)
        self.rolling_volatility_window = int(rolling_volatility_window)

        self.action_space = spaces.Box(low=0, high=1, shape=(self.asset_dim,), dtype=np.float32)
        self.observation_space_shape = (self.window_size, self.asset_dim * self.num_features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_space_shape, dtype=np.float32
        )
        
        # For reward shaping
        self.recent_portfolio_returns = collections.deque(maxlen=self.rolling_volatility_window)

        self.current_step = 0
        self.portfolio_value = 0.0
        self.weights = np.zeros(self.asset_dim, dtype=np.float32)
        self.previous_weights = np.zeros(self.asset_dim, dtype=np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.weights = np.full(self.asset_dim, 1.0 / self.asset_dim, dtype=np.float32)
        self.previous_weights = self.weights.copy()
        self.recent_portfolio_returns.clear() # Clear history for new episode
        # Fill deque with some initial neutral returns (e.g. 0) to have a full window sooner if desired, or let it fill naturally
        # for _ in range(self.rolling_volatility_window):
        # self.recent_portfolio_returns.append(0.0)


        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray):
        self.previous_weights = self.weights.copy()

        action = np.clip(action, 0, None).astype(np.float32)
        action_sum = np.sum(action)
        if action_sum > 1e-6:
            self.weights = action / action_sum
        else:
            self.weights = np.full(self.asset_dim, 1.0 / self.asset_dim, dtype=np.float32)

        turnover = np.sum(np.abs(self.weights - self.previous_weights))
        costs = self.portfolio_value * turnover * self.transaction_cost_pct
        self.portfolio_value -= costs

        try:
            close_feature_idx = self.feature_names.index('close')
        except ValueError:
            raise ValueError("The feature 'close' must be in feature_columns_ordered to calculate returns.")

        prev_close_prices = []
        current_close_prices = []
        for i in range(self.asset_dim):
            base_col_idx_for_asset_i = i * self.num_features
            prev_close_prices.append(self.df.iloc[self.current_step - 1, base_col_idx_for_asset_i + close_feature_idx])
            current_close_prices.append(self.df.iloc[self.current_step, base_col_idx_for_asset_i + close_feature_idx])

        prev_close_prices = np.array(prev_close_prices, dtype=np.float32)
        current_close_prices = np.array(current_close_prices, dtype=np.float32)

        if not (np.all(np.isfinite(prev_close_prices)) and np.all(np.isfinite(current_close_prices)) and np.all(prev_close_prices > 0)):
            daily_asset_returns = np.zeros(self.asset_dim, dtype=np.float32)
            logging.warning(f"Step {self.current_step}: Invalid close price data. Asset returns set to 0.")
        else:
            daily_asset_returns = (current_close_prices / prev_close_prices) - 1

        portfolio_daily_return = np.dot(self.weights, daily_asset_returns)
        self.portfolio_value *= (1 + portfolio_daily_return)
        
        # -- Reward Shaping --
        self.recent_portfolio_returns.append(portfolio_daily_return)
        
        # 1. Loss Aversion Penalty
        shaped_reward = portfolio_daily_return
        if portfolio_daily_return < 0:
            shaped_reward *= self.loss_aversion_factor

        # 2. Volatility Penalty
        current_volatility_penalty = 0
        if len(self.recent_portfolio_returns) == self.recent_portfolio_returns.maxlen: # Ensure deque is full
            portfolio_std_dev = np.std(list(self.recent_portfolio_returns))
            current_volatility_penalty = self.volatility_penalty_weight * portfolio_std_dev
        
        final_reward = shaped_reward - current_volatility_penalty
        # -- End Reward Shaping --

        self.current_step += 1
        terminated = False
        truncated = False
        if self.current_step >= len(self.df) -1 :
            truncated = True

        observation = self._get_observation()
        info = self._get_info()
        info["transaction_costs"] = costs
        info["raw_daily_return"] = portfolio_daily_return # Log raw return for analysis
        info["volatility_penalty"] = current_volatility_penalty # Log penalty for analysis

        return observation, float(final_reward), terminated, truncated, info

    def _get_observation(self):
        obs_start_idx = self.current_step - self.window_size
        obs_end_idx = self.current_step
        raw_obs_window = self.df.iloc[obs_start_idx:obs_end_idx].values.astype(np.float32)
        normalized_obs_window = np.zeros_like(raw_obs_window)

        for i in range(self.asset_dim):
            for j, feature_name in enumerate(self.feature_names):
                col_idx = i * self.num_features + j
                feature_window = raw_obs_window[:, col_idx].copy()

                if feature_name == 'close':
                    if feature_window[0] > 1e-9:
                        normalized_obs_window[:, col_idx] = (feature_window / feature_window[0]) - 1.0
                    else:
                        normalized_obs_window[:, col_idx] = 0.0
                elif feature_name == 'rsi':
                    normalized_obs_window[:, col_idx] = feature_window / 100.0
                elif 'volatility' in feature_name or feature_name == 'atr':
                    normalized_obs_window[:, col_idx] = np.log1p(feature_window)
                elif 'bollinger_mavg' in feature_name or 'bollinger_hband' in feature_name or 'bollinger_lband' in feature_name: # Treat all bollinger bands like price series
                     if feature_window[0] > 1e-9:
                        normalized_obs_window[:, col_idx] = (feature_window / feature_window[0]) - 1.0
                     else:
                        normalized_obs_window[:, col_idx] = 0.0
                else:
                    if feature_window[0] != 0 and np.abs(feature_window[0]) > 1e-9:
                         normalized_obs_window[:, col_idx] = feature_window / feature_window[0] - 1.0
                    else:
                         normalized_obs_window[:, col_idx] = feature_window
        
        if not np.all(np.isfinite(normalized_obs_window)):
            logging.warning(f"Step {self.current_step}: NaN/Inf found in normalized_obs_window, replacing with 0.")
            normalized_obs_window = np.nan_to_num(normalized_obs_window, nan=0.0, posinf=0.0, neginf=0.0)
            
        return normalized_obs_window.astype(np.float32)

    def _get_info(self):
        return {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.tolist(),
            "current_step": self.current_step,
            "transaction_costs": 0.0,
            "raw_daily_return": 0.0,
            "volatility_penalty": 0.0
        }

    def render(self, mode='human'):
        if mode == 'human':
            actual_trading_step = self.current_step - self.window_size
            print(f"Trading Step: {actual_trading_step}, Portfolio Value: ${self.portfolio_value:,.2f}")
        else:
            super().render(mode=mode)

    def close(self):
        pass