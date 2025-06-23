# rl/portfolio_env_short_term.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import collections

class PortfolioEnvShortTerm(gym.Env):
    """
    An environment that rewards the agent based on the rolling Sharpe ratio of its returns,
    encouraging active, short-term trading.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, feature_columns_ordered: list, **kwargs):
        super().__init__()

        if df.empty:
            raise ValueError("Input DataFrame `df` cannot be empty.")

        # --- ENV SETUP ---
        self.df = df
        self.feature_names = feature_columns_ordered
        self.tickers = sorted(list(set(col.split('_')[0] for col in df.columns)))
        self.asset_dim = len(self.tickers)
        self.num_features = len(self.feature_names)

        # --- KWARGS WITH DEFAULTS ---
        self.window_size = int(kwargs.get('window_size', 30))
        self.initial_balance = float(kwargs.get('initial_balance', 10000.0))
        self.transaction_cost_pct = float(kwargs.get('transaction_cost_pct', 0.001))
        self.rolling_volatility_window = int(kwargs.get('rolling_volatility_window', 20))
        self.max_concentration_per_asset = float(kwargs.get('max_concentration_per_asset', 1.0))
        self.turnover_penalty_weight = float(kwargs.get('turnover_penalty_weight', 0.01))

        # --- SPACES ---
        self.action_space = spaces.Box(low=0, high=1, shape=(self.asset_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.asset_dim * self.num_features), dtype=np.float32)
        
        # --- STATE VARIABLES ---
        self.recent_portfolio_returns = collections.deque(maxlen=self.rolling_volatility_window)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.weights = np.full(self.asset_dim, 1.0 / self.asset_dim, dtype=np.float32)
        self.previous_weights = self.weights.copy()
        self.recent_portfolio_returns.clear()
        
        return self._get_observation(), self._get_info()

    # In rl/portfolio_env_short_term.py, replace the step method

    def step(self, action: np.ndarray):
        self.previous_weights = self.weights.copy()
        action = np.clip(action, 0, self.max_concentration_per_asset)
        action_sum = np.sum(action)
        if action_sum > 1e-6:
            self.weights = action / action_sum
        else:
            self.weights = np.full(self.asset_dim, 1.0 / self.asset_dim, dtype=np.float32)

        turnover = np.sum(np.abs(self.weights - self.previous_weights))
        costs = self.portfolio_value * turnover * self.transaction_cost_pct
        self.portfolio_value -= costs

        # Calculate portfolio return for this step
        # Note: 'close' is not used in the short-term feature set, so we find its index in the full df
        try:
            close_col_names = [col for col in self.df.columns if col.endswith('_close')]
            close_prices_df = self.df[close_col_names]
            prev_close_prices = close_prices_df.iloc[self.current_step - 1].values
            current_close_prices = close_prices_df.iloc[self.current_step].values
            daily_asset_returns = (current_close_prices / prev_close_prices) - 1
        except Exception:
             daily_asset_returns = np.zeros(self.asset_dim)

        portfolio_daily_return = np.dot(self.weights, daily_asset_returns)
        self.portfolio_value *= (1 + portfolio_daily_return)
        self.recent_portfolio_returns.append(portfolio_daily_return)

        # --- ROLLING SHARPE REWARD LOGIC ---
        final_reward = 0.0
        if len(self.recent_portfolio_returns) == self.rolling_volatility_window:
            recent_returns = np.array(list(self.recent_portfolio_returns))
            if np.std(recent_returns) > 1e-6:
                final_reward = np.mean(recent_returns) / np.std(recent_returns)
        
        final_reward -= self.turnover_penalty_weight * turnover

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = terminated
        
        # --- THIS IS THE CORRECTED PART ---
        # Get the base info and add all per-step details for evaluation
        info = self._get_info()
        info['transaction_costs'] = costs
        info['raw_daily_return'] = portfolio_daily_return
        info['turnover'] = turnover
        # --- END OF CORRECTION ---

        return self._get_observation(), float(final_reward), terminated, truncated, info

    def _get_observation(self):
        obs_window = self.df.iloc[self.current_step - self.window_size:self.current_step].values.astype(np.float32)
        # Simple normalization for brevity, can be enhanced
        return (obs_window - np.mean(obs_window, axis=0)) / (np.std(obs_window, axis=0) + 1e-9)

    def _get_info(self):
        return {"portfolio_value": self.portfolio_value, "weights": self.weights.tolist()}

    def close(self):
        pass