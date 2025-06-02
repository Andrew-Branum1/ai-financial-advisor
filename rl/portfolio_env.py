# rl/portfolio_env.py
import gymnasium as gym # Use Gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

class PortfolioEnv(gym.Env):
    """
    A simple portfolio management environment using Gymnasium.

    Args:
        df (pd.DataFrame): DataFrame with asset prices.
                           Index should be timestamps (dates).
                           Columns should be asset tickers.
                           Values are prices (e.g., 'close').
        window_size (int): Number of past timesteps to include in the observation.
        initial_balance (float): The initial cash balance of the portfolio.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df: pd.DataFrame, window_size: int = 30, initial_balance: float = 10000.0):
        super(PortfolioEnv, self).__init__()

        if df.empty:
            raise ValueError("Input DataFrame `df` cannot be empty.")
        if df.shape[0] <= window_size: # Need enough data for at least one full window
            raise ValueError(f"DataFrame length ({df.shape[0]}) must be greater than window_size ({window_size}).")

        self.df = df.reset_index(drop=True) # Ensure integer-based indexing for .iloc
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.asset_dim = df.shape[1]

        # Action space: desired weights for each asset (will be normalized)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.asset_dim,), dtype=np.float32)

        # Observation space: (window_size, num_assets) prices
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.asset_dim), dtype=np.float32
        )

        # These will be initialized in reset()
        self.current_step = 0
        self.portfolio_value = 0.0
        self.weights = np.zeros(self.asset_dim, dtype=np.float32)
        # self.done is replaced by terminated and truncated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Gymnasium expects this for reproducibility

        self.current_step = self.window_size # Start after the first full window
        self.portfolio_value = self.initial_balance
        # Initial weights: equally distributed or could be learned/set differently
        self.weights = np.full(self.asset_dim, 1.0 / self.asset_dim, dtype=np.float32)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action: np.ndarray):
        # 1. Normalize action to represent portfolio weights (sum to 1)
        action = np.clip(action, 0, None).astype(np.float32) # ensure non-negative
        action_sum = np.sum(action)
        if action_sum > 1e-6: # Avoid division by zero or near-zero sum
            self.weights = action / action_sum
        else:
            # Default to equal weights if action sum is too small (or all zeros)
            self.weights = np.full(self.asset_dim, 1.0 / self.asset_dim, dtype=np.float32)

        # 2. Calculate portfolio return
        # Prices at t-1 (previous step's close) and t (current step's close)
        prev_prices = self.df.iloc[self.current_step - 1].values.astype(np.float32)
        current_prices = self.df.iloc[self.current_step].values.astype(np.float32)

        if not (np.all(np.isfinite(prev_prices)) and np.all(np.isfinite(current_prices)) and np.all(prev_prices > 0)):
            # If price data is bad (e.g., zeros, NaNs), assume no change in value for this step
            daily_asset_returns = np.zeros(self.asset_dim, dtype=np.float32)
            logging.warning(f"Step {self.current_step}: Invalid price data. Prev: {prev_prices}, Curr: {current_prices}")
        else:
            daily_asset_returns = (current_prices / prev_prices) - 1
        
        portfolio_daily_return = np.dot(self.weights, daily_asset_returns)
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_daily_return)
        
        # 3. Calculate reward
        reward = portfolio_daily_return # Simple reward: portfolio's daily return

        # 4. Update state and check for termination/truncation
        self.current_step += 1
        
        terminated = False # Task-specific ending condition
        truncated = False  # Time limit or out-of-data condition

        # Episode ends if we run out of data for the next observation
        if self.current_step >= len(self.df) -1: # -1 as current_step is new state, df.iloc[current_step] is needed
            truncated = True
        
        # Optional: Terminate if portfolio value is too low
        # if self.portfolio_value < 0.1 * self.initial_balance: # Example: 90% loss
        #     terminated = True
        #     reward -= 10 # Heavy penalty for "going broke"

        observation = self._get_observation()
        info = self._get_info()
        
        return observation, float(reward), terminated, truncated, info

    def _get_observation(self):
        # Observation is the window of prices leading up to the current_step
        obs_start_idx = self.current_step - self.window_size
        obs_end_idx = self.current_step
        
        if obs_start_idx < 0: # Should not happen if reset and step logic is correct
            logging.error(f"Observation start index {obs_start_idx} is less than 0 at current_step {self.current_step}")
            # Handle error, e.g., return a zero observation or pad
            return np.zeros((self.window_size, self.asset_dim), dtype=np.float32)

        obs_data = self.df.iloc[obs_start_idx:obs_end_idx].values.astype(np.float32)
        
        if not np.all(np.isfinite(obs_data)):
            # This ideally shouldn't happen if data loader cleans NaNs
            logging.warning(f"Step {self.current_step}: NaN/Inf found in observation data, replacing with 0.")
            obs_data = np.nan_to_num(obs_data, nan=0.0, posinf=0.0, neginf=0.0)
            
        return obs_data

    def _get_info(self):
        return {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.tolist(), # Convert numpy array to list for info
            "current_step": self.current_step
        }

    def render(self, mode='human'):
        if mode == 'human':
            actual_trading_step = self.current_step - self.window_size
            print(f"Trading Step: {actual_trading_step}, Portfolio Value: ${self.portfolio_value:,.2f}")
            # print(f"Weights: {['{:.2f}'.format(w) for w in self.weights]}") # Formatted weights
        else:
            super().render(mode=mode)

    def close(self):
        pass # Any cleanup operations