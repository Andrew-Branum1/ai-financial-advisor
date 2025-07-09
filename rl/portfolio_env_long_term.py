import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnvLongTerm(gym.Env):
    """
    A long-term portfolio management environment for reinforcement learning.

    This environment is designed for strategic, long-horizon asset allocation,
    penalizing frequent trading and rewarding stable, risk-adjusted growth.

    Args:
        df (pd.DataFrame): DataFrame with historical market data in a wide format.
        features_for_observation (list): List of feature names to construct the observation.
        window_size (int): The number of past time steps for the observation.
        rebalancing_frequency (int): How often (in days) the agent is allowed to rebalance.
        initial_balance (float): The starting portfolio balance.
        transaction_cost_pct (float): The percentage cost for each transaction.
        turnover_penalty_weight (float): Penalty for high portfolio turnover.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, features_for_observation, window_size=60, rebalancing_frequency=20,
                 initial_balance=100_000, transaction_cost_pct=0.001,
                 turnover_penalty_weight=0.02, **kwargs):
        super().__init__()

        self.df = df
        self.window_size = window_size
        self.rebalancing_frequency = rebalancing_frequency
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.turnover_penalty_weight = turnover_penalty_weight
        
        # --- CORE FIX: SEPARATE OBSERVATION DATA FROM CALCULATION DATA ---
        
        self.tickers = sorted(list(set([c.split('_')[0] for c in df.columns])))
        self.num_tickers = len(self.tickers)

        self.features_for_observation = features_for_observation
        self.observation_columns = [f"{ticker}_{feat}" for ticker in self.tickers for feat in self.features_for_observation]
        
        self.df_observation = self.df[self.observation_columns].copy()
        
        # --- END FIX ---

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_tickers,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.observation_columns)),
            dtype=np.float32,
        )

        # Initialize state variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.weights = np.zeros(self.num_tickers)
        self.portfolio_value = self.initial_balance
        self.done = False

    def _get_observation(self):
        """Returns the observation for the current step."""
        obs = self.df_observation.iloc[self.current_step : self.current_step + self.window_size].values
        return obs

    def _get_reward(self, portfolio_returns):
        """
        Calculates the reward, focusing on long-term risk-adjusted returns.
        Uses Sortino ratio to penalize downside volatility more heavily.
        """
        # Calculate downside returns
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std()
        
        # If there's no downside deviation, Sortino is infinite; return a large positive number.
        if downside_std == 0 or np.isnan(downside_std):
            return portfolio_returns.mean() * 100

        sortino_ratio = portfolio_returns.mean() / (downside_std + 1e-7)
        return sortino_ratio

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.weights = np.zeros(self.num_tickers)
        self.current_step = self.window_size
        self.done = False
        self.last_rebalance_step = self.window_size

        initial_observation = self._get_observation()
        info = {"message": "Environment reset."}

        return initial_observation, info

    def step(self, action):
        """Executes one time step within the environment."""
        self.current_step += 1
        
        # Check for end of data
        if self.current_step >= len(self.df) - self.window_size - 1 or self.portfolio_value <= 0:
            self.done = True
            return self._get_observation(), 0.0, True, False, {}

        # Only allow rebalancing at specified intervals
        if (self.current_step - self.last_rebalance_step) >= self.rebalancing_frequency:
            target_weights = np.array(action)
            
            # Normalize to sum to 1
            if np.sum(target_weights) > 0:
                target_weights = target_weights / np.sum(target_weights)
            
            previous_weights = self.weights.copy()
            turnover = np.sum(np.abs(target_weights - previous_weights))
            transaction_costs = self.portfolio_value * turnover * self.transaction_cost_pct
            
            self.portfolio_value -= transaction_costs
            self.weights = target_weights
            self.last_rebalance_step = self.current_step
        else:
            # If not a rebalancing day, turnover is zero
            turnover = 0

        # Calculate portfolio return based on current weights
        return_cols = [f"{ticker}_daily_return" for ticker in self.tickers]
        daily_asset_returns = self.df[return_cols].iloc[self.current_step + self.window_size].values
        daily_asset_returns = np.nan_to_num(daily_asset_returns)
        
        portfolio_daily_return = np.dot(self.weights, daily_asset_returns)
        self.portfolio_value *= (1 + portfolio_daily_return)

        # Calculate reward based on the performance over the rebalancing period
        window_start = self.last_rebalance_step
        window_end = self.current_step + self.window_size
        
        window_returns_df = self.df[return_cols].iloc[window_start:window_end]
        portfolio_window_returns = (window_returns_df * self.weights).sum(axis=1)
        
        reward = self._get_reward(portfolio_window_returns)
        
        # Penalize turnover only on rebalancing days
        reward -= self.turnover_penalty_weight * turnover
        
        observation = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "reward": reward,
        }

        return observation, reward, self.done, False, info
