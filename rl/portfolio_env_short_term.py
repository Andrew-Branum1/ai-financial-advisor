import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnvShortTerm(gym.Env):
    """
    A short-term portfolio management environment for reinforcement learning.

    This environment simulates managing a portfolio of assets over time, with a focus
    on short-term metrics.

    Args:
        df (pd.DataFrame): A DataFrame with historical market data in a wide format.
            Must contain columns like 'TICKER_feature'.
        features_for_observation (list): A list of feature names (e.g., 'rsi', 'macd')
            that will be used to construct the agent's observation.
        window_size (int): The number of past time steps to include in the observation.
        initial_balance (float): The starting balance of the portfolio.
        transaction_cost_pct (float): The percentage cost for each transaction.
        volatility_target (float): The desired annualized volatility for the portfolio.
        turnover_penalty_weight (float): A penalty multiplier for high portfolio turnover.
        max_concentration_per_asset (float): The maximum weight any single asset can have.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, features_for_observation, window_size=30, initial_balance=100_000,
                 transaction_cost_pct=0.001, volatility_target=0.15,
                 turnover_penalty_weight=0.01, max_concentration_per_asset=0.35, **kwargs):
        super().__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.volatility_target = volatility_target
        self.turnover_penalty_weight = turnover_penalty_weight
        self.max_concentration_per_asset = max_concentration_per_asset
        
        # --- CORE FIX: SEPARATE OBSERVATION DATA FROM CALCULATION DATA ---
        
        # 1. Identify all tickers from the wide-format DataFrame.
        self.tickers = sorted(list(set([c.split('_')[0] for c in df.columns])))
        self.num_tickers = len(self.tickers)

        # 2. Create a specific list of columns that will form the agent's observation.
        self.features_for_observation = features_for_observation
        self.observation_columns = [f"{ticker}_{feat}" for ticker in self.tickers for feat in self.features_for_observation]
        
        # 3. Create a new, smaller DataFrame containing only the observation columns.
        #    This ensures that _get_observation() always returns the correct shape.
        self.df_observation = self.df[self.observation_columns].copy()
        
        # --- END FIX ---

        # Action space: weights for each ticker. The agent doesn't need to decide the cash weight.
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_tickers,), dtype=np.float32)

        # Observation space: defined by the window size and the number of features *for observation*.
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
        # Slices the pre-filtered observation DataFrame. This now guarantees the correct shape.
        obs = self.df_observation.iloc[self.current_step : self.current_step + self.window_size].values
        return obs

    def _get_reward(self, portfolio_return, volatility, turnover):
        """Calculates the reward for a given step."""
        # A simple Sharpe ratio-like reward, penalized for high turnover and deviation from volatility target.
        sharpe_ratio = portfolio_return / (volatility + 1e-7)
        turnover_penalty = self.turnover_penalty_weight * turnover
        volatility_penalty = abs(volatility - self.volatility_target)
        
        reward = sharpe_ratio - turnover_penalty - volatility_penalty
        return reward

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.weights = np.zeros(self.num_tickers)
        self.current_step = self.window_size
        self.done = False

        initial_observation = self._get_observation()
        info = {"message": "Environment reset."}

        return initial_observation, info

    def step(self, action):
        """Executes one time step within the environment."""
        # 1. Process Action: Normalize weights and apply constraints.
        target_weights = np.array(action)
        target_weights = np.clip(target_weights, 0, self.max_concentration_per_asset)
        
        # Normalize to sum to 1
        if np.sum(target_weights) > 0:
            target_weights = target_weights / np.sum(target_weights)
        
        # 2. Calculate Metrics
        previous_weights = self.weights.copy()
        turnover = np.sum(np.abs(target_weights - previous_weights))
        transaction_costs = self.portfolio_value * turnover * self.transaction_cost_pct

        # 3. Update Portfolio
        self.portfolio_value -= transaction_costs
        self.weights = target_weights

        # 4. Calculate Portfolio Return for the current day
        # We use the full `self.df` here to get the 'daily_return' columns.
        return_cols = [f"{ticker}_daily_return" for ticker in self.tickers]
        
        # Ensure we don't go out of bounds
        if self.current_step + self.window_size >= len(self.df):
            self.done = True
            # Return zero reward if we are at the end
            return self._get_observation(), 0.0, True, False, {}

        daily_asset_returns = self.df[return_cols].iloc[self.current_step + self.window_size].values
        daily_asset_returns = np.nan_to_num(daily_asset_returns) # Handle missing data
        
        portfolio_daily_return = np.dot(self.weights, daily_asset_returns)
        self.portfolio_value *= (1 + portfolio_daily_return)
        
        # 5. Calculate Volatility
        # Use the full `self.df` to get 'daily_return' columns for the past window.
        window_returns = self.df[return_cols].iloc[self.current_step : self.current_step + self.window_size]
        portfolio_window_returns = (window_returns * self.weights).sum(axis=1)
        volatility = portfolio_window_returns.std() * np.sqrt(252) # Annualized
        volatility = np.nan_to_num(volatility)

        # 6. Advance Time and Check for Completion
        self.current_step += 1
        if self.current_step >= len(self.df) - self.window_size - 1 or self.portfolio_value <= 0:
            self.done = True
        
        # 7. Calculate Reward and Get Next Observation
        reward = self._get_reward(portfolio_daily_return, volatility, turnover)
        observation = self._get_observation()
        
        info = {
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "volatility": volatility,
            "reward": reward,
        }

        return observation, reward, self.done, False, info
