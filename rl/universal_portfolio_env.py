# rl/universal_portfolio_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class UniversalPortfolioEnv(gym.Env):
    """
    A scalable portfolio environment that works with a large, dynamic universe of stocks.
    The observation space is a dictionary, and the environment expects a custom
    policy that can handle this structure (e.g., an Attention/Transformer model).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        window_size: int = 30,
        initial_balance: float = 10000.0,
        top_k_stocks: int = 20,  # The agent will allocate among the top K ranked stocks
        transaction_cost_pct: float = 0.001,
        reward_scaling: float = 1e-4,
        # Parameters for different goals
        sharpe_window: int = 252,  # For long-term goals
        drawdown_penalty_weight: float = 0.5,
    ):

        super().__init__()
        self.df = df
        self.feature_columns = feature_columns
        self.num_features = len(feature_columns)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.top_k_stocks = top_k_stocks
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.sharpe_window = sharpe_window
        self.drawdown_penalty_weight = drawdown_penalty_weight

        # Unique dates and tickers in the dataset
        self.trade_dates = sorted(self.df.index.unique())
        self.all_tickers = sorted(self.df["ticker"].unique())

        # Action space: weights for the top K stocks
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.top_k_stocks,), dtype=np.float32
        )

        # Observation space: Dictionary space
        # We define a placeholder shape. The custom policy will handle variable numbers of stocks.
        self.observation_space = spaces.Dict(
            {
                "features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self.all_tickers), self.window_size, self.num_features),
                    dtype=np.float32,
                ),
                "mask": spaces.Box(
                    low=0, high=1, shape=(len(self.all_tickers),), dtype=np.int8
                ),  # Mask for stocks active on a given day
            }
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_index = self.window_size
        self.portfolio_value = self.initial_balance
        self.portfolio_composition = {}  # Will hold {ticker: shares}
        self.daily_returns_history = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        top_k_indices, allocation_weights = action

        # --- FIX 1: Sanitize Agent's Action ---
        # If the model outputs nan/inf, replace them with zeros and clip to a valid range.
        if not np.all(np.isfinite(allocation_weights)):
            allocation_weights = np.nan_to_num(
                allocation_weights, nan=0.0, posinf=1.0, neginf=0.0
            )
        allocation_weights = np.clip(
            allocation_weights, 0, 1e6
        )  # Clip to a large but finite number

        # Normalize weights to sum to 1
        if np.sum(allocation_weights) > 1e-8:
            allocation_weights = allocation_weights / np.sum(allocation_weights)
        else:
            allocation_weights = np.zeros(self.top_k_stocks)

        # --- (Liquidation logic is fine) ---
        current_date = self.trade_dates[self.current_step_index]
        prev_date = self.trade_dates[self.current_step_index - 1]
        prev_data = self.df.loc[prev_date].set_index("ticker")

        liquidated_value = 0
        for ticker, shares in self.portfolio_composition.items():
            if ticker in prev_data.index:
                liquidated_value += shares * prev_data.loc[ticker, "close"]

        cash = (
            self.portfolio_value if not self.portfolio_composition else liquidated_value
        )

        # --- FIX 2: Check for Catastrophic Portfolio State ---
        # If cash is not a finite number or is bankrupt, the episode has failed.
        if not np.isfinite(cash) or cash <= 0:
            # End the episode immediately with a large penalty.
            terminated = True
            truncated = True
            return (
                self._get_observation(),
                -1e6,
                terminated,
                truncated,
                self._get_info(),
            )

        # --- (New portfolio calculation is fine) ---
        self.portfolio_composition = {}
        target_tickers = [self.all_tickers[i] for i in top_k_indices]
        current_data_indexed = self.df.loc[current_date].set_index("ticker")
        for i, ticker in enumerate(target_tickers):
            if ticker in current_data_indexed.index:
                investment = cash * allocation_weights[i]
                price = current_data_indexed.loc[ticker, "close"]
                investment -= investment * self.transaction_cost_pct
                self.portfolio_composition[ticker] = investment / price

        # --- (Next day value calculation is fine) ---
        next_date = self.trade_dates[self.current_step_index + 1]
        next_data = self.df.loc[next_date].set_index("ticker")

        next_day_value = 0
        for ticker, shares in self.portfolio_composition.items():
            if ticker in next_data.index:
                next_day_value += shares * next_data.loc[ticker, "close"]

        # --- FIX 3: Sanitize and Clamp the Daily Return ---
        if not np.isfinite(next_day_value):
            daily_return = -1.0  # Penalize for inf portfolio value
        else:
            daily_return = (next_day_value / cash) - 1 if cash > 0 else 0

        # Clamp return to a realistic range to prevent single-step explosions from affecting reward scaling
        daily_return = np.clip(daily_return, -1.0, 10.0)

        self.daily_returns_history.append(daily_return)
        self.portfolio_value = (
            next_day_value if np.isfinite(next_day_value) else self.portfolio_value
        )

        reward = self._calculate_reward()

        # --- FIX 4: Final Check on Reward ---
        if not np.isfinite(reward):
            reward = 0.0  # Use a neutral reward if calculation fails, to avoid poisoning the agent

        self.current_step_index += 1
        terminated = self.current_step_index >= len(self.trade_dates) - 2
        truncated = terminated

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # ... inside the UniversalPortfolioEnv class ...
    def _get_observation(self):
        end_idx = self.current_step_index
        start_idx = end_idx - self.window_size

        window_dates = self.trade_dates[start_idx:end_idx]
        obs_df = self.df[self.df.index.isin(window_dates)]

        features_array = np.zeros(
            self.observation_space["features"].shape, dtype=np.float32
        )
        mask_array = np.zeros(self.observation_space["mask"].shape, dtype=np.int8)

        last_day_of_window = window_dates[-1]
        active_tickers_today = self.df.loc[last_day_of_window]["ticker"].unique()

        for i, ticker in enumerate(self.all_tickers):
            if ticker in active_tickers_today:
                ticker_df = obs_df[obs_df["ticker"] == ticker][self.feature_columns]
                if len(ticker_df) == self.window_size:

                    # --- ADD THIS NORMALIZATION STEP ---
                    # Normalize each feature window independently.
                    # This prevents lookahead bias and handles different scales.
                    normalized_df = (ticker_df - ticker_df.mean()) / (
                        ticker_df.std() + 1e-8
                    )

                    features_array[i, :, :] = normalized_df.values
                    mask_array[i] = 1

        return {"features": features_array, "mask": mask_array}

    # ... rest of the class ...

    def _calculate_reward(self):
        # Example for a LONG-TERM goal (customize for others)
        if len(self.daily_returns_history) < self.sharpe_window:
            return 0.0

        recent_returns = np.array(self.daily_returns_history[-self.sharpe_window :])
        sharpe_ratio = (
            np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
        ) * np.sqrt(252)

        # Penalize for large drawdowns
        portfolio_values = (
            self.initial_balance * (1 + np.array(self.daily_returns_history)).cumprod()
        )
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        drawdown_penalty = np.min(drawdown) * self.drawdown_penalty_weight

        reward = sharpe_ratio + drawdown_penalty
        return reward * self.reward_scaling

    def _get_info(self):
        return {
            "portfolio_value": self.portfolio_value,
            "date": self.trade_dates[self.current_step_index],
        }
