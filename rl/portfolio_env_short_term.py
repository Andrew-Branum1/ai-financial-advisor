# rl/portfolio_env_short_term.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

class PortfolioEnvShortTerm(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, features_for_observation, window_size=30, initial_balance=100_000,
                 transaction_cost_pct=0.001, turnover_penalty_weight=0.01, 
                 max_concentration_per_asset=0.35, **kwargs):
        super().__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.turnover_penalty_weight = turnover_penalty_weight
        self.max_concentration_per_asset = max_concentration_per_asset
        
        self.tickers = sorted(list(set([c.split('_')[0] for c in df.columns])))
        self.num_tickers = len(self.tickers)

        self.features_for_observation = features_for_observation
        self.observation_columns = [f"{ticker}_{feat}" for ticker in self.tickers for feat in self.features_for_observation]
        self.df_observation = self.df[self.observation_columns].copy()
        
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_tickers,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.observation_columns)),
            dtype=np.float32,
        )

        self.current_step = 0
        self.weights = np.zeros(self.num_tickers)
        self.portfolio_value = self.initial_balance
        self.done = False
        self.portfolio_returns_history = deque(maxlen=20) 

    def _get_observation(self):
        end_idx = self.current_step + self.window_size
        obs = self.df_observation.iloc[self.current_step : end_idx].values
        return obs

    def _calculate_reward(self):
        history = list(self.portfolio_returns_history)
        if len(history) < 2:
            return 0.0
        history_np = np.array(history)
        sharpe_ratio = np.mean(history_np) / (np.std(history_np) + 1e-9)
        annualized_sharpe = sharpe_ratio * np.sqrt(252)
        return annualized_sharpe

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.portfolio_value = self.initial_balance
        self.weights = np.zeros(self.num_tickers)
        self.current_step = self.window_size
        self.done = False
        self.portfolio_returns_history.clear()

        initial_observation = self._get_observation()
        info = {"message": "Environment reset."}
        return initial_observation, info

    def step(self, action):
        # ensure it's a 1D array
        if isinstance(action, tuple) and len(action) > 0:
            target_weights = np.array(action[0]).flatten()
        else:
            target_weights = np.array(action).flatten()
            
        target_weights = np.clip(target_weights, 0, self.max_concentration_per_asset)

        #allow for holding cash
        cash_weight = 1.0 - np.sum(target_weights)
        if cash_weight < 0:
            target_weights = target_weights / np.sum(target_weights)
            cash_weight = 0
        
        previous_weights = self.weights.copy()
        turnover = np.sum(np.abs(target_weights - previous_weights))
        transaction_costs = self.portfolio_value * turnover * self.transaction_cost_pct

        self.portfolio_value -= transaction_costs
        self.weights = target_weights

        self.current_step += 1
        if self.current_step >= len(self.df) - self.window_size - 1 or self.portfolio_value <= 0:
            self.done = True
            return self._get_observation(), 0.0, True, False, {}

        return_cols = [f"{ticker}_daily_return" for ticker in self.tickers]
        daily_asset_returns = self.df[return_cols].iloc[self.current_step + self.window_size].values
        daily_asset_returns = np.nan_to_num(daily_asset_returns)
        
        portfolio_daily_return = np.dot(self.weights, daily_asset_returns)
        self.portfolio_value *= (1 + portfolio_daily_return)
        
        self.portfolio_returns_history.append(portfolio_daily_return)
        
        reward = self._calculate_reward()
        reward -= self.turnover_penalty_weight * turnover
        
        observation = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
            "reward": reward,
        }

        return observation, reward, self.done, False, info
