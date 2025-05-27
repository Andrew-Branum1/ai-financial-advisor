import gym
from gym import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, window_size=30, initial_balance=1.0):
        super(PortfolioEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.asset_dim = df.shape[1]

        self.action_space = spaces.Box(low=0, high=1, shape=(self.asset_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, self.asset_dim), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.done = False
        self.weights = np.array([1.0 / self.asset_dim] * self.asset_dim)
        return self._get_observation()

    def step(self, action):
        action = np.clip(action, 0, 1)
        total = np.sum(action)
        if total == 0 or not np.isfinite(total):
            action = np.ones_like(action) / len(action)
        else:
            action = action / total

        prev_prices = self.df.iloc[self.current_step - 1].values
        new_prices = self.df.iloc[self.current_step].values

        if not np.all(np.isfinite(prev_prices)) or not np.all(np.isfinite(new_prices)):
            reward = 0.0  # skip bad data rows
        else:
            returns = new_prices / prev_prices - 1
            reward = np.dot(action, returns)
            self.portfolio_value *= (1 + reward)

        self.weights = action
        self.current_step += 1
        self.done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs)
        return obs
