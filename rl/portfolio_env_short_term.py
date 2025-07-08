# rl/portfolio_env_short_term.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

import collections


class PortfolioEnvShortTerm(gym.Env):
    """
    An enhanced environment that rewards the agent based on multiple factors:
    - Rolling Sharpe ratio for short-term performance
    - Momentum and mean reversion signals
    - Volatility targeting
    - Enhanced risk management
    """

    metadata = {"render_modes": ["human"]}


    def __init__(self, df: pd.DataFrame, feature_columns_ordered: list, **kwargs):
        super().__init__()


        if not isinstance(df, pd.DataFrame) or df.empty:

            raise ValueError("Input DataFrame `df` cannot be empty.")

        # --- ENV SETUP ---
        self.df = df
        self.feature_names = feature_columns_ordered

        self.tickers = sorted(list(set(col.split("_")[0] for col in df.columns)))

        self.asset_dim = len(self.tickers)
        self.num_features = len(self.feature_names)

        # --- KWARGS WITH DEFAULTS ---

        self.window_size = int(kwargs.get("window_size", 30))
        self.initial_balance = float(kwargs.get("initial_balance", 10000.0))
        self.transaction_cost_pct = float(kwargs.get("transaction_cost_pct", 0.001))
        self.rolling_volatility_window = int(
            kwargs.get("rolling_volatility_window", 20)
        )
        self.max_concentration_per_asset = float(
            kwargs.get("max_concentration_per_asset", 1.0)
        )
        self.turnover_penalty_weight = float(
            kwargs.get("turnover_penalty_weight", 0.01)
        )

        # Enhanced parameters
        self.momentum_weight = float(kwargs.get("momentum_weight", 0.3))
        self.mean_reversion_weight = float(kwargs.get("mean_reversion_weight", 0.2))
        self.volatility_target = float(kwargs.get("volatility_target", 0.15))
        self.momentum_lookback = int(kwargs.get("momentum_lookback", 20))
        self.mean_reversion_lookback = int(kwargs.get("mean_reversion_lookback", 60))

        # --- SPACES ---
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.asset_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.asset_dim * self.num_features),
            dtype=np.float32,
        )

        # --- STATE VARIABLES ---
        self.recent_portfolio_returns = collections.deque(
            maxlen=self.rolling_volatility_window
        )
        self.portfolio_values_history = collections.deque(
            maxlen=max(self.momentum_lookback, self.mean_reversion_lookback)
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.weights = np.full(self.asset_dim, 1.0 / self.asset_dim, dtype=np.float32)
        self.previous_weights = self.weights.copy()
        self.recent_portfolio_returns.clear()

        self.portfolio_values_history.clear()
        self.portfolio_values_history.append(self.initial_balance)

        return self._get_observation(), self._get_info()


    def step(self, action: np.ndarray):
        self.previous_weights = self.weights.copy()
        action = np.clip(action, 0, self.max_concentration_per_asset)

        # --- ENFORCE TOP-5 SELECTION ---
        top_n = 5
        if len(action) > top_n:
            top_indices = np.argsort(action)[-top_n:]
            mask = np.zeros_like(action)
            mask[top_indices] = 1
            action = action * mask

        action_sum = np.sum(action)
        if action_sum > 1e-6:
            self.weights = action / action_sum
        else:

            self.weights = np.full(
                self.asset_dim, 1.0 / self.asset_dim, dtype=np.float32
            )

        # Calculate transaction costs

        turnover = np.sum(np.abs(self.weights - self.previous_weights))
        costs = self.portfolio_value * turnover * self.transaction_cost_pct
        self.portfolio_value -= costs

        # Calculate portfolio return for this step

        try:
            close_col_names = [col for col in self.df.columns if col.endswith("_close")]

            close_prices_df = self.df[close_col_names]
            prev_close_prices = close_prices_df.iloc[self.current_step - 1].values
            current_close_prices = close_prices_df.iloc[self.current_step].values
            daily_asset_returns = (current_close_prices / prev_close_prices) - 1
        except Exception:

            daily_asset_returns = np.zeros(self.asset_dim)

        portfolio_daily_return = np.dot(self.weights, daily_asset_returns)
        self.portfolio_value *= 1 + portfolio_daily_return
        # Ensure portfolio_daily_return is a scalar (float)
        if isinstance(portfolio_daily_return, (np.ndarray, list)):
            scalar_return = float(np.sum(portfolio_daily_return))
        else:
            scalar_return = float(portfolio_daily_return)
        self.recent_portfolio_returns.append(scalar_return)
        self.portfolio_values_history.append(self.portfolio_value)

        # Use scalar_return for reward calculation
        final_reward = self._calculate_enhanced_reward(scalar_return, turnover)


        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = terminated


        info = self._get_info()
        info["transaction_costs"] = costs
        info["raw_daily_return"] = portfolio_daily_return
        info["turnover"] = turnover

        return self._get_observation(), float(final_reward), terminated, truncated, info

    def _calculate_enhanced_reward(self, daily_return: float, turnover: float) -> float:
        """
        Enhanced reward function incorporating multiple factors for better prediction.
        """
        reward = 0.0

        # 1. Sharpe ratio component (short-term focus)
        if len(self.recent_portfolio_returns) == self.rolling_volatility_window:
            recent_returns = np.array(list(self.recent_portfolio_returns))
            if np.std(recent_returns) > 1e-6:
                sharpe_component = np.mean(recent_returns) / np.std(recent_returns)
                reward += float(sharpe_component)

        # 2. Momentum component
        if len(self.portfolio_values_history) >= self.momentum_lookback:
            recent_values = list(self.portfolio_values_history)[
                -self.momentum_lookback :
            ]
            if len(recent_values) >= 2:
                momentum = (recent_values[-1] / recent_values[0]) - 1
                reward += self.momentum_weight * float(momentum)

        # 3. Mean reversion component
        if len(self.portfolio_values_history) >= self.mean_reversion_lookback:
            long_term_values = list(self.portfolio_values_history)[
                -self.mean_reversion_lookback :
            ]
            if len(long_term_values) >= 2:
                long_term_return = (long_term_values[-1] / long_term_values[0]) - 1
                # Penalize if too far from long-term trend
                mean_reversion_penalty = -self.mean_reversion_weight * abs(
                    long_term_return
                )
                reward += float(mean_reversion_penalty)

        # 4. Volatility targeting
        if len(self.recent_portfolio_returns) >= 20:
            current_vol = np.std(self.recent_portfolio_returns) * np.sqrt(252)
            vol_penalty = -abs(current_vol - self.volatility_target) * 0.1
            reward += float(vol_penalty)

        # 5. Turnover penalty
        reward -= self.turnover_penalty_weight * turnover

        # 6. Drawdown penalty
        if len(self.portfolio_values_history) >= 2:
            peak = max(self.portfolio_values_history)
            current_drawdown = (self.portfolio_value - peak) / peak
            if current_drawdown < -0.1:  # Penalize drawdowns > 10%
                reward += float(current_drawdown * 2.0)

        return float(reward)

    def _get_observation(self):
        obs_window = self.df.iloc[
            self.current_step - self.window_size : self.current_step
        ].values.astype(np.float32)
        obs_mean = np.nanmean(obs_window, axis=0)
        obs_std = np.nanstd(obs_window, axis=0)
        obs_std_safe = np.where((obs_std == 0) | np.isnan(obs_std), 1.0, obs_std)
        obs_norm = (obs_window - obs_mean) / obs_std_safe
        obs_norm = np.where(np.isnan(obs_norm), 0, obs_norm)
        return obs_norm

    def _get_info(self):
        return {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.tolist(),
        }

    def close(self):
        pass
