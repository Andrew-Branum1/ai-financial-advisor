# rl/portfolio_env_long_term.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import collections


class PortfolioEnvLongTerm(gym.Env):
    """
    A sophisticated environment optimized for long-term stock market prediction.
    Features include:
    - Multi-timeframe analysis
    - Risk parity allocation
    - Sector rotation capabilities
    - Dynamic rebalancing
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
        self.window_size = int(kwargs.get("window_size", 60))
        self.initial_balance = float(kwargs.get("initial_balance", 10000.0))
        self.transaction_cost_pct = float(kwargs.get("transaction_cost_pct", 0.001))
        self.rolling_volatility_window = int(
            kwargs.get("rolling_volatility_window", 252)
        )
        self.max_concentration_per_asset = float(
            kwargs.get("max_concentration_per_asset", 0.4)
        )
        self.turnover_penalty_weight = float(
            kwargs.get("turnover_penalty_weight", 0.01)
        )

        # Long-term specific parameters
        self.sharpe_target = float(kwargs.get("sharpe_target", 1.2))
        self.max_drawdown_limit = float(kwargs.get("max_drawdown_limit", 0.20))
        self.rebalancing_frequency = int(kwargs.get("rebalancing_frequency", 10))
        self.momentum_lookback = int(kwargs.get("momentum_lookback", 60))
        self.mean_reversion_lookback = int(kwargs.get("mean_reversion_lookback", 252))
        self.growth_weight = float(kwargs.get("growth_weight", 0.4))
        self.stability_weight = float(kwargs.get("stability_weight", 0.3))
        self.diversification_weight = float(kwargs.get("diversification_weight", 0.3))

        # Strategy parameters
        self.min_holding_period = int(kwargs.get("min_holding_period", 30))
        self.trend_following_weight = float(kwargs.get("trend_following_weight", 0.6))
        self.value_weight = float(kwargs.get("value_weight", 0.4))
        self.sector_rotation_enabled = bool(kwargs.get("sector_rotation_enabled", True))
        self.risk_parity_enabled = bool(kwargs.get("risk_parity_enabled", True))
        self.dynamic_rebalancing = bool(kwargs.get("dynamic_rebalancing", True))

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
        self.holding_periods = np.zeros(self.asset_dim)  # Track holding periods
        self.sector_allocations = {}  # Track sector allocations
        self.risk_contributions = (
            np.ones(self.asset_dim) / self.asset_dim
        )  # Risk parity weights

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
        self.holding_periods.fill(0)
        self.last_rebalancing_step = self.current_step

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray):
        self.previous_weights = self.weights.copy()
        # --- ENFORCE TOP-5 SELECTION ---
        top_n = 5
        if len(action) > top_n:
            top_indices = np.argsort(action)[-top_n:]
            mask = np.zeros_like(action)
            mask[top_indices] = 1
            action = action * mask
        # Apply minimum holding period constraint
        action = self._apply_holding_period_constraints(action)

        # Apply concentration limits
        action = np.clip(action, 0, self.max_concentration_per_asset)
        action_sum = np.sum(action)
        if action_sum > 1e-6:
            self.weights = action / action_sum
        else:
            self.weights = np.full(
                self.asset_dim, 1.0 / self.asset_dim, dtype=np.float32
            )

        # Apply risk parity if enabled
        if self.risk_parity_enabled:
            self.weights = self._apply_risk_parity(self.weights)

        # Calculate transaction costs
        turnover = np.sum(np.abs(self.weights - self.previous_weights))
        costs = self.portfolio_value * turnover * self.transaction_cost_pct
        self.portfolio_value -= costs

        # Calculate portfolio return
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
        self.recent_portfolio_returns.append(portfolio_daily_return)
        self.portfolio_values_history.append(self.portfolio_value)

        # Update holding periods
        self.holding_periods += 1
        self.holding_periods[self.weights < 0.01] = 0  # Reset for sold positions

        # Calculate enhanced reward
        final_reward = self._calculate_long_term_reward(
            portfolio_daily_return, turnover
        )

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = terminated

        info = self._get_info()
        info["transaction_costs"] = costs
        info["raw_daily_return"] = portfolio_daily_return
        info["turnover"] = turnover
        info["holding_periods"] = self.holding_periods.tolist()

        return self._get_observation(), float(final_reward), terminated, truncated, info

    def _apply_holding_period_constraints(self, action):
        action = np.asarray(action).reshape(-1)
        self.previous_weights = np.asarray(self.previous_weights).reshape(-1)
        if action.shape != self.previous_weights.shape:
            raise ValueError(
                f"Shape mismatch: action {action.shape}, previous_weights {self.previous_weights.shape}"
            )
        constrained_action = np.copy(action)

        for i in range(self.asset_dim):
            if (
                self.holding_periods[i] < self.min_holding_period
                and self.previous_weights[i] > 0.01
            ):
                # If holding period is too short, maintain current weight
                constrained_action[i] = self.previous_weights[i]

        return constrained_action

    def _apply_risk_parity(self, weights: np.ndarray) -> np.ndarray:
        """Apply risk parity allocation to equalize risk contributions."""
        if len(self.recent_portfolio_returns) < 20:
            return weights

        # Calculate asset volatilities
        asset_returns = []
        for i, ticker in enumerate(self.tickers):
            try:
                close_col = f"{ticker}_close"
                if close_col in self.df.columns:
                    prices = (
                        self.df[close_col]
                        .iloc[self.current_step - 20 : self.current_step]
                        .values
                    )
                    returns = np.diff(prices) / prices[:-1]
                    asset_returns.append(returns)
                else:
                    asset_returns.append(np.zeros(19))
            except:
                asset_returns.append(np.zeros(19))

        asset_volatilities = np.array([np.std(returns) for returns in asset_returns])

        # Risk parity weights: inverse of volatility
        risk_weights = 1.0 / (asset_volatilities + 1e-8)
        risk_weights = risk_weights / np.sum(risk_weights)

        # Blend with original weights
        blended_weights = 0.7 * weights + 0.3 * risk_weights
        return blended_weights / np.sum(blended_weights)

    def _calculate_long_term_reward(
        self, daily_return: float, turnover: float
    ) -> float:
        """
        Enhanced reward function for long-term growth with multiple objectives.
        """
        reward = 0.0

        # 1. Long-term Sharpe ratio component
        if len(self.recent_portfolio_returns) == self.rolling_volatility_window:
            recent_returns = np.array(list(self.recent_portfolio_returns))
            if np.std(recent_returns) > 1e-6:
                sharpe_component = np.mean(recent_returns) / np.std(recent_returns)
                reward += self.growth_weight * float(sharpe_component)

        # 2. Growth component (long-term momentum)
        if len(self.portfolio_values_history) >= self.momentum_lookback:
            recent_values = list(self.portfolio_values_history)[
                -self.momentum_lookback :
            ]
            if len(recent_values) >= 2:
                momentum = (recent_values[-1] / recent_values[0]) - 1
                reward += self.growth_weight * float(momentum)

        # 3. Stability component (low volatility)
        if len(self.recent_portfolio_returns) >= 20:
            current_vol = np.std(self.recent_portfolio_returns) * np.sqrt(252)
            stability_bonus = max(0, 1.0 - current_vol)  # Reward low volatility
            reward += self.stability_weight * stability_bonus

        # 4. Diversification component
        concentration_penalty = -np.sum(self.weights**2)  # Herfindahl index
        reward += self.diversification_weight * concentration_penalty

        # 5. Drawdown penalty
        if len(self.portfolio_values_history) >= 2:
            peak = max(self.portfolio_values_history)
            current_drawdown = (self.portfolio_value - peak) / peak
            if current_drawdown < -self.max_drawdown_limit:
                reward += float(
                    current_drawdown * 5.0
                )  # Heavy penalty for large drawdowns

        # 6. Turnover penalty (higher for long-term strategies)
        reward -= self.turnover_penalty_weight * turnover

        # 7. Trend following component
        if len(self.portfolio_values_history) >= 30:
            short_trend = self._calculate_trend(30)
            long_trend = self._calculate_trend(90)
            trend_alignment = np.sign(short_trend) == np.sign(long_trend)
            if trend_alignment:
                reward += self.trend_following_weight * abs(short_trend)

        return float(reward)

    def _calculate_trend(self, lookback: int) -> float:
        """Calculate trend strength over a given lookback period."""
        if len(self.portfolio_values_history) < lookback:
            return 0.0

        values = list(self.portfolio_values_history)[-lookback:]
        if len(values) < 2:
            return 0.0

        # Linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope / values[0])  # Normalize by initial value

    def _get_observation(self):
        obs_window = self.df.iloc[
            self.current_step - self.window_size : self.current_step
        ].values.astype(np.float32)
        obs_mean = np.nanmean(obs_window, axis=0)
        obs_std = np.nanstd(obs_window, axis=0)
        obs_std_safe = np.where((obs_std == 0) | np.isnan(obs_std), 1.0, obs_std)
        obs_norm = (obs_window - obs_mean) / obs_std_safe
        obs_norm = np.where(np.isnan(obs_norm), 0, obs_norm)
        obs_norm = np.clip(obs_norm, -3, 3)
        return obs_norm

    def _get_info(self):
        return {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.tolist(),
            "holding_periods": self.holding_periods.tolist(),
        }

    def close(self):
        pass
