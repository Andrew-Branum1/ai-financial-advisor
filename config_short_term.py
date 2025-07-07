# config_short_term.py
"""
Configuration for the ACTIVE, short-term RL agent.
Enhanced with additional features for better short-term prediction.
"""

# --- Data Configuration ---
# Expanded to 20 stocks for better diversification and more options
AGENT_TICKERS = [
    "AAPL", "AMZN", "BAC", "CVX", "GOOGL", "JNJ", "JPM", "LLY", "MRK", "MSFT",
    "NVDA", "PFE", "SAP", "SPY", "TM", "UNH", "V", "WMT", "XOM"
]
BENCHMARK_TICKER = "SPY"
FEATURES_TO_USE_IN_MODEL = [
    "close",
    "volume",
    "daily_return",
    "close_vs_sma_10",
    "close_vs_sma_20",
    "close_vs_sma_50",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bollinger_width",
    "bollinger_position",
    "volume_ratio",
    "obv",
    "atr",
    "stoch_k",
    "stoch_d",
    "williams_r",
    "cci",
    "mfi",
    "adx",
    "trend_strength",
    "volatility_ratio",
    "roc_10",
    "roc_20",
    "volume_roc",
    "price_efficiency",
    "mean_reversion_signal",
]

# --- Enhanced Hyperparameters for Short-Term Active Trading ---
BEST_PPO_PARAMS = {
    "learning_rate": 0.0009042436291875808,
    "n_epochs": 11,
    "gae_lambda": 0.9271494454220346,
    "ent_coef": 0.0164583897291383,
    "vf_coef": 0.46612025760783965,
    "gamma": 0.9414231682044852,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "batch_size": 64,
    "n_steps": 2048,
    "policy_kwargs": {
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        "activation_fn": "relu",
    },
}

# --- Environment Parameters for Short-Term Active Trading ---
ENV_PARAMS = {
    "initial_balance": 100000,
    "transaction_cost_pct": 0.001,
    "window_size": 30,
    "max_position_pct": 0.3,
    "min_holding_period": 1,
    "max_holding_period": 5,
    "volatility_target": 0.15,
    "drawdown_penalty": 0.10,
    "momentum_window": 20,
    "mean_reversion_window": 60,
}
