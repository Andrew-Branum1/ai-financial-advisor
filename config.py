# config.py
"""
Central configuration file for the AI Financial Advisor project.
This file serves as the single source of truth for all shared settings,
ensuring consistency across data collection, training, and evaluation scripts.
Enhanced for better short and long term prediction capabilities.
"""

# --- Data Configuration ---
# Define the universe of tickers the agent will be trained on.
# This list is now the single source of truth.
# In both config.py and config_aggressive.py

AGENT_TICKERS = [
    "AAPL",  # US Tech
    "MSFT",  # US Tech
    "GOOGL", # US Tech
    "NVDA",  # US Tech
    "META",  # US Tech
    "JNJ",   # US Healthcare
    "PFE",   # US Healthcare
    "LLY",   # US Healthcare
    "UNH",   # US Healthcare
    "MRK",   # US Healthcare
    "JPM",   # US Financials
    "BAC",   # US Financials
    "V",     # US Financials
    "XOM",   # US Energy
    "CVX",   # US Energy
    "AMZN",  # US Consumer
    "WMT",   # US Consumer
    "BABA",  # China
    "TM",    # Japan
    "SAP",   # Europe
]

# Define the ticker to be used for the buy-and-hold benchmark.
BENCHMARK_TICKER = "SPY"

# Define all features to be calculated and stored in the database.
# This ensures that all necessary data is available for experiments.
# Enhanced with additional features for better prediction
FEATURES_TO_CALCULATE = [
    "close",
    "rsi",
    "volatility_20",
    "bollinger_hband",
    "bollinger_lband",
    "bollinger_mavg",
    "atr",
    "sma_10",
    "sma_50",
    "macd",
    "macd_signal",
    "daily_return",
    "volatility_5",
    "momentum_10",
    "avg_volume_10",
    "close_vs_sma_50",
    "bollinger_width",
    "obv",
    "mfi",
    "close_vs_sma_10",
    # Additional features for enhanced prediction
    "ema_12",
    "ema_26",
    "stoch_k",
    "stoch_d",
    "williams_r",
    "cci",
    "adx",
    "volume_sma_ratio",
    "price_momentum_5",
    "volatility_ratio",
    "trend_strength",
]

# Define the specific subset of features the model will use for its observation space.
# Enhanced feature set for better prediction accuracy
FEATURES_TO_USE_IN_MODEL = [
    "close",
    "close_vs_sma_50",
    "mfi",
    "bollinger_width",
    "obv",
    "atr",
    "rsi",
    "macd",
    "volatility_20",
    "momentum_10",
    "daily_return",
    "volume_sma_ratio",
    "trend_strength",
]

# Best hyperparameters for the 5-stock portfolio.
# Optimized for long-term growth with risk management
BEST_PPO_PARAMS = {
    "learning_rate": 2.0221608802590983e-05,
    "n_epochs": 12,
    "gae_lambda": 0.9271494454220346,
    "ent_coef": 0.08684848168594887,
    "vf_coef": 0.46612025760783965,
    "gamma": 0.9946448619546133,
    "clip_range": 0.2720116965455601,
    "n_steps": 1024,
}

# Environment parameters for the 5-stock portfolio.
# Enhanced with additional parameters for better performance
ENV_PARAMS = {
    "window_size": 30,
    "initial_balance": 10000.0,
    "transaction_cost_pct": 0.001,
    "volatility_penalty_weight": 0.8287413582328335,
    "loss_aversion_factor": 2.2001849979841603,
    "rolling_volatility_window": 113,
    "turnover_penalty_weight": 0.12950481652566614,
    "max_concentration_per_asset": 0.9991397452797207,
    # Additional parameters for enhanced performance
    "sharpe_target": 1.5,  # Target Sharpe ratio
    "max_drawdown_limit": 0.25,  # Maximum acceptable drawdown
    "rebalancing_frequency": 5,  # Days between rebalancing
    "momentum_lookback": 20,  # Days for momentum calculation
    "mean_reversion_lookback": 60,  # Days for mean reversion signals
}
