# config.py
"""
Central configuration file for the AI Financial Advisor project.
This file serves as the single source of truth for all shared settings,
ensuring consistency across data collection, training, and evaluation scripts.
"""

# --- Data Configuration ---
# Define the universe of tickers the agent will be trained on.
# This list is now the single source of truth.
# In both config.py and config_aggressive.py

AGENT_TICKERS = [
    'MSFT', # Technology
    'NVDA', # Technology (High-Growth/Volatility)
    'AMZN', # Consumer Discretionary
    'JNJ',  # Healthcare (Stable/Defensive)
    'JPM',  # Financials
]

# Define the ticker to be used for the buy-and-hold benchmark.
BENCHMARK_TICKER = 'SPY'

# Define all features to be calculated and stored in the database.
# This ensures that all necessary data is available for experiments.
# In config.py
FEATURES_TO_CALCULATE = [
    'close', 'rsi', 'volatility_20',
    'bollinger_hband', 'bollinger_lband', 'bollinger_mavg', 'atr',
    'sma_10', 'sma_50', 'macd', 'macd_signal', 'daily_return',
    'volatility_5', 'momentum_10', 'avg_volume_10',
    'close_vs_sma_50', 'bollinger_width',
    'obv', 'mfi', 'close_vs_sma_10'  # <-- ADD NEW VOLUME FEATURES HERE
]
# Define the specific subset of features the model will use for its observation space.
# This allows for easy experimentation without changing the data collector.
# In config.py
FEATURES_TO_USE_IN_MODEL = [
    'close',
    'close_vs_sma_50',
    'mfi',
    'bollinger_width',
    'obv',
    'atr'
]

# Best hyperparameters for the 5-stock portfolio.
BEST_PPO_PARAMS = {
    'learning_rate': 2.0221608802590983e-05,
    'n_epochs': 12,
    'gae_lambda': 0.9271494454220346,
    'ent_coef': 0.08684848168594887,
    'vf_coef': 0.46612025760783965,
    'gamma': 0.9946448619546133,
    'clip_range': 0.2720116965455601,
    'n_steps': 1024,
}

# Environment parameters for the 5-stock portfolio.
ENV_PARAMS = {
    'window_size': 30,
    'initial_balance': 10000.0,
    'transaction_cost_pct': 0.001,
    'volatility_penalty_weight': 0.8287413582328335,
    'loss_aversion_factor': 2.2001849979841603,
    'rolling_volatility_window': 113,
    'turnover_penalty_weight': 0.12950481652566614,
    'max_concentration_per_asset': 0.9991397452797207,
}