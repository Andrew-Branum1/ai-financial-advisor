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
    'obv', 'mfi'  # <-- ADD NEW VOLUME FEATURES HERE
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


# --- Environment & Model Hyperparameters ---
BEST_PPO_PARAMS = {
    'learning_rate': 1.007366221655747e-05,
    'n_epochs': 16,
    'gae_lambda': 0.9987285068366246,
    'ent_coef': 0.017908276296944192,
    'vf_coef': 0.38887640327346684,
    'gamma': 0.9950391730877265,
    'clip_range': 0.19740440204467932,
    'n_steps': 1024,
}

# Environment parameters discovered from the final Optuna study.
ENV_PARAMS = {
    'window_size': 30,
    'initial_balance': 10000.0,
    'transaction_cost_pct': 0.001,
    'volatility_penalty_weight': 0.003142761800601024,
    'loss_aversion_factor': 1.6304835502462258,
    'rolling_volatility_window': 113,
    'turnover_penalty_weight': 0.05,
    'max_concentration_per_asset': 1,
}