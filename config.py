# config.py
"""
Central configuration file for the AI Financial Advisor project.
This file serves as the single source of truth for all shared settings,
ensuring consistency across data collection, training, and evaluation scripts.
"""

# --- Data Configuration ---
# Define the universe of tickers the agent will be trained on.
# This list is now the single source of truth.
AGENT_TICKERS = ['AAPL', 'MSFT', 'GOOGL']

# Define the ticker to be used for the buy-and-hold benchmark.
BENCHMARK_TICKER = 'SPY'

# Define all features to be calculated and stored in the database.
# This ensures that all necessary data is available for experiments.
FEATURES_TO_CALCULATE = [
    'close', 'rsi', 'volatility_20',
    'bollinger_hband', 'bollinger_lband', 'bollinger_mavg', 'atr',
    'sma_10', 'sma_50', 'macd', 'macd_signal', 'daily_return',
    'volatility_5', 'momentum_10', 'avg_volume_10'
]

# Define the specific subset of features the model will use for its observation space.
# This allows for easy experimentation without changing the data collector.
FEATURES_TO_USE_IN_MODEL = [
    'close', 'rsi', 'volatility_20',
    'bollinger_hband', 'bollinger_lband', 'atr'
]


# --- Environment & Model Hyperparameters ---

# Best hyperparameters found from the Optuna study.
# This is the single source for the final model's configuration.
BEST_PPO_PARAMS = {
    'learning_rate': 0.00019990304577983145,
    'n_epochs': 17,
    'gae_lambda': 0.9383531599343731,
    'ent_coef': 0.052884046332963695,
    'vf_coef': 0.36980761456220457,
    'gamma': 0.9943256589140542,
    'clip_range': 0.3430035418999476,
    'n_steps': 512,
}

# Environment parameters, including reward shaping and safety constraints.
ENV_PARAMS = {
    'window_size': 30,
    'volatility_penalty_weight': 0.5246614265918629,
    'loss_aversion_factor': 2.3008289743538675,
    'rolling_volatility_window': 139,
    # Safety constraints
    'max_concentration_per_asset': 0.50,
    'turnover_penalty_weight': 0.1,
    'initial_balance': 10000.0, 
    'transaction_cost_pct': 0.001
}
