# config_alpha.py
"""
Configuration for the final, alpha-seeking RL agent.
"""

# --- Data Configuration ---
AGENT_TICKERS = [
    'MSFT', 'NVDA', 'AMZN', 'JNJ', 'JPM'
]
BENCHMARK_TICKER = 'SPY'
FEATURES_TO_USE_IN_MODEL = [
    'close', 'close_vs_sma_50', 'mfi',
    'bollinger_width', 'obv', 'atr'
]

# --- Environment & Model Hyperparameters ---
# Best hyperparameters for the 5-stock ALPHA portfolio.
BEST_PPO_PARAMS = {
    'learning_rate': 1.015743796765418e-05,
    'n_epochs': 18,
    'gae_lambda': 0.9626806061446497,
    'ent_coef': 0.040126240058211596,
    'vf_coef': 0.33694931556942503,
    'gamma': 0.9901620237679383,
    'clip_range': 0.10455619747924472,
    'n_steps': 512,
}

# Environment parameters for the 5-stock ALPHA portfolio.
ENV_PARAMS = {
    'window_size': 30,
    'initial_balance': 10000.0,
    'transaction_cost_pct': 0.001,
    'volatility_penalty_weight': 0.7018132328989477,
    'loss_aversion_factor': 1.0311445779819872,
    'rolling_volatility_window': 179,
    'turnover_penalty_weight': 0.028894174005784548,
    'max_concentration_per_asset': 0.9981176663680391,
}