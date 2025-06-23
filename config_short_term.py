# config_short_term.py
"""
Configuration for the ACTIVE, short-term RL agent.
"""

# --- Data Configuration ---
AGENT_TICKERS = ['MSFT', 'NVDA', 'AMZN', 'JNJ', 'JPM']
BENCHMARK_TICKER = 'SPY'
FEATURES_TO_USE_IN_MODEL = [
    'close_vs_sma_10', 'rsi', 'volatility_5', 'bollinger_width', 'mfi', 'close'
]

# --- Hyperparameters for the Short-Term Active Trader ---
BEST_PPO_PARAMS = {
    'learning_rate':  0.0009042436291875808,
    'n_epochs': 11,
    'gae_lambda': 0.9271494454220346,
    'ent_coef':  0.0164583897291383,
    'vf_coef': 0.46612025760783965,
    'gamma':   0.9414231682044852,
    'clip_range':  0.2705124729061021,
    'n_steps': 1024,
}

ENV_PARAMS = {
    'window_size': 30,
    'initial_balance': 10000.0,
    'transaction_cost_pct': 0.001,
    'rolling_volatility_window': 14,
    'turnover_penalty_weight':  0.004552354112205691,
    'max_concentration_per_asset': 0.7452189793808334,
}