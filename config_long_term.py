# config_long_term.py
"""
Configuration for the PASSIVE, long-term RL agent.
Optimized for buy-and-hold strategies with periodic rebalancing.
"""

# --- Data Configuration ---
# Expanded to 20 stocks for better diversification and more options
AGENT_TICKERS = [
    # Technology (High Growth)
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN', 'TSLA', 'META', 'NFLX', 'ADBE', 'CRM',
    # Healthcare (Stable Growth)
    'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'LLY', 'MRK', 'BMY', 'AMGN', 'GILD'
]
BENCHMARK_TICKER = 'SPY'
FEATURES_TO_USE_IN_MODEL = [
    'close', 'volume', 'daily_return',
    'close_vs_sma_10', 'close_vs_sma_20', 'close_vs_sma_50',
    'volatility_5', 'volatility_10', 'volatility_20', 'volatility_60',
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60',
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'bollinger_width', 'bollinger_position',
    'volume_ratio', 'obv', 'atr',
    'stoch_k', 'stoch_d', 'williams_r',
    'cci', 'mfi', 'adx',
    'trend_strength', 'volatility_ratio',
    'roc_10', 'roc_20', 'volume_roc',
    'price_efficiency', 'mean_reversion_signal'
]

# --- Optimized Hyperparameters for Long-Term Passive Investing ---
BEST_PPO_PARAMS = {
    'learning_rate': 0.0003,
    'n_epochs': 10,
    'gae_lambda': 0.95,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'gamma': 0.99,
    'clip_range': 0.2,
    'max_grad_norm': 0.5,
    'batch_size': 64,
    'n_steps': 2048,
    'policy_kwargs': {
        'net_arch': [dict(pi=[128, 128], vf=[128, 128])],
        'activation_fn': 'relu'
    }
}

# --- Environment Parameters for Long-Term Passive Investing ---
ENV_PARAMS = {
    'initial_balance': 100000,
    'transaction_cost_pct': 0.001,
    'window_size': 252,  # One year of trading days
    'max_position_pct': 0.4,
    'min_holding_period': 30,  # Minimum 30 days
    'max_holding_period': 252,  # Maximum 1 year
    'volatility_target': 0.12,  # Lower volatility target for long-term
    'drawdown_penalty': 0.05,  # Lower drawdown penalty
    'momentum_window': 60,  # Longer momentum window
    'mean_reversion_window': 252,  # One year for mean reversion
    'rebalancing_frequency': 30,  # Rebalance every 30 days
    'diversification_bonus': 0.1  # Bonus for portfolio diversification
} 