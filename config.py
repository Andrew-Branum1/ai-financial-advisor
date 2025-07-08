# config.py

# ==============================================================================
# 1. GLOBAL CONFIGURATION
# ==============================================================================

# Benchmark for all strategies
BENCHMARK_TICKER = "SPY"

# Full list of tickers available for analysis
# We can define different subsets for different risk profiles
STABLE_TICKERS = ["AAPL", "MSFT", "JNJ", "PFE", "UNH", "WMT", "JPM", "V", "SPY"]
MODERATE_TICKERS = STABLE_TICKERS + ["GOOGL", "AMZN", "COST", "CRM", "XOM"]
AGGRESSIVE_TICKERS = MODERATE_TICKERS + ["NVDA", "TSLA", "META", "ADBE"]

# A comprehensive list of features that can be calculated.
# Each strategy will select a subset of these.
FEATURES_TO_CALCULATE = [
    'close', 'volume', 'daily_return', 'close_vs_sma_10', 'close_vs_sma_20',
    'close_vs_sma_50', 'volatility_5', 'volatility_10', 'volatility_20',
    'momentum_10', 'momentum_20', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'bollinger_width', 'bollinger_position', 'obv', 'atr', 'stoch_k', 'stoch_d', 'mfi'
]

# ==============================================================================
# 2. STRATEGY-SPECIFIC CONFIGURATIONS (Term and Risk)
# ==============================================================================

STRATEGY_CONFIGS = {
    "short_term": {
        "conservative": {
            "tickers": STABLE_TICKERS,
            "features_to_use": ["close", "daily_return", "close_vs_sma_20", "volatility_10", "rsi", "bollinger_position"],
            "env_params": {"window_size": 20, "volatility_target": 0.10, "turnover_penalty_weight": 0.01},
            "ppo_params": {"learning_rate": 5e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "gamma": 0.98}
        },
        "moderate": {
            "tickers": MODERATE_TICKERS,
            "features_to_use": ["close", "daily_return", "close_vs_sma_20", "volatility_10", "momentum_10", "rsi", "macd_histogram", "bollinger_position"],
            "env_params": {"window_size": 30, "volatility_target": 0.15, "turnover_penalty_weight": 0.005},
            "ppo_params": {"learning_rate": 3e-4, "n_steps": 512, "batch_size": 64, "n_epochs": 10, "gamma": 0.99}
        },
        "aggressive": {
            "tickers": AGGRESSIVE_TICKERS,
            "features_to_use": ["close", "volume", "daily_return", "close_vs_sma_10", "volatility_5", "momentum_10", "mfi", "atr"],
            "env_params": {"window_size": 30, "volatility_target": 0.20, "turnover_penalty_weight": 0.002},
            "ppo_params": {"learning_rate": 1e-4, "n_steps": 512, "batch_size": 32, "n_epochs": 15, "gamma": 0.99}
        }
    },
    "long_term": {
        "conservative": {
            "tickers": STABLE_TICKERS,
            "features_to_use": ["close", "daily_return", "close_vs_sma_50", "volatility_20", "rsi", "obv"],
            "env_params": {"window_size": 80, "rebalancing_frequency": 20, "volatility_target": 0.08, "turnover_penalty_weight": 0.02},
            "ppo_params": {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 128, "n_epochs": 10, "gamma": 0.995}
        },
        "moderate": {
            "tickers": MODERATE_TICKERS,
            "features_to_use": ["close", "daily_return", "close_vs_sma_50", "volatility_20", "momentum_20", "rsi", "macd", "obv"],
            "env_params": {"window_size": 60, "rebalancing_frequency": 10, "volatility_target": 0.12, "turnover_penalty_weight": 0.008},
            "ppo_params": {"learning_rate": 1e-4, "n_steps": 1024, "batch_size": 128, "n_epochs": 10, "gamma": 0.995}
        },
        "aggressive": {
            "tickers": AGGRESSIVE_TICKERS,
            "features_to_use": ["close", "daily_return", "close_vs_sma_50", "volatility_20", "momentum_20", "bollinger_width", "mfi", "atr"],
            "env_params": {"window_size": 60, "rebalancing_frequency": 5, "volatility_target": 0.18, "turnover_penalty_weight": 0.005},
            "ppo_params": {"learning_rate": 1e-5, "n_steps": 2048, "batch_size": 256, "n_epochs": 15, "gamma": 0.998}
        }
    }
}