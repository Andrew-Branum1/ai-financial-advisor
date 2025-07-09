# config.py
# Last updated: July 8, 2025 by Andrew
#
# This is the central configuration file for the entire project.
# It holds all the important parameters, from the list of stocks we're looking at
# to the specific settings for each of our reinforcement learning models.
#
# Keeping everything in one place makes it way easier to manage experiments
# and make sure everything is consistent.

# ==============================================================================
# 1. GLOBAL DATA & BENCHMARK SETTINGS
# ==============================================================================

# Standard benchmark for comparing our portfolio performance.
BENCHMARK_TICKER = "SPY"

# This is the full list of tickers we'll ever pull data for.
# I've sorted them to keep things clean. Any new stocks should be added here.
ALL_TICKERS = sorted(list(set([
    "AAPL", "AMZN", "BAC", "CVX", "GOOGL", "JNJ", "JPM", "LLY", "MRK", "MSFT",
    "NVDA", "PFE", "SAP", "SPY", "TM", "UNH", "V", "WMT", "XOM"
])))

# This is the master list of all possible technical indicators we can use.
# The data manager will calculate and store all of these, and then each model
# can pick and choose which ones it wants to use from this list.
ALL_FEATURES = [
    # Price & Volume
    "close",
    "volume",
    "daily_return",

    # Trend Indicators
    "close_vs_sma_10",
    "close_vs_sma_20",
    "close_vs_sma_50",
    "macd",
    "macd_signal",
    "macd_histogram",
    "obv",

    # Momentum Indicators
    "momentum_10",
    "momentum_20",
    "rsi",
    "stoch_k",
    "stoch_d",
    "mfi",

    # Volatility Indicators
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "bollinger_width",
    "bollinger_position",
    "atr",
]


# ==============================================================================
# 2. MODEL-SPECIFIC CONFIGURATIONS
# ==============================================================================
# Each entry defines a unique trading model. The training script will loop through these.

MODEL_CONFIGS = {

    # --- SHORT-TERM MODELS (Focus: Tactical, 1-6 week horizon) ---

    "short_term_conservative": {
        "description": "A cautious short-term model that prioritizes capital preservation and low turnover.",
        "env_class": "PortfolioEnvShortTerm",
        "features_to_use": ["daily_return", "volatility_10", "rsi", "bollinger_position", "close_vs_sma_20"],
        "env_params": {
            "volatility_target": 0.10,
            "turnover_penalty_weight": 0.015,
            "max_concentration_per_asset": 0.25,
        },
    },

    "short_term_balanced": {
        "description": "A balanced short-term model seeking a mix of safety and performance.",
        "env_class": "PortfolioEnvShortTerm",
        "features_to_use": ["daily_return", "volatility_20", "rsi", "macd_histogram", "momentum_10"],
        "env_params": {
            "volatility_target": 0.15,
            "turnover_penalty_weight": 0.008,
            "max_concentration_per_asset": 0.35,
        },
    },

    "short_term_aggressive": {
        "description": "A more aggressive short-term model that seeks to capitalize on momentum.",
        "env_class": "PortfolioEnvShortTerm",
        "features_to_use": ["daily_return", "momentum_10", "momentum_20", "rsi", "macd_histogram", "volatility_20"],
        "env_params": {
            "volatility_target": 0.20,
            "turnover_penalty_weight": 0.005,
            "max_concentration_per_asset": 0.40,
        },
    },

    # --- LONG-TERM MODELS (Focus: Strategic, 3-12+ month horizon) ---

    "long_term_conservative": {
        "description": "A defensive long-term model focused on stability and diversification.",
        "env_class": "PortfolioEnvLongTerm",
        "features_to_use": ["daily_return", "close_vs_sma_50", "volatility_20", "obv", "rsi"], # FIXED: Replaced 'mfi' with 'rsi'
        "env_params": {
            "rebalancing_frequency": 22, # Rebalance about once a month
            "turnover_penalty_weight": 0.025, # High penalty for infrequent trading
        },
    },

    "long_term_balanced": {
        "description": "A balanced long-term model focused on steady growth and diversification.",
        "env_class": "PortfolioEnvLongTerm",
        "features_to_use": ["daily_return", "close_vs_sma_50", "volatility_20", "rsi", "obv"], # FIXED: Removed 'mfi'
        "env_params": {
            "rebalancing_frequency": 20,
            "turnover_penalty_weight": 0.015,
        },
    },

    "long_term_aggressive": {
        "description": "A growth-focused long-term model that takes on more risk for higher potential returns.",
        "env_class": "PortfolioEnvLongTerm",
        "features_to_use": ["daily_return", "close_vs_sma_20", "close_vs_sma_50", "momentum_20", "volatility_20", "obv"],
        "env_params": {
            "rebalancing_frequency": 15, # Rebalance more often to catch trends
            "turnover_penalty_weight": 0.01,
        },
    },
}
