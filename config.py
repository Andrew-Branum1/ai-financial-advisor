BENCHMARK_TICKER = "SPY"

TECH = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "SAP", "V"]
FINANCIAL = ["BAC", "JPM"]
HEALTHCARE = ["JNJ", "LLY", "MRK", "PFE", "UNH"]
ENERGY = ["CVX", "XOM"]
OTHER = ["TM", "WMT"] 

ALL_TICKERS = sorted(list(set(
    TECH + 
    FINANCIAL + 
    HEALTHCARE + 
    ENERGY + 
    OTHER + 
    [BENCHMARK_TICKER]
)))


ALL_FEATURES = [
    "close",
    "volume",
    "daily_return",
    "close_vs_sma_10",
    "close_vs_sma_20",
    "close_vs_sma_50",
    "macd",
    "macd_signal",
    "macd_histogram",
    "obv",
    "momentum_10",
    "momentum_20",
    "rsi",
    "stoch_k",
    "stoch_d",
    "mfi",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "bollinger_width",
    "bollinger_position",
    "atr",
]


MODEL_CONFIGS = {

        "short_term_conservative": {
            "description": "A cautious short-term model.",
            "env_class": "PortfolioEnvShortTerm",
            "features_to_use": ["bollinger_position", "volatility_20"],
            "env_params": {
                "volatility_target": 0.10,
                "turnover_penalty_weight": 0.012,
                "max_concentration_per_asset": 0.30,
            },
        },

        "short_term_balanced": {
            "description": "A balanced short-term model.",
            "env_class": "PortfolioEnvShortTerm",
            "features_to_use": ["daily_return", "bollinger_position"],
            "env_params": {
                "volatility_target": 0.15,
                "turnover_penalty_weight": 0.008,
                "max_concentration_per_asset": 0.35,
            },
        },

        "short_term_aggressive": {
            "description": "An aggressive short-term model.",
            "env_class": "PortfolioEnvShortTerm",
            "features_to_use": ["daily_return", "rsi"],
            "env_params": {
                "volatility_target": 0.20,
                "turnover_penalty_weight": 0.002,
                "max_concentration_per_asset": 0.50,
            },
        },

    "long_term_conservative": {
        "description": "A conservative long-term model.",
        "env_class": "PortfolioEnvLongTerm",
        "features_to_use": ["daily_return", "close_vs_sma_50", "volatility_20", "obv", "rsi"],
        "env_params": {
            "rebalancing_frequency": 22, 
            "turnover_penalty_weight": 0.025, 
        },
    },

    "long_term_balanced": {
        "description": "A balanced long-term model.",
        "env_class": "PortfolioEnvLongTerm",
        "features_to_use": ["daily_return", "close_vs_sma_50", "volatility_20", "rsi", "obv"], 
        "env_params": {
            "rebalancing_frequency": 20,
            "turnover_penalty_weight": 0.015,
        },
    },

    "long_term_aggressive": {
        "description": "An aggressive long-term model.",
        "env_class": "PortfolioEnvLongTerm",
        "features_to_use": ["daily_return", "close_vs_sma_20", "close_vs_sma_50", "momentum_20", "volatility_20", "obv"],
        "env_params": {
            "rebalancing_frequency": 25, 
            "turnover_penalty_weight": 0.010,
        },
    },

}
