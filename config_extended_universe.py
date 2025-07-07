# config_extended_universe.py
"""
Extended Universe Configuration
Defines a larger universe of stocks for the model to choose from,
while only showing top recommendations to users.
"""

# --- Extended Universe of Stocks ---
# The model will be trained on this larger universe but only show top 5 to users

EXTENDED_UNIVERSE_TICKERS = [
    # Technology (High Growth)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'ADBE',
    'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'ADI', 'KLAC',
    
    # Healthcare (Stable Growth)
    'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'LLY', 'MRK', 'BMY', 'AMGN',
    'GILD', 'CVS', 'CI', 'ANTM', 'HUM', 'CNC', 'WBA', 'ZTS', 'REGN', 'VRTX',
    
    # Financial Services
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
    'AXP', 'BLK', 'SCHW', 'CME', 'ICE', 'SPGI', 'MCO', 'FIS', 'FISV', 'V',
    
    # Consumer Discretionary
    'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'ROST', 'LOW', 'TGT', 'WMT', 'COST',
    'AMZN', 'TSLA', 'NFLX', 'BKNG', 'MAR', 'HLT', 'YUM', 'CMG', 'DPZ', 'SBUX',
    
    # Consumer Staples (Defensive)
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'CL', 'GIS', 'K',
    'HSY', 'SJM', 'CAG', 'KMB', 'CLX', 'CHD', 'EL', 'ULTA', 'DG', 'DLTR',
    
    # Industrials
    'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'RTX', 'LMT', 'NOC',
    'EMR', 'ETN', 'ITW', 'PH', 'DOV', 'XYL', 'AME', 'FTV', 'IEX', 'ROK',
    
    # Energy
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'PSX', 'VLO', 'MPC',
    'OXY', 'PXD', 'DVN', 'HES', 'APA', 'FANG', 'MRO', 'KMI', 'WMB', 'ENB',
    
    # Materials
    'LIN', 'APD', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'STLD', 'RS', 'BLL',
    'ALB', 'LVS', 'VMC', 'MLM', 'NVR', 'PHM', 'LEN', 'DHI', 'TOL', 'KBH',
    
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'DLR', 'PSA', 'SPG', 'O', 'WELL', 'VICI',
    'EQR', 'AVB', 'MAA', 'UDR', 'ESS', 'CPT', 'BXP', 'SLG', 'VNO', 'KIM',
    
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'WEC', 'DTE', 'ED',
    'PEG', 'EIX', 'AEE', 'CMS', 'CNP', 'NI', 'LNT', 'ATO', 'BKH', 'PNW',
    
    # Communication Services
    'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'CHTR', 'TMUS', 'VZ', 'T',
    'ATVI', 'EA', 'TTWO', 'ZNGA', 'MTCH', 'SNAP', 'PINS', 'TWTR', 'LYV', 'LVS',
    
    # Emerging Growth
    'SQ', 'PYPL', 'ZM', 'DOCU', 'CRWD', 'OKTA', 'ZS', 'PLTR', 'SNOW', 'DDOG',
    'NET', 'MDB', 'TWLO', 'SPOT', 'UBER', 'LYFT', 'RBLX', 'HOOD', 'COIN', 'RIVN'
]

# --- Sector Classifications ---
SECTOR_MAPPING = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'NVDA', 'TSLA', 'META', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'ADI', 'KLAC'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'LLY', 'MRK', 'BMY', 'AMGN', 'GILD', 'CVS', 'CI', 'ANTM', 'HUM', 'CNC', 'WBA', 'ZTS', 'REGN', 'VRTX'],
    'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF', 'AXP', 'BLK', 'SCHW', 'CME', 'ICE', 'SPGI', 'MCO', 'FIS', 'FISV', 'V'],
    'Consumer_Discretionary': ['HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'ROST', 'LOW', 'TGT', 'WMT', 'COST', 'AMZN', 'TSLA', 'NFLX', 'BKNG', 'MAR', 'HLT', 'YUM', 'CMG', 'DPZ'],
    'Consumer_Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'CL', 'GIS', 'K', 'HSY', 'SJM', 'CAG', 'KMB', 'CLX', 'CHD', 'EL', 'ULTA', 'DG', 'DLTR'],
    'Industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'RTX', 'LMT', 'NOC', 'EMR', 'ETN', 'ITW', 'PH', 'DOV', 'XYL', 'AME', 'FTV', 'IEX', 'ROK'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR', 'PSX', 'VLO', 'MPC', 'OXY', 'PXD', 'DVN', 'HES', 'APA', 'FANG', 'MRO', 'KMI', 'WMB', 'ENB'],
    'Materials': ['LIN', 'APD', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'STLD', 'RS', 'BLL', 'ALB', 'LVS', 'VMC', 'MLM', 'NVR', 'PHM', 'LEN', 'DHI', 'TOL', 'KBH'],
    'Real_Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'DLR', 'PSA', 'SPG', 'O', 'WELL', 'VICI', 'EQR', 'AVB', 'MAA', 'UDR', 'ESS', 'CPT', 'BXP', 'SLG', 'VNO', 'KIM'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'WEC', 'DTE', 'ED', 'PEG', 'EIX', 'AEE', 'CMS', 'CNP', 'NI', 'LNT', 'ATO', 'BKH', 'PNW'],
    'Communication': ['GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'CHTR', 'TMUS', 'VZ', 'T', 'ATVI', 'EA', 'TTWO', 'ZNGA', 'MTCH', 'SNAP', 'PINS', 'TWTR', 'LYV', 'LVS'],
    'Emerging_Growth': ['SQ', 'PYPL', 'ZM', 'DOCU', 'CRWD', 'OKTA', 'ZS', 'PLTR', 'SNOW', 'DDOG', 'NET', 'MDB', 'TWLO', 'SPOT', 'UBER', 'LYFT', 'RBLX', 'HOOD', 'COIN', 'RIVN']
}

# --- Risk Classifications ---
RISK_LEVELS = {
    'Conservative': ['JNJ', 'PFE', 'PG', 'KO', 'WMT', 'JPM', 'BAC', 'XOM', 'CVX', 'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'WEC', 'DTE', 'ED', 'PEG'],
    'Moderate': ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'NVDA', 'UNH', 'HD', 'MCD', 'NKE', 'SBUX', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF', 'AXP', 'BLK'],
    'Aggressive': ['TSLA', 'META', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'ADI', 'KLAC', 'ABBV', 'TMO', 'DHR', 'LLY', 'MRK', 'BMY']
}

# --- Current Universe (for backward compatibility) ---
AGENT_TICKERS = ['MSFT', 'NVDA', 'AMZN', 'JNJ', 'JPM']
BENCHMARK_TICKER = 'SPY'

# --- Features to use (same as current config) ---
FEATURES_TO_USE_IN_MODEL = [
    'close', 'volume', 'daily_return',
    'close_vs_sma_10', 'close_vs_sma_20', 'close_vs_sma_50',
    'volatility_5', 'volatility_10', 'volatility_20',
    'momentum_5', 'momentum_10', 'momentum_20',
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'bollinger_width', 'bollinger_position',
    'volume_ratio', 'obv', 'atr',
    'stoch_k', 'stoch_d', 'williams_r',
    'cci', 'mfi', 'adx',
    'trend_strength', 'volatility_ratio',
    'roc_10', 'roc_20', 'volume_roc',
    'price_efficiency', 'mean_reversion_signal'
]

# --- Configuration for different model sizes ---
MODEL_CONFIGS = {
    'small': {
        'max_tickers': 20,
        'top_recommendations': 5,
        'description': 'Small universe - 20 stocks, show top 5'
    },
    'medium': {
        'max_tickers': 50,
        'top_recommendations': 5,
        'description': 'Medium universe - 50 stocks, show top 5'
    },
    'large': {
        'max_tickers': 100,
        'top_recommendations': 5,
        'description': 'Large universe - 100 stocks, show top 5'
    },
    'full': {
        'max_tickers': len(EXTENDED_UNIVERSE_TICKERS),
        'top_recommendations': 5,
        'description': 'Full universe - all stocks, show top 5'
    }
}

# --- Selection strategies ---
SELECTION_STRATEGIES = {
    'diversified': {
        'description': 'Select stocks from different sectors for diversification',
        'sectors_per_model': 8,
        'stocks_per_sector': 6
    },
    'momentum': {
        'description': 'Select stocks based on recent performance',
        'lookback_period': 60,
        'top_percentile': 0.3
    },
    'value': {
        'description': 'Select stocks based on fundamental metrics',
        'metrics': ['pe_ratio', 'pb_ratio', 'dividend_yield'],
        'top_percentile': 0.3
    },
    'balanced': {
        'description': 'Mix of different selection criteria',
        'momentum_weight': 0.4,
        'value_weight': 0.3,
        'diversification_weight': 0.3
    }
}

def get_ticker_universe(size='medium', strategy='diversified'):
    """
    Get a subset of tickers based on size and selection strategy.
    
    Args:
        size: 'small', 'medium', 'large', or 'full'
        strategy: 'diversified', 'momentum', 'value', or 'balanced'
    
    Returns:
        List of ticker symbols
    """
    config = MODEL_CONFIGS.get(size, MODEL_CONFIGS['medium'])
    max_tickers = config['max_tickers']
    
    if size == 'full':
        return EXTENDED_UNIVERSE_TICKERS[:max_tickers]
    
    if strategy == 'diversified':
        return _select_diversified_tickers(max_tickers)
    elif strategy == 'momentum':
        return _select_momentum_tickers(max_tickers)
    elif strategy == 'value':
        return _select_value_tickers(max_tickers)
    elif strategy == 'balanced':
        return _select_balanced_tickers(max_tickers)
    else:
        return EXTENDED_UNIVERSE_TICKERS[:max_tickers]

def _select_diversified_tickers(max_tickers):
    """Select tickers from different sectors for diversification."""
    selected = []
    sectors_per_model = 8
    stocks_per_sector = max_tickers // sectors_per_model
    
    # Get top sectors by market cap or other criteria
    top_sectors = list(SECTOR_MAPPING.keys())[:sectors_per_model]
    
    for sector in top_sectors:
        sector_tickers = SECTOR_MAPPING[sector]
        selected.extend(sector_tickers[:stocks_per_sector])
    
    # Fill remaining slots if needed
    remaining = max_tickers - len(selected)
    if remaining > 0:
        all_tickers = [t for tickers in SECTOR_MAPPING.values() for t in tickers]
        unused_tickers = [t for t in all_tickers if t not in selected]
        selected.extend(unused_tickers[:remaining])
    
    return selected[:max_tickers]

def _select_momentum_tickers(max_tickers):
    """Select tickers based on momentum (placeholder - would need price data)."""
    # This would require historical price data to calculate momentum
    # For now, return a subset based on market cap or other criteria
    return EXTENDED_UNIVERSE_TICKERS[:max_tickers]

def _select_value_tickers(max_tickers):
    """Select tickers based on value metrics (placeholder - would need fundamental data)."""
    # This would require fundamental data (P/E, P/B, etc.)
    # For now, return a subset
    return EXTENDED_UNIVERSE_TICKERS[:max_tickers]

def _select_balanced_tickers(max_tickers):
    """Select tickers using a balanced approach."""
    # Combine different selection criteria
    # For now, return a diversified subset
    return _select_diversified_tickers(max_tickers)

def get_top_recommendations(weights_dict, top_n=5):
    """
    Get top N recommendations from model weights.
    
    Args:
        weights_dict: Dictionary of {ticker: weight}
        top_n: Number of top recommendations to return
    
    Returns:
        List of tuples (ticker, weight, percentage)
    """
    # Sort by weight (descending)
    sorted_items = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_items = sorted_items[:top_n]
    
    # Calculate percentages
    total_weight = sum(weight for _, weight in top_items)
    if total_weight > 0:
        recommendations = [
            (ticker, weight, weight / total_weight * 100)
            for ticker, weight in top_items
        ]
    else:
        recommendations = [(ticker, weight, 0) for ticker, weight in top_items]
    
    return recommendations

def get_sector_breakdown(recommendations):
    """
    Get sector breakdown of recommendations.
    
    Args:
        recommendations: List of (ticker, weight, percentage) tuples
    
    Returns:
        Dictionary of {sector: percentage}
    """
    sector_weights = {}
    
    for ticker, weight, percentage in recommendations:
        for sector, tickers in SECTOR_MAPPING.items():
            if ticker in tickers:
                sector_weights[sector] = sector_weights.get(sector, 0) + percentage
                break
    
    return sector_weights

def get_risk_profile(recommendations):
    """
    Get risk profile of recommendations.
    
    Args:
        recommendations: List of (ticker, weight, percentage) tuples
    
    Returns:
        Dictionary of {risk_level: percentage}
    """
    risk_weights = {}
    
    for ticker, weight, percentage in recommendations:
        for risk_level, tickers in RISK_LEVELS.items():
            if ticker in tickers:
                risk_weights[risk_level] = risk_weights.get(risk_level, 0) + percentage
                break
    
    return risk_weights 