#!/usr/bin/env python3
"""
Collect Expanded Data
Collects data for the expanded universe of stocks used in the updated configs.
"""

import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collector_enhanced import run_data_collection_job
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Collect data for the expanded universe."""
    logger.info("Starting data collection for expanded universe...")
    
    # Collect data for the expanded universe (20 stocks from short-term and long-term configs)
    expanded_tickers = [
        # Technology (High Growth)
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN', 'TSLA', 'META', 'NFLX', 'ADBE', 'CRM',
        # Healthcare (Stable Growth)
        'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'LLY', 'MRK', 'BMY', 'AMGN', 'GILD'
    ]
    
    logger.info(f"Collecting data for {len(expanded_tickers)} stocks:")
    for ticker in expanded_tickers:
        logger.info(f"  - {ticker}")
    
    # Use the enhanced data collector with all technical indicators
    try:
        # Run the data collection job
        run_data_collection_job()
        
        logger.info("Data collection completed successfully!")
        logger.info("The expanded universe is now ready for training and inference.")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Data collection completed successfully!")
        print("You can now train models with the expanded universe of stocks.")
    else:
        print("\n❌ Data collection failed. Please check the logs for details.") 