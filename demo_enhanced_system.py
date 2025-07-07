#!/usr/bin/env python3
"""
Enhanced AI Financial Advisor Demo
Demonstrates the capabilities of the short and long term stock market prediction system.
"""

import os
import sys
import logging
import asyncio
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collector import run_data_collection_job
from src.utils import load_market_data_from_db
from rl.portfolio_env_short_term import PortfolioEnvShortTerm
from rl.portfolio_env_long_term import PortfolioEnvLongTerm

# Import configurations
from config_short_term import AGENT_TICKERS as SHORT_TERM_TICKERS, ENV_PARAMS as SHORT_TERM_PARAMS
from config_long_term import AGENT_TICKERS as LONG_TERM_TICKERS, ENV_PARAMS as LONG_TERM_PARAMS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedAIFinancialAdvisor:
    """
    Enhanced AI Financial Advisor that demonstrates both short and long term prediction capabilities.
    """
    
    def __init__(self):
        self.short_term_env = None
        self.long_term_env = None
        self.market_data = None
        
    def collect_market_data(self):
        """Collect and prepare market data for analysis."""
        logging.info("🔄 Collecting market data...")
        
        try:
            # Run data collection
            run_data_collection_job()
            logging.info("✅ Market data collection completed successfully!")
            
            # Load data for analysis
            all_tickers = list(set(SHORT_TERM_TICKERS + LONG_TERM_TICKERS + ['SPY']))
            self.market_data = load_market_data_from_db(
                tickers_list=all_tickers,
                start_date="2020-01-01",
                end_date="2024-01-01",
                min_data_points=100,
                feature_columns=['close', 'rsi', 'volatility_20', 'macd', 'bollinger_width']
            )
            
            logging.info(f"📊 Loaded market data with shape: {self.market_data.shape}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to collect market data: {e}")
            return False
    
    def setup_environments(self):
        """Setup both short-term and long-term portfolio environments."""
        logging.info("🔧 Setting up portfolio environments...")
        
        try:
            # Prepare feature columns for short-term strategy
            short_term_features = ['close', 'rsi', 'volatility_20', 'macd', 'bollinger_width']
            short_term_cols = [f"{t}_{f}" for t in SHORT_TERM_TICKERS for f in short_term_features]
            
            # Prepare feature columns for long-term strategy
            long_term_features = ['close', 'rsi', 'volatility_20', 'macd', 'bollinger_width']
            long_term_cols = [f"{t}_{f}" for t in LONG_TERM_TICKERS for f in long_term_features]
            
            # Create short-term environment
            short_term_data = self.market_data[short_term_cols].copy()
            self.short_term_env = PortfolioEnvShortTerm(
                df=short_term_data,
                feature_columns_ordered=short_term_features,
                **SHORT_TERM_PARAMS
            )
            
            # Create long-term environment
            long_term_data = self.market_data[long_term_cols].copy()
            self.long_term_env = PortfolioEnvLongTerm(
                df=long_term_data,
                feature_columns_ordered=long_term_features,
                **LONG_TERM_PARAMS
            )
            
            logging.info("✅ Portfolio environments setup completed!")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to setup environments: {e}")
            return False
    
    def demonstrate_short_term_strategy(self):
        """Demonstrate short-term trading strategy capabilities."""
        logging.info("📈 Demonstrating Short-term Trading Strategy...")
        
        if not self.short_term_env:
            logging.error("❌ Short-term environment not initialized")
            return
        
        try:
            # Reset environment
            obs, info = self.short_term_env.reset()
            
            # Simulate a few trading days
            portfolio_values = [info['portfolio_value']]
            weights_history = [info['weights']]
            
            for day in range(30):  # Simulate 30 trading days
                # Generate random action (in real scenario, this would be from trained model)
                action = np.random.dirichlet(np.ones(len(SHORT_TERM_TICKERS)))
                
                # Take step in environment
                obs, reward, terminated, truncated, info = self.short_term_env.step(action)
                
                portfolio_values.append(info['portfolio_value'])
                weights_history.append(info['weights'])
                
                if terminated or truncated:
                    break
            
            # Calculate performance metrics
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            logging.info(f"📊 Short-term Strategy Results:")
            logging.info(f"   Total Return: {total_return:.2%}")
            logging.info(f"   Annualized Volatility: {volatility:.2%}")
            logging.info(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
            logging.info(f"   Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'final_value': portfolio_values[-1]
            }
            
        except Exception as e:
            logging.error(f"❌ Short-term strategy demonstration failed: {e}")
            return None
    
    def demonstrate_long_term_strategy(self):
        """Demonstrate long-term growth strategy capabilities."""
        logging.info("🌱 Demonstrating Long-term Growth Strategy...")
        
        if not self.long_term_env:
            logging.error("❌ Long-term environment not initialized")
            return
        
        try:
            # Reset environment
            obs, info = self.long_term_env.reset()
            
            # Simulate a longer period for long-term strategy
            portfolio_values = [info['portfolio_value']]
            weights_history = [info['weights']]
            
            for day in range(90):  # Simulate 90 trading days
                # Generate random action (in real scenario, this would be from trained model)
                action = np.random.dirichlet(np.ones(len(LONG_TERM_TICKERS)))
                
                # Take step in environment
                obs, reward, terminated, truncated, info = self.long_term_env.step(action)
                
                portfolio_values.append(info['portfolio_value'])
                weights_history.append(info['weights'])
                
                if terminated or truncated:
                    break
            
            # Calculate performance metrics
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Calculate maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)
            
            logging.info(f"📊 Long-term Strategy Results:")
            logging.info(f"   Total Return: {total_return:.2%}")
            logging.info(f"   Annualized Volatility: {volatility:.2%}")
            logging.info(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
            logging.info(f"   Max Drawdown: {max_drawdown:.2%}")
            logging.info(f"   Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': portfolio_values[-1]
            }
            
        except Exception as e:
            logging.error(f"❌ Long-term strategy demonstration failed: {e}")
            return None
    
    def compare_strategies(self, short_term_results, long_term_results):
        """Compare the performance of both strategies."""
        logging.info("🔄 Comparing Strategy Performance...")
        
        if not short_term_results or not long_term_results:
            logging.error("❌ Cannot compare strategies - missing results")
            return
        
        print("\n" + "="*80)
        print(" STRATEGY COMPARISON ".center(80, "="))
        print("="*80)
        
        print(f"{'Metric':<25} {'Short-term':<15} {'Long-term':<15} {'Winner':<10}")
        print("-" * 80)
        
        # Compare key metrics
        metrics = [
            ('Total Return', f"{short_term_results['total_return']:.2%}", f"{long_term_results['total_return']:.2%}", 'return'),
            ('Volatility', f"{short_term_results['volatility']:.2%}", f"{long_term_results['volatility']:.2%}", 'lower'),
            ('Sharpe Ratio', f"{short_term_results['sharpe_ratio']:.2f}", f"{long_term_results['sharpe_ratio']:.2f}", 'higher'),
            ('Final Value', f"${short_term_results['final_value']:,.0f}", f"${long_term_results['final_value']:,.0f}", 'higher')
        ]
        
        for metric, short_val, long_val, better in metrics:
            if better == 'return' or better == 'higher':
                winner = 'Short-term' if short_term_results[metric.replace(' ', '_').lower()] > long_term_results[metric.replace(' ', '_').lower()] else 'Long-term'
            else:
                winner = 'Short-term' if short_term_results[metric.replace(' ', '_').lower()] < long_term_results[metric.replace(' ', '_').lower()] else 'Long-term'
            
            print(f"{metric:<25} {short_val:<15} {long_val:<15} {winner:<10}")
        
        if 'max_drawdown' in long_term_results:
            print(f"{'Max Drawdown':<25} {'N/A':<15} {long_term_results['max_drawdown']:.2%:<15} {'N/A':<10}")
        
        print("="*80)
        
        # Strategy recommendations
        print("\n📋 Strategy Recommendations:")
        print("• Short-term Strategy: Best for active traders seeking quick gains")
        print("• Long-term Strategy: Best for patient investors focused on sustainable growth")
        print("• Consider combining both strategies for diversified portfolio management")
    
    def run_full_demo(self):
        """Run the complete demonstration of the enhanced AI financial advisor."""
        print("🚀 Enhanced AI Financial Advisor Demo")
        print("="*50)
        
        # Step 1: Collect market data
        if not self.collect_market_data():
            return False
        
        # Step 2: Setup environments
        if not self.setup_environments():
            return False
        
        # Step 3: Demonstrate short-term strategy
        short_term_results = self.demonstrate_short_term_strategy()
        
        # Step 4: Demonstrate long-term strategy
        long_term_results = self.demonstrate_long_term_strategy()
        
        # Step 5: Compare strategies
        self.compare_strategies(short_term_results, long_term_results)
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Train models: python -m rl.train_short_term")
        print("2. Train models: python -m rl.train_long_term")
        print("3. Evaluate: python -m rl.evaluate_enhanced")
        print("4. Launch web interface: python app.py")
        
        return True

def main():
    """Main function to run the demo."""
    import numpy as np
    
    advisor = EnhancedAIFinancialAdvisor()
    success = advisor.run_full_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
    else:
        print("\n❌ Demo failed. Please check the logs for details.")

if __name__ == "__main__":
    main() 