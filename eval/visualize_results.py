# /eval/visualize_results.py

import argparse
import importlib
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from src.data_manager import DataManager
from src.utils import get_final_portfolio_value, get_sharpe_ratio, get_max_drawdown

def run_backtest(env, model):
    """Runs the backtest and returns the performance history."""
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.unwrapped.get_history()

def get_benchmark_performance(df, initial_investment=100000):
    """Calculates the performance of a buy-and-hold strategy on the first asset."""
    benchmark_df = df.copy()
    # Assume the first ticker is the benchmark (e.g., SPY)
    benchmark_ticker = benchmark_df.columns.get_level_values('ticker').unique()[0]
    benchmark_price = benchmark_df[('Close', benchmark_ticker)]
    
    returns = benchmark_price.pct_change().dropna()
    benchmark_values = initial_investment * (1 + returns).cumprod()
    # Add the initial investment value at the start
    benchmark_values = pd.concat([pd.Series([initial_investment], index=[benchmark_values.index[0] - pd.Timedelta(days=1)]), benchmark_values])
    benchmark_values.index = pd.to_datetime(benchmark_values.index)
    return benchmark_values


def plot_results(history, benchmark_history, config_name, output_dir='charts'):
    """
    Generates and saves a set of plots analyzing the backtest results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    history.index = pd.to_datetime(history.index)

    # --- 1. Equity Curve vs. Benchmark ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig1, ax1 = plt.subplots(figsize=(15, 7))
    
    ax1.plot(history.index, history['portfolio_value'], label='RL Agent', color='royalblue', linewidth=2)
    ax1.plot(benchmark_history.index, benchmark_history.values, label='Buy & Hold Benchmark (SPY)', color='grey', linestyle='--', linewidth=2)
    
    ax1.set_title(f'Equity Curve: RL Agent vs. Benchmark ({config_name})', fontsize=16)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, f'{config_name}_equity_curve.png'), dpi=300)
    print(f"Equity curve plot saved to {output_dir}/{config_name}_equity_curve.png")

    # --- 2. Asset Allocation Stacked Area Chart ---
    fig2, ax2 = plt.subplots(figsize=(15, 7))
    weights = history['weights'].apply(pd.Series)
    weights.columns = history['tickers'] + ['Cash'] # Add cash to the legend
    
    ax2.stackplot(weights.index, weights.T, labels=weights.columns, alpha=0.8)
    
    ax2.set_title(f'Asset Allocation Over Time ({config_name})', fontsize=16)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Portfolio Allocation (%)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, f'{config_name}_asset_allocation.png'), dpi=300)
    print(f"Asset allocation plot saved to {output_dir}/{config_name}_asset_allocation.png")

    # --- 3. Drawdown Plot ---
    fig3, ax3 = plt.subplots(figsize=(15, 7))
    portfolio_values = history['portfolio_value']
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max * 100

    ax3.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax3.plot(drawdown.index, drawdown, color='red', linewidth=1.5)
    
    ax3.set_title(f'Portfolio Drawdown ({config_name})', fontsize=16)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Drawdown (%)', fontsize=12)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, f'{config_name}_drawdown.png'), dpi=300)
    print(f"Drawdown plot saved to {output_dir}/{config_name}_drawdown.png")
    
    plt.close('all') # Close all figures to free up memory


def main(config_name, model_path):
    """Main function to run the visualization script."""
    # --- Load Config and Data ---
    try:
        config_module = importlib.import_module(f'config_{config_name}')
    except ImportError:
        print(f"Error: Could not find config file 'config_{config_name}.py'.")
        sys.exit(1)

    data_manager = DataManager(config=config_module.DATA_CONFIG, data_path=config_module.DATA_PATH)
    _, _, test_df = data_manager.get_train_val_test_data()
    
    # --- Set up Environment ---
    PortfolioEnvClass = config_module.ENV_CLASS
    env = PortfolioEnvClass(df=test_df, config=config_module.ENV_CONFIG)
    
    # --- Load Model ---
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        sys.exit(1)

    # --- Run Backtest and Generate Plots ---
    print("Running backtest on test data...")
    history = run_backtest(env, model)
    
    print("Calculating benchmark performance...")
    benchmark_history = get_benchmark_performance(test_df, initial_investment=config_module.ENV_CONFIG['initial_balance'])
    
    print("Generating plots...")
    plot_results(history, benchmark_history, config_name)
    print("\nAll visualizations have been saved to the 'charts/' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the performance of a trained RL model.")
    parser.add_argument('--config', type=str, required=True, help="Name of the configuration (e.g., 'long_term').")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model file (.zip).")
    args = parser.parse_args()
    
    main(config_name=args.config, model_path=args.model)
