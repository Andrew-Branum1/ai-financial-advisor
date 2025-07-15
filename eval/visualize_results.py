import json
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from src.data_manager import load_market_data
from config import MODEL_CONFIGS, ALL_TICKERS, BENCHMARK_TICKER
from rl.portfolio_env_long_term import PortfolioEnvLongTerm
from rl.portfolio_env_short_term import PortfolioEnvShortTerm

def run_backtest(env, model):
    obs, _ = env.reset()
    done = False
    
    portfolio_values = [env.portfolio_value]
    asset_weights = [env.weights] 
    dates = [env.df.index[env.current_step + env.window_size]]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if not done:
            portfolio_values.append(env.portfolio_value)
            asset_weights.append(env.weights)
            dates.append(env.df.index[env.current_step + env.window_size])
            
    return pd.DataFrame({
        'portfolio_value': portfolio_values,
        'weights': asset_weights
    }, index=pd.to_datetime(dates))

def get_benchmark_performance(df, start_date, end_date, initial_investment=100000):
    benchmark_col = f"{BENCHMARK_TICKER}_daily_return"
    benchmark_returns = df.loc[start_date:end_date, benchmark_col]
    
    cumulative_return = (1 + benchmark_returns).cumprod()
    benchmark_values = initial_investment * cumulative_return
    return benchmark_values

def plot_results(history, benchmark_history, config_name, tickers, output_dir='charts'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Model vs. Benchmark 
    plt.style.use('seaborn-v0_8-darkgrid')
    fig1, ax1 = plt.subplots(figsize=(15, 7))
    
    ax1.plot(history.index, history['portfolio_value'], label='RL Agent', color='royalblue', linewidth=2)
    ax1.plot(benchmark_history.index, benchmark_history.values, label=f'Buy & Hold ({BENCHMARK_TICKER})', color='grey', linestyle='--', linewidth=2)
    
    ax1.set_title(f'Equity Curve: {config_name}', fontsize=16, weight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, f'{config_name}_equity_curve.png'), dpi=300)

    # Portfolio Allocation
    fig2, ax2 = plt.subplots(figsize=(15, 7))
    weights_df = pd.DataFrame(history['weights'].tolist(), index=history.index, columns=tickers)
    weights_df['Cash'] = 1.0 - weights_df.sum(axis=1) 

    ax2.stackplot(weights_df.index, weights_df.T, labels=weights_df.columns, alpha=0.8)
    
    ax2.set_title(f'Asset Allocation Over Time: {config_name}', fontsize=16, weight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Portfolio Allocation', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, f'{config_name}_asset_allocation.png'), dpi=300)
    print(f"✅ Asset allocation plot saved for {config_name}")
    
    plt.close('all')

def generate_visuals_for_model(config_name, model_path):
    env_class_map = {"PortfolioEnvLongTerm": PortfolioEnvLongTerm, "PortfolioEnvShortTerm": PortfolioEnvShortTerm}
    
    df = load_market_data()
    df.index = pd.to_datetime(df.index)
    config = MODEL_CONFIGS[config_name]
    env_class = env_class_map[config['env_class']]
    
    with open(os.path.join(os.path.dirname(model_path), "training_info.json"), 'r') as f:
        training_info = json.load(f)
    
    hyperparams = training_info['best_hyperparameters']
    features_from_training = training_info['features_used']
    env_hyperparams = {k: v for k, v in hyperparams.items() if k in ["window_size", "turnover_penalty_weight", "max_concentration_per_asset", "rebalancing_frequency"]}
    
    # collect data
    train_start_date_check = training_info.get("train_start_date", "2015-01-01")
    train_end_date_check = training_info.get("train_end_date", "2021-12-31")
    min_data_points = 252 * 2
    df_for_check = df.loc[train_start_date_check:train_end_date_check]
    tickers_to_drop = [t for t in ALL_TICKERS if f'{t}_close' in df_for_check and df_for_check[f'{t}_close'].count() < min_data_points]
    tickers_in_model = [t for t in ALL_TICKERS if t not in tickers_to_drop]
    
    required_features = ['close', 'daily_return']
    all_necessary_features = sorted(list(set(features_from_training + required_features)))
    env_columns = [f"{t}_{f}" for t in tickers_in_model for f in all_necessary_features if f"{t}_{f}" in df.columns]
    
    #test data
    test_start_date = "2022-01-01"
    test_df_env = df.loc[test_start_date:, env_columns].copy()
    
    env = env_class(test_df_env, features_for_observation=features_from_training, **env_hyperparams)
    
    model = PPO.load(model_path, env=env)
    history = run_backtest(env, model)
    benchmark_history = get_benchmark_performance(df, test_df_env.index.min(), test_df_env.index.max())
    plot_results(history, benchmark_history, config_name, tickers_in_model)

def find_latest_model(config_name: str) -> str | None:
    models_dir = "models"
    if not os.path.isdir(models_dir):
        return None

    subfolders = [f for f in os.listdir(models_dir) if f.startswith(config_name) and os.path.isdir(os.path.join(models_dir, f))]
    if not subfolders:
        return None
        
    latest_folder = sorted(subfolders, reverse=True)[0]
    model_path = os.path.join(models_dir, latest_folder, "model.zip")
    
    return model_path if os.path.exists(model_path) else None

def visualize_all_models():
    for config_name in MODEL_CONFIGS.keys():
        latest_model_path = find_latest_model(config_name)
        generate_visuals_for_model(config_name, latest_model_path)

            

visualize_all_models()