import sys
import os
import argparse
import pandas as pd
from stable_baselines3 import PPO
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_manager import load_market_data
from config import MODEL_CONFIGS, ALL_TICKERS, BENCHMARK_TICKER
from rl.portfolio_env_long_term import PortfolioEnvLongTerm
from rl.portfolio_env_short_term import PortfolioEnvShortTerm

def calculate_sharpe_ratio(rewards):
    """Calculates the annualized Sharpe ratio from a list of daily rewards."""
    rewards = pd.Series(rewards)
    if rewards.std() == 0 or rewards.empty:
        return 0.0
    return np.sqrt(252) * (rewards.mean() / rewards.std())

def calculate_benchmark_performance(df, start_date, end_date, initial_balance=100_000):
    """Calculates the performance of a buy-and-hold strategy on the benchmark."""
    benchmark_col = f"{BENCHMARK_TICKER}_daily_return"
    if benchmark_col not in df.columns:
        return 0, 0.0
    
    benchmark_returns = df.loc[start_date:end_date, benchmark_col]
    if benchmark_returns.empty:
        return initial_balance, 0.0

    final_value = initial_balance * (1 + benchmark_returns).cumprod().iloc[-1]
    sharpe = calculate_sharpe_ratio(benchmark_returns)
    return final_value, sharpe

def run_evaluation(model_path, config_name):
    """Loads a trained model and evaluates its performance against a benchmark."""
    print(f"\n--- Running Overfitting Analysis for: {config_name} ---")
    
    # 1. Load Data and Configuration
    config = MODEL_CONFIGS[config_name]
    env_class = PortfolioEnvLongTerm if config['env_class'] == 'PortfolioEnvLongTerm' else PortfolioEnvShortTerm
    
    with open(os.path.join(os.path.dirname(model_path), "training_info.json"), 'r') as f:
        training_info = json.load(f)

    df = load_market_data()
    df.index = pd.to_datetime(df.index)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # Replicate the exact feature and ticker set from the training run
    hyperparams = training_info['best_hyperparameters']
    features_from_training = training_info['features_used']
    env_params = {k: v for k, v in hyperparams.items() if k in ["window_size", "turnover_penalty_weight", "max_concentration_per_asset", "rebalancing_frequency"]}
    
    df_check = df.loc[training_info["train_start_date"]:training_info["train_end_date"]]
    
    # --- START OF FIX ---
    # More robustly filter tickers to match the training conditions.
    # A ticker is kept only if it has sufficient data for ALL features used in the observation space.
    tickers_kept = []
    for ticker in ALL_TICKERS:
        # Check that all required feature columns exist for this ticker
        required_obs_cols = [f"{ticker}_{feat}" for feat in features_from_training]
        if not all(col in df_check.columns for col in required_obs_cols):
            continue  # Skip ticker if a feature column is missing entirely

        # Check that all required feature columns have sufficient non-null data.
        # Technical indicators can have leading NaNs, so the threshold is slightly lower.
        min_data_points = 252 * 1.5 
        has_sufficient_data = all(df_check[col].count() >= min_data_points for col in required_obs_cols)
        
        if has_sufficient_data:
            tickers_kept.append(ticker)
    # --- END OF FIX ---
    
    # This list contains ALL columns the environment needs to function
    required_features = ['close', 'daily_return']
    all_features_for_env = sorted(list(set(features_from_training + required_features)))
    env_columns = [f"{t}_{f}" for t in tickers_kept for f in all_features_for_env if f"{t}_{f}" in df.columns]

    # 2. Split Data for training and testing periods
    train_df = df.loc[training_info["train_start_date"]:training_info["train_end_date"], env_columns].copy()
    test_df = df.loc["2022-01-01":, env_columns].copy()
    
    # 3. Create the environments
    # The environment will use `features_from_training` to create the correctly shaped observation space
    train_env = env_class(train_df, features_for_observation=features_from_training, **env_params)
    test_env = env_class(test_df, features_for_observation=features_from_training, **env_params)

    # 4. Load the model and evaluate
    # Pass the training environment to `load` to ensure spaces match
    model = PPO.load(model_path, env=train_env, device='cpu')
    results = {}
    
    # Evaluate on the training environment
    train_rewards = []
    obs, _ = train_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = train_env.step(action)
        train_rewards.append(reward)
        if terminated or truncated: break
    results['train_sharpe'] = calculate_sharpe_ratio(train_rewards)
    results['train_value'] = train_env.portfolio_value
    
    # Evaluate on the testing environment
    test_rewards = []
    obs, _ = test_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        test_rewards.append(reward)
        if terminated or truncated: break
    results['test_sharpe'] = calculate_sharpe_ratio(test_rewards)
    results['test_value'] = test_env.portfolio_value

    # 5. Calculate benchmark performance for both periods
    results['benchmark_train_value'], results['benchmark_train_sharpe'] = calculate_benchmark_performance(df, train_df.index.min(), train_df.index.max())
    results['benchmark_test_value'], results['benchmark_test_sharpe'] = calculate_benchmark_performance(df, test_df.index.min(), test_df.index.max())

    # 6. Display the final results
    print("\n--- IN-SAMPLE (TRAINING) RESULTS ---")
    print(f"   Model Value:      ${results['train_value']:,.2f} (Sharpe: {results['train_sharpe']:.2f})")
    print(f"   Benchmark Value:  ${results['benchmark_train_value']:,.2f} (Sharpe: {results['benchmark_train_sharpe']:.2f})")

    print("\n--- OUT-OF-SAMPLE (TESTING) RESULTS ---")
    print(f"   Model Value:      ${results['test_value']:,.2f} (Sharpe: {results['test_sharpe']:.2f})")
    print(f"   Benchmark Value:  ${results['benchmark_test_value']:,.2f} (Sharpe: {results['benchmark_test_sharpe']:.2f})")

    # 7. Provide a final analysis of performance
    print("\n--- ANALYSIS ---")
    if results['test_sharpe'] < 0:
        print("   Result: FAILURE. The model produced a negative Sharpe ratio on test data.")
    elif results['test_value'] < results['benchmark_test_value']:
        print("   Result: WARNING. The model underperformed the benchmark on test data.")
    else:
        sharpe_degradation = 1.0 - (results['test_sharpe'] / (results['train_sharpe'] + 1e-9))
        if sharpe_degradation > 0.4:
            print(f"   Result: CAUTION. Moderate overfitting suspected (Sharpe dropped by {sharpe_degradation:.1%}).")
        else:
            print(f"   Result: SUCCESS. Model appears robust and beat the benchmark.")
    print("-" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an overfitting test on a trained model.")
    parser.add_argument("--config", type=str, required=True, help="The model configuration name.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model .zip file.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
    else:
        run_evaluation(args.model, args.config)