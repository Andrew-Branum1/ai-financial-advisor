# eval/test_overfitting.py
import sys
import os
import argparse
import pandas as pd
from stable_baselines3 import PPO
import numpy as np
import json

# Add project root to path to find 'src' and 'rl' modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary components from their actual locations
from src.data_manager import load_market_data_from_db
from config import MODEL_CONFIGS, ALL_TICKERS, BENCHMARK_TICKER
# Import the actual environment classes
from rl.portfolio_env_long_term import PortfolioEnvLongTerm
from rl.portfolio_env_short_term import PortfolioEnvShortTerm

def calculate_sharpe_ratio(rewards):
    """Calculates the annualized Sharpe ratio from a list of daily rewards."""
    rewards_series = pd.Series(rewards)
    if rewards_series.std() == 0 or rewards_series.empty:
        return 0.0
    return np.sqrt(252) * (rewards_series.mean() / rewards_series.std())

def calculate_benchmark_performance(df, start_date, end_date, initial_balance=100_000):
    """Calculates the performance of a buy-and-hold strategy on the benchmark."""
    benchmark_col = f"{BENCHMARK_TICKER}_daily_return"
    if benchmark_col not in df.columns:
        return 0, 0.0
        
    benchmark_returns = df.loc[start_date:end_date, benchmark_col]
    if benchmark_returns.empty:
        return initial_balance, 0.0

    cumulative_return = (1 + benchmark_returns).cumprod()
    final_value = initial_balance * cumulative_return.iloc[-1]
    sharpe_ratio = calculate_sharpe_ratio(benchmark_returns)
    return final_value, sharpe_ratio


def run_evaluation(model_path, config_name):
    """
    Loads a trained model and evaluates its performance against a benchmark.
    """
    print(f"\n{'='*60}")
    print(f"🔬 RUNNING OVERFITTING ANALYSIS FOR: {config_name}")
    print(f"   MODEL PATH: {model_path}")
    print(f"{'='*60}")

    # 1. Load Data and Configuration
    try:
        env_class_map = {
            "PortfolioEnvLongTerm": PortfolioEnvLongTerm,
            "PortfolioEnvShortTerm": PortfolioEnvShortTerm,
        }

        df = load_market_data_from_db()
        df.index = pd.to_datetime(df.index)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True); df.bfill(inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        if config_name not in MODEL_CONFIGS:
            print(f"❌ Error: Config '{config_name}' not found in config.py."); return

        config = MODEL_CONFIGS[config_name]
        env_class = env_class_map[config['env_class']]
        
        model_dir = os.path.dirname(model_path)
        info_path = os.path.join(model_dir, "training_info.json")
        if not os.path.exists(info_path):
            print(f"❌ Error: training_info.json not found in {model_dir}."); return

        with open(info_path, 'r') as f: training_info = json.load(f)

        hyperparams = training_info['best_hyperparameters']
        features_from_training = training_info['features_used']
        env_hyperparams = {k: v for k, v in hyperparams.items() if k in ["window_size", "turnover_penalty_weight", "max_concentration_per_asset", "rebalancing_frequency"]}
        print(f"   Loaded training hyperparameters: {env_hyperparams}")
        
        train_start_date_check = training_info.get("train_start_date", "2015-01-01")
        train_end_date_check = training_info.get("train_end_date", "2021-12-31")
        min_data_points = 252 * 2

        df_for_check = df.loc[train_start_date_check:train_end_date_check]
        
        tickers_to_drop = [t for t in ALL_TICKERS if f'{t}_close' in df_for_check and df_for_check[f'{t}_close'].count() < min_data_points]
        tickers_kept = [t for t in ALL_TICKERS if t not in tickers_to_drop]
        if tickers_to_drop: print(f"   Replicating training setup by dropping tickers: {tickers_to_drop}")
        
        required_features = ['close', 'daily_return']
        all_necessary_features = sorted(list(set(features_from_training + required_features)))
        env_columns = [f"{t}_{f}" for t in tickers_kept for f in all_necessary_features]
        env_columns = [c for c in env_columns if c in df.columns]

        print(f"   Using Environment: {env_class.__name__}")

    except Exception as e:
        print(f"❌ Error during setup: {e}", exc_info=True); return

    # 2. Split Data and Calculate Benchmark Performance
    train_start_date = training_info["train_start_date"]
    train_end_date = training_info["train_end_date"]
    test_start_date = "2022-01-01"

    train_df_env = df.loc[train_start_date:train_end_date, env_columns].copy()
    test_df_env = df.loc[test_start_date:, env_columns].copy()
    
    benchmark_train_value, benchmark_train_sharpe = calculate_benchmark_performance(df, train_start_date, train_end_date)
    benchmark_test_value, benchmark_test_sharpe = calculate_benchmark_performance(df, test_start_date, df.index.max())

    if train_df_env.empty or test_df_env.empty:
        print("❌ Error: Training or testing dataframe is empty."); return

    print(f"   Training data period: {train_df_env.index.min().date()} to {train_df_env.index.max().date()}")
    print(f"   Testing data period:  {test_df_env.index.min().date()} to {test_df_env.index.max().date()}")

    # 3. Create trading environments
    train_env = env_class(train_df_env, features_for_observation=features_from_training, **env_hyperparams)
    test_env = env_class(test_df_env, features_for_observation=features_from_training, **env_hyperparams)

    # 4. Load Model and Evaluate
    try:
        model = PPO.load(model_path, env=train_env, device='cpu')
        
        # In-Sample Evaluation
        train_rewards = []; obs, _ = train_env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = train_env.step(action)
            train_rewards.append(reward)
            if terminated or truncated: break
        train_sharpe = calculate_sharpe_ratio(train_rewards)
        train_value = train_env.portfolio_value
        
        # Out-of-Sample Evaluation
        test_rewards = []; obs, _ = test_env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            test_rewards.append(reward)
            if terminated or truncated: break
        test_sharpe = calculate_sharpe_ratio(test_rewards)
        test_value = test_env.portfolio_value

    except Exception as e:
        print(f"❌ Error during model evaluation: {e}", exc_info=True); return

    # 5. Display Results
    print("\n--- IN-SAMPLE (TRAINING DATA) RESULTS ---")
    print(f"   Model Portfolio Value:      ${train_value:,.2f}")
    print(f"   Model Annualized Sharpe:    {train_sharpe:.2f}")
    print(f"   Benchmark ({BENCHMARK_TICKER}) Value:   ${benchmark_train_value:,.2f}")
    print(f"   Benchmark ({BENCHMARK_TICKER}) Sharpe: {benchmark_train_sharpe:.2f}")

    print("\n--- OUT-OF-SAMPLE (TESTING DATA) RESULTS ---")
    print(f"   Model Portfolio Value:      ${test_value:,.2f}")
    print(f"   Model Annualized Sharpe:    {test_sharpe:.2f}")
    print(f"   Benchmark ({BENCHMARK_TICKER}) Value:   ${benchmark_test_value:,.2f}")
    print(f"   Benchmark ({BENCHMARK_TICKER}) Sharpe: {benchmark_test_sharpe:.2f}")

    # 6. Final Analysis
    print("\n--- ANALYSIS ---")
    initial_balance = 100_000
    
    # Check for negative Sharpe ratios, which is a major failure
    if test_sharpe < 0:
        print(f"   🚨 FAILURE: Model produced a negative Sharpe ratio ({test_sharpe:.2f}) on out-of-sample data.")
        print("   This indicates the learned strategy is fundamentally flawed and value-destroying.")
    elif test_value < benchmark_test_value:
        print(f"   🚨 WARNING: Model underperformed the benchmark ({BENCHMARK_TICKER}) on out-of-sample data.")
        print(f"   The model is not adding value compared to a simple buy-and-hold strategy.")
    else:
        # It beat the benchmark, now check for overfitting
        sharpe_degradation = 1.0 - (test_sharpe / (train_sharpe + 1e-9))
        if sharpe_degradation > 0.4:
            print(f"   ⚠️ CAUTION: Moderate Overfitting Suspected. The model's Sharpe ratio dropped by {sharpe_degradation:.1%}.")
            print("   While it beat the benchmark, its performance is not as stable as training results imply.")
        else:
            print(f"   ✅ SUCCESS: Model appears robust and beat the {BENCHMARK_TICKER} benchmark on out-of-sample data.")
    
    print(f"{'-'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an overfitting test by comparing a model's performance on training and testing data.")
    parser.add_argument("--config", type=str, required=True, help="The name of the model configuration to test.")
    parser.add_argument("--model", type=str, required=True, help="The path to the trained model .zip file.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found at {args.model}")
    else:
        run_evaluation(args.model, args.config)
