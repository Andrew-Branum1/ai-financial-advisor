# /eval/test_overfitting.py

import argparse
import importlib
import pandas as pd
from stable_baselines3 import PPO
import numpy as np
import sys
import os

# Add the project root to the Python path
# This allows us to import modules from the project (e.g., src, rl)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_manager import DataManager
from src.utils import get_final_portfolio_value, get_sharpe_ratio, get_max_drawdown

def evaluate_model(env, model):
    """
    Evaluates a given model in a given environment.

    Args:
        env (gym.Env): The environment to evaluate the model in.
        model (PPO): The trained PPO model.

    Returns:
        dict: A dictionary containing performance metrics.
    """
    obs, _ = env.reset()
    done = False
    
    # Run the model through the environment
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Extract performance metrics from the environment's history
    history = env.unwrapped.get_history()
    final_value = get_final_portfolio_value(history)
    returns = history['portfolio_value'].pct_change().dropna()
    sharpe_ratio = get_sharpe_ratio(returns)
    max_drawdown = get_max_drawdown(history['portfolio_value'])
    
    return {
        "final_portfolio_value": final_value,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_steps": len(history)
    }

def main(config_name, model_path):
    """
    Main function to run the overfitting test.
    It loads a model and evaluates it on both the training and testing datasets,
    then prints a comparison of the performance.
    """
    # --- 1. Load Configuration and Data ---
    try:
        config_module = importlib.import_module(f'config_{config_name}')
        print(f"Successfully loaded configuration: {config_name}")
    except ImportError:
        print(f"Error: Could not find config file 'config_{config_name}.py'.")
        sys.exit(1)

    # Use the DataManager to get the same data splits used for training
    data_manager = DataManager(
        config=config_module.DATA_CONFIG,
        data_path=config_module.DATA_PATH
    )
    train_df, val_df, test_df = data_manager.get_train_val_test_data()
    print(f"Data loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # --- 2. Set up Environments ---
    # The environment class is specified in the config
    PortfolioEnvClass = config_module.ENV_CLASS
    
    # Create an environment for the TRAINING data
    env_train = PortfolioEnvClass(
        df=train_df,
        config=config_module.ENV_CONFIG
    )
    
    # Create an environment for the TESTING data
    env_test = PortfolioEnvClass(
        df=test_df,
        config=config_module.ENV_CONFIG
    )

    # --- 3. Load Model ---
    try:
        model = PPO.load(model_path)
        print(f"Successfully loaded model from: {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        sys.exit(1)

    # --- 4. Evaluate and Compare ---
    print("\n--- Evaluating on TRAINING data ---")
    train_metrics = evaluate_model(env_train, model)
    print(pd.Series(train_metrics).to_string())

    print("\n--- Evaluating on TESTING data (unseen) ---")
    test_metrics = evaluate_model(env_test, model)
    print(pd.Series(test_metrics).to_string())

    # --- 5. Overfitting Analysis ---
    print("\n--- Overfitting Analysis ---")
    sharpe_diff = train_metrics['sharpe_ratio'] - test_metrics['sharpe_ratio']
    value_diff_pct = ((train_metrics['final_portfolio_value'] - test_metrics['final_portfolio_value']) / 
                      abs(test_metrics['final_portfolio_value'])) * 100 if test_metrics['final_portfolio_value'] != 0 else float('inf')

    print(f"Sharpe Ratio Difference (Train - Test): {sharpe_diff:.4f}")
    print(f"Final Value Difference (Train vs. Test): {value_diff_pct:.2f}%")

    if sharpe_diff > 0.5 or value_diff_pct > 25:
        print("\nWARNING: Potential overfitting detected.")
        print("Performance on training data is significantly higher than on unseen test data.")
    else:
        print("\nSUCCESS: Model appears to generalize well.")
        print("Performance on training and test data is comparable.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained RL model for overfitting.")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help="Name of the configuration to use (e.g., 'long_term', 'short_term')."
    )
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        help="Path to the trained model file (.zip)."
    )
    args = parser.parse_args()
    
    main(config_name=args.config, model_path=args.model)

