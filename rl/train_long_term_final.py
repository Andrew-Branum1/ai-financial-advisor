#!/usr/bin/env python3
"""
Final training script for long-term portfolio models.
This script loads the best hyperparameters from Optuna optimization and trains the final model.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from datetime import datetime
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import our modules
from rl.portfolio_env_long_term import PortfolioEnvLongTerm
from src.utils import load_market_data_with_indicators
from config_long_term import AGENT_TICKERS, BENCHMARK_TICKER, FEATURES_TO_USE_IN_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_best_hyperparameters(strategy="long_term"):
    """
    Load the best hyperparameters from Optuna optimization results.

    Args:
        strategy: Strategy type ("short_term" or "long_term")

    Returns:
        dict: Best hyperparameters
    """
    # Look for the most recent optimization results
    optimization_dir = "optimization_results"
    if not os.path.exists(optimization_dir):
        raise FileNotFoundError(
            f"Optimization directory {optimization_dir} not found. Run optimization first!"
        )

    # Find the most recent optimization results for this strategy
    strategy_dirs = [d for d in os.listdir(optimization_dir) if d.startswith(strategy)]
    if not strategy_dirs:
        raise FileNotFoundError(
            f"No optimization results found for {strategy}. Run optimization first!"
        )

    # Get the most recent one
    latest_dir = sorted(strategy_dirs)[-1]
    hyperparams_file = os.path.join(
        optimization_dir, latest_dir, "best_hyperparameters.json"
    )

    if not os.path.exists(hyperparams_file):
        raise FileNotFoundError(f"Hyperparameters file {hyperparams_file} not found!")

    with open(hyperparams_file, "r") as f:
        best_params = json.load(f)

    logger.info(f"Loaded best hyperparameters from {hyperparams_file}")
    return best_params


def create_env(df, env_params, is_training=True):
    """
    Create a portfolio environment with the given data and parameters.

    Args:
        df: Market data DataFrame
        env_params: Environment parameters
        is_training: Whether this is for training or evaluation

    Returns:
        PortfolioEnvLongTerm: Configured environment
    """
    # Prepare feature columns
    feature_columns = []
    for ticker in AGENT_TICKERS:
        for feature in FEATURES_TO_USE_IN_MODEL:
            feature_columns.append(f"{ticker}_{feature}")

    available_cols = [col for col in feature_columns if col in df.columns]
    if not available_cols:
        raise ValueError("No available features found in data")

    # Ensure DataFrame type for environment data
    if len(available_cols) == 1:
        df_subset = df[[available_cols[0]]]
    else:
        df_subset = df[available_cols]

    if not isinstance(df_subset, pd.DataFrame):
        df_subset = pd.DataFrame(df_subset)

    # Create environment
    env = PortfolioEnvLongTerm(
        df=df_subset,
        feature_columns_ordered=FEATURES_TO_USE_IN_MODEL,
        initial_balance=100000,
        transaction_cost_pct=0.001,
        **env_params,
    )

    # Wrap with Monitor for logging
    env = Monitor(env)

    return env


def train_long_term_final_model():
    """
    Train the final long-term portfolio model using optimized hyperparameters.
    """
    logger.info("Starting final long-term portfolio model training...")

    # Load best hyperparameters
    try:
        best_params = load_best_hyperparameters("long_term")
        env_params = best_params["env_params"]
        ppo_params = best_params["ppo_params"]
        logger.info("Successfully loaded optimized hyperparameters")
    except Exception as e:
        logger.warning(f"Failed to load optimized hyperparameters: {e}")
        logger.info("Using default hyperparameters")
        # Fallback to default parameters for long-term
        env_params = {
            "window_size": 60,
            "rolling_volatility_window": 40,
            "momentum_weight": 0.5,
            "mean_reversion_weight": 0.1,
            "volatility_target": 0.12,
            "turnover_penalty_weight": 0.008,
            "max_concentration_per_asset": 0.25,
            "min_holding_period": 15,
            "rebalancing_frequency": 10,
            "dividend_reinvestment": True,
        }
        ppo_params = {
            "learning_rate": 1e-4,
            "n_steps": 1024,
            "batch_size": 128,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.005,
        }

    # Load training data
    logger.info("Loading training data...")
    all_tickers = sorted(list(set(AGENT_TICKERS + [BENCHMARK_TICKER])))

    # Use full training period for long-term
    train_start_date = "2010-01-01"
    train_end_date = "2023-12-31"

    df_training = load_market_data_with_indicators(
        tickers_list=all_tickers,
        start_date=train_start_date,
        end_date=train_end_date,
        min_data_points=252 + 100,
    )

    if not isinstance(df_training, pd.DataFrame) or df_training.empty:
        raise ValueError("Failed to load training data.")

    logger.info(f"Training data loaded. Shape: {df_training.shape}")

    # Create training environment
    train_env = create_env(df_training, env_params, is_training=True)

    # Create evaluation environment (using different time period)
    eval_start_date = "2024-01-01"
    eval_end_date = "2024-12-31"

    df_eval = load_market_data_with_indicators(
        tickers_list=all_tickers,
        start_date=eval_start_date,
        end_date=eval_end_date,
        min_data_points=252 + 50,
    )

    if not isinstance(df_eval, pd.DataFrame) or df_eval.empty:
        logger.warning(
            "Failed to load evaluation data. Using training data for evaluation."
        )
        df_eval = df_training

    eval_env = create_env(df_eval, env_params, is_training=False)

    # Create model directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = f"models/long_term_final_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=15000, save_path=model_dir, name_prefix="long_term_model"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=7500,
        deterministic=True,
        render=False,
    )

    # Create and train model
    logger.info("Creating PPO model with optimized parameters...")
    logger.info(f"Environment parameters: {env_params}")
    logger.info(f"PPO parameters: {ppo_params}")

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=f"logs/long_term_final_{timestamp}",
        device="cpu",
        **ppo_params,
    )

    # Train the model (more timesteps for long-term)
    total_timesteps = 500000  # More timesteps for long-term learning
    logger.info(f"Starting training for {total_timesteps} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    logger.info(f"Training completed. Model saved to: {final_model_path}")

    # Save training configuration
    training_config = {
        "strategy": "long_term",
        "training_date": timestamp,
        "env_params": env_params,
        "ppo_params": ppo_params,
        "training_period": {"start": train_start_date, "end": train_end_date},
        "evaluation_period": {"start": eval_start_date, "end": eval_end_date},
        "total_timesteps": total_timesteps,
    }

    config_path = os.path.join(model_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)

    # Evaluate final model
    logger.info("Evaluating final model...")
    obs, _ = eval_env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += float(reward)
        steps += 1

        if terminated or truncated:
            break

    logger.info(f"Final evaluation - Total reward: {total_reward:.4f}, Steps: {steps}")
    logger.info(f"Final portfolio value: ${info['portfolio_value']:.2f}")

    return model, model_dir


if __name__ == "__main__":
    try:
        model, model_dir = train_long_term_final_model()
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {model_dir}")
        print(f"Best model: {model_dir}/best_model.zip")
        print(f"Final model: {model_dir}/final_model.zip")
        print(f"Training config: {model_dir}/training_config.json")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
