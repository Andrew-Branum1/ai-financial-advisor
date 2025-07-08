#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for long-term portfolio models.
This script ONLY finds the best hyperparameters, it does NOT train the final model.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Import our modules
from rl.portfolio_env_long_term import PortfolioEnvLongTerm
from src.utils import load_market_data_with_indicators
from config_long_term import AGENT_TICKERS, BENCHMARK_TICKER, FEATURES_TO_USE_IN_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def objective(trial):
    """
    Optuna objective function for long-term model hyperparameter optimization.
    Returns the Sharpe ratio for the given hyperparameters.
    """
    print(f"--- Starting Trial {trial.number} ---")

    # Load data for optimization
    all_tickers = sorted(list(set(AGENT_TICKERS + [BENCHMARK_TICKER])))

    # Use longer periods for long-term optimization
    train_start_date = "2010-01-01"
    train_end_date = "2022-12-31"
    val_start_date = "2023-01-01"
    val_end_date = "2024-01-01"

    # Load training data
    df_train = load_market_data_with_indicators(
        tickers_list=all_tickers,
        start_date=train_start_date,
        end_date=train_end_date,
        min_data_points=252 + 100,
    )

    # Load validation data
    df_val = load_market_data_with_indicators(
        tickers_list=all_tickers,
        start_date=val_start_date,
        end_date=val_end_date,
        min_data_points=252 + 50,
    )

    if (
        not isinstance(df_train, pd.DataFrame)
        or df_train.empty
        or not isinstance(df_val, pd.DataFrame)
        or df_val.empty
    ):
        raise optuna.exceptions.TrialPruned("No data available")

    # Prepare feature columns
    feature_columns = []
    for ticker in AGENT_TICKERS:
        for feature in FEATURES_TO_USE_IN_MODEL:
            feature_columns.append(f"{ticker}_{feature}")

    available_cols = [col for col in feature_columns if col in df_train.columns]
    if not available_cols:
        raise optuna.exceptions.TrialPruned("No available features")

    # Hyperparameter search space for long-term strategy
    env_params = {
        "window_size": trial.suggest_int(
            "window_size", 30, 90
        ),  # Longer windows for long-term
        "rolling_volatility_window": trial.suggest_int(
            "rolling_volatility_window", 20, 60
        ),
        "momentum_weight": trial.suggest_float(
            "momentum_weight", 0.2, 0.8
        ),  # Higher momentum for long-term
        "mean_reversion_weight": trial.suggest_float(
            "mean_reversion_weight", 0.05, 0.3
        ),
        "volatility_target": trial.suggest_float(
            "volatility_target", 0.08, 0.20
        ),  # Lower volatility for long-term
        "turnover_penalty_weight": trial.suggest_float(
            "turnover_penalty_weight", 0.002, 0.015
        ),
        "max_concentration_per_asset": trial.suggest_float(
            "max_concentration_per_asset", 0.15, 0.4
        ),
        "min_holding_period": trial.suggest_int(
            "min_holding_period", 5, 30
        ),  # Longer holding periods
        "rebalancing_frequency": trial.suggest_int("rebalancing_frequency", 5, 20),
        "dividend_reinvestment": trial.suggest_categorical(
            "dividend_reinvestment", [True, False]
        ),
    }

    ppo_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical(
            "n_steps", [256, 512, 1024, 2048]
        ),  # More steps for long-term
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs": trial.suggest_int("n_epochs", 5, 15),
        "gamma": trial.suggest_float(
            "gamma", 0.95, 0.999
        ),  # Higher gamma for long-term
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float(
            "ent_coef", 0.0, 0.005
        ),  # Lower entropy for long-term
    }

    try:
        # Ensure DataFrame type for environment data
        if len(available_cols) == 1:
            df_train_subset = df_train[[available_cols[0]]]
            df_val_subset = df_val[[available_cols[0]]]
        else:
            df_train_subset = df_train[available_cols]
            df_val_subset = df_val[available_cols]

        if not isinstance(df_train_subset, pd.DataFrame):
            df_train_subset = pd.DataFrame(df_train_subset)
        if not isinstance(df_val_subset, pd.DataFrame):
            df_val_subset = pd.DataFrame(df_val_subset)

        # Create environments
        train_env = PortfolioEnvLongTerm(
            df=df_train_subset,
            feature_columns_ordered=FEATURES_TO_USE_IN_MODEL,
            initial_balance=100000,
            transaction_cost_pct=0.001,
            **env_params,
        )

        val_env = PortfolioEnvLongTerm(
            df=df_val_subset,
            feature_columns_ordered=FEATURES_TO_USE_IN_MODEL,
            initial_balance=100000,
            transaction_cost_pct=0.001,
            **env_params,
        )

        # Import PPO here to avoid issues
        from stable_baselines3 import PPO

        # Create model
        model = PPO("MlpPolicy", train_env, verbose=0, device="cpu", **ppo_params)

        # Quick training for optimization (reduced timesteps)
        print(f"Trial {trial.number}: Training for optimization...")
        model.learn(total_timesteps=75000)  # More timesteps for long-term optimization
        print(f"Trial {trial.number}: Training complete.")

        # Evaluate on validation set
        obs, _ = val_env.reset()
        total_reward = 0
        portfolio_values = [val_env.portfolio_value]

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = val_env.step(action)
            total_reward += reward
            portfolio_values.append(info["portfolio_value"])
            done = terminated or truncated

        # Calculate Sharpe ratio
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if np.std(returns) > 0
                else 0
            )
        else:
            sharpe_ratio = 0

        # Store additional metrics
        trial.set_user_attr("total_reward", total_reward)
        trial.set_user_attr("final_portfolio_value", portfolio_values[-1])
        trial.set_user_attr(
            "total_return", (portfolio_values[-1] / portfolio_values[0]) - 1
        )

        print(f"--- Trial {trial.number} Results ---")
        print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Final Portfolio Value: ${portfolio_values[-1]:.2f}")

        return sharpe_ratio

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned(f"Trial failed: {e}")


def optimize_long_term_hyperparameters():
    """
    Run hyperparameter optimization for long-term models.
    """
    print("Starting hyperparameter optimization for long-term models...")
    print("This will find the best parameters but NOT train the final model.")
    print("=" * 80)

    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
    )

    # Run optimization
    try:
        study.optimize(
            objective, n_trials=30, timeout=10800
        )  # 3 hours timeout for long-term
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"optimization_results/long_term_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save best parameters
    best_params = {
        "env_params": {},
        "ppo_params": {},
        "optimization_info": {
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "optimization_date": timestamp,
            "strategy": "long_term",
        },
    }

    # Extract best parameters
    for key, value in study.best_trial.params.items():
        if key in [
            "window_size",
            "rolling_volatility_window",
            "momentum_weight",
            "mean_reversion_weight",
            "volatility_target",
            "turnover_penalty_weight",
            "max_concentration_per_asset",
            "min_holding_period",
            "rebalancing_frequency",
            "dividend_reinvestment",
        ]:
            best_params["env_params"][key] = value
        else:
            best_params["ppo_params"][key] = value

    # Save to file
    with open(f"{results_dir}/best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Best Sharpe Ratio: {study.best_value:.4f}")
    print(f"Number of Trials: {len(study.trials)}")
    print(f"Best Trial: {study.best_trial.number}")

    print("\nBest Environment Parameters:")
    for key, value in best_params["env_params"].items():
        print(f"  {key}: {value}")

    print("\nBest PPO Parameters:")
    for key, value in best_params["ppo_params"].items():
        print(f"  {key}: {value}")

    print(f"\nResults saved to: {results_dir}/best_hyperparameters.json")
    print("\nNext step: Use these parameters to train the final model!")

    return best_params


if __name__ == "__main__":
    try:
        best_params = optimize_long_term_hyperparameters()
        print("\n✅ Hyperparameter optimization completed successfully!")

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise
