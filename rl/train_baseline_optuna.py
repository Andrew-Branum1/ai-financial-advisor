import optuna
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import numpy as np

# Assuming portfolio_env and utils are in the correct path
from rl.portfolio_env import PortfolioEnv
from src.utils import calculate_evaluation_kpis, load_and_split_data
from config import FEATURES_TO_USE_IN_MODEL

def objective(trial):
    """
    The objective function for Optuna optimization.
    NOW MODIFIED to log detailed KPIs for each trial.
    """
    print(f"--- Starting Trial {trial.number} ---")
    
    # --- Data Loading ---
    # (This part remains the same)
    optuna_train_start_date = "2007-01-01"
    optuna_train_end_date = "2018-12-31"
    optuna_validation_start_date = "2019-01-01"
    optuna_validation_end_date = "2020-12-31"

    df_optuna_train = load_and_split_data(
        start_date=optuna_train_start_date,
        end_date=optuna_train_end_date,
        feature_columns=FEATURES_TO_USE_IN_MODEL
    )
    df_optuna_validation = load_and_split_data(
        start_date=optuna_validation_start_date,
        end_date=optuna_validation_end_date,
        feature_columns=FEATURES_TO_USE_IN_MODEL
    )

    if df_optuna_train.empty or df_optuna_validation.empty:
        print("Data loading failed, skipping trial.")
        raise optuna.exceptions.TrialPruned()

    # --- Hyperparameter Search Space ---
    # (This part remains the same)
    learning_rate_val = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_epochs_val = trial.suggest_int("n_epochs", 5, 20)
    gae_lambda_val = trial.suggest_float("gae_lambda", 0.9, 1.0)
    ent_coef_val = trial.suggest_float("ent_coef", 0.0, 0.1)
    vf_coef_val = trial.suggest_float("vf_coef", 0.2, 0.8)
    gamma_val = trial.suggest_float("gamma", 0.99, 0.9999)
    clip_range_val = trial.suggest_float("clip_range", 0.1, 0.4)
    n_steps_val = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    env_volatility_penalty = trial.suggest_float("volatility_penalty_weight", 0.0, 2.0)
    env_loss_aversion = trial.suggest_float("loss_aversion_factor", 1.0, 2.5)
    env_rolling_vol_window = trial.suggest_int("rolling_volatility_window", 30, 252)
    env_turnover_penalty = trial.suggest_float("turnover_penalty_weight", 0.0, 0.2)
    env_max_concentration = trial.suggest_float("max_concentration_per_asset", 0.4, 1.0)

    # --- Environment Setup ---
    # (This part remains the same, assuming you already added feature_columns_ordered)
    train_env_trial = PortfolioEnv(df=df_optuna_train, feature_columns_ordered=FEATURES_TO_USE_IN_MODEL, volatility_penalty_weight=env_volatility_penalty, loss_aversion_factor=env_loss_aversion, rolling_volatility_window=env_rolling_vol_window, turnover_penalty_weight=env_turnover_penalty, max_concentration_per_asset=env_max_concentration)
    validation_env_trial = PortfolioEnv(df=df_optuna_validation, feature_columns_ordered=FEATURES_TO_USE_IN_MODEL, volatility_penalty_weight=0.0, loss_aversion_factor=1.0)

    train_env_trial = Monitor(train_env_trial)
    validation_env_trial = Monitor(validation_env_trial)

    # --- Model Training ---
    # (This part remains the same, assuming you already added device='cpu')
    total_timesteps_per_trial = 150000 
    model_trial = PPO("MlpPolicy", train_env_trial, learning_rate=learning_rate_val, n_steps=n_steps_val, n_epochs=n_epochs_val, gamma=gamma_val, gae_lambda=gae_lambda_val, clip_range=clip_range_val, ent_coef=ent_coef_val, vf_coef=vf_coef_val, verbose=0, device='cpu')
    
    print(f"Trial {trial.number}: Training for {total_timesteps_per_trial} timesteps...")
    model_trial.learn(total_timesteps=total_timesteps_per_trial)
    print(f"Trial {trial.number}: Training complete.")

    # --- NEW: Evaluation and Logging ---
    kpis = calculate_evaluation_kpis(model_trial, validation_env_trial)
    
    # Log the detailed KPIs for this trial
    print(f"--- Trial {trial.number} Results ---")
    for key, value in kpis.items():
        print(f"  {key}: {value:.4f}")
        # Save each KPI as a user attribute for later analysis
        trial.set_user_attr(key, value)
    
    # Return the single objective value for Optuna to optimize
    return kpis.get("sharpe_ratio", -1.0)

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    )
    
    # MODIFICATION: Adjusted n_trials. With longer trials, you might run fewer, 
    # or leave it running longer depending on your hardware.
    try:
        study.optimize(objective, n_trials=25) # You might adjust this number
    except KeyboardInterrupt:
        print("Study interrupted by user.")

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Sharpe Ratio): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")