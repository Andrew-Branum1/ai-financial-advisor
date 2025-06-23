import optuna
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import numpy as np

# Assuming portfolio_env and utils are in the correct path
from rl.portfolio_env_short_term import PortfolioEnvShortTerm
from src.utils import calculate_evaluation_kpis, load_and_split_data
from config_short_term import FEATURES_TO_USE_IN_MODEL

# In train_short_term_optuna.py, replace the objective function

def objective(trial):
    print(f"--- Starting Trial {trial.number} ---")
    
    optuna_train_start_date = "2010-01-01"
    optuna_train_end_date = "2022-12-31"
    optuna_validation_start_date = "2023-01-01"
    optuna_validation_end_date = "2024-12-31"

    df_train = load_and_split_data(start_date=optuna_train_start_date, end_date=optuna_train_end_date, feature_columns=FEATURES_TO_USE_IN_MODEL)
    df_eval = load_and_split_data(start_date=optuna_validation_start_date, end_date=optuna_validation_end_date, feature_columns=FEATURES_TO_USE_IN_MODEL)

    if df_train.empty or df_eval.empty:
        raise optuna.exceptions.TrialPruned()

    # --- HYPERPARAMETER SEARCH FOR A SHORT-TERM AGENT ---
    env_params = {
        "rolling_volatility_window": trial.suggest_int("rolling_volatility_window", 10, 40),
        "turnover_penalty_weight": trial.suggest_float("turnover_penalty_weight", 0.0, 0.05),
        "max_concentration_per_asset": trial.suggest_float("max_concentration_per_asset", 0.5, 1.0)
    }
    
    ppo_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_epochs": trial.suggest_int("n_epochs", 5, 15),
        "gamma": trial.suggest_float("gamma", 0.90, 0.99), # Lower gamma for short-term focus
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024]),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05)
    }

    train_env = PortfolioEnvShortTerm(df=df_train, feature_columns_ordered=FEATURES_TO_USE_IN_MODEL, **env_params)
    eval_env = PortfolioEnvShortTerm(df=df_eval, feature_columns_ordered=FEATURES_TO_USE_IN_MODEL, **env_params)

    train_env = Monitor(train_env)
    eval_env = Monitor(eval_env)
    
    model = PPO("MlpPolicy", train_env, verbose=0, device='cpu', **ppo_params)
    
    print(f"Trial {trial.number}: Training...")
    model.learn(total_timesteps=100000) # Reduced timesteps for faster iteration
    print(f"Trial {trial.number}: Training complete.")

    kpis = calculate_evaluation_kpis(model, eval_env)
    
    print(f"--- Trial {trial.number} Results ---")
    for key, value in kpis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
            trial.set_user_attr(key, value)
    
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