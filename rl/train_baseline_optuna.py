# rl/train_baseline_optuna.py
import logging
import os
import pandas as pd
import numpy as np
import optuna # Import Optuna

# Ensure Stable Baselines3 is imported
try:
    from stable_baselines3 import PPO
    # from stable_baselines3.common.env_checker import check_env # Optional for trials
except ImportError:
    logging.error("Stable Baselines3 (PPO) not found. Please ensure it's installed.")
    PPO = None # To prevent further NameErrors

# Assuming your custom environment and data loader are correctly imported
# These paths assume the script is run from the project root, e.g., python -m rl.train_baseline_optuna
from rl.portfolio_env import PortfolioEnv
from src.utils import load_market_data_from_db

# --- Helper: Learning Rate Schedule (Optional, can tune fixed LR too) ---
DEFAULT_INITIAL_LR = 1e-4
def linear_schedule(progress_remaining: float, initial_lr_for_schedule: float = DEFAULT_INITIAL_LR) -> float:
    """
    Linear learning rate schedule.
    :param progress_remaining: Progress remaining (starts at 1.0 and goes to 0.0)
    :param initial_lr_for_schedule: The initial learning rate for this schedule.
    :return: current learning rate
    """
    return progress_remaining * initial_lr_for_schedule

# --- Helper: Simplified Sharpe Ratio for Optuna Objective ---
def calculate_sharpe_for_optuna(portfolio_values: np.ndarray, risk_free_rate_daily: float = 0.0) -> float:
    """Calculates annualized Sharpe ratio from an array of portfolio values."""
    if len(portfolio_values) < 2:
        return -float('inf') # Not enough data for Sharpe
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(daily_returns) == 0:
        return -float('inf')
    excess_returns = daily_returns - risk_free_rate_daily
    std_dev_excess_returns = np.std(excess_returns)
    if std_dev_excess_returns < 1e-9: # Avoid division by zero or near-zero std dev
        return 0.0 if np.mean(excess_returns) >= 0 else -float('inf') # Penalize if mean return is negative with no risk
        
    sharpe = np.mean(excess_returns) / std_dev_excess_returns
    annualized_sharpe = sharpe * np.sqrt(252) # Assuming 252 trading days
    return annualized_sharpe if np.isfinite(annualized_sharpe) else -float('inf')


# --- Objective Function for Optuna ---
def objective(trial: optuna.Trial):
    """
    Objective function for Optuna to optimize.
    Trains a PPO model with suggested hyperparameters and returns a score.
    """
    if PPO is None:
        logging.critical("PPO module not available in objective function. Exiting trial.")
        raise optuna.exceptions.TrialPruned("PPO not available.")

    logging.info(f"Starting Trial {trial.number}...")

    # === 1. Suggest Hyperparameters ===
    # PPO Hyperparameters
    # Using a fixed learning rate for simplicity in Optuna, or use the schedule with a tuned initial_lr
    learning_rate_val = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_epochs_val = trial.suggest_int("n_epochs", 2, 10)
    gae_lambda_val = trial.suggest_float("gae_lambda", 0.9, 0.99)
    ent_coef_val = trial.suggest_float("ent_coef", 1e-8, 0.05, log=True) # Adjusted upper bound
    vf_coef_val = trial.suggest_float("vf_coef", 0.2, 0.8)
    gamma_val = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999]) # More focused gamma
    clip_range_val = trial.suggest_categorical("clip_range", [0.1, 0.15, 0.2, 0.25]) # More focused clip_range

    # PortfolioEnv Reward Shaping Hyperparameters
    env_vol_penalty = trial.suggest_float("volatility_penalty_weight", 0.0, 0.15) # Adjusted range
    env_loss_aversion = trial.suggest_float("loss_aversion_factor", 1.0, 2.0) # Adjusted range
    env_rolling_vol_window = trial.suggest_int("rolling_volatility_window", 15, 35) # Adjusted range

    # === 2. Configuration for this trial ===
    tickers_for_training_trial = ['MSFT', 'GOOGL', 'AMZN'] # Focus on the 3-asset model
    training_start_date_trial = "2009-01-01"
    # For faster Optuna trials, you might use a shorter training period or a specific validation period.
    # Here, we'll use a portion of the full training data.
    # Let's define a training end date for loading, and then potentially split for validation.
    training_end_date_for_loading = "2020-12-31" 
    
    window_size_env_trial = 30
    initial_balance_env_trial = 10000.0
    transaction_cost_percentage_trial = 0.001

    features_to_use_trial = [
        'close', 'rsi', 'volatility_20',
        'bollinger_hband', 'bollinger_lband', 'bollinger_mavg', 'atr'
    ]
    
    # Reduced timesteps for faster Optuna trials
    total_timesteps_per_trial = 100000 # Increased slightly for more learning signal

    # === 3. Load Data ===
    # It's crucial to have a separate validation set for a robust Optuna objective.
    # For simplicity in this example, we'll train on a period and evaluate on a subsequent period.
    # Example: Train 2009-2018, Validate 2019-2020
    optuna_train_end_date = "2018-12-31"
    optuna_validation_start_date = "2019-01-01"
    optuna_validation_end_date = "2020-12-31"

    df_optuna_train = load_market_data_from_db(
        tickers_list=tickers_for_training_trial,
        start_date=training_start_date_trial,
        end_date=optuna_train_end_date,
        feature_columns=features_to_use_trial,
        min_data_points=window_size_env_trial + 50 + env_rolling_vol_window 
    )

    df_optuna_validation = load_market_data_from_db(
        tickers_list=tickers_for_training_trial,
        start_date=optuna_validation_start_date,
        end_date=optuna_validation_end_date,
        feature_columns=features_to_use_trial,
        min_data_points=window_size_env_trial + 50 + env_rolling_vol_window # Min points for validation period
    )

    if df_optuna_train.empty or len(df_optuna_train) < (window_size_env_trial + total_timesteps_per_trial / 100): # Heuristic check
        logging.warning(f"Trial {trial.number}: Not enough training data ({len(df_optuna_train)} rows), pruning.")
        raise optuna.exceptions.TrialPruned()
    if df_optuna_validation.empty or len(df_optuna_validation) < (window_size_env_trial + 60): # Need at least ~3 months for validation
        logging.warning(f"Trial {trial.number}: Not enough validation data ({len(df_optuna_validation)} rows), pruning.")
        raise optuna.exceptions.TrialPruned()

    # === 4. Setup Training Environment ===
    try:
        train_env_trial = PortfolioEnv(
            df_optuna_train,
            feature_columns_ordered=features_to_use_trial,
            window_size=window_size_env_trial,
            initial_balance=initial_balance_env_trial,
            transaction_cost_pct=transaction_cost_percentage_trial,
            volatility_penalty_weight=env_vol_penalty,
            loss_aversion_factor=env_loss_aversion,
            rolling_volatility_window=env_rolling_vol_window
        )
    except ValueError as e:
        logging.warning(f"Trial {trial.number}: Training Env init error {e}, pruning.")
        raise optuna.exceptions.TrialPruned()

    # === 5. Train Model ===
    model_trial = PPO(
        "MlpPolicy",
        train_env_trial,
        verbose=0, 
        learning_rate=learning_rate_val, # Using fixed learning rate suggested by Optuna
        n_epochs=n_epochs_val,
        gae_lambda=gae_lambda_val,
        ent_coef=ent_coef_val,
        vf_coef=vf_coef_val,
        gamma=gamma_val,
        clip_range=clip_range_val,
        # Consider adding n_steps if you want to tune it:
        # n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096]) 
    )

    try:
        model_trial.learn(total_timesteps=total_timesteps_per_trial, progress_bar=False)
    except Exception as e: # Catch broader exceptions during learning
        logging.warning(f"Trial {trial.number}: Model training error '{e}', pruning.")
        raise optuna.exceptions.TrialPruned()

    # === 6. Evaluate Model on Validation Set and Return Score ===
    try:
        validation_env_trial = PortfolioEnv(
            df_optuna_validation, # Use the separate validation DataFrame
            feature_columns_ordered=features_to_use_trial,
            window_size=window_size_env_trial,
            initial_balance=initial_balance_env_trial,
            transaction_cost_pct=transaction_cost_percentage_trial,
            # Reward shaping params for validation env can be defaults or match training
            # They don't affect model.predict, only the logged reward if you were using it
            volatility_penalty_weight=0.0, # Set to 0 for raw performance eval
            loss_aversion_factor=1.0,    # Set to 1 for raw performance eval
            rolling_volatility_window=env_rolling_vol_window
        )
    except ValueError as e:
        logging.warning(f"Trial {trial.number}: Validation Env init error {e}, pruning.")
        raise optuna.exceptions.TrialPruned()

    obs_val, _ = validation_env_trial.reset()
    portfolio_values_validation = [validation_env_trial.portfolio_value]
    
    terminated_val, truncated_val = False, False
    while not (terminated_val or truncated_val):
        action_val, _ = model_trial.predict(obs_val, deterministic=True)
        obs_val, _, terminated_val, truncated_val, info_step_val = validation_env_trial.step(action_val)
        portfolio_values_validation.append(info_step_val['portfolio_value'])
    
    # Calculate a score. Higher is better for maximization.
    # Using Sharpe Ratio on the validation set is a good objective.
    score = calculate_sharpe_for_optuna(np.array(portfolio_values_validation))
    # Alternative: score = portfolio_values_validation[-1] # Final portfolio value

    logging.info(f"Trial {trial.number} finished. Score (Validation Sharpe): {score:.4f}. Params: {trial.params}")
    
    # Clean up to free memory
    del model_trial, train_env_trial, validation_env_trial, df_optuna_train, df_optuna_validation
    
    return score


if __name__ == "__main__":
    if PPO is None:
        logging.critical("PPO module not available. Exiting Optuna script.")
        exit()

    # Setup logging for the main script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Optuna Study ---
    study_name = "ppo_3asset_sharpe_optimization_v2" # Give a unique name
    storage_name = f"sqlite:///{study_name}.db" # SQLite database to store study results
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        logging.info(f"Resuming existing Optuna study: {study_name} from {storage_name}")
    except KeyError: 
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize", # We want to maximize Sharpe Ratio
            # Example pruner: stops unpromising trials early
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5) 
        )
        logging.info(f"Created new Optuna study: {study_name} stored at {storage_name}")

    try:
        # Adjust n_trials: more trials = better search, but takes longer
        # Start with a moderate number like 50-100 to test.
        study.optimize(objective, n_trials=100, timeout=3600*4) # e.g., 50 trials, or max 4 hours
    except KeyboardInterrupt:
        logging.info("Optuna study interrupted by user. Results so far are saved.")
    except Exception as e:
        logging.error(f"An error occurred during the Optuna study: {e}", exc_info=True)

    print("\nOptuna Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")

    try:
        best_trial = study.best_trial
        print("Best trial:")
        print(f"  Value (Best Validation Sharpe): {best_trial.value:.4f}")
        print("  Best Hyperparameters found:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
        # Save best parameters to a file
        best_params_file = os.path.join("models", f"{study_name}_best_params.txt")
        os.makedirs("models", exist_ok=True)
        with open(best_params_file, "w") as f:
            f.write(f"Best trial value (Validation Sharpe): {best_trial.value:.4f}\n")
            f.write("Best Hyperparameters:\n")
            for key, value in best_trial.params.items():
                f.write(f"  {key}: {value}\n")
        logging.info(f"Best hyperparameters saved to {best_params_file}")

    except ValueError:
        logging.warning("No trials were completed successfully in the Optuna study (e.g., all pruned or errored).")


    logging.info(f"Optuna script finished. Study '{study_name}' results are in '{storage_name}'.")
    logging.info("To use the best parameters, manually update your main train_baseline.py and retrain for the full duration.")
