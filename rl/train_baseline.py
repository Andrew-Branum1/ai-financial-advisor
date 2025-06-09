# rl/train_baseline.py
import logging
import os
import pandas as pd

from rl.portfolio_env import PortfolioEnv # Ensure this is the version with reward shaping params
from src.utils import load_market_data_from_db

try:
    from stable_baselines3 import PPO
    # from stable_baselines3.common.env_checker import check_env
except ImportError:
    logging.error("Stable Baselines3 (PPO) not found. Please ensure it's installed.")
    PPO = None

# --- Learning Rate Schedule Definition ---
# This was likely the version of linear_schedule used for the "best pre-Optuna" model,
# taking one argument and using a global initial_lr.
initial_lr = 1e-4  # A common default, and what was in your earlier train_baseline.py

def linear_schedule(progress_remaining: float) -> float:
    """
    Linear learning rate schedule.
    :param progress_remaining: Progress remaining (starts at 1.0 and goes to 0.0)
    :return: current learning rate
    """
    return progress_remaining * initial_lr

if __name__ == "__main__":
    if PPO is None:
        logging.critical("PPO module not available. Exiting training.")
        exit()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration for the "Best Pre-Optuna 3-Asset Model" ---
    tickers_for_training = ['MSFT', 'GOOGL', 'AMZN'] # Key change for this model
    training_start_date = "2009-01-01"
    training_end_date = "2020-12-31" 
    window_size_env = 30
    initial_balance_env = 10000.0
    transaction_cost_percentage = 0.001

    # Reward Shaping Hyperparameters for PortfolioEnv
    # These were the values that led to better drawdown control before Optuna
    env_volatility_penalty_weight = 0.05
    env_loss_aversion_factor = 1.5
    env_rolling_volatility_window = 20

    features_to_use = [
        'close', 'rsi', 'volatility_20',
        'bollinger_hband', 'bollinger_lband', 'bollinger_mavg', 'atr'
    ]

    logging.info(f"Attempting to load training data for tickers: {tickers_for_training} with features: {features_to_use}")
    df_train = load_market_data_from_db(
        tickers_list=tickers_for_training,
        start_date=training_start_date,
        end_date=training_end_date,
        feature_columns=features_to_use,
        min_data_points=window_size_env + 50 + env_rolling_volatility_window 
    )

    if df_train.empty:
        logging.error("Failed to load training data or DataFrame is empty. Exiting.")
        exit()
    logging.info(f"Training data loaded. Shape: {df_train.shape}.")

    # --- Environment Setup ---
    logging.info("Initializing PortfolioEnv for training...")
    env = PortfolioEnv(
        df_train,
        feature_columns_ordered=features_to_use,
        window_size=window_size_env,
        initial_balance=initial_balance_env,
        transaction_cost_pct=transaction_cost_percentage,
        volatility_penalty_weight=env_volatility_penalty_weight,
        loss_aversion_factor=env_loss_aversion_factor,
        rolling_volatility_window=env_rolling_volatility_window
    )
    
    # --- Model Training ---
    log_dir = os.path.join(os.getcwd(), "logs", "ppo_3asset_preOptuna_tensorboard") # Suggest new log dir
    os.makedirs(log_dir, exist_ok=True)

    # PPO Hyperparameters (likely from your earlier settings before Optuna)
    # These are based on common defaults or values mentioned in earlier versions of your script.
    # Please verify against your own records if you had specific custom values here.
    ppo_n_epochs = 4  # Often seen as 'tuned_n_epochs' in earlier scripts
    ppo_gae_lambda = 0.92 # Often seen as 'tuned_gae_lambda'
    ppo_ent_coef = 0.001  # Often seen as 'tuned_ent_coef'
    ppo_vf_coef = 0.7     # Often seen as 'tuned_vf_coef'
    # Gamma and clip_range would likely have been PPO defaults (0.99 and 0.2 respectively) if not explicitly set.

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=linear_schedule, # Uses the 1-argument version with global initial_lr
        n_epochs=ppo_n_epochs,
        gae_lambda=ppo_gae_lambda,
        ent_coef=ppo_ent_coef,
        vf_coef=ppo_vf_coef
        # gamma and clip_range would use SB3 defaults if not specified here
    )

    total_timesteps_to_train = 300000 # Or 500,000 if you used that for the pre-Optuna 3-asset model

    model_tag = (
        f"ppo_3asset_preOptuna_tx{transaction_cost_percentage*100:.2f}_"
        f"volP{env_volatility_penalty_weight}_lossA{env_loss_aversion_factor}_"
        f"rollWin{env_rolling_volatility_window}_steps{total_timesteps_to_train//1000}k"
    )
    logging.info(f"Starting PPO model training ({model_tag}) for {total_timesteps_to_train} timesteps...")
    
    model.learn(
        total_timesteps=total_timesteps_to_train,
        progress_bar=True
    )
    logging.info("Model training complete.")

    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_name = f"{model_tag}_bestPerforming_v1.zip" # Example name
    model_save_path = os.path.join(models_dir, model_save_name)
    model.save(model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    logging.info("Training script finished.")