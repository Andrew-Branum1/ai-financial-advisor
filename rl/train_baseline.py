# rl/train_baseline.py
import logging
import os
import pandas as pd

from rl.portfolio_env import PortfolioEnv
from src.utils import load_market_data_from_db

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    logging.error("Stable Baselines3 (PPO) not found. Please ensure it's installed.")
    PPO = None

initial_lr = 1e-4

def linear_schedule(progress_remaining: float) -> float:
    return progress_remaining * initial_lr

if __name__ == "__main__":
    if PPO is None:
        logging.critical("PPO module not available. Exiting training.")
        exit()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    tickers_for_training = ['MSFT', 'GOOGL', 'AMZN']
    training_start_date = "2009-01-01"
    training_end_date = "2020-12-31"
    window_size_env = 30
    initial_balance_env = 10000.0
    transaction_cost_percentage = 0.001

    # Reward Shaping Hyperparameters for PortfolioEnv (tune these)
    env_volatility_penalty_weight = 0.05
    env_loss_aversion_factor = 1.5
    env_rolling_volatility_window = 20

    features_to_use = [
        'close', 'rsi', 'volatility_20',
        'bollinger_hband', 'bollinger_lband', 'bollinger_mavg', 'atr'
    ]

    logging.info(f"Attempting to load training data for tickers: {tickers_for_training} with features: {features_to_use}")
    try:
        df_train = load_market_data_from_db(
            tickers_list=tickers_for_training,
            start_date=training_start_date,
            end_date=training_end_date,
            feature_columns=features_to_use,
            min_data_points=window_size_env + 100 + env_rolling_volatility_window # Ensure enough data for rolling window
        )
    except Exception as e:
        logging.critical(f"Fatal error loading data: {e}", exc_info=True)
        df_train = pd.DataFrame()

    if df_train.empty:
        logging.error("Failed to load training data or DataFrame is empty. Exiting.")
        exit()

    logging.info(f"Training data loaded. Shape: {df_train.shape}. Columns: {df_train.columns.tolist()}")

    # --- Environment Setup ---
    logging.info("Initializing PortfolioEnv for training...")
    try:
        env = PortfolioEnv(
            df_train,
            feature_columns_ordered=features_to_use,
            window_size=window_size_env,
            initial_balance=initial_balance_env,
            transaction_cost_pct=transaction_cost_percentage,
            volatility_penalty_weight=env_volatility_penalty_weight, # New param
            loss_aversion_factor=env_loss_aversion_factor,           # New param
            rolling_volatility_window=env_rolling_volatility_window  # New param
        )
        # check_env(env) # Optional, can be verbose
    except ValueError as e:
        logging.error(f"Error initializing PortfolioEnv: {e}", exc_info=True)
        exit()
    except Exception as e:
        logging.error(f"Unexpected error initializing PortfolioEnv: {e}", exc_info=True)
        exit()

    # --- Model Training ---
    log_dir = os.path.join(os.getcwd(), "logs", "ppo_portfolio_tensorboard")
    os.makedirs(log_dir, exist_ok=True)

    # PPO Hyperparameters (good candidates for tuning to reduce drawdown)
    tuned_n_epochs = 4
    tuned_gae_lambda = 0.95     # Consider values like 0.95-0.99 for potentially more stable advantage estimates
    tuned_ent_coef = 0.001      # Tune for exploration vs exploitation balance
    tuned_vf_coef = 0.5         # Default is 0.5, ensure value function is well learned
    # ppo_gamma = 0.99            # Discount factor, already default. Higher (e.g. 0.995) = more future-focused.
    # ppo_clip_range = 0.2        # Default. Smaller (e.g. 0.1) = more conservative updates.

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=linear_schedule,
        n_epochs=tuned_n_epochs,
        gae_lambda=tuned_gae_lambda,        # Consider tuning
        ent_coef=tuned_ent_coef,            # Consider tuning
        vf_coef=tuned_vf_coef,
        # gamma=ppo_gamma,                  # Uncomment to tune
        # clip_range=ppo_clip_range,        # Uncomment to tune
        # net_arch=[dict(pi=[128, 128], vf=[128, 128])] # Example: Tune network architecture
    )

    total_timesteps_to_train = 300000 # Increase if needed
    model_tag = (f"ppo_tx{transaction_cost_percentage*100:.2f}_"
                 f"volP{env_volatility_penalty_weight}_lossA{env_loss_aversion_factor}_"
                 f"feats{len(features_to_use)}_{total_timesteps_to_train//1000}k_drawdownFocus")
    logging.info(f"Starting PPO model training ({model_tag}) for {total_timesteps_to_train} timesteps...")

    try:
        model.learn(
            total_timesteps=total_timesteps_to_train,
            progress_bar=True
        )
        logging.info("Model training complete.")
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}", exc_info=True)
        exit()

    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_name = f"{model_tag}_v1.zip"
    model_save_path = os.path.join(models_dir, model_save_name)
    try:
        model.save(model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}", exc_info=True)

    logging.info("Training script finished.")