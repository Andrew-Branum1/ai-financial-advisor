# rl/train_baseline.py
import logging
import os
import pandas as pd

# Absolute imports from the project root perspective
from rl.portfolio_env import PortfolioEnv
from src.utils import load_market_data_from_db

# Ensure Stable Baselines3 is imported (can be here or later)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env # Optional
except ImportError:
    logging.error("Stable Baselines3 (PPO) not found. Please ensure it's installed.")
    # Depending on how you structure, you might exit here or let it fail later
    # For now, let's assume it will be caught if used before PPO() call

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    # Define tickers for training, ensure they are collected by data_collector.py
    tickers_for_training = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    # Optional: Define a specific date range for training data
    # training_start_date = "2020-01-01"
    # training_end_date = "2023-12-31"
    window_size_env = 30
    initial_balance_env = 10000.0

    logging.info(f"Attempting to load training data for tickers: {tickers_for_training}")
    try:
        df_train = load_market_data_from_db(
            tickers_list=tickers_for_training,
            # start_date=training_start_date, # Uncomment to use specific dates
            # end_date=training_end_date,     # Uncomment to use specific dates
            min_data_points=window_size_env + 100 # Ensure enough data for window and some learning
        )
    except Exception as e:
        logging.critical(f"Fatal error loading data: {e}")
        df_train = pd.DataFrame() # Ensure df_train is defined for the empty check

    if df_train.empty:
        logging.error("Failed to load training data or DataFrame is empty. Exiting.")
        exit()

    logging.info(f"Training data loaded successfully. Shape: {df_train.shape}")
    # print("Training Data Head:\n", df_train.head()) # For debugging

    # --- Environment Setup ---
    logging.info("Initializing PortfolioEnv for training...")
    try:
        env = PortfolioEnv(df_train, window_size=window_size_env, initial_balance=initial_balance_env)
        # check_env(env) # Optional: Check custom environment (can be verbose)
        # logging.info("Environment check passed (if uncommented).")
    except ValueError as e:
        logging.error(f"Error initializing PortfolioEnv: {e}")
        exit()
    
    # --- Model Training ---
    log_dir = "./logs/ppo_portfolio_tensorboard/" # For TensorBoard logs
    os.makedirs(log_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        # Common hyperparameters to consider tuning:
        learning_rate=1e-4,  # 0.0003
        # n_steps=2048,        # Steps to run for each environment per update
        # batch_size=64,
        n_epochs=4,
        # gamma=0.99,
        gae_lambda=0.92,
        # clip_range=0.2,
        ent_coef=0.001,        # Entropy coefficient for exploration
        vf_coef=0.7          # Value function coefficient
    )

    total_timesteps_to_train = 150000 # Adjust based on complexity and available time
    logging.info(f"Starting PPO model training for {total_timesteps_to_train} timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps_to_train,
            progress_bar=True
        )
        logging.info("Model training complete.")
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}", exc_info=True)
        exit()

    # --- Save Model ---
    models_dir = os.path.join(os.getcwd(), "models") # Use absolute or well-defined relative path
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "ppo_portfolio_baseline_150k_run_fix_attmpt.zip")
    try:
        model.save(model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

    logging.info("Training script finished.")