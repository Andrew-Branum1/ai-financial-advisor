# rl/train_baseline.py
import logging
import os
import pandas as pd

# Absolute imports from the project root perspective
from rl.portfolio_env import PortfolioEnv
from src.utils import load_market_data_from_db

# Ensure Stable Baselines3 is imported
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env # Optional
except ImportError:
    logging.error("Stable Baselines3 (PPO) not found. Please ensure it's installed.")
    PPO = None # To prevent further NameErrors if script is partially run

# --- Learning Rate Schedule Definition ---
# Define your initial learning rate for the schedule
initial_lr = 1e-4  # This was the good fixed learning rate you were using

def linear_schedule(progress_remaining: float) -> float:
    """
    Linear learning rate schedule.
    :param progress_remaining: Progress remaining (starts at 1.0 and goes to 0.0)
    :return: current learning rate
    """
    return progress_remaining * initial_lr

# You could also define other schedules here, e.g., exponential decay if desired

if __name__ == "__main__":
    if PPO is None:
        logging.critical("PPO module not available. Exiting training.")
        exit()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    tickers_for_training = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    window_size_env = 30
    initial_balance_env = 10000.0

    logging.info(f"Attempting to load training data for tickers: {tickers_for_training}")
    try:
        df_train = load_market_data_from_db(
            tickers_list=tickers_for_training,
            min_data_points=window_size_env + 100
        )
    except Exception as e:
        logging.critical(f"Fatal error loading data: {e}", exc_info=True)
        df_train = pd.DataFrame()

    if df_train.empty:
        logging.error("Failed to load training data or DataFrame is empty. Exiting.")
        exit()

    logging.info(f"Training data loaded successfully. Shape: {df_train.shape}")

    # --- Environment Setup ---
    logging.info("Initializing PortfolioEnv for training...")
    try:
        env = PortfolioEnv(df_train, window_size=window_size_env, initial_balance=initial_balance_env)
        # check_env(env) # Optional
    except ValueError as e:
        logging.error(f"Error initializing PortfolioEnv: {e}", exc_info=True)
        exit()
    except Exception as e:
        logging.error(f"Unexpected error initializing PortfolioEnv: {e}", exc_info=True)
        exit()
    
    # --- Model Training ---
    log_dir = os.path.join(os.getcwd(), "logs", "ppo_portfolio_tensorboard")
    os.makedirs(log_dir, exist_ok=True)

    # Your existing tuned hyperparameters
    tuned_n_epochs = 4
    tuned_gae_lambda = 0.92
    tuned_ent_coef = 0.001
    tuned_vf_coef = 0.7

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=linear_schedule,  # <-- Pass the schedule function here
        n_epochs=tuned_n_epochs,
        gae_lambda=tuned_gae_lambda,
        ent_coef=tuned_ent_coef,
        vf_coef=tuned_vf_coef
        # Other PPO parameters will use their defaults (e.g., n_steps=2048, batch_size=64, gamma=0.99 etc.)
        # unless you explicitly set them.
    )

    # For a run with LR scheduling, you might want to train for a longer duration
    # to see the full effect of the decaying learning rate.
    total_timesteps_to_train = 300000 # Example: try 300k or 500k
    logging.info(f"Starting PPO model training for {total_timesteps_to_train} timesteps with LR schedule (initial LR: {initial_lr})...")
    
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
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    # IMPORTANT: Give this model a new unique name!
    model_save_name = f"ppo_lr_schedule_{total_timesteps_to_train//1000}k_v1.zip"
    model_save_path = os.path.join(models_dir, model_save_name)
    try:
        model.save(model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}", exc_info=True)

    logging.info("Training script finished.")