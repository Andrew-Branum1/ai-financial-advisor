# rl/train_baseline.py
import logging
import os
from datetime import datetime

# Absolute imports from the project root perspective
from rl.portfolio_env_alpha import PortfolioEnvAlpha
from src.utils import load_and_split_data
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# --- MODIFICATION: Import all settings from the central config file ---
from config_alpha import (
    AGENT_TICKERS,
    FEATURES_TO_USE_IN_MODEL,
    BEST_PPO_PARAMS,
    ENV_PARAMS
)

# --- Configuration for file paths ---
LOGS_DIR = "logs/"
MODELS_DIR = "models/"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_NAME = f"PPO_Portfolio_Alpha_{timestamp}"
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
LOG_PATH = os.path.join(LOGS_DIR, MODEL_NAME)


def train_final_model():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Data Loading ---
    train_start_date = "2007-01-01"
    train_end_date = "2022-12-31"
    eval_start_date = "2023-01-01"
    eval_end_date = "2023-12-31"

    logging.info("Loading training data...")
    # --- MODIFICATION: Pass consistent tickers and features to the data loader ---
    df_train = load_and_split_data(
        tickers_list=AGENT_TICKERS,
        feature_columns=FEATURES_TO_USE_IN_MODEL,
        start_date=train_start_date,
        end_date=train_end_date
    )
    logging.info("Loading evaluation data for callback...")
    df_eval = load_and_split_data(
        tickers_list=AGENT_TICKERS,
        feature_columns=FEATURES_TO_USE_IN_MODEL,
        start_date=eval_start_date,
        end_date=eval_end_date
    )

    if df_train.empty or df_eval.empty:
        logging.error("Failed to load data. Exiting training.")
        return

    # --- Environment Setup ---
    logging.info("Setting up environments...")
    # --- MODIFICATION: Pass consistent features and params to the environment ---
    train_env = PortfolioEnvAlpha(df=df_train, feature_columns_ordered=FEATURES_TO_USE_IN_MODEL, **ENV_PARAMS)
    eval_env = PortfolioEnvAlpha(df=df_eval, feature_columns_ordered=FEATURES_TO_USE_IN_MODEL, **ENV_PARAMS)
    
    train_env = Monitor(train_env, LOG_PATH)
    eval_env = Monitor(eval_env)
    
    # --- Callback to Save Best Model ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, f"best_{MODEL_NAME}"),
        log_path=LOG_PATH,
        eval_freq=max(BEST_PPO_PARAMS['n_steps'] * 2, 1000),
        n_eval_episodes=1,
        deterministic=True,
        render=False
    )

    # --- Model Training ---
    logging.info("Creating PPO model with best hyperparameters...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        device='cpu',
        **BEST_PPO_PARAMS
    )

    total_training_timesteps = 1_000_000 
    logging.info(f"Starting final model training for {total_training_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=total_training_timesteps,
        callback=eval_callback,
        tb_log_name=MODEL_NAME
    )
    
    logging.info(f"Training complete. Saving final model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_final_model()
