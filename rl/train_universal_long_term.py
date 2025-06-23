# rl/train_universal_long_term.py
import logging
import os
import pandas as pd
from datetime import datetime
import torch

# Import our custom classes
from src.utils import load_market_data_for_universal_env
from rl.universal_portfolio_env import UniversalPortfolioEnv
# --- FIX 1: Import the AttentionFeaturesExtractor as well ---
from rl.attention_policy import AttentionPolicy, AttentionFeaturesExtractor
from rl.custom_ppo import CustomPPO

# --- Configuration ---
TICKERS = ['MSFT', 'NVDA'] 
FEATURES_TO_USE = ['close', 'close_vs_sma_50', 'mfi', 'bollinger_width', 'obv', 'atr']
FEATURES_DIM = 64
TOP_K_STOCKS = 2
LOG_DIR = "logs"
MODEL_DIR = "models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train_long_term_model():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # --- Data Loading ---
    logging.info("Loading market data...")
    df = load_market_data_for_universal_env(
        tickers_list=TICKERS,
        feature_columns=FEATURES_TO_USE,
        start_date="2018-01-01",
        end_date="2023-01-01"
    )
    if df.empty:
        logging.error("Failed to load data. Ensure data_collector.py has been run for the specified tickers.")
        return

    # --- Environment Setup for LONG-TERM Goal ---
    logging.info("Setting up the Universal Portfolio Environment...")
    env = UniversalPortfolioEnv(
        df=df,
        feature_columns=FEATURES_TO_USE,
        window_size=30,
        top_k_stocks=TOP_K_STOCKS,
        sharpe_window=252,
        drawdown_penalty_weight=0.5
    )

    # We must explicitly tell the policy to use our custom features extractor class.
    policy_kwargs = {
        # --- FIX 2: Refer to the imported class directly ---
        "features_extractor_class": AttentionFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": FEATURES_DIM}
    }
    
    logging.info("Initializing CustomPPO model with AttentionPolicy...")
    model = CustomPPO(
        AttentionPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        device='cpu'
    )

    logging.info("--- Starting Integration Test (short training run) ---")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model.learn(
        total_timesteps=50_000, 
        tb_log_name=f"custom_ppo_long_term_{timestamp}"
    )
    
    logging.info("--- ✅ Integration Test Passed! The training loop runs without crashing. ---")


if __name__ == "__main__":
    train_long_term_model()