# train.py
import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# --- Import custom modules ---
# Add the project root to the Python path to find our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import STRATEGY_CONFIGS
from src.data_manager import load_market_data_from_db
# Import both environment classes
from rl.portfolio_env_short_term import PortfolioEnvShortTerm
from rl.portfolio_env_long_term import PortfolioEnvLongTerm

# --- Custom Logging Callback ---
class TrainingLogCallback(BaseCallback):
    """A simple callback to log training progress."""
    def __init__(self, check_freq: int = 2000, verbose: int = 1):
        super(TrainingLogCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.check_freq == 0:
            reward = self.locals['rewards'][0]
            info = self.locals['infos'][0]
            value = info.get('portfolio_value', 0)
            print(f"  Step: {self.num_timesteps}, Reward: {reward:.4f}, Portfolio Value: ${value:,.2f}")
        return True

def train_model(term, risk_profile, config, full_data):
    """
    Prepares data and trains a single model, intelligently handling NaN values
    from different IPO dates.
    """
    model_name = f"ppo_portfolio_{term}_{risk_profile}"
    model_dir = f"models/{model_name}/"
    model_path = f"{model_dir}/model.zip"
    log_dir = f"logs/{model_name}/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "="*60)
    print(f"--- Preparing Model: {model_name} ---")

    try:
        # 1. Select the required columns for this specific model
        required_cols = [f"{ticker}_{feature}" for ticker in config['tickers'] for feature in config['features_to_use']]
        
        # Filter the full dataset to only the columns needed for this model
        available_cols = [col for col in required_cols if col in full_data.columns]
        model_data = full_data[available_cols].copy()

        # 2. Find the valid start date for training this model
        first_valid_date = model_data.dropna().index.min()
        if pd.isna(first_valid_date):
            print(f"!!! Skipping {model_name}: No window where all ticker data is available.")
            return

        train_data = model_data.loc[first_valid_date:].copy()
        print(f"Training data available from {first_valid_date.date()}. Shape: {train_data.shape}")

        # 3. Setup RL Environment - *** THIS IS THE KEY FIX ***
        # Dynamically select the correct environment class based on the 'term'
        if term == "short_term":
            env_class = PortfolioEnvShortTerm
        elif term == "long_term":
            env_class = PortfolioEnvLongTerm
        else:
            raise ValueError(f"Unknown term: {term}")

        env = DummyVecEnv([lambda: env_class(df=train_data, 
                                             feature_columns_ordered=config['features_to_use'],
                                             **config['env_params'])])

        # 4. Train the PPO Model
        model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=log_dir, device='cpu', **config['ppo_params'])

        print(f"--- Training Model: {model_name} ---")
        total_timesteps = max(10000, len(train_data) * 5)
        model.learn(total_timesteps=total_timesteps, callback=TrainingLogCallback())

        # 5. Save the final model
        model.save(model_path)
        print(f"--- Successfully trained and saved model to {model_path} ---")

    except Exception as e:
        print(f"!!! ERROR training {model_name}: {e} !!!")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to load data once, then train all models.
    """
    print("--- Starting AI Financial Advisor Model Training Pipeline ---")

    try:
        market_data = load_market_data_from_db()
        # *** THE FIX: Ensure the index is a DatetimeIndex ***
        market_data.index = pd.to_datetime(market_data.index)
        
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        print("Please run 'python src/data_manager.py' first to build the database.")
        return
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}")
        return

    # Loop through all strategies defined in the config and train them
    for term, risk_profiles in STRATEGY_CONFIGS.items():
        for risk, config in risk_profiles.items():
            train_model(term, risk, config, market_data)

    print("\n--- All model training sessions are complete. ---")

if __name__ == "__main__":
    main()