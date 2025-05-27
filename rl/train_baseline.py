# rl/train_baseline.py
from stable_baselines3 import PPO
from rl.portfolio_env import PortfolioEnv
from utils.data_loader import load_market_data
import os

if __name__ == "__main__":
    df = load_market_data()
    env = PortfolioEnv(df)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/tensorboard/")
    model.learn(total_timesteps=100_000)
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_portfolio_baseline")