# test_env.py
import logging
from src.utils import load_market_data_from_db
from rl.portfolio_env import PortfolioEnv # Ensure rl.portfolio_env is correct
from stable_baselines3.common.env_checker import check_env

logging.basicConfig(level=logging.INFO)

env_tickers = ['AAPL', 'MSFT', 'GOOGL'] # Use a few tickers
df_for_env = load_market_data_from_db(tickers_list=env_tickers, min_data_points=100) # Need enough for window + steps

if not df_for_env.empty:
    print("\n--- Testing Portfolio Environment ---")
    try:
        env = PortfolioEnv(df_for_env, window_size=30, initial_balance=10000.0)
        print("✅ PortfolioEnv instantiated.")

        # Optional: Full environment check (can be verbose)
        # print("\nRunning SB3 environment checker (this might be verbose)...")
        # check_env(env)
        # print("✅ SB3 environment check completed (if uncommented and no errors).")

        print("\nTesting env.reset():")
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape}, dtype: {obs.dtype}")
        print(f"  Info: {info}")
        assert obs.shape == (env.window_size, env.asset_dim), "Observation shape mismatch after reset."

        print("\nTesting env.step() with a sample action:")
        action = env.action_space.sample() # Get a random valid action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Action taken: {action.round(2)}")
        print(f"  Next observation shape: {obs.shape}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Info: {info}")
        assert obs.shape == (env.window_size, env.asset_dim), "Observation shape mismatch after step."
        print("\n✅ Basic environment interactions seem OK.")

    except Exception as e:
        print(f"\n❌ Error during environment testing: {e}", exc_info=True)
else:
    print("\nSkipping environment test as data loading failed.")