# eval/test_universal_env.py
import logging
import pandas as pd
import numpy as np

from src.utils import load_market_data_for_universal_env
from rl.universal_portfolio_env import UniversalPortfolioEnv

# --- Configuration for a quick test ---
TEST_TICKERS = ['NVDA', 'MSFT'] # Use a minimal set of tickers
FEATURES_TO_USE = ['close', 'close_vs_sma_50', 'mfi', 'bollinger_width', 'obv', 'atr']
TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2022-06-30"
TOP_K_STOCKS = 2 # Must be <= number of tickers

def run_manual_env_test():
    """
    This test manually checks the environment's core logic (reset and step)
    by providing a correctly formatted compound action.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- Starting Manual Environment Test ---")

    # 1. Load a small amount of data
    df = load_market_data_for_universal_env(
        tickers_list=TEST_TICKERS,
        feature_columns=FEATURES_TO_USE,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE
    )
    if df.empty:
        logging.error("Test failed: No data loaded. Ensure data_collector.py has run for NVDA and MSFT.")
        return

    # 2. Instantiate the environment
    logging.info("Instantiating UniversalPortfolioEnv...")
    env = UniversalPortfolioEnv(
        df=df,
        feature_columns=FEATURES_TO_USE,
        top_k_stocks=TOP_K_STOCKS
    )
    
    # 3. Test the reset method
    logging.info("Testing reset()...")
    obs, info = env.reset()
    assert isinstance(obs, dict) and "features" in obs and "mask" in obs, "Reset did not return a valid observation."
    assert obs['features'].shape == (len(TEST_TICKERS), env.window_size, len(FEATURES_TO_USE)), "Observation shape is incorrect."
    logging.info("Reset successful.")

    # 4. Manually create a valid compound action
    # This simulates what our custom policy will do: provide indices and allocation weights.
    # The indices correspond to the positions in the `env.all_tickers` list.
    sample_indices = np.array([0, 1])  # Corresponds to NVDA and MSFT
    sample_allocations = np.array([0.6, 0.4]) # 60% to NVDA, 40% to MSFT
    action = (sample_indices, sample_allocations)
    
    logging.info(f"Testing step() with a manual action: {action}")
    
    # 5. Test the step method
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        logging.info("Step successful.")
        assert isinstance(obs, dict), "Step did not return a valid observation."
        assert isinstance(reward, float), "Step did not return a float reward."
        logging.info("--- ✅ Manual Environment Test Passed! ---")
    except Exception as e:
        logging.error(f"--- ❌ Manual Environment Test Failed during step! ---")
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_manual_env_test()