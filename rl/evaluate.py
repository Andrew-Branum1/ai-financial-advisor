# rl/evaluate.py
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Absolute imports from the project root perspective
from rl.portfolio_env import PortfolioEnv
from src.utils import load_market_data_from_db

# Ensure Stable Baselines3 PPO is imported
try:
    from stable_baselines3 import PPO
except ImportError:
    logging.error("Stable Baselines3 (PPO) not found. Please ensure it's installed.")
    # Exiting or handling as appropriate
    PPO = None # To prevent NameError if script continues briefly

# --- Performance Metrics ---
def calculate_sharpe_ratio(portfolio_values: np.ndarray, risk_free_rate_daily: float = 0.0) -> float:
    if len(portfolio_values) < 2: return 0.0
    daily_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
    # daily_returns = pd.Series(portfolio_values).pct_change().dropna().values # Alternative
    
    excess_returns = daily_returns - risk_free_rate_daily
    if np.std(excess_returns) == 0: return 0.0 # Avoid division by zero
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    annualized_sharpe = sharpe * np.sqrt(252) # Assuming 252 trading days
    return annualized_sharpe

def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    if len(portfolio_values) == 0: return 0.0
    roll_max = np.maximum.accumulate(portfolio_values)
    # Prevent division by zero if roll_max is 0 (though portfolio values should be positive)
    drawdown = (portfolio_values - roll_max) / np.where(roll_max == 0, 1e-9, roll_max)
    max_dd = np.min(drawdown) # Will be negative or zero
    return -max_dd # Return as a positive value (percentage)

if __name__ == "__main__":
    if PPO is None: # Check if import failed
        logging.critical("PPO module not available. Exiting evaluation.")
        exit()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    tickers_for_evaluation = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    eval_start_date = "2021-01-01"  # Example
    eval_end_date = "2023-12-31"    # Example
    model_load_path = os.path.join(os.getcwd(), "models", "ppo_lr_schedule_15yr_train_300k_v1.zip") # Corrected path
    window_size_env_eval = 30        # Must match the window_size used for the loaded model's env
    initial_balance_env_eval = 10000.0 # Should ideally match training or be set appropriately
    features_to_use = ['close', 'rsi', 'volatility_20'] 
    
    # Optional: Define specific date range for out-of-sample evaluation
    # eval_start_date = "2024-01-01"
    # eval_end_date = None

    logging.info(f"Attempting to load evaluation data for: {tickers_for_evaluation}")
    try:
        df_eval = load_market_data_from_db(
            tickers_list=tickers_for_evaluation,
            start_date=eval_start_date, # Uncomment to use
            end_date=eval_end_date,     # Uncomment to use
            min_data_points=window_size_env_eval + 50, # Enough for env + some eval steps
            feature_columns=features_to_use,
        )
    except Exception as e:
        logging.critical(f"Fatal error loading evaluation data: {e}")
        df_eval = pd.DataFrame()

    if df_eval.empty:
        logging.error("Failed to load evaluation data or DataFrame is empty. Exiting.")
        exit()

    # The manual column name cleaning `df.columns = ...` should NO LONGER be needed here
    # if `load_market_data_from_db` returns clean column names (ticker symbols).
    logging.info(f"Evaluation data loaded. Shape: {df_eval.shape}. Columns: {df_eval.columns.tolist()}")

    # --- Initialize Environment and Load Model ---
    logging.info("Initializing PortfolioEnv for evaluation...")
    try:
        eval_env = PortfolioEnv(df_eval, window_size=window_size_env_eval, initial_balance=initial_balance_env_eval,feature_columns=features_to_use,)
        model = PPO.load(model_load_path, env=eval_env) # Pass env to ensure compatibility
        logging.info(f"Model {model_load_path} loaded successfully.")
    except Exception as e:
        logging.error(f"Error initializing evaluation environment or loading model: {e}", exc_info=True)
        exit()

    # --- Run RL Agent Evaluation ---
    logging.info("Evaluating RL Agent...")
    obs, info = eval_env.reset()
    rl_portfolio_values = [eval_env.portfolio_value] # Store initial balance
    
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info_step = eval_env.step(action)
        rl_portfolio_values.append(info_step['portfolio_value'])
    rl_portfolio_values = np.array(rl_portfolio_values)

    logging.info(f"\n--- RL Agent Performance ---")
    logging.info(f"Final Portfolio Value: ${rl_portfolio_values[-1]:.2f}")
    logging.info(f"Sharpe Ratio: {calculate_sharpe_ratio(rl_portfolio_values):.4f}")
    logging.info(f"Max Drawdown: {calculate_max_drawdown(rl_portfolio_values):.4%}")

    # --- Benchmarks ---
    num_actual_trading_days = len(rl_portfolio_values) - 1 # -1 for initial balance

    # ====== 1. Buy and Hold 'SPY' Benchmark ======
    spy_ticker_symbol = 'SPY'
    buy_and_hold_spy_values = []
    if spy_ticker_symbol in df_eval.columns:
        logging.info(f"\n--- Buy & Hold ({spy_ticker_symbol}) Benchmark ---")
        # Align SPY data with the agent's trading period
        # Agent's first trade uses prices at index `window_size_env_eval`
        # Agent makes `num_actual_trading_days` trades.
        
        if len(df_eval) >= window_size_env_eval + num_actual_trading_days :
            spy_prices_eval_period = df_eval[spy_ticker_symbol].iloc[window_size_env_eval : window_size_env_eval + num_actual_trading_days].values
            
            if len(spy_prices_eval_period) > 0:
                # Calculate portfolio values, starting with initial_balance_env_eval
                buy_and_hold_spy_values_temp = (spy_prices_eval_period / spy_prices_eval_period[0]) * initial_balance_env_eval
                buy_and_hold_spy_values = np.insert(buy_and_hold_spy_values_temp, 0, initial_balance_env_eval) # Add initial balance at start

                logging.info(f"Final Portfolio Value: ${buy_and_hold_spy_values[-1]:.2f}")
                logging.info(f"Sharpe Ratio: {calculate_sharpe_ratio(buy_and_hold_spy_values):.4f}")
                logging.info(f"Max Drawdown: {calculate_max_drawdown(buy_and_hold_spy_values):.4%}")
            else:
                logging.warning(f"Not enough SPY price data points for the evaluation period.")
        else:
            logging.warning(f"DataFrame too short for SPY Buy & Hold benchmark over agent's trading period.")
    else:
        logging.warning(f"Ticker '{spy_ticker_symbol}' not found in evaluation data. Skipping Buy & Hold SPY benchmark.")
        
    # ====== 2. Equal-Weight Portfolio Benchmark (using all assets in df_eval) ======
    logging.info("\n--- Equal-Weight Portfolio Benchmark ---")
    equal_weight_portfolio_values = [initial_balance_env_eval]
    num_assets_for_eq_weight = df_eval.shape[1]
    eq_weights = np.ones(num_assets_for_eq_weight) / num_assets_for_eq_weight

    if num_actual_trading_days > 0:
        # Iterate through the actual trading period of the agent
        # The prices used are from window_size to window_size + num_actual_trading_days
        for i in range(window_size_env_eval, window_size_env_eval + num_actual_trading_days):
            prev_day_prices_all_assets = df_eval.iloc[i].values      # Prices at start of day t (or end of t-1)
            current_day_prices_all_assets = df_eval.iloc[i+1].values # Prices at end of day t

            if not (np.all(np.isfinite(prev_day_prices_all_assets)) and \
                    np.all(np.isfinite(current_day_prices_all_assets)) and \
                    np.all(prev_day_prices_all_assets > 0)):
                daily_asset_returns_eq = np.zeros(num_assets_for_eq_weight)
            else:
                daily_asset_returns_eq = (current_day_prices_all_assets / prev_day_prices_all_assets) - 1
            
            portfolio_daily_return_eq = np.dot(daily_asset_returns_eq, eq_weights)
            current_portfolio_value_eq = equal_weight_portfolio_values[-1] * (1 + portfolio_daily_return_eq)
            equal_weight_portfolio_values.append(current_portfolio_value_eq)
        
        equal_weight_portfolio_values = np.array(equal_weight_portfolio_values)
        if len(equal_weight_portfolio_values) > 1:
            logging.info(f"Final Portfolio Value: ${equal_weight_portfolio_values[-1]:.2f}")
            logging.info(f"Sharpe Ratio: {calculate_sharpe_ratio(equal_weight_portfolio_values):.4f}")
            logging.info(f"Max Drawdown: {calculate_max_drawdown(equal_weight_portfolio_values):.4%}")
        else:
            logging.warning("Not enough data to calculate Equal-Weight benchmark values.")
    else:
        logging.warning("No trading steps recorded for agent, skipping Equal-Weight benchmark.")


    # ====== Plot Comparison ======
    plt.figure(figsize=(14, 7))
    plt.plot(rl_portfolio_values, label="RL Agent", linewidth=2)
    
    if len(buy_and_hold_spy_values) > 0:
        plt.plot(buy_and_hold_spy_values[:len(rl_portfolio_values)], label=f"Buy & Hold {spy_ticker_symbol}", linestyle='--')

    if len(equal_weight_portfolio_values) > 1: # Ensure there's more than just initial balance
        plt.plot(equal_weight_portfolio_values[:len(rl_portfolio_values)], label="Equal-Weight Portfolio", linestyle=':')
        
    plt.title("Portfolio Value Comparison During Evaluation")
    plt.xlabel("Trading Timestep")
    plt.ylabel(f"Portfolio Value ($) - Initial: ${initial_balance_env_eval:,.0f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plots_dir = "plots" # Relative to where evaluate.py is run from
    os.makedirs(plots_dir, exist_ok=True)
    plot_save_path = os.path.join(plots_dir, "portfolio_evaluation_comparison.png")
    try:
        plt.savefig(plot_save_path)
        logging.info(f"Evaluation plot saved to {plot_save_path}")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
    plt.show()

    logging.info("\nEvaluation script finished.")