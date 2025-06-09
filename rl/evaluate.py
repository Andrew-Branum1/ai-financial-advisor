# rl/evaluate.py
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns # For KDE plots

# Absolute imports from the project root perspective
from rl.portfolio_env import PortfolioEnv # Assuming portfolio_env.py is in the 'rl' directory
from src.utils import load_market_data_from_db # Assuming utils.py is in the 'src' directory

try:
    from stable_baselines3 import PPO
except ImportError:
    logging.error("Stable Baselines3 (PPO) not found. Please ensure it's installed.")
    PPO = None

# --- Performance Metrics ---
def calculate_sharpe_ratio(portfolio_values: np.ndarray, risk_free_rate_daily: float = 0.0) -> float:
    if len(portfolio_values) < 2: return 0.0
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(daily_returns) == 0: return 0.0
    excess_returns = daily_returns - risk_free_rate_daily
    if np.std(excess_returns) == 0: return 0.0
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    annualized_sharpe = sharpe * np.sqrt(252)
    return annualized_sharpe

def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    if len(portfolio_values) == 0: return 0.0
    roll_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - roll_max) / np.where(roll_max == 0, 1e-9, roll_max)
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0
    return -max_dd

if __name__ == "__main__":
    if PPO is None:
        logging.critical("PPO module not available. Exiting evaluation.")
        exit()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    tickers_for_agent_evaluation = ['MSFT', 'GOOGL', 'AMZN']
    spy_ticker_symbol = 'SPY'
    tickers_to_load_for_data = sorted(list(set(tickers_for_agent_evaluation + [spy_ticker_symbol])))

    eval_start_date = "2023-01-01"
    eval_end_date = "2025-06-02"
    model_load_path = os.path.join(os.getcwd(), "models", "ppo_3asset_preOptuna_tx0.10_volP0.05_lossA1.5_rollWin20_steps300k_bestPerforming_v1.zip") # EXAMPLE PATH
    
    window_size_env_eval = 30
    initial_balance_env_eval = 10000.0
    transaction_cost_percentage_eval = 0.001

    eval_volatility_penalty_weight = 0.05
    eval_loss_aversion_factor = 1.5
    eval_rolling_volatility_window = 20

    features_to_use_eval = [
        'close', 'rsi', 'volatility_20',
        'bollinger_hband', 'bollinger_lband', 'bollinger_mavg', 'atr'
    ]

    logging.info(f"Attempting to load full evaluation data for: {tickers_to_load_for_data} with features: {features_to_use_eval}")
    df_full_eval_data = load_market_data_from_db(
        tickers_list=tickers_to_load_for_data,
        start_date=eval_start_date,
        end_date=eval_end_date,
        min_data_points=window_size_env_eval + 50 + eval_rolling_volatility_window,
        feature_columns=features_to_use_eval,
    )

    if df_full_eval_data.empty:
        logging.error("Failed to load full evaluation data. Exiting.")
        exit()
    logging.info(f"Full evaluation data loaded. Shape: {df_full_eval_data.shape}.")

    agent_specific_columns = []
    for ticker in tickers_for_agent_evaluation:
        for feature in features_to_use_eval:
            col_name = f"{ticker}_{feature}"
            if col_name in df_full_eval_data.columns:
                agent_specific_columns.append(col_name)
            else:
                logging.error(f"CRITICAL: Column {col_name} for agent not found.")
                exit()
    
    expected_agent_cols = len(tickers_for_agent_evaluation) * len(features_to_use_eval)
    if len(agent_specific_columns) != expected_agent_cols:
        logging.error(f"Agent column mismatch. Expected {expected_agent_cols}, got {len(agent_specific_columns)}.")
        exit()
    df_agent_env_data = df_full_eval_data[agent_specific_columns].copy()

    logging.info(f"Initializing PortfolioEnv for evaluation with {len(tickers_for_agent_evaluation)} agent assets...")
    eval_env = PortfolioEnv(
        df_agent_env_data,
        feature_columns_ordered=features_to_use_eval,
        window_size=window_size_env_eval,
        initial_balance=initial_balance_env_eval,
        transaction_cost_pct=transaction_cost_percentage_eval,
        volatility_penalty_weight=eval_volatility_penalty_weight,
        loss_aversion_factor=eval_loss_aversion_factor,
        rolling_volatility_window=eval_rolling_volatility_window
    )
    try:
        model = PPO.load(model_load_path, env=eval_env)
        logging.info(f"Model {model_load_path} loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_load_path}.")
        exit()
    except Exception as e:
        logging.error(f"Error initializing/loading model: {e}", exc_info=True)
        exit()

    # --- Run RL Agent Evaluation & Log Data for Analysis ---
    logging.info("Evaluating RL Agent...")
    obs, info = eval_env.reset()
    
    rl_portfolio_values = [eval_env.portfolio_value]
    agent_weights_history = [eval_env.weights.copy()]
    agent_turnover_history = [] 
    agent_transaction_costs_stepwise = [] # Log step-wise costs
    agent_cumulative_costs_history = [0.0] 
    agent_raw_daily_returns_history = []
    
    previous_agent_weights = eval_env.weights.copy()

    terminated, truncated = False, False
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info_step = eval_env.step(action)
        
        current_agent_weights = np.array(info_step['weights'])
        agent_weights_history.append(current_agent_weights)
        
        turnover = np.sum(np.abs(current_agent_weights - previous_agent_weights))
        agent_turnover_history.append(turnover)
        previous_agent_weights = current_agent_weights.copy()
        
        agent_transaction_costs_stepwise.append(info_step['transaction_costs'])
        agent_cumulative_costs_history.append(agent_cumulative_costs_history[-1] + info_step['transaction_costs'])
        agent_raw_daily_returns_history.append(info_step['raw_daily_return'])
        rl_portfolio_values.append(info_step['portfolio_value'])

    rl_portfolio_values = np.array(rl_portfolio_values)
    agent_weights_history_df = pd.DataFrame(agent_weights_history, columns=tickers_for_agent_evaluation)
    agent_raw_daily_returns_history = np.array(agent_raw_daily_returns_history)
    agent_turnover_history = np.array(agent_turnover_history)
    agent_cumulative_costs_history = np.array(agent_cumulative_costs_history) # Already includes initial 0


    logging.info(f"\n--- RL Agent Performance ({', '.join(tickers_for_agent_evaluation)}) ---")
    logging.info(f"Final Portfolio Value: ${rl_portfolio_values[-1]:.2f}")
    logging.info(f"Sharpe Ratio: {calculate_sharpe_ratio(rl_portfolio_values):.4f}")
    logging.info(f"Max Drawdown: {calculate_max_drawdown(rl_portfolio_values):.4%}")
    logging.info(f"Total Transaction Costs: ${agent_cumulative_costs_history[-1]:.2f}")
    logging.info(f"Average Daily Turnover: {np.mean(agent_turnover_history):.4f}" if len(agent_turnover_history) > 0 else "N/A")

    # --- Benchmarks ---
    num_actual_trading_days = len(rl_portfolio_values) - 1
    buy_and_hold_spy_values = np.array([])
    spy_daily_returns = np.array([])
    equal_weight_portfolio_values = np.array([])
    eq_daily_returns = np.array([])

    spy_close_col = f"{spy_ticker_symbol}_close"
    if spy_close_col in df_full_eval_data.columns:
        logging.info(f"\n--- Buy & Hold ({spy_ticker_symbol}) Benchmark ---")
        start_idx_spy = window_size_env_eval -1 
        end_idx_spy = start_idx_spy + num_actual_trading_days 
        if end_idx_spy < len(df_full_eval_data):
            spy_prices = df_full_eval_data[spy_close_col].iloc[start_idx_spy : end_idx_spy + 1].values
            if len(spy_prices) > 1:
                buy_and_hold_spy_values = (spy_prices / spy_prices[0]) * initial_balance_env_eval
                if len(buy_and_hold_spy_values) > 1:
                    spy_daily_returns = np.diff(buy_and_hold_spy_values) / buy_and_hold_spy_values[:-1]
                logging.info(f"Final Val: ${buy_and_hold_spy_values[-1]:.2f}, Sharpe: {calculate_sharpe_ratio(buy_and_hold_spy_values):.4f}, Max DD: {calculate_max_drawdown(buy_and_hold_spy_values):.4%}")
        else: logging.warning(f"SPY data too short.")
    else: logging.warning(f"SPY col '{spy_close_col}' not found.")
        
    logging.info(f"\n--- Equal-Weight Portfolio Benchmark ({', '.join(tickers_for_agent_evaluation)}) ---")
    eq_weight_close_cols = [f"{t}_close" for t in tickers_for_agent_evaluation if f"{t}_close" in df_full_eval_data.columns]
    if eq_weight_close_cols and len(eq_weight_close_cols) == len(tickers_for_agent_evaluation) and num_actual_trading_days > 0:
        num_assets_eq = len(eq_weight_close_cols)
        eq_w = np.ones(num_assets_eq) / num_assets_eq
        start_idx_eq = window_size_env_eval -1
        end_idx_eq = start_idx_eq + num_actual_trading_days
        if end_idx_eq < len(df_full_eval_data):
            prices_df_eq = df_full_eval_data[eq_weight_close_cols].iloc[start_idx_eq : end_idx_eq +1]
            if len(prices_df_eq) > 1:
                asset_returns_eq = prices_df_eq.pct_change().dropna() # First row is NaN, then N-1 returns
                portfolio_returns_eq = asset_returns_eq.dot(eq_w) # Series of N-1 daily returns
                
                temp_eq_values = [initial_balance_env_eval]
                for r in portfolio_returns_eq.values: temp_eq_values.append(temp_eq_values[-1] * (1 + r))
                equal_weight_portfolio_values = np.array(temp_eq_values)

                if len(equal_weight_portfolio_values) > 1:
                    eq_daily_returns = np.diff(equal_weight_portfolio_values) / equal_weight_portfolio_values[:-1]
                logging.info(f"Final Val: ${equal_weight_portfolio_values[-1]:.2f}, Sharpe: {calculate_sharpe_ratio(equal_weight_portfolio_values):.4f}, Max DD: {calculate_max_drawdown(equal_weight_portfolio_values):.4%}")
        else: logging.warning(f"Equal-weight data too short.")
    else: logging.warning(f"Equal-weight setup failed.")

    # ====== Plotting ======
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    base_plot_name = os.path.basename(model_load_path).replace(".zip", "") if model_load_path else "evaluation"

    # 1. Portfolio Value Comparison (remains the same)
    plt.figure(figsize=(14, 7))
    plt.plot(rl_portfolio_values, label=f"RL Agent ({len(tickers_for_agent_evaluation)} assets)", lw=2, zorder=3)
    if len(buy_and_hold_spy_values) > 0: plt.plot(buy_and_hold_spy_values, label=f"B&H {spy_ticker_symbol}", ls='--', zorder=2)
    if len(equal_weight_portfolio_values) > 0: plt.plot(equal_weight_portfolio_values, label=f"Eq-W. ({len(eq_weight_close_cols)} assets)", ls=':', zorder=1)
    plt.title("Portfolio Value Comparison"); plt.xlabel("Trading Timestep (Days)"); plt.ylabel("Portfolio Value ($)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{base_plot_name}_performance.png")); plt.close()
    logging.info(f"Performance plot saved to {os.path.join(plots_dir, f'{base_plot_name}_performance.png')}")

    # 2. Portfolio Weights Allocation
    if not agent_weights_history_df.empty:
        plt.figure(figsize=(14, 7))
        plt.stackplot(range(len(agent_weights_history_df)), agent_weights_history_df.T, labels=agent_weights_history_df.columns)
        plt.title("RL Agent Portfolio Weights Allocation"); plt.xlabel("Trading Timestep (Days)"); plt.ylabel("Weight Proportion")
        plt.legend(loc='upper left'); plt.margins(0,0); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{base_plot_name}_weights.png")); plt.close()
        logging.info(f"Weights allocation plot saved to {os.path.join(plots_dir, f'{base_plot_name}_weights.png')}")

    # 3. Portfolio Turnover
    if len(agent_turnover_history) > 0:
        plt.figure(figsize=(14, 7))
        plt.plot(agent_turnover_history, label="Daily Turnover", lw=1.5)
        plt.title("RL Agent Portfolio Turnover"); plt.xlabel("Trading Timestep (Days)"); plt.ylabel("Turnover (Sum of Abs Weight Changes)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{base_plot_name}_turnover.png")); plt.close()
        logging.info(f"Turnover plot saved to {os.path.join(plots_dir, f'{base_plot_name}_turnover.png')}")

    # 4. Cumulative Transaction Costs
    # Ensure agent_cumulative_costs_history has the same length as the number of steps if plotting on same x-axis
    # rl_portfolio_values has N+1 points, agent_raw_daily_returns has N points.
    # agent_cumulative_costs_history also has N+1 points (starts with 0)
    if len(agent_cumulative_costs_history) > 0 :
        plt.figure(figsize=(14, 7))
        plt.plot(agent_cumulative_costs_history, label="Cumulative Transaction Costs", lw=1.5, color='red')
        plt.title("RL Agent Cumulative Transaction Costs"); plt.xlabel("Trading Timestep (Days)"); plt.ylabel("Total Costs ($)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{base_plot_name}_transaction_costs.png")); plt.close()
        logging.info(f"Transaction costs plot saved to {os.path.join(plots_dir, f'{base_plot_name}_transaction_costs.png')}")

    # 5. Distribution of Daily Returns
    plt.figure(figsize=(12, 7))
    plot_dist_flag = False
    if len(agent_raw_daily_returns_history) > 0:
        sns.histplot(agent_raw_daily_returns_history, kde=True, label="RL Agent Raw Returns", stat="density", alpha=0.6, bins=50)
        plot_dist_flag = True
    if len(spy_daily_returns) > 0:
        sns.histplot(spy_daily_returns, kde=True, label=f"SPY B&H Returns", stat="density", alpha=0.6, bins=50)
        plot_dist_flag = True
    if len(eq_daily_returns) > 0:
        sns.histplot(eq_daily_returns, kde=True, label=f"Equal-Weight Returns", stat="density", alpha=0.6, bins=50)
        plot_dist_flag = True
    
    if plot_dist_flag:
        plt.title("Distribution of Daily Returns"); plt.xlabel("Daily Return"); plt.ylabel("Density")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{base_plot_name}_returns_distribution.png")); plt.close()
        logging.info(f"Returns distribution plot saved to {os.path.join(plots_dir, f'{base_plot_name}_returns_distribution.png')}")
    else:
        logging.info("Skipping returns distribution plot as no return series were available.")
        plt.close() # Close the figure if nothing was plotted


    # ====== CSV Data Export ======
    # Create DataFrames for CSV export - ensuring consistent lengths for time series data
    # Number of trading days (steps where returns/turnover occur)
    num_trading_days = num_actual_trading_days

    # Timestep index for series of length N (daily returns, turnover)
    day_index = pd.Index(range(1, num_trading_days + 1), name="Trading_Day")
    # Timestep index for series of length N+1 (portfolio values, weights, cumulative costs)
    value_index = pd.Index(range(num_trading_days + 1), name="Timestep_Value_Point")


    # Main summary DataFrame for time series data
    summary_data_dict = {}
    summary_data_dict['RL_Agent_Portfolio_Value'] = pd.Series(rl_portfolio_values, index=value_index)
    if len(agent_raw_daily_returns_history) == num_trading_days:
        summary_data_dict['RL_Agent_Raw_Daily_Return'] = pd.Series(agent_raw_daily_returns_history, index=day_index)
    if len(agent_turnover_history) == num_trading_days:
        summary_data_dict['RL_Agent_Daily_Turnover'] = pd.Series(agent_turnover_history, index=day_index)
    if len(agent_cumulative_costs_history) == num_trading_days + 1:
         summary_data_dict['RL_Agent_Cumulative_Costs'] = pd.Series(agent_cumulative_costs_history, index=value_index)


    if len(buy_and_hold_spy_values) == num_trading_days + 1:
        summary_data_dict['SPY_Portfolio_Value'] = pd.Series(buy_and_hold_spy_values, index=value_index)
    if len(spy_daily_returns) == num_trading_days:
        summary_data_dict['SPY_Daily_Return'] = pd.Series(spy_daily_returns, index=day_index)
    
    if len(equal_weight_portfolio_values) == num_trading_days + 1:
        summary_data_dict['EqualWeight_Portfolio_Value'] = pd.Series(equal_weight_portfolio_values, index=value_index)
    if len(eq_daily_returns) == num_trading_days:
        summary_data_dict['EqualWeight_Daily_Return'] = pd.Series(eq_daily_returns, index=day_index)

    # Combine into a single DataFrame, aligning by index (outer join to keep all timesteps)
    # Since some series are N and some N+1, we'll save weights separately.
    # For main summary, we can use day_index and add portfolio values carefully.
    
    # Let's create a DataFrame with a common index from 0 to num_trading_days
    # Portfolio values and cumulative costs are naturally on this index (N+1 points)
    # Daily returns and turnover are for day 1 to N (N points) - we can assign them to index 1 to N
    
    # Max length for DataFrame
    max_len = num_trading_days + 1
    df_index = pd.RangeIndex(max_len, name="Timestep")

    data_for_csv = {'Timestep': df_index}
    data_for_csv['RL_Agent_Value'] = pd.Series(rl_portfolio_values, index=df_index)
    data_for_csv['RL_Agent_Raw_Return'] = pd.Series([np.nan] + list(agent_raw_daily_returns_history) if len(agent_raw_daily_returns_history) == num_trading_days else [np.nan]*max_len, index=df_index)
    data_for_csv['RL_Agent_Turnover'] = pd.Series([np.nan] + list(agent_turnover_history) if len(agent_turnover_history) == num_trading_days else [np.nan]*max_len, index=df_index)
    data_for_csv['RL_Agent_Cumulative_Costs'] = pd.Series(agent_cumulative_costs_history, index=df_index)

    if len(buy_and_hold_spy_values) > 0:
        data_for_csv['SPY_Value'] = pd.Series(buy_and_hold_spy_values, index=df_index)
        data_for_csv['SPY_Return'] = pd.Series([np.nan] + list(spy_daily_returns) if len(spy_daily_returns) == num_trading_days else [np.nan]*max_len, index=df_index)
    
    if len(equal_weight_portfolio_values) > 0:
        data_for_csv['EqualWeight_Value'] = pd.Series(equal_weight_portfolio_values, index=df_index)
        data_for_csv['EqualWeight_Return'] = pd.Series([np.nan] + list(eq_daily_returns) if len(eq_daily_returns) == num_trading_days else [np.nan]*max_len, index=df_index)

    summary_df = pd.DataFrame(data_for_csv)
    summary_csv_path = os.path.join(plots_dir, f"{base_plot_name}_summary_timeseries_data.csv")
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.6f')
    logging.info(f"Summary timeseries data saved to {summary_csv_path}")

    # Weights history (already a DataFrame: agent_weights_history_df)
    # Index should represent the point in time the weights are set FOR (0 to N)
    agent_weights_history_df.index.name = "Timestep_Weights_Set_For_Next_Day"
    weights_csv_path = os.path.join(plots_dir, f"{base_plot_name}_weights_history.csv")
    agent_weights_history_df.to_csv(weights_csv_path, float_format='%.6f')
    logging.info(f"Agent weights history saved to {weights_csv_path}")


    logging.info("\nEvaluation script finished.")