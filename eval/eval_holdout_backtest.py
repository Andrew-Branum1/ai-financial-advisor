#!/usr/bin/env python3
"""
Holdout Backtest and Benchmark Comparison for Expanded Universe
Evaluates the trained model on a recent, never-seen holdout period (2023-2024)
Compares to SPY and equal-weighted portfolio.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import json
import csv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_short_term import AGENT_TICKERS, BENCHMARK_TICKER, FEATURES_TO_USE_IN_MODEL
from src.utils import load_market_data_with_indicators
from rl.portfolio_env_short_term import PortfolioEnvShortTerm

try:
    from stable_baselines3 import PPO
except ImportError:
    print("Stable Baselines3 (PPO) not found. Please ensure it's installed.")
    PPO = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Parameters ---
HOLDOUT_START = "2023-01-01"
HOLDOUT_END = "2024-12-31"
INITIAL_BALANCE = 100000
TRANSACTION_COST = 0.001


# --- Helper Functions ---
def get_latest_model(strategy="short_term"):
    models_dir = "models"
    strategy_prefix = f"{strategy}_final_"
    strategy_dirs = [d for d in os.listdir(models_dir) if d.startswith(strategy_prefix)]
    if not strategy_dirs:
        logger.error(f"No {strategy} models found in {models_dir}")
        return None
    latest_dir = sorted(strategy_dirs)[-1]
    model_dir = os.path.join(models_dir, latest_dir)
    for model_file in ["best_model.zip", "final_model.zip"]:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            logger.info(f"Found {strategy} model: {model_path}")
            return model_path
    logger.error(f"No model files found in {model_dir}")
    return None


def get_equal_weight_returns(df, tickers):
    close_cols = [f"{ticker}_close" for ticker in tickers]
    prices = df[close_cols]
    returns = prices.pct_change().fillna(0)
    ew_returns = returns.mean(axis=1)
    ew_cum = (1 + ew_returns).cumprod()
    return ew_cum, ew_returns


def get_spy_returns(df):
    spy_col = "SPY_close"
    if spy_col not in df.columns:
        logger.warning("SPY_close not in columns!")
        return None, None
    prices = df[spy_col]
    returns = prices.pct_change().fillna(0)
    cum = (1 + returns).cumprod()
    return cum, returns


def compute_metrics(cum_returns, returns):
    total_return = cum_returns.iloc[-1] - 1
    ann_return = cum_returns.iloc[-1] ** (252 / len(cum_returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = (
        (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        if returns.std() > 0
        else 0
    )
    drawdown = (cum_returns / cum_returns.cummax() - 1).min()
    return {
        "Total Return": f"{total_return*100:.2f}%",
        "Annualized Return": f"{ann_return*100:.2f}%",
        "Volatility": f"{volatility*100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{drawdown*100:.2f}%",
    }


def plot_results(dates, model_cum, spy_cum, ew_cum):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, model_cum, label="RL Model Portfolio")
    if spy_cum is not None:
        plt.plot(dates, spy_cum, label="SPY (S&P 500)")
    plt.plot(dates, ew_cum, label="Equal-Weighted Portfolio")
    plt.title("Holdout Backtest: Cumulative Returns (2023-2024)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (normalized)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    logger.info("Loading holdout data for 2023-2024...")
    all_tickers = sorted(list(set(AGENT_TICKERS + [BENCHMARK_TICKER])))
    df = load_market_data_with_indicators(
        tickers_list=all_tickers,
        start_date=HOLDOUT_START,
        end_date=HOLDOUT_END,
        min_data_points=252,
    )
    if df is None or df.empty:
        logger.error("No data loaded for holdout period!")
        return
    logger.info(f"Data loaded. Shape: {df.shape}")

    # --- Model Portfolio ---
    logger.info("Loading trained RL model...")
    model_path = get_latest_model("short_term")
    if model_path is None:
        logger.error("No trained model found!")
        return
    model = PPO.load(model_path)

    # Load window_size and features_to_use from training_config.json
    model_dir = os.path.dirname(model_path)
    with open(os.path.join(model_dir, "training_config.json")) as f:
        training_config = json.load(f)
    window_size = training_config["env_params"]["window_size"]
    features_to_use = training_config.get("ppo_params", {}).get(
        "features_to_use", FEATURES_TO_USE_IN_MODEL
    )
    if not features_to_use:
        features_to_use = training_config.get(
            "features_to_use", FEATURES_TO_USE_IN_MODEL
        )

    # Prepare environment with correct features and window size
    feature_columns = []
    for ticker in AGENT_TICKERS:
        for feature in features_to_use:
            feature_columns.append(f"{ticker}_{feature}")
    available_cols = [col for col in feature_columns if col in df.columns]
    df_subset = df[available_cols]
    env = PortfolioEnvShortTerm(
        df=df_subset,
        feature_columns_ordered=features_to_use,
        initial_balance=INITIAL_BALANCE,
        transaction_cost_pct=TRANSACTION_COST,
        window_size=window_size,
    )
    obs, _ = env.reset()
    model_portfolio_values = [env.portfolio_value]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        model_portfolio_values.append(info["portfolio_value"])
        done = terminated or truncated

    # --- Equal-Weighted Portfolio ---
    ew_cum, ew_returns = get_equal_weight_returns(df, AGENT_TICKERS)

    # --- SPY Benchmark ---
    spy_cum, spy_returns = get_spy_returns(df)

    # Align all series to the same length for plotting
    n = len(model_portfolio_values)
    plot_dates = df.index[-n:]
    model_cum = pd.Series(model_portfolio_values, index=plot_dates) / INITIAL_BALANCE
    model_returns = model_cum.pct_change().fillna(0)
    ew_cum = ew_cum[-n:]
    spy_cum = spy_cum[-n:] if spy_cum is not None else None

    # --- Metrics ---
    print("\n=== HOLDOUT BACKTEST RESULTS (2023-2024) ===")
    rl_metrics = compute_metrics(model_cum, model_returns)
    print("RL Model Portfolio:")
    print(rl_metrics)
    ew_metrics = compute_metrics(ew_cum, ew_returns[-n:])
    print("\nEqual-Weighted Portfolio:")
    print(ew_metrics)
    spy_metrics = None
    if spy_cum is not None:
        spy_metrics = compute_metrics(spy_cum, spy_returns[-n:])
        print("\nSPY (S&P 500):")
        print(spy_metrics)

    # --- Save metrics to CSV ---
    metrics_path = "holdout_backtest_metrics.csv"
    with open(metrics_path, "w", newline="") as csvfile:
        fieldnames = ["Strategy"] + list(rl_metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"Strategy": "RL Model", **rl_metrics})
        writer.writerow({"Strategy": "Equal-Weighted", **ew_metrics})
        if spy_metrics:
            writer.writerow({"Strategy": "SPY", **spy_metrics})
    print(f"\nMetrics saved to {metrics_path}")

    # --- Plot ---
    plot_results(plot_dates, model_cum, spy_cum, ew_cum)


if __name__ == "__main__":
    main()
