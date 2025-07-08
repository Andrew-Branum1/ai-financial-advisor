import sys
import os
import argparse
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import json
import asyncio
from src.utils import (
    UserProfile,
    map_user_profile_to_env_params,
    load_market_data_with_indicators,
)
from app import ModelManager, InvestmentProfile
from llm.advisor import generate_investment_report

# --- Parameters ---
ROLLING_WINDOW_YEARS = 2
TEST_WINDOW_YEARS = 1
START_YEAR = 2018
END_YEAR = 2024

# Example user profile (customize as needed)
user_profile = UserProfile(
    name="Demo User",
    age=30,
    income=60000,
    investment_amount=10000,
    time_horizon="long_term",
    risk_tolerance="moderate",
    goal="growth",
)

# Map user profile to RL environment parameters
env_params = map_user_profile_to_env_params(user_profile)

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--model_name", required=True)
args = parser.parse_args()
model_dir = args.model_dir
model_name = args.model_name

# PATCH: Write results to a unique CSV per model
OUTPUT_CSV = f"rolling_backtest_results_{model_name}.csv"

with open(os.path.join(model_dir, "training_info.json")) as f:
    model_info = json.load(f)
tickers = model_info["tickers"]
features = model_info["features"]
expected_cols = [f"{ticker}_{feature}" for ticker in tickers for feature in features]

# --- Hardcode window_size for long-term models ---
if model_name.startswith("long_term"):
    window_size = 60
else:
    window_size = 30
env_params["window_size"] = window_size

# Prepare rolling windows
def get_rolling_windows(start_year, end_year, train_years, test_years):
    windows = []
    for train_start in range(start_year, end_year - train_years - test_years + 2):
        train_end = train_start + train_years - 1
        test_start = train_end + 1
        test_end = test_start + test_years - 1
        if test_end > end_year:
            break
        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return windows


windows = get_rolling_windows(
    START_YEAR, END_YEAR, ROLLING_WINDOW_YEARS, TEST_WINDOW_YEARS
)

# Initialize model manager
model_manager = ModelManager()

results = []

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add this function to compute metrics for a static portfolio
def compute_portfolio_metrics(df, weights, tickers):
    close_cols = [f"{ticker}_close" for ticker in tickers]
    prices = df[close_cols]
    returns = prices.pct_change().fillna(0)
    # Portfolio returns: weighted sum
    port_returns = (returns * np.array([weights.get(t, 0) for t in tickers])).sum(
        axis=1
    )
    cum_returns = (1 + port_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
    ann_return = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    return {
        "Cumulative Return": total_return,
        "Max Drawdown": max_drawdown,
        "Annualized Sharpe Ratio": sharpe,
    }


def suitability_check(kpis, weights):
    checks = {}
    # Max drawdown < 20%
    checks["max_drawdown_ok"] = kpis["Max Drawdown"] > -0.20
    # No single stock > 50%
    checks["no_overconcentration"] = all(w <= 0.5 for w in weights.values())
    # At least 5 stocks allocated
    checks["min_diversification"] = sum(1 for w in weights.values() if w > 0.01) >= 5
    # Overall suitability: all checks pass
    checks["suitable"] = all(checks.values())
    return checks


for window in windows:
    logger.info(
        f"\n=== Rolling Window: Train {window['train_start']}-{window['train_end']} | Test {window['test_start']}-{window['test_end']} ==="
    )
    logger.info(f"   [DEBUG] Using window_size={env_params['window_size']} for environment creation")
    investment_profile = InvestmentProfile(
        age=user_profile.age,
        investment_amount=user_profile.investment_amount,
        time_horizon=user_profile.time_horizon,
        risk_tolerance=user_profile.risk_tolerance,
    )
    try:
        # PATCH: Always pass the correct window_size in env_params to ModelManager
        portfolio_rec = model_manager.get_portfolio_recommendation(
            strategy=model_name,
            profile=investment_profile,
            env_params=env_params,  # This now always has the correct window_size
            model_path=os.path.join(model_dir, "best_model.zip"),
        )
        if "error" in portfolio_rec:
            logger.error(f"   ❌ Error: {portfolio_rec['error']}")
            continue
        weights = portfolio_rec["recommended_weights"]
        # Load test window data using the model's tickers/features
        test_start_date = f"{window['test_start']}-01-01"
        test_end_date = f"{window['test_end']}-12-31"
        df_test = load_market_data_with_indicators(
            tickers_list=tickers,
            start_date=test_start_date,
            end_date=test_end_date,
            min_data_points=252,
        )
        if df_test is None or df_test.empty:
            logger.error(f"   ❌ No data for test window {test_start_date} to {test_end_date}")
            # --- Add diagnostics below ---
            if df_test is not None:
                logger.error(f"   [DIAG] df_test shape: {df_test.shape}")
                logger.error(f"   [DIAG] df_test columns: {df_test.columns.tolist()}")
                if not df_test.empty:
                    logger.error(f"   [DIAG] df_test index: {df_test.index.min()} to {df_test.index.max()}")
            continue
        # Filter to expected columns (from model_info)
        missing_cols = [col for col in expected_cols if col not in df_test.columns]
        for col in missing_cols:
            df_test[col] = 0.0
        df_test = df_test[expected_cols]
        logger.info(f"   [DEBUG] df_test shape after filling: {df_test.shape}, expected: (n_rows, {len(expected_cols)})")
        if len(missing_cols) > 0:
            logger.warning(f"   [DEBUG] Missing columns in test data (filled with zeros): {missing_cols}")
        if df_test.shape[1] != len(expected_cols):
            logger.error(f"   [ERROR] Test data shape mismatch: got {df_test.shape[1]} columns, expected {len(expected_cols)}. Missing: {missing_cols}")
        logger.info(f"   [DEBUG] Tickers in training: {tickers}")
        logger.info(f"   [DEBUG] Features in training: {features}")
        logger.info(f"   [DEBUG] Columns in test data: {list(df_test.columns)[:10]} ... {list(df_test.columns)[-10:]}")
        # Compute real KPIs using the model's tickers
        kpis = compute_portfolio_metrics(df_test, weights, tickers)
        suitability = suitability_check(kpis, weights)
        # PATCH: Convert all suitability values to native bools for JSON serialization
        suitability = {k: bool(v) for k, v in suitability.items()}
        user_profile_dict = {
            "age": user_profile.age,
            "income": user_profile.income,
            "investment_amount": user_profile.investment_amount,
            "risk_tolerance": user_profile.risk_tolerance,
            "goal": user_profile.goal,
        }
        explanation = asyncio.run(
            generate_investment_report(
                kpis=kpis,
                weights=weights,
                user_goal=user_profile.goal,
                user_profile=user_profile_dict,
            )
        )
        results.append(
            {
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "cumulative_return": kpis["Cumulative Return"],
                "max_drawdown": kpis["Max Drawdown"],
                "sharpe": kpis["Annualized Sharpe Ratio"],
                "weights": json.dumps(weights),
                "suitability": suitability["suitable"],
                "suitability_checks": json.dumps(suitability),
                "explanation": explanation,
            }
        )
    except Exception as e:
        logger.error(f"   ❌ Exception: {e}")
        continue

# Define stress test windows (add more as needed)
stress_windows = [
    {"name": "COVID Crash", "start": "2020-02-01", "end": "2020-05-31"},
    {"name": "2022 Bear Market", "start": "2022-01-01", "end": "2022-10-31"},
]

for stress in stress_windows:
    print(
        f"\n=== Stress Test: {stress['name']} ({stress['start']} to {stress['end']}) ==="
    )
    investment_profile = InvestmentProfile(
        age=user_profile.age,
        investment_amount=user_profile.investment_amount,
        time_horizon=user_profile.time_horizon,
        risk_tolerance=user_profile.risk_tolerance,
    )
    try:
        portfolio_rec = model_manager.get_portfolio_recommendation(
            strategy=model_name,
            profile=investment_profile,
            env_params=env_params,
            model_path=os.path.join(model_dir, "best_model.zip"),
        )
        if "error" in portfolio_rec:
            print(f"   ❌ Error: {portfolio_rec['error']}")
            continue
        weights = portfolio_rec["recommended_weights"]
        df_test = load_market_data_with_indicators(
            tickers_list=tickers,
            start_date=stress["start"],
            end_date=stress["end"],
            min_data_points=30,
        )
        if df_test is None or df_test.empty:
            print(
                f"   ❌ No data for stress window {stress['start']} to {stress['end']}"
            )
            continue
        for col in expected_cols:
            if col not in df_test.columns:
                df_test[col] = 0.0
        df_test = df_test[expected_cols]
        kpis = compute_portfolio_metrics(df_test, weights, tickers)
        suitability = suitability_check(kpis, weights)
        # PATCH: Convert all suitability values to native bools for JSON serialization
        suitability = {k: bool(v) for k, v in suitability.items()}
        user_profile_dict = {
            "age": user_profile.age,
            "income": user_profile.income,
            "investment_amount": user_profile.investment_amount,
            "risk_tolerance": user_profile.risk_tolerance,
            "goal": user_profile.goal,
        }
        explanation = asyncio.run(
            generate_investment_report(
                kpis=kpis,
                weights=weights,
                user_goal=user_profile.goal,
                user_profile=user_profile_dict,
            )
        )
        results.append(
            {
                "window_type": "stress",
                "window_name": stress["name"],
                "test_start": stress["start"],
                "test_end": stress["end"],
                "cumulative_return": kpis["Cumulative Return"],
                "max_drawdown": kpis["Max Drawdown"],
                "sharpe": kpis["Annualized Sharpe Ratio"],
                "weights": json.dumps(weights),
                "suitability": suitability["suitable"],
                "suitability_checks": json.dumps(suitability),
                "explanation": explanation,
            }
        )
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        continue

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nRolling backtest results saved to {OUTPUT_CSV}")
