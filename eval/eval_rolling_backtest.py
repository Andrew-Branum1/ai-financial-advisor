import os
import sys
import pandas as pd
import numpy as np
import json
import asyncio
from datetime import datetime, timedelta
from src.utils import UserProfile, map_user_profile_to_env_params, load_market_data_with_indicators
from app import ModelManager, InvestmentProfile
from llm.advisor import generate_investment_report

# --- Parameters ---
ROLLING_WINDOW_YEARS = 2
TEST_WINDOW_YEARS = 1
START_YEAR = 2018
END_YEAR = 2024
OUTPUT_CSV = "rolling_backtest_results.csv"

# Example user profile (customize as needed)
user_profile = UserProfile(
    name="Demo User",
    age=30,
    income=60000,
    investment_amount=10000,
    time_horizon="long_term",
    risk_tolerance="moderate",
    goal="growth"
)

# Map user profile to RL environment parameters
env_params = map_user_profile_to_env_params(user_profile)

# Prepare rolling windows
def get_rolling_windows(start_year, end_year, train_years, test_years):
    windows = []
    for train_start in range(start_year, end_year - train_years - test_years + 2):
        train_end = train_start + train_years - 1
        test_start = train_end + 1
        test_end = test_start + test_years - 1
        if test_end > end_year:
            break
        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end
        })
    return windows

windows = get_rolling_windows(START_YEAR, END_YEAR, ROLLING_WINDOW_YEARS, TEST_WINDOW_YEARS)

# Initialize model manager
model_manager = ModelManager()

results = []

# Add this function to compute metrics for a static portfolio
def compute_portfolio_metrics(df, weights, tickers):
    close_cols = [f"{ticker}_close" for ticker in tickers]
    prices = df[close_cols]
    returns = prices.pct_change().fillna(0)
    # Portfolio returns: weighted sum
    port_returns = (returns * np.array([weights.get(t, 0) for t in tickers])).sum(axis=1)
    cum_returns = (1 + port_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
    ann_return = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    return {
        "Cumulative Return": total_return,
        "Max Drawdown": max_drawdown,
        "Annualized Sharpe Ratio": sharpe
    }

def suitability_check(kpis, weights):
    checks = {}
    # Max drawdown < 20%
    checks['max_drawdown_ok'] = kpis['Max Drawdown'] > -0.20
    # No single stock > 50%
    checks['no_overconcentration'] = all(w <= 0.5 for w in weights.values())
    # At least 5 stocks allocated
    checks['min_diversification'] = sum(1 for w in weights.values() if w > 0.01) >= 5
    # Overall suitability: all checks pass
    checks['suitable'] = all(checks.values())
    return checks

for window in windows:
    print(f"\n=== Rolling Window: Train {window['train_start']}-{window['train_end']} | Test {window['test_start']}-{window['test_end']} ===")
    # For demo, we use the latest trained model (in practice, retrain for each window)
    investment_profile = InvestmentProfile(
        age=user_profile.age,
        investment_amount=user_profile.investment_amount,
        time_horizon=user_profile.time_horizon,
        risk_tolerance=user_profile.risk_tolerance
    )
    # Get portfolio recommendation for the test window
    try:
        portfolio_rec = model_manager.get_portfolio_recommendation(
            strategy=user_profile.time_horizon,
            profile=investment_profile,
            env_params=env_params
        )
        if 'error' in portfolio_rec:
            print(f"   ❌ Error: {portfolio_rec['error']}")
            continue
        weights = portfolio_rec['recommended_weights']
        # Load test window data
        test_start_date = f"{window['test_start']}-01-01"
        test_end_date = f"{window['test_end']}-12-31"
        config = model_manager.configs[user_profile.time_horizon]
        all_tickers = sorted(list(set(config["agent_tickers"] + [config["benchmark_ticker"]])))
        df_test = load_market_data_with_indicators(
            tickers_list=all_tickers,
            start_date=test_start_date,
            end_date=test_end_date,
            min_data_points=252
        )
        if df_test is None or df_test.empty:
            print(f"   ❌ No data for test window {test_start_date} to {test_end_date}")
            continue
        # Compute real KPIs
        kpis = compute_portfolio_metrics(df_test, weights, config["agent_tickers"])
        suitability = suitability_check(kpis, weights)
        user_profile_dict = {
            "age": user_profile.age,
            "income": user_profile.income,
            "investment_amount": user_profile.investment_amount,
            "risk_tolerance": user_profile.risk_tolerance,
            "goal": user_profile.goal
        }
        # Generate LLM explanation
        explanation = asyncio.run(generate_investment_report(
            kpis=kpis,
            weights=weights,
            user_goal=user_profile.goal,
            user_profile=user_profile_dict
        ))
        # Log results
        results.append({
            "train_start": window['train_start'],
            "train_end": window['train_end'],
            "test_start": window['test_start'],
            "test_end": window['test_end'],
            "cumulative_return": kpis["Cumulative Return"],
            "max_drawdown": kpis["Max Drawdown"],
            "sharpe": kpis["Annualized Sharpe Ratio"],
            "weights": json.dumps(weights),
            "suitability": suitability['suitable'],
            "suitability_checks": json.dumps(suitability),
            "explanation": explanation
        })
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        continue

# Define stress test windows (add more as needed)
stress_windows = [
    {"name": "COVID Crash", "start": "2020-02-01", "end": "2020-05-31"},
    {"name": "2022 Bear Market", "start": "2022-01-01", "end": "2022-10-31"}
]

for stress in stress_windows:
    print(f"\n=== Stress Test: {stress['name']} ({stress['start']} to {stress['end']}) ===")
    investment_profile = InvestmentProfile(
        age=user_profile.age,
        investment_amount=user_profile.investment_amount,
        time_horizon=user_profile.time_horizon,
        risk_tolerance=user_profile.risk_tolerance
    )
    try:
        portfolio_rec = model_manager.get_portfolio_recommendation(
            strategy=user_profile.time_horizon,
            profile=investment_profile,
            env_params=env_params
        )
        if 'error' in portfolio_rec:
            print(f"   ❌ Error: {portfolio_rec['error']}")
            continue
        weights = portfolio_rec['recommended_weights']
        config = model_manager.configs[user_profile.time_horizon]
        all_tickers = sorted(list(set(config["agent_tickers"] + [config["benchmark_ticker"]])))
        df_test = load_market_data_with_indicators(
            tickers_list=all_tickers,
            start_date=stress['start'],
            end_date=stress['end'],
            min_data_points=30
        )
        if df_test is None or df_test.empty:
            print(f"   ❌ No data for stress window {stress['start']} to {stress['end']}")
            continue
        kpis = compute_portfolio_metrics(df_test, weights, config["agent_tickers"])
        suitability = suitability_check(kpis, weights)
        user_profile_dict = {
            "age": user_profile.age,
            "income": user_profile.income,
            "investment_amount": user_profile.investment_amount,
            "risk_tolerance": user_profile.risk_tolerance,
            "goal": user_profile.goal
        }
        explanation = asyncio.run(generate_investment_report(
            kpis=kpis,
            weights=weights,
            user_goal=user_profile.goal,
            user_profile=user_profile_dict
        ))
        results.append({
            "window_type": "stress",
            "window_name": stress['name'],
            "test_start": stress['start'],
            "test_end": stress['end'],
            "cumulative_return": kpis["Cumulative Return"],
            "max_drawdown": kpis["Max Drawdown"],
            "sharpe": kpis["Annualized Sharpe Ratio"],
            "weights": json.dumps(weights),
            "suitability": suitability['suitable'],
            "suitability_checks": json.dumps(suitability),
            "explanation": explanation
        })
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        continue

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nRolling backtest results saved to {OUTPUT_CSV}") 