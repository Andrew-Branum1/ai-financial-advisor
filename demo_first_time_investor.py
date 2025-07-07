#!/usr/bin/env python3
"""
Demo script for the First-Time Investor AI Financial Advisor.
This script demonstrates the new user-friendly interface with profile-based model selection.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
import time
import threading
import queue

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from app import InvestmentProfile, ModelManager
from llm.advisor import generate_investment_report
from src.utils import UserProfile, map_user_profile_to_env_params, load_market_data_with_indicators
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Persistent LLM worker
class LLMWorker:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while True:
            args, kwargs = self.task_queue.get()
            if args is None:  # Poison pill for shutdown
                break
            try:
                result = loop.run_until_complete(generate_investment_report(*args, **kwargs))
            except Exception as e:
                result = f"LLM error: {e}"
            self.result_queue.put(result)

    def call(self, *args, **kwargs):
        self.task_queue.put((args, kwargs))
        return self.result_queue.get()

llm_worker = LLMWorker()

def safe_generate_investment_report(*args, **kwargs):
    return llm_worker.call(*args, **kwargs)

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title} ".center(60, "="))
    print("="*60)

def demo_investment_profiles():
    """Demonstrate different investment profiles and their recommendations."""
    
    print_header("FIRST-TIME INVESTOR AI ADVISOR DEMO")
    
    # Sample user profiles
    profiles = [
        UserProfile(
            name="Young Professional (Sarah, 24)",
            age=24,
            income=5000,
            investment_amount=5000,
            time_horizon="long_term",
            risk_tolerance="aggressive",
            goal="growth"
        ),
        UserProfile(
            name="Mid-Career Professional (Mike, 35)",
            age=35,
            income=25000,
            investment_amount=25000,
            time_horizon="long_term",
            risk_tolerance="moderate",
            goal="balanced"
        ),
        UserProfile(
            name="Active Trader (Alex, 28)",
            age=28,
            income=15000,
            investment_amount=15000,
            time_horizon="short_term",
            risk_tolerance="aggressive",
            goal="growth"
        ),
        UserProfile(
            name="Conservative Investor (Linda, 45)",
            age=45,
            income=50000,
            investment_amount=50000,
            time_horizon="long_term",
            risk_tolerance="conservative",
            goal="preservation"
        )
    ]
    
    # Show mapping from user profile to RL environment parameters
    print_section("User Profile to RL Environment Parameter Mapping")
    for profile in profiles:
        env_params = map_user_profile_to_env_params(profile)
        print(f"{profile.name}:")
        for k, v in env_params.items():
            print(f"  {k}: {v}")
        print()
    
    # Initialize model manager
    print_section("Initializing AI Models")
    model_manager = ModelManager()
    
    # Check available models
    print("Checking available models...")
    for strategy in ['short_term', 'long_term']:
        model_path = model_manager.get_model_path(strategy)
        status = "✅ Available" if model_path and os.path.exists(model_path) else "❌ Not Found"
        print(f"  {strategy.replace('_', ' ').title()}: {status}")
        if model_path:
            print(f"    Path: {model_path}")
    
    print_section("Demo Profiles Analysis")
    
    for profile in profiles:
        print(f"\n📋 Analyzing Profile: {profile.name}")
        print("-" * 60)
        print(f"Age: {profile.age}")
        print(f"Income: ${profile.income:,}")
        print(f"Investment Amount: ${profile.investment_amount:,}")
        print(f"Time Horizon: {profile.time_horizon.replace('_', ' ').title()}")
        print(f"Risk Tolerance: {profile.risk_tolerance.title()}")
        print(f"Goal: {profile.goal.title()}")

        # Map user profile to RL environment parameters
        env_params = map_user_profile_to_env_params(profile)
        print(f"\n🔧 RL Environment Parameters:")
        for k, v in env_params.items():
            print(f"  {k}: {v}")

        # Create a temporary InvestmentProfile for compatibility
        investment_profile = InvestmentProfile(
            age=profile.age,
            investment_amount=profile.investment_amount,
            time_horizon=profile.time_horizon,
            risk_tolerance=profile.risk_tolerance
        )
        try:
            # Select model key based on both time horizon and risk tolerance
            if profile.time_horizon in ["long_term", "short_term"]:
                model_key = f"{profile.time_horizon}_{profile.risk_tolerance}"
            else:
                model_key = profile.time_horizon
            model_path = model_manager.get_model_path(model_key) or ""
            # Patch: always use the base config for the time horizon
            if model_key.startswith("long_term"):
                config = model_manager.configs["long_term"]
            elif model_key.startswith("short_term"):
                config = model_manager.configs["short_term"]
            else:
                config = model_manager.configs[model_key]
            portfolio_rec = model_manager.get_portfolio_recommendation(
                strategy=model_key,  # Use the combined key
                profile=investment_profile,
                env_params=env_params,
                model_path=model_path
            )
            if 'error' in portfolio_rec:
                print(f"   ❌ Error: {portfolio_rec['error']}")
            else:
                weights = portfolio_rec['recommended_weights']
                amounts = portfolio_rec['allocation_amounts']
                print(f"   Total Investment: ${portfolio_rec['total_investment']:,}")
                print(f"   Benchmark: {portfolio_rec['benchmark']}")
                print(f"   Recommended Allocation:")
                for ticker, weight in weights.items():
                    amount = amounts[ticker]
                    percentage = weight * 100
                    print(f"     {ticker}: {percentage:.1f}% (${amount:,.0f})")
                # Compute and print KPIs
                all_tickers = sorted(list(set(config["agent_tickers"] + [config["benchmark_ticker"]])))
                df_hist = load_market_data_with_indicators(
                    tickers_list=all_tickers,
                    start_date="2022-01-01",
                    end_date="2024-01-01",
                    min_data_points=252
                )
                kpis = None
                if df_hist is not None and not df_hist.empty:
                    kpis = compute_portfolio_metrics(df_hist, weights, config["agent_tickers"])
                    print("   Portfolio KPIs (2022-2024):")
                    for k, v in kpis.items():
                        if "Sharpe" in k:
                            print(f"     {k}: {v:.2f}")
                        else:
                            print(f"     {k}: {v:.2%}")
                else:
                    print("   (No historical data available for KPI calculation.)")
                # Generate the explanation (async) only if KPIs are available
                if kpis is not None:
                    user_profile_dict = {
                        "age": profile.age,
                        "income": profile.income,
                        "investment_amount": profile.investment_amount,
                        "risk_tolerance": profile.risk_tolerance,
                        "goal": profile.goal
                    }
                    try:
                        report = safe_generate_investment_report(
                            kpis=kpis,
                            weights=weights,
                            user_goal=profile.goal,
                            user_profile=user_profile_dict
                        )
                        print("\n🤖 AI Investment Analysis:")
                        print("-" * 60)
                        print(report)
                    except Exception as e:
                        print(f"\n🤖 AI Investment Analysis:")
                        print("-" * 60)
                        print(f"Analysis temporarily unavailable: {e}")
                        print("Portfolio recommendations are still valid and ready to use.")
                    time.sleep(0.5)
            with open("explanations_log.txt", "a") as f:
                f.write(f"User: {user_profile_dict}\n")
                f.write(f"KPIs: {kpis}\n")
                f.write(f"Weights: {weights}\n")
                f.write(f"Explanation:\n{report}\n")
                f.write("="*80 + "\n")
        except Exception as e:
            print(f"   ❌ Error generating recommendation: {e}")
        print("\n" + "="*60)

async def demo_llm_analysis():
    """Demonstrate LLM-powered investment analysis."""
    
    print_section("AI-Powered Investment Analysis Demo")
    
    # Sample portfolio data
    sample_kpis = {
        "Investment Amount": 25000,
        "Age": 30,
        "Time Horizon": "long_term",
        "Risk Tolerance": "moderate"
    }
    
    sample_weights = {
        "AAPL": 0.25,
        "MSFT": 0.20,
        "GOOGL": 0.15,
        "TSLA": 0.10,
        "NVDA": 0.15,
        "SPY": 0.15
    }
    
    print("Generating AI investment analysis...")
    print("This may take a moment as we connect to the AI service...")
    
    try:
        report = await generate_investment_report(
            kpis=sample_kpis,
            weights=sample_weights,
            user_goal="Long-Term Investment"
        )
        
        print("\n🤖 AI Investment Analysis:")
        print("-" * 60)
        print(report)
        
    except Exception as e:
        print(f"❌ Error generating AI analysis: {e}")
        print("This might be due to API rate limits or connectivity issues.")

def demo_web_interface():
    """Provide instructions for using the web interface."""
    
    print_section("Web Interface Instructions")
    
    print("🌐 To use the interactive web interface:")
    print("1. Start the Flask application:")
    print("   python app.py")
    print("\n2. Open your web browser and go to:")
    print("   http://localhost:5001")
    print("\n3. Fill out the investment profile form:")
    print("   - Enter your age")
    print("   - Specify investment amount")
    print("   - Choose time horizon (short-term vs long-term)")
    print("   - Select risk tolerance")
    print("\n4. Click 'Get My Personalized Investment Plan'")
    print("\n5. Review your personalized recommendations!")
    
    print("\n✨ Features of the web interface:")
    print("   • User-friendly form with helpful tooltips")
    print("   • Real-time validation")
    print("   • Beautiful, responsive design")
    print("   • Age-specific investment advice")
    print("   • Visual portfolio allocation charts")
    print("   • AI-powered investment analysis")
    print("   • Step-by-step next steps guidance")

def compute_portfolio_metrics(df, weights, tickers):
    close_cols = [f"{ticker}_close" for ticker in tickers]
    prices = df[close_cols]
    returns = prices.pct_change().fillna(0)
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

def main():
    """Main demo function."""
    
    print_header("AI FINANCIAL ADVISOR - FIRST-TIME INVESTOR DEMO")
    
    print("This demo showcases our AI-powered financial advisor designed specifically")
    print("for first-time investors. The system provides personalized investment")
    print("recommendations based on your age, investment amount, time horizon,")
    print("and risk tolerance.")
    
    try:
        # Demo 1: Investment Profiles
        demo_investment_profiles()
        
        # Demo 2: LLM Analysis (if available)
        print("\n" + "="*80)
        print(" Would you like to see the AI-powered analysis demo? (y/n) ".center(80, "="))
        print("="*80)
        response = input("Enter 'y' to continue with AI analysis demo: ").lower().strip()
        if response == 'y':
            # Use the first profile as an example
            profiles = [
                UserProfile(
                    name="Young Professional (Sarah, 24)",
                    age=24,
                    income=5000,
                    investment_amount=5000,
                    time_horizon="long_term",
                    risk_tolerance="aggressive",
                    goal="growth"
                ),
                UserProfile(
                    name="Mid-Career Professional (Mike, 35)",
                    age=35,
                    income=25000,
                    investment_amount=25000,
                    time_horizon="long_term",
                    risk_tolerance="moderate",
                    goal="balanced"
                ),
                UserProfile(
                    name="Active Trader (Alex, 28)",
                    age=28,
                    income=15000,
                    investment_amount=15000,
                    time_horizon="short_term",
                    risk_tolerance="aggressive",
                    goal="growth"
                ),
                UserProfile(
                    name="Conservative Investor (Linda, 45)",
                    age=45,
                    income=50000,
                    investment_amount=50000,
                    time_horizon="long_term",
                    risk_tolerance="conservative",
                    goal="preservation"
                )
            ]
            profile = profiles[0]
            sample_kpis = {
                "Cumulative Return": 0.25,
                "Max Drawdown": -0.08,
                "Annualized Sharpe Ratio": 1.5
            }
            sample_weights = {"AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.15, "TSLA": 0.10, "NVDA": 0.15, "SPY": 0.15}
            user_profile_dict = {
                "age": profile.age,
                "income": profile.income,
                "investment_amount": profile.investment_amount,
                "risk_tolerance": profile.risk_tolerance,
                "goal": profile.goal
            }
            report = safe_generate_investment_report(
                kpis=sample_kpis,
                weights=sample_weights,
                user_goal=profile.goal,
                user_profile=user_profile_dict
            )
        
        # Demo 3: Web Interface Instructions
        demo_web_interface()
        
        print_header("DEMO COMPLETE")
        print("🎉 Thank you for exploring our AI Financial Advisor!")
        print("\nThe system is designed to make investing accessible and understandable")
        print("for first-time investors, providing personalized guidance based on")
        print("your unique financial situation and goals.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)

if __name__ == "__main__":
    main() 