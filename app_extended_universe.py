#!/usr/bin/env python3
"""
Enhanced AI Financial Advisor with Extended Universe
Supports a larger universe of stocks while showing only top recommendations to users.
"""

import sys
import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass
from flask import Flask, render_template, request, jsonify
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_market_data_from_db
from config_extended_universe import (
    get_ticker_universe,
    get_top_recommendations,
    get_sector_breakdown,
    get_risk_profile,
    MODEL_CONFIGS,
    SECTOR_MAPPING,
    RISK_LEVELS,
)

# Import LLM advisor
try:
    from llm.advisor import generate_investment_report

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("LLM advisor not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def convert_numpy_types(obj):
    """Convert NumPy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@dataclass
class InvestmentProfile:
    """Investment profile for a user."""

    age: int
    investment_amount: float
    strategy: str
    risk_tolerance: str
    universe_size: str = "medium"
    selection_strategy: str = "diversified"


class ExtendedUniverseModelManager:
    """Manages models for the extended universe of stocks."""

    def __init__(self):
        self.models = {}
        self.configs = {}
        self._load_configurations()

    def _load_configurations(self):
        """Load different strategy configurations."""
        try:
            from config_short_term import (
                AGENT_TICKERS as ST_TICKERS,
                BENCHMARK_TICKER as ST_BENCHMARK,
                FEATURES_TO_USE_IN_MODEL as ST_FEATURES,
            )

            self.configs["short_term"] = {
                "agent_tickers": ST_TICKERS,
                "benchmark_ticker": ST_BENCHMARK,
                "features_to_use": ST_FEATURES,
            }
        except ImportError:
            logging.warning("Short-term config not found")

        try:
            from config_long_term import (
                AGENT_TICKERS as LT_TICKERS,
                BENCHMARK_TICKER as LT_BENCHMARK,
                FEATURES_TO_USE_IN_MODEL as LT_FEATURES,
            )

            self.configs["long_term"] = {
                "agent_tickers": LT_TICKERS,
                "benchmark_ticker": LT_BENCHMARK,
                "features_to_use": LT_FEATURES,
            }
        except ImportError:
            logging.warning("Long-term config not found")

    def load_model(self, strategy: str):
        """Load a trained model for a strategy."""
        if strategy in self.models:
            return self.models[strategy]

        try:
            from stable_baselines3 import PPO
            import os

            # Look for model files
            model_dir = f"models/{strategy}_final_*"
            import glob

            model_paths = glob.glob(model_dir)

            if not model_paths:
                logger.error(f"No model found for strategy: {strategy}")
                return None

            # Use the most recent model
            latest_model_path = max(model_paths, key=os.path.getctime)
            best_model_path = os.path.join(latest_model_path, "best_model.zip")

            if os.path.exists(best_model_path):
                logger.info(f"Found {strategy} model: {best_model_path}")
                model = PPO.load(best_model_path)
                self.models[strategy] = model
                return model
            else:
                logger.error(f"Best model not found in {latest_model_path}")
                return None

        except Exception as e:
            logger.error(f"Error loading model for {strategy}: {e}")
            return None

    def get_environment(self, strategy: str, df: pd.DataFrame, tickers: List[str]):
        """Create the appropriate environment for a strategy with custom tickers."""
        config = self.configs.get(strategy)
        if not config:
            logging.error(f"No configuration found for strategy: {strategy}")
            return None

        # Prepare feature columns for the custom tickers
        feature_columns = []
        for ticker in tickers:
            for feature in config["features_to_use"]:
                feature_columns.append(f"{ticker}_{feature}")

        # Filter available columns
        available_cols = [col for col in feature_columns if col in df.columns]
        if not available_cols:
            logging.error(f"No available features found for {strategy}")
            return None

        # Always ensure available_cols is a list of strings
        if len(available_cols) == 1:
            df_subset = df[[available_cols[0]]]  # double brackets for DataFrame
        else:
            df_subset = df[available_cols]
        if not isinstance(df_subset, pd.DataFrame):
            df_subset = pd.DataFrame(df_subset)

        # Load environment parameters from training config
        env_params = self._load_env_params(strategy)

        # Create environment based on strategy
        if strategy == "short_term":
            from rl.portfolio_env_short_term import PortfolioEnvShortTerm

            env = PortfolioEnvShortTerm(
                df=df_subset,
                feature_columns_ordered=config["features_to_use"],
                initial_balance=100000,
                transaction_cost_pct=0.001,
                **env_params,
            )
        else:  # long_term
            from rl.portfolio_env_long_term import PortfolioEnvLongTerm

            env = PortfolioEnvLongTerm(
                df=df_subset,
                feature_columns_ordered=config["features_to_use"],
                initial_balance=100000,
                transaction_cost_pct=0.001,
                **env_params,
            )

        return env

    def _load_env_params(self, strategy: str) -> Dict:
        """Load environment parameters from training configuration."""
        try:
            import glob
            import os

            model_dir = f"models/{strategy}_final_*"
            model_paths = glob.glob(model_dir)

            if not model_paths:
                return {}

            latest_model_path = max(model_paths, key=os.path.getctime)
            config_path = os.path.join(latest_model_path, "training_config.json")

            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                return config.get("env_params", {})
            else:
                return {}

        except Exception as e:
            logger.warning(f"Could not load env params for {strategy}: {e}")
            return {}

    def get_portfolio_recommendation(self, profile: InvestmentProfile) -> Any:
        """Get portfolio recommendation for the extended universe."""
        model = self.load_model(profile.strategy)
        if not model:
            return {"error": "Model not available"}

        config = self.configs.get(profile.strategy)
        if not config:
            return {"error": "Configuration not available"}

        try:
            # Get ticker universe based on profile settings
            tickers = get_ticker_universe(
                profile.universe_size, profile.selection_strategy
            )

            # Add benchmark ticker
            all_tickers = sorted(list(set(tickers + [config["benchmark_ticker"]])))

            # Get market data
            df_data = load_market_data_from_db(
                tickers_list=all_tickers,
                start_date="2020-01-01",
                end_date="2024-01-01",
                min_data_points=252 + 50,
                feature_columns=config["features_to_use"],
            )

            if not isinstance(df_data, pd.DataFrame) or df_data.empty:
                return {"error": "Market data not available"}

            # Create environment with extended universe
            env = self.get_environment(profile.strategy, df_data, tickers)
            if not env:
                return {"error": "Failed to create environment"}

            # Get model prediction
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)

            # Normalize action to weights
            if isinstance(action, np.ndarray):
                action = action.flatten()
            else:
                action = np.array(action).flatten()

            action_sum = np.sum(action)
            if action_sum > 1e-6:
                predicted_weights = action / action_sum
            else:
                predicted_weights = np.full(len(tickers), 1.0 / len(tickers))

            # Create weights dictionary
            weights_dict = dict(zip(tickers, predicted_weights))

            # Get top recommendations
            top_recommendations = get_top_recommendations(weights_dict, top_n=5)

            # Get sector and risk breakdowns
            sector_breakdown = get_sector_breakdown(top_recommendations)
            risk_profile = get_risk_profile(top_recommendations)

            # Calculate allocation amounts for top recommendations
            top_tickers = [rec[0] for rec in top_recommendations]
            top_weights = [
                rec[2] / 100 for rec in top_recommendations
            ]  # Convert percentage to decimal

            allocation_amounts = {
                ticker: weight * profile.investment_amount
                for ticker, weight in zip(top_tickers, top_weights)
            }

            result = {
                "strategy": profile.strategy,
                "universe_size": profile.universe_size,
                "selection_strategy": profile.selection_strategy,
                "total_universe_size": len(tickers),
                "top_recommendations": [
                    {
                        "ticker": rec[0],
                        "weight": rec[1],
                        "percentage": rec[2],
                        "allocation": allocation_amounts[rec[0]],
                    }
                    for rec in top_recommendations
                ],
                "sector_breakdown": sector_breakdown,
                "risk_profile": risk_profile,
                "total_investment": profile.investment_amount,
                "available_tickers": tickers,
                "benchmark": config["benchmark_ticker"],
            }

            # Convert NumPy types to JSON-serializable types
            result = convert_numpy_types(result)

            return result

        except Exception as e:
            logging.error(f"Error getting portfolio recommendation: {e}")
            return {"error": f"Failed to generate recommendation: {str(e)}"}


# Initialize model manager
model_manager = ExtendedUniverseModelManager()

# Initialize LLM advisor if available
if LLM_AVAILABLE:
    llm_advisor = generate_investment_report
else:
    llm_advisor = None


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index_extended.html")


@app.route("/api/analyze_portfolio", methods=["POST"])
def analyze_portfolio():
    """Analyze portfolio with extended universe."""
    try:
        data = request.get_json()

        # Create investment profile
        profile = InvestmentProfile(
            age=int(data.get("age", 30)),
            investment_amount=float(data.get("investment_amount", 10000)),
            strategy=data.get("strategy", "short_term"),
            risk_tolerance=data.get("risk_tolerance", "moderate"),
            universe_size=data.get("universe_size", "medium"),
            selection_strategy=data.get("selection_strategy", "diversified"),
        )

        # Get portfolio recommendation
        recommendation = model_manager.get_portfolio_recommendation(profile)

        if "error" in recommendation:
            return jsonify({"error": recommendation["error"]}), 400

        # Generate LLM analysis if available
        if llm_advisor:
            try:
                # Prepare data for LLM
                top_stocks = [
                    rec["ticker"] for rec in recommendation["top_recommendations"]
                ]
                allocations = [
                    rec["percentage"] for rec in recommendation["top_recommendations"]
                ]

                # Create weights dictionary for LLM
                weights_dict = {
                    ticker: allocation / 100
                    for ticker, allocation in zip(top_stocks, allocations)
                }

                # Create KPIs for LLM
                kpis = {
                    "Investment Amount": profile.investment_amount,
                    "Age": profile.age,
                    "Strategy": profile.strategy,
                    "Risk Tolerance": profile.risk_tolerance,
                }

                # Generate LLM report asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    llm_analysis = loop.run_until_complete(
                        llm_advisor(
                            kpis=kpis,
                            weights=weights_dict,
                            user_goal=f"{profile.strategy.replace('_', ' ').title()} Investment",
                        )
                    )
                finally:
                    loop.close()

                recommendation["llm_analysis"] = llm_analysis

            except Exception as e:
                logger.error(f"Error generating LLM analysis: {e}")
                recommendation["llm_analysis"] = "AI analysis temporarily unavailable."
        else:
            recommendation["llm_analysis"] = "AI analysis not available."

        return jsonify(recommendation)

    except Exception as e:
        logger.error(f"Error in analyze_portfolio: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/universe_info")
def get_universe_info():
    """Get information about available universes."""
    try:
        universe_info = {
            "model_configs": MODEL_CONFIGS,
            "sectors": list(SECTOR_MAPPING.keys()),
            "risk_levels": list(RISK_LEVELS.keys()),
            "total_stocks": len(get_ticker_universe("full", "diversified")),
        }
        return jsonify(universe_info)
    except Exception as e:
        logger.error(f"Error getting universe info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sector_stocks/<sector>")
def get_sector_stocks(sector):
    """Get stocks in a specific sector."""
    try:
        if sector in SECTOR_MAPPING:
            stocks = SECTOR_MAPPING[sector]
            return jsonify({"sector": sector, "stocks": stocks, "count": len(stocks)})
        else:
            return jsonify({"error": f"Sector '{sector}' not found"}), 404
    except Exception as e:
        logger.error(f"Error getting sector stocks: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/risk_stocks/<risk_level>")
def get_risk_stocks(risk_level):
    """Get stocks for a specific risk level."""
    try:
        if risk_level in RISK_LEVELS:
            stocks = RISK_LEVELS[risk_level]
            return jsonify(
                {"risk_level": risk_level, "stocks": stocks, "count": len(stocks)}
            )
        else:
            return jsonify({"error": f"Risk level '{risk_level}' not found"}), 404
    except Exception as e:
        logger.error(f"Error getting risk stocks: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting Enhanced AI Financial Advisor with Extended Universe...")
    logger.info("Available universe sizes: small, medium, large, full")
    logger.info(
        "Available selection strategies: diversified, momentum, value, balanced"
    )

    app.run(debug=True, host="0.0.0.0", port=5000)
