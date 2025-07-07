# app.py
from flask import Flask, render_template, jsonify, request
import os
import logging
import subprocess
import json
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import threading
import queue

# Import our custom modules
from src.utils import load_market_data_from_db
from llm.advisor import generate_investment_report
from rl.portfolio_env_short_term import PortfolioEnvShortTerm
from rl.portfolio_env_long_term import PortfolioEnvLongTerm

try:
    from stable_baselines3 import PPO
except ImportError:
    logging.error("Stable Baselines3 (PPO) not found. Please ensure it's installed.")
    PPO = None

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Path to the directory where plots and reports are saved
OUTPUT_DIR = os.path.join(os.getcwd(), 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

class InvestmentProfile:
    """Class to handle investment profile and model selection logic."""
    
    def __init__(self, age: int, investment_amount: float, time_horizon: str, risk_tolerance: str = "moderate"):
        self.age = age
        self.investment_amount = investment_amount
        self.time_horizon = time_horizon  # 'short_term' or 'long_term'
        self.risk_tolerance = risk_tolerance
        
    def get_recommended_strategy(self) -> Dict:
        """Determine recommended strategy based on profile."""
        if self.time_horizon == 'short_term':
            return {
                'strategy': 'short_term',
                'description': 'Active trading strategy for short-term gains',
                'risk_level': 'Higher',
                'expected_volatility': 'High',
                'recommended_frequency': 'Daily monitoring'
            }
        else:  # long_term
            return {
                'strategy': 'long_term',
                'description': 'Growth-focused strategy for long-term wealth building',
                'risk_level': 'Moderate to High',
                'expected_volatility': 'Medium',
                'recommended_frequency': 'Weekly monitoring'
            }
    
    def get_age_based_advice(self) -> Dict:
        """Provide age-specific investment advice."""
        if self.age < 25:
            return {
                'risk_capacity': 'High',
                'advice': 'You have time on your side! Consider more aggressive growth strategies.',
                'focus': 'Growth and learning about markets'
            }
        elif self.age < 35:
            return {
                'risk_capacity': 'High to Moderate',
                'advice': 'Great time to build wealth. Balance growth with some stability.',
                'focus': 'Growth with moderate risk management'
            }
        elif self.age < 50:
            return {
                'risk_capacity': 'Moderate',
                'advice': 'Consider diversifying and protecting your gains.',
                'focus': 'Balanced growth and preservation'
            }
        else:
            return {
                'risk_capacity': 'Moderate to Low',
                'advice': 'Focus on capital preservation while maintaining growth potential.',
                'focus': 'Stability with moderate growth'
            }

class ModelManager:
    """Manages model loading and prediction for different strategies."""
    
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
                FEATURES_TO_USE_IN_MODEL as ST_FEATURES
            )
            self.configs['short_term'] = {
                'agent_tickers': ST_TICKERS,
                'benchmark_ticker': ST_BENCHMARK,
                'features_to_use': ST_FEATURES
            }
        except ImportError:
            logging.warning("Short-term config not found")
            
        try:
            from config_long_term import (
                AGENT_TICKERS as LT_TICKERS,
                BENCHMARK_TICKER as LT_BENCHMARK,
                FEATURES_TO_USE_IN_MODEL as LT_FEATURES
            )
            self.configs['long_term'] = {
                'agent_tickers': LT_TICKERS,
                'benchmark_ticker': LT_BENCHMARK,
                'features_to_use': LT_FEATURES
            }
        except ImportError:
            logging.warning("Long-term config not found")
    
    def get_model_path(self, strategy: str) -> Optional[str]:
        """Get the most recent model path for a strategy, regardless of folder name pattern."""
        models_dir = "models"
        if not os.path.exists(models_dir):
            logging.warning(f"Models directory {models_dir} not found")
            return None
        # Look for any folder starting with the strategy name
        strategy_prefix = f"{strategy}_"
        strategy_dirs = [d for d in os.listdir(models_dir) if d.startswith(strategy_prefix)]
        if not strategy_dirs:
            logging.warning(f"No {strategy} models found in {models_dir}")
            return None
        # Sort by timestamp in folder name (assumes format: strategy_YYYYMMDD-HHMMSS)
        def extract_timestamp(d):
            parts = d.split("_")
            if len(parts) >= 3:
                return parts[-1]
            elif len(parts) == 2:
                return parts[1]
            return ""
        latest_dir = sorted(strategy_dirs, key=extract_timestamp)[-1]
        model_dir = os.path.join(models_dir, latest_dir)
        # Try to find the best model first, then final model
        model_files = ["best_model.zip", "final_model.zip"]
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                logging.info(f"Found {strategy} model: {model_path}")
                return model_path
        logging.warning(f"No model files found in {model_dir}")
        return None
    
    def load_model(self, strategy: str) -> 'Optional[Any]':
        """Load a model for the specified strategy."""
        if strategy in self.models:
            return self.models[strategy]
        model_path = self.get_model_path(strategy)
        if not model_path or not os.path.exists(model_path):
            logging.error(f"Model not found for strategy: {strategy}")
            return None
        try:
            if PPO is None:
                logging.error("Stable Baselines3 (PPO) is not installed.")
                return None
            model = PPO.load(model_path, device='cpu')
            self.models[strategy] = model
            logging.info(f"Successfully loaded {strategy} model from {model_path}")
            return model
        except Exception as e:
            logging.error(f"Failed to load {strategy} model: {e}")
            return None
    
    def get_environment(self, strategy: str, df: pd.DataFrame, env_params: Optional[dict] = None, model_path: str = ""):
        """Create the appropriate environment for a strategy, using model's training config for window_size/features."""
        # Always use the base config for the time horizon
        if strategy.startswith("long_term"):
            config = self.configs.get("long_term")
        elif strategy.startswith("short_term"):
            config = self.configs.get("short_term")
        else:
            config = self.configs.get(strategy)
        if not config:
            logging.error(f"No configuration found for strategy: {strategy}")
            return None
        # If model_path is provided, load training_config.json for window_size/features
        features_to_use = config["features_to_use"]
        window_size = None
        if model_path:
            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, "training_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    training_config = json.load(f)
                window_size = training_config.get("env_params", {}).get("window_size")
                features_to_use = training_config.get("ppo_params", {}).get("features_to_use", features_to_use)
                if not features_to_use:
                    features_to_use = training_config.get("features_to_use", features_to_use)
                logging.info(f"Loaded window_size from config: {window_size}")
        if window_size is None:
            window_size = env_params.get("window_size") if env_params and "window_size" in env_params else 30
        # Remove window_size from env_params to avoid conflict
        if env_params and "window_size" in env_params:
            env_params = dict(env_params)  # make a copy
            env_params.pop("window_size")
        # Prepare feature columns
        feature_columns = []
        for ticker in config["agent_tickers"]:
            for feature in features_to_use:
                feature_columns.append(f"{ticker}_{feature}")
        available_cols = [col for col in feature_columns if col in df.columns]
        df_subset = df[available_cols]
        env_kwargs = dict()
        if env_params:
            env_kwargs.update(env_params)
        # Always override with model's window_size/features
        env_kwargs["window_size"] = window_size
        logging.info(f"Creating environment with window_size={window_size}")
        # Choose environment class
        if not isinstance(df_subset, pd.DataFrame):
            logging.error("df_subset is not a DataFrame. Cannot create environment.")
            return None
        if strategy.startswith('short_term'):
            return PortfolioEnvShortTerm(df=df_subset, feature_columns_ordered=features_to_use, **env_kwargs)
        else:
            return PortfolioEnvLongTerm(df=df_subset, feature_columns_ordered=features_to_use, **env_kwargs)
    
    def _load_env_params_from_config(self, strategy: str) -> dict:
        """Load environment parameters from the training config file."""
        try:
            # Find the most recent model directory for this strategy
            models_dir = "models"
            strategy_prefix = f"{strategy}_"
            strategy_dirs = [d for d in os.listdir(models_dir) if d.startswith(strategy_prefix)]
            
            if not strategy_dirs:
                logging.warning(f"No {strategy} models found, using default parameters")
                return {}
            
            # Get the most recent model directory
            latest_dir = sorted(strategy_dirs)[-1]
            config_path = os.path.join(models_dir, latest_dir, "training_config.json")
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    env_params = config_data.get('env_params', {})
                    logging.info(f"Loaded {strategy} env params: {env_params}")
                    return env_params
            else:
                logging.warning(f"Config file not found at {config_path}, using default parameters")
                return {}
                
        except Exception as e:
            logging.warning(f"Failed to load env params from config: {e}, using default parameters")
            return {}
    
    def get_portfolio_recommendation(self, strategy: str, profile: InvestmentProfile, env_params: Optional[dict] = None, model_path: str = "") -> dict:
        """Get portfolio recommendation for a given strategy and profile, with optional custom env_params and model_path."""
        # Patch: always use the base config for the time horizon
        if strategy.startswith("long_term"):
            config = self.configs.get("long_term")
        elif strategy.startswith("short_term"):
            config = self.configs.get("short_term")
        else:
            config = self.configs.get(strategy)
        if config is None:
            return {"error": "Configuration not available"}
        model = self.load_model(strategy)
        if model is None:
            return {"error": "Model not available"}
        try:
            # Get current market data
            all_tickers = sorted(list(set(config["agent_tickers"] + [config["benchmark_ticker"]])))
            df_data = load_market_data_from_db(
                tickers_list=all_tickers,
                start_date="2020-01-01",
                end_date="2024-01-01",
                min_data_points=252 + 50,
                feature_columns=config["features_to_use"],
            )
            logging.info(f"Data shape for KPI calculation: {df_data.shape if isinstance(df_data, pd.DataFrame) else 'None'}")
            logging.info(f"Tickers used: {config['agent_tickers']}")
            if not isinstance(df_data, pd.DataFrame) or df_data.empty:
                return {"error": "Market data not available"}
            if env_params is None:
                env_params = {}
            if not model_path:
                model_path = self.get_model_path(strategy) or ""
            if not model_path or not os.path.exists(model_path):
                return {"error": "Model file not found"}
            env = self.get_environment(strategy, df_data, env_params=env_params, model_path=model_path)
            if env is None:
                return {"error": "Failed to create environment"}
            # Get model prediction
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.flatten()
            else:
                action = np.array(action).flatten()
            logging.info(f"Predicted action: {action}")
            action_sum = np.sum(action)
            if action_sum > 1e-6:
                predicted_weights = action / action_sum
            else:
                predicted_weights = np.full(len(config["agent_tickers"]), 1.0 / len(config["agent_tickers"]))
            logging.info(f"Predicted weights: {predicted_weights}")
            ticker_weights = list(zip(config["agent_tickers"], predicted_weights))
            ticker_weights.sort(key=lambda x: x[1], reverse=True)
            top_recommendations = ticker_weights[:8]
            top_weights_sum = sum(weight for _, weight in top_recommendations)
            if top_weights_sum > 1e-6:
                normalized_top_weights = [weight / top_weights_sum for _, weight in top_recommendations]
            else:
                normalized_top_weights = [1.0 / len(top_recommendations)] * len(top_recommendations)
            final_recommendations = {
                ticker: weight 
                for (ticker, _), weight in zip(top_recommendations, normalized_top_weights)
            }
            allocation_amounts = {
                ticker: weight * profile.investment_amount 
                for ticker, weight in final_recommendations.items()
            }
            sector_breakdown = self._get_sector_breakdown([ticker for ticker, _ in top_recommendations])
            result = {
                "strategy": strategy,
                "recommended_weights": final_recommendations,
                "allocation_amounts": allocation_amounts,
                "total_investment": profile.investment_amount,
                "top_recommendations": [ticker for ticker, _ in top_recommendations],
                "available_tickers": config["agent_tickers"],
                "benchmark": config["benchmark_ticker"],
                "sector_breakdown": sector_breakdown,
                "total_stocks_analyzed": len(config["agent_tickers"]),
                "recommendations_shown": len(top_recommendations)
            }
            converted = convert_numpy_types(result)
            if not isinstance(converted, dict):
                return {"result": converted}
            return converted
        except Exception as e:
            logging.error(f"Error getting portfolio recommendation: {e}")
            return {"error": f"Failed to generate recommendation: {str(e)}"}
    
    def _get_sector_breakdown(self, tickers):
        """Get sector breakdown for given tickers."""
        try:
            from config_extended_universe import SECTOR_MAPPING
            
            sector_counts = {}
            for ticker in tickers:
                for sector, sector_tickers in SECTOR_MAPPING.items():
                    if ticker in sector_tickers:
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                        break
                else:
                    # If ticker not found in any sector, categorize as "Other"
                    sector_counts["Other"] = sector_counts.get("Other", 0) + 1
            
            return sector_counts
        except ImportError:
            return {"Unknown": len(tickers)}

    def get_model_performance(self, strategy: str) -> dict:
        """Calculate actual performance metrics for a given strategy."""
        model = self.load_model(strategy)
        if not model:
            return {}
        
        config = self.configs.get(strategy)
        if not config:
            return {}
        
        try:
            # Get current market data
            all_tickers = sorted(list(set(config["agent_tickers"] + [config["benchmark_ticker"]])))
            df_data = load_market_data_from_db(
                tickers_list=all_tickers,
                start_date="2020-01-01",
                end_date="2024-01-01",
                min_data_points=252 + 50,
                feature_columns=config["features_to_use"],
            )
            
            if not isinstance(df_data, pd.DataFrame) or df_data.empty:
                return {}
            
            # Create environment
            env = self.get_environment(strategy, df_data)
            if not env:
                return {}
            
            # Get model prediction
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            
            # Use the model's predicted action as weights
            # Ensure action is properly normalized (sums to 1)
            if isinstance(action, np.ndarray):
                action = action.flatten()  # Ensure 1D array
            else:
                action = np.array(action).flatten()
            
            # Normalize weights to sum to 1
            action_sum = np.sum(action)
            if action_sum > 1e-6:
                predicted_weights = action / action_sum
            else:
                # Fallback to equal weights if action is all zeros
                predicted_weights = np.full(len(config['agent_tickers']), 1.0 / len(config['agent_tickers']))
            
            # Calculate performance metrics based on the predicted weights
            cumulative_return = 0.0
            max_drawdown = 0.0
            annualized_sharpe_ratio = 0.0
            
            # Calculate weighted portfolio performance
            portfolio_returns = pd.Series(0.0, index=df_data.index)
            
            for ticker, weight in zip(config['agent_tickers'], predicted_weights):
                close_col = f"{ticker}_close"
                if close_col in df_data.columns:
                    # Calculate daily returns for this ticker
                    ticker_returns = df_data[close_col].pct_change().fillna(0)
                    # Add weighted contribution to portfolio
                    portfolio_returns += weight * ticker_returns
            
            # Calculate cumulative return
            cumulative_return = (1 + portfolio_returns).prod() - 1
            
            # Calculate max drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            # Calculate annualized Sharpe ratio
            if len(portfolio_returns) > 1:
                annualized_return = portfolio_returns.mean() * 252
                annualized_volatility = portfolio_returns.std() * np.sqrt(252)
                sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            else:
                sharpe_ratio = 0
            
            return {
                "cumulative_return": cumulative_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio
            }
        except Exception as e:
            logging.error(f"Failed to calculate model performance: {e}")
            return {}

# Initialize global model manager
model_manager = ModelManager()

# Persistent LLM worker for Gemini API
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
                from llm.advisor import generate_investment_report
                result = loop.run_until_complete(generate_investment_report(*args, **kwargs))
            except Exception as e:
                result = f"LLM error: {e}"
            self.result_queue.put(result)

    def call(self, *args, **kwargs):
        self.task_queue.put((args, kwargs))
        return self.result_queue.get()

llm_worker = LLMWorker()

@app.route('/')
def home():
    """Renders the main dashboard page with user profile form."""
    return render_template('index.html')

@app.route('/api/analyze_portfolio', methods=['POST'])
def analyze_portfolio():
    """API endpoint to analyze portfolio based on user profile."""
    try:
        data = request.get_json()
        age = int(data.get('age', 25))
        investment_amount = float(data.get('investment_amount', 10000))
        time_horizon = data.get('time_horizon', 'long_term')
        risk_tolerance = data.get('risk_tolerance', 'moderate')
        
        # Create investment profile
        profile = InvestmentProfile(age, investment_amount, time_horizon, risk_tolerance)
        
        # Get strategy recommendation
        strategy_info = profile.get_recommended_strategy()
        age_advice = profile.get_age_based_advice()
        
        # Compose the model key using both time horizon and risk tolerance
        model_key = f"{time_horizon}_{risk_tolerance}"
        
        # Get portfolio recommendation using the correct model key
        portfolio_rec = model_manager.get_portfolio_recommendation(model_key, profile)
        
        if 'error' in portfolio_rec:
            return jsonify({"error": portfolio_rec['error']}), 400
        
        # --- PATCH: Compute KPIs using historical data and recommended weights ---
        all_tickers = sorted(list(set(portfolio_rec["available_tickers"] + [portfolio_rec["benchmark"]])))
        df_hist = load_market_data_from_db(
            tickers_list=all_tickers,
            start_date="2022-01-01",
            end_date="2024-01-01",
            min_data_points=252,
            feature_columns=["close"]
        )
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
        kpis = None
        if df_hist is not None and not df_hist.empty:
            kpis = compute_portfolio_metrics(df_hist, portfolio_rec['recommended_weights'], portfolio_rec["available_tickers"])
        else:
            kpis = {
                "Cumulative Return": 0.0,
                "Max Drawdown": 0.0,
                "Annualized Sharpe Ratio": 0.0
            }
        # --- END PATCH ---
        
        # Generate LLM report
        try:
            # Use the new KPIs for LLM
            try:
                llm_report = llm_worker.call(
                    kpis=kpis,
                    weights=portfolio_rec['recommended_weights'],
                    user_goal=f"{time_horizon.replace('_', ' ').title()} Investment"
                )
            except Exception as e:
                logging.warning(f"LLM report generation failed: {e}")
                llm_report = "AI analysis temporarily unavailable. Please check back later."
                
        except Exception as e:
            logging.warning(f"LLM report generation failed: {e}")
            llm_report = "AI analysis temporarily unavailable. Please check back later."
        
        return jsonify({
            "profile": {
                "age": age,
                "investment_amount": investment_amount,
                "time_horizon": time_horizon,
                "risk_tolerance": risk_tolerance
            },
            "strategy": strategy_info,
            "age_advice": age_advice,
            "portfolio": portfolio_rec,
            "llm_report": llm_report
        })
        
    except Exception as e:
        logging.error(f"Portfolio analysis failed: {e}")
        return jsonify({"error": "Analysis failed. Please try again."}), 500

@app.route('/api/get_available_models')
def get_available_models():
    """Get list of available models."""
    models_info = {}
    
    for strategy in ['short_term', 'long_term']:
        model_path = model_manager.get_model_path(strategy)
        models_info[strategy] = {
            "available": model_path is not None and os.path.exists(model_path),
            "path": model_path
        }
    
    return jsonify(models_info)

@app.route('/run_evaluation')
def run_evaluation():
    """API endpoint to trigger a new evaluation run."""
    logging.info("Received request to run evaluation...")
    try:
        process = subprocess.run(
            ['python', '-m', 'rl.evaluate_enhanced'],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        logging.info("Evaluation script completed successfully.")
        return jsonify({"status": "success", "message": "Evaluation completed successfully."})
    except subprocess.CalledProcessError as e:
        logging.error(f"Evaluation script failed with exit code {e.returncode}.")
        return jsonify({"status": "error", "message": "Evaluation script failed.", "details": e.stderr}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({"status": "error", "message": "An unexpected server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)

