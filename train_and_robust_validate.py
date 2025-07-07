#!/usr/bin/env python3
"""
Robust training script for AI Financial Advisor models.
- Uses a wider Optuna search space and more trials for hyperparameter tuning.
- After training each model, automatically runs rolling backtest validation.
- Prints and saves a summary of rolling backtest results for each model.

Usage:
    python train_and_robust_validate.py
"""
import os
import sys
import logging
from datetime import datetime
import optuna
import json
import subprocess
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from optuna.pruners import MedianPruner

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_market_data_from_db
from rl.portfolio_env_short_term import PortfolioEnvShortTerm
from rl.portfolio_env_long_term import PortfolioEnvLongTerm
from rl.custom_ppo import CustomPPO
import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OVERFIT_MODELS = [
    "short_term_moderate"
]

# === TEST MODE TOGGLE ===
TEST_MODE = False  # Set to True for quick dry runs, False for full training

class RobustModelTrainer:
    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_configs = {
            # ... (copy model_configs from train_all_models.py) ...
        }
        self._load_configurations()

    def _load_configurations(self):
        try:
            from config_short_term import (
                AGENT_TICKERS as ST_TICKERS,
                BENCHMARK_TICKER as ST_BENCHMARK,
                FEATURES_TO_USE_IN_MODEL as ST_FEATURES,
            )
            self.short_term_config = {
                "agent_tickers": ST_TICKERS,
                "benchmark_ticker": ST_BENCHMARK,
                "features_to_use": ST_FEATURES,
            }
        except ImportError:
            logger.warning("Short-term config not found, using defaults")
            self.short_term_config = {
                "agent_tickers": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "SPY"],
                "benchmark_ticker": "SPY",
                "features_to_use": [
                    "close",
                    "volume",
                    "rsi",
                    "macd",
                    "bb_upper",
                    "bb_lower",
                ],
            }
        try:
            from config_long_term import (
                AGENT_TICKERS as LT_TICKERS,
                BENCHMARK_TICKER as LT_BENCHMARK,
                FEATURES_TO_USE_IN_MODEL as LT_FEATURES,
            )
            self.long_term_config = {
                "agent_tickers": LT_TICKERS,
                "benchmark_ticker": LT_BENCHMARK,
                "features_to_use": LT_FEATURES,
            }
        except ImportError:
            logger.warning("Long-term config not found, using defaults")
            self.long_term_config = {
                "agent_tickers": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "SPY"],
                "benchmark_ticker": "SPY",
                "features_to_use": [
                    "close",
                    "volume",
                    "rsi",
                    "macd",
                    "bb_upper",
                    "bb_lower",
                ],
            }
        # Model configs (copy from train_all_models.py)
        self.model_configs = {
            "short_term_conservative": {
                "strategy": "short_term",
                "risk_tolerance": "conservative",
                "description": "Short-term conservative strategy",
                "env_params": {
                    "window_size": 30,
                    "rolling_volatility_window": 14,
                    "momentum_weight": 0.1,
                    "mean_reversion_weight": 0.4,
                    "volatility_target": 0.08,
                    "turnover_penalty_weight": 0.01,
                    "max_concentration_per_asset": 0.25,
                    "min_holding_period": 5,
                },
            },
            "short_term_moderate": {
                "strategy": "short_term",
                "risk_tolerance": "moderate",
                "description": "Short-term moderate strategy",
                "env_params": {
                    "window_size": 30,
                    "rolling_volatility_window": 14,
                    "momentum_weight": 0.3,
                    "mean_reversion_weight": 0.2,
                    "volatility_target": 0.15,
                    "turnover_penalty_weight": 0.005,
                    "max_concentration_per_asset": 0.35,
                    "min_holding_period": 3,
                },
            },
            "short_term_aggressive": {
                "strategy": "short_term",
                "risk_tolerance": "aggressive",
                "description": "Short-term aggressive strategy",
                "env_params": {
                    "window_size": 30,
                    "rolling_volatility_window": 14,
                    "momentum_weight": 0.5,
                    "mean_reversion_weight": 0.1,
                    "volatility_target": 0.25,
                    "turnover_penalty_weight": 0.002,
                    "max_concentration_per_asset": 0.5,
                    "min_holding_period": 1,
                },
            },
            "long_term_conservative": {
                "strategy": "long_term",
                "risk_tolerance": "conservative",
                "description": "Long-term conservative strategy",
                "env_params": {
                    "window_size": 60,
                    "rolling_volatility_window": 252,
                    "min_holding_period": 60,
                    "risk_parity_enabled": True,
                    "sector_rotation_enabled": False,
                    "max_concentration_per_asset": 0.20,
                    "volatility_target": 0.08,
                    "turnover_penalty_weight": 0.02,
                },
            },
            "long_term_moderate": {
                "strategy": "long_term",
                "risk_tolerance": "moderate",
                "description": "Long-term moderate strategy",
                "env_params": {
                    "window_size": 60,
                    "rolling_volatility_window": 252,
                    "min_holding_period": 30,
                    "risk_parity_enabled": True,
                    "sector_rotation_enabled": True,
                    "max_concentration_per_asset": 0.30,
                    "volatility_target": 0.12,
                    "turnover_penalty_weight": 0.01,
                },
            },
            "long_term_aggressive": {
                "strategy": "long_term",
                "risk_tolerance": "aggressive",
                "description": "Long-term aggressive strategy",
                "env_params": {
                    "window_size": 60,
                    "rolling_volatility_window": 252,
                    "min_holding_period": 15,
                    "risk_parity_enabled": False,
                    "sector_rotation_enabled": True,
                    "max_concentration_per_asset": 0.40,
                    "volatility_target": 0.18,
                    "turnover_penalty_weight": 0.005,
                },
            },
        }

    def _load_data(self, strategy: str) -> pd.DataFrame:
        # Use a consistent start date for all tickers with IPOs before 2010
        start_date = "2010-01-01"
        end_date = "2021-12-31"
        if strategy == "short_term":
            config = self.short_term_config
        else:
            config = self.long_term_config
        all_tickers = sorted(list(set(config["agent_tickers"] + [config["benchmark_ticker"]])))
        logger.info(f"[DATA] Loading data for {strategy} strategy...")
        logger.info(f"[DATA] Training date range: {start_date} to {end_date} (holdout: 2022-2024)")
        df_data = load_market_data_from_db(
            tickers_list=all_tickers,
            start_date=start_date,
            end_date=end_date,
            min_data_points=252 + 100,
            feature_columns=config["features_to_use"],
        )
        if not isinstance(df_data, pd.DataFrame) or df_data.empty:
            raise ValueError(f"Failed to load data for {strategy} strategy")
        logger.info(f"Loaded data shape: {df_data.shape}")
        # After df_data is loaded
        bad_cols = df_data.columns[df_data.isnull().all() | (df_data.nunique() <= 1)].tolist()
        if len(bad_cols) > 0:
            print("Dropping bad columns:", bad_cols)
            df_data = df_data.drop(columns=bad_cols)
        if df_data.isnull().values.any():
            print("Warning: NaNs remain in df_data after cleaning!")
        all_nan_cols = df_data.columns[df_data.isnull().all()].tolist()
        if len(all_nan_cols) > 0:
            print("All-NaN columns in df_data:", all_nan_cols)
            df_data = df_data.drop(columns=all_nan_cols)
        df_data = df_data.ffill().fillna(0)
        return df_data

    def _create_environment(self, model_name: str, df_data: pd.DataFrame) -> Tuple:
        config = self.model_configs[model_name]
        strategy = config["strategy"]
        if strategy == "short_term":
            ticker_config = self.short_term_config
            env_class = PortfolioEnvShortTerm
        else:
            ticker_config = self.long_term_config
            env_class = PortfolioEnvLongTerm
        agent_specific_cols = [
            f"{t}_{f}"
            for t in ticker_config["agent_tickers"]
            for f in ticker_config["features_to_use"]
        ]
        available_cols = [col for col in agent_specific_cols if col in df_data.columns]
        if not available_cols:
            raise ValueError(f"No available features found for {model_name}")
        if len(available_cols) == 1:
            df_env_data = df_data[[available_cols[0]]].copy()
        else:
            df_env_data = df_data[available_cols].copy()
        if not isinstance(df_env_data, pd.DataFrame):
            df_env_data = pd.DataFrame(df_env_data)
        env_params = {
            "initial_balance": 100000,
            "transaction_cost_pct": 0.001,
            **config["env_params"],
        }
        env = env_class(df_env_data, ticker_config["features_to_use"], **env_params)
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs)), "NaN in initial observation!"
        return env, ticker_config, env_params

    def _train_model(self, model_name: str, env, ticker_config: Dict, env_params: dict, df_data: pd.DataFrame) -> Tuple[str, str]:
        logger.info(f"Training {model_name} with robust Optuna search...")
        if TEST_MODE:
            logger.warning("[TEST MODE] Running with n_trials=2, total_timesteps=1000 for quick dry run.")
            n_trials = 2
            total_timesteps = 1000
        else:
            logger.info("[FULL TRAINING MODE] Running with n_trials=20, total_timesteps=100_000.")
            n_trials = 20
            total_timesteps = 100_000
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = f"{self.models_dir}/{model_name}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        def objective(trial, env_params=env_params, model_dir=model_dir):
            n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
            
            # Debug logging to see what values are suggested
            logger.info(f"[DEBUG] Trial suggested: n_steps={n_steps}, batch_size={batch_size}")
            logger.info(f"[DEBUG] Checking compatibility: {n_steps} % {batch_size} = {n_steps % batch_size}")
            
            if n_steps % batch_size != 0:
                logger.warning(f"[DEBUG] Pruning trial due to incompatible n_steps={n_steps} and batch_size={batch_size}")
                raise optuna.TrialPruned()
            
            learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-3, log=True)
            n_epochs = trial.suggest_int("n_epochs", 5, 20)
            gamma = trial.suggest_float("gamma", 0.93, 0.999)
            gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.99)
            clip_range = trial.suggest_float("clip_range", 0.05, 0.4)
            ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05)
            logger.info(f"[OPTUNA] Trial n_steps: {n_steps}, batch_size: {batch_size} (divisor)")
            model = CustomPPO(
                "MlpPolicy", env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                verbose=0,
                tensorboard_log=f"{model_dir}/tensorboard",
                device="cpu",
            )
            eval_interval = max(1, total_timesteps // 20)
            best_reward = -np.inf
            for step in range(0, total_timesteps, eval_interval):
                model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
                eval_env = env.__class__(env.df, ticker_config["features_to_use"], **env_params)
                obs, _ = eval_env.reset()
                total_reward = 0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                # Report intermediate value for pruning
                trial.report(total_reward, step)
                if trial.should_prune():
                    logger.info(f"[OPTUNA] Trial pruned at step {step} with reward {total_reward}")
                    raise optuna.TrialPruned()
                if total_reward > best_reward:
                    best_reward = total_reward
            return best_reward
        # Use MedianPruner for pruning - less aggressive settings
        pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=3)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(objective, n_trials=n_trials, timeout=3600 if not TEST_MODE else 60)
        best_params = study.best_params
        logger.info(f"Best parameters for {model_name}: {best_params}")
        logger.info(f"[SUMMARY] Training {model_name} on {df_data.index[0]} to {df_data.index[-1]} (holdout: 2022-2024)")
        logger.info(f"[SUMMARY] Best n_steps: {best_params.get('n_steps')}, batch_size: {best_params.get('batch_size')}")
        # Retrain final model with best params and full timesteps
        final_model = CustomPPO(
            "MlpPolicy", env, **best_params,
            verbose=1,
            tensorboard_log=f"{model_dir}/tensorboard",
            device="cpu",
        )
        final_model.learn(total_timesteps=total_timesteps)
        model_path = f"{model_dir}/best_model.zip"
        final_model.save(model_path)
        training_info = {
            "model_name": model_name,
            "strategy": self.model_configs[model_name]["strategy"],
            "risk_tolerance": self.model_configs[model_name]["risk_tolerance"],
            "description": self.model_configs[model_name]["description"],
            "best_params": best_params,
            "tickers": ticker_config["agent_tickers"],
            "features": ticker_config["features_to_use"],
            "training_date": timestamp,
            "total_timesteps": total_timesteps,
            "train_start_date": str(df_data.index[0]),
            "train_end_date": str(df_data.index[-1]),
            "holdout_start_date": "2022-01-01",
            "holdout_end_date": "2024-12-31",
        }
        with open(f"{model_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        logger.info(f"Model {model_name} saved to {model_path}")
        return model_path, model_dir

    def train_and_robust_validate_all(self):
        logger.info("Starting robust training and validation for all models...")
        for model_name in [
            "short_term_conservative",
            "short_term_moderate",
            "short_term_aggressive",
            "long_term_conservative",
            "long_term_moderate",
            "long_term_aggressive",
        ]:
            if model_name not in self.model_configs:
                logger.warning(f"Skipping unknown model: {model_name}")
                continue
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_name}")
                logger.info(f"{'='*60}")
                strategy = self.model_configs[model_name]["strategy"]
                df_data = self._load_data(strategy)
                env, ticker_config, env_params = self._create_environment(model_name, df_data)
                model_path, model_dir = self._train_model(model_name, env, ticker_config, env_params, df_data)
                logger.info(f"✅ Successfully trained {model_name}")
                # --- Run rolling backtest validation ---
                logger.info(f"Running rolling backtest for {model_name}...")
                result = subprocess.run([
                    sys.executable, "eval/eval_rolling_backtest.py",
                    "--model_dir", model_dir,
                    "--model_name", model_name
                ], capture_output=True, text=True)
                if result.stdout.strip():
                    with open(f"{model_dir}/rolling_backtest_output.txt", "w") as f:
                        f.write(result.stdout)
                    logger.info(f"Rolling backtest results saved to {model_dir}/rolling_backtest_output.txt")
                    print(result.stdout)
                else:
                    logger.error(f"Rolling backtest produced no output. STDERR: {result.stderr}")
                    with open(f"{model_dir}/rolling_backtest_output.txt", "w") as f:
                        f.write(result.stderr)
            except Exception as e:
                logger.error(f"❌ Error training or validating {model_name}: {e}")

if __name__ == "__main__":
    trainer = RobustModelTrainer()
    trainer.train_and_robust_validate_all() 