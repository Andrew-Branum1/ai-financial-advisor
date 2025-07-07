#!/usr/bin/env python3
"""
Comprehensive training script for all 6 AI Financial Advisor models.
Trains models for different strategies and risk tolerances.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Tuple
import optuna
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from src.utils import load_market_data_from_db
from rl.portfolio_env_short_term import PortfolioEnvShortTerm
from rl.portfolio_env_long_term import PortfolioEnvLongTerm
from rl.custom_ppo import CustomPPO
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Comprehensive trainer for all model variants."""

    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

        # Define all model configurations
        self.model_configs = {
            # Short-term models
            "short_term_conservative": {
                "strategy": "short_term",
                "risk_tolerance": "conservative",
                "description": "Short-term conservative strategy",
                "env_params": {
                    "window_size": 30,
                    "rolling_volatility_window": 14,
                    "momentum_weight": 0.2,
                    "mean_reversion_weight": 0.3,
                    "volatility_target": 0.10,  # Lower volatility target
                    "turnover_penalty_weight": 0.01,  # Higher turnover penalty
                    "max_concentration_per_asset": 0.25,  # More diversified
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
                    "momentum_weight": 0.4,
                    "mean_reversion_weight": 0.1,
                    "volatility_target": 0.25,  # Higher volatility target
                    "turnover_penalty_weight": 0.002,  # Lower turnover penalty
                    "max_concentration_per_asset": 0.5,  # Allow higher concentration
                    "min_holding_period": 1,
                },
            },
            # Long-term models
            "long_term_conservative": {
                "strategy": "long_term",
                "risk_tolerance": "conservative",
                "description": "Long-term conservative strategy",
                "env_params": {
                    "window_size": 60,
                    "rolling_volatility_window": 252,
                    "min_holding_period": 60,  # Longer holding periods
                    "risk_parity_enabled": True,
                    "sector_rotation_enabled": False,  # Less active
                    "max_concentration_per_asset": 0.20,  # Very diversified
                    "volatility_target": 0.08,  # Low volatility target
                    "turnover_penalty_weight": 0.02,  # High turnover penalty
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
                    "min_holding_period": 15,  # Shorter holding periods
                    "risk_parity_enabled": False,  # More active management
                    "sector_rotation_enabled": True,
                    "max_concentration_per_asset": 0.40,  # Allow higher concentration
                    "volatility_target": 0.18,  # Higher volatility target
                    "turnover_penalty_weight": 0.005,  # Lower turnover penalty
                },
            },
        }

        # Load configurations
        self._load_configurations()

    def _load_configurations(self):
        """Load ticker and feature configurations."""
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

    def _load_data(self, strategy: str) -> pd.DataFrame:
        """Load market data for training."""
        if strategy == "short_term":
            config = self.short_term_config
            start_date = "2018-01-01"  # Longer history, excludes 2022 volatility
            end_date = "2021-12-31"  # Stop before 2022 crash
        else:  # long_term
            config = self.long_term_config
            start_date = "2015-01-01"  # Much longer history for long-term
            end_date = "2021-12-31"  # Stop before 2022 crash

        all_tickers = sorted(
            list(set(config["agent_tickers"] + [config["benchmark_ticker"]]))
        )

        logger.info(f"Loading data for {strategy} strategy...")
        df_data = load_market_data_from_db(
            tickers_list=all_tickers,
            start_date=start_date,
            end_date=end_date,
            min_data_points=252 + 100,  # At least 1 year + buffer
            feature_columns=config["features_to_use"],
        )

        if not isinstance(df_data, pd.DataFrame) or df_data.empty:
            raise ValueError(f"Failed to load data for {strategy} strategy")

        logger.info(f"Loaded data shape: {df_data.shape}")
        return df_data

    def _create_environment(self, model_name: str, df_data: pd.DataFrame) -> Tuple:
        """Create training environment for a specific model."""
        config = self.model_configs[model_name]
        strategy = config["strategy"]

        if strategy == "short_term":
            ticker_config = self.short_term_config
            env_class = PortfolioEnvShortTerm
        else:
            ticker_config = self.long_term_config
            env_class = PortfolioEnvLongTerm

        # Prepare feature columns
        agent_specific_cols = [
            f"{t}_{f}"
            for t in ticker_config["agent_tickers"]
            for f in ticker_config["features_to_use"]
        ]
        available_cols = [col for col in agent_specific_cols if col in df_data.columns]

        if not available_cols:
            raise ValueError(f"No available features found for {model_name}")

        # Ensure DataFrame type for environment data
        if len(available_cols) == 1:
            df_env_data = df_data[[available_cols[0]]].copy()
        else:
            df_env_data = df_data[available_cols].copy()

        if not isinstance(df_env_data, pd.DataFrame):
            df_env_data = pd.DataFrame(df_env_data)

        # Create environment with model-specific parameters
        env_params = {
            "initial_balance": 100000,
            "transaction_cost_pct": 0.001,
            **config["env_params"],
        }

        # Create environment for training
        env = env_class(df_env_data, ticker_config["features_to_use"], **env_params)

        return env, ticker_config, env_params

    def _train_model(
        self, model_name: str, env, ticker_config: Dict, env_params: dict
    ) -> str:
        """Train a single model with hyperparameter optimization."""
        logger.info(f"Training {model_name}...")

        # Create model directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = f"{self.models_dir}/{model_name}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)

        # Run hyperparameter optimization
        def objective(trial, env_params=env_params, model_dir=model_dir):
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            n_steps = trial.suggest_categorical(
                "n_steps", [128, 256, 512, 1024]
            )  # Increased
            batch_size = trial.suggest_categorical(
                "batch_size", [64, 128, 256]
            )  # Increased
            n_epochs = trial.suggest_int("n_epochs", 5, 15)  # Increased
            gamma = trial.suggest_float(
                "gamma", 0.95, 0.999
            )  # Higher gamma for long-term
            gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
            clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
            ent_coef = trial.suggest_float(
                "ent_coef", 0.001, 0.02
            )  # Increased exploration
            model = CustomPPO(
                "MlpPolicy",
                env,
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
            model.learn(total_timesteps=100000)  # Increased from 50k to 100k
            eval_env = env.__class__(
                env.df, ticker_config["features_to_use"], **env_params
            )
            obs, _ = eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                done = terminated or truncated
            return total_reward

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective, n_trials=30, timeout=7200
        )  # Increased trials and timeout

        # Train final model with best parameters
        best_params = study.best_params
        logger.info(f"Best parameters for {model_name}: {best_params}")

        final_model = CustomPPO(
            "MlpPolicy",
            env,
            **best_params,
            verbose=1,
            tensorboard_log=f"{model_dir}/tensorboard",
            device="cpu",
        )

        # Train final model
        final_model.learn(total_timesteps=200000)  # More training for final model

        # Save the model
        model_path = f"{model_dir}/best_model.zip"
        final_model.save(model_path)

        # Save training info
        training_info = {
            "model_name": model_name,
            "strategy": self.model_configs[model_name]["strategy"],
            "risk_tolerance": self.model_configs[model_name]["risk_tolerance"],
            "description": self.model_configs[model_name]["description"],
            "best_params": best_params,
            "tickers": ticker_config["agent_tickers"],
            "features": ticker_config["features_to_use"],
            "training_date": timestamp,
            "total_timesteps": 200000,
        }

        import json

        with open(f"{model_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

        logger.info(f"Model {model_name} saved to {model_path}")
        return model_path

    def train_all_models(self):
        """Train all 6 models."""
        logger.info("Starting training for all 6 models...")

        trained_models = {}

        for model_name in self.model_configs.keys():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_name}")
                logger.info(f"{'='*60}")

                # Load data
                strategy = self.model_configs[model_name]["strategy"]
                df_data = self._load_data(strategy)

                # Create environment
                env, ticker_config, env_params = self._create_environment(
                    model_name, df_data
                )

                # Train model
                model_path = self._train_model(
                    model_name, env, ticker_config, env_params
                )
                trained_models[model_name] = model_path

                logger.info(f"✅ Successfully trained {model_name}")

            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                continue

        # Create a summary
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")

        for model_name, model_path in trained_models.items():
            logger.info(f"✅ {model_name}: {model_path}")

        logger.info(
            f"\nTrained {len(trained_models)} out of {len(self.model_configs)} models"
        )

        return trained_models


def main():
    """Main training function."""
    print("🤖 AI Financial Advisor - Model Training")
    print("=" * 60)
    print("Choose training option:")
    print("1. Train all 6 models (takes several hours)")
    print("2. Train only long-term models (recommended)")
    print("3. Train only short-term models")
    print("=" * 60)
    print("Note: Your short-term model is already performing well!")
    print("Long-term models need improvement (2.21% return, -32.78% drawdown)")
    print("=" * 60)

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        print("\nTraining all 6 models...")
        models_to_train = list(ModelTrainer().model_configs.keys())
    elif choice == "2":
        print("\nTraining only long-term models...")
        models_to_train = [
            "long_term_conservative",
            "long_term_moderate",
            "long_term_aggressive",
        ]
    elif choice == "3":
        print("\nTraining only short-term models...")
        models_to_train = [
            "short_term_conservative",
            "short_term_moderate",
            "short_term_aggressive",
        ]
    else:
        print("Invalid choice. Exiting.")
        return

    response = (
        input(
            f"Do you want to proceed with training {len(models_to_train)} models? (y/n): "
        )
        .lower()
        .strip()
    )
    if response != "y":
        print("Training cancelled.")
        return

    try:
        trainer = ModelTrainer()

        # Filter to only train selected models
        original_configs = trainer.model_configs.copy()
        trainer.model_configs = {
            k: v for k, v in original_configs.items() if k in models_to_train
        }

        trainer.train_all_models()

        print("\n🎉 Training completed!")
        print("You can now use the web interface to get personalized investment plans.")
        print("Run: python app.py")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.error(f"Training failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
