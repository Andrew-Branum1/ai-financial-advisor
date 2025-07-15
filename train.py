import os
import sys
import json
from datetime import datetime
from typing import Dict
import optuna
import pandas as pd
import numpy as np
import gymnasium as gym
import sqlite3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rl.portfolio_env_short_term import PortfolioEnvShortTerm
from rl.portfolio_env_long_term import PortfolioEnvLongTerm
from rl.custom_ppo import CustomPPO
import config

def load_market_data_from_db(tickers_list, start_date, end_date, min_data_points, feature_columns):
    """
    Loads and preprocesses market data from the local SQLite database.
    """
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'market_data.db')
    
    try:
        with sqlite3.connect(db_path) as conn:
            query = f'SELECT * FROM features_market_data WHERE "Date" >= ? AND "Date" <= ? ORDER BY "Date" ASC'
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
    except Exception as e:
        print(f"Error querying database: {e}")
        return None

    df.rename(columns={'Date': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Drop tickers with insufficient data for the given period
    for ticker in tickers_list:
        close_col = f'{ticker}_close'
        if close_col in df and df[close_col].count() < min_data_points:
            print(f"Warning: Ticker {ticker} has insufficient data and will be dropped.")
            cols_to_drop = [col for col in df.columns if col.startswith(f'{ticker}_')]
            df.drop(columns=cols_to_drop, inplace=True)
            
    return df

class ModelTrainer:
    """
    Handles the end-to-end process of training and validating RL models.
    """
    def __init__(self):
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

        self.model_configs = config.MODEL_CONFIGS
        self.all_tickers = config.ALL_TICKERS
        self.all_features = config.ALL_FEATURES

        # Hardcoded training parameters
        self.optuna_trials = 30
        self.optuna_timesteps = 10000
        self.training_timesteps = 150000

    def _load_data(self) -> pd.DataFrame:
        """Loads and prepares the full dataset for training."""
        print("Loading market data...")
        df = load_market_data_from_db(
            tickers_list=self.all_tickers,
            start_date="2007-01-01",
            end_date="2021-12-31",
            min_data_points=252 * 2,  # Approx 2 years of trading days
            feature_columns=self.all_features,
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Failed to load market data.")
            
        # Clean and preprocess the data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.dropna(how='all', axis=1, inplace=True)
        
        print(f"Data loaded. Shape: {df.shape}")
        return df

    def _create_environment(self, model_name: str, df: pd.DataFrame, env_params: dict) -> gym.Env:
        """Creates the appropriate RL environment for a given model."""
        model_cfg = self.model_configs[model_name]
        env_class = PortfolioEnvShortTerm if model_cfg['env_class'] == 'PortfolioEnvShortTerm' else PortfolioEnvLongTerm
        
        features_for_observation = model_cfg['features_to_use']
        # Ensure 'close' and 'daily_return' are always available for calculations
        required_features = ['close', 'daily_return']
        all_necessary_features = sorted(list(set(features_for_observation + required_features)))

        # Construct column names like 'AAPL_close', 'MSFT_rsi', etc.
        agent_specific_cols = [f"{t}_{f}" for t in self.all_tickers for f in all_necessary_features]
        available_cols = [col for col in agent_specific_cols if col in df.columns]
        
        if not available_cols:
            raise ValueError(f"No available features found for model {model_name}")

        df_env_data = df[available_cols].copy()
        
        # Set base environment parameters and merge with optimized ones
        full_env_params = {"initial_balance": 100_000, "transaction_cost_pct": 0.001, **env_params}
        return env_class(df_env_data, features_for_observation, **full_env_params)

    def _run_hyperparameter_optimization(self, model_name: str, train_df: pd.DataFrame) -> Dict:
        """Uses Optuna to find the best hyperparameters for a model."""
        def objective(trial: optuna.Trial) -> float:
            try:
                base_env_params = self.model_configs[model_name]['env_params']
                
                # Define search space for environment parameters
                env_params = {
                    "window_size": trial.suggest_int("window_size", 20, 80, step=10),
                    "max_concentration_per_asset": trial.suggest_float("max_concentration_per_asset", 0.2, 0.6)
                }
                # Adjust penalty based on model's risk profile
                if 'conservative' in model_name:
                    env_params["turnover_penalty_weight"] = trial.suggest_float("turnover_penalty_weight", 0.015, 0.03)
                elif 'balanced' in model_name:
                    env_params["turnover_penalty_weight"] = trial.suggest_float("turnover_penalty_weight", 0.008, 0.015)
                else: # aggressive
                    env_params["turnover_penalty_weight"] = trial.suggest_float("turnover_penalty_weight", 0.001, 0.008)

                # Define search space for PPO agent parameters
                ppo_params = {
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                    "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
                    "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                    "n_epochs": trial.suggest_int("n_epochs", 5, 15),
                    "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                    "ent_coef": trial.suggest_float("ent_coef", 0.01, 0.05),
                }
                # Prune trials where batch size is larger than the number of steps
                if ppo_params['n_steps'] < ppo_params['batch_size']:
                    raise optuna.TrialPruned()

                env = self._create_environment(model_name, train_df, {**base_env_params, **env_params})
                model = CustomPPO("MlpPolicy", env, verbose=0, device="cpu", **ppo_params)
                model.learn(total_timesteps=self.optuna_timesteps)

                # Evaluate the trial
                obs, _ = env.reset()
                done, total_reward = False, 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                return total_reward
            except Exception as e:
                print(f"Optuna trial failed for {model_name}: {e}")
                return -1e9

        print(f"Starting hyperparameter search for {model_name}...")
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=self.optuna_trials)
        
        print(f"Optimization complete. Best value for {model_name}: {study.best_value:.2f}")
        return study.best_params

    def train_single_model(self, model_name: str, full_df: pd.DataFrame):
        """Trains, optimizes, and saves a single model with early stopping."""
        print(f"\n{'='*60}\n--- Starting process for model: {model_name} ---\n{'='*60}")
        
        # Split data to prevent leakage into the hyperparameter search
        train_len = int(len(full_df) * 0.8)
        train_df = full_df.iloc[:train_len]
        eval_df = full_df.iloc[train_len:]

        try:
            # Run hyperparameter search on the training data only
            best_params = self._run_hyperparameter_optimization(model_name, train_df)
            print(f"Best hyperparameters found: {json.dumps(best_params, indent=2)}")

            final_env_params = {**self.model_configs[model_name]['env_params'], **best_params}
            
            # Create final environments with the clean data splits
            final_train_env = self._create_environment(model_name, train_df, final_env_params)
            final_eval_env = Monitor(self._create_environment(model_name, eval_df, final_env_params))
            
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_dir = os.path.join(self.models_dir, f"{model_name}_{timestamp}")
            os.makedirs(model_dir, exist_ok=True)
            
            # Set up callback for early stopping and saving the best model
            eval_callback = EvalCallback(
                final_eval_env,
                best_model_save_path=model_dir,
                log_path=model_dir,
                eval_freq=5000,
                deterministic=True,
                render=False,
                n_eval_episodes=3
            )

            # Separate PPO params from environment params
            final_ppo_params = {k: v for k, v in best_params.items() if k not in final_env_params}
            final_model = CustomPPO("MlpPolicy", final_train_env, verbose=1, device="cpu", **final_ppo_params)

            print(f"Training final model for {model_name} with early stopping...")
            final_model.learn(total_timesteps=self.training_timesteps, callback=eval_callback)

            # Rename the best model to a standard name
            best_model_path = os.path.join(model_dir, 'best_model.zip')
            if os.path.exists(best_model_path):
                 os.rename(best_model_path, os.path.join(model_dir, 'model.zip'))
            else: 
                 final_model.save(os.path.join(model_dir, "model.zip"))

            # Save training metadata
            training_info = {
                "model_name": model_name,
                "description": self.model_configs[model_name].get('description', 'N/A'),
                "training_date": timestamp,
                "total_timesteps": "N/A (Early Stopping)",
                "best_hyperparameters": best_params,
                "features_used": self.model_configs[model_name]['features_to_use'],
                "train_start_date": str(train_df.index.min().date()),
                "train_end_date": str(train_df.index.max().date()),
            }
            with open(os.path.join(model_dir, "training_info.json"), "w") as f:
                json.dump(training_info, f, indent=4)
                
            print(f"Successfully trained and saved model for {model_name} to {model_dir}")
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")

    def run_training_pipeline(self):
        """Main function to run the entire pipeline."""
        full_df = self._load_data()
        
        #target_models = list(self.model_configs.keys())
        target_models = [
            "short_term_conservative",
            "short_term_aggressive",
            "long_term_aggressive"
        ]

        print(f"Starting training for the following models: {target_models}")
        for model_name in target_models:
            self.train_single_model(model_name, full_df)
            
        print("\nAll selected training sessions are complete.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training_pipeline()