# rl/train_universal_optuna.py
import logging
import os
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Import our custom classes
from src.utils import load_market_data_for_universal_env
from rl.universal_portfolio_env import UniversalPortfolioEnv
from rl.attention_policy import AttentionPolicy, AttentionFeaturesExtractor
from rl.custom_ppo import CustomPPO

# --- NEW Self-Contained Pruning Callback ---
# This class replaces the problematic import from sb3-contrib
class ManualOptunaPruningCallback(BaseCallback):
    """A custom callback for Optuna pruning."""
    def __init__(self, trial: optuna.Trial, eval_env: UniversalPortfolioEnv):
        super().__init__(verbose=0)
        self.trial = trial
        self.eval_env = eval_env
        # It's better to wrap the eval_env in a VecEnv for consistency
        self.eval_env_vec = DummyVecEnv([lambda: self.eval_env])

    def _on_step(self) -> bool:
        # The EvalCallback runs first and saves the latest score in the logger
        if "eval/mean_reward" in self.logger.name_to_value:
            mean_reward = self.logger.name_to_value["eval/mean_reward"]
            
            self.trial.report(mean_reward, self.num_timesteps)

            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned(f"Trial pruned at step {self.num_timesteps}")
        
        return True

# --- NEW Custom Evaluation Callback ---
# This class fixes the bug you found. It evaluates our agent correctly.
class CustomEvalCallback(EvalCallback):
    """
    Custom evaluation callback that correctly handles our tuple-based action space.
    """
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # We call our own evaluation logic instead of the generic one
            self.run_custom_evaluation()
        return True

    def run_custom_evaluation(self):
        # --- This is a simplified evaluation loop that works with our custom action ---
        total_rewards = 0
        total_steps = 0
        
        obs = self.eval_env.reset()[0]
        done = False
        while not done:
            # Predict the action tuple
            action_tuple, _ = self.model.predict(obs, deterministic=self.deterministic)
            
            # Pass the tuple directly to the environment's step function
            obs, reward, terminated, truncated, info = self.eval_env.step(action_tuple)
            done = terminated or truncated
            total_rewards += reward
            total_steps += 1
        
        mean_reward = total_rewards / total_steps
        self.logger.record("eval/mean_reward", mean_reward)
        print(f"Eval @ step {self.num_timesteps}: mean_reward={mean_reward:.2f}")

        if self.best_model_save_path is not None:
            # (Optional) Save the best model logic
            pass
        return True

# --- Configuration (remains the same) ---
TICKERS = ['MSFT', 'NVDA'] 
FEATURES_TO_USE = ['close', 'close_vs_sma_50', 'mfi', 'bollinger_width', 'obv', 'atr']
FEATURES_DIM = 64
TOP_K_STOCKS = 2
LOG_DIR = "logs_optuna"
MODEL_DIR = "models_optuna"
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2021-12-31"
EVAL_START_DATE = "2022-01-01"
EVAL_END_DATE = "2022-06-30"
N_STEPS_PER_TRIAL = 75_000 

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def objective(trial: optuna.Trial) -> float:
    # --- Refined Hyperparameter Search Space ---
    # Narrow the learning rate to a more stable range
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True) 
    
    # Use a larger, more stable rollout buffer size
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096]) 
    
    # Keep gamma high for long-term rewards
    gamma = trial.suggest_float("gamma", 0.98, 0.999, log=True) 
    
    # A slightly higher entropy can help prevent getting stuck
    ent_coef = trial.suggest_float("ent_coef", 1e-7, 0.05, log=True) 
    
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2]) # 0.3 can be unstable
    
    # A larger batch size can lead to more stable updates
    batch_size = trial.suggest_categorical("batch_size", [64, 128])


    # --- Environment setup (remains the same) ---
    train_df = load_market_data_for_universal_env(tickers_list=TICKERS, feature_columns=FEATURES_TO_USE, start_date="2018-01-01", end_date=TRAIN_END_DATE)
    train_env = UniversalPortfolioEnv(df=train_df[train_df.index >= TRAIN_START_DATE], feature_columns=FEATURES_TO_USE, top_k_stocks=TOP_K_STOCKS)
    
    eval_df = load_market_data_for_universal_env(tickers_list=TICKERS, feature_columns=FEATURES_TO_USE, start_date="2018-01-01", end_date=EVAL_END_DATE)
    eval_env = UniversalPortfolioEnv(df=eval_df[eval_df.index >= EVAL_START_DATE], feature_columns=FEATURES_TO_USE, top_k_stocks=TOP_K_STOCKS)
    
    # --- Use our new CustomEvalCallback ---
    eval_callback = CustomEvalCallback(eval_env, eval_freq=5000, deterministic=True)
    pruning_callback = ManualOptunaPruningCallback(trial, eval_env) # This will now work
    callback_list = CallbackList([eval_callback, pruning_callback])

    model = CustomPPO(
        AttentionPolicy,
        train_env,
        policy_kwargs={"features_extractor_class": AttentionFeaturesExtractor, "features_extractor_kwargs": {"features_dim": FEATURES_DIM}},
        learning_rate=learning_rate, n_steps=n_steps, gamma=gamma, ent_coef=ent_coef,
        clip_range=clip_range, batch_size=batch_size, max_grad_norm=0.5, verbose=0
    )
    
    final_portfolio_value = -1.0
    try:
        model.learn(total_timesteps=N_STEPS_PER_TRIAL, callback=callback_list)
        
        # --- Final evaluation after full trial ---
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
        
        final_portfolio_value = eval_env.portfolio_value
        logging.info(f"--- Trial {trial.number} Finished. Final Portfolio Value: ${final_portfolio_value:.2f} ---")

    except optuna.exceptions.TrialPruned as e:
        logging.info(f"Trial {trial.number} pruned successfully.")
        raise
    except Exception as e:
        logging.warning(f"Trial {trial.number} failed with error {e}. Pruning.")
        raise optuna.exceptions.TrialPruned()

    return final_portfolio_value

if __name__ == "__main__":
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30000)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    
    try:
        study.optimize(objective, n_trials=20, show_progress_bar=True)
    except KeyboardInterrupt:
        print("Interrupted by user")

    print("\n" + "="*50)
    print(" Optuna Search Finished ".center(50, "="))
    print("="*50)
    print("Number of finished trials: ", len(study.trials))
    
    try:
        trial = study.best_trial
        print(f"  Value (Final Portfolio Value): ${trial.value:.2f}")
        print("  Best Hyperparameters: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    except ValueError:
        print("No successful trials completed.")