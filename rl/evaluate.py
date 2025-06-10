# rl/evaluate.py
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import asyncio
from llm.advisor import generate_investment_report

# Absolute imports from the project root perspective
from rl.portfolio_env import PortfolioEnv
from src.utils import load_market_data_from_db
# from llm.advisor import generate_report_async # Can be uncommented when ready
from config import (
    AGENT_TICKERS,
    BENCHMARK_TICKER,
    FEATURES_TO_USE_IN_MODEL,
    ENV_PARAMS
)

try:
    from stable_baselines3 import PPO
except ImportError:
    logging.error("Stable Baselines3 (PPO) not found. Please ensure it's installed.")
    PPO = None

# --- Configuration Dictionary ---
# IMPORTANT: This path must point to the model you want to evaluate.
# You might make this a command-line argument for easier use.
MODEL_PATH = "models/best_PPO_Portfolio_Final_20250610-093858/best_model.zip"
PLOTS_DIR = "plots"
START_DATE = "2008-01-01"
END_DATE = "2009-12-31"


class PortfolioEvaluator:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.eval_env = None
        self.df_full_eval_data = None
        self.results = {}
        
        os.makedirs(self.config["plots_dir"], exist_ok=True)
        model_dir = os.path.dirname(self.config["model_path"])
        self.base_plot_name = os.path.basename(model_dir) if "best_model.zip" in self.config["model_path"] else os.path.basename(self.config["model_path"]).replace(".zip", "")

    def _load_data(self):
        logging.info("Loading evaluation data...")
        all_tickers = sorted(list(set(self.config["agent_tickers"] + [self.config["benchmark_ticker"]])))
        self.df_full_eval_data = load_market_data_from_db(
            tickers_list=all_tickers,
            start_date=self.config["start_date"],
            end_date=self.config["end_date"],
            min_data_points=self.config["env_params"]["window_size"] + 50,
            feature_columns=self.config["features_to_use"],
        )
        if self.df_full_eval_data.empty: raise ValueError("Failed to load evaluation data.")
        logging.info(f"Full evaluation data loaded. Shape: {self.df_full_eval_data.shape}")

    def _setup_environment_and_model(self):
        logging.info("Setting up environment and loading model...")
        agent_specific_cols = [f"{t}_{f}" for t in self.config["agent_tickers"] for f in self.config["features_to_use"]]
        
        # This check will now pass because the config is consistent
        if not all(col in self.df_full_eval_data.columns for col in agent_specific_cols):
            missing = [col for col in agent_specific_cols if col not in self.df_full_eval_data.columns]
            raise ValueError(f"Agent-specific columns missing from data: {missing}")

        df_agent_env_data = self.df_full_eval_data[agent_specific_cols].copy()
        self.eval_env = PortfolioEnv(df=df_agent_env_data, feature_columns_ordered=self.config["features_to_use"], **self.config["env_params"])
        
        if not os.path.exists(self.config["model_path"]): raise FileNotFoundError(f"Model file not found at {self.config['model_path']}")
        self.model = PPO.load(self.config["model_path"], env=self.eval_env)
        logging.info(f"Model {self.config['model_path']} loaded successfully.")
        
    def _run_rl_agent_simulation(self):
        logging.info("Evaluating RL Agent...")
        obs, _ = self.eval_env.reset()
        terminated, truncated = False, False
        self.results['rl_agent'] = {'portfolio_values': [self.eval_env.portfolio_value], 'weights_history': [self.eval_env.weights.copy()], 'turnover_history': [], 'cumulative_costs': [0.0]}
        prev_weights = self.eval_env.weights.copy()
        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self.eval_env.step(action)
            self.results['rl_agent']['portfolio_values'].append(info['portfolio_value'])
            self.results['rl_agent']['weights_history'].append(info['weights'])
            self.results['rl_agent']['cumulative_costs'].append(self.results['rl_agent']['cumulative_costs'][-1] + info['transaction_costs'])
            turnover = np.sum(np.abs(np.array(info['weights']) - prev_weights))
            self.results['rl_agent']['turnover_history'].append(turnover)
            prev_weights = np.array(info['weights']).copy()

    def _run_benchmarks(self):
        logging.info("Evaluating benchmarks...")
        num_trading_days = len(self.results['rl_agent']['portfolio_values']) - 1
        initial_balance = self.config["env_params"]["initial_balance"]
        if num_trading_days <= 0: return
        start_idx = self.config["env_params"]["window_size"]
        spy_col = f"{self.config['benchmark_ticker']}_close"
        if spy_col in self.df_full_eval_data.columns:
            spy_prices = self.df_full_eval_data[spy_col].iloc[start_idx : start_idx + num_trading_days + 1].values
            if len(spy_prices) > 1: self.results['spy_benchmark'] = {'portfolio_values': (spy_prices / spy_prices[0]) * initial_balance}
        eq_cols = [f"{t}_close" for t in self.config["agent_tickers"]]
        if all(c in self.df_full_eval_data.columns for c in eq_cols):
            prices_df = self.df_full_eval_data[eq_cols].iloc[start_idx : start_idx + num_trading_days + 1]
            returns = prices_df.pct_change().dropna()
            portfolio_returns = returns.mean(axis=1)
            eq_values = [initial_balance]
            for r in portfolio_returns: eq_values.append(eq_values[-1] * (1 + r))
            self.results['equal_weight_benchmark'] = {'portfolio_values': np.array(eq_values)}

    def _calculate_and_log_kpis(self):
        for name, data in self.results.items():
            values = np.array(data.get('portfolio_values', []))
            if len(values) < 2: continue
            daily_returns = np.diff(values) / values[:-1]
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            max_dd = self._calculate_max_drawdown(values)
            cum_return = (values[-1] / values[0]) - 1
            annual_vol = np.std(daily_returns) * np.sqrt(252)
            num_years = len(values) / 252.0
            annual_return = (1 + cum_return)**(1/num_years) - 1 if num_years > 0 else 0
            calmar = annual_return / max_dd if max_dd > 0 else float('inf')
            data['kpis'] = {"Final Portfolio Value": values[-1], "Cumulative Return": cum_return, "Annualized Volatility": annual_vol, "Annualized Sharpe Ratio": sharpe, "Max Drawdown": max_dd, "Calmar Ratio": calmar}
            if 'turnover_history' in data:
                data['kpis']["Average Daily Turnover"] = np.mean(data['turnover_history'])
                data['kpis']["Total Transaction Costs"] = data['cumulative_costs'][-1]
            logging.info(f"\n--- {name.replace('_', ' ').title()} Performance ---")
            for k, v in data['kpis'].items():
                format_str = ".2%" if any(s in k.lower() for s in ["return", "volatility", "drawdown"]) else ",.2f" if "value" in k.lower() else ".4f"
                logging.info(f"{k}: {v:{format_str}}")

    def _generate_plots(self):
        logging.info("Generating plots...")
        plt.figure(figsize=(14, 7));
        for name, data in self.results.items():
            if 'portfolio_values' in data: plt.plot(data['portfolio_values'], label=name.replace('_', ' ').title(), lw=2)
        plt.title("Portfolio Value Comparison"); plt.xlabel("Trading Day"); plt.ylabel("Portfolio Value ($)")
        plt.legend(); plt.grid(True, alpha=0.5); plt.tight_layout()
        plt.savefig(os.path.join(self.config["plots_dir"], f"{self.base_plot_name}_performance.png")); plt.close()
        if 'rl_agent' in self.results and 'weights_history' in self.results['rl_agent']:
            weights_df = pd.DataFrame(self.results['rl_agent']['weights_history'], columns=self.config["agent_tickers"])
            plt.figure(figsize=(14, 7)); plt.stackplot(weights_df.index, weights_df.T, labels=weights_df.columns)
            plt.title("RL Agent Portfolio Weights"); plt.xlabel("Trading Day"); plt.ylabel("Weight Allocation")
            plt.legend(loc='upper left'); plt.margins(0, 0); plt.tight_layout()
            plt.savefig(os.path.join(self.config["plots_dir"], f"{self.base_plot_name}_weights.png")); plt.close()
        logging.info(f"Plots saved to '{self.config['plots_dir']}' directory.")
    


    # In rl/evaluate.py, replace the existing method with this new version:

    async def _generate_llm_report(self):
        """
        Generates and prints investment reports for different user goals,
        with a robust delay and retry mechanism to handle strict free tier rate limits.
        """
        if 'rl_agent' not in self.results:
            logging.warning("RL Agent results not found, skipping LLM report.")
            return

        agent_kpis = self.results['rl_agent'].get('kpis', {})
        final_weights_list = self.results['rl_agent'].get('weights_history', [[]])[-1]
        final_weights_dict = {ticker: weight for ticker, weight in zip(self.config["agent_tickers"], final_weights_list)}

        user_goals = ["Long-Term Growth", "Mid-Term Balanced", "Short-Term Speculation"]

        print("\n" + "="*80)
        print(" GENERATING PERSONALIZED INVESTMENT REPORTS ".center(80, "="))
        print("="*80)

        for goal in user_goals:
            max_retries = 2
            for attempt in range(max_retries):
                report = await generate_investment_report(
                    kpis=agent_kpis,
                    weights=final_weights_dict,
                    user_goal=goal
                )
                
                # Check if the report contains an error message
                if "An error occurred" in report and "429" in report:
                    logging.warning(f"Rate limit hit on attempt {attempt + 1} for goal '{goal}'. Retrying in 65 seconds...")
                    await asyncio.sleep(65) # Wait a bit longer than a minute
                    continue
                else:
                    # If successful, print the report and break the retry loop
                    print(f"\n--- REPORT FOR USER GOAL: {goal} ---")
                    print(report)
                    
                    # Wait before starting the next goal's generation
                    if goal != user_goals[-1]:
                         print("\n--- [Rate Limiter] Waiting for 65 seconds before next API call... ---")
                         await asyncio.sleep(65)
                    break # Exit the retry loop
        
    async def run_full_evaluation(self):
        self._load_data()
        self._setup_environment_and_model()
        self._run_rl_agent_simulation()
        self._run_benchmarks()
        self._calculate_and_log_kpis()
        self._generate_plots()
        await self._generate_llm_report()
        logging.info("Evaluation script finished successfully.")

    @staticmethod
    def _calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
        if len(portfolio_values) < 2: return 0.0
        roll_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - roll_max) / np.where(roll_max == 0, 1e-9, roll_max)
        return -np.min(drawdown) if len(drawdown) > 0 else 0.0

# rl/evaluate.py (CORRECTED main function)
async def main():
    if PPO is None:
        return

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Build the configuration dictionary from imported settings
    # For evaluation, we often turn off reward shaping to get a pure performance metric.
    eval_env_params = ENV_PARAMS.copy()
    eval_env_params.update({
        'volatility_penalty_weight': 0.0,
        'loss_aversion_factor': 1.0,
        'turnover_penalty_weight': 0.0
    })

    # This dictionary now correctly uses the imported, centralized settings.
    config = {
        "agent_tickers": AGENT_TICKERS,
        "benchmark_ticker": BENCHMARK_TICKER,
        "features_to_use": FEATURES_TO_USE_IN_MODEL,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "model_path": MODEL_PATH,
        "plots_dir": PLOTS_DIR,
        "env_params": eval_env_params
    }
    
    try:
        evaluator = PortfolioEvaluator(config)
        await evaluator.run_full_evaluation()
    except Exception as e:
        logging.error(f"An error occurred during the evaluation pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    # Ensure you are running this with asyncio
    asyncio.run(main())
