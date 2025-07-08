#!/usr/bin/env python3
"""
Model Validation Script
Comprehensive validation of the trained models to ensure they're performing correctly.

How to check for overfitting:
- Run holdout or rolling backtests on periods not used for training.
- Compare model performance to SPY and equal-weight portfolios on out-of-sample data.
- Look for stable performance across different market regimes.

How to tune short-term models:
- Increase turnover penalty to reduce excessive trading and drawdown.
- Lower volatility target for less risk.
- Adjust window size (lookback period).
- Remove noisy features.
- Use Optuna or similar for hyperparameter search.
- Try ensembling or regularization if using neural nets.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import ModelManager
from src.utils import load_market_data_from_db
from config_short_term import AGENT_TICKERS, BENCHMARK_TICKER, FEATURES_TO_USE_IN_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAMES = [
    "short_term_conservative",
    "short_term_moderate",
    "short_term_aggressive",
    "long_term_conservative",
    "long_term_moderate",
    "long_term_aggressive",
]

MODEL_ENV_PARAMS = {
    "short_term_conservative": {
        "window_size": 30,
        "rolling_volatility_window": 14,
        "momentum_weight": 0.2,
        "mean_reversion_weight": 0.3,
        "volatility_target": 0.10,
        "turnover_penalty_weight": 0.01,
        "max_concentration_per_asset": 0.25,
        "min_holding_period": 5,
    },
    "short_term_moderate": {
        "window_size": 30,
        "rolling_volatility_window": 14,
        "momentum_weight": 0.3,
        "mean_reversion_weight": 0.2,
        "volatility_target": 0.15,
        "turnover_penalty_weight": 0.005,
        "max_concentration_per_asset": 0.35,
        "min_holding_period": 3,
    },
    "short_term_aggressive": {
        "window_size": 30,
        "rolling_volatility_window": 14,
        "momentum_weight": 0.4,
        "mean_reversion_weight": 0.1,
        "volatility_target": 0.25,
        "turnover_penalty_weight": 0.002,
        "max_concentration_per_asset": 0.5,
        "min_holding_period": 1,
    },
    "long_term_conservative": {
        "window_size": 60,
        "rolling_volatility_window": 252,
        "min_holding_period": 60,
        "risk_parity_enabled": True,
        "sector_rotation_enabled": False,
        "max_concentration_per_asset": 0.20,
        "volatility_target": 0.08,
        "turnover_penalty_weight": 0.02,
    },
    "long_term_moderate": {
        "window_size": 60,
        "rolling_volatility_window": 252,
        "min_holding_period": 30,
        "risk_parity_enabled": True,
        "sector_rotation_enabled": True,
        "max_concentration_per_asset": 0.30,
        "volatility_target": 0.12,
        "turnover_penalty_weight": 0.01,
    },
    "long_term_aggressive": {
        "window_size": 60,
        "rolling_volatility_window": 252,
        "min_holding_period": 15,
        "risk_parity_enabled": False,
        "sector_rotation_enabled": True,
        "max_concentration_per_asset": 0.40,
        "volatility_target": 0.18,
        "turnover_penalty_weight": 0.005,
    },
}

class ModelValidator:
    """Comprehensive model validation and backtesting."""

    def __init__(self):
        self.model_manager = ModelManager()
        self.results = {}

    def validate_model(self, model_name):
        logger.info("=" * 60)
        logger.info(f"VALIDATING {model_name.upper()}")
        logger.info("=" * 60)
        # Determine strategy
        if model_name.startswith("short_term"):
            strategy = "short_term"
            config = self.model_manager.configs["short_term"]
            in_sample_start = "2018-01-01"
        else:
            strategy = "long_term"
            config = self.model_manager.configs["long_term"]
            in_sample_start = "2015-01-01"
        env_params = MODEL_ENV_PARAMS[model_name]
        # Test 1: Model Loading
        logger.info("Test 1: Model Loading")
        try:
            model = self.model_manager.load_model(model_name)
            if model:
                logger.info("✅ Model loaded successfully")
            else:
                logger.error("❌ Failed to load model")
                return False
        except Exception as e:
            logger.error(f"❌ Model loading error: {e}")
            return False
        # Test 2: Data Loading (In-sample)
        logger.info("\nTest 2: Data Loading (In-sample)")
        try:
            all_tickers = sorted(list(set(config["agent_tickers"] + [config["benchmark_ticker"]])))
            df_data = load_market_data_from_db(
                tickers_list=all_tickers,
                start_date=in_sample_start,
                end_date="2021-12-31",
                min_data_points=252 + 50,
                feature_columns=config["features_to_use"],
            )
            if isinstance(df_data, pd.DataFrame) and not df_data.empty:
                logger.info(f"✅ Data loaded successfully. Shape: {df_data.shape}")
                logger.info(f"   Date range: {df_data.index[0]} to {df_data.index[-1]}")
                logger.info(f"   Available tickers: {[col.split('_')[0] for col in df_data.columns if col.endswith('_close')]}")
            else:
                logger.error("❌ Failed to load data")
                return False
        except Exception as e:
            logger.error(f"❌ Data loading error: {e}")
            return False
        # Test 3: Environment Creation (In-sample)
        logger.info("\nTest 3: Environment Creation (In-sample)")
        try:
            model_path = self.model_manager.get_model_path(model_name) or ""
            env = self.model_manager.get_environment(strategy, df_data, env_params=env_params, model_path=model_path)
            if env:
                logger.info("✅ Environment created successfully")
                logger.info(f"   Observation space: {env.observation_space}")
                logger.info(f"   Action space: {env.action_space}")
            else:
                logger.error("❌ Failed to create environment")
                return False
        except Exception as e:
            logger.error(f"❌ Environment creation error: {e}")
            return False
        # Test 4: Model Prediction (In-sample)
        logger.info("\nTest 4: Model Prediction (In-sample)")
        try:
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            logger.info(f"✅ Model prediction successful")
            logger.info(f"   Observation shape: {obs.shape}")
            logger.info(f"   Action shape: {action.shape}")
            logger.info(f"   Action values: {action}")
            # Normalize action to weights
            if isinstance(action, np.ndarray):
                action = action.flatten()
            else:
                action = np.array(action).flatten()
            action_sum = np.sum(action)
            if action_sum > 1e-6:
                weights = action / action_sum
            else:
                weights = np.full(len(config["agent_tickers"]), 1.0 / len(config["agent_tickers"]))
            logger.info(f"   Normalized weights: {dict(zip(config['agent_tickers'], weights))}")
        except Exception as e:
            logger.error(f"❌ Model prediction error: {e}")
            return False
        # Test 5: Backtesting Performance (In-sample)
        logger.info("\nTest 5: Backtesting Performance (In-sample)")
        try:
            performance = self.backtest_model(model, env, df_data, strategy)
            if performance:
                logger.info("✅ Backtesting completed successfully")
                self.print_performance_summary(performance)
            else:
                logger.error("❌ Backtesting failed")
                return False
        except Exception as e:
            logger.error(f"❌ Backtesting error: {e}")
            return False
        # Test 6: Risk Analysis (In-sample)
        logger.info("\nTest 6: Risk Analysis (In-sample)")
        try:
            risk_metrics = self.analyze_risk(performance)
            logger.info("✅ Risk analysis completed")
            self.print_risk_summary(risk_metrics)
        except Exception as e:
            logger.error(f"❌ Risk analysis error: {e}")
            return False
        # Test 7: Benchmark Comparison (In-sample)
        logger.info("\nTest 7: Benchmark Comparison (In-sample)")
        try:
            benchmark_comparison = self.compare_to_benchmarks(df_data, performance)
            logger.info("✅ Benchmark comparison completed")
            self.print_benchmark_summary(benchmark_comparison)
            self.save_benchmark_results(model_name + "_in_sample", benchmark_comparison)
        except Exception as e:
            logger.error(f"❌ Benchmark comparison error: {e}")
            return False
        # --- Out-of-sample (holdout) validation ---
        logger.info("\n================ OUT-OF-SAMPLE (HOLDOUT) VALIDATION ================")
        try:
            df_holdout = load_market_data_from_db(
                tickers_list=all_tickers,
                start_date="2022-01-01",
                end_date="2024-12-31",
                min_data_points=252,
                feature_columns=config["features_to_use"],
            )
            if not isinstance(df_holdout, pd.DataFrame) or df_holdout.empty:
                logger.warning("No holdout data available for out-of-sample validation.")
                holdout_performance = None
            else:
                env_holdout = self.model_manager.get_environment(strategy, df_holdout, env_params=env_params, model_path=model_path)
                if env_holdout is not None:
                    obs, _ = env_holdout.reset()
                    total_reward = 0
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env_holdout.step(action)
                        total_reward += reward
                        done = terminated or truncated
                    # Use the same backtest method for metrics
                    holdout_performance = self.backtest_model(model, env_holdout, df_holdout, strategy)
                    logger.info("\n📊 OUT-OF-SAMPLE PERFORMANCE SUMMARY")
                    self.print_performance_summary(holdout_performance)
                    holdout_bench = self.compare_to_benchmarks(df_holdout, holdout_performance)
                    self.print_benchmark_summary(holdout_bench)
                    self.save_benchmark_results(model_name + "_holdout", holdout_bench)
                else:
                    logger.warning("Could not create holdout environment for out-of-sample validation.")
                    holdout_performance = None
        except Exception as e:
            logger.error(f"❌ Out-of-sample validation error: {e}")
            holdout_performance = None
        # --- Overfitting check ---
        overfit_flag = False
        if holdout_performance and performance:
            in_sharpe = performance.get("sharpe_ratio", 0)
            out_sharpe = holdout_performance.get("sharpe_ratio", 0)
            in_return = performance.get("total_return", 0)
            out_return = holdout_performance.get("total_return", 0)
            if out_sharpe < 0 or (in_sharpe > 0 and out_sharpe < 0.5 * in_sharpe) or out_return < 0:
                overfit_flag = True
                logger.warning(f"⚠️  POTENTIAL OVERFITTING: Out-of-sample Sharpe dropped from {in_sharpe:.3f} to {out_sharpe:.3f}, or total return is negative.")
        # --- Save overfitting flag in summary ---
        self.results[model_name] = {
            "in_sample_sharpe": performance.get("sharpe_ratio", None) if performance else None,
            "holdout_sharpe": holdout_performance.get("sharpe_ratio", None) if holdout_performance else None,
            "in_sample_return": performance.get("total_return", None) if performance else None,
            "holdout_return": holdout_performance.get("total_return", None) if holdout_performance else None,
            "overfit_flag": overfit_flag,
        }
        logger.info("\n" + "=" * 60)
        logger.info(f"VALIDATION COMPLETED FOR {model_name.upper()}")
        logger.info("=" * 60)
        return not overfit_flag

    def backtest_model(self, model, env, df_data, strategy):
        """Run comprehensive backtesting of the model."""
        logger.info("Running backtesting...")

        # Reset environment
        obs, _ = env.reset()

        # Initialize tracking
        portfolio_values = [env.portfolio_value]
        weights_history = [env.weights.copy()]
        daily_returns = []
        transaction_costs = [0.0]

        step = 0
        total_reward = 0.0

        while True:
            try:
                # Get model prediction
                action, _ = model.predict(obs, deterministic=True)

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)

                # Track metrics
                total_reward += float(reward)
                portfolio_values.append(info["portfolio_value"])
                weights_history.append(info["weights"].copy())
                daily_returns.append(info.get("raw_daily_return", 0.0))
                transaction_costs.append(
                    transaction_costs[-1] + info.get("transaction_costs", 0.0)
                )

                step += 1

                if terminated or truncated:
                    break

            except Exception as e:
                logger.error(f"Error in backtesting step {step}: {e}")
                break

        # Calculate performance metrics
        portfolio_values = np.array(portfolio_values)
        daily_returns = np.array(daily_returns)

        if len(portfolio_values) < 2:
            logger.error("Insufficient data for backtesting")
            return None

        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Performance metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # Additional metrics
        win_rate = np.sum(returns > 0) / len(returns)
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = (
            np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        )
        sortino_ratio = (
            annualized_return / downside_volatility if downside_volatility > 0 else 0
        )

        # Calmar ratio
        calmar_ratio = (
            annualized_return / max_drawdown if max_drawdown > 0 else float("inf")
        )

        # Turnover analysis
        weights_array = np.array(weights_history)
        turnover = np.sum(np.abs(np.diff(weights_array, axis=0)), axis=1)
        avg_turnover = np.mean(turnover)

        return {
            "portfolio_values": portfolio_values,
            "weights_history": weights_history,
            "daily_returns": daily_returns,
            "returns": returns,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_turnover": avg_turnover,
            "total_reward": total_reward,
            "steps": step,
            "transaction_costs": transaction_costs[-1],
        }

    def analyze_risk(self, performance):
        """Analyze risk metrics of the model."""
        if not performance:
            return None

        returns = performance["returns"]

        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # Conditional Value at Risk (CVaR)
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])

        # Skewness and Kurtosis
        skewness = np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3)
        kurtosis = np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4) - 3

        return {
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "volatility": performance["volatility"],
            "max_drawdown": performance["max_drawdown"],
        }

    def compare_to_benchmarks(self, df_data, performance):
        """Compare model performance to benchmarks."""
        if not performance:
            return None

        # SPY benchmark
        spy_col = f"{BENCHMARK_TICKER}_close"
        if spy_col in df_data.columns:
            spy_prices = df_data[spy_col].dropna()
            spy_returns = spy_prices.pct_change().dropna()

            # Align with model performance period
            start_idx = len(df_data) - len(performance["returns"])
            spy_returns_aligned = spy_returns.iloc[
                start_idx : start_idx + len(performance["returns"])
            ]

            if len(spy_returns_aligned) == len(performance["returns"]):
                spy_total_return = (1 + spy_returns_aligned).prod() - 1
                spy_annualized_return = (1 + spy_total_return) ** (
                    252 / len(spy_returns_aligned)
                ) - 1
                spy_volatility = spy_returns_aligned.std() * np.sqrt(252)
                spy_sharpe = (
                    spy_annualized_return / spy_volatility if spy_volatility > 0 else 0
                )

                # Equal weight benchmark
                ticker_cols = [f"{ticker}_close" for ticker in AGENT_TICKERS]
                available_cols = [col for col in ticker_cols if col in df_data.columns]

                if len(available_cols) > 0:
                    ticker_prices = df_data[available_cols].dropna()
                    ticker_returns = ticker_prices.pct_change().dropna()

                    # Align with model performance period
                    ticker_returns_aligned = ticker_returns.iloc[
                        start_idx : start_idx + len(performance["returns"])
                    ]

                    if len(ticker_returns_aligned) == len(performance["returns"]):
                        # Equal weight returns
                        eq_weight_returns = ticker_returns_aligned.mean(axis=1)
                        eq_total_return = (1 + eq_weight_returns).prod() - 1
                        eq_annualized_return = (1 + eq_total_return) ** (
                            252 / len(eq_weight_returns)
                        ) - 1
                        eq_volatility = eq_weight_returns.std() * np.sqrt(252)
                        eq_sharpe = (
                            eq_annualized_return / eq_volatility
                            if eq_volatility > 0
                            else 0
                        )

                        return {
                            "model": {
                                "total_return": performance["total_return"],
                                "annualized_return": performance["annualized_return"],
                                "volatility": performance["volatility"],
                                "sharpe_ratio": performance["sharpe_ratio"],
                            },
                            "spy": {
                                "total_return": spy_total_return,
                                "annualized_return": spy_annualized_return,
                                "volatility": spy_volatility,
                                "sharpe_ratio": spy_sharpe,
                            },
                            "equal_weight": {
                                "total_return": eq_total_return,
                                "annualized_return": eq_annualized_return,
                                "volatility": eq_volatility,
                                "sharpe_ratio": eq_sharpe,
                            },
                        }

        return None

    def print_performance_summary(self, performance):
        """Print detailed performance summary."""
        logger.info("\n📊 PERFORMANCE SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Total Return: {performance['total_return']:.2%}")
        logger.info(f"Annualized Return: {performance['annualized_return']:.2%}")
        logger.info(f"Volatility: {performance['volatility']:.2%}")
        logger.info(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
        logger.info(f"Sortino Ratio: {performance['sortino_ratio']:.3f}")
        logger.info(f"Maximum Drawdown: {performance['max_drawdown']:.2%}")
        logger.info(f"Calmar Ratio: {performance['calmar_ratio']:.3f}")
        logger.info(f"Win Rate: {performance['win_rate']:.2%}")
        logger.info(f"Profit Factor: {performance['profit_factor']:.3f}")
        logger.info(f"Average Turnover: {performance['avg_turnover']:.3f}")
        logger.info(f"Total Transaction Costs: ${performance['transaction_costs']:.2f}")
        logger.info(f"Total Reward: {performance['total_reward']:.4f}")
        logger.info(f"Steps: {performance['steps']}")

    def print_risk_summary(self, risk_metrics):
        """Print risk analysis summary."""
        if not risk_metrics:
            return

        logger.info("\n⚠️  RISK ANALYSIS")
        logger.info("-" * 40)
        logger.info(f"VaR (95%): {risk_metrics['var_95']:.2%}")
        logger.info(f"VaR (99%): {risk_metrics['var_99']:.2%}")
        logger.info(f"CVaR (95%): {risk_metrics['cvar_95']:.2%}")
        logger.info(f"CVaR (99%): {risk_metrics['cvar_99']:.2%}")
        logger.info(f"Skewness: {risk_metrics['skewness']:.3f}")
        logger.info(f"Kurtosis: {risk_metrics['kurtosis']:.3f}")
        logger.info(f"Volatility: {risk_metrics['volatility']:.2%}")
        logger.info(f"Max Drawdown: {risk_metrics['max_drawdown']:.2%}")

    def print_benchmark_summary(self, comparison):
        """Print benchmark comparison summary as a table."""
        if not comparison:
            return
        logger.info("\n🏆 BENCHMARK COMPARISON (Model vs. SPY vs. Equal Weight)")
        logger.info("-" * 60)
        logger.info(f"{'Strategy':<15} {'Total Return':>12} {'Ann. Return':>12} {'Volatility':>12} {'Sharpe':>8}")
        for key, name in zip(['model', 'spy', 'equal_weight'], ['AI Model', 'SPY', 'Equal Weight']):
            if key in comparison:
                data = comparison[key]
                logger.info(f"{name:<15} {data['total_return']*100:11.2f}% {data['annualized_return']*100:11.2f}% {data['volatility']*100:11.2f}% {data['sharpe_ratio']:8.3f}")
        # Outperformance
        model_return = comparison['model']['total_return']
        spy_return = comparison['spy']['total_return']
        eq_return = comparison['equal_weight']['total_return']
        logger.info(f"\nOutperformance vs SPY: {(model_return - spy_return)*100:.2f}%")
        logger.info(f"Outperformance vs Equal Weight: {(model_return - eq_return)*100:.2f}%")

    def save_benchmark_results(self, model_name, comparison):
        """Save benchmark comparison to CSV for each model."""
        import csv
        if not comparison:
            return
        filename = f"benchmark_comparison_{model_name}.csv"
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Strategy", "Total Return", "Annualized Return", "Volatility", "Sharpe Ratio"])
            for key, name in zip(['model', 'spy', 'equal_weight'], ['AI Model', 'SPY', 'Equal Weight']):
                if key in comparison:
                    data = comparison[key]
                    writer.writerow([
                        name,
                        f"{data['total_return']:.6f}",
                        f"{data['annualized_return']:.6f}",
                        f"{data['volatility']:.6f}",
                        f"{data['sharpe_ratio']:.6f}",
                    ])

    def save_validation_report(self, filename="validation_report.json"):
        """Save validation results to file."""
        report = {
            "validation_date": datetime.now().isoformat(),
            "model_type": "short_term_moderate",
            "results": self.results,
        }

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to {filename}")


def main():
    """Main validation function."""
    validator = ModelValidator()
    results = {}
    for model_name in MODEL_NAMES:
        logger.info(f"\n\n{'#' * 80}\nEvaluating {model_name}\n{'#' * 80}")
        success = validator.validate_model(model_name)
        results[model_name] = success
    logger.info("\nSUMMARY OF ALL MODEL VALIDATIONS:")
    for model_name, success in results.items():
        overfit = validator.results.get(model_name, {}).get("overfit_flag", False)
        msg = "⚠️  OVERFITTING DETECTED" if overfit else ("✅ PASS" if success else "❌ FAIL")
        logger.info(f"{model_name}: {msg}")
    logger.info("\nNOTE: To check for overfitting, always validate on out-of-sample data and compare to benchmarks. See the top of this script for more tips.")
    # Optionally save a summary report
    with open("validation_summary.json", "w") as f:
        json.dump(validator.results, f, indent=2)
    logger.info("Validation summary saved to validation_summary.json")


if __name__ == "__main__":
    main()
