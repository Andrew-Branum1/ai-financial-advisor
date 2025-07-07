# AI Financial Advisor: Project Flow & Architecture

## Overview

This project is an end-to-end AI-powered financial advisor system designed to provide personalized investment recommendations for first-time investors. It leverages reinforcement learning (RL), advanced technical indicators, and LLM-powered (Large Language Model) investment reports. The system is modular, robust, and designed for both research and practical deployment.

---

## 1. Project Structure

```
├── app.py                    # Main Flask web app (user interface)
├── app_extended_universe.py  # Web app for expanded universe features
├── train_all_models.py       # Comprehensive training script for all RL models
├── config.py                 # Main configuration
├── config_short_term.py      # Short-term strategy config
├── config_long_term.py       # Long-term strategy config
├── config_extended_universe.py # Extended universe config
├── requirements.txt          # Python dependencies
├── README.md, PROJECT_FLOW.md # Documentation
│
├── rl/                      # RL environments, custom PPO, and training logic
│   ├── portfolio_env_short_term.py # Short-term trading environment
│   ├── portfolio_env_long_term.py  # Long-term growth environment
│   ├── attention_policy.py         # Attention-based policy network
│   ├── custom_ppo.py               # Custom PPO implementation
│   ├── train_short_term_final.py   # Short-term model training (final)
│   ├── train_long_term_final.py    # Long-term model training (final)
│   ├── optimize_short_term.py      # Hyperparameter optimization (short-term)
│   ├── optimize_long_term.py       # Hyperparameter optimization (long-term)
│   ├── evaluate_enhanced.py        # Enhanced evaluation script
│   └── universal_portfolio_env.py  # Env for large/variable universes
│
├── src/                           # Core data processing and utilities
│   ├── data_collector_enhanced.py # Enhanced data collection
│   ├── data_collector_extended.py # Extended universe data collection
│   └── utils.py                   # Utility functions
│
├── llm/                           # LLM-powered investment report generation
│   └── advisor.py                 # AI-powered investment reports
│
├── eval/                          # Evaluation and validation scripts
│   ├── validate_model.py          # Model validation and performance metrics
│   ├── eval_holdout_backtest.py   # Holdout backtest evaluation
│   └── eval_rolling_backtest.py   # Rolling window backtest
│
├── templates/                     # HTML templates for the web UI
│   ├── index.html                 # First-time investor interface
│   └── index_extended.html        # Extended universe interface
│
├── data/                          # Market data and SQLite DB
├── models/                        # Trained model checkpoints
├── logs/                          # Log files
├── optimization_results/          # Hyperparameter optimization results
├── tests/                         # Unit and integration tests
├── plots/                         # Performance plots and charts
├── archived/                      # Old/unused scripts and docs
└── .gitignore, Dockerfile, ...    # Miscellaneous files
```

---

## 2. Data Pipeline

### a. Data Collection

- **Script:** `collect_expanded_data.py`
- **Function:** Downloads historical market data for a universe of stocks (e.g., S&P 500 subset) and computes a comprehensive set of technical indicators (momentum, volatility, trend, volume, etc.).
- **Output:** Data is stored in `data/market_data.db` (SQLite) and/or as processed DataFrames.

### b. Feature Engineering

- **Location:** `src/utils.py`, `src/data_collector*.py`
- **Function:** Computes rolling statistics, technical indicators, and normalizes features for RL environments.

---

## 3. Model Training Pipeline

### a. Model Profiles

- **Defined in:** `train_all_models.py`
- **Profiles:**
  - Short-term: Conservative, Moderate, Aggressive
  - Long-term: Conservative, Moderate, Aggressive
- **Each profile** has its own environment parameters (e.g., volatility target, holding period, turnover penalty) reflecting different investor goals and risk tolerances.

### b. Training Script

- **Script:** `train_all_models.py`
- **Flow:**
  1. **Loads profile-specific config** (from `config_short_term.py`, `config_long_term.py`).
  2. **Loads market data** for the relevant tickers, features, and time window.
  3. **Creates the RL environment** (`PortfolioEnvShortTerm` or `PortfolioEnvLongTerm`) with profile-specific parameters.
  4. **Runs Optuna hyperparameter optimization** for PPO (20 trials per model).
  5. **Trains the final model** with the best hyperparameters for 200,000 timesteps.
  6. **Saves the model** and training metadata in `models/`.

### c. RL Environments

- **Location:** `rl/portfolio_env_short_term.py`, `rl/portfolio_env_long_term.py`
- **Reward Functions:**
  - Parameterized by profile (e.g., Sharpe ratio, momentum, drawdown, turnover penalty).
  - No need for separate reward functions per profile; all logic is parameterized.

---

## 4. Model Evaluation

- **Scripts:** `eval/validate_model.py`, `eval/`
- **Function:** Evaluates trained models on holdout or rolling windows, computes KPIs (return, drawdown, Sharpe), and generates plots.
- **Output:** Metrics and visualizations for model performance.

---

## 5. User-Facing Applications

### a. CLI Demo

- **Script:** `demo_first_time_investor.py`
- **Function:** Simulates a first-time investor experience, mapping user profiles to RL environment parameters, running the trained model, and generating an LLM-powered investment report.

### b. Web App

- **Script:** `app.py`
- **Function:** Flask web interface where users enter their profile (age, investment amount, time horizon, risk tolerance). The app selects the appropriate model, runs inference, and presents results and an LLM-generated explanation.
- **Templates:** Located in `templates/`.

---

## 6. LLM-Powered Investment Reports

- **Location:** `llm/advisor.py`
- **Function:** Uses a large language model (e.g., Gemini, GPT) to generate plain-English, beginner-friendly investment analysis and explanations based on model outputs and user profile.

---

## 7. Configuration & Extensibility

- **Config Files:** `config_short_term.py`, `config_long_term.py`, etc.
- **Purpose:** Define tickers, features, and environment parameters for each strategy/profile.
- **Easy to extend:** Add new profiles, strategies, or features by updating config files and model configs.

---

## 8. Best Practices & Reproducibility

- **All major steps are logged** (to console and/or log files).
- **Training metadata is saved** with each model (hyperparameters, tickers, features, date).
- **Random seeds can be set** for reproducibility (recommended for research).
- **Code is modular and well-documented** for easy onboarding and extension.

---

## 9. Typical End-to-End Workflow

1. **Collect Data:**
   ```bash
   python -m src.data_collector_enhanced
   ```
2. **Train All Models:**
   ```bash
   python train_all_models.py
   ```
3. **Validate Models:**
   ```bash
   python eval/validate_model.py
   ```
4. **Run Web App:**
   ```bash
   python app.py
   # or for expanded universe features
   python app_extended_universe.py
   # Then visit http://localhost:5001
   ```

---

## 10. Key Files & Their Roles

| File/Folder                | Purpose                                         |
| -------------------------- | ----------------------------------------------- |
| app.py                     | Web app (Flask)                                 |
| app_extended_universe.py   | Web app for expanded universe                   |
| train_all_models.py        | Trains all RL models for all profiles           |
| src/                       | Data processing, feature engineering, utilities |
| rl/                        | RL environments, custom PPO, training logic     |
| llm/                       | LLM-powered investment report generation        |
| models/                    | Trained model checkpoints                       |
| data/                      | Market data and SQLite DB                       |
| templates/                 | HTML templates for web UI                       |
| config\_\*.py              | Config files for different strategies/profiles  |
| eval/                      | Evaluation and validation scripts               |
| tests/                     | Unit and integration tests                      |
| logs/                      | Log files                                       |
| optimization_results/      | Hyperparameter optimization results             |
| plots/                     | Performance plots and charts                    |
| archived/                  | Old/unused scripts and docs                     |
| README.md, PROJECT_FLOW.md | Documentation                                   |

---

## 11. Extending or Modifying the System

- **To add a new profile:**
  - Update `train_all_models.py` with a new config.
  - Add any new features/tickers to the relevant config file.
- **To change the reward function:**
  - Edit the relevant method in `rl/portfolio_env_short_term.py` or `rl/portfolio_env_long_term.py`.
- **To add new features:**
  - Update feature engineering in `src/utils.py` and config files.

---

## 12. Troubleshooting & Tips

- **If training fails:** Check logs for errors, ensure data is available, and dependencies are installed.
- **If models are not used in the app:** Ensure the model directories and config files are up to date.
- **For reproducibility:** Set random seeds in all scripts.

---

## 13. Contact & Further Reading

- See `README.md` and `TRAINING_GUIDE.md` for more details.
- For questions, contact the project maintainer or refer to the code comments.

## Updated Project Flow (2025)

1. **Data Collection**

   - Collect historical data for 20 diverse tickers spanning US tech, healthcare, financials, energy, consumer, and international markets.
   - Use `src/data_collector_enhanced.py` or similar, which now fetches the new universe from `config.py`.

2. **Model Training**

   - All models (short/long term, all risk levels) are trained on the same 20-ticker universe.
   - The action space allows allocation to all 20 tickers.

3. **Top-5 Selection Logic**

   - In both short-term and long-term RL environments, after the model outputs its action vector, only the top 5 tickers (by predicted weight) are selected for allocation; all other weights are set to zero, and the top 5 are re-normalized to sum to 1.
   - This ensures the model focuses on its 5 best ideas at each step, improving interpretability and reducing overfitting.

4. **Validation & Benchmarking**

   - Validation script (`eval/validate_model.py`) evaluates all models on the same 20-ticker universe, comparing to SPY and equal-weight benchmarks.
   - Out-of-sample (holdout) validation is performed to check for overfitting.

5. **Documentation**
   - All configs and scripts now reference the same 20-ticker universe for consistency.
