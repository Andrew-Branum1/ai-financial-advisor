# AI Financial Advisor: Project Flow & Architecture

## Overview

This project is an end-to-end AI-powered financial advisor system designed to provide personalized investment recommendations for first-time investors. It leverages reinforcement learning (RL), advanced technical indicators, and LLM-powered (Large Language Model) investment reports. The system is modular, robust, and designed for both research and practical deployment.

---

## 1. Project Structure

```
├── app.py                    # Main Flask web app (user interface)
├── demo_first_time_investor.py # CLI demo for first-time investors
├── train_all_models.py       # Comprehensive training script for all RL models
├── collect_expanded_data.py  # Data collection and feature engineering
├── src/                     # Core data processing and utilities
├── rl/                      # RL environments, custom PPO, and training logic
├── llm/                     # LLM-powered investment report generation
├── models/                  # Trained model checkpoints
├── data/                    # Market data and SQLite DB
├── templates/               # HTML templates for the web UI
├── tests/, eval/            # Testing and evaluation scripts
├── config_*.py              # Config files for different strategies
├── requirements.txt         # Python dependencies
├── README.md, PROJECT_FLOW.md # Documentation
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

- **Scripts:** `validate_model.py`, `eval/`
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
   python collect_expanded_data.py
   ```
2. **Train All Models:**
   ```bash
   python train_all_models.py
   ```
3. **Validate Models:**
   ```bash
   python validate_model.py
   ```
4. **Run Demo (CLI):**
   ```bash
   python demo_first_time_investor.py
   ```
5. **Run Web App:**
   ```bash
   python app.py
   # Then visit http://localhost:5001
   ```

---

## 10. Key Files & Their Roles

| File/Folder                 | Purpose                                         |
| --------------------------- | ----------------------------------------------- |
| app.py                      | Web app (Flask)                                 |
| demo_first_time_investor.py | CLI demo for first-time investors               |
| train_all_models.py         | Trains all RL models for all profiles           |
| collect_expanded_data.py    | Data collection and feature engineering         |
| src/                        | Data processing, feature engineering, utilities |
| rl/                         | RL environments, custom PPO, training logic     |
| llm/                        | LLM-powered investment report generation        |
| models/                     | Trained model checkpoints                       |
| data/                       | Market data and SQLite DB                       |
| templates/                  | HTML templates for web UI                       |
| config\_\*.py               | Config files for different strategies/profiles  |
| tests/, eval/               | Testing and evaluation scripts                  |
| README.md, PROJECT_FLOW.md  | Documentation                                   |

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
