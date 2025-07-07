# AI Financial Advisor - Training Guide

## 🎯 **Overview**

This guide explains how to train the AI Financial Advisor models using the **proper two-step process**:

1. **Step 1: Hyperparameter Optimization** - Use Optuna to find the best parameters
2. **Step 2: Final Model Training** - Train models with the optimized parameters

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│  STEP 1: HYPERPARAMETER OPTIMIZATION                        │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Short-term      │    │ Long-term       │                │
│  │ Optimization    │    │ Optimization    │                │
│  │ (2-3 hours)     │    │ (2-3 hours)     │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           └───────┬───────────────┘                        │
│                   ▼                                        │
│  ┌─────────────────────────────────┐                        │
│  │ optimization_results/           │                        │
│  │ ├── short_term_YYYYMMDD-HHMMSS/ │                        │
│  │ │   └── best_hyperparameters.json│                        │
│  │ └── long_term_YYYYMMDD-HHMMSS/  │                        │
│  │     └── best_hyperparameters.json│                        │
│  └─────────────────────────────────┘                        │
├─────────────────────────────────────────────────────────────┤
│  STEP 2: FINAL MODEL TRAINING                              │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Short-term      │    │ Long-term       │                │
│  │ Final Training  │    │ Final Training  │                │
│  │ (1-2 hours)     │    │ (1-2 hours)     │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           └───────┬───────────────┘                        │
│                   ▼                                        │
│  ┌─────────────────────────────────┐                        │
│  │ models/                         │                        │
│  │ ├── short_term_final_YYYYMMDD/  │                        │
│  │ │   ├── final_model.zip         │                        │
│  │ │   ├── best_model.zip          │                        │
│  │ │   └── training_config.json    │                        │
│  │ └── long_term_final_YYYYMMDD/   │                        │
│  │     ├── final_model.zip         │                        │
│  │     ├── best_model.zip          │                        │
│  │     └── training_config.json    │                        │
│  └─────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 **Quick Start**

### **Option 1: Simple Training (Recommended)**

Run the complete pipeline with one command:

```bash
python train_models_simple.py
```

This will:

- ✅ Check prerequisites
- 🔍 Run hyperparameter optimization (2-3 hours)
- 🎯 Train final models (1-2 hours)
- ✅ Verify results

### **Option 2: Manual Step-by-Step**

If you want more control or need to debug:

#### **Step 1: Hyperparameter Optimization**

```bash
# Optimize short-term model parameters
python rl/optimize_short_term.py

# Optimize long-term model parameters
python rl/optimize_long_term.py
```

#### **Step 2: Final Model Training**

```bash
# Train short-term model with best parameters
python rl/train_short_term_final.py

# Train long-term model with best parameters
python rl/train_long_term_final.py
```

## 📊 **Model Strategies**

### **Short-Term Strategy**

- **Time Horizon**: 1-12 months
- **Focus**: Momentum, mean reversion, volatility targeting
- **Risk Profile**: Higher volatility tolerance
- **Rebalancing**: Frequent (weekly/monthly)
- **Features**: Technical indicators, short-term patterns

### **Long-Term Strategy**

- **Time Horizon**: 1-10 years
- **Focus**: Value investing, dividend growth, compound returns
- **Risk Profile**: Lower volatility tolerance
- **Rebalancing**: Infrequent (quarterly/annually)
- **Features**: Fundamental indicators, long-term trends

## ⚙️ **Hyperparameter Optimization Details**

### **Environment Parameters (Optimized)**

- `window_size`: Lookback period for features
- `rolling_volatility_window`: Volatility calculation window
- `momentum_weight`: Weight for momentum signals
- `mean_reversion_weight`: Weight for mean reversion signals
- `volatility_target`: Target portfolio volatility
- `turnover_penalty_weight`: Penalty for frequent trading
- `max_concentration_per_asset`: Maximum allocation per asset
- `min_holding_period`: Minimum holding period for positions

### **PPO Parameters (Optimized)**

- `learning_rate`: Learning rate for policy updates
- `n_steps`: Number of steps per update
- `batch_size`: Batch size for training
- `n_epochs`: Number of epochs per update
- `gamma`: Discount factor for future rewards
- `gae_lambda`: GAE lambda parameter
- `clip_range`: PPO clipping parameter
- `ent_coef`: Entropy coefficient for exploration

## 📁 **File Structure**

```
ai-financial-advisor/
├── rl/
│   ├── optimize_short_term.py      # Step 1: Short-term optimization
│   ├── optimize_long_term.py       # Step 1: Long-term optimization
│   ├── train_short_term_final.py   # Step 2: Short-term training
│   ├── train_long_term_final.py    # Step 2: Long-term training
│   ├── portfolio_env_short_term.py # Short-term environment
│   ├── portfolio_env_long_term.py  # Long-term environment
│   └── ...
├── optimization_results/           # Step 1 outputs
│   ├── short_term_YYYYMMDD-HHMMSS/
│   │   └── best_hyperparameters.json
│   └── long_term_YYYYMMDD-HHMMSS/
│       └── best_hyperparameters.json
├── models/                         # Step 2 outputs
│   ├── short_term_final_YYYYMMDD/
│   │   ├── final_model.zip
│   │   ├── best_model.zip
│   │   └── training_config.json
│   └── long_term_final_YYYYMMDD/
│       ├── final_model.zip
│       ├── best_model.zip
│       └── training_config.json
└── ...
```

## 🔧 **Configuration**

### **Data Configuration**

- **Training Period**: 2010-2023 (14 years)
- **Validation Period**: 2024 (1 year)
- **Features**: Technical indicators, price data, volume data
- **Assets**: S&P 500 stocks, ETFs, bonds

### **Training Configuration**

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MLP (Multi-Layer Perceptron)
- **Device**: CPU (configurable for GPU)
- **Timesteps**: 300K (short-term), 500K (long-term)

## 📈 **Performance Metrics**

### **Optimization Metrics**

- **Primary**: Sharpe Ratio
- **Secondary**: Total Return, Maximum Drawdown
- **Tertiary**: Volatility, Turnover

### **Training Metrics**

- **Training**: Loss curves, reward progression
- **Validation**: Out-of-sample performance
- **Final**: Portfolio value, risk-adjusted returns

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **1. "No optimization results found"**

```bash
# Run optimization first
python rl/optimize_short_term.py
python rl/optimize_long_term.py
```

#### **2. "Failed to load training data"**

```bash
# Check data availability
python -c "from src.utils import load_market_data_from_db; print('Data OK')"
```

#### **3. "Out of memory"**

```bash
# Reduce batch size in optimization
# Edit rl/optimize_*.py and reduce batch_size options
```

#### **4. "Training too slow"**

```bash
# Reduce timesteps for faster iteration
# Edit total_timesteps in train_*_final.py
```

### **Performance Optimization**

#### **For Faster Training**

- Reduce `n_trials` in optimization (default: 30)
- Reduce `total_timesteps` in training
- Use smaller data windows

#### **For Better Results**

- Increase `n_trials` in optimization
- Increase `total_timesteps` in training
- Use longer data periods

## ✅ **Success Criteria**

### **Optimization Success**

- ✅ Best Sharpe Ratio > 0.5
- ✅ Convergence in optimization curves
- ✅ Reasonable parameter ranges

### **Training Success**

- ✅ Training loss decreases
- ✅ Validation performance improves
- ✅ Final model files created
- ✅ No errors in training logs

### **Model Quality**

- ✅ Sharpe Ratio > 0.3 (out-of-sample)
- ✅ Positive total return
- ✅ Reasonable volatility (< 20%)
- ✅ Acceptable maximum drawdown (< 30%)

## 🎯 **Next Steps**

After successful training:

1. **Test the Web Interface**

   ```bash
   python app.py
   # Open http://localhost:5001
   ```

2. **Run the Demo**

   ```bash
   python demo_first_time_investor.py
   ```

3. **Evaluate Performance**

   ```bash
   python rl/evaluate_enhanced.py
   ```

4. **Monitor and Improve**
   - Track real-world performance
   - Retrain with new data
   - Optimize hyperparameters further

## 📚 **Additional Resources**

- **README.md**: Project overview and setup
- **app.py**: Web interface implementation
- **demo\_\*.py**: Example usage and demonstrations
- **config\_\*.py**: Configuration files for different strategies

---

**Note**: This training process follows best practices for reinforcement learning in finance, ensuring robust hyperparameter optimization and proper model training separation.
