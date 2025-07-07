# 🚀 Expanded Universe AI Financial Advisor - Quick Start Guide

## Overview

This guide will help you set up and run the AI Financial Advisor with an expanded universe of 20 stocks (10 Technology + 10 Healthcare) for better diversification and more investment options.

## 🎯 What's New in the Expanded Universe

### Stock Universe

- **Technology (10 stocks)**: AAPL, MSFT, GOOGL, NVDA, AMZN, TSLA, META, NFLX, ADBE, CRM
- **Healthcare (10 stocks)**: JNJ, PFE, UNH, ABBV, TMO, LLY, MRK, BMY, AMGN, GILD
- **Total**: 20 stocks instead of the original 5

### Enhanced Features

- Better portfolio diversification
- More investment options
- Sector-based recommendations
- Enhanced technical indicators (36 per stock)
- Optimized hyperparameters for both strategies

## 🚀 Quick Start (One Command)

### Option 1: Complete Automated Pipeline

```bash
python run_expanded_universe_complete.py
```

This single command will:

1. ✅ Collect data for all 20 stocks
2. 🔍 Run Optuna optimization for both strategies
3. 🎯 Train final models with optimized parameters
4. ✅ Validate everything works
5. 🎉 Ready to run the app!

**Expected time**: 3-4 hours (depending on your system)

### Option 2: Step-by-Step (If you prefer control)

#### Step 1: Collect Data

```bash
python collect_expanded_data_simple.py
```

#### Step 2: Run Optimizations

```bash
python rl/optimize_short_term.py
python rl/optimize_long_term.py
```

#### Step 3: Train Models

```bash
python rl/train_short_term_final.py
python rl/train_long_term_final.py
```

#### Step 4: Run the App

```bash
python app.py
```

## 📊 What You'll Get

### Enhanced Portfolio Recommendations

- **20 stocks** instead of 5
- **Sector breakdown** (Tech vs Healthcare)
- **Better diversification** recommendations
- **More investment options** for users

### Improved Models

- **Optimized hyperparameters** for 20-stock universe
- **Better performance** through Optuna optimization
- **Enhanced features** (36 technical indicators per stock)
- **Strategy-specific** optimizations (short-term vs long-term)

## 🔧 Technical Details

### Data Collection

- **Enhanced data collector** with all 36 technical indicators
- **SQLite database** for efficient storage
- **Automatic feature calculation** (RSI, MACD, Bollinger Bands, etc.)

### Model Training

- **PPO (Proximal Policy Optimization)** with attention mechanisms
- **Optuna hyperparameter optimization** for best performance
- **Strategy-specific environments** (short-term vs long-term)
- **Comprehensive evaluation** with multiple KPIs

### App Features

- **Dynamic model loading** based on user profile
- **Age-based recommendations** (25, 35, 50+ age groups)
- **Investment horizon** selection (short-term vs long-term)
- **LLM-powered reports** with actionable insights

## 📈 Expected Performance Improvements

### Diversification Benefits

- **Lower portfolio volatility** through sector diversification
- **Better risk-adjusted returns** with 20 stocks
- **Reduced concentration risk** (max 5% per stock)

### Model Improvements

- **Higher Sharpe ratios** through optimization
- **Better drawdown management** with enhanced features
- **More stable performance** across market conditions

## 🛠️ Troubleshooting

### Common Issues

#### 1. Data Collection Fails

```bash
# Check if yfinance is working
python -c "import yfinance as yf; print(yf.download('AAPL', period='1d'))"
```

#### 2. Optimization Takes Too Long

- Reduce trials in optimization scripts (change `n_trials=30` to `n_trials=10`)
- Use shorter time periods for faster testing

#### 3. Model Training Fails

- Check available memory (models need ~4GB RAM)
- Ensure all dependencies are installed: `pip install -r requirements.txt`

#### 4. App Won't Start

- Verify models exist: `ls models/`
- Check logs: `tail -f expanded_universe_training.log`

### Log Files

- **Training logs**: `expanded_universe_training.log`
- **App logs**: Check console output
- **Data logs**: Check `src/data_collector_enhanced.py` output

## 🎯 Next Steps

### After Training Completes

1. **Start the app**: `python app.py`
2. **Open browser**: `http://localhost:5000`
3. **Enter your profile**: Age, investment amount, time horizon
4. **Get recommendations**: View portfolio suggestions and insights

### Customization Options

- **Add more stocks**: Edit `config_short_term.py` and `config_long_term.py`
- **Modify features**: Adjust `FEATURES_TO_USE_IN_MODEL` in configs
- **Change strategies**: Modify environment parameters in configs

## 📚 Additional Resources

- **Full Documentation**: `README.md`
- **Training Guide**: `TRAINING_GUIDE.md`
- **Extended Universe Guide**: `EXTENDED_UNIVERSE_GUIDE.md`
- **Evaluation Scripts**: `eval/` directory

## 🎉 Success Indicators

You'll know everything is working when:

- ✅ Data collection shows "20 stocks with data"
- ✅ Optimization completes with best parameters saved
- ✅ Model training shows "best_model.zip" created
- ✅ App loads without errors
- ✅ Portfolio recommendations show 20 stocks with sector breakdown

---

**Ready to get started? Run: `python run_expanded_universe_complete.py`**
