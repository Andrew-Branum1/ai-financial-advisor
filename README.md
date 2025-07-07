# AI Financial Advisor - First-Time Investor Guide

A sophisticated AI-powered financial advisor designed specifically for first-time investors. This system combines advanced reinforcement learning, personalized investment profiles, and AI-powered insights to make investing accessible and understandable for everyone.

## 🎯 For First-Time Investors

### **Personalized Investment Guidance**

- **Age-Based Advice**: Tailored recommendations based on your life stage
- **Risk Profile Assessment**: Understand your risk tolerance and capacity
- **Investment Amount Optimization**: Smart allocation based on your budget
- **Time Horizon Planning**: Short-term vs long-term strategy selection

### **User-Friendly Interface**

- **Simple Profile Form**: Just answer a few questions about yourself
- **Visual Portfolio Charts**: See your recommended allocation clearly
- **Plain Language Explanations**: No complex financial jargon
- **Step-by-Step Guidance**: Clear next steps to start investing

### **AI-Powered Insights**

- **Personalized Analysis**: AI-generated investment recommendations
- **Market Education**: Learn about investing in simple terms
- **Risk Management**: Understand how to protect your investments
- **Growth Strategies**: Long-term wealth building guidance

## 🚀 Quick Start for First-Time Investors

### 1. Launch the Application

```bash
python app.py
```

### 2. Open Your Browser

Visit `http://localhost:5001`

### 3. Fill Out Your Profile

- **Your Age**: Helps determine your investment timeline
- **Investment Amount**: How much you want to invest
- **Time Horizon**: Short-term (1-3 years) or Long-term (5+ years)
- **Risk Tolerance**: Conservative, Moderate, or Aggressive

### 4. Get Your Personalized Plan

The AI will analyze your profile and provide:

- 📊 **Investment Strategy**: Tailored to your goals
- 🎯 **Age-Specific Advice**: Based on your life stage
- 💼 **Portfolio Recommendation**: Specific stock allocations
- 🤖 **AI Analysis**: Plain-language investment insights
- 📝 **Next Steps**: How to start investing

## 🎯 Demo the System

Run our interactive demo to see how the system works:

```bash
python demo_first_time_investor.py
```

This demo shows:

- Sample investor profiles (different ages and goals)
- How the AI selects appropriate strategies
- Portfolio recommendations for each profile
- AI-powered investment analysis

## 🏗️ Advanced Features

### **Multi-Strategy Portfolio Management**

- **Short-term Trading**: Active trading strategy for quick gains
- **Long-term Growth**: Sustained growth strategy for wealth building
- **Dynamic Model Selection**: AI automatically chooses the best strategy for you
- **Risk Management**: Built-in protection against market volatility

### **Advanced Technical Analysis**

- **20+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic
- **Volume Analysis**: OBV, MFI, volume ratios, and momentum indicators
- **Multi-timeframe Analysis**: Short-term and long-term trend analysis
- **Volatility Modeling**: Smart risk management based on market conditions

### **Reinforcement Learning Models**

- **PPO (Proximal Policy Optimization)**: State-of-the-art AI algorithm
- **Attention Mechanisms**: Advanced pattern recognition
- **Custom Reward Functions**: Optimized for different investment goals
- **Hyperparameter Optimization**: Automated model tuning

### **Comprehensive Evaluation**

- **Performance Metrics**: Sharpe ratio, Sortino ratio, max drawdown
- **Benchmark Comparison**: Compare against market indices
- **Risk Analysis**: Value at Risk, win rate, profit factor
- **LLM-Powered Reports**: AI-generated investment insights

## 🚀 New Pipeline Highlights (2025)

- **20-Ticker Universe:** All models are now trained and validated on a diverse set of 20 tickers spanning multiple sectors and geographies.
- **Top-5 Selection Logic:** At each step, the RL agent can only allocate to its 5 best tickers (by predicted weight); all other weights are set to zero and the top 5 are re-normalized. This is enforced in the environment logic.
- **Consistent Training & Validation:** All scripts reference the same universe, ensuring robust and fair evaluation.

### Why Top-5 Selection?

- Improves interpretability for users and stakeholders.
- Reduces overfitting by forcing the model to focus on its best ideas.
- Mimics real-world portfolio construction constraints.

## 📁 Project Structure

```
ai-financial-advisor/
├── app.py                          # Main web application (First-time investor interface)
├── app_extended_universe.py        # Web app for expanded universe features
├── train_all_models.py             # Comprehensive training script for all RL models
├── config.py                       # Main configuration
├── config_short_term.py            # Short-term strategy config
├── config_long_term.py             # Long-term strategy config
├── config_extended_universe.py     # Extended universe config
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── PROJECT_FLOW.md                 # Project flow and architecture
│
├── rl/                            # Reinforcement Learning
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
├── src/                           # Core utilities and data collection
│   ├── data_collector_enhanced.py # Enhanced data collection
│   ├── data_collector_extended.py # Extended universe data collection
│   └── utils.py                   # Utility functions
│
├── llm/                           # Language Model Integration
│   └── advisor.py                 # AI-powered investment reports
│
├── eval/                          # Evaluation and validation scripts
│   ├── validate_model.py          # Model validation and performance metrics
│   ├── eval_holdout_backtest.py   # Holdout backtest evaluation
│   └── eval_rolling_backtest.py   # Rolling window backtest
│
├── templates/                     # Web interface templates
│   └── index.html                 # First-time investor interface
│   └── index_extended.html        # Extended universe interface
│
├── data/                          # Market data storage
├── models/                        # Trained model files
├── logs/                          # Log files
├── optimization_results/          # Hyperparameter optimization results
├── tests/                         # Unit and integration tests
├── plots/                         # Performance plots and charts
├── archived/                      # Old/unused scripts and docs
└── .gitignore, Dockerfile, ...    # Miscellaneous files
```

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ai-financial-advisor
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the database**
   ```bash
   python -m src.data_collector
   ```

## 🚀 Advanced Usage

### For Developers and Researchers

#### 1. Data Collection

```bash
# Collect market data for all configured tickers
python -m src.data_collector_enhanced
```

#### 2. Train Models

```bash
python train_all_models.py
```

#### 3. Evaluate Performance

```bash
python eval/validate_model.py
```

#### 4. Run the Web App

```bash
python app.py
# or for expanded universe features
python app_extended_universe.py
```

## 📊 Strategy Overview

### Short-term Trading Strategy

- **Objective**: Maximize daily to weekly returns with active trading
- **Features**: 14-day volatility window, momentum signals, mean reversion
- **Risk Management**: Volatility targeting, turnover penalties, drawdown limits
- **Best For**: Active traders, day traders, momentum investors

### Long-term Growth Strategy

- **Objective**: Sustainable growth over 3-5+ year periods
- **Features**: 252-day analysis window, risk parity, sector rotation
- **Risk Management**: Minimum holding periods, diversification, trend following
- **Best For**: Long-term investors, retirement planning, wealth building

## 🔧 Configuration

### Key Parameters

**Short-term Strategy:**

```python
ENV_PARAMS = {
    'window_size': 30,                    # Observation window
    'rolling_volatility_window': 14,      # Volatility calculation period
    'momentum_weight': 0.3,               # Momentum signal weight
    'mean_reversion_weight': 0.2,         # Mean reversion weight
    'volatility_target': 0.15,            # Target annualized volatility
    'turnover_penalty_weight': 0.005,     # Trading cost penalty
}
```

**Long-term Strategy:**

```python
ENV_PARAMS = {
    'window_size': 60,                    # Longer observation window
    'rolling_volatility_window': 252,     # Full year analysis
    'min_holding_period': 30,             # Minimum holding days
    'risk_parity_enabled': True,          # Risk parity allocation
    'sector_rotation_enabled': True,      # Sector rotation
    'max_concentration_per_asset': 0.4,   # Diversification limit
}
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Return Metrics**: Cumulative return, annualized return, Sharpe ratio, Sortino ratio
- **Risk Metrics**: Maximum drawdown, Value at Risk (VaR), Conditional VaR (CVaR)
- **Trading Metrics**: Win rate, profit factor, average turnover, transaction costs
- **Risk-Adjusted Metrics**: Calmar ratio, information ratio, Treynor ratio

## 🤖 AI-Powered Insights

The system generates intelligent investment reports using LLM integration:

- **Strategy Analysis**: Detailed breakdown of model performance
- **Risk Assessment**: Comprehensive risk analysis and recommendations
- **Portfolio Optimization**: Suggestions for weight allocation and rebalancing
- **Market Outlook**: AI-generated market insights and predictions

## 🔬 Advanced Features

### Technical Indicators

- **Trend Indicators**: SMA, EMA, MACD, ADX, Parabolic SAR
- **Momentum Indicators**: RSI, Stochastic, Williams %R, CCI
- **Volatility Indicators**: Bollinger Bands, ATR, Keltner Channels
- **Volume Indicators**: OBV, MFI, Volume Rate of Change

### Risk Management

- **Dynamic Position Sizing**: Adjust positions based on volatility
- **Stop-Loss Mechanisms**: Automatic risk control
- **Correlation Analysis**: Diversification optimization
- **Stress Testing**: Portfolio resilience under extreme conditions

## 📚 Educational Resources

### For First-Time Investors

- **Investment Basics**: Learn about stocks, bonds, and ETFs
- **Risk Management**: Understand how to protect your money
- **Market Cycles**: Learn about bull and bear markets
- **Dollar-Cost Averaging**: A smart way to invest regularly

### For Advanced Users

- **Technical Analysis**: Deep dive into chart patterns and indicators
- **Portfolio Theory**: Modern Portfolio Theory and optimization
- **Algorithmic Trading**: Understanding the AI behind the recommendations
- **Backtesting**: How to test investment strategies

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Bug reports and feature requests
- Code contributions and improvements
- Documentation updates
- Testing and validation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Investment decisions should be made with careful consideration of your financial situation and goals. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## 🆘 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs or request features via GitHub issues
- **Demo**: Run `python demo_first_time_investor.py` for interactive examples
- **Web Interface**: Use the web app for the easiest experience

---

**Made with ❤️ for first-time investors who want to build wealth intelligently.**
