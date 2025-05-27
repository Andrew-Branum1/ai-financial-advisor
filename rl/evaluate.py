from stable_baselines3 import PPO
from rl.portfolio_env import PortfolioEnv
from utils.data_loader import load_market_data
import matplotlib.pyplot as plt
import numpy as np


def sharpe_ratio(values):
    returns = np.diff(values) / values[:-1]
    return np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0


def max_drawdown(values):
    peak = np.maximum.accumulate(values)
    drawdowns = 1 - values / peak
    return np.max(drawdowns)


if __name__ == "__main__":
    df = load_market_data()

    # Flatten column names
    df.columns = [col.replace("('", "").replace("', '", ",").replace("')", "").split(",")[0] if col.startswith("('") else col for col in df.columns]

    print("Cleaned Columns:", df.columns)

    env = PortfolioEnv(df)
    model = PPO.load("models/ppo_portfolio_baseline")

    obs = env.reset()
    portfolio_values_rl = [env.portfolio_value]

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        portfolio_values_rl.append(env.portfolio_value)
        if done:
            break

    print(f"Final Portfolio Value (RL Agent): ${env.portfolio_value:.4f}")
    print(f"Sharpe Ratio (RL Agent): {sharpe_ratio(portfolio_values_rl):.4f}")
    print(f"Max Drawdown (RL Agent): {max_drawdown(portfolio_values_rl):.4f}")

    # ====== Buy and Hold S&P 500 ======
    if "SPY" in df.columns:
        spy_prices = df["SPY"].values
        initial_price = spy_prices[env.window_size]
        buy_and_hold_values = spy_prices[env.window_size:] / initial_price * env.initial_balance
        pad_length = len(portfolio_values_rl) - len(buy_and_hold_values)
        buy_and_hold_values = np.pad(buy_and_hold_values, (pad_length, 0), 'edge')

        print(f"Final Portfolio Value (Buy & Hold SPY): ${buy_and_hold_values[-1]:.4f}")
        print(f"Sharpe Ratio (Buy & Hold SPY): {sharpe_ratio(buy_and_hold_values):.4f}")
        print(f"Max Drawdown (Buy & Hold SPY): {max_drawdown(buy_and_hold_values):.4f}")

    # ====== Equal-Weight Baseline ======
    equal_weight_values = [env.initial_balance]
    prices = df.values
    weights = np.ones(prices.shape[1]) / prices.shape[1]
    for i in range(env.window_size + 1, len(prices)):
        prev = prices[i - 1]
        curr = prices[i]
        if np.any(prev == 0) or not np.all(np.isfinite(prev)) or not np.all(np.isfinite(curr)):
            growth = 1
        else:
            returns = curr / prev - 1
            growth = 1 + np.dot(returns, weights)
        equal_weight_values.append(equal_weight_values[-1] * growth)

    print(f"Final Portfolio Value (Equal-Weight): ${equal_weight_values[-1]:.4f}")
    print(f"Sharpe Ratio (Equal-Weight): {sharpe_ratio(equal_weight_values):.4f}")
    print(f"Max Drawdown (Equal-Weight): {max_drawdown(equal_weight_values):.4f}")

    # ====== Plot Comparison ======
    plt.plot(portfolio_values_rl, label="RL Agent")
    if 'buy_and_hold_values' in locals():
        plt.plot(buy_and_hold_values, label="Buy & Hold SPY")
    plt.plot(equal_weight_values, label="Equal-Weight Portfolio")
    plt.title("Portfolio Value Comparison")
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()