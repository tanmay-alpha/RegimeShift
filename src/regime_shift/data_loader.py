import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ASSETS = ["equity", "gold", "bonds"]


def _simulate_prices(start="2019-01-01", end="2022-12-31"):
    """
    Simulate multi-asset prices with regime-switching drift and volatility.

    Returns a DataFrame indexed by business days with columns for
    equity, gold, and bonds, priced to start at 100.
    """
    np.random.seed(42)
    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)

    # Fixed correlation matrix (PSD by construction)
    corr = np.array([
        [1.00, 0.08, 0.12],
        [0.08, 1.00, 0.08],
        [0.12, 0.08, 1.00],
    ])
    L = np.linalg.cholesky(corr)  # lower-triangular, corr = L @ L.T

    # Regime-specific drift and volatility
    # vol_map values are annualized; convert to daily for use in returns
    SQRT_252 = np.sqrt(252)
    regimes = np.repeat([0, 1, 2], [n // 3, n // 3, n - 2 * n // 3])
    np.random.shuffle(regimes)

    vol_map  = {0: 0.15 / SQRT_252, 1: 0.18 / SQRT_252, 2: 0.35 / SQRT_252}
    drift_map = {0: 0.0008, 1: -0.0003, 2: -0.003}
    # Asset-specific vol multipliers: equity, gold, bonds
    vol_mult = np.array([1.0, 0.08, 0.12])
    drift_vec = np.array([1.0, 0.3, 0.5])

    Z = np.random.randn(n, 3)
    rets = np.zeros((n, 3))
    for i, r in enumerate(regimes):
        sigma = vol_map[r] * vol_mult
        mu = drift_map[r] * drift_vec
        rets[i] = (Z[i] @ L.T) * sigma + mu

    prices = pd.DataFrame(rets, index=idx, columns=ASSETS)
    prices = (1 + prices).cumprod() * 100
    return prices


def run_backtest_simulated():
    """Load simulated prices, run the strategy, and display results."""
    prices = _simulate_prices()
    print("Simulated prices (first 5 rows):")
    print(prices.head())
    print(f"\nShape: {prices.shape}")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    return prices


if __name__ == "__main__":
    prices = run_backtest_simulated()
    prices.plot(figsize=(12, 6), title="Simulated Multi-Asset Prices")
    plt.ylabel("Price (base = 100)")
    plt.show()
