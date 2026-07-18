import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ASSETS = ["equity", "gold", "bonds"]
ASSET_LABELS = {t: t for t in ASSETS}  # ticker → display label map


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


def compute_features(returns, tickers, window=20):
    """
    Build rolling-statistic features for regime detection.

    Parameters
    ----------
    returns : DataFrame
        Daily returns indexed by date, one column per ticker.
    tickers : list[str]
        Ordered list of ticker symbols (at least 3 expected).
    window : int
        Rolling window length for statistics.

    Returns
    -------
    DataFrame with columns:
        ret_<ticker>      — annualised rolling mean return (252×)
        vol_<ticker>      — annualised rolling std (√252×)
        corr_eq_gold      — rolling correlation equity ↔ gold
        corr_eq_bond      — rolling correlation equity ↔ bonds
        corr_gold_bond    — rolling correlation gold ↔ bonds

    Total: 6 + 3 = 9 features (reduced from 13 to avoid overfitting).
    """
    tickers = returns.columns.tolist()
    labels = [ASSET_LABELS.get(t, t) for t in tickers]
    features = pd.DataFrame(index=returns.index)

    window = int(window)
    min_p = max(int(window * 0.8), 10)

    for ticker, label in zip(tickers, labels):
        roll = returns[ticker].rolling(window, min_periods=min_p)
        features[f"ret_{label}"]  = roll.mean() * 252
        features[f"vol_{label}"]  = roll.std() * np.sqrt(252)

    if len(tickers) == 3:
        roll_ret = returns[tickers].rolling(window, min_periods=min_p)
        eq_corr = roll_ret.corr(returns[tickers[0]], pairwise=False)
        features["corr_eq_gold"]   = eq_corr[tickers[1]]
        features["corr_eq_bond"]   = eq_corr[tickers[2]]
        gd_corr = roll_ret.corr(returns[tickers[1]], pairwise=False)
        features["corr_gold_bond"] = gd_corr[tickers[2]]

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features


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
