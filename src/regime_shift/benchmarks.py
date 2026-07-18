"""
Benchmark strategies for comparison against RegimeShift.

Current benchmarks:
- Buy-and-hold (equal-weight)
- 60/40 portfolio (60% equity, 40% bonds)
"""

import numpy as np
import pandas as pd


class BenchmarkResult:
    """Simple container for a benchmark's returns and metadata."""

    def __init__(self, name, returns):
        self.name = name
        self.returns = returns  # pd.Series indexed by date


def buy_and_hold(prices):
    """Equal-weight buy-and-hold across all assets."""
    returns = prices.pct_change().dropna()
    n = len(returns.columns)
    port_ret = returns.mean(axis=1)
    return BenchmarkResult("Buy-and-Hold", port_ret)


def sixty_forty(prices):
    """60/40 portfolio: 60% equity, 40% bonds (no gold)."""
    returns = prices.pct_change().dropna()
    if "equity" in prices.columns and "bonds" in prices.columns:
        port_ret = 0.6 * returns["equity"] + 0.4 * returns["bonds"]
    else:
        # Fallback: first two columns
        port_ret = 0.6 * returns.iloc[:, 0] + 0.4 * returns.iloc[:, 1]
    return BenchmarkResult("60/40", port_ret)


def run_benchmarks(prices, returns=None):
    """Run all benchmarks and return dict of BenchmarkResult."""
    if returns is None:
        returns = prices.pct_change().dropna()
    return {
        "buy_and_hold": buy_and_hold(prices),
        "sixty_forty": sixty_forty(prices),
    }


def compute_sharpe(result, rf_annual=0.0):
    """Annualised Sharpe ratio from a BenchmarkResult."""
    rets = result.returns.dropna()
    if len(rets) < 2:
        return 0.0
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    excess = rets - rf_daily
    std = excess.std()
    if std == 0:
        return 0.0
    return (excess.mean() / std) * np.sqrt(252)


def compute_total_return(result):
    """Total return over the period."""
    rets = result.returns.dropna()
    if len(rets) == 0:
        return 0.0
    return float((1.0 + rets).prod() - 1.0)
