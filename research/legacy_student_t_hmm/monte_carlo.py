"""
monte_carlo.py — Statistical significance testing for RegimeShift.

Methods implemented:
  1. Bootstrap Sharpe test (Politis & Romano 1994 — block bootstrap)
     Tests whether the strategy's Sharpe ratio is significantly > 0
     p-value = fraction of bootstrap samples that beat the real Sharpe

  2. Permutation test on trade returns
     Randomly shuffles trade order N times, measures how often random
     shuffling beats the real cumulative PnL

  3. Monte Carlo equity curves
     Simulates 1000 equity paths by resampling daily returns with replacement
     Shows the distribution of possible outcomes

Reference: Politis & Romano (1994), "The Stationary Bootstrap",
           JASA Vol. 89, No. 428, pp. 1303-1313.
"""

import numpy as np
import pandas as pd
import math
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def bootstrap_sharpe_test(
    returns: pd.Series,
    n_bootstrap: int = 5000,
    block_size: int = 21,
    ann_factor: int = 365,
    risk_free_annual: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Block Bootstrap test for Sharpe Ratio significance.

    Null hypothesis: Strategy's Sharpe ratio is no better than random.

    Method:
      1. Compute real Sharpe ratio from observed returns
      2. Block-bootstrap N samples of same length (block_size ≈ 1 month)
      3. Compute Sharpe for each bootstrap sample
      4. p-value = P(SR_bootstrap >= SR_real)  [fraction of samples >= real]

    Block bootstrap preserves autocorrelation structure of returns,
    which is critical for time series (Politis & Romano 1994).

    Parameters
    ----------
    returns : pd.Series — daily portfolio returns
    n_bootstrap : int — number of bootstrap iterations (default 5000)
    block_size : int — block length in days (default 21 ≈ 1 month)
    ann_factor : int — 365 for crypto
    risk_free_annual : float — annual risk-free rate
    seed : int — random seed

    Returns
    -------
    dict with keys:
        real_sharpe      — observed Sharpe ratio
        bootstrap_sharpes — array of N bootstrap Sharpe ratios
        p_value          — fraction of bootstraps >= real Sharpe
        ci_95            — (low, high) 95% confidence interval for Sharpe
        significant      — True if p_value < 0.05
    """
    from src.regime_shift.stats import sharpe_ratio

    rng  = np.random.default_rng(seed)
    rets = returns.dropna().values
    T    = len(rets)

    if T < block_size * 3:
        logger.warning("Insufficient returns for bootstrap (%d < %d)", T, block_size * 3)
        return {"p_value": 1.0, "significant": False, "real_sharpe": 0.0}

    # Real Sharpe ratio
    real_sr = sharpe_ratio(returns, risk_free_annual, ann_factor)

    # Block bootstrap
    n_blocks  = math.ceil(T / block_size)
    boot_srs  = []
    rf_daily  = (1.0 + risk_free_annual) ** (1.0 / ann_factor) - 1.0

    for _ in range(n_bootstrap):
        blocks = []
        for _ in range(n_blocks):
            start = rng.integers(0, max(1, T - block_size + 1))
            blocks.append(rets[start : start + block_size])
        sample = np.concatenate(blocks)[:T]
        excess = sample - rf_daily
        std    = excess.std()
        sr     = (excess.mean() / std * math.sqrt(ann_factor)) if std > 0 else 0.0
        boot_srs.append(sr)

    boot_srs_arr = np.array(boot_srs)
    p_value      = float((boot_srs_arr >= real_sr).mean())
    ci_95        = (float(np.percentile(boot_srs_arr, 2.5)),
                    float(np.percentile(boot_srs_arr, 97.5)))

    return {
        "real_sharpe"        : float(real_sr),
        "bootstrap_sharpes"  : boot_srs_arr,
        "p_value"            : p_value,
        "ci_95"              : ci_95,
        "bootstrap_mean"     : float(boot_srs_arr.mean()),
        "bootstrap_std"      : float(boot_srs_arr.std()),
        "significant"        : p_value < 0.05,
    }


def permutation_test_pnl(
    trade_pnls: List[float],
    n_permutations: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Permutation test on trade PnL sequence.

    Null hypothesis: The order of trades does not matter (random).

    Method:
      1. Compute real cumulative PnL
      2. Randomly shuffle trade order N times
      3. p-value = fraction of shuffles that achieve >= real cum PnL

    This tests whether the SEQUENCE of winning/losing trades has structure
    (i.e., good trades cluster in good regimes), vs. being random.

    Returns
    -------
    dict with keys:
        real_total_pnl  — sum of all trade PnLs
        p_value         — fraction of shuffles >= real cumulative PnL
        ci_95           — 95% CI of shuffle distribution
        significant     — True if p_value < 0.05
    """
    rng   = np.random.default_rng(seed)
    pnls  = np.array(trade_pnls)
    real_total = pnls.sum()

    shuffle_totals = []
    for _ in range(n_permutations):
        shuffled = pnls.copy()
        rng.shuffle(shuffled)
        shuffle_totals.append(shuffled.sum())

    arr     = np.array(shuffle_totals)
    p_value = float((arr >= real_total).mean())
    ci_95   = (float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)))

    return {
        "real_total_pnl"   : float(real_total),
        "shuffle_totals"   : arr,
        "p_value"          : p_value,
        "ci_95"            : ci_95,
        "significant"      : p_value < 0.05,
    }


def simulate_equity_paths(
    returns: pd.Series,
    initial_capital: float = 1000.0,
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo equity curve simulation via return resampling.

    For each path:
      1. Resample daily returns WITH replacement (stationary bootstrap)
      2. Simulate equity curve from initial_capital

    This shows the distribution of possible equity outcomes
    given the observed return distribution.

    Parameters
    ----------
    returns : pd.Series — daily returns
    initial_capital : float — starting portfolio value
    n_paths : int — number of simulated paths
    seed : int

    Returns
    -------
    pd.DataFrame — shape (T, n_paths), each column = one equity path
    """
    rng  = np.random.default_rng(seed)
    rets = returns.dropna().values
    T    = len(rets)

    paths = np.zeros((T, n_paths))
    for j in range(n_paths):
        sample      = rng.choice(rets, size=T, replace=True)
        equity      = initial_capital * (1 + sample).cumprod()
        paths[:, j] = equity

    return pd.DataFrame(paths, index=returns.dropna().index)


def print_monte_carlo_report(bootstrap_result: dict, perm_result: dict) -> None:
    """Print formatted Monte Carlo significance report."""
    print("\n" + "=" * 55)
    print("  MONTE CARLO SIGNIFICANCE REPORT")
    print("=" * 55)

    # Bootstrap Sharpe test
    print("\n  Block Bootstrap Sharpe Test (Politis & Romano 1994):")
    print(f"    Real Sharpe Ratio : {bootstrap_result['real_sharpe']:.4f}")
    print(f"    Bootstrap Mean SR : {bootstrap_result['bootstrap_mean']:.4f}")
    print(f"    95% CI            : [{bootstrap_result['ci_95'][0]:.4f}, {bootstrap_result['ci_95'][1]:.4f}]")
    print(f"    p-value           : {bootstrap_result['p_value']:.4f}")
    sig = "✓ SIGNIFICANT (p < 0.05)" if bootstrap_result['significant'] else "✗ NOT significant"
    print(f"    Result            : {sig}")

    # Permutation test
    print("\n  Permutation Test on Trade PnL Sequence:")
    print(f"    Real Total PnL    : ${perm_result['real_total_pnl']:.2f}")
    print(f"    95% CI (shuffles) : [${perm_result['ci_95'][0]:.2f}, ${perm_result['ci_95'][1]:.2f}]")
    print(f"    p-value           : {perm_result['p_value']:.4f}")
    sig2 = "✓ SIGNIFICANT (p < 0.05)" if perm_result['significant'] else "✗ NOT significant"
    print(f"    Result            : {sig2}")

    print("=" * 55 + "\n")
