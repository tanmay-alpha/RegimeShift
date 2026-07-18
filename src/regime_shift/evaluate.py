"""Evaluation metrics with bootstrap confidence intervals."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def bootstrap_metrics(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    block_size: int = 21,
    seed: int = 42,
) -> dict:
    """Block bootstrap for Sharpe, return, and max drawdown.

    Parameters
    ----------
    returns : pd.Series
        Daily returns indexed by date.
    n_bootstrap : int
        Number of bootstrap samples (default 1000).
    block_size : int
        Block length in trading days (default 21 ≈ 1 month).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        "sharpe"       → (median, 2.5th pct, 97.5th pct)
        "ann_return"   → (median, 2.5th pct, 97.5th pct)
        "max_drawdown" → (median, 2.5th pct, 97.5th pct)
    """
    rng = np.random.default_rng(seed)
    rets = returns.dropna().values
    if len(rets) < block_size:
        logger.warning("Not enough returns for bootstrap (%d < %d)", len(rets), block_size)
        return {}

    n = len(rets)
    n_blocks = n // block_size + 1

    sharpes = []
    returns_ann = []
    mdd_list = []

    for _ in range(n_bootstrap):
        blocks = []
        for _ in range(n_blocks):
            start = rng.integers(0, n - block_size + 1)
            blocks.append(rets[start : start + block_size])
        sample = np.concatenate(blocks)[:n]

        ann_ret = (1.0 + sample).prod() ** (252.0 / len(sample)) - 1.0
        ann_vol = sample.std() * np.sqrt(252.0)
        sharpe = ann_ret / (ann_vol + 1e-12)

        cum = (1.0 + sample).cumprod()
        peak = np.maximum.accumulate(cum)
        mdd = np.min((cum - peak) / peak)

        sharpes.append(sharpe)
        returns_ann.append(ann_ret)
        mdd_list.append(mdd)

    def ci(vals, p_low=2.5, p_high=97.5):
        vals = np.array(vals)
        return float(np.median(vals)), float(np.percentile(vals, p_low)), float(np.percentile(vals, p_high))

    return {
        "sharpe": ci(sharpes),
        "ann_return": ci(returns_ann),
        "max_drawdown": ci(mdd_list),
    }


def print_confidence_intervals(result):
    """Print bootstrap confidence intervals from a backtest result."""
    ci = bootstrap_metrics(result["returns"].sum(axis=1))
    if not ci:
        print("Insufficient data for bootstrap CIs.")
        return

    print("Bootstrap confidence intervals (1000 draws, 21-day blocks):")
    for metric, (med, lo, hi) in ci.items():
        print(f"  {metric}: {med:.3f} [{lo:.3f}, {hi:.3f}]")
