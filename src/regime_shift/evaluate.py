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


# ─────────────────────────────────────────────────────────────────────────────
# Regime-Specific Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_regime_metrics(
    returns: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    Compute performance metrics broken down by regime.

    This reveals which regimes the strategy profits from vs. loses in.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns indexed by date.
    regimes : pd.Series
        Regime labels indexed by date (must align with returns index).

    Returns
    -------
    pd.DataFrame
        Rows = regime labels, columns = metrics:
        - days: number of days in regime
        - total_return: cumulative return during regime
        - annualized_return: annualized return
        - annualized_vol: annualized volatility
        - sharpe_ratio: return / vol
        - max_drawdown: worst peak-to-trough
        - win_rate: fraction of positive return days
    """
    # Align indices
    common_idx = returns.index.intersection(regimes.index)
    if len(common_idx) == 0:
        logger.warning("No overlapping dates between returns and regimes")
        return pd.DataFrame()

    aligned_returns = returns.loc[common_idx]
    aligned_regimes = regimes.loc[common_idx]

    rows = []
    for regime in aligned_regimes.unique():
        mask = aligned_regimes == regime
        if mask.sum() == 0:
            continue

        regime_rets = aligned_returns[mask]
        n_days = mask.sum()

        # Basic metrics
        total_ret = (1.0 + regime_rets).prod() - 1.0
        ann_ret = (1.0 + total_ret) ** (252.0 / max(n_days, 1)) - 1.0
        ann_vol = regime_rets.std() * np.sqrt(252.0)
        sharpe = ann_ret / (ann_vol + 1e-12) if ann_vol > 0 else 0.0

        # Max drawdown
        cum = (1.0 + regime_rets).cumprod()
        peak = np.maximum.accumulate(cum.values)
        mdd = np.min((cum.values - peak) / (peak + 1e-12))

        # Win rate
        win_rate = (regime_rets > 0).mean()

        rows.append({
            "regime": regime,
            "days": n_days,
            "total_return": float(total_ret),
            "annualized_return": float(ann_ret),
            "annualized_vol": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(mdd),
            "win_rate": float(win_rate),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("regime")

    return df
