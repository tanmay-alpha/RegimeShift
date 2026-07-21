"""Evaluation metrics with bootstrap confidence intervals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for performance metrics."""
    name: str
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0

    @classmethod
    def from_returns(cls, returns: pd.Series, name: str = "Strategy") -> MetricsResult:
        """Compute all metrics from a return series."""
        rets = returns.dropna()
        if len(rets) == 0:
            return cls(name=name)

        n = len(rets)
        total_ret = (1.0 + rets).prod() - 1.0
        ann_ret = (1.0 + total_ret) ** (252.0 / max(n, 1)) - 1.0 if n > 0 else 0.0
        ann_vol = rets.std() * np.sqrt(252.0) if n > 1 else 0.0
        sharpe = ann_ret / (ann_vol + 1e-12) if ann_vol > 1e-12 else 0.0

        cum = (1.0 + rets).cumprod()
        peak = np.maximum.accumulate(cum.values)
        mdd = float(np.min((cum.values - peak) / (peak + 1e-12))) if len(cum) > 0 else 0.0

        win_rate = float((rets > 0).mean()) if n > 0 else 0.0

        return cls(
            name=name,
            total_return=total_ret,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            win_rate=win_rate,
        )


def compute_metrics(returns: pd.Series, name: str = "Strategy") -> MetricsResult:
    """
    Compute performance metrics from a return series.

    Args:
        returns: Daily portfolio returns indexed by date
        name: Strategy name for the result

    Returns:
        MetricsResult with total_return, annualized_return, annualized_volatility,
        sharpe_ratio, max_drawdown, win_rate
    """
    return MetricsResult.from_returns(returns, name)

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


def compute_turnover_metrics(
    weights_history: pd.DataFrame,
    regimes: Optional[pd.Series] = None,
) -> dict:
    """
    Compute turnover statistics.

    Args:
        weights_history: DataFrame of daily portfolio weights (n_days, n_assets)
        regimes: Optional Series of regime labels (same index as weights_history)

    Returns:
        Dict with keys:
            - avg_daily_turnover: average daily turnover (fraction)
            - avg_annual_turnover: annualized turnover (fraction * 252)
            - max_single_day_turnover: max daily turnover
            - turnover_by_regime: dict of regime -> avg turnover
            - cost_drag: total costs / absolute total return
    """
    if len(weights_history) < 2:
        return {
            "avg_daily_turnover": 0.0,
            "avg_annual_turnover": 0.0,
            "max_single_day_turnover": 0.0,
            "turnover_by_regime": {},
            "cost_drag": 0.0,
        }

    diffs = weights_history.diff().dropna()
    daily_turnovers = diffs.abs().sum(axis=1) / 2.0

    result: dict = {
        "avg_daily_turnover": float(daily_turnovers.mean()),
        "avg_annual_turnover": float(daily_turnovers.mean() * 252),
        "max_single_day_turnover": float(daily_turnovers.max()),
        "turnover_by_regime": {},
        "cost_drag": 0.0,
    }

    if regimes is not None:
        aligned = pd.DataFrame({"turnover": daily_turnovers, "regime": regimes}).dropna()
        for regime, group in aligned.groupby("regime"):
            result["turnover_by_regime"][regime] = float(group["turnover"].mean())

    return result
