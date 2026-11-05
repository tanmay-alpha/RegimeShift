"""Dependence-aware inference and selection-bias controls.

* Paired moving-block bootstrap of metric differences on identical dates, with
  percentile CIs, centred two-sided p-values and Holm family-wise adjustment.
* Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2012/2014).
* Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de
  Prado & Zhu, 2017).  CSCV measures selection-rank stability among a fixed set
  of trials; it is NOT a substitute for chronological out-of-sample testing.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from regime_shift.research.spec import ANNUALIZATION_FACTOR

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Row-wise metrics on (n_samples, n_days) arrays
# ---------------------------------------------------------------------------

def sharpe_rows(r: np.ndarray, ann: int = ANNUALIZATION_FACTOR) -> np.ndarray:
    sd = r.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r.mean(axis=-1) / sd * np.sqrt(ann)
    return np.where(sd > 1e-12, out, np.nan)


def cagr_rows(r: np.ndarray, ann: int = ANNUALIZATION_FACTOR) -> np.ndarray:
    return np.exp(np.log1p(r).sum(axis=-1) * ann / r.shape[-1]) - 1.0


def max_drawdown_rows(r: np.ndarray) -> np.ndarray:
    wealth = np.cumprod(1.0 + r, axis=-1)
    return np.max(1.0 - wealth / np.maximum(np.maximum.accumulate(wealth, axis=-1), 1.0), axis=-1)


def ann_vol_rows(r: np.ndarray, ann: int = ANNUALIZATION_FACTOR) -> np.ndarray:
    return r.std(axis=-1, ddof=1) * np.sqrt(ann)


METRIC_FUNCS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sharpe": sharpe_rows, "cagr": cagr_rows, "max_drawdown": max_drawdown_rows, "ann_vol": ann_vol_rows,
}


# ---------------------------------------------------------------------------
# Moving-block bootstrap
# ---------------------------------------------------------------------------

def moving_block_indices(n: int, block_len: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Künsch moving-block resampling indices, shape (n_boot, n). Blocks never wrap."""
    if not 1 <= block_len <= n:
        raise ValueError(f"block_len must be in [1, n={n}] (got {block_len}).")
    n_blocks = math.ceil(n / block_len)
    starts = rng.integers(0, n - block_len + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n]


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1].")
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def paired_block_bootstrap(
    returns: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    metrics: Sequence[str] = ("sharpe", "cagr", "max_drawdown"),
    block_len: int = 21,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], np.ndarray]]:
    """Paired moving-block bootstrap of ``metric(a) - metric(b)``.

    ``returns`` must already be aligned (one index, no missing values); the same
    resampled day indices are applied to every column, preserving cross-strategy
    dependence.  p-values use the centred bootstrap: P(|d* - d| >= |d|).
    Holm adjustment is applied across every (pair, metric) row in the call.
    """
    if returns.isna().any().any():
        raise ValueError("Bootstrap input contains NaN; align strategies before testing (never fill).")
    data = returns.to_numpy(dtype=float)
    if not np.all(np.isfinite(data)) or np.any(data <= -1) or len(returns) < 2:
        raise ValueError("Bootstrap requires finite returns > -1 and at least two observations.")
    cols = {c: i for i, c in enumerate(returns.columns)}
    idx = moving_block_indices(len(returns), block_len, n_boot, np.random.default_rng(seed))
    observed = {name: METRIC_FUNCS[name](data.T) for name in metrics}
    boot = {name: np.stack([METRIC_FUNCS[name](data[idx, j]) for j in range(data.shape[1])]) for name in metrics}
    rows, draws = [], {}
    for a, b in pairs:
        for name in metrics:
            d_obs = float(observed[name][cols[a]] - observed[name][cols[b]])
            if not np.isfinite(d_obs):
                raise ValueError(f"Undefined observed {name} for {a} versus {b}.")
            d = boot[name][cols[a]] - boot[name][cols[b]]
            d = d[np.isfinite(d)]
            if len(d) < max(10, int(0.9 * n_boot)):
                raise ValueError(f"Insufficient finite bootstrap draws for {name}: {len(d)}/{n_boot}.")
            draws[(a, b, name)] = d
            lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            p = float((1 + np.count_nonzero(np.abs(d - d_obs) >= abs(d_obs))) / (len(d) + 1))
            rows.append({
                "strategy": a, "benchmark": b, "metric": name, "observed_diff": d_obs,
                "boot_mean_diff": float(d.mean()), "ci_lower": float(lo), "ci_upper": float(hi),
                "p_value": p, "n_valid": int(len(d)), "block_len": block_len, "n_boot": n_boot, "seed": seed,
            })
    table = pd.DataFrame(rows)
    table["p_value_holm"] = holm_adjust(table["p_value"].to_numpy())
    table["significant_holm_5pct"] = table["p_value_holm"] < alpha
    return table, draws


# ---------------------------------------------------------------------------
# Probabilistic / Deflated Sharpe ratio
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period SR > ``sr_benchmark``) given skew/kurtosis of ``returns``."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    sd = r.std(ddof=1)
    if n < 3 or sd <= 1e-12:
        return float("nan")
    sr = r.mean() / sd
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return float("nan")
    return float(sps.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(trial_sharpes: Sequence[float]) -> float:
    """E[max per-period SR] of N independent zero-skill trials with the observed SR variance."""
    srs = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
    n = len(srs)
    if n < 2:
        return 0.0
    sd = srs.std(ddof=1)
    return float(sd * ((1 - EULER_GAMMA) * sps.norm.ppf(1 - 1.0 / n)
                       + EULER_GAMMA * sps.norm.ppf(1 - 1.0 / (n * math.e))))


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes_per_period: Sequence[float]) -> Dict[str, float]:
    """DSR = PSR evaluated at the expected maximum SR across all registered trials.

    ``trial_sharpes_per_period`` must include every trial evaluated on the same
    sample (the selected one too), as non-annualised per-period Sharpe ratios.
    """
    sr0 = expected_max_sharpe(trial_sharpes_per_period)
    return {
        "n_trials": int(np.isfinite(np.asarray(trial_sharpes_per_period, dtype=float)).sum()),
        "sr_threshold_per_period": sr0,
        "sr_threshold_annualised": sr0 * math.sqrt(ANNUALIZATION_FACTOR),
        "psr_vs_zero": probabilistic_sharpe_ratio(returns, 0.0),
        "dsr": probabilistic_sharpe_ratio(returns, sr0),
    }


# ---------------------------------------------------------------------------
# CSCV / Probability of Backtest Overfitting
# ---------------------------------------------------------------------------

def cscv_pbo(returns: pd.DataFrame, n_splits: int = 16,
             metric: Callable[[np.ndarray], np.ndarray] = sharpe_rows) -> Dict:
    """Combinatorially symmetric cross-validation PBO over the columns of ``returns``.

    The T rows are cut into ``n_splits`` contiguous blocks; every half of the
    blocks is used once as in-sample (IS) and its complement as out-of-sample
    (OOS).  For each split, the IS-best trial's OOS relative rank w gives
    logit = ln(w / (1 - w)); PBO = P(logit <= 0).
    """
    if n_splits % 2 or n_splits < 2:
        raise ValueError("n_splits must be an even integer >= 2.")
    data = returns.to_numpy(dtype=float)
    n_trials = data.shape[1]
    if n_trials < 2:
        raise ValueError("CSCV needs at least two trials.")
    if len(data) < 2 * n_splits or not np.all(np.isfinite(data)):
        raise ValueError("CSCV requires finite data and at least two observations per block.")
    blocks = np.array_split(np.arange(len(data)), n_splits)
    logits: List[float] = []
    degradations: List[Tuple[float, float]] = []
    for is_blocks in itertools.combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[i] for i in is_blocks])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in is_blocks])
        is_perf = metric(data[is_idx].T)
        oos_perf = metric(data[oos_idx].T)
        if not np.all(np.isfinite(is_perf)) or not np.all(np.isfinite(oos_perf)):
            raise ValueError("CSCV cannot rank undefined split scores; all trials must have finite scores.")
        best = int(np.nanargmax(is_perf))
        rank = float(sps.rankdata(oos_perf)[best])  # 1 = worst
        w = rank / (n_trials + 1.0)
        logits.append(math.log(w / (1.0 - w)))
        degradations.append((float(is_perf[best]), float(oos_perf[best])))
    logits_arr = np.asarray(logits)
    deg = np.asarray(degradations)
    slope = float(np.polyfit(deg[:, 0], deg[:, 1], 1)[0]) if len(deg) > 2 else float("nan")
    return {
        "pbo": float(np.mean(logits_arr <= 0.0)),
        "n_combinations": len(logits),
        "n_trials": n_trials,
        "n_splits": n_splits,
        "median_logit": float(np.median(logits_arr)),
        "is_oos_slope": slope,
        "prob_oos_loss": float(np.mean(deg[:, 1] < 0.0)),
        "logits": logits_arr,
        "is_oos_pairs": deg,
    }
