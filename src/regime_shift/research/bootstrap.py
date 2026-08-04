"""Paired moving-block bootstrap uncertainty estimates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_shift.metrics import compute_performance_metrics
from regime_shift.research.artefacts import write_json


def _sample_positions(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
    return np.concatenate([(start + np.arange(block_length)) % n for start in starts])[:n]


def _values(series: np.ndarray) -> dict:
    metrics = compute_performance_metrics(pd.Series(series))
    return {"Annualized mean return": float(np.mean(series) * 252), "CAGR": metrics.cagr, "Volatility": metrics.annualized_volatility, "Sharpe": metrics.sharpe_ratio, "Maximum Drawdown": metrics.maximum_drawdown, "Calmar": metrics.calmar_ratio}


def _interval(samples: np.ndarray) -> tuple[float, float, float]:
    return float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 50)), float(np.nanpercentile(samples, 97.5))


def run_bootstrap(daily: pd.DataFrame, output_dir: Path, samples: int = 2000, block_length: int = 21, seed: int = 42) -> None:
    strategies = ["RegimeShift", "Static 60/40", "Equal Weight"]
    arrays = {name: daily[name].to_numpy(dtype=float) for name in strategies}
    collected = {name: {metric: [] for metric in _values(arrays[name]).keys()} for name in strategies}
    paired = {"RegimeShift - Static 60/40": {"Sharpe": [], "CAGR": [], "Maximum Drawdown": []}, "RegimeShift - Equal Weight": {"Sharpe": [], "CAGR": [], "Maximum Drawdown": []}}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        pos = _sample_positions(len(daily), block_length, rng)
        values = {name: _values(array[pos]) for name, array in arrays.items()}
        for name, result in values.items():
            for metric, value in result.items(): collected[name][metric].append(value)
        for benchmark, label in [("Static 60/40", "RegimeShift - Static 60/40"), ("Equal Weight", "RegimeShift - Equal Weight")]:
            for metric in paired[label]: paired[label][metric].append(values["RegimeShift"][metric] - values[benchmark][metric])
    rows = []
    for name, result in collected.items():
        for metric, values in result.items():
            low, median, high = _interval(np.asarray(values)); rows.append({"Strategy": name, "Metric": metric, "ci_low": low, "median": median, "ci_high": high})
    pd.DataFrame(rows).to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)
    diffs = []
    for comparison, result in paired.items():
        for metric, values in result.items():
            low, median, high = _interval(np.asarray(values)); diffs.append({"Comparison": comparison, "Metric": metric, "ci_low": low, "median": median, "ci_high": high, "zero_in_interval": bool(low <= 0 <= high)})
    pd.DataFrame(diffs).to_csv(output_dir / "bootstrap_benchmark_differences.csv", index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5)); ax.hist(paired["RegimeShift - Static 60/40"]["Sharpe"], bins=45, alpha=.8); ax.axvline(0, color="black", linewidth=1); ax.set_title("Paired bootstrap: RegimeShift minus 60/40 Sharpe"); fig.tight_layout(); fig.savefig(output_dir / "bootstrap_sharpe_distributions.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.hist(paired["RegimeShift - Equal Weight"]["CAGR"], bins=45, alpha=.8); ax.axvline(0, color="black", linewidth=1); ax.set_title("Paired bootstrap: RegimeShift minus Equal Weight CAGR"); fig.tight_layout(); fig.savefig(output_dir / "bootstrap_difference_distributions.png", dpi=180); plt.close(fig)
    write_json(output_dir / "bootstrap_metadata.json", {"method": "paired moving-block bootstrap", "block_length_trading_days": block_length, "samples": samples, "seed": seed, "confidence_level": 0.95, "same_resampled_positions_for_all_strategies": True})
