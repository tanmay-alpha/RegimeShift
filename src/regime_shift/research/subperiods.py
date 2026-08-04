"""Chronological retrospective subperiod summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_shift.metrics import compute_performance_metrics
from regime_shift.research.artefacts import plot_drawdowns, write_json


PERIODS = {"Development": ("2010-10-05", "2018-12-31"), "Validation": ("2019-01-01", "2021-12-31"), "Retrospective evaluation": ("2022-01-01", "2026-07-27")}


def run_subperiods(daily: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for period, (start, end) in PERIODS.items():
        sample = daily.loc[start:end]
        for strategy in sample:
            metrics = compute_performance_metrics(sample[strategy]).to_dict()
            rows.append({"Period": period, "Strategy": strategy, **metrics})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "subperiod_summary.csv", index=False)
    weights = pd.read_csv(output_dir / "ablation_weights.csv", parse_dates=["date"])
    weight_rows = []
    for period, (start, end) in PERIODS.items():
        sample = weights.loc[(weights["date"] >= start) & (weights["date"] <= end)]
        grouped = sample.groupby(["Strategy", "asset"], as_index=False)["weight"].mean()
        grouped.insert(0, "Period", period); grouped.rename(columns={"weight": "average_weight"}, inplace=True)
        weight_rows.append(grouped)
    pd.concat(weight_rows, ignore_index=True).to_csv(output_dir / "subperiod_weights.csv", index=False)
    pivot = summary.pivot(index="Strategy", columns="Period", values="Sharpe")
    ax = pivot.plot.bar(figsize=(10, 5), rot=30, title="Subperiod Sharpe"); ax.figure.tight_layout(); ax.figure.savefig(output_dir / "subperiod_sharpe.png", dpi=180); ax.figure.clf()
    plot_drawdowns(daily, output_dir / "subperiod_drawdowns.png", "Full-horizon drawdowns by strategy")
    write_json(output_dir / "subperiod_metadata.json", {"periods": PERIODS, "retrospective_disclosure": "This is a retrospective evaluation period, not a pristine holdout, because the complete historical sample had been inspected during earlier development.", "n_observations": int(len(daily))})
    return summary
