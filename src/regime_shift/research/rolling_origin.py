"""Causal rolling-origin summaries from the frozen causal daily series."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_shift.metrics import compute_performance_metrics
from regime_shift.research.artefacts import plot_drawdowns, write_json


FOLDS = {
    "2013-2015": ("2013-01-01", "2015-12-31"),
    "2016-2018": ("2016-01-01", "2018-12-31"),
    "2019-2021": ("2019-01-01", "2021-12-31"),
    "2022-2024": ("2022-01-01", "2024-12-31"),
    "2025-2026": ("2025-01-01", "2026-07-27"),
}


def run_rolling_origin(daily: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows, selected = [], []
    for fold, (start, end) in FOLDS.items():
        sample = daily.loc[start:end]
        if sample.empty:
            raise ValueError(f"No observations for fold {fold}")
        selected.append(sample.assign(fold=fold))
        for strategy in sample:
            rows.append({"Fold": fold, "Strategy": strategy, **compute_performance_metrics(sample[strategy]).to_dict()})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "rolling_origin_summary.csv", index=False)
    pd.concat(selected).to_csv(output_dir / "rolling_origin_daily_returns.csv", index_label="date")
    pivot = summary.pivot(index="Fold", columns="Strategy", values="Sharpe")
    ax = pivot.plot.bar(figsize=(11, 5), rot=20, title="Rolling-origin fold Sharpe")
    ax.figure.tight_layout(); ax.figure.savefig(output_dir / "rolling_origin_sharpe.png", dpi=180); ax.figure.clf()
    plot_drawdowns(daily, output_dir / "rolling_origin_drawdowns.png", "Causal common-horizon drawdowns")
    write_json(output_dir / "rolling_origin_metadata.json", {
        "folds": FOLDS,
        "causal_truncation": "Each reported fold is sliced from a NEXT_CLOSE causal engine run; decisions at a date use only observations through that date and no fold reports data after its end.",
        "common_index": True,
    })
    return summary
