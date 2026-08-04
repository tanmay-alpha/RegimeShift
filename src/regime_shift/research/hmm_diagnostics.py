"""Aggregate fixed-configuration HMM restart diagnostics without retuning."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from regime_shift.research.artefacts import write_json


def run_hmm_diagnostics(output_dir: Path) -> None:
    payload = json.loads((output_dir / "hmm_raw_diagnostics.json").read_text(encoding="utf-8"))
    records = payload["diagnostics"]
    rebalances, restart_rows = [], []
    for item in records:
        rebalances.append({key: value for key, value in item.items() if key != "restarts"})
        for restart in item.get("restarts", []): restart_rows.append({"date": item["date"], "selected_regime": item["regime"], **restart})
    rebalance = pd.DataFrame(rebalances); restarts = pd.DataFrame(restart_rows)
    rebalance.to_csv(output_dir / "hmm_rebalance_diagnostics.csv", index=False)
    summary = pd.DataFrame([{"hmm_fits": len(rebalance), "restarts_per_fit": 3, "converged_candidates": int(restarts.get("converged", pd.Series(dtype=bool)).sum()), "fallback_to_non_converged": int((~restarts.get("converged", pd.Series(dtype=bool))).sum()) if not restarts.empty else 0, "mean_iterations": float(restarts["n_iter"].mean()) if "n_iter" in restarts else None, "median_log_likelihood": float(restarts["log_likelihood"].median()) if "log_likelihood" in restarts else None}])
    summary.to_csv(output_dir / "hmm_restart_summary.csv", index=False)
    occupancy = rebalance["regime"].value_counts().rename_axis("state").reset_index(name="rebalance_count")
    occupancy.to_csv(output_dir / "hmm_state_occupancy.csv", index=False)
    probabilities = pd.read_csv(output_dir / "hmm_posterior_probabilities.csv", index_col="date", parse_dates=True)
    confidence = pd.DataFrame({
        "posterior_maximum_probability": probabilities.max(axis=1),
        "posterior_entropy": -(probabilities.clip(lower=1e-12) * probabilities.clip(lower=1e-12).apply(lambda x: __import__("numpy").log(x))).sum(axis=1),
    })
    confidence.to_csv(output_dir / "hmm_posterior_confidence.csv", index_label="date")
    write_json(output_dir / "hmm_diagnostics_metadata.json", {"diagnostic_only": True, "configuration_changed_after_inspection": False, "source": "full RegimeShift causal run HMM restart diagnostics"})
