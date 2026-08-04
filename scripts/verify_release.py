"""Fail fast when the committed resume-facing artefacts lose integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "submission_market_data.csv"
RESULTS = ROOT / "results" / "submission"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    required = ["performance_summary.csv", "daily_results.csv", "execution_timeline.csv", "run_metadata.json", "experiment_manifest.json"]
    missing = [name for name in required if not (RESULTS / name).is_file()]
    if not DATA.is_file() or missing:
        raise SystemExit(f"Missing release inputs: dataset={DATA.is_file()}, results={missing}")
    manifest = json.loads((RESULTS / "experiment_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((RESULTS / "run_metadata.json").read_text(encoding="utf-8"))
    if manifest["dataset_sha256"] != _sha256(DATA):
        raise SystemExit("Dataset hash does not match experiment manifest.")
    if metadata.get("execution_model") != "NEXT_CLOSE":
        raise SystemExit("Official results must use NEXT_CLOSE.")
    if "same-close" in (ROOT / "README.md").read_text(encoding="utf-8").lower() and "baseline_legacy" not in (ROOT / "README.md").read_text(encoding="utf-8"):
        raise SystemExit("README mentions legacy execution without isolating it.")
    daily = pd.read_csv(RESULTS / "daily_results.csv")
    timeline = pd.read_csv(RESULTS / "execution_timeline.csv")
    summary = pd.read_csv(RESULTS / "performance_summary.csv")
    if daily.empty or timeline.empty or summary.empty:
        raise SystemExit("Final results must not be empty.")
    if not len(daily) == len(timeline):
        raise SystemExit("Daily results and execution timeline must align.")
    if not {"signal_date", "execution_date", "return_date", "rebalance_flag"}.issubset(timeline.columns):
        raise SystemExit("Execution timeline does not expose causal event fields.")
    if (timeline.loc[timeline["rebalance_flag"], "execution_date"] != timeline.loc[timeline["rebalance_flag"], "return_date"]).any():
        raise SystemExit("Execution and return dates must be clearly aligned on close event rows.")
    print("Release verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
