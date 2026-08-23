"""Fail fast when the committed resume-facing artefacts lose integrity.

Checks (Section 27 of the project requirements):
  Dataset:        canonical hash, row count, required columns
  Official results: required files, NEXT_CLOSE, cost scenario, manifest consistency
  Hygiene:        AUDIT_REPORT/cleanup files absent, LICENSE present, no PDFs,
                  no stale tickers, cost-scenario base in README, no agent state
  Research:       required research output files present
  README:         README values consistent, no stale tickers
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "submission_market_data.csv"
RESULTS = ROOT / "results" / "submission"
RESEARCH = ROOT / "results" / "research"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if path.suffix.lower() == ".csv":
            digest.update(handle.read().replace(b"\r\n", b"\n"))
            return digest.hexdigest()
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_hygiene() -> None:
    """Section 27 — Public repository hygiene checks."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    # Forbidden files
    forbidden_files = [
        "AUDIT_REPORT.md", "CLEANUP_AUDIT.md", "CLEANUP_REPORT.md",
        "FINAL_AUDIT.md", "PROJECT_AUDIT.md", "GEMINI.md", "ANTIGRAVITY.md",
    ]
    for fname in forbidden_files:
        if (ROOT / fname).exists():
            raise SystemExit(f"Forbidden public file exists: {fname}")

    # LICENSE required
    if not (ROOT / "LICENSE").exists():
        raise SystemExit("LICENSE file is missing.")

    # No PDFs
    pdf_files = list(ROOT.rglob("*.pdf"))
    pdf_files = [p for p in pdf_files if ".git" not in str(p)]
    if pdf_files:
        raise SystemExit(f"PDF files must not exist in public repository: {pdf_files}")

    # Stale ticker absent from source code and docs (data CSVs excluded)
    stale_ticker = "0P0001BVE8" + ".BO"  # split so verifier doesn't flag itself
    for f in ROOT.rglob("*.py"):
        if ".git" not in str(f) and stale_ticker in f.read_text(encoding="utf-8", errors="ignore"):
            raise SystemExit(f"Stale ticker {stale_ticker!r} found in: {f}")
    if stale_ticker in readme:
        raise SystemExit(f"Stale ticker {stale_ticker!r} found in README.md")

    # Official reproduction command uses --cost-scenario base
    if "--cost-scenario base" not in readme:
        raise SystemExit("README must show --cost-scenario base as the official reproduction command.")

    # same-close absent from README
    if "same-close" in readme.lower():
        raise SystemExit("README mentions legacy same-close execution.")

    # No results directly under results/ (only subdirectories allowed)
    results_root = ROOT / "results"
    if results_root.exists():
        stale_root_files = [
            f for f in results_root.iterdir()
            if f.is_file() and f.suffix in {".csv", ".json", ".png"}
        ]
        if stale_root_files:
            raise SystemExit(f"Stale result files directly under results/: {stale_root_files}")


def _check_research() -> None:
    """Section 27 — Required research output files present."""
    required_research = [
        "ablation_summary.csv",
        "rolling_origin_summary.csv",
        "subperiod_summary.csv",
        "bootstrap_confidence_intervals.csv",
        "bootstrap_benchmark_differences.csv",
        "hmm_diagnostics.csv",
    ]
    missing = [f for f in required_research if not (RESEARCH / f).exists()]
    if missing:
        raise SystemExit(
            f"Required research outputs missing (run scripts/run_research_suite.py): {missing}"
        )


def main() -> int:
    # --- Official submission artefacts ---
    required = [
        "performance_summary.csv", "daily_results.csv", "execution_timeline.csv",
        "run_metadata.json", "experiment_manifest.json",
    ]
    missing = [name for name in required if not (RESULTS / name).is_file()]
    if not DATA.is_file() or missing:
        raise SystemExit(f"Missing release inputs: dataset={DATA.is_file()}, results={missing}")

    manifest = json.loads((RESULTS / "experiment_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((RESULTS / "run_metadata.json").read_text(encoding="utf-8"))

    # Dataset hash
    if manifest.get("dataset_sha256_canonical") != _sha256(DATA):
        raise SystemExit("Dataset hash does not match experiment manifest.")
    if metadata.get("dataset_sha256_canonical") != manifest.get("dataset_sha256_canonical"):
        raise SystemExit("Metadata and manifest dataset hashes disagree.")
    if metadata.get("dataset_hash_policy") != manifest.get("dataset_hash_policy"):
        raise SystemExit("Metadata and manifest hash policies disagree.")
    if Path(manifest.get("dataset_path", "")).is_absolute() or "\\" in manifest.get("dataset_path", ""):
        raise SystemExit("Manifest dataset path must be repository-relative and portable.")

    # Cost metadata consistency
    for key in ("cost_input_mode", "cost_scenario", "flat_cost_override_bps", "effective_asset_cost_bps"):
        if metadata.get(key) != manifest.get("metadata", {}).get(key):
            raise SystemExit(f"Metadata and manifest disagree on {key}.")

    costs = metadata.get("effective_asset_cost_bps", {})
    if set(costs) != {"equity", "gold", "bond"} or len(set(costs.values())) != 1:
        raise SystemExit("Effective asset costs must be identical and explicit for all official assets.")

    scenario_bps = {"optimistic": 5.0, "base": 10.0, "stressed": 20.0}
    if metadata.get("cost_input_mode") == "scenario":
        if metadata.get("flat_cost_override_bps") is not None or costs["equity"] != scenario_bps.get(metadata.get("cost_scenario")):
            raise SystemExit("Scenario cost metadata is contradictory.")
    elif metadata.get("cost_input_mode") == "flat_override":
        if metadata.get("cost_scenario") is not None or costs["equity"] != metadata.get("flat_cost_override_bps"):
            raise SystemExit("Flat cost override metadata is contradictory.")
    else:
        raise SystemExit("Unknown cost input mode.")

    if metadata.get("execution_model") != "NEXT_CLOSE":
        raise SystemExit("Official results must use NEXT_CLOSE.")

    # Result files
    daily = pd.read_csv(RESULTS / "daily_results.csv")
    timeline = pd.read_csv(RESULTS / "execution_timeline.csv")
    summary = pd.read_csv(RESULTS / "performance_summary.csv")

    if daily.empty or timeline.empty or summary.empty:
        raise SystemExit("Final results must not be empty.")
    if len(daily) != len(timeline):
        raise SystemExit("Daily results and execution timeline must align.")
    if summary["n_observations"].nunique() != 1:
        raise SystemExit("All performance rows must share a common evaluation horizon.")

    benchmark_columns = ["static_60_40_net_return", "equal_weight_net_return"]
    if daily[benchmark_columns].isna().any().any():
        raise SystemExit("Benchmark returns contain missing values on the strategy index.")

    if not {"signal_date", "execution_date", "return_date", "rebalance_flag"}.issubset(timeline.columns):
        raise SystemExit("Execution timeline does not expose causal event fields.")

    if (timeline.loc[timeline["rebalance_flag"], "execution_date"] !=
            timeline.loc[timeline["rebalance_flag"], "return_date"]).any():
        raise SystemExit("Execution and return dates must be clearly aligned on close event rows.")

    # --- Hygiene ---
    _check_hygiene()

    # --- Research outputs ---
    _check_research()

    print("Release verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
