"""Fail fast when the committed resume-facing artefacts lose integrity."""

from __future__ import annotations

import hashlib
import json
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


def main() -> int:
    required = ["performance_summary.csv", "daily_results.csv", "execution_timeline.csv", "run_metadata.json", "experiment_manifest.json"]
    missing = [name for name in required if not (RESULTS / name).is_file()]
    if not DATA.is_file() or missing:
        raise SystemExit(f"Missing release inputs: dataset={DATA.is_file()}, results={missing}")
    manifest = json.loads((RESULTS / "experiment_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((RESULTS / "run_metadata.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256_canonical") != _sha256(DATA):
        raise SystemExit("Dataset hash does not match experiment manifest.")
    if metadata.get("dataset_sha256_canonical") != manifest.get("dataset_sha256_canonical"):
        raise SystemExit("Metadata and manifest dataset hashes disagree.")
    if metadata.get("dataset_hash_policy") != manifest.get("dataset_hash_policy"):
        raise SystemExit("Metadata and manifest hash policies disagree.")
    if Path(manifest.get("dataset_path", "")).is_absolute() or "\\" in manifest.get("dataset_path", ""):
        raise SystemExit("Manifest dataset path must be repository-relative and portable.")
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
    stale = [path.name for path in (ROOT / "results").iterdir() if path.is_file() and path.suffix.lower() in {".csv", ".json", ".png"}]
    if stale:
        raise SystemExit(f"Stale official-looking files exist directly under results/: {stale}")
    if "same-close" in (ROOT / "README.md").read_text(encoding="utf-8").lower() and "baseline_legacy" not in (ROOT / "README.md").read_text(encoding="utf-8"):
        raise SystemExit("README mentions legacy execution without isolating it.")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "--cost-scenario base" not in readme or "--transaction-cost-bps 10" in readme:
        raise SystemExit("README must reproduce the official scenario-mode command.")
    daily = pd.read_csv(RESULTS / "daily_results.csv")
    timeline = pd.read_csv(RESULTS / "execution_timeline.csv")
    summary = pd.read_csv(RESULTS / "performance_summary.csv")
    if daily.empty or timeline.empty or summary.empty:
        raise SystemExit("Final results must not be empty.")
    if not len(daily) == len(timeline):
        raise SystemExit("Daily results and execution timeline must align.")
    if summary["n_observations"].nunique() != 1:
        raise SystemExit("All performance rows must share a common evaluation horizon.")
    benchmark_columns = ["static_60_40_net_return", "equal_weight_net_return"]
    if daily[benchmark_columns].isna().any().any():
        raise SystemExit("Benchmark returns contain missing values on the strategy index.")
    if not {"signal_date", "execution_date", "return_date", "rebalance_flag"}.issubset(timeline.columns):
        raise SystemExit("Execution timeline does not expose causal event fields.")
    if (timeline.loc[timeline["rebalance_flag"], "execution_date"] != timeline.loc[timeline["rebalance_flag"], "return_date"]).any():
        raise SystemExit("Execution and return dates must be clearly aligned on close event rows.")
    expected_rows = {
        "RegimeShift Net": ("RegimeShift", "6.02%", "0.400", "35.77%"),
        "Static 60/40 Net": ("Static 60/40", "6.60%", "0.697", "23.39%"),
        "Equal Weight Net": ("Equal Weight", "8.75%", "0.570", "32.96%"),
    }
    for csv_label, (label, cagr, sharpe, drawdown) in expected_rows.items():
        row = summary.loc[summary["Strategy"] == csv_label].iloc[0]
        if f"| {label} | {cagr} | {sharpe} | {drawdown} |" not in readme:
            raise SystemExit(f"README table does not match submission CSV for {label}.")
        if not (round(float(row["CAGR"]) * 100, 2) == float(cagr[:-1]) and round(float(row["Sharpe"]), 3) == float(sharpe)):
            raise SystemExit(f"Submission CSV does not match frozen {label} values.")
    if "0P0001BVE8.BO" in (ROOT / "data" / "README.md").read_text(encoding="utf-8"):
        raise SystemExit("Retired gilt-fund ticker remains in active data documentation.")
    for path in (ROOT / "AUDIT_REPORT.md", ROOT / "CLEANUP_AUDIT.md", ROOT / "CLEANUP_REPORT.md"):
        if path.exists():
            raise SystemExit(f"Stale audit content remains: {path.name}")
    for path in (ROOT / "LICENSE", ROOT / "DATA_LICENSE_NOTICE.md", ROOT / "reports" / "RegimeShift_Quant_Research_Report.pdf"):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Required release file missing or empty: {path}")
    required_research = {"ablation_summary.csv", "ablation_daily_returns.csv", "ablation_turnover.csv", "ablation_metadata.json", "ablation_equity_curves.png", "ablation_drawdowns.png", "ablation_sharpe_turnover.png", "subperiod_summary.csv", "subperiod_weights.csv", "subperiod_metadata.json", "subperiod_sharpe.png", "subperiod_drawdowns.png", "rolling_origin_summary.csv", "rolling_origin_daily_returns.csv", "rolling_origin_metadata.json", "rolling_origin_sharpe.png", "rolling_origin_drawdowns.png", "bootstrap_confidence_intervals.csv", "bootstrap_benchmark_differences.csv", "bootstrap_metadata.json", "bootstrap_sharpe_distributions.png", "bootstrap_difference_distributions.png", "hmm_restart_summary.csv", "hmm_rebalance_diagnostics.csv", "hmm_state_occupancy.csv", "hmm_posterior_confidence.csv", "hmm_diagnostics_metadata.json"}
    missing_research = sorted(name for name in required_research if not (RESEARCH / name).is_file())
    if missing_research:
        raise SystemExit(f"Missing required research artefacts: {missing_research}")
    ablation = pd.read_csv(RESEARCH / "ablation_daily_returns.csv", index_col="date", parse_dates=True)
    if len(ablation) != 3732 or ablation.isna().any().any():
        raise SystemExit("Ablation returns must share the 3,732-date complete index.")
    rolling = json.loads((RESEARCH / "rolling_origin_metadata.json").read_text(encoding="utf-8"))
    if not rolling.get("common_index") or len(rolling.get("folds", {})) != 5:
        raise SystemExit("Rolling-origin metadata is invalid.")
    bootstrap = json.loads((RESEARCH / "bootstrap_metadata.json").read_text(encoding="utf-8"))
    if (bootstrap.get("block_length_trading_days"), bootstrap.get("samples"), bootstrap.get("seed"), bootstrap.get("confidence_level")) != (21, 2000, 42, 0.95):
        raise SystemExit("Bootstrap parameters differ from the frozen protocol.")
    notebook_path = ROOT / "notebooks" / "RegimeShift_Submission.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    codes = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    counts = [cell.get("execution_count") for cell in codes]
    if not counts or any(count is None for count in counts) or counts != list(range(1, len(counts) + 1)):
        raise SystemExit("Notebook execution counts must be sequential and non-null.")
    if any(output.get("output_type") == "error" for cell in codes for output in cell.get("outputs", [])):
        raise SystemExit("Notebook contains error output.")
    notebook_text = notebook_path.read_text(encoding="utf-8")
    if manifest["dataset_sha256_canonical"] not in notebook_text or "possible backward-fill detected" in notebook_text.lower():
        raise SystemExit("Notebook does not reflect the canonical current protocol.")
    print("Release verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
