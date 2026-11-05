"""Run the frozen regime_intelligence_v1 experiment on an existing local CSV."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import regime_shift
from regime_shift.config import RegimeShiftConfig
from regime_shift.data import load_market_data_csv
from regime_shift.execution import ExecutionCostModel
from regime_shift.experiment import sha256_file
from regime_shift.research.evaluation import metrics_table, net_return_frame, require_common_index
from regime_shift.research.regime_intelligence import (
    BASELINE_CRISIS_VOL,
    BASELINE_HIGH_VOL,
    BASELINE_WINDOW,
    ENTROPY_ALERT_THRESHOLD,
    EXPERIMENT_ID,
    MAX_POSTERIOR_ALERT_THRESHOLD,
    MAX_PRICE_ROWS,
    MAX_SIGNAL_CLOSES,
    NEGATIVE_CONTROL_SEED,
    STRATEGY_IDS,
    run_regime_intelligence,
)
from regime_shift.research.registry import register_attempt, write_experiment_manifest
from regime_shift.research.spec import TIMESTAMP_CONVENTION
from regime_shift.research.strategies import REGIME_FIXED_WEIGHTS, build_context

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/submission_market_data.csv",
                        help="Existing offline CSV; relative paths resolve against this worktree.")
    parser.add_argument("--cost-scenario", default="base", choices=("optimistic", "base", "stressed"))
    return parser.parse_args(argv)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def _register_failure(dataset: Path, registry: Path, summary: dict, error: str, n_trials: int) -> None:
    """Preserve failed attempts, including absent datasets with a null hash."""
    register_attempt(experiment_id=EXPERIMENT_ID, status="failed", n_trials=n_trials,
                     summary=summary, dataset_path=dataset, registry_dir=registry, error=error)


def _write_results(result, ctx, out: Path) -> pd.DataFrame:
    runs = list(result.runs.values())
    require_common_index(runs)
    summary = metrics_table(runs, ctx.cash_returns)
    summary["evaluation_type"] = "full_sample"
    summary.to_csv(out / "strategy_metrics.csv", index=False)
    net_return_frame(runs).to_csv(out / "net_returns.csv", index_label="date")
    accounting = {}
    for strategy_id, run in result.runs.items():
        frame = pd.DataFrame({"gross": run.gross, "net": run.net, "cost": run.costs,
                              "turnover": run.turnover})
        if run.weights is not None:
            frame = frame.join(run.weights.add_prefix("end_of_day_post_trade_"))
        accounting[strategy_id] = frame
    pd.concat(accounting, names=["strategy", "date"]).reset_index().to_csv(
        out / "daily_accounting.csv", index=False,
    )
    for name in ("signals", "diagnostics", "transitions", "persistence", "stability",
                 "feature_means", "spells", "restarts"):
        getattr(result, name).to_csv(out / f"{name}.csv", index=False)
    metrics = summary.set_index("strategy")
    comparisons = []
    for strategy in STRATEGY_IDS:
        for reference in ("hmm_fixed", "static_6040", "equal_weight"):
            if strategy == reference:
                continue
            comparisons.append({
                "strategy": strategy, "reference": reference, "n_obs": len(runs[0].index),
                "start": str(runs[0].index[0].date()), "end": str(runs[0].index[-1].date()),
                "evaluation_type": "full_sample_descriptive_difference",
                **{f"delta_{metric}": metrics.loc[strategy, metric] - metrics.loc[reference, metric]
                   for metric in ("cagr", "ann_vol", "sharpe", "sharpe_vs_cash", "max_drawdown",
                                  "ann_turnover", "ann_cost", "cost_drag")},
            })
    pd.DataFrame(comparisons).to_csv(out / "same_index_comparisons.csv", index=False)
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    # Shared provenance helpers use cwd; keep them anchored to this worktree.
    os.chdir(ROOT)
    dataset = Path(args.data_path).resolve()
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    out = ROOT / "results" / "experiments" / EXPERIMENT_ID / attempt_id
    registry = ROOT / "results" / "registry"
    started = []
    output_created = False
    config = RegimeShiftConfig()
    config.cost_scenario = args.cost_scenario
    config.execution_cost_model = ExecutionCostModel.scenario(args.cost_scenario, config.core_assets)
    specification = {
        "regimeshift_config": asdict(config), "strategies": list(STRATEGY_IDS),
        "regime_fixed_weights": REGIME_FIXED_WEIGHTS,
        "uncertainty_alerts_exploratory": {
            "normalized_entropy_gte": ENTROPY_ALERT_THRESHOLD,
            "maximum_posterior_lte": MAX_POSTERIOR_ALERT_THRESHOLD, "combination": "OR",
        },
        "momentum_volatility": {"window": BASELINE_WINDOW, "high_vol_gt": BASELINE_HIGH_VOL,
                                "crisis_vol_gt": BASELINE_CRISIS_VOL, "bear_momentum_lte": 0.0},
        "negative_control": {"seed": NEGATIVE_CONTROL_SEED, "generator": "PCG64",
                             "assignment": "IID uniform Bull/Bear/Crisis at eligible signal closes"},
        "bounds": {"max_price_rows": MAX_PRICE_ROWS, "max_signal_closes": MAX_SIGNAL_CLOSES,
                   "strategy_trials": len(STRATEGY_IDS), "parameter_search": False},
        "vix_policy": "omit, matching the existing research suite",
        "timestamp_convention": TIMESTAMP_CONVENTION,
    }
    attempt_summary = {"attempt_id": attempt_id, "output_dir": out.relative_to(ROOT).as_posix(),
                       "dataset_path": dataset.as_posix(), "cost_scenario": args.cost_scenario,
                       "planned_strategies": list(STRATEGY_IDS), "started_strategies": started}
    try:
        out.mkdir(parents=True, exist_ok=False)
        output_created = True
        _write_json(out / "attempt.json", {**attempt_summary, "specification": specification})
        if Path(regime_shift.__file__).resolve().parent != ROOT / "src" / "regime_shift":
            raise RuntimeError("Set PYTHONPATH to this worktree's src; imported regime_shift belongs elsewhere.")
        # Absolute input avoids the shared loader's parent-directory fallback.
        prices = load_market_data_csv(path=str(dataset), config=config)
        data_diagnostics = dict(prices.attrs.get("data_diagnostics", {}))
        if config.vix_col in prices.columns:
            prices = prices.drop(columns=[config.vix_col])
        ctx = build_context(prices, config)
        result = run_regime_intelligence(ctx, on_strategy_start=started.append)
        summary = _write_results(result, ctx, out)
        attempt_summary.update({
            "n_obs": len(result.runs["hmm_fixed"].index),
            "n_hmm_signal_closes": len(result.diagnostics),
            "start": str(result.runs["hmm_fixed"].index[0].date()),
            "end": str(result.runs["hmm_fixed"].index[-1].date()),
        })
        _write_json(out / "attempt.json", {**attempt_summary, "status": "completed",
                                          "specification": specification})
        write_experiment_manifest(
            out, experiment_id=EXPERIMENT_ID, description=__doc__, dataset_path=dataset,
            config=specification,
            extra={"attempt_id": attempt_id, "evaluation_type": "full_sample",
                   "data_diagnostics": data_diagnostics,
                   "comparison": "Identical engine, eligibility, date index and per-asset costs; no filling.",
                   "posterior_interpretation": "Latent state at signal close; not a future crash probability.",
                   "source_sha256": {
                       path.relative_to(ROOT).as_posix(): sha256_file(path)
                       for path in (Path(__file__).resolve(), ROOT / "src" / "regime_shift" / "research" / "regime_intelligence.py")
                   }, **attempt_summary},
        )
        register_attempt(experiment_id=EXPERIMENT_ID, status="completed", n_trials=len(started),
                         summary=attempt_summary, dataset_path=dataset, registry_dir=registry)
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        failure = {**attempt_summary, "status": "failed", "error": error,
                   "n_trials": len(started), "specification": specification}
        try:
            if output_created:
                (out / "manifest.json").unlink(missing_ok=True)
                _write_json(out / "failure.json", failure)
                _write_json(out / "attempt.json", failure)
            else:
                print("Attempt output directory could not be created; existing files were preserved.", file=sys.stderr)
        except OSError as output_error:
            print(f"Could not write failure output: {output_error}", file=sys.stderr)
        try:
            _register_failure(dataset, registry, attempt_summary, error, len(started))
        except Exception as registration_error:
            print(f"Could not register failed attempt: {registration_error}", file=sys.stderr)
        print(f"{error}\nFailed attempt: {out}", file=sys.stderr)
        return 1
    print(summary.to_string(index=False))
    print(f"Completed attempt: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
