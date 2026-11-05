"""Run the fixed risk_lab_v1 experiment on canonical local data, without downloads."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Always prefer this checkout to an installed package or another worktree.
sys.path.insert(0, str(ROOT / "src"))

from regime_shift.config import RegimeShiftConfig  # noqa: E402
from regime_shift.data import load_market_data_csv  # noqa: E402
from regime_shift.execution import ExecutionCostModel  # noqa: E402
from regime_shift.experiment import sha256_file  # noqa: E402
from regime_shift.research.evaluation import metrics_table, require_common_index  # noqa: E402
from regime_shift.research.registry import register_attempt, write_experiment_manifest  # noqa: E402
from regime_shift.research.risk_lab import (  # noqa: E402
    COST_BPS,
    HISTORICAL_WINDOWS,
    RiskLabConfig,
    estimate_covariance,
    historical_replay,
    hypothetical_stress,
    risk_contributions,
    risk_lab_targets,
    trade_deltas,
)
from regime_shift.research.spec import TIMESTAMP_CONVENTION  # noqa: E402
from regime_shift.research.strategies import build_context, run_target_strategy  # noqa: E402

EXPERIMENT_ID = "risk_lab_v1"
DATASET = ROOT / "data" / "submission_market_data.csv"


def run_experiment(out: Path, attempts: list, settings: RiskLabConfig) -> dict:
    """Write only new output files; retain every failed trial alongside successes."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to mix risk lab outputs with existing files: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = RegimeShiftConfig()
    prices = load_market_data_csv(str(DATASET), config=config)
    prices = prices[config.core_assets]  # frozen canonical feature set: no optional VIX
    accounting, holdings, trades, metrics, history, hypothetical, contributions = [], [], [], [], [], [], []
    all_runs = []
    for bps in COST_BPS:
        config.execution_cost_model = ExecutionCostModel.flat(bps, config.core_assets,
                                                             name=f"assumed_{bps}bps")
        ctx = build_context(prices, config)
        runs = []
        for sid, target in risk_lab_targets(ctx, settings).items():
            attempt = {"strategy": sid, "cost_bps": bps, "status": "started"}
            attempts.append(attempt)
            try:
                signal_contributions = []
                if sid == "equal_risk_contribution":
                    erc_target = target

                    def target(date, features, returns, previous):
                        w = erc_target(date, features, returns, previous)
                        pc = config.portfolio_config
                        cov = estimate_covariance(returns[ctx.assets].tail(pc.estimation_lookback),
                                                  pc.covariance_ridge)
                        rc = risk_contributions(w, cov)
                        signal_contributions.append({
                            "signal_date": str(date.date()), "cost_bps": bps,
                            "has_next_close": date < ctx.asset_returns.index[-1],
                            **{f"{a}_risk_share": float(rc[i]) for i, a in enumerate(ctx.assets)},
                            "max_budget_error": float(np.max(np.abs(rc - 1 / len(rc))))})
                        return w

                run = run_target_strategy(ctx, sid, target)
                delta = trade_deltas(run, ctx.asset_returns)
                if not np.allclose(run.costs, delta.abs().sum(axis=1) * bps / 10000,
                                   atol=1e-12, rtol=0):
                    raise ValueError("Assumed costs do not reconcile to genuine trade deltas.")
                if not np.isclose(run.turnover.iloc[0], 1) or not np.isclose(run.costs.iloc[0], bps / 10000):
                    raise ValueError("Missing initial cash allocation cost.")
                require_common_index([*all_runs, *runs, run])
                tags = {"strategy": sid, "cost_bps": bps}
                accounting.append(pd.DataFrame({"gross": run.gross, "net": run.net,
                                                "cost": run.costs, "turnover": run.turnover})
                                  .rename_axis("date").reset_index().assign(**tags))
                holdings.append(run.weights.rename_axis("date").reset_index().assign(**tags))
                trades.append(delta.rename_axis("date").reset_index().assign(**tags))
                metrics.append(metrics_table([run], ctx.cash_returns).assign(cost_bps=bps))
                history.append(historical_replay(run, ctx.cash_returns).assign(cost_bps=bps))
                pc = config.portfolio_config
                cov = estimate_covariance(ctx.asset_returns[ctx.assets].tail(pc.estimation_lookback),
                                          pc.covariance_ridge)
                hypothetical.append(hypothetical_stress(
                    run.weights.iloc[-1].to_numpy(), cov, [-0.20, -0.10, -0.02])
                    .assign(**tags, snapshot_date=str(run.index[-1].date())))
                runs.append(run)
                contributions.extend(signal_contributions)
                attempt.update(status="completed", n_obs=len(run.index),
                               first_fill=str(run.index[0].date()))
            except Exception as exc:
                attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        all_runs.extend(runs)
    require_common_index(all_runs)
    for name, frames in (("daily_accounting", accounting), ("weights", holdings),
                         ("trade_deltas", trades), ("metrics", metrics),
                         ("historical_replay", history), ("hypothetical_stress", hypothetical)):
        pd.concat(frames, ignore_index=True).to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame(contributions).to_csv(out / "risk_contributions.csv", index=False)
    return {"data_diagnostics": prices.attrs.get("data_diagnostics", {}),
            "common_start": str(all_runs[0].index[0].date()),
            "common_end": str(all_runs[0].index[-1].date()),
            "n_obs": len(all_runs[0].index)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "experiments" / EXPERIMENT_ID)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    attempts = []
    settings = RiskLabConfig()
    error, summary = None, {}
    # An attempt may be registered even if setup fails before the first trial.
    owned_output = not out.exists() or (out.is_dir() and not any(out.iterdir()))
    try:
        summary = run_experiment(out, attempts, settings)
        if any(a["status"] != "completed" for a in attempts):
            raise RuntimeError("Some strategy/cost trials failed; inspect trials.json.")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    status = "failed" if error else "completed"
    summary.update(planned_trials=28, attempted_trials=len(attempts),
                   completed_trials=sum(a["status"] == "completed" for a in attempts),
                   failed_trials=sum(a["status"] == "failed" for a in attempts),
                   candidate_definitions=7, novel_variants=4, cost_scenarios=4,
                   selection_rule="none; report every prespecified variant and baseline")
    try:
        if owned_output:
            out.mkdir(parents=True, exist_ok=True)
            (out / "trials.json").write_text(json.dumps(
                {"status": status, "error": error, "summary": summary, "trials": attempts},
                indent=2), encoding="utf-8")
            write_experiment_manifest(
                out, experiment_id=EXPERIMENT_ID, description=__doc__.splitlines()[0],
                dataset_path=DATASET,
                config={"risk_lab": asdict(settings), "engine": asdict(RegimeShiftConfig()),
                        "cost_bps": COST_BPS, "timestamp_convention": TIMESTAMP_CONVENTION,
                        "historical_windows": HISTORICAL_WINDOWS,
                        "hypothetical": {"asset_shocks": [-0.20, -0.10, -0.02],
                                         "correlation": 0.90, "correlation_blend": 0.5,
                                         "concentration_transfer": 0.25},
                        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in (
                            Path(__file__).resolve(), ROOT / "src/regime_shift/research/risk_lab.py")}},
                extra={"status": status, "error": error, "trial_summary": summary})
    except Exception as exc:
        status = "failed"
        error = f"{error or ''}; manifest failure: {type(exc).__name__}: {exc}"
    # Registry writing is independent of output setup/manifest success.
    register_attempt(experiment_id=EXPERIMENT_ID, status=status, n_trials=len(attempts),
                     summary={**summary, "output_dir": str(out)}, dataset_path=DATASET,
                     registry_dir=ROOT / "results" / "registry", error=error)
    print(json.dumps({"status": status, "error": error, **summary}, indent=2))
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
