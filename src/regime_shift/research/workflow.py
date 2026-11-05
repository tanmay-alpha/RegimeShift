"""Offline, JSON-only research experiments shared by the CLI and optional UI.

Nothing in this module imports Streamlit, downloads prices, or places orders.
Evaluation windows attribute one continuous causal run; they do not select a
winner, establish significance, or create an untouched holdout.
"""
from __future__ import annotations

import io
import json
import math
import re
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from regime_shift.config import RegimeShiftConfig
from regime_shift.experiment import sha256_file
from regime_shift.research.evaluation import metrics_table, require_common_index
from regime_shift.research.registry import register_attempt, write_experiment_manifest
from regime_shift.research.spec import (
    COST_UNITS,
    PROTOCOL_VERSION,
    RETURN_UNITS,
    STRATEGIES,
    TIMESTAMP_CONVENTION,
    TURNOVER_UNITS,
)
from regime_shift.research.strategies import ABLATION_TARGETS, run_core_suite
from regime_shift.validation import validate_price_data

SCHEMA_VERSION = 1
DEFAULT_DATASET = "data/cache/validated_market_data.csv"
DEFAULT_RESULTS = Path("results/experiments")
MAX_ROWS = 20_000
CAVEATS = [
    "Research only: no orders, broker connectivity, live trading, or vendor downloads.",
    "Prepared history may contain ex-post price repairs; preparation is separate from strict experiment ingestion and is not point-in-time.",
    "The equity proxy (^NSEI) is an index, not directly tradable; the bond leg is a liquid/cash proxy, not a duration hedge.",
    "Costs are assumed flat one-way bps, not calibrated execution estimates; taxes, impact and capacity are not established.",
    "Evaluation is chronological attribution of a continuous run, not independent refitting or an untouched holdout.",
    "Descriptive comparisons and HMM probabilities do not establish significance, causal explanations, or future performance.",
    "HMM state names are training-derived interpretations; posterior uncertainty is conditional on the fitted model, not calibrated forecast confidence.",
]


class WorkflowError(RuntimeError):
    """An attempted experiment failed; its versioned artifacts remain inspectable."""

    def __init__(self, message: str, output_dir: Path):
        super().__init__(message)
        self.output_dir = output_dir


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: Path
    manifest: dict


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False), encoding="utf-8")


def _object(value: Any, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    return value


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    return value


def _date(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{name} must be YYYY-MM-DD.")
    try:
        parsed = pd.Timestamp(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid calendar date.") from exc
    if pd.isna(parsed):
        raise ValueError(f"{name} is not a valid calendar date.")
    return value


def validate_config(value: Any) -> dict:
    """Validate an allowlisted JSON schema; no imports, expressions or callables."""
    obj = _object(value, "config", {
        "schema_version", "description", "dataset_path", "strategies", "evaluation", "engine",
    })
    if type(obj.get("schema_version", SCHEMA_VERSION)) is not int or obj.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")
    dataset = obj.get("dataset_path", DEFAULT_DATASET)
    if not isinstance(dataset, str) or not dataset or Path(dataset).suffix.lower() != ".csv" or "://" in dataset:
        raise ValueError("dataset_path must be a local canonical-schema CSV path.")
    description = obj.get("description", "Independent offline research experiment")
    if not isinstance(description, str) or len(description) > 2000:
        raise ValueError("description must be a string of at most 2000 characters.")
    strategies = obj.get("strategies", ["regimeshift", "static_6040", "equal_weight"])
    if not isinstance(strategies, list) or not strategies or any(not isinstance(s, str) or s not in STRATEGIES for s in strategies):
        raise ValueError(f"strategies must be a nonempty list drawn from {list(STRATEGIES)}.")
    if len(set(strategies)) != len(strategies):
        raise ValueError("strategies must not contain duplicates.")
    evaluation = _object(obj.get("evaluation"), "evaluation", {"start", "end"})
    start, end = (_date(evaluation.get(k), f"evaluation.{k}") for k in ("start", "end"))
    if start > end:
        raise ValueError("evaluation.start must not exceed evaluation.end.")
    engine = _object(obj.get("engine", {}), "engine", {
        "train_window", "rebalance_frequency", "minimum_training_observations", "transaction_cost_bps",
        "random_seed", "hmm_n_iter", "hmm_n_restarts",
    })
    defaults = {
        "train_window": (252, 60, 2520), "rebalance_frequency": (21, 1, 252),
        "minimum_training_observations": (126, 50, 2520), "random_seed": (42, 0, 2**31 - 1),
        "hmm_n_iter": (200, 1, 1000), "hmm_n_restarts": (3, 1, 10),
    }
    normalized = {k: _integer(engine.get(k, default), f"engine.{k}", low, high)
                  for k, (default, low, high) in defaults.items()}
    if normalized["minimum_training_observations"] > normalized["train_window"]:
        raise ValueError("minimum_training_observations must not exceed train_window.")
    cost = engine.get("transaction_cost_bps", 10.0)
    if type(cost) not in (int, float) or not math.isfinite(cost) or not 0 <= cost <= 1000:
        raise ValueError("transaction_cost_bps must be a finite number in [0, 1000].")
    normalized["transaction_cost_bps"] = float(cost)
    return {"schema_version": SCHEMA_VERSION, "description": description, "dataset_path": dataset,
            "strategies": strategies, "evaluation": {"start": start, "end": end}, "engine": normalized}


def read_config(path: str | Path) -> dict:
    """Read JSON only. Relative dataset paths are relative to the working directory."""
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def build_engine_config(config: dict) -> RegimeShiftConfig:
    settings = config["engine"]
    engine = RegimeShiftConfig(**{k: settings[k] for k in (
        "train_window", "rebalance_frequency", "minimum_training_observations", "transaction_cost_bps",
    )}, random_seed=settings["random_seed"])
    engine.hmm_config.random_state = settings["random_seed"]
    engine.hmm_config.n_iter = settings["hmm_n_iter"]
    engine.hmm_config.n_restarts = settings["hmm_n_restarts"]
    engine.hmm_config.validate()
    engine.validate()
    return engine


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Strict canonical-schema ingestion, with no repairs, sorting or filling."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local CSV not found: {path}")
    frame = pd.read_csv(path, index_col=0, nrows=MAX_ROWS + 1)
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Dataset exceeds the bounded {MAX_ROWS}-row research limit.")
    if set(frame.columns) not in ({"equity", "gold", "bond"}, {"equity", "gold", "bond", "vix"}):
        raise ValueError("CSV columns must be equity, gold, bond and optionally vix, with a date index.")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if frame.empty or frame.index.hasnans or not frame.index.equals(frame.index.normalize()):
        raise ValueError("CSV requires nonempty daily dates without missing dates or intraday timestamps.")
    validate_price_data(frame, required_cols=list(frame.columns), allow_missing=False, forward_fill_limit=0)
    return frame


def _accounting(run) -> pd.DataFrame:
    wealth = (1 + run.net).cumprod()
    gross_wealth = (1 + run.gross).cumprod()
    # Include initial wealth 1 in the high-water mark, including the first allocation cost.
    peak = wealth.cummax().clip(lower=1.0)
    return pd.DataFrame({"gross_return": run.gross, "net_return": run.net,
                         "transaction_cost": run.costs, "turnover": run.turnover,
                         "gross_equity": gross_wealth, "net_equity": wealth,
                         "drawdown": wealth / peak - 1.0, "cumulative_cost_fraction_sum": run.costs.cumsum()})


def _rule(strategy: str, regime: str) -> str:
    if strategy == "static_6040":
        return "Fixed target: equity 0.60, gold 0, bond 0.40; drifts between reference fill dates."
    if strategy == "equal_weight":
        return "Fixed target: equity/gold/bond each 1/3; drifts between reference fill dates."
    if strategy == "min_variance":
        return "Sample-covariance minimum variance; long-only, sum=1, each target weight <=0.80."
    if strategy in ("hmm_fixed", "vol_rule"):
        return ("Frozen regime targets Bull=(.65,.20,.15), Bear=(.35,.30,.35), Crisis=(.10,.30,.60); "
                + ("63-day equity volatility: Bear >20%, Crisis >30%." if strategy == "vol_rule" else "HMM state selects target; no optimizer."))
    prefix = "Bull" if strategy == "no_regime_optimizer" else regime
    descriptions = {
        "Bull": "equity in [.45,.80], gold <=.35, bond <=.45; mean-variance objective",
        "Bear": "equity <=.40, gold+bond >=.60; defensive objective with return reward",
        "Crisis": "equity <=.15, gold >=.25, bond >=.40; minimum-variance objective",
    }
    return (f"{prefix}: {descriptions.get(prefix, 'no recorded signal regime')}; long-only, sum=1, each target <=.80. "
            "Constraints apply to targets, not drifted holdings; this is a rule explanation, not a causal attribution.")


def _explanations(run, reference, ctx) -> pd.DataFrame:
    index = run.index
    # Link held weights to the most recent executed decision, never today's unfilled signal.
    decisions = {pd.Timestamp(d["execution_date"]): d for d in reference.hmm_diagnostics
                 if d.get("execution_date") is not None and not pd.isna(d["execution_date"])}
    vol_decisions = {}
    if run.strategy_id == "vol_rule":
        dates = ctx.raw_features.index
        for pos in sorted(ctx.rebalance_positions):
            if pos + 1 < ctx.min_history or pos + 1 >= len(dates):
                continue
            date = dates[pos]
            vol = float(ctx.asset_returns.loc[:date, ctx.config.equity_col].tail(63).std(ddof=1)) * np.sqrt(252)
            vol_decisions[dates[pos + 1]] = {"date": date, "regime": "Crisis" if vol > .30 else "Bear" if vol > .20 else "Bull"}
        decisions = vol_decisions
    rows, last = [], None
    # Retain the decision before a calendar slice begins.
    for date in ctx.asset_returns.index:
        if date in decisions:
            last = decisions[date]
        if date not in index:
            continue
        regime = "Bull" if run.strategy_id == "no_regime_optimizer" else (last["regime"] if last else "Unknown")
        uses_regime = run.strategy_id in ("regimeshift", "hmm_fixed", "vol_rule", "no_regime_optimizer")
        rows.append({"date": date, "strategy": run.strategy_id,
                     "signal_date_for_held_target": str(last["date"]) if last and uses_regime else None,
                     "regime_for_held_target": regime if uses_regime else "not applicable",
                     "rule_explanation": _rule(run.strategy_id, regime),
                     "probability_source": "reference RegimeShift HMM; context only for other strategies",
                     "weight_semantics": "end-of-day post-trade weights; can drift beyond target constraints"})
    return pd.DataFrame(rows).set_index("date")


def run_experiment(config: Mapping[str, Any] | str | Path, *, results_dir: str | Path = DEFAULT_RESULTS,
                   registry_dir: str | Path = "results/registry") -> ExperimentResult:
    """Run one explicit attempt, preserving a manifest and registry entry on failure.

    The suite always computes RegimeShift and static references to establish its
    official common execution calendar; only explicitly selected outputs are
    published. No strategy is chosen from performance in this function.
    """
    experiment_id = f"v{PROTOCOL_VERSION}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"
    out = Path(results_dir) / experiment_id
    out.mkdir(parents=True, exist_ok=False)
    raw: Any = {}
    normalized: dict = {}
    dataset = Path(DEFAULT_DATASET)
    n_trials = 0
    status, error, summary = "failed", None, {}
    try:
        raw = json.loads(Path(config).read_text(encoding="utf-8")) if isinstance(config, (str, Path)) else dict(config)
        if isinstance(raw, dict) and isinstance(raw.get("dataset_path"), str):
            dataset = Path(raw["dataset_path"])
        normalized = validate_config(raw)
        dataset = Path(normalized["dataset_path"])
        _write_json(out / "config.json", normalized)
        engine = build_engine_config(normalized)
        prices = load_dataset(dataset)
        start, end = (pd.Timestamp(normalized["evaluation"][k]) for k in ("start", "end"))
        if start < prices.index[0] or end > prices.index[-1]:
            raise ValueError("Evaluation boundaries must lie within the supplied dataset calendar span.")
        # No rows after the evaluation endpoint enter feature construction or fitting.
        bounded = prices.loc[:end]
        if len(bounded) // engine.rebalance_frequency > 2000:
            raise ValueError("Configuration exceeds the 2000 scheduled-refit bound; increase rebalance_frequency.")
        variants = [s for s in normalized["strategies"] if s in ABLATION_TARGETS]
        evaluated = list(dict.fromkeys(["regimeshift", *variants, "static_6040", "equal_weight"]))
        n_trials = len(evaluated)
        runs, reference, ctx = run_core_suite(bounded, engine, variants=variants)
        require_common_index(runs.values())
        common = require_common_index([runs[s] for s in normalized["strategies"]])
        if start < common[0]:
            raise ValueError(f"Evaluation starts before the first eligible fill ({common[0].date()}); provide more warmup history or move start.")
        selected = [runs[s].window(start, end) for s in normalized["strategies"]]
        index = require_common_index(selected)
        if len(index) < 2:
            raise ValueError("Evaluation requires at least two common accounting observations.")
        metrics = metrics_table(selected, cash_returns=ctx.cash_returns)
        metrics.to_csv(out / "metrics.csv", index=False)
        _write_json(out / "metrics.json", metrics.to_dict(orient="records"))
        probabilities = reference.regime_probabilities.loc[index].copy()
        probabilities["regime_at_close"] = reference.regime_series.loc[index]
        p = probabilities[["Bull", "Bear", "Crisis"]]
        probabilities["entropy_nats"] = -(p * np.log(p.clip(lower=np.finfo(float).tiny))).sum(axis=1)
        probabilities["max_probability"] = p.max(axis=1)
        probabilities.to_csv(out / "regime_probabilities.csv", index_label="date")
        _write_json(out / "diagnostics.json", reference.hmm_diagnostics)
        _write_json(out / "metadata.json", {
            "caveats": CAVEATS, "timestamp_convention": TIMESTAMP_CONVENTION,
            "return_units": RETURN_UNITS, "cost_units": COST_UNITS, "turnover_units": TURNOVER_UNITS,
            "drawdown_units": "Signed fraction below high-water mark including starting wealth 1; <=0.",
            "evaluation_type": "chronological_attribution", "computed_strategies": evaluated,
            "published_strategies": normalized["strategies"], "dataset_rows": len(prices),
            "engine_rows_through_end": len(bounded), "raw_data_exported": False,
            "probability_source": "Reference RegimeShift HMM, last known signal at each close; not the executed target's signal necessarily.",
            "hmm_fixed_explanation": "Reference HMM signal regime under the same deterministic configuration; not separately instrumented ablation diagnostics.",
            "engine_metadata": reference.config_metadata,
        })
        for run in selected:
            folder = out / run.strategy_id
            folder.mkdir()
            _accounting(run).to_csv(folder / "accounting.csv", index_label="date")
            if run.weights is None:
                raise ValueError(f"{run.strategy_id}: engine did not emit required accounting weights.")
            run.weights.to_csv(folder / "weights.csv", index_label="date")
            _explanations(run, reference, ctx).to_csv(folder / "explanations.csv", index_label="date")
        if not reference.event_log.empty:
            reference.event_log.loc[index].to_csv(out / "reference_events.csv", index_label="date")
        summary = {"strategies": normalized["strategies"], "computed_strategies": evaluated,
                   "start": str(index[0].date()), "end": str(index[-1].date()), "n_obs": len(index),
                   "evaluation_type": "chronological_attribution", "output_dir": str(out)}
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_json(out / "error.json", {"error": error, "status": "failed"})
        if not (out / "config.json").exists():
            _write_json(out / "config.json", raw)
        summary = {"output_dir": str(out), "dataset_path": str(dataset), "evaluation_type": "chronological_attribution"}
    record = register_attempt(experiment_id=experiment_id, status=status, n_trials=n_trials,
                              summary=summary, dataset_path=dataset, registry_dir=registry_dir, error=error)
    _write_json(out / "attempt.json", record)
    write_experiment_manifest(out, experiment_id=experiment_id,
                              description=normalized.get("description", "Failed research attempt"),
                              dataset_path=dataset, config=normalized or (raw if isinstance(raw, dict) else {}),
                              extra={"status": status, "error": error, "summary": summary,
                                     "workflow_schema_version": SCHEMA_VERSION})
    manifest = inspect_experiment(out)
    if status == "failed":
        raise WorkflowError(error or "Experiment failed", out)
    return ExperimentResult(out, manifest)


def inspect_experiment(output_dir: str | Path) -> dict:
    """Read existing results only; never trigger the engine."""
    return json.loads((Path(output_dir) / "manifest.json").read_text(encoding="utf-8"))


def list_experiments(results_dir: str | Path = DEFAULT_RESULTS) -> list[Path]:
    return sorted((p.parent for p in Path(results_dir).glob("*/manifest.json")), reverse=True)


def load_table(output_dir: str | Path, name: str, strategy: str | None = None) -> pd.DataFrame:
    """Read allowlisted generated tables for inspection without evaluating models."""
    root = Path(output_dir)
    if strategy is None:
        if name not in {"metrics", "regime_probabilities", "reference_events"}:
            raise ValueError("Unknown experiment table.")
        path = root / f"{name}.csv"
    else:
        if strategy not in STRATEGIES or name not in {"accounting", "weights", "explanations"}:
            raise ValueError("Unknown strategy or table.")
        path = root / strategy / f"{name}.csv"
    if name == "metrics":
        return pd.read_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _exportable(name: str) -> bool:
    parts = Path(name).parts
    if len(parts) == 1:
        return name in {"config.json", "metrics.csv", "metrics.json", "regime_probabilities.csv",
                        "diagnostics.json", "metadata.json", "reference_events.csv", "attempt.json", "error.json"}
    return len(parts) == 2 and parts[0] in STRATEGIES and parts[1] in {"accounting.csv", "weights.csv", "explanations.csv"}


def export_experiment_zip(output_dir: str | Path, destination: str | Path | None = None) -> bytes:
    """Export only hash-verified generated artifacts, never raw market data.

    Extra files, links, traversal paths and dataset CSVs are excluded even if
    someone adds them to a manifest after the run.
    """
    root = Path(output_dir).resolve()
    manifest = inspect_experiment(root)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(root / "manifest.json", "manifest.json")
        for name, expected in sorted(manifest["outputs"].items()):
            if not _exportable(name) or "\\" in name or ":" in name:
                continue
            path = root / name
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
                raise ValueError(f"Refusing symbolic-link artifact: {name}")
            if root not in path.resolve().parents:
                raise ValueError(f"Artifact escapes experiment directory: {name}")
            if sha256_file(path) != expected:
                raise ValueError(f"Artifact checksum mismatch: {name}")
            archive.write(path, name)
    payload = buffer.getvalue()
    if destination is not None:
        destination = Path(destination)
        if destination.resolve() == root or root in destination.resolve().parents:
            raise ValueError("ZIP destination must be outside the immutable experiment directory.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return payload
