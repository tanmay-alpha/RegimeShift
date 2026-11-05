"""Offline, append-only freeze/evaluate protocol; never a prospective certificate."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from regime_shift.backtest import Signal, _cost_model, run_benchmark, simulate_lagged_execution
from regime_shift.benchmarks import equal_weight_weights, static_60_40_weights
from regime_shift.config import RegimeShiftConfig
from regime_shift.execution import AssetCost, ExecutionCostModel, ExecutionModel
from regime_shift.experiment import sha256_file
from regime_shift.portfolio import optimize_portfolio
from regime_shift.research.evaluation import StrategyRun, metrics_table, require_common_index
from regime_shift.research.registry import environment, register_attempt
from regime_shift.research.spec import PROTOCOL_VERSION, TIMESTAMP_CONVENTION
from regime_shift.research.strategies import build_context, hmm_regime

FORWARD_VERSION = 1
ROOT = Path(__file__).resolve().parents[3]
TRAINING_POLICY = {
    "name": "rolling_refit_at_each_signal",
    "initial_signal": "cutoff_close; all strategies start from cash",
    "subsequent_signals": "every rebalance_frequency observed sessions after cutoff",
    "training": "historical prefix plus accepted incremental observations through signal close only",
    "window": "frozen train_window; scaler and HMM refitted on that trailing window",
    "execution": TIMESTAMP_CONVENTION,
    "baselines": "static_6040 and equal_weight; same first fill and RegimeShift fill dates",
    "selection": "none; no tuning or strategy selection during evaluation",
}


class ForwardIntegrityError(ValueError):
    """A frozen artifact, chronology or input no longer satisfies the protocol."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=lambda obj: obj.value if isinstance(obj, Enum) else str(obj))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_json(value) + "\n")


def _timestamp(value=None) -> pd.Timestamp:
    stamp = pd.Timestamp(value if value is not None else datetime.now(timezone.utc))
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ForwardIntegrityError("observed_at must be an explicit timezone-aware timestamp.")
    return stamp.tz_convert("UTC")


def _date(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ForwardIntegrityError("Trading dates must be timezone-naive midnight dates.")
    return stamp


def _source_hashes(root: Path) -> dict[str, str]:
    paths = list((root / "src" / "regime_shift").rglob("*.py"))
    paths.append(root / "scripts" / "forward_evaluation.py")
    paths.extend(p for p in (root / "pyproject.toml", root / "requirements.txt") if p.is_file())
    if len(paths) < 2 or any(not p.is_file() for p in paths):
        raise ForwardIntegrityError("Complete source checkout and forward command are required.")
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(paths)}


def _restore(template, value):
    if is_dataclass(template):
        expected = {field.name for field in fields(template)}
        if set(value) != expected:
            raise ForwardIntegrityError("Frozen configuration fields differ from this implementation.")
        return type(template)(**{key: _restore(getattr(template, key), val) for key, val in value.items()})
    if isinstance(template, Enum):
        return type(template)(value)
    return value


def _config(value: Mapping[str, Any]) -> RegimeShiftConfig:
    payload = copy.deepcopy(dict(value))
    costs = payload.get("execution_cost_model")
    payload["execution_cost_model"] = None
    config = _restore(RegimeShiftConfig(), payload)
    if costs is not None:
        config.execution_cost_model = ExecutionCostModel(
            name=costs["name"], market_impact_modeled=costs["market_impact_modeled"],
            assets={asset: AssetCost(**entry) for asset, entry in costs["assets"].items()},
        )
    config.validate()
    if config.execution_model is not ExecutionModel.NEXT_CLOSE:
        raise ForwardIntegrityError("Only NEXT_CLOSE is available for local close-only CSVs.")
    if config.minimum_training_observations < 1 or config.rebalance_frequency < 1:
        raise ForwardIntegrityError("Training minimum and rebalance frequency must be positive.")
    if config.train_window < max(config.minimum_training_observations,
                                 config.hmm_config.minimum_training_observations):
        raise ForwardIntegrityError("train_window is shorter than the frozen training minimum.")
    if config.annualization_factor != 252:
        raise ForwardIntegrityError("The shared research metrics require annualization_factor=252.")
    return config


def _read_csv(path: Path, config: RegimeShiftConfig) -> pd.DataFrame:
    # Deliberately do not use the legacy cleaning loader: sorting, duplicate removal,
    # spike repair and filling would conceal changes in the user-supplied evidence.
    frame = pd.read_csv(path, index_col=0)
    try:
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    except (ValueError, TypeError) as exc:
        raise ForwardIntegrityError("CSV index must contain valid trading dates.") from exc
    index = frame.index
    if (not len(index) or index.hasnans or index.tz is not None or index.has_duplicates
            or not index.is_monotonic_increasing or not index.equals(index.normalize())):
        raise ForwardIntegrityError("CSV dates must be sorted, unique timezone-naive midnight dates; never repaired.")
    if not set(config.core_assets).issubset(frame.columns) or frame.columns.has_duplicates:
        raise ForwardIntegrityError("CSV must have unique columns including all frozen core assets.")
    allowed = set(config.core_assets + [config.vix_col])
    if set(frame.columns) - allowed:
        raise ForwardIntegrityError("CSV contains columns outside the frozen price schema.")
    try:
        frame = frame.astype(float)
    except (ValueError, TypeError) as exc:
        raise ForwardIntegrityError("CSV prices must be numeric.") from exc
    if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
        raise ForwardIntegrityError("CSV prices must be finite and positive; no filling or cleaning is performed.")
    return frame


def _row_hashes(frame: pd.DataFrame) -> dict[str, str]:
    columns = list(frame.columns)
    return {str(date.date()): _digest({"columns": columns, "values": row.tolist()})
            for date, row in frame.iterrows()}


def _verify_input(record: Mapping, config: RegimeShiftConfig) -> pd.DataFrame:
    path = Path(record["path"])
    if sha256_file(path) != record["sha256_canonical"]:
        raise ForwardIntegrityError(f"Previously observed input was rewritten: {path}")
    frame = _read_csv(path, config)
    if _row_hashes(frame) != record["row_hashes"]:
        raise ForwardIntegrityError(f"Previously observed rows changed: {path}")
    return frame


def _input_record(path: Path, frame: pd.DataFrame) -> dict:
    return {"path": str(path.resolve()), "sha256_canonical": sha256_file(path),
            "row_hashes": _row_hashes(frame), "start": str(frame.index[0].date()),
            "end": str(frame.index[-1].date()), "n_rows": len(frame)}


@contextmanager
def _lock(directory: Path):
    path = directory / ".forward.lock"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ForwardIntegrityError("Another attempt holds the experiment lock; never overwrite it.") from exc
    try:
        handle.write(str(uuid.uuid4()))
        handle.close()
        yield
    finally:
        path.unlink()


def _registry(directory: Path, *, status: str, phase: str, dataset_path: Path,
              summary: Mapping, error=None) -> dict:
    return register_attempt(experiment_id=directory.name, status=status, n_trials=summary.get("n_trials", 0),
                            summary={"phase": phase, **summary}, dataset_path=dataset_path,
                            registry_dir=directory.parent / "registry", error=error)


def freeze_experiment(experiment_dir: str | Path, *, dataset_path: str | Path,
                      cutoff, config: RegimeShiftConfig | Mapping | str | Path | None = None, observed_at=None,
                      release_at=None, source_root: str | Path = ROOT) -> dict:
    """Create a new freeze, exclusively. An existing directory is never reused.

    Timestamps are local assertions, not independent proof of prior registration.
    Keep the returned freeze_sha256 outside this mutable local filesystem.
    """
    directory, dataset = Path(experiment_dir), Path(dataset_path).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", directory.name):
        raise ForwardIntegrityError("Experiment name must be a safe registry identifier.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Existing directories (including incomplete freezes) must not be repurposed.
    try:
        directory.mkdir(exist_ok=False)
    except FileExistsError as exc:
        _registry(directory, status="failed", phase="freeze", dataset_path=dataset,
                  summary={"requested_dataset": str(dataset)}, error=f"{type(exc).__name__}: {exc}")
        raise
    try:
        stamp, last_date = _timestamp(observed_at), _date(cutoff)
        if isinstance(config, (str, Path)):
            cfg = _config(json.loads(Path(config).read_text(encoding="utf-8")))
        elif isinstance(config, Mapping):
            cfg = _config(config)
        else:
            cfg = copy.deepcopy(config or RegimeShiftConfig())
        if cfg.execution_cost_model is None:
            cfg.execution_cost_model = _cost_model(cfg, None)
        config_json = json.loads(_json(asdict(cfg)))
        cfg = _config(config_json)
        history = _read_csv(dataset, cfg)
        if history.index[-1] != last_date:
            raise ForwardIntegrityError("The frozen historical CSV must end exactly at cutoff; no known future rows.")
        if last_date > stamp.tz_convert("Asia/Kolkata").tz_localize(None).normalize():
            raise ForwardIntegrityError("Cutoff cannot be later than observed_at's trading date.")
        ctx = build_context(history, cfg)
        needed = max(ctx.min_history, cfg.hmm_config.minimum_training_observations)
        if len(ctx.raw_features) < needed:
            raise ForwardIntegrityError("Historical prefix cannot train the frozen model at cutoff.")
        manifest = {
            "forward_version": FORWARD_VERSION, "protocol_version": PROTOCOL_VERSION,
            "experiment_id": directory.name, "observed_at": stamp.isoformat(),
            "timestamp_authority": "unverified_local_clock", "cutoff": str(last_date.date()),
            "release_at": None if release_at is None else _timestamp(release_at).isoformat(),
            "config": config_json, "training_policy": TRAINING_POLICY,
            "historical_input": _input_record(dataset, history),
            "sources": _source_hashes(Path(source_root)), "environment": environment(),
            "prospective_certified": False,
        }
        digest = _digest(manifest)
        _write_new(directory / "freeze.json", manifest)
        _write_new(directory / "freeze.sha256.json", {"freeze_sha256": digest})
        _registry(directory, status="completed", phase="freeze", dataset_path=dataset,
                  summary={"freeze_sha256": digest, "observed_at": stamp.isoformat()})
        return {**manifest, "freeze_sha256": digest}
    except Exception as exc:
        _registry(directory, status="failed", phase="freeze", dataset_path=dataset,
                  summary={"requested_dataset": str(dataset)}, error=f"{type(exc).__name__}: {exc}")
        raise


def _load_freeze(directory: Path, source_root: Path, registry_records: list[dict]) -> tuple[dict, str]:
    manifest = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    digest = _digest(manifest)
    sidecar = json.loads((directory / "freeze.sha256.json").read_text(encoding="utf-8"))
    anchors = [r for r in registry_records if r["summary"].get("phase") == "freeze" and r["status"] == "completed"]
    if (sidecar.get("freeze_sha256") != digest or len(anchors) != 1
            or anchors[0]["summary"].get("freeze_sha256") != digest):
        raise ForwardIntegrityError("Frozen manifest/config/protocol has been mutated or its registry anchor is absent.")
    if (manifest["forward_version"] != FORWARD_VERSION or manifest["protocol_version"] != PROTOCOL_VERSION
            or manifest["training_policy"] != TRAINING_POLICY):
        raise ForwardIntegrityError("Frozen training protocol differs from the running protocol.")
    if manifest["sources"] != _source_hashes(source_root):
        raise ForwardIntegrityError("Frozen code source hashes changed; create a new experiment instead.")
    if manifest["environment"] != environment():
        raise ForwardIntegrityError("Frozen runtime environment changed; create a new experiment instead.")
    return manifest, digest


def _records(directory: Path) -> list[dict]:
    path = directory.parent / "registry" / f"{directory.name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _accepted_history(directory: Path, records: list[dict], manifest: Mapping,
                      config: RegimeShiftConfig) -> list[pd.DataFrame]:
    frames = [_verify_input(manifest["historical_input"], config)]
    recorded_ids = set()
    for record in records:
        summary = record["summary"]
        if summary.get("phase") == "evaluate":
            attempt_dir = directory / "attempts" / summary["attempt_id"]
            for field, filename in (("request_sha256", "request.json"), ("failure_sha256", "failure.json")):
                expected = summary.get(field)
                if expected is not None and sha256_file(attempt_dir / filename) != expected:
                    raise ForwardIntegrityError("A prior attempt record has been mutated.")
        if summary.get("phase") != "evaluate" or record["status"] != "completed":
            continue
        attempt_id = summary["attempt_id"]
        recorded_ids.add(attempt_id)
        attempt_dir = directory / "attempts" / attempt_id
        result_path = attempt_dir / "result.json"
        if sha256_file(result_path) != summary["result_sha256"]:
            raise ForwardIntegrityError("An accepted evaluation result has been mutated.")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for name, digest in result["outputs"].items():
            if sha256_file(attempt_dir / name) != digest:
                raise ForwardIntegrityError("An accepted accounting output has been mutated.")
        frame = _verify_input(result["input"], config)
        if list(frame.columns) != list(frames[0].columns) or frame.index[0] <= frames[-1].index[-1]:
            raise ForwardIntegrityError("Accepted input history overlaps or has a changed schema.")
        frames.append(frame)
    # Detect completed artifacts left unregistered by interruption instead of silently
    # repeating an already evaluated period. Failed/incomplete attempts remain visible.
    for result_path in (directory / "attempts").glob("*/result.json"):
        if result_path.parent.name not in recorded_ids:
            raise ForwardIntegrityError("Unregistered completed attempt requires audit; refusing an overlapping replay.")
    return frames


def _run_frozen(prices: pd.DataFrame, config: RegimeShiftConfig, cutoff: pd.Timestamp,
                on_trial_start=None) -> tuple[dict, Any]:
    ctx = build_context(prices, config)
    positions = ctx.asset_returns.index.get_indexer([cutoff])
    if positions[0] < 0:
        raise ForwardIntegrityError("Cutoff is absent from the feature/return execution index.")
    start = int(positions[0])
    schedule = set(range(start, len(ctx.asset_returns), config.rebalance_frequency))

    def signal_fn(pos, date, weights, allocated):
        features = ctx.raw_features.iloc[:pos + 1]
        regime = hmm_regime(ctx, features)
        returns = ctx.asset_returns.iloc[:pos + 1].tail(config.portfolio_config.estimation_lookback)
        previous = dict(zip(ctx.assets, weights)) if allocated else None
        portfolio = optimize_portfolio(regime, returns, previous, config=config.portfolio_config)
        return Signal(np.array([portfolio.weights[a] for a in ctx.assets], dtype=float))

    if on_trial_start is not None:
        on_trial_start("regimeshift")
    trace = simulate_lagged_execution(ctx.asset_returns, schedule, signal_fn, _cost_model(config, None))
    runs = {"regimeshift": StrategyRun.from_result("regimeshift", trace)}
    for strategy_id, weights in (("static_6040", static_60_40_weights()),
                                 ("equal_weight", equal_weight_weights())):
        if on_trial_start is not None:
            on_trial_start(strategy_id)
        benchmark = run_benchmark(prices, weights, config, rebalance_flags=trace.executed,
                                  start_date=trace.net.index[0])
        runs[strategy_id] = StrategyRun.from_result(strategy_id, benchmark)
    require_common_index(runs.values())
    return runs, ctx


def _classification(manifest: Mapping, index: pd.DatetimeIndex) -> str:
    boundary = _timestamp(manifest["observed_at"])
    if manifest["release_at"] is not None:
        boundary = max(boundary, _timestamp(manifest["release_at"]))
    boundary_date = boundary.tz_convert("Asia/Kolkata").tz_localize(None).normalize()
    return "historical" if index[0] <= boundary_date else "post_release_unverified"


def evaluate_forward(experiment_dir: str | Path, *, dataset_path: str | Path,
                     observed_at=None, source_root: str | Path = ROOT) -> dict:
    """Append one evaluation attempt using *only new* rows after the accepted end.

    Prior input files must remain available and unchanged. Every attempt gets a
    fresh directory and registry line, including rejected chronology and failures.
    """
    directory, dataset = Path(experiment_dir), Path(dataset_path).resolve()
    if not directory.is_dir():
        raise ForwardIntegrityError("Freeze this experiment before evaluating it.")
    attempt_id = uuid.uuid4().hex
    attempts = directory / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt_dir = attempts / attempt_id
    attempt_dir.mkdir(exist_ok=False)
    stamp = None
    trials_started: list[str] = []
    try:
        with _lock(directory):
            stamp = _timestamp(observed_at)
            request = {"attempt_id": attempt_id, "observed_at": stamp.isoformat(),
                       "requested_dataset": str(dataset)}
            _write_new(attempt_dir / "request.json", request)
            records = _records(directory)
            manifest, freeze_hash = _load_freeze(directory, Path(source_root), records)
            prior_times = [_timestamp(manifest["observed_at"])]
            # Requests include failed and interrupted attempts: observed timestamps
            # cannot be reused, even when a prior model fit or validation failed.
            for path in attempts.glob("*/request.json"):
                if path.parent != attempt_dir:
                    prior_times.append(_timestamp(json.loads(path.read_text(encoding="utf-8"))["observed_at"]))
            if stamp <= max(prior_times):
                raise ForwardIntegrityError("observed_at overlaps a freeze or prior attempt timestamp.")
            config = _config(manifest["config"])
            frames = _accepted_history(directory, records, manifest, config)
            incremental = _read_csv(dataset, config)
            if list(incremental.columns) != list(frames[0].columns):
                raise ForwardIntegrityError("Incremental CSV schema differs from frozen history.")
            if incremental.index[0] <= _date(manifest["cutoff"]):
                raise ForwardIntegrityError("Incremental dates must be strictly after cutoff.")
            if incremental.index[0] <= frames[-1].index[-1]:
                raise ForwardIntegrityError("Incremental dates overlap an accepted evaluation period.")
            if incremental.index[-1] > stamp.tz_convert("Asia/Kolkata").tz_localize(None).normalize():
                raise ForwardIntegrityError("Incremental dates exceed the declared observation date.")
            input_record = _input_record(dataset, incremental)
            prices = pd.concat([*frames, incremental])
            runs, ctx = _run_frozen(prices, config, _date(manifest["cutoff"]),
                                    on_trial_start=trials_started.append)
            # Replay from original cash entry; only new dates are reported. Later
            # batches retain holdings and do not invent a fresh initial allocation.
            selected = {sid: run.on_index(incremental.index) for sid, run in runs.items()}
            require_common_index(selected.values())
            summary = metrics_table(list(selected.values()), ctx.cash_returns)
            summary.to_csv(attempt_dir / "summary.csv", index=False)
            daily = pd.concat([
                pd.DataFrame({"strategy": sid, "gross": run.gross, "net": run.net,
                              "costs": run.costs, "turnover": run.turnover})
                .rename_axis("date").reset_index() for sid, run in selected.items()
            ], ignore_index=True)
            daily.to_csv(attempt_dir / "daily_accounting.csv", index=False)
            # Catch concurrent external mutation during execution before accepting.
            _verify_input(input_record, config)
            for frame_record in [manifest["historical_input"], *[
                json.loads((directory / "attempts" / r["summary"]["attempt_id"] / "result.json")
                           .read_text(encoding="utf-8"))["input"]
                for r in records if r["summary"].get("phase") == "evaluate" and r["status"] == "completed"
            ]]:
                _verify_input(frame_record, config)
            _load_freeze(directory, Path(source_root), records)
            result = {"attempt_id": attempt_id, "freeze_sha256": freeze_hash,
                      "observed_at": stamp.isoformat(), "input": input_record,
                      "evaluation_type": _classification(manifest, incremental.index),
                      "prospective_certified": False, "timestamp_authority": "unverified_local_clock",
                      "n_trials": len(trials_started), "outputs": {name: sha256_file(attempt_dir / name)
                          for name in ("summary.csv", "daily_accounting.csv")}}
            _write_new(attempt_dir / "result.json", result)
            _registry(directory, status="completed", phase="evaluate", dataset_path=dataset,
                      summary={"attempt_id": attempt_id, "freeze_sha256": freeze_hash,
                               "n_trials": len(trials_started), "trials_started": trials_started,
                               "observed_at": stamp.isoformat(), "evaluation_type": result["evaluation_type"],
                               "result_sha256": sha256_file(attempt_dir / "result.json"),
                               "request_sha256": sha256_file(attempt_dir / "request.json")})
            return result
    except Exception as exc:
        failure = {"attempt_id": attempt_id, "observed_at": None if stamp is None else stamp.isoformat(),
                   "requested_dataset": str(dataset), "error": f"{type(exc).__name__}: {exc}"}
        _write_new(attempt_dir / "failure.json", failure)
        failure_summary = {key: value for key, value in failure.items() if key != "error"}
        failure_summary["n_trials"] = len(trials_started)
        failure_summary["trials_started"] = trials_started
        failure_summary["failure_sha256"] = sha256_file(attempt_dir / "failure.json")
        if (attempt_dir / "request.json").is_file():
            failure_summary["request_sha256"] = sha256_file(attempt_dir / "request.json")
        _registry(directory, status="failed", phase="evaluate", dataset_path=dataset,
                  summary=failure_summary, error=failure["error"])
        raise
