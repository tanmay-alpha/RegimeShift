"""Deterministic workflow behavior without stochastic fitting or UI dependencies."""
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from regime_shift.cli import main
from regime_shift.exceptions import DataValidationError
from regime_shift.experiment import sha256_file
from regime_shift.research.evaluation import StrategyRun
from regime_shift.research.workflow import (
    WorkflowError,
    export_experiment_zip,
    inspect_experiment,
    list_experiments,
    load_dataset,
    load_table,
    run_experiment,
    validate_config,
)


@pytest.fixture
def research_case(tmp_path, monkeypatch):
    dates = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({"equity": 100 * np.exp(np.arange(300) * .0002),
                           "gold": 80 * np.exp(np.arange(300) * .0001),
                           "bond": np.full(300, 100.0)}, index=dates)
    dataset = tmp_path / "canonical.csv"
    prices.to_csv(dataset, index_label="Date")
    config = {"dataset_path": str(dataset), "strategies": ["vol_rule", "equal_weight"],
              "evaluation": {"start": str(dates[220].date()), "end": str(dates[260].date())},
              "engine": {"transaction_cost_bps": 10.0}}
    calls = []

    def deterministic_suite(data, engine, variants):
        calls.append((data.copy(), engine, list(variants)))
        index = data.index[200:]
        gross = pd.Series(np.where(np.arange(len(index)) % 2, -.002, .003), index=index)
        cost = pd.Series(.0001, index=index)
        cost.iloc[0] = .001
        turn = cost / .001
        weights = pd.DataFrame([[.6, 0, .4]] * len(index), index=index, columns=["equity", "gold", "bond"])
        identifiers = ["regimeshift", *variants, "static_6040", "equal_weight"]
        runs = {s: StrategyRun(s, gross, (1 + gross) * (1 - cost) - 1,
                               cost, turn, weights=weights) for s in identifiers}
        reference = SimpleNamespace(
            regime_probabilities=pd.DataFrame([[.7, .2, .1]] * len(index), index=index, columns=["Bull", "Bear", "Crisis"]),
            regime_series=pd.Series("Bull", index=index),
            hmm_diagnostics=[{"date": data.index[199], "execution_date": index[0], "regime": "Bull",
                              "converged": True, "n_iter": 5, "restarts": []}],
            event_log=pd.DataFrame({"rebalance_flag": True}, index=index), config_metadata={"test_fixture": True},
        )
        ctx = SimpleNamespace(cash_returns=pd.Series(0.0, index=data.index),
                              asset_returns=data.pct_change().iloc[1:], raw_features=data,
                              rebalance_positions=set(range(0, len(data), 21)), min_history=126, config=engine)
        return runs, reference, ctx

    monkeypatch.setattr("regime_shift.research.workflow.run_core_suite", deterministic_suite)
    return SimpleNamespace(root=tmp_path, dates=dates, prices=prices, dataset=dataset,
                           config=config, calls=calls, suite=deterministic_suite,
                           results=tmp_path / "experiments", registry=tmp_path / "registry")


def _execute(case, config=None):
    return run_experiment(case.config if config is None else config,
                          results_dir=case.results, registry_dir=case.registry)


def _records(case):
    return [json.loads(line) for path in case.registry.glob("*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()]


def test_selected_outputs_use_bounded_engine_and_real_accounting(research_case):
    case = research_case
    result = _execute(case)
    data, engine, variants = case.calls[0]
    assert data.index[-1] == case.dates[260]
    assert variants == ["vol_rule"]
    assert engine.transaction_cost_bps == 10
    assert engine.hmm_config.random_state == 42
    manifest = result.manifest
    assert manifest["status"] == "completed"
    assert manifest["summary"]["n_obs"] == 41
    assert manifest["summary"]["computed_strategies"] == ["regimeshift", "vol_rule", "static_6040", "equal_weight"]
    assert not (result.output_dir / "regimeshift").exists()
    assert not (result.output_dir / "static_6040").exists()
    for strategy in case.config["strategies"]:
        accounting = load_table(result.output_dir, "accounting", strategy)
        assert accounting.index.equals(case.dates[220:261])
        np.testing.assert_allclose(accounting.net_return,
                                   (1 + accounting.gross_return) * (1 - accounting.transaction_cost) - 1)
        np.testing.assert_allclose(accounting.net_equity, (1 + accounting.net_return).cumprod())
        assert accounting.transaction_cost.sum() > 0
        assert accounting.turnover.sum() > 0
        assert (accounting.drawdown <= 0).all()
        assert load_table(result.output_dir, "weights", strategy).index.equals(accounting.index)
        assert load_table(result.output_dir, "explanations", strategy).index.equals(accounting.index)
    probabilities = load_table(result.output_dir, "regime_probabilities")
    assert probabilities.index.equals(case.dates[220:261])
    assert probabilities.entropy_nats.iloc[0] == pytest.approx(-np.dot([.7, .2, .1], np.log([.7, .2, .1])))
    assert _records(case)[0]["n_trials"] == 4


def test_manifest_hashes_and_versioned_attempts_are_reproducible(research_case):
    case = research_case
    first, second = _execute(case), _execute(case)
    assert first.output_dir != second.output_dir
    assert len(list_experiments(case.results)) == 2
    manifest = inspect_experiment(first.output_dir)
    assert manifest["dataset_sha256_canonical"] == sha256_file(case.dataset)
    assert manifest["config"] == validate_config(case.config)
    assert manifest["protocol_version"] >= 3
    assert "environment" in manifest and "git_commit" in manifest and "source_tree_dirty" in manifest
    for name, expected in manifest["outputs"].items():
        assert sha256_file(first.output_dir / name) == expected
    assert len(_records(case)) == 2


def test_engine_failure_is_registered_and_manifested(research_case, monkeypatch):
    case = research_case

    def broken(*args, **kwargs):
        raise RuntimeError("deterministic optimizer failure")

    monkeypatch.setattr("regime_shift.research.workflow.run_core_suite", broken)
    with pytest.raises(WorkflowError, match="deterministic optimizer failure") as caught:
        _execute(case)
    manifest = inspect_experiment(caught.value.output_dir)
    assert manifest["status"] == "failed"
    assert "optimizer failure" in manifest["error"]
    assert "error.json" in manifest["outputs"]
    record = _records(case)[0]
    assert record["status"] == "failed" and record["n_trials"] == 4
    assert "optimizer failure" in record["error"]
    archive = zipfile.ZipFile(io.BytesIO(export_experiment_zip(caught.value.output_dir)))
    assert "error.json" in archive.namelist()


def test_invalid_selection_and_missing_dataset_preserve_failure_records(research_case):
    case = research_case
    invalid = dict(case.config, strategies=["__import__('os').system('bad')"])
    with pytest.raises(WorkflowError, match="strategies") as caught:
        _execute(case, invalid)
    assert inspect_experiment(caught.value.output_dir)["status"] == "failed"
    missing = dict(case.config, dataset_path=str(case.root / "missing.csv"))
    with pytest.raises(WorkflowError, match="Local CSV not found") as caught:
        _execute(case, missing)
    assert inspect_experiment(caught.value.output_dir)["dataset_sha256_canonical"] is None
    records = _records(case)
    assert len(records) == 2 and all(r["status"] == "failed" for r in records)
    assert not case.calls


@pytest.mark.parametrize("change,match", [
    ({"strategies": []}, "nonempty"),
    ({"strategies": ["equal_weight", "equal_weight"]}, "duplicates"),
    ({"engine": {"train_window": True}}, "integer"),
    ({"engine": {"transaction_cost_bps": float("nan")}}, "finite"),
    ({"engine": {"train_window": 60, "minimum_training_observations": 126}}, "must not exceed"),
    ({"engine": {"python": "print('unsafe')"}}, "Unknown"),
    ({"dataset_path": "https://vendor/prices.csv"}, "local"),
    ({"evaluation": {"start": "2020-02-30", "end": "2020-12-01"}}, "valid calendar"),
])
def test_json_schema_rejects_unsafe_or_unbounded_requests(research_case, change, match):
    with pytest.raises(ValueError, match=match):
        validate_config(dict(research_case.config, **change))


def test_start_before_eligible_fill_is_not_silently_clipped(research_case):
    case = research_case
    config = dict(case.config, evaluation={"start": str(case.dates[100].date()), "end": str(case.dates[260].date())})
    with pytest.raises(WorkflowError, match="first eligible fill"):
        _execute(case, config)
    assert _records(case)[0]["status"] == "failed"


def test_outside_dataset_boundaries_do_not_run_engine(research_case):
    case = research_case
    config = dict(case.config, evaluation={"start": "2019-01-01", "end": "2030-01-01"})
    with pytest.raises(WorkflowError, match="dataset calendar span"):
        _execute(case, config)
    assert not case.calls


def test_misaligned_engine_outputs_are_not_filled(research_case, monkeypatch):
    case = research_case

    def misaligned(*args, **kwargs):
        runs, reference, ctx = case.suite(*args, **kwargs)
        original = runs["equal_weight"]
        runs["equal_weight"] = original.on_index(original.index[1:])
        return runs, reference, ctx

    monkeypatch.setattr("regime_shift.research.workflow.run_core_suite", misaligned)
    with pytest.raises(WorkflowError, match="dates only in"):
        _execute(case)
    assert _records(case)[0]["status"] == "failed"


def test_strict_dataset_loader_never_repairs_missing_or_sorts_dates(research_case):
    case = research_case
    broken = case.prices.copy()
    broken.iloc[10, 0] = np.nan
    broken.to_csv(case.dataset)
    with pytest.raises(DataValidationError, match="NaN|missing"):
        load_dataset(case.dataset)
    case.prices.iloc[::-1].to_csv(case.dataset)
    with pytest.raises(DataValidationError, match="sorted"):
        load_dataset(case.dataset)


def test_zip_excludes_raw_data_and_refuses_changed_outputs(research_case):
    case = research_case
    result = _execute(case)
    raw = result.output_dir / "raw_market.csv"
    raw.write_bytes(case.dataset.read_bytes())
    manifest = result.manifest
    manifest["outputs"]["raw_market.csv"] = sha256_file(raw)
    (result.output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    payload = export_experiment_zip(result.output_dir)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert "manifest.json" in archive.namelist()
        assert "equal_weight/accounting.csv" in archive.namelist()
        assert "raw_market.csv" not in archive.namelist()
        assert "canonical.csv" not in archive.namelist()
    (result.output_dir / "metrics.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        export_experiment_zip(result.output_dir)


def test_cli_run_inspect_and_export_share_saved_artifacts(research_case, capsys):
    case = research_case
    config_path = case.root / "experiment.json"
    config_path.write_text(json.dumps(case.config), encoding="utf-8")
    assert main(["run", "--config", str(config_path), "--results-dir", str(case.results),
                 "--registry-dir", str(case.registry)]) == 0
    completed = json.loads(capsys.readouterr().out)
    out = Path(completed["experiment_dir"])
    assert main(["inspect", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    zip_path = case.root / "reproduction.zip"
    assert main(["export", str(out), "--output", str(zip_path)]) == 0
    assert zip_path.is_file()
    assert len(case.calls) == 1  # inspect/export never refit a model


def test_explanations_use_executed_signal_not_new_unfilled_regime(research_case, monkeypatch):
    case = research_case

    def changing_reference(*args, **kwargs):
        runs, reference, ctx = case.suite(*args, **kwargs)
        reference.hmm_diagnostics.append({
            "date": case.dates[239], "execution_date": case.dates[240],
            "regime": "Bear", "converged": True, "n_iter": 5, "restarts": [],
        })
        reference.regime_probabilities.loc[case.dates[239]:, ["Bull", "Bear", "Crisis"]] = [.2, .7, .1]
        reference.regime_series.loc[case.dates[239]:] = "Bear"
        return runs, reference, ctx

    monkeypatch.setattr("regime_shift.research.workflow.run_core_suite", changing_reference)
    result = _execute(case, dict(case.config, strategies=["regimeshift"]))
    probabilities = load_table(result.output_dir, "regime_probabilities")
    explanations = load_table(result.output_dir, "explanations", "regimeshift")
    assert probabilities.loc[case.dates[239], "regime_at_close"] == "Bear"
    assert explanations.loc[case.dates[239], "regime_for_held_target"] == "Bull"
    assert explanations.loc[case.dates[240], "regime_for_held_target"] == "Bear"
    assert pd.Timestamp(explanations.loc[case.dates[240], "signal_date_for_held_target"]) == case.dates[239]
    assert "equity <=.40" in explanations.loc[case.dates[240], "rule_explanation"]
