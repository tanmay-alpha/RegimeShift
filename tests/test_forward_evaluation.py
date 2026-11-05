"""Behavioral tests of immutable freezes and incremental causal accounting."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from regime_shift.config import RegimeShiftConfig
from regime_shift.research import forward


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    # A standalone source fixture permits actual byte-level tamper tests without
    # mutating the repository running the test suite.
    source = tmp_path / "checkout"
    (source / "src" / "regime_shift").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "src" / "regime_shift" / "model.py").write_bytes(b"frozen model implementation\n")
    (source / "scripts" / "forward_evaluation.py").write_bytes(b"frozen command implementation\n")
    monkeypatch.setattr(forward, "environment", lambda: {"test_runtime": "fixed"})
    calls = []

    def regime(ctx, features):
        calls.append((features.index[-1], len(features)))
        return "Bull"

    monkeypatch.setattr(forward, "hmm_regime", regime)
    monkeypatch.setattr(forward, "optimize_portfolio", lambda *args, **kwargs: SimpleNamespace(
        weights={"equity": 1.0, "gold": 0.0, "bond": 0.0}))
    index = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(17)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.006, (300, 3)), axis=0)),
                          index=index, columns=["equity", "gold", "bond"])
    history = tmp_path / "history.csv"
    prices.iloc[:260].to_csv(history)
    first = tmp_path / "first.csv"
    prices.iloc[260:280].to_csv(first)
    second = tmp_path / "second.csv"
    prices.iloc[280:].to_csv(second)
    config = RegimeShiftConfig(minimum_training_observations=60, train_window=100,
                              rebalance_frequency=5, transaction_cost_bps=10)
    directory = tmp_path / "experiments" / "frozen_test"
    # Date supplied after cutoff but before new rows. The clock remains untrusted.
    observed = (index[259] + pd.Timedelta(hours=12)).tz_localize("UTC")  # 17:30 IST, after close
    manifest = forward.freeze_experiment(directory, dataset_path=history, cutoff=index[259],
                                         config=config, observed_at=observed, source_root=source)
    return SimpleNamespace(directory=directory, source=source, prices=prices, history=history,
                           first=first, second=second, manifest=manifest, calls=calls,
                           eval_time=pd.Timestamp("2022-01-01T12:00:00Z"))


def evaluate(exp, path=None, timestamp=None):
    return forward.evaluate_forward(exp.directory, dataset_path=path or exp.first,
                                    observed_at=timestamp or exp.eval_time, source_root=exp.source)


def records(exp):
    return forward._records(exp.directory)


def accounting(exp, result):
    return pd.read_csv(exp.directory / "attempts" / result["attempt_id"] / "daily_accounting.csv",
                       parse_dates=["date"])


def test_freeze_cannot_be_overwritten_and_configuration_is_bound(experiment):
    exp = experiment
    before = (exp.directory / "freeze.json").read_bytes()
    with pytest.raises(FileExistsError):
        forward.freeze_experiment(exp.directory, dataset_path=exp.history,
                                  cutoff=exp.prices.index[259], source_root=exp.source)
    assert (exp.directory / "freeze.json").read_bytes() == before
    assert records(exp)[-1]["status"] == "failed"
    payload = json.loads(before)
    payload["config"]["train_window"] += 1
    (exp.directory / "freeze.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(forward.ForwardIntegrityError, match="mutated"):
        evaluate(exp)
    assert records(exp)[-1]["status"] == "failed"


@pytest.mark.parametrize("change", ["modify", "add", "delete"])
def test_source_tree_changes_are_rejected(experiment, change):
    exp = experiment
    source = exp.source / "src" / "regime_shift" / "model.py"
    if change == "modify":
        source.write_bytes(b"changed implementation\n")
    elif change == "add":
        source.with_name("new_model.py").write_bytes(b"new code\n")
    else:
        source.unlink()
    with pytest.raises(forward.ForwardIntegrityError, match="source"):
        evaluate(exp)


def test_rewritten_historical_prices_are_rejected_without_cleaning(experiment):
    exp = experiment
    rewritten = exp.prices.iloc[:260].copy()
    rewritten.iloc[10, 0] *= 1.001
    rewritten.to_csv(exp.history)
    with pytest.raises(forward.ForwardIntegrityError, match="rewritten"):
        evaluate(exp)
    assert not exp.calls


@pytest.mark.parametrize("kind", ["cutoff", "duplicate", "unsorted", "missing", "intraday"])
def test_bad_incremental_chronology_is_recorded_as_distinct_failures(experiment, kind):
    exp = experiment
    bad = exp.prices.iloc[260:265].copy()
    if kind == "cutoff":
        bad = exp.prices.iloc[259:265]
    elif kind == "duplicate":
        bad.index = pd.DatetimeIndex([bad.index[0], bad.index[0], *bad.index[2:]])
    elif kind == "unsorted":
        bad = bad.iloc[::-1]
    elif kind == "missing":
        bad.iloc[1, 0] = np.nan
    else:
        bad.index = bad.index + pd.Timedelta(hours=1)
    path = exp.first.with_name("invalid.csv")
    bad.to_csv(path)
    for offset in (0, 1):
        with pytest.raises(forward.ForwardIntegrityError):
            evaluate(exp, path, exp.eval_time + pd.Timedelta(seconds=offset))
    failed = [r for r in records(exp) if r["status"] == "failed"]
    assert len(failed) == 2
    assert len({r["summary"]["attempt_id"] for r in failed}) == 2
    for record in failed:
        attempt = exp.directory / "attempts" / record["summary"]["attempt_id"]
        assert (attempt / "failure.json").is_file()
        assert not (attempt / "result.json").exists()


def test_initial_entry_and_baselines_have_identical_dates_and_real_costs(experiment):
    exp = experiment
    result = evaluate(exp)
    daily = accounting(exp, result)
    for _, run in daily.groupby("strategy"):
        assert list(run.date) == list(exp.prices.index[260:280])
        assert run.gross.iloc[0] == pytest.approx(0)
        assert run.net.iloc[0] == pytest.approx(-0.001)
        assert run.costs.iloc[0] == pytest.approx(0.001)
        assert run.turnover.iloc[0] == pytest.approx(1)
    equity = daily[daily.strategy == "regimeshift"]
    expected = exp.prices.equity.pct_change().iloc[261]
    assert equity.gross.iloc[1] == pytest.approx(expected)
    assert exp.calls[0][0] == exp.prices.index[259]
    assert [date for date, _ in exp.calls] == list(exp.prices.index[259:280:5])
    assert result["evaluation_type"] == "post_release_unverified"
    assert result["prospective_certified"] is False


def test_incremental_replay_preserves_holdings_and_rejects_repeated_period(experiment):
    exp = experiment
    first = evaluate(exp)
    old_bytes = (exp.directory / "attempts" / first["attempt_id"] / "result.json").read_bytes()
    second = evaluate(exp, exp.second, exp.eval_time + pd.Timedelta(seconds=1))
    new = accounting(exp, second)
    equity = new[new.strategy == "regimeshift"]
    assert equity.gross.iloc[0] == pytest.approx(exp.prices.equity.pct_change().iloc[280])
    assert equity.turnover.iloc[0] == pytest.approx(0)
    assert (exp.directory / "attempts" / first["attempt_id"] / "result.json").read_bytes() == old_bytes
    with pytest.raises(forward.ForwardIntegrityError, match="overlap"):
        evaluate(exp, exp.second, exp.eval_time + pd.Timedelta(seconds=2))
    assert [r["status"] for r in records(exp)] == ["completed", "completed", "completed", "failed"]


def test_replay_matches_one_continuous_evaluation(experiment):
    exp = experiment
    first = evaluate(exp)
    second = evaluate(exp, exp.second, exp.eval_time + pd.Timedelta(seconds=1))
    config = forward._config(exp.manifest["config"])
    runs, _ = forward._run_frozen(exp.prices, config, exp.prices.index[259])
    combined = pd.concat([accounting(exp, first), accounting(exp, second)])
    for sid, run in runs.items():
        observed = combined[combined.strategy == sid].set_index("date").sort_index()
        for column in ("gross", "net", "costs", "turnover"):
            np.testing.assert_allclose(observed[column], getattr(run, column), atol=1e-12)


@pytest.mark.parametrize("target", ["accepted_input", "result", "daily", "request"])
def test_previously_accepted_artifact_tamper_blocks_next_batch(experiment, target):
    exp = experiment
    result = evaluate(exp)
    attempt = exp.directory / "attempts" / result["attempt_id"]
    if target == "accepted_input":
        changed = exp.prices.iloc[260:280].copy()
        changed.iloc[0, 1] *= 1.001
        changed.to_csv(exp.first)
    else:
        path = attempt / {"result": "result.json", "daily": "daily_accounting.csv", "request": "request.json"}[target]
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(forward.ForwardIntegrityError, match="mutated|rewritten"):
        evaluate(exp, exp.second, exp.eval_time + pd.Timedelta(seconds=1))


def test_overlapping_observation_timestamps_are_rejected_even_after_failure(experiment):
    exp = experiment
    missing = exp.first.with_name("missing.csv")
    with pytest.raises(FileNotFoundError):
        evaluate(exp, missing)
    with pytest.raises(forward.ForwardIntegrityError, match="observed_at overlaps"):
        evaluate(exp)
    assert len([r for r in records(exp) if r["status"] == "failed"]) == 2


def test_historical_dates_are_never_promoted_by_a_user_clock(experiment):
    exp = experiment
    manifest = {**exp.manifest, "observed_at": "2025-01-01T00:00:00Z"}
    assert forward._classification(manifest, exp.prices.index[260:280]) == "historical"
    # Backdating a clock can change a descriptive label, never confer certification.
    result = evaluate(exp)
    assert result["timestamp_authority"] == "unverified_local_clock"
    assert result["prospective_certified"] is False


def test_freeze_rejects_a_history_with_known_future_rows(tmp_path):
    history = tmp_path / "history.csv"
    frame = pd.DataFrame(100.0, index=pd.bdate_range("2020-01-01", periods=5),
                         columns=["equity", "gold", "bond"])
    frame.to_csv(history)
    with pytest.raises(forward.ForwardIntegrityError, match="end exactly at cutoff"):
        forward.freeze_experiment(tmp_path / "bad_freeze", dataset_path=history,
                                  cutoff=frame.index[-2], observed_at="2021-01-01T00:00:00Z")
    record = json.loads((tmp_path / "registry" / "bad_freeze.jsonl").read_text())
    assert record["status"] == "failed"


def test_runtime_environment_mutation_requires_a_new_freeze(experiment, monkeypatch):
    monkeypatch.setattr(forward, "environment", lambda: {"test_runtime": "changed"})
    with pytest.raises(forward.ForwardIntegrityError, match="runtime environment"):
        evaluate(experiment)


def test_failed_attempt_artifacts_cannot_be_rewritten(experiment):
    exp = experiment
    with pytest.raises(FileNotFoundError):
        evaluate(exp, exp.first.with_name("missing.csv"))
    failed = records(exp)[-1]
    failure = exp.directory / "attempts" / failed["summary"]["attempt_id"] / "failure.json"
    failure.write_bytes(failure.read_bytes() + b" ")
    with pytest.raises(forward.ForwardIntegrityError, match="attempt record has been mutated"):
        evaluate(exp, timestamp=exp.eval_time + pd.Timedelta(seconds=1))


def test_user_declared_release_timestamp_cannot_certify_prospective(experiment):
    manifest = {**experiment.manifest, "release_at": "2025-01-01T00:00:00Z"}
    assert forward._classification(manifest, experiment.prices.index[260:280]) == "historical"
    assert experiment.manifest["prospective_certified"] is False
