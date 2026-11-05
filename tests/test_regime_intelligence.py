"""Behavioral boundaries; synthetic fixtures exercise mechanics, not research skill."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from regime_shift.config import RegimeShiftConfig
from regime_shift.execution import ExecutionCostModel
from regime_shift.regime_model import RegimeSolution, StateStatistics
from regime_shift.research import regime_intelligence as intelligence
from regime_shift.research.evaluation import require_common_index
from regime_shift.research.strategies import ResearchContext, hmm_fixed, run_target_strategy


def test_entropy_extremes_and_exploratory_alert_boundary():
    assert intelligence.normalized_entropy([1, 0, 0]) == 0
    assert intelligence.normalized_entropy([1 / 3] * 3) == pytest.approx(1)
    boundary = intelligence.uncertainty_diagnostics([0.6, 0.4, 0])
    assert boundary["low_max_posterior_alert"]
    assert not boundary["high_entropy_alert"]
    assert boundary["uncertainty_alert"]
    assert not intelligence.uncertainty_diagnostics([0.61, 0.39, 0])["uncertainty_alert"]
    assert intelligence.uncertainty_diagnostics([1 / 3] * 3)["high_entropy_alert"]


@pytest.mark.parametrize("probabilities", [
    [0, 0, 0], [-0.1, 0.6, 0.5], [np.nan, 0, 1], [np.inf, 0, 0],
    [0.2, 0.2, 0.2], [1], [[0.5, 0.5]],
])
def test_entropy_rejects_invalid_probabilities_without_silent_normalization(probabilities):
    with pytest.raises(ValueError, match="probability vector"):
        intelligence.normalized_entropy(probabilities)


def test_transition_duration_includes_absorbing_and_one_step_states():
    labels = list(intelligence.REGIMES)
    matrix = pd.DataFrame([[0.8, 0.2, 0], [0, 1, 0], [1, 0, 0]], index=labels, columns=labels)
    table = intelligence.transition_persistence(matrix).set_index("regime")
    assert table.loc["Bull", "expected_duration_feature_rows"] == pytest.approx(5)
    assert np.isinf(table.loc["Bear", "expected_duration_feature_rows"])
    assert table.loc["Bear", "absorbing"]
    assert table.loc["Crisis", "expected_duration_feature_rows"] == 1
    matrix.iloc[0, 0] = 0.9
    with pytest.raises(ValueError, match="row-stochastic"):
        intelligence.transition_persistence(matrix)


def test_refit_stability_uses_only_overlap_and_ignores_state_id_permutation():
    dates = pd.bdate_range("2020-01-01", periods=8)
    old = pd.Series([9, 9, 0, 0, 1, 1], index=dates[:6])
    new = pd.Series([2, 2, 0, 0, 8, 8], index=dates[2:])
    result = intelligence.overlap_state_stability(old, new)
    assert result["n_overlap"] == 4
    assert result["overlap_end"] == dates[5]
    assert result["adjusted_rand_index"] == 1
    new.iloc[1] = 0
    assert intelligence.overlap_state_stability(old, new)["adjusted_rand_index"] < 1
    assert np.isnan(intelligence.overlap_state_stability(old.iloc[:2], new)["adjusted_rand_index"])
    degenerate = intelligence.overlap_state_stability(old * 0, old * 0)
    assert degenerate["stability_status"] == "degenerate_partition"
    assert np.isnan(degenerate["adjusted_rand_index"])


def test_empirical_spells_count_observed_closes_and_censor_boundaries():
    dates = pd.bdate_range("2020-01-01", periods=10)
    labels = pd.Series(["Bull", "Bull", "Bear", "Bull"], index=dates[[0, 2, 5, 8]])
    spells = intelligence.empirical_regime_spells(labels, dates)
    assert spells["duration_trading_closes"].tolist() == [5, 3, 2]
    assert spells["n_signal_closes"].tolist() == [2, 1, 1]
    assert spells["right_censored"].tolist() == [False, False, True]
    assert spells["left_censored"].tolist() == [True, False, False]
    assert spells.iloc[-1]["last_observed_close"] == dates[-1]
    assert pd.isna(spells.iloc[-1]["next_switch_signal_close"])
    single = intelligence.empirical_regime_spells(labels.iloc[:1], dates)
    assert single.iloc[0]["left_censored"] and single.iloc[0]["right_censored"]
    with pytest.raises(ValueError, match="Every signal"):
        intelligence.empirical_regime_spells(labels, dates[1:])


def test_momentum_volatility_rule_has_explicit_precedence_and_warmup():
    assert intelligence.momentum_volatility_regime(pd.Series([0.001] * 63)) == "Bull"
    assert intelligence.momentum_volatility_regime(pd.Series([0.0] * 63)) == "Bear"
    assert intelligence.momentum_volatility_regime(pd.Series([-0.001] * 63)) == "Bear"
    elevated = pd.Series(np.tile([0.018, -0.014], 32)[:63])
    assert np.prod(1 + elevated) > 1
    assert intelligence.momentum_volatility_regime(elevated) == "Bear"
    returns = pd.Series(np.tile([0.04, -0.03], 32)[:63])
    assert np.prod(1 + returns) > 1  # positive momentum still yields Crisis at high volatility
    assert intelligence.momentum_volatility_regime(returns) == "Crisis"
    with pytest.raises(ValueError, match="63 finite"):
        intelligence.momentum_volatility_regime(returns.iloc[:62])


def test_negative_control_is_repeatable_prefix_stable_and_does_not_touch_global_rng():
    dates = pd.bdate_range("2020-01-01", periods=60)
    state_before = np.random.get_state()
    full = intelligence.negative_control_assignments(dates)
    pd.testing.assert_series_equal(full, intelligence.negative_control_assignments(dates))
    pd.testing.assert_series_equal(full.iloc[:20], intelligence.negative_control_assignments(dates[:20]))
    assert full.isin(intelligence.REGIMES).all()
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def _context():
    dates = pd.bdate_range("2020-01-01", periods=148)
    values = np.column_stack([
        np.where(np.arange(len(dates)) % 5 == 0, -0.002, 0.001),
        np.full(len(dates), 0.0002), np.full(len(dates), 0.0001),
    ])
    # The first eligible signal is pos 63, its next-close fill at 64 cannot earn this jump.
    values[64, 0] = 0.20
    returns = pd.DataFrame(values, index=dates, columns=["equity", "gold", "bond"])
    prices = 100 * (1 + returns).cumprod()
    features = pd.DataFrame({"equity_momentum_63d": np.arange(len(dates), dtype=float)}, index=dates)
    config = RegimeShiftConfig(minimum_training_observations=63, train_window=84, rebalance_frequency=21)
    config.execution_cost_model = ExecutionCostModel.flat(10, config.core_assets)
    return ResearchContext(prices, config, features, returns, set(range(21, len(dates), 21)))


def _stub_hmm(monkeypatch):
    """Replace expensive fitting only; keep the actual gated execution and accounting."""
    from regime_shift.research import strategies

    fits = []

    def fit(scaled, raw, config):
        fits.append(raw.copy())
        hidden = raw.iloc[:, 0].to_numpy(dtype=int) % 3
        stats = [StateStatistics(s, intelligence.REGIMES[s], raw.loc[hidden == s].mean(),
                                 int(np.sum(hidden == s))) for s in range(3)]
        model = SimpleNamespace(stats=stats, _restart_diagnostics=[
            {"seed": 42, "converged": True, "n_iter": 2, "selected": True, "fallback_used": False},
        ])
        return model, hidden, None

    def predict(model, scaler, sequence, train, config):
        labels = list(intelligence.REGIMES)
        regime = labels[int(train.iloc[-1, 0]) % 3]
        return RegimeSolution(
            regime, pd.Series([float(r == regime) for r in labels], index=labels),
            pd.DataFrame(np.eye(3), index=labels, columns=labels), model.stats, True, 2, 0.0,
        )

    for module in (intelligence, strategies):
        monkeypatch.setattr(module, "fit_feature_scaler", lambda train: object())
        monkeypatch.setattr(module, "transform_features", lambda scaler, train: train)
        monkeypatch.setattr(module, "fit_hmm", fit)
        monkeypatch.setattr(module, "predict_current_state", predict)
    return fits


def test_suite_preserves_official_hmm_targets_and_same_index_lagged_cost_accounting(monkeypatch):
    ctx = _context()
    fits = _stub_hmm(monkeypatch)
    result = intelligence.run_regime_intelligence(ctx)
    require_common_index(result.runs.values())
    assert len(fits) == len(result.diagnostics)
    for train, diagnostic in zip(fits, result.diagnostics.to_dict("records")):
        assert train.index[-1] == diagnostic["signal_close"]
        assert len(train) <= ctx.config.train_window
    official = run_target_strategy(ctx, "official_hmm_fixed", hmm_fixed(ctx))
    pd.testing.assert_series_equal(official.net, result.runs["hmm_fixed"].net)
    for run in result.runs.values():
        assert run.index[0] == ctx.asset_returns.index[64]
        assert run.gross.iloc[0] == 0  # jump into the fill close belongs to the preceding cash book
        assert run.turnover.iloc[0] == 1
        assert run.costs.iloc[0] == pytest.approx(0.001)
        np.testing.assert_allclose(run.net, (1 + run.gross) * (1 - run.costs) - 1, atol=1e-12)
    last = result.signals[result.signals.signal_close == ctx.asset_returns.index[-1]]
    assert len(last) == 5 and not last.filled.any()  # terminal signal recorded, never filled
    assert result.diagnostics.signal_close.is_unique
    assert result.spells.iloc[-1].right_censored
    assert result.stability.iloc[0].n_overlap == 0
    assert (result.stability.iloc[1:].n_overlap > 0).all()


def test_future_mutation_cannot_change_signal_close_diagnostics_or_controls(monkeypatch):
    _stub_hmm(monkeypatch)
    ctx = _context()
    first = intelligence.run_regime_intelligence(ctx)
    cutoff = ctx.asset_returns.index[105]
    ctx.raw_features.loc[ctx.raw_features.index > cutoff] += 1000
    ctx.asset_returns.loc[ctx.asset_returns.index > cutoff, "equity"] = -0.02
    second = intelligence.run_regime_intelligence(ctx)
    for field in ("diagnostics", "signals", "feature_means", "stability", "persistence"):
        before, after = getattr(first, field), getattr(second, field)
        pd.testing.assert_frame_equal(before[before.signal_close <= cutoff], after[after.signal_close <= cutoff])


def test_bound_rejects_experiment_before_any_strategy_or_fit(monkeypatch):
    fits = _stub_hmm(monkeypatch)
    ctx = _context()
    monkeypatch.setattr(intelligence, "MAX_SIGNAL_CLOSES", 1)
    started = []
    with pytest.raises(ValueError, match="at most 1 HMM fits"):
        intelligence.run_regime_intelligence(ctx, on_strategy_start=started.append)
    assert started == []
    assert fits == []


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_regime_intelligence.py"
    spec = importlib.util.spec_from_file_location("regime_intelligence_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_dataset_failure_is_registered_without_an_invented_hash(tmp_path, monkeypatch):
    cli = _load_cli()
    # Limit writes to the test directory while keeping imports tied to the actual worktree.
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    missing = tmp_path / "missing.csv"
    cli._register_failure(missing, tmp_path / "registry", {"started_strategies": []}, "missing input", 0)
    records = [json.loads(line) for line in (tmp_path / "registry" / f"{intelligence.EXPERIMENT_ID}.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "failed" and records[0]["n_trials"] == 0
    assert records[0]["dataset_sha256_canonical"] is None
    assert records[0]["dataset_path"] == missing.as_posix()


def test_output_accounting_and_comparisons_keep_the_engine_sample(tmp_path, monkeypatch):
    _stub_hmm(monkeypatch)
    ctx = _context()
    result = intelligence.run_regime_intelligence(ctx)
    cli = _load_cli()
    table = cli._write_results(result, ctx, tmp_path)
    assert set(table.strategy) == set(intelligence.STRATEGY_IDS)
    assert table.n_obs.nunique() == 1
    accounting = pd.read_csv(tmp_path / "daily_accounting.csv", parse_dates=["date"])
    for strategy, run in result.runs.items():
        saved = accounting[accounting.strategy == strategy].set_index("date")
        assert pd.DatetimeIndex(saved.index).equals(run.index)
        np.testing.assert_allclose(saved["net"], run.net, atol=1e-14)
        np.testing.assert_allclose(saved["cost"], run.costs, atol=1e-14)
        np.testing.assert_allclose(saved["turnover"], run.turnover, atol=1e-14)
    comparisons = pd.read_csv(tmp_path / "same_index_comparisons.csv")
    assert set(comparisons.reference) == {"hmm_fixed", "static_6040", "equal_weight"}
    assert (comparisons.n_obs == len(result.runs["hmm_fixed"].index)).all()


def test_cli_partial_failure_records_started_trials_without_success_manifest(tmp_path, monkeypatch):
    cli = _load_cli()
    ctx = _context()
    dataset = tmp_path / "input.csv"
    ctx.prices.to_csv(dataset)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "regime_shift", SimpleNamespace(__file__=str(tmp_path / "src" / "regime_shift" / "__init__.py")))
    monkeypatch.setattr(cli, "load_market_data_csv", lambda **kwargs: ctx.prices)
    monkeypatch.setattr(cli, "build_context", lambda prices, config: ctx)

    def fail(context, on_strategy_start):
        on_strategy_start("hmm_fixed")
        on_strategy_start("momentum_volatility")
        raise RuntimeError("injected second-strategy failure")

    monkeypatch.setattr(cli, "run_regime_intelligence", fail)
    assert cli.main(["--data-path", str(dataset)]) == 1
    records = [json.loads(line) for line in (tmp_path / "results" / "registry" / f"{intelligence.EXPERIMENT_ID}.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["status"] == "failed"
    assert records[0]["n_trials"] == 2
    assert records[0]["summary"]["started_strategies"] == ["hmm_fixed", "momentum_volatility"]
    directory = tmp_path / records[0]["summary"]["output_dir"]
    assert not (directory / "manifest.json").exists()
    failure = json.loads((directory / "failure.json").read_text())
    assert "injected second-strategy failure" in failure["error"]
