"""Behavioral acceptance checks for the bounded, causal risk lab."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from regime_shift.config import RegimeShiftConfig
from regime_shift.execution import ExecutionCostModel
from regime_shift.research.evaluation import MisalignedSamplesError, require_common_index
from regime_shift.research.risk_lab import (
    COST_BPS,
    RiskLabConfig,
    equal_risk_contribution,
    estimate_covariance,
    historical_replay,
    hypothetical_stress,
    minimum_variance,
    risk_contributions,
    risk_lab_targets,
    trade_deltas,
    volatility_target_weights,
)
from regime_shift.research.strategies import build_context, run_target_strategy


@pytest.fixture
def context():
    rng = np.random.default_rng(71)
    returns = rng.normal(0.0002, [0.015, 0.009, 0.001], (300, 3))
    prices = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0),
                          index=pd.bdate_range("2019-07-01", periods=300),
                          columns=["equity", "gold", "bond"])
    cfg = RegimeShiftConfig(minimum_training_observations=80, train_window=120)
    cfg.execution_cost_model = ExecutionCostModel.flat(10, cfg.core_assets)
    return build_context(prices, cfg)


def test_erc_matches_inverse_volatility_on_diagonal_and_validates_correlated_risk():
    sd = np.array([0.03, 0.02, 0.01])
    diagonal = np.diag(sd ** 2)
    w = equal_risk_contribution(diagonal)
    np.testing.assert_allclose(w, (1 / sd) / (1 / sd).sum(), atol=0.002)
    corr = np.array([[1, 0.6, -0.2], [0.6, 1, 0.1], [-0.2, 0.1, 1]])
    cov = np.outer(sd, sd) * corr
    w = equal_risk_contribution(cov)
    np.testing.assert_allclose(risk_contributions(w, cov), [1 / 3] * 3, atol=0.002)
    assert w.min() > 0 and w.sum() == pytest.approx(1)


@pytest.mark.parametrize("cov", [np.zeros((3, 3)), np.diag([1, -1, 1]),
                                     np.full((3, 3), np.nan), np.ones((2, 3))])
def test_bad_covariance_is_rejected(cov):
    with pytest.raises(ValueError):
        equal_risk_contribution(cov)


def test_solver_success_is_rejected_if_contributions_are_wrong(monkeypatch):
    import cvxpy as cp

    def false_success(problem, **kwargs):
        problem._status = "optimal"
        problem.variables()[0].value = np.ones(3)

    monkeypatch.setattr(cp.Problem, "solve", false_success)
    with pytest.raises(RuntimeError, match="contribution validation"):
        equal_risk_contribution(np.diag([0.01, 0.02, 0.03]))


def test_lw_estimator_and_binding_minimum_variance_cap():
    rng = np.random.default_rng(4)
    window = pd.DataFrame(rng.normal(size=(100, 3)) * [0.01, 0.02, 0.0001])
    ridge = 1e-6
    cov = estimate_covariance(window, ridge, shrink=True)
    np.testing.assert_allclose(cov, LedoitWolf().fit(window).covariance_ + ridge * np.eye(3))
    w = minimum_variance(np.diag([0.01, 0.02, 0.000001]), cap=0.6)
    assert w[2] == pytest.approx(0.6, abs=1e-6)
    assert w.max() <= 0.6000001 and w.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="Infeasible"):
        minimum_variance(cov, cap=0.30)


def test_turnover_penalty_reduces_distance_to_actual_previous_holdings():
    cov = np.diag([0.0004, 0.0002, 0.00001])
    previous = np.array([0.5, 0.3, 0.2])
    plain = minimum_variance(cov, previous=previous)
    penalized = minimum_variance(cov, previous=previous, penalty=1.0)
    assert np.abs(penalized - previous).sum() < np.abs(plain - previous).sum() / 10
    # The initial allocation has no previous invested book to penalize.
    np.testing.assert_allclose(minimum_variance(cov, penalty=1), minimum_variance(cov), atol=1e-6)


def test_vol_target_uses_residual_bond_covariance_and_never_levers():
    cov = np.diag([0.25 ** 2, 0.20 ** 2, 0.02 ** 2]) / 252
    w = volatility_target_weights(cov, 2, 0.06)
    assert w[0] == pytest.approx(w[1]) and w[2] > 1 / 3
    assert w.min() >= 0 and w.sum() == pytest.approx(1)
    assert np.sqrt(w @ cov @ w * 252) == pytest.approx(0.06, abs=1e-8)
    np.testing.assert_allclose(volatility_target_weights(cov, 2, 1), [1 / 3] * 3)
    with pytest.raises(ValueError, match="infeasible"):
        volatility_target_weights(np.eye(3) / 252, 2, 0.01)


def test_every_variant_common_dates_initial_costs_and_genuine_trade_deltas(context):
    targets = risk_lab_targets(context)
    assert len(targets) == 7
    runs = [run_target_strategy(context, sid, target) for sid, target in targets.items()]
    require_common_index(runs)
    for run in runs:
        assert run.gross.iloc[0] == 0
        assert run.costs.iloc[0] == pytest.approx(0.001)
        assert run.turnover.iloc[0] == pytest.approx(1)
        delta = trade_deltas(run, context.asset_returns)
        np.testing.assert_allclose(run.costs, delta.abs().sum(axis=1) * 0.001, atol=1e-12)
        for bps in COST_BPS:
            costs = delta.abs().sum(axis=1) * bps / 10000
            net = (1 + run.gross) * (1 - costs) - 1
            assert net.iloc[0] == pytest.approx(-bps / 10000)
    with pytest.raises(MisalignedSamplesError):
        require_common_index([runs[0], runs[1].window(runs[1].index[1], runs[1].index[-1])])


def test_targets_only_use_history_through_signal(context):
    sid = "lw_min_variance"
    target = risk_lab_targets(context)[sid]
    cutoff = 100
    date = context.asset_returns.index[cutoff]
    past = context.asset_returns.iloc[:cutoff + 1].copy()
    before = target(date, context.raw_features.loc[:date], past, None)
    context.asset_returns.iloc[cutoff + 1:] = 0.9
    after = target(date, context.raw_features.loc[:date], context.asset_returns.loc[:date], None)
    np.testing.assert_allclose(before, after, atol=1e-8)


def test_engine_cost_sensitivity_preserves_gross_and_charges_actual_fills(context):
    runs = []
    for bps in COST_BPS:
        context.config.execution_cost_model = ExecutionCostModel.flat(bps, context.assets)
        run = run_target_strategy(context, "equal_weight", risk_lab_targets(context)["equal_weight"])
        delta = trade_deltas(run, context.asset_returns)
        np.testing.assert_allclose(run.costs, delta.abs().sum(axis=1) * bps / 10000, atol=1e-12)
        runs.append(run)
    require_common_index(runs)
    for run in runs[1:]:
        np.testing.assert_allclose(run.gross, runs[0].gross, atol=1e-12)
    assert (1 + runs[0].net).prod() > (1 + runs[-1].net).prod()


def test_next_close_target_cannot_capture_signal_to_fill_jump(context):
    first_signal = min(p for p in context.rebalance_positions if p + 1 >= context.min_history)
    context.asset_returns.iloc[first_signal + 1] = [0.5, 0, 0]
    run = run_target_strategy(context, "equity_only", lambda *args: [1, 0, 0])
    assert run.index[0] == context.asset_returns.index[first_signal + 1]
    assert run.gross.iloc[0] == 0
    assert run.gross.iloc[1] == pytest.approx(context.asset_returns.iloc[first_signal + 2, 0])


def test_historical_is_real_slice_and_hypothetical_is_separately_labelled(context):
    run = run_target_strategy(context, "equal_weight", risk_lab_targets(context)["equal_weight"])
    historical = historical_replay(run, context.cash_returns).set_index("scenario")
    assert historical.loc["euro_crisis_2011", "status"] == "unavailable"
    covid = run.window("2020-02-01", "2020-04-30")
    assert historical.loc["covid_2020", "window_return"] == pytest.approx((1 + covid.net).prod() - 1)
    stress = hypothetical_stress([0.5, 0.3, 0.2], np.diag([0.04, 0.02, 0.001]) / 252,
                                 [-0.2, -0.1, -0.02]).set_index("scenario")
    assert set(stress.evaluation_type) == {"hypothetical_terminal_snapshot"}
    assert stress.loc["correlation_shock", "ann_vol"] > stress.loc["reference", "ann_vol"]
    assert stress.loc["concentration_shock", "hhi"] > stress.loc["reference", "hhi"]
    assert stress.loc["reference", "assumed_one_period_return"] == pytest.approx(-0.134)


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        RiskLabConfig(turnover_penalty=float("nan")).validate(3)


def test_runner_registers_setup_failure_with_zero_trials(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/run_risk_lab.py"
    spec = importlib.util.spec_from_file_location("risk_lab_runner_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    records = []
    monkeypatch.setattr(runner, "run_experiment", lambda *args: (_ for _ in ()).throw(ValueError("bad data")))
    monkeypatch.setattr(runner, "register_attempt", lambda **kwargs: records.append(kwargs))
    monkeypatch.setattr(runner, "write_experiment_manifest", lambda *args, **kwargs: None)
    assert runner.main(["--output-dir", str(tmp_path / "outputs")]) == 1
    assert records[0]["status"] == "failed" and records[0]["n_trials"] == 0
    assert "bad data" in records[0]["error"]
    assert json.loads((tmp_path / "outputs/trials.json").read_text())["status"] == "failed"
