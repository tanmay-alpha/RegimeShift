"""Regression tests for the signal-at-close-t / fill-at-close-t+1 contract."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from regime_shift import backtest
from regime_shift.config import RegimeShiftConfig
from regime_shift.execution import ExecutionCostModel


def _prices(n=90):
    index = pd.bdate_range("2020-01-01", periods=n)
    # Jump A lands on the bar right after the first signal close (price row 62);
    # jump B lands well after the fill (price row 70).
    equity = np.ones(n) * 100.0
    equity[62:] = 110.0
    equity[70:] = 121.0
    return pd.DataFrame({"equity": equity, "gold": 100.0, "bond": 100.0}, index=index)


def _fast_model(monkeypatch):
    monkeypatch.setattr(backtest, "compute_raw_features", lambda prices, config: prices[["equity", "gold", "bond"]])
    monkeypatch.setattr(backtest, "drop_feature_warmup", lambda features, config: features)
    monkeypatch.setattr(backtest, "fit_feature_scaler", lambda train: object())
    monkeypatch.setattr(backtest, "transform_features", lambda scaler, train: train)
    monkeypatch.setattr(backtest, "fit_hmm", lambda *args, **kwargs: (object(), None, None))
    solution = SimpleNamespace(
        regime="Bull", probabilities=pd.Series([1.0, 0.0, 0.0], index=["Bull", "Bear", "Crisis"]),
        transition_matrix=pd.DataFrame(np.eye(3), index=["Bull", "Bear", "Crisis"], columns=["Bull", "Bear", "Crisis"]),
        convergence=True, n_iter=1, log_likelihood=0.0,
    )
    monkeypatch.setattr(backtest, "predict_current_state", lambda *args, **kwargs: solution)
    monkeypatch.setattr(backtest, "optimize_portfolio", lambda *args, **kwargs: SimpleNamespace(weights={"equity": 1.0, "gold": 0.0, "bond": 0.0}))


def test_signal_at_close_t_is_filled_at_close_t_plus_1(monkeypatch):
    _fast_model(monkeypatch)
    config = RegimeShiftConfig(minimum_training_observations=1, train_window=60, rebalance_frequency=60)
    result = backtest.run_walk_forward_backtest(_prices(), config, transaction_cost_bps=0)
    fill = result.event_log[result.event_log.rebalance_flag].iloc[0]
    prices = _prices()
    # The signal close is the bar before the fill close.
    assert prices.index.get_loc(fill.signal_date) + 1 == prices.index.get_loc(fill.execution_date)
    # Jump A accrues between signal close and fill close: a same-close fill would earn it, a t+1 fill must not.
    assert fill.execution_date == prices.index[62]
    assert result.gross_returns.loc[fill.execution_date] == 0.0
    # Once filled, the position earns subsequent returns (jump B).
    assert result.gross_returns.loc[prices.index[70]] == pytest.approx(0.10)


def test_cost_is_charged_on_execution_and_benchmark_has_same_order(monkeypatch):
    _fast_model(monkeypatch)
    config = RegimeShiftConfig(minimum_training_observations=1, train_window=60, rebalance_frequency=60)
    strategy = backtest.run_walk_forward_backtest(_prices(), config, transaction_cost_bps=10)
    execution = strategy.event_log[strategy.event_log.rebalance_flag].index[0]
    benchmark = backtest.run_benchmark(_prices(), {"equity": 1.0, "gold": 0.0, "bond": 0.0}, config, transaction_cost_bps=10, rebalance_flags=strategy.rebalance_flags, start_date=strategy.net_returns.index[0])
    assert strategy.transaction_costs.loc[execution] == pytest.approx(0.001)
    assert strategy.gross_returns.loc[execution] == benchmark.gross_returns.loc[execution] == 0.0
    assert strategy.net_returns.loc[execution] == pytest.approx(-0.001)
    assert benchmark.net_returns.loc[execution] == pytest.approx(-0.001)


def test_future_mutation_cannot_change_preceding_decisions(monkeypatch):
    _fast_model(monkeypatch)
    config = RegimeShiftConfig(minimum_training_observations=1, train_window=60, rebalance_frequency=60)
    original = _prices()
    altered = original.copy(); altered.iloc[-1, 0] = 1_000.0
    first = backtest.run_walk_forward_backtest(original, config, transaction_cost_bps=0)
    second = backtest.run_walk_forward_backtest(altered, config, transaction_cost_bps=0)
    boundary = altered.index[-1]
    pd.testing.assert_frame_equal(first.event_log.loc[:boundary].iloc[:-1], second.event_log.loc[:boundary].iloc[:-1])
