"""Regression tests for the protocol-v3 research layer.

Covers the confirmed audit flaws: paired comparisons on unequal samples,
benchmarks reported with zero turnover/cost inside sub-windows, degenerate
bootstrap Sharpe set to 0, and walk-forward switching costs.
"""

import numpy as np
import pandas as pd
import pytest

from regime_shift.execution import ExecutionCostModel
from regime_shift.research.evaluation import (
    MisalignedSamplesError,
    StrategyRun,
    metric_row,
    net_return_frame,
    require_common_index,
    window_table,
)
from regime_shift.research.stats import (
    cscv_pbo,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    holm_adjust,
    moving_block_indices,
    paired_block_bootstrap,
    probabilistic_sharpe_ratio,
)
from regime_shift.research.walk_forward import walk_forward_select


def _run(sid, gross, costs=None, turnover=None, weights=None):
    gross = pd.Series(gross, dtype=float)
    costs = pd.Series(0.0, index=gross.index) if costs is None else pd.Series(costs, index=gross.index, dtype=float)
    turnover = pd.Series(0.0, index=gross.index) if turnover is None else pd.Series(turnover, index=gross.index, dtype=float)
    net = (1 + gross) * (1 - costs) - 1
    return StrategyRun(sid, gross, net, costs, turnover, weights=weights)


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_comparison_on_unequal_samples_raises_instead_of_filling():
    a = _run("a", _series(np.full(30, 0.001)))
    b = _run("b", _series(np.full(29, 0.001), start="2020-01-02"))
    with pytest.raises(MisalignedSamplesError, match="dates only in"):
        require_common_index([a, b])
    with pytest.raises(MisalignedSamplesError):
        net_return_frame([a, b])
    with pytest.raises(MisalignedSamplesError):
        a.on_index(b.index.union(pd.DatetimeIndex(["2021-06-01"])))


def test_strategy_run_rejects_net_inconsistent_with_gross_and_costs():
    gross = _series([0.01, 0.02])
    with pytest.raises(ValueError, match="net !="):
        StrategyRun("x", gross, gross, gross * 0 + 0.001, gross * 0)


def test_window_attribution_keeps_benchmark_turnover_and_costs():
    """Sub-period rows must use the benchmark's genuine accounting, not zeros."""
    gross = _series(np.full(60, 0.001))
    costs = pd.Series(0.0, index=gross.index)
    turnover = pd.Series(0.0, index=gross.index)
    costs.iloc[[0, 21, 42]] = 0.0002
    turnover.iloc[[0, 21, 42]] = [1.0, 0.1, 0.1]
    bench = _run("static_6040", gross, costs, turnover)
    strat = _run("regimeshift", gross)
    table = window_table([strat, bench], [("late", str(gross.index[20].date()), str(gross.index[-1].date()))])
    row = table.set_index("strategy").loc["static_6040"]
    assert row["ann_turnover"] == pytest.approx(0.2 * 252 / 40)
    assert row["cost_drag"] > 0
    assert row["cagr"] < table.set_index("strategy").loc["regimeshift", "cagr"]
    assert set(table["evaluation_type"]) == {"chronological_attribution"}


def test_sharpe_vs_cash_subtracts_cash_proxy_daily():
    r = _series([0.002, 0.001, 0.003, 0.0, 0.002])
    cash = _series([0.001] * 5)
    row = metric_row(_run("x", r), cash_returns=cash)
    excess = r - cash
    assert row["sharpe_vs_cash"] == pytest.approx(excess.mean() / excess.std(ddof=1) * np.sqrt(252))


def test_holm_adjustment_matches_step_down_definition():
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])


def test_moving_blocks_are_contiguous_and_in_range():
    idx = moving_block_indices(100, 21, 50, np.random.default_rng(0))
    assert idx.shape == (50, 100) and idx.min() >= 0 and idx.max() <= 99
    assert np.all(np.diff(idx[:, :21], axis=1) == 1)


def test_bootstrap_rejects_nan_and_identical_strategies_are_not_significant():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0004, 0.01, 500), index=pd.bdate_range("2020-01-01", periods=500))
    frame = pd.DataFrame({"a": r, "b": r})
    table, _ = paired_block_bootstrap(frame, [("a", "b")], n_boot=200)
    assert (table["observed_diff"] == 0).all() and (table["p_value"] == 1).all()
    frame.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="never fill"):
        paired_block_bootstrap(frame, [("a", "b")], n_boot=10)


def test_bootstrap_detects_large_paired_difference():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, 1500)
    frame = pd.DataFrame({"good": noise + 0.002, "bad": noise - 0.002},
                         index=pd.bdate_range("2015-01-01", periods=1500))
    table, _ = paired_block_bootstrap(frame, [("good", "bad")], metrics=["sharpe"], n_boot=500)
    assert table.loc[0, "ci_lower"] > 0 and table.loc[0, "p_value_holm"] < 0.01


def test_psr_and_dsr_penalise_more_trials():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.01, 2000)
    assert probabilistic_sharpe_ratio(r, 0.0) > 0.9
    few = deflated_sharpe_ratio(r, [0.05, 0.0, -0.01])
    many = deflated_sharpe_ratio(r, list(rng.normal(0, 0.02, 200)))
    assert many["sr_threshold_per_period"] > few["sr_threshold_per_period"]
    assert many["dsr"] < few["dsr"]
    assert expected_max_sharpe([0.1]) == 0.0


def test_cscv_pbo_low_for_persistent_winner_and_high_for_noise():
    rng = np.random.default_rng(4)
    noise = pd.DataFrame(rng.normal(0, 0.01, (1600, 10)))
    persistent = noise.copy()
    persistent[0] += 0.003
    assert cscv_pbo(persistent, n_splits=8)["pbo"] < 0.05
    assert 0.2 < cscv_pbo(noise, n_splits=8)["pbo"] < 0.9


def test_walk_forward_selection_uses_only_prior_data_and_charges_switch_cost():
    index = pd.bdate_range("2015-01-01", "2020-12-31")
    assets = ["equity", "gold", "bond"]
    # "eq" wins 2015-2017, "bd" wins 2018-2020; the 2018 choice must still be "eq".
    eq_g = pd.Series(np.where(index.year <= 2017, 0.001, -0.001), index=index)
    eq_g[::7] -= 0.0005
    bd_g = pd.Series(np.where(index.year <= 2017, -0.0005, 0.002), index=index)
    bd_g[::5] -= 0.0004
    w_eq = pd.DataFrame([[1.0, 0.0, 0.0]] * len(index), index=index, columns=assets)
    w_bd = pd.DataFrame([[0.0, 0.0, 1.0]] * len(index), index=index, columns=assets)
    runs = {"eq": _run("eq", eq_g, weights=w_eq), "bd": _run("bd", bd_g, weights=w_bd)}
    asset_returns = pd.DataFrame(0.0, index=index, columns=assets)
    asset_returns["equity"] = 0.01  # the outgoing equity book earns this on the switch day

    result = walk_forward_select(runs, asset_returns, ExecutionCostModel.flat(10, assets), lookback_years=3)

    chosen = result.selections.set_index("evaluation_year")["selected"]
    assert chosen[2018] == "eq" and chosen[2020] == "bd"
    assert (pd.to_datetime(result.selections["validation_end"]) < pd.to_datetime(result.selections["fill_close"])).all()
    switch_year = next(y for y in (2019, 2020) if chosen[y] == "bd")
    switch_day = result.run.index[result.run.index.year == switch_year][0]
    assert result.run.gross[switch_day] == pytest.approx(0.01)
    assert result.run.costs[switch_day] == pytest.approx(2 * 10 / 10_000)
    assert result.run.turnover[switch_day] == pytest.approx(1.0)
    assert result.run.net[switch_day + pd.offsets.BDay(1)] == pytest.approx(bd_g[switch_day + pd.offsets.BDay(1)])


def test_all_variants_share_first_fill_and_index(monkeypatch):
    """Variants start on RegimeShift's first fill, so initial-allocation costs are inside every sample."""
    from tests.test_causal_execution import _fast_model

    from regime_shift.config import RegimeShiftConfig
    from regime_shift.research.strategies import run_core_suite

    _fast_model(monkeypatch)
    rng = np.random.default_rng(5)
    index = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 3)), axis=0)),
                          index=index, columns=["equity", "gold", "bond"])
    config = RegimeShiftConfig(minimum_training_observations=80, train_window=120, rebalance_frequency=21)
    config.execution_cost_model = ExecutionCostModel.flat(10, config.core_assets)
    runs, _, _ = run_core_suite(prices, config, variants=["vol_rule", "min_variance"])
    first = {sid: run.index[0] for sid, run in runs.items()}
    assert len(set(first.values())) == 1
    for run in runs.values():
        assert run.turnover.iloc[0] == pytest.approx(1.0)
        assert run.costs.iloc[0] == pytest.approx(10 / 10_000)
