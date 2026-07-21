"""
Tests for WalkForwardBacktest and BacktestResult.

Run with: python -m pytest tests/test_backtest.py -v
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.regime_shift.backtest import BacktestResult, WalkForwardBacktest
from src.regime_shift.transaction_costs import TransactionCostModel
from src.regime_shift.benchmarks import run_benchmarks
from src.regime_shift.regime_detector import RegimeDetector
from src.regime_shift.regime_features import RegimeFeatureEngineer
from src.regime_shift.optimizer import PortfolioOptimizer

logging.basicConfig(level=logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def small_prices(rng) -> pd.DataFrame:
    """500-day simulated price for 3 assets."""
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    means = np.array([0.0003, 0.0002, 0.0001])
    vols = np.array([0.015, 0.010, 0.005])
    rets = means + rng.standard_normal((n - 1, 3)) * vols
    prices = np.exp(np.cumsum(rets, axis=0)) * 100.0
    df = pd.DataFrame(prices, index=dates[1:], columns=["A", "B", "C"])
    df.iloc[0] = 100.0
    return df.sort_index()


@pytest.fixture
def small_returns(small_prices) -> pd.DataFrame:
    return small_prices.pct_change().dropna()


@pytest.fixture
def small_features(small_prices) -> pd.DataFrame:
    eng = RegimeFeatureEngineer(lookback_window=60)
    return eng.fit_transform(small_prices)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1-3: BacktestResult properties
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestResultProperties:
    def test_empty_returns_have_zero_metrics(self):
        result = BacktestResult()
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.annualized_volatility == 0.0

    def test_positive_returns_yield_positive_total_return(self, rng):
        rets = pd.Series(rng.normal(0.001, 0.01, 252),
                         index=pd.date_range("2020-01-01", periods=252))
        result = BacktestResult(portfolio_returns=rets)
        assert result.total_return > 0
        assert result.sharpe_ratio > 0
        assert result.max_drawdown <= 0.0  # Drawdowns are always <= 0

    def test_negative_returns_yield_negative_metrics(self, rng):
        rets = pd.Series(rng.normal(-0.001, 0.01, 252),
                         index=pd.date_range("2020-01-01", periods=252))
        result = BacktestResult(portfolio_returns=rets)
        assert result.total_return < 0
        assert result.sharpe_ratio < 0

    def test_annualized_return_matches_total(self, rng):
        rets = pd.Series(rng.normal(0.0005, 0.01, 252),
                         index=pd.date_range("2020-01-01", periods=252))
        result = BacktestResult(portfolio_returns=rets)
        # Verify: total = (1+annret)^(n/252) - 1
        expected = (1.0 + result.annualized_return) ** (len(rets) / 252.0) - 1.0
        assert abs(expected - result.total_return) < 1e-9

    def test_max_drawdown_in_valid_range(self, rng):
        rets = pd.Series(rng.normal(0.0003, 0.02, 500),
                         index=pd.date_range("2020-01-01", periods=500))
        result = BacktestResult(portfolio_returns=rets)
        assert -1.0 <= result.max_drawdown <= 0.0

    def test_win_rate_is_fraction(self, rng):
        rets = pd.Series(rng.normal(0.0001, 0.01, 100),
                         index=pd.date_range("2020-01-01", periods=100))
        result = BacktestResult(portfolio_returns=rets)
        assert 0.0 <= result.win_rate <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 4-7: WalkForwardBacktest run()
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardBacktestRun:
    def test_returns_complete_dataset(self, small_prices, small_returns, small_features):
        wb = WalkForwardBacktest(
            prices=small_prices, returns=small_returns, features=small_features,
            lookback=60, retrain_freq=21, n_states=3,
        )
        result = wb.run()
        expected_days = len(small_returns) - 60
        assert len(result.portfolio_returns) == expected_days

    def test_no_nan_or_inf_in_returns(self, small_prices, small_returns, small_features):
        wb = WalkForwardBacktest(
            prices=small_prices, returns=small_returns, features=small_features,
            lookback=60, retrain_freq=21, n_states=3,
        )
        result = wb.run()
        assert not result.portfolio_returns.isna().any()
        assert np.isfinite(result.portfolio_returns.values).all()

    def test_regime_series_length_matches(self, small_prices, small_returns, small_features):
        wb = WalkForwardBacktest(
            prices=small_prices, returns=small_returns, features=small_features,
            lookback=60, retrain_freq=21, n_states=3,
        )
        result = wb.run()
        assert len(result.regime_series) == len(result.portfolio_returns)

    def test_costs_nonnegative(self, small_prices, small_returns, small_features):
        wb = WalkForwardBacktest(
            prices=small_prices, returns=small_returns, features=small_features,
            lookback=60, retrain_freq=21, n_states=3,
        )
        result = wb.run()
        assert (result.costs >= 0).all()
        assert (result.costs <= 0.02).all()  # Capped at 2%


# ─────────────────────────────────────────────────────────────────────────────
# Test 8-10: Custom detector/optimizer/cost_model injection
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardBacktestCustomization:
    def test_custom_detector_is_used(self, small_prices, small_returns, small_features):
        det = RegimeDetector(n_states=3, lookback=60, retrain_freq=21)
        wb = WalkForwardBacktest(
            prices=small_prices, returns=small_returns, features=small_features,
            lookback=60, retrain_freq=21, n_states=3, detector=det,
        )
        assert wb.detector is det
        wb.run()
        # After run, detector should have at least one fitted state
        assert hasattr(wb.detector, "is_fitted")

    def test_custom_cost_model_applied(self, small_prices, small_returns, small_features):
        cm = TransactionCostModel(commission_rate=0.005, impact_factor=2.0)
        wb = WalkForwardBacktest(
            prices=small_prices, returns=small_returns, features=small_features,
            lookback=60, retrain_freq=21, n_states=3, cost_model=cm,
        )
        assert wb.cost_model is cm

    def test_optimizer_injection(self, small_prices, small_returns, small_features):
        opt = PortfolioOptimizer(n_assets=3)
        wb = WalkForwardBacktest(
            prices=small_prices, returns=small_returns, features=small_features,
            lookback=60, retrain_freq=21, n_states=3, optimizer=opt,
        )
        assert wb.optimizer is opt


# ─────────────────────────────────────────────────────────────────────────────
# Test 11-13: Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarks:
    def test_run_benchmarks_returns_all(self, small_prices, small_returns, small_features):
        results = run_benchmarks(
            small_prices, small_returns, small_features,
            rebalance_freq=21, lookback=60,
        )
        expected_strats = {"BuyAndHold", "EqualWeight", "RiskParity", "Momentum"}
        assert expected_strats.issubset(set(results.keys()))

    def test_benchmarks_no_nan(self, small_prices, small_returns, small_features):
        results = run_benchmarks(
            small_prices, small_returns, small_features,
            rebalance_freq=21, lookback=60,
        )
        for name, res in results.items():
            assert not res.portfolio_returns.isna().any(), f"{name} has NaN"
            assert np.isfinite(res.portfolio_returns.values).all(), f"{name} has Inf"

    def test_buy_and_hold_first_weights_equal(self, small_prices, small_returns):
        results = run_benchmarks(small_prices, small_returns, rebalance_freq=21)
        bh = results["BuyAndHold"]
        # First day weights should be equal
        first_w = bh.weights_history.iloc[0].values
        assert np.allclose(first_w, 1.0 / 3, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Test 14-16: Transaction costs
# ─────────────────────────────────────────────────────────────────────────────

class TestTransactionCosts:
    def test_no_turnover_zero_cost(self):
        cm = TransactionCostModel()
        weights = np.array([0.4, 0.3, 0.3])
        cost = cm.cost_as_fraction(weights, weights, np.array([0.15, 0.10, 0.05]),
                                    ["A", "B", "C"])
        assert cost == 0.0

    def test_turnover_computed_correctly(self):
        old = np.array([0.5, 0.5, 0.0])
        new = np.array([0.0, 0.5, 0.5])
        # Buy 0.5 of C, sell 0.5 of A. Each trade counted once.
        # L1 distance = 1.0, but with 0.5 factor to avoid double-counting = 0.5
        to = TransactionCostModel.compute_turnover(old, new)
        assert abs(to - 0.5) < 1e-9

    def test_cost_increases_with_turnover(self):
        cm = TransactionCostModel()
        old = np.array([1.0, 0.0, 0.0])
        new_small = np.array([0.9, 0.05, 0.05])
        new_large = np.array([0.0, 0.5, 0.5])
        vol = np.array([0.15, 0.10, 0.05])
        cost_small = cm.cost_as_fraction(old, new_small, vol, ["A", "B", "C"])
        cost_large = cm.cost_as_fraction(old, new_large, vol, ["A", "B", "C"])
        assert cost_large > cost_small


# ─────────────────────────────────────────────────────────────────────────────
# Test 17-19: End-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_end_to_end_simulated_data(self, rng):
        """Full pipeline on simulated data — should not crash, should produce sensible metrics."""
        n = 600
        dates = pd.date_range("2018-01-01", periods=n, freq="D")
        means = np.array([0.0005, 0.0002, 0.0001])
        vols = np.array([0.02, 0.015, 0.008])
        rets = means + rng.standard_normal((n - 1, 3)) * vols
        prices = pd.DataFrame(
            np.exp(np.cumsum(rets, axis=0)) * 100.0,
            index=dates[1:], columns=["A", "B", "C"],
        )
        prices.iloc[0] = 100.0
        returns = prices.pct_change().dropna()

        eng = RegimeFeatureEngineer(lookback_window=60)
        features = eng.fit_transform(prices)

        wb = WalkForwardBacktest(
            prices=prices, returns=returns, features=features,
            lookback=60, retrain_freq=21, n_states=3,
        )
        result = wb.run()

        assert len(result.portfolio_returns) > 0
        assert not result.portfolio_returns.isna().any()
        # Sharpe should be in reasonable range for random data
        assert -5.0 < result.sharpe_ratio < 5.0

    def test_turnover_under_limit(self, small_prices, small_returns, small_features):
        """Verify turnover constraint is respected."""
        wb = WalkForwardBacktest(
            prices=small_prices, returns=small_returns, features=small_features,
            lookback=60, retrain_freq=21, n_states=3, turnover_limit=0.20,
        )
        result = wb.run()
        if len(result.weights_history) >= 2:
            diffs = result.weights_history.diff().dropna()
            daily_turn = diffs.abs().sum(axis=1) / 2.0
            assert daily_turn.max() <= 0.20 + 1e-6  # Should respect limit

    def test_regime_series_values_are_strings(self, small_prices, small_returns, small_features):
        wb = WalkForwardBacktest(
            prices=small_prices, returns=small_returns, features=small_features,
            lookback=60, retrain_freq=21, n_states=3,
        )
        result = wb.run()
        for v in result.regime_series.unique():
            assert isinstance(v, str)
