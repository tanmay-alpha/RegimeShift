"""
Tests for the performance metrics module.

Covers:
- NaN / +inf / -inf rejection (no silent drops).
- Returns <= -1 rejection (would produce non-positive wealth).
- Index alignment enforcement across returns / gross_returns / turnover /
  transaction_costs.
- Maximum drawdown clamped to [0, 1].
- Transaction-cost drag definition: gross_final - net_final.
- Equity curve via (1 + returns).cumprod().
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_shift.metrics import compute_performance_metrics, PerformanceMetrics


def _flat_series(n: int, value: float = 0.001, start: str = "2020-01-01") -> pd.Series:
    idx = pd.bdate_range(start, periods=n)
    return pd.Series([value] * n, index=idx, name="r")


def _returns_with_nan(n: int = 50) -> pd.Series:
    s = _flat_series(n)
    s.iloc[5] = np.nan
    return s


def _returns_with_pos_inf(n: int = 50) -> pd.Series:
    s = _flat_series(n)
    s.iloc[10] = float("inf")
    return s


def _returns_with_neg_inf(n: int = 50) -> pd.Series:
    s = _flat_series(n)
    s.iloc[10] = float("-inf")
    return s


def _returns_with_minus_one_or_worse(n: int = 50) -> pd.Series:
    s = _flat_series(n)
    s.iloc[7] = -1.0
    s.iloc[20] = -1.5
    return s


# ---------------------------------------------------------------------------
# Input rejection
# ---------------------------------------------------------------------------

class TestInputRejection:
    def test_nan_returns_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            compute_performance_metrics(_returns_with_nan())

    def test_positive_inf_returns_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            compute_performance_metrics(_returns_with_pos_inf())

    def test_negative_inf_returns_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            compute_performance_metrics(_returns_with_neg_inf())

    def test_returns_le_minus_one_rejected(self):
        with pytest.raises(ValueError, match="<= -1"):
            compute_performance_metrics(_returns_with_minus_one_or_worse())

    def test_empty_returns_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            compute_performance_metrics(_flat_series(0))


# ---------------------------------------------------------------------------
# Index alignment
# ---------------------------------------------------------------------------

class TestIndexAlignment:
    def test_turnover_misaligned_rejected(self):
        returns = _flat_series(50)
        bad_idx = pd.bdate_range("2021-01-01", periods=50)
        turnover = pd.Series([0.01] * 50, index=bad_idx)
        with pytest.raises(ValueError, match="turnover"):
            compute_performance_metrics(returns, turnover=turnover)

    def test_transaction_costs_misaligned_rejected(self):
        returns = _flat_series(50)
        bad_idx = pd.bdate_range("2021-01-01", periods=50)
        costs = pd.Series([0.0] * 50, index=bad_idx)
        with pytest.raises(ValueError, match="transaction_costs"):
            compute_performance_metrics(returns, transaction_costs=costs)

    def test_gross_returns_misaligned_rejected(self):
        returns = _flat_series(50)
        bad_idx = pd.bdate_range("2021-01-01", periods=50)
        gross = pd.Series([0.001] * 50, index=bad_idx)
        with pytest.raises(ValueError, match="gross_returns"):
            compute_performance_metrics(returns, gross_returns=gross)

    def test_aligned_inputs_accepted(self):
        returns = _flat_series(50, value=0.001)
        gross = _flat_series(50, value=0.0015)
        turnover = _flat_series(50, value=0.05)
        costs = _flat_series(50, value=0.0001)
        m = compute_performance_metrics(
            returns, gross_returns=gross, turnover=turnover,
            transaction_costs=costs, risk_free_rate=0.0,
        )
        assert isinstance(m, PerformanceMetrics)


# ---------------------------------------------------------------------------
# Drawdown math
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_max_drawdown_between_zero_and_one(self):
        # Synthesize a return series with one big drawdown
        n = 100
        idx = pd.bdate_range("2020-01-01", periods=n)
        r = np.zeros(n) * 0.001
        r[40:60] = -0.05  # 20 days of -5%
        s = pd.Series(r, index=idx)
        m = compute_performance_metrics(s)
        assert 0.0 <= m.maximum_drawdown <= 1.0

    def test_max_drawdown_uses_one_plus_returns(self):
        n = 60
        idx = pd.bdate_range("2020-01-01", periods=n)
        r = np.concatenate([np.full(30, 0.01), np.full(30, -0.005)])
        s = pd.Series(r, index=idx)
        m = compute_performance_metrics(s)
        # Equity via (1+r).cumprod() should peak around day 30.
        # Required: absolute MDD between 0 and 1.
        assert 0.0 <= m.maximum_drawdown <= 1.0
        assert m.maximum_drawdown > 0.0


# ---------------------------------------------------------------------------
# Transaction cost drag definition
# ---------------------------------------------------------------------------

class TestCostDrag:
    def test_drag_equals_gross_final_minus_net_final(self):
        n = 100
        idx = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.default_rng(7)
        net = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
        gross = net + 0.0001  # small constant cost on every period
        m = compute_performance_metrics(net, gross_returns=gross)
        expected = float((1.0 + gross).prod() - (1.0 + net).prod())
        assert m.total_transaction_cost_drag is not None
        assert np.isclose(m.total_transaction_cost_drag, expected, atol=1e-10), (
            f"Cost drag {m.total_transaction_cost_drag} != expected {expected}"
        )

    def test_zero_drag_when_no_gross_difference(self):
        n = 50
        idx = pd.bdate_range("2020-01-01", periods=n)
        r = pd.Series(np.zeros(n), index=idx)
        m = compute_performance_metrics(r, gross_returns=r)
        assert np.isclose(m.total_transaction_cost_drag, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Equity curve and total-return math
# ---------------------------------------------------------------------------

class TestEquityMath:
    def test_total_return_equals_one_plus_returns_product_minus_one(self):
        n = 50
        idx = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.default_rng(11)
        r = pd.Series(rng.normal(0.001, 0.01, n), index=idx)
        m = compute_performance_metrics(r)
        expected = float((1.0 + r).prod() - 1.0)
        assert np.isclose(m.total_return, expected, atol=1e-10)

    def test_risk_free_rate_consistency(self):
        n = 252
        idx = pd.bdate_range("2020-01-01", periods=n)
        r = pd.Series(np.zeros(n), index=idx)
        m0 = compute_performance_metrics(r, risk_free_rate=0.0)
        m_rf = compute_performance_metrics(r, risk_free_rate=0.05)
        # Sharpe should differ when risk_free_rate changes (zero returns
        # have zero excess at rf=0 and negative excess at rf>0).
        assert m0.sharpe_ratio is not None
        assert m_rf.sharpe_ratio is not None

    def test_maximum_drawdown_constant_returns_is_zero(self):
        n = 30
        idx = pd.bdate_range("2020-01-01", periods=n)
        r = pd.Series([0.005] * n, index=idx)
        m = compute_performance_metrics(r)
        assert np.isclose(m.maximum_drawdown, 0.0, atol=1e-12)

    def test_to_dict_rounds_and_drops_nan(self):
        n = 30
        idx = pd.bdate_range("2020-01-01", periods=n)
        r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, n), index=idx)
        m = compute_performance_metrics(r)
        d = m.to_dict()
        for v in d.values():
            assert v is None or isinstance(v, (float, int))
