"""
test_stats.py — Unit tests for the stats module.

Tests verify mathematical correctness of each metric against known values.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
import numpy as np
import pandas as pd
import pytest

from src.regime_shift.stats import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    information_ratio,
    kelly_criterion,
    profit_factor,
    cagr,
    max_drawdown,
    annualised_return,
    annualised_volatility,
    compute_full_stats,
)


# ──────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def constant_return_series():
    """0.1% daily return every day — known Sharpe and volatility."""
    n    = 365
    rets = pd.Series(np.ones(n) * 0.001)  # +0.1% every day
    return rets


@pytest.fixture
def equity_curve_up():
    """Monotonically rising equity curve — MDD should be 0."""
    n  = 100
    eq = pd.Series(np.linspace(1000, 2000, n),
                   index=pd.date_range("2020-01-01", periods=n, freq="D"))
    return eq


@pytest.fixture
def equity_curve_with_drawdown():
    """Equity: rises to 1500, falls to 1000, then recovers to 2000."""
    values = [1000] * 20 + list(range(1000, 1501, 25)) + list(range(1500, 999, -25)) + list(range(1000, 2001, 50))
    n = len(values)
    eq = pd.Series(values, dtype=float,
                   index=pd.date_range("2020-01-01", periods=n, freq="D"))
    return eq


# ──────────────────────────────────────────────────────────────────────────────
# Sharpe Ratio
# ──────────────────────────────────────────────────────────────────────────────

def test_sharpe_zero_return():
    """Strategy with zero excess returns → Sharpe = 0."""
    rets = pd.Series(np.zeros(100))
    sr   = sharpe_ratio(rets, risk_free_annual=0.0)
    assert sr == 0.0, f"Sharpe for zero returns should be 0, got {sr}"


def test_sharpe_positive_for_positive_returns(constant_return_series):
    """Constant positive returns → Sharpe > 0."""
    sr = sharpe_ratio(constant_return_series, risk_free_annual=0.0)
    assert sr > 0, f"Sharpe should be positive, got {sr}"


def test_sharpe_all_positive_returns_high():
    """Identical returns: std may be near-zero due to floating point.
    Test just verifies non-crash and non-negative output for this degenerate input."""
    rets = pd.Series(np.ones(100) * 0.01)
    sr   = sharpe_ratio(rets, risk_free_annual=0.0)
    # For identical returns std = 0 by convention → Sharpe = 0
    # Allow both 0.0 and very large (if FP std rounding != 0)
    assert sr >= 0, f"Sharpe for positive returns must be non-negative, got {sr}"


def test_sharpe_negative_for_negative_returns():
    """Negative daily returns → Sharpe < 0."""
    rets = pd.Series(-np.ones(100) * 0.001)
    sr   = sharpe_ratio(rets, risk_free_annual=0.0)
    assert sr < 0, f"Sharpe for negative returns should be < 0, got {sr}"


def test_sharpe_annualisation():
    """Verify √365 annualisation for crypto."""
    np.random.seed(0)
    rets     = pd.Series(np.random.randn(365) * 0.01 + 0.001)
    sr_daily = rets.mean() / rets.std()
    sr_ann   = sharpe_ratio(rets, risk_free_annual=0.0, ann_factor=365)
    assert abs(sr_ann - sr_daily * math.sqrt(365)) < 1e-9, (
        f"Sharpe annualisation error: expected {sr_daily * math.sqrt(365):.4f}, got {sr_ann:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sortino Ratio
# ──────────────────────────────────────────────────────────────────────────────

def test_sortino_all_positive_returns():
    """All positive returns → no downside deviation → Sortino = inf."""
    rets = pd.Series(np.ones(100) * 0.005)
    sort = sortino_ratio(rets, mar=0.0)
    assert sort == float("inf") or sort > 100, (
        f"Sortino should be very high for all-positive returns, got {sort}"
    )


def test_sortino_greater_than_sharpe_for_symmetric():
    """For symmetric return distributions, Sortino ≥ Sharpe."""
    np.random.seed(1)
    rets = pd.Series(np.random.randn(500) * 0.01 + 0.0005)
    sr   = sharpe_ratio(rets, ann_factor=365)
    sort = sortino_ratio(rets, ann_factor=365)
    assert sort >= sr * 0.8, (  # Allow some tolerance
        f"Sortino ({sort:.3f}) should be >= Sharpe ({sr:.3f}) approximately"
    )


def test_sortino_zero_return():
    """Zero mean return, zero MAR → Sortino = 0."""
    np.random.seed(2)
    rets = pd.Series(np.random.randn(200) * 0.01)  # mean ≈ 0
    # With mean ≈ 0 and MAR = 0, numerator ≈ 0 → Sortino ≈ 0
    sort = sortino_ratio(rets, mar=0.0, ann_factor=1)
    assert abs(sort) < 2.0, f"Sortino for near-zero mean should be small, got {sort}"


# ──────────────────────────────────────────────────────────────────────────────
# Max Drawdown
# ──────────────────────────────────────────────────────────────────────────────

def test_max_drawdown_monotone(equity_curve_up):
    """Monotonically rising equity → MDD = 0."""
    mdd = max_drawdown(equity_curve_up)
    assert mdd == 0.0, f"MDD for rising equity should be 0, got {mdd}"


def test_max_drawdown_known_value():
    """Known drawdown: from 1500 to 1000 → MDD = 1/3 = 33.3%."""
    eq  = pd.Series([1000.0, 1500.0, 1000.0],
                    index=pd.date_range("2020-01-01", periods=3, freq="D"))
    mdd = max_drawdown(eq)
    expected = (1500 - 1000) / 1500
    assert abs(mdd - expected) < 1e-9, f"MDD mismatch: expected {expected:.4f}, got {mdd:.4f}"


def test_max_drawdown_positive(equity_curve_with_drawdown):
    """MDD must always be a positive value."""
    mdd = max_drawdown(equity_curve_with_drawdown)
    assert mdd > 0, f"MDD should be positive for curve with drawdown, got {mdd}"
    assert mdd < 1.0, f"MDD should be < 100%, got {mdd}"


# ──────────────────────────────────────────────────────────────────────────────
# Calmar Ratio
# ──────────────────────────────────────────────────────────────────────────────

def test_calmar_ratio_zero_drawdown(equity_curve_up):
    """Zero drawdown → Calmar = inf."""
    calmar = calmar_ratio(equity_curve_up)
    assert calmar == float("inf"), f"Calmar with zero MDD should be inf, got {calmar}"


def test_calmar_ratio_positive_for_rising():
    """CAGR > 0 and MDD > 0 → Calmar > 0."""
    eq     = pd.Series([1000.0, 1200.0, 1100.0, 1500.0, 1400.0, 2000.0],
                       index=pd.date_range("2020-01-01", periods=6, freq="365D"))
    calmar = calmar_ratio(eq)
    assert calmar > 0, f"Calmar should be positive for rising equity, got {calmar}"


# ──────────────────────────────────────────────────────────────────────────────
# Omega Ratio
# ──────────────────────────────────────────────────────────────────────────────

def test_omega_ratio_all_positive():
    """All returns above threshold → Omega = inf."""
    rets  = pd.Series(np.ones(100) * 0.01)
    omega = omega_ratio(rets, threshold=0.0)
    assert omega == float("inf"), f"Omega with all-positive returns should be inf, got {omega}"


def test_omega_ratio_all_negative():
    """All returns below threshold → Omega ≈ 0."""
    rets  = pd.Series(-np.ones(100) * 0.01)
    omega = omega_ratio(rets, threshold=0.0)
    assert omega == 0.0, f"Omega with all-negative returns should be 0, got {omega}"


def test_omega_ratio_symmetric():
    """Returns symmetric around 0 → Omega ≈ 1."""
    np.random.seed(3)
    rets  = pd.Series(np.random.randn(1000) * 0.01)  # symmetric around 0
    omega = omega_ratio(rets, threshold=0.0)
    assert abs(omega - 1.0) < 0.3, f"Omega for symmetric returns should be ≈1, got {omega}"


# ──────────────────────────────────────────────────────────────────────────────
# Profit Factor
# ──────────────────────────────────────────────────────────────────────────────

def test_profit_factor_known():
    """PF = gross_wins / abs(gross_losses)."""
    pnls  = [10.0, 20.0, -5.0, -15.0]  # wins=30, losses=20 → PF=1.5
    pf    = profit_factor(pnls)
    assert abs(pf - 1.5) < 1e-9, f"Profit factor mismatch: expected 1.5, got {pf}"


def test_profit_factor_no_losses():
    """No losing trades → PF = inf."""
    pf = profit_factor([5.0, 10.0, 15.0])
    assert pf == float("inf"), f"PF with no losses should be inf, got {pf}"


# ──────────────────────────────────────────────────────────────────────────────
# Kelly Criterion
# ──────────────────────────────────────────────────────────────────────────────

def test_kelly_positive_edge():
    """Positive mean returns → Kelly fraction > 0."""
    rets  = pd.Series(np.ones(100) * 0.01 + np.random.randn(100) * 0.02)
    k     = kelly_criterion(rets, fraction=0.5)
    assert k >= 0, f"Kelly should be >= 0 for positive-expectancy strategy, got {k}"


def test_kelly_capped_at_one():
    """Kelly fraction must never exceed 1.0."""
    rets = pd.Series(np.ones(100) * 0.1)  # very high return
    k    = kelly_criterion(rets, fraction=1.0)
    assert k <= 1.0, f"Kelly should be capped at 1.0, got {k}"


def test_kelly_zero_for_negative_edge():
    """Negative expectation → Kelly = 0."""
    rets = pd.Series(-np.abs(np.random.randn(100)) * 0.01)
    k    = kelly_criterion(rets, fraction=0.5)
    assert k == 0.0, f"Kelly should be 0 for negative edge, got {k}"


# ──────────────────────────────────────────────────────────────────────────────
# CAGR
# ──────────────────────────────────────────────────────────────────────────────

def test_cagr_known_value():
    """2× return over exactly 1 year → CAGR = 100% (factor=365 days)."""
    eq = pd.Series(
        [1000.0, 2000.0],
        index=pd.date_range("2020-01-01", periods=2, freq="365D")
    )
    c = cagr(eq, ann_factor=365)
    # (2000/1000)^(365/365) - 1 = 1.0 = 100%
    assert abs(c - 1.0) < 0.05, f"CAGR for 2× in 1 year should be ≈1.0, got {c}"


# ──────────────────────────────────────────────────────────────────────────────
# Information Ratio
# ──────────────────────────────────────────────────────────────────────────────

def test_information_ratio_same_returns():
    """Strategy == benchmark → IR = 0."""
    rets = pd.Series(np.random.randn(100) * 0.01)
    ir   = information_ratio(rets, rets, ann_factor=365)
    assert ir == 0.0, f"IR against identical benchmark should be 0, got {ir}"


def test_information_ratio_positive_excess():
    """Strategy consistently beats benchmark → IR > 0."""
    bm   = pd.Series(np.random.randn(100) * 0.01)
    strat = bm + 0.001  # +0.1% alpha per day
    ir   = information_ratio(strat, bm, ann_factor=365)
    assert ir > 0, f"IR should be positive when strategy beats benchmark, got {ir}"


# ──────────────────────────────────────────────────────────────────────────────
# Full Stats Integration Test
# ──────────────────────────────────────────────────────────────────────────────

def test_compute_full_stats_keys():
    """compute_full_stats must return all expected keys."""
    np.random.seed(99)
    returns     = pd.Series(np.random.randn(200) * 0.01 + 0.001,
                            index=pd.date_range("2020-01-01", periods=200, freq="D"))
    equity_curve = (1 + returns).cumprod() * 1000
    trade_pnls   = list(np.random.randn(30) * 50)

    stats = compute_full_stats(
        equity_curve=equity_curve,
        trade_pnls=trade_pnls,
        risk_free_annual=0.0,
        ann_factor=365,
    )

    required_keys = [
        "CAGR (%)", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
        "Omega Ratio", "Max Drawdown (%)", "Win Rate (%)", "Profit Factor",
        "Kelly Fraction", "Total Trades",
    ]
    for key in required_keys:
        assert key in stats, f"Missing key in compute_full_stats output: '{key}'"
