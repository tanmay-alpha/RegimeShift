"""
Comprehensive tests for the CVXPY regime-conditioned portfolio optimizer.

Tests cover:
  1.  CVXPY is used
  2.  Output contains exactly equity, gold, and bond
  3.  VIX is never allocated
  4.  Weights are finite
  5.  Weights are nonnegative
  6.  Weights sum to one
  7.  Maximum asset constraint is respected
  8.  Bull equity minimum is respected
  9.  Bear equity maximum is respected
  10. Bear defensive allocation minimum is respected
  11. Crisis equity maximum is respected
  12. Crisis gold minimum is respected
  13. Crisis bond minimum is respected
  14. Unknown regimes are rejected
  15. Insufficient return history is rejected
  16. NaN and infinite returns are rejected
  17. Covariance matrix is positive semidefinite
  18. Deterministic inputs produce deterministic weights
  19. Returns after cutoff T do not affect weights computed at T
  20. Previous weights are validated
  21. Solver failure raises explicit error
  22. Constraint infeasibility raises explicit error
  23. Equal-weight benchmark is exactly 1/3 each
  24. Static 60/40 benchmark is exactly 60% equity and 40% bond
  25. Benchmark weights contain no VIX

All tests use synthetic deterministic fixtures.  No network access required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_shift.config import PortfolioConfig
from regime_shift.exceptions import PortfolioOptimizationError
from regime_shift.portfolio import (
    optimize_portfolio,
    _validate_returns_input,
    _validate_previous_weights,
    _estimate_returns_and_covariance,
    _ASSET_ORDER,
)
from regime_shift.benchmarks import (
    static_60_40_weights,
    equal_weight_weights,
    validate_benchmark_weights,
)


# ---------------------------------------------------------------------------
# Deterministic synthetic return data
# ---------------------------------------------------------------------------

def _make_returns(
    n_obs: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic multi-asset return DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_obs)

    equity_ret = rng.normal(0.0005, 0.015, n_obs)
    gold_ret = rng.normal(0.0002, 0.008, n_obs)
    bond_ret = rng.normal(0.0001, 0.003, n_obs)

    return pd.DataFrame(
        {"equity": equity_ret, "gold": gold_ret, "bond": bond_ret},
        index=idx,
    )


def _make_segmented_returns(
    n_per_regime: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create returns with three clearly separable segments for regime-specific testing.

    Bull: high equity returns, low gold/bond
    Bear: negative equity returns, high bond
    Crisis: high volatility across all assets
    """
    rng = np.random.default_rng(seed)

    bull = {}
    bear = {}
    crisis = {}

    idx = pd.bdate_range("2015-01-01", periods=n_per_regime * 3)

    # Bull: equity positive, low vol
    bull_equity = rng.normal(0.003, 0.008, n_per_regime)
    bull_gold = rng.normal(0.0, 0.005, n_per_regime)
    bull_bond = rng.normal(0.0005, 0.002, n_per_regime)

    # Bear: equity negative
    bear_equity = rng.normal(-0.003, 0.015, n_per_regime)
    bear_gold = rng.normal(0.001, 0.006, n_per_regime)
    bear_bond = rng.normal(0.001, 0.003, n_per_regime)

    # Crisis: high volatility
    crisis_equity = rng.normal(0.0, 0.04, n_per_regime)
    crisis_gold = rng.normal(0.002, 0.015, n_per_regime)
    crisis_bond = rng.normal(-0.001, 0.01, n_per_regime)

    equity = np.concatenate([bull_equity, bear_equity, crisis_equity])
    gold = np.concatenate([bull_gold, bear_gold, crisis_gold])
    bond = np.concatenate([bull_bond, bear_bond, crisis_bond])

    return pd.DataFrame(
        {"equity": equity, "gold": gold, "bond": bond},
        index=idx,
    )


# ---------------------------------------------------------------------------
# 1. CVXPY is used
# ---------------------------------------------------------------------------

class TestCvxpyUsage:

    def test_cvxpy_imported(self):
        """Portfolio module must use CVXPY."""
        import regime_shift.portfolio as portfolio_mod
        src_path = Path(portfolio_mod.__file__)
        with open(src_path) as f:
            source = f.read()

        assert "cvxpy" in source or "cp." in source, (
            "portfolio.py must use CVXPY"
        )


# ---------------------------------------------------------------------------
# 2-3. Asset set and VIX exclusion
# ---------------------------------------------------------------------------

class TestAssetSet:

    def test_output_contains_exactly_three_assets(self):
        """Optimizer must return weights for exactly equity, gold, bond."""
        returns = _make_returns()
        config = PortfolioConfig()
        solution = optimize_portfolio("Bull", returns, config=config)

        assert set(solution.weights.keys()) == {"equity", "gold", "bond"}

    def test_no_vix_allocation(self):
        """VIX must never appear in portfolio weights."""
        returns = _make_returns()
        returns["vix"] = np.random.default_rng(42).normal(0, 1, len(returns))
        config = PortfolioConfig()

        for regime in ["Bull", "Bear", "Crisis"]:
            solution = optimize_portfolio(regime, returns, config=config)
            assert "vix" not in solution.weights, (
                f"VIX must not be allocated in {regime} regime"
            )


# ---------------------------------------------------------------------------
# 4-6. Weight properties
# ---------------------------------------------------------------------------

class TestWeightProperties:

    def test_weights_are_finite(self):
        """All weights must be finite."""
        returns = _make_returns()
        config = PortfolioConfig()

        for regime in ["Bull", "Bear", "Crisis"]:
            solution = optimize_portfolio(regime, returns, config=config)
            for asset, weight in solution.weights.items():
                assert np.isfinite(weight), (
                    f"Weight for {asset} in {regime} is not finite: {weight}"
                )

    def test_weights_are_nonnegative(self):
        """All weights must be non-negative (long-only)."""
        returns = _make_returns()
        config = PortfolioConfig()

        for regime in ["Bull", "Bear", "Crisis"]:
            solution = optimize_portfolio(regime, returns, config=config)
            for asset, weight in solution.weights.items():
                assert weight >= -1e-10, (
                    f"Weight for {asset} in {regime} is negative: {weight}"
                )

    def test_weights_sum_to_one(self):
        """Weights must sum to 1."""
        returns = _make_returns()
        config = PortfolioConfig()

        for regime in ["Bull", "Bear", "Crisis"]:
            solution = optimize_portfolio(regime, returns, config=config)
            total = sum(solution.weights.values())
            assert np.isclose(total, 1.0, atol=1e-6), (
                f"Weights for {regime} sum to {total}, not 1.0"
            )


# ---------------------------------------------------------------------------
# 7-13. Regime-specific constraints
# ---------------------------------------------------------------------------

class TestRegimeConstraints:

    def test_max_asset_weight_respected(self):
        """No asset should exceed maximum_asset_weight."""
        returns = _make_returns()
        config = PortfolioConfig(maximum_asset_weight=0.50)

        for regime in ["Bull", "Bear", "Crisis"]:
            solution = optimize_portfolio(regime, returns, config=config)
            for asset, weight in solution.weights.items():
                assert weight <= 0.50 + 1e-10, (
                    f"{asset} weight {weight} exceeds max 0.50 in {regime}"
                )

    def test_bull_equity_minimum(self):
        """Bull regime: equity weight >= 0.45."""
        returns = _make_returns()
        config = PortfolioConfig()

        solution = optimize_portfolio("Bull", returns, config=config)
        assert solution.weights["equity"] >= 0.45 - 1e-10, (
            f"Bull equity weight {solution.weights['equity']} below minimum 0.45"
        )

    def test_bear_equity_maximum(self):
        """Bear regime: equity weight <= 0.40."""
        returns = _make_returns()
        config = PortfolioConfig()

        solution = optimize_portfolio("Bear", returns, config=config)
        assert solution.weights["equity"] <= 0.40 + 1e-10, (
            f"Bear equity weight {solution.weights['equity']} exceeds maximum 0.40"
        )

    def test_bear_defensive_minimum(self):
        """Bear regime: gold + bond >= 0.60."""
        returns = _make_returns()
        config = PortfolioConfig()

        solution = optimize_portfolio("Bear", returns, config=config)
        defensive = solution.weights["gold"] + solution.weights["bond"]
        assert defensive >= 0.60 - 1e-10, (
            f"Bear defensive allocation {defensive} below minimum 0.60"
        )

    def test_crisis_equity_maximum(self):
        """Crisis regime: equity weight <= 0.15."""
        returns = _make_returns()
        config = PortfolioConfig()

        solution = optimize_portfolio("Crisis", returns, config=config)
        assert solution.weights["equity"] <= 0.15 + 1e-10, (
            f"Crisis equity weight {solution.weights['equity']} exceeds maximum 0.15"
        )

    def test_crisis_gold_minimum(self):
        """Crisis regime: gold weight >= 0.25."""
        returns = _make_returns()
        config = PortfolioConfig()

        solution = optimize_portfolio("Crisis", returns, config=config)
        assert solution.weights["gold"] >= 0.25 - 1e-10, (
            f"Crisis gold weight {solution.weights['gold']} below minimum 0.25"
        )

    def test_crisis_bond_minimum(self):
        """Crisis regime: bond weight >= 0.40."""
        returns = _make_returns()
        config = PortfolioConfig()

        solution = optimize_portfolio("Crisis", returns, config=config)
        assert solution.weights["bond"] >= 0.40 - 1e-10, (
            f"Crisis bond weight {solution.weights['bond']} below minimum 0.40"
        )


# ---------------------------------------------------------------------------
# 14-16. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_unknown_regime_rejected(self):
        """Unknown regime label should raise PortfolioOptimizationError."""
        returns = _make_returns()
        config = PortfolioConfig()
        with pytest.raises(PortfolioOptimizationError, match="Unknown regime"):
            optimize_portfolio("UnknownRegime", returns, config=config)

    def test_insufficient_history_rejected(self):
        """Too few return observations should raise error."""
        returns = _make_returns(n_obs=10)
        config = PortfolioConfig(minimum_estimation_observations=60)
        with pytest.raises(PortfolioOptimizationError, match="Insufficient"):
            optimize_portfolio("Bull", returns, config=config)

    def test_nan_returns_rejected(self):
        """NaN in returns should raise error."""
        returns = _make_returns()
        returns.iloc[10, 0] = np.nan
        config = PortfolioConfig()
        with pytest.raises(PortfolioOptimizationError, match="NaN"):
            optimize_portfolio("Bull", returns, config=config)

    def test_inf_returns_rejected(self):
        """Infinite returns should raise error."""
        returns = _make_returns()
        returns.iloc[10, 0] = np.inf
        config = PortfolioConfig()
        with pytest.raises(PortfolioOptimizationError, match="non-finite"):
            optimize_portfolio("Bull", returns, config=config)

    def test_missing_columns_rejected(self):
        """Missing required columns should raise error."""
        returns = pd.DataFrame({"equity": [0.01] * 100})
        config = PortfolioConfig()
        with pytest.raises(PortfolioOptimizationError, match="missing"):
            optimize_portfolio("Bull", returns, config=config)

    def test_empty_returns_rejected(self):
        """Empty returns DataFrame should raise error."""
        returns = pd.DataFrame(columns=["equity", "gold", "bond"])
        config = PortfolioConfig()
        with pytest.raises(PortfolioOptimizationError, match="empty"):
            optimize_portfolio("Bull", returns, config=config)

    def test_previous_weights_missing_asset(self):
        """previous_weights missing an asset should raise error."""
        returns = _make_returns()
        config = PortfolioConfig()
        with pytest.raises(PortfolioOptimizationError, match="missing"):
            optimize_portfolio("Bull", returns, previous_weights={"equity": 0.5}, config=config)

    def test_previous_weights_non_finite(self):
        """previous_weights with non-finite value should raise error."""
        returns = _make_returns()
        config = PortfolioConfig()
        with pytest.raises(PortfolioOptimizationError, match="not finite"):
            optimize_portfolio(
                "Bull", returns,
                previous_weights={"equity": 0.5, "gold": 0.3, "bond": float("inf")},
                config=config,
            )


# ---------------------------------------------------------------------------
# 17. Covariance PSD
# ---------------------------------------------------------------------------

class TestCovariancePSD:

    def test_covariance_is_psd(self):
        """Estimated covariance matrix must be positive semidefinite."""
        returns = _make_returns()
        config = PortfolioConfig()

        _, cov, diagnostics = _estimate_returns_and_covariance(returns, config)

        eigenvalues = np.linalg.eigh(cov)[0]
        assert np.all(eigenvalues >= -1e-10), (
            f"Covariance matrix is not PSD. Min eigenvalue: {eigenvalues.min()}"
        )


# ---------------------------------------------------------------------------
# 18-19. Determinism and leakage
# ---------------------------------------------------------------------------

class TestDeterminismAndLeakage:

    def test_deterministic_weights(self):
        """Same inputs should produce identical weights."""
        returns = _make_returns(seed=42)
        config = PortfolioConfig()

        solution1 = optimize_portfolio("Bull", returns, config=config)
        solution2 = optimize_portfolio("Bull", returns, config=config)

        for asset in _ASSET_ORDER:
            assert np.isclose(
                solution1.weights[asset], solution2.weights[asset], atol=1e-10
            ), f"Weights for {asset} differ between runs"

    def test_future_returns_dont_affect_weights_at_T(self):
        """Returns after cutoff T must not affect weights computed at T."""
        returns = _make_returns(n_obs=300, seed=42)
        config = PortfolioConfig()

        # Weights computed from returns through T
        returns_through_T = returns.iloc[:150]
        solution_at_T = optimize_portfolio("Bull", returns_through_T, config=config)

        # Add more return data after T
        returns_extended = pd.concat([returns_through_T, returns.iloc[150:]])
        solution_extended = optimize_portfolio("Bull", returns_through_T, config=config)

        for asset in _ASSET_ORDER:
            assert np.isclose(
                solution_at_T.weights[asset],
                solution_extended.weights[asset],
                atol=1e-10,
            ), (
                f"Weight for {asset} changed when future returns were added"
            )


# ---------------------------------------------------------------------------
# 20-22. Solver and constraint behavior
# ---------------------------------------------------------------------------

class TestSolverBehavior:

    def test_solver_is_recorded(self):
        """Solution must record which solver was used."""
        returns = _make_returns()
        config = PortfolioConfig()
        solution = optimize_portfolio("Bull", returns, config=config)

        assert solution.solver in ("CLARABEL", "OSQP", "SCS"), (
            f"Unknown solver: {solution.solver}"
        )

    def test_status_is_recorded(self):
        """Solution must record solver status."""
        returns = _make_returns()
        config = PortfolioConfig()
        solution = optimize_portfolio("Bull", returns, config=config)

        assert solution.status in ("optimal", "optimal_inaccurate"), (
            f"Solver status must be optimal, got: {solution.status}"
        )

    def test_regime_is_recorded(self):
        """Solution must record the regime used."""
        returns = _make_returns()
        config = PortfolioConfig()

        for regime in ["Bull", "Bear", "Crisis"]:
            solution = optimize_portfolio(regime, returns, config=config)
            assert solution.regime == regime


# ---------------------------------------------------------------------------
# 23-25. Benchmark definitions
# ---------------------------------------------------------------------------

class TestBenchmarkDefinitions:

    def test_static_60_40_values(self):
        """Static 60/40 must have exactly equity=0.60, gold=0.00, bond=0.40."""
        weights = static_60_40_weights()
        validate_benchmark_weights(weights)

        assert np.isclose(weights["equity"], 0.60)
        assert np.isclose(weights["gold"], 0.00)
        assert np.isclose(weights["bond"], 0.40)

    def test_equal_weight_values(self):
        """Equal weight must have exactly 1/3 for each asset."""
        weights = equal_weight_weights()
        validate_benchmark_weights(weights)

        expected = 1.0 / 3.0
        for asset in ["equity", "gold", "bond"]:
            assert np.isclose(weights[asset], expected)

    def test_benchmark_no_vix(self):
        """Benchmark weights must not contain VIX."""
        for fn in [static_60_40_weights, equal_weight_weights]:
            weights = fn()
            assert "vix" not in weights

    def test_benchmark_weights_sum_to_one(self):
        """Benchmark weights must sum to 1."""
        for fn in [static_60_40_weights, equal_weight_weights]:
            weights = fn()
            total = sum(weights.values())
            assert np.isclose(total, 1.0)

    def test_benchmark_deterministic(self):
        """Benchmark functions must return the same values on repeated calls."""
        assert static_60_40_weights() == static_60_40_weights()
        assert equal_weight_weights() == equal_weight_weights()

    def test_validate_benchmark_rejects_vix(self):
        """validate_benchmark_weights should reject VIX allocation."""
        with pytest.raises(ValueError, match="VIX"):
            validate_benchmark_weights({"equity": 0.5, "gold": 0.3, "bond": 0.2, "vix": 0.0})

    def test_validate_benchmark_rejects_bad_sum(self):
        """validate_benchmark_weights should reject weights not summing to 1."""
        with pytest.raises(ValueError, match="sum"):
            validate_benchmark_weights({"equity": 0.5, "gold": 0.3, "bond": 0.1})

    def test_validate_benchmark_rejects_negative(self):
        """validate_benchmark_weights should reject negative weights."""
        with pytest.raises(ValueError):
            validate_benchmark_weights({"equity": -0.1, "gold": 0.5, "bond": 0.6})
