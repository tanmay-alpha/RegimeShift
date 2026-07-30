"""
Regime-conditioned portfolio optimization using CVXPY.

Implements three optimization objectives (one per regime) with regime-specific
constraints on equity, gold, and bond weights.  VIX is never allocated.

Tradable assets are exactly equity, gold, and bond.

Expected returns are estimated from historical returns using a configurable
lookback window.  The covariance matrix is symmetrized and regularized with
a diagonal ridge to ensure positive semidefiniteness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from regime_shift.config import PortfolioConfig
from regime_shift.exceptions import PortfolioOptimizationError

logger = logging.getLogger(__name__)

# Tradable asset names in deterministic order
_ASSET_ORDER = ["equity", "gold", "bond"]


@dataclass
class PortfolioSolution:
    """
    Result of regime-conditioned portfolio optimization.

    Attributes:
        weights: Dict mapping asset name to weight.  Sums to 1.
        status: CVXPY solver status string.
        objective_value: Value of the optimization objective.
        solver: Name of the solver used.
        regime: Regime label that was optimized.
        expected_return: Estimated expected portfolio return (annualized).
        expected_volatility: Estimated portfolio volatility (annualized).
        covariance_diagnostics: Dict with covariance matrix diagnostics.
        warning: Optional warning message.
        fallback_used: Whether an explicit fallback was used instead of optimal.
    """

    weights: Dict[str, float]
    status: str
    objective_value: float
    solver: str
    regime: str
    expected_return: float
    expected_volatility: float
    covariance_diagnostics: Dict[str, any]
    warning: Optional[str] = None
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Return and covariance estimation
# ---------------------------------------------------------------------------

def _estimate_returns_and_covariance(
    returns: pd.DataFrame,
    config: PortfolioConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Estimate expected returns and covariance matrix from historical returns.

    Uses simple historical mean for expected returns and sample covariance
    with symmetrization and diagonal ridge for PSD guarantee.

    Args:
        returns: DataFrame of asset returns (rows = observations).
        config: PortfolioConfig with estimation parameters.

    Returns:
        (expected_returns, covariance_matrix, diagnostics)
    """
    if len(returns) < config.minimum_estimation_observations:
        raise PortfolioOptimizationError(
            f"Insufficient return history: {len(returns)} observations, "
            f"need at least {config.minimum_estimation_observations}."
        )

    # Simple historical mean as expected return estimate
    # Use only required asset columns (ignore extra columns like vix)
    returns_subset = returns[_ASSET_ORDER]
    expected_returns = returns_subset.mean().values  # shape: (n_assets,)

    # Sample covariance (use only required assets)
    cov = returns_subset.cov().values  # shape: (n_assets, n_assets)

    # Symmetrize (numerical noise can make it slightly asymmetric)
    cov = (cov + cov.T) / 2.0

    # Eigenvalue decomposition for PSD guarantee
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Clip very small negative eigenvalues to zero
    eigenvalues = np.where(eigenvalues < 0, 0.0, eigenvalues)

    # Add diagonal ridge
    eigenvalues += config.covariance_ridge

    # Reconstruct covariance matrix
    cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Final symmetrization
    cov = (cov + cov.T) / 2.0

    # Verify PSD
    final_eigenvalues = np.linalg.eigh(cov)[0]
    min_eigenvalue = float(final_eigenvalues.min())
    is_psd = min_eigenvalue >= -1e-10

    diagnostics = {
        "n_observations": len(returns),
        "lookback": config.estimation_lookback,
        "min_eigenvalue": min_eigenvalue,
        "is_positive_semidefinite": is_psd,
        "covariance_ridge_applied": config.covariance_ridge,
        "eigenvalue_range": (float(final_eigenvalues.min()), float(final_eigenvalues.max())),
    }

    if not is_psd:
        logger.warning(
            "Covariance matrix is not positive semidefinite (min eigenvalue: %.2e). "
            "Ridge regularization may be insufficient.",
            min_eigenvalue,
        )

    return expected_returns, cov, diagnostics


# ---------------------------------------------------------------------------
# Regime-specific optimization
# ---------------------------------------------------------------------------

def optimize_portfolio(
    regime: str,
    returns_through_date: pd.DataFrame,
    previous_weights: Optional[Dict[str, float]] = None,
    config: Optional[PortfolioConfig] = None,
) -> PortfolioSolution:
    """
    Optimize portfolio weights for the given regime.

    Uses CVXPY with regime-specific objectives and constraints.

    Args:
        regime: Market regime label (Bull, Bear, or Crisis).
        returns_through_date: Historical asset returns through the decision date.
            Columns must be exactly ['equity', 'gold', 'bond'].
        previous_weights: Optional dict of previous portfolio weights for
            turnover penalty.  Must contain all three assets.
        config: PortfolioConfig instance.  Defaults to PortfolioConfig() if None.

    Returns:
        PortfolioSolution with optimal weights and diagnostics.

    Raises:
        PortfolioOptimizationError: If optimization fails or inputs are invalid.
    """
    config = config or PortfolioConfig()
    config.validate()

    _validate_returns_input(returns_through_date)
    _validate_previous_weights(previous_weights)

    # Limit to lookback window
    if len(returns_through_date) > config.estimation_lookback:
        returns = returns_through_date.iloc[-config.estimation_lookback:]
    else:
        returns = returns_through_date

    n_assets = len(_ASSET_ORDER)

    # Estimate returns and covariance
    expected_returns, cov_matrix, cov_diagnostics = _estimate_returns_and_covariance(
        returns, config
    )

    # Build previous weights vector if provided
    prev_w_vec = None
    if previous_weights is not None:
        prev_w_vec = np.array([previous_weights[a] for a in _ASSET_ORDER])

    # Set up optimization problem
    w = cp.Variable(n_assets)

    # Expected portfolio return
    portfolio_return = expected_returns @ w
    # Portfolio variance: w^T * Sigma * w
    portfolio_variance = cp.quad_form(w, cp.psd_wrap(cov_matrix))

    # Build constraints based on regime
    constraints = _build_constraints(w, regime, config)

    # Build objective based on regime
    objective = _build_objective(
        w, portfolio_return, portfolio_variance, regime, config, prev_w_vec
    )

    problem = cp.Problem(objective, constraints)

    # Solve with preferred solvers
    solver_used = None
    status = None
    warning = None

    for solver_name in config.preferred_solvers:
        try:
            solver_obj = getattr(cp, solver_name)
            problem.solve(solver=solver_obj, verbose=False)
            solver_used = solver_name
            status = problem.status
            break
        except Exception as exc:
            logger.debug("Solver %s failed: %s", solver_name, exc)
            continue

    if solver_used is None:
        raise PortfolioOptimizationError(
            "All preferred solvers failed. Check CVXPY installation.",
            regime=regime,
            solver=", ".join(config.preferred_solvers),
            solver_status="none",
        )

    # Check solution quality
    if status not in ("optimal", "optimal_inaccurate"):
        raise PortfolioOptimizationError(
            f"Solver returned non-optimal status: {status}",
            regime=regime,
            solver=solver_used,
            solver_status=status,
        )

    if status == "optimal_inaccurate":
        warning = (
            f"Solver {solver_used} returned 'optimal_inaccurate'. "
            "Results may not be reliable."
        )
        logger.warning(warning)

    # Extract weights
    if w.value is None:
        raise PortfolioOptimizationError(
            "Optimization returned None for weights.",
            regime=regime,
            solver=solver_used,
            solver_status=status,
        )

    weight_values = np.maximum(w.value, 0.0)  # Ensure non-negative

    # Renormalize to sum to 1 (handles numerical drift)
    weight_sum = float(weight_values.sum())
    if weight_sum > 0 and not np.isclose(weight_sum, 1.0):
        weight_values = weight_values / weight_sum

    # Ensure finite values
    if not np.all(np.isfinite(weight_values)):
        raise PortfolioOptimizationError(
            "Optimization produced non-finite weights.",
            regime=regime,
            solver=solver_used,
            solver_status=status,
        )

    weights_dict = {asset: float(weight_values[i]) for i, asset in enumerate(_ASSET_ORDER)}

    # Compute diagnostics
    est_ret = float(expected_returns @ weight_values)
    est_vol = float(np.sqrt(weight_values @ cov_matrix @ weight_values))

    solution = PortfolioSolution(
        weights=weights_dict,
        status=status,
        objective_value=float(problem.value),
        solver=solver_used,
        regime=regime,
        expected_return=est_ret,
        expected_volatility=est_vol,
        covariance_diagnostics=cov_diagnostics,
        warning=warning,
        fallback_used=False,
    )

    logger.info(
        "Portfolio optimized for %s: %s (solver=%s, status=%s)",
        regime,
        {k: f"{v:.3f}" for k, v in weights_dict.items()},
        solver_used,
        status,
    )

    return solution


# ---------------------------------------------------------------------------
# Regime-specific objectives and constraints
# ---------------------------------------------------------------------------

def _build_constraints(
    w: cp.Variable,
    regime: str,
    config: PortfolioConfig,
) -> List:
    """Build CVXPY constraints for the given regime."""
    constraints = []

    # Global constraints (all regimes)
    # sum(weights) == 1
    constraints.append(cp.sum(w) == 1.0)
    # weights >= 0 (long-only)
    constraints.append(w >= 0.0)
    # weights <= max_asset_weight
    constraints.append(w <= config.maximum_asset_weight)

    if regime == "Bull":
        bc = config.bull
        constraints.append(w[_ASSET_ORDER.index("equity")] >= bc.equity_min)
        constraints.append(w[_ASSET_ORDER.index("equity")] <= bc.equity_max)
        constraints.append(w[_ASSET_ORDER.index("gold")] <= bc.gold_max)
        constraints.append(w[_ASSET_ORDER.index("bond")] <= bc.bond_max)

    elif regime == "Bear":
        bc = config.bear
        constraints.append(w[_ASSET_ORDER.index("equity")] <= bc.equity_max)
        # gold + bond >= defensive_min
        gold_idx = _ASSET_ORDER.index("gold")
        bond_idx = _ASSET_ORDER.index("bond")
        constraints.append(w[gold_idx] + w[bond_idx] >= bc.defensive_min)

    elif regime == "Crisis":
        cc = config.crisis
        constraints.append(w[_ASSET_ORDER.index("equity")] <= cc.equity_max)
        constraints.append(w[_ASSET_ORDER.index("gold")] >= cc.gold_min)
        constraints.append(w[_ASSET_ORDER.index("bond")] >= cc.bond_min)

    else:
        raise PortfolioOptimizationError(
            f"Unknown regime: '{regime}'. Must be Bull, Bear, or Crisis."
        )

    return constraints


def _build_objective(
    w: cp.Variable,
    portfolio_return: cp.Expression,
    portfolio_variance: cp.Expression,
    regime: str,
    config: PortfolioConfig,
    prev_w_vec: Optional[np.ndarray],
) -> cp.Problem:
    """Build CVXPY objective for the given regime."""
    turnover = 0.0
    if prev_w_vec is not None and config.turnover_penalty > 0:
        turnover = config.turnover_penalty * cp.norm1(w - prev_w_vec)

    if regime == "Bull":
        objective = cp.Maximize(
            portfolio_return - config.bull_risk_aversion * portfolio_variance - turnover
        )
    elif regime == "Bear":
        objective = cp.Minimize(
            portfolio_variance - config.bear_return_reward * portfolio_return + turnover
        )
    elif regime == "Crisis":
        objective = cp.Minimize(
            portfolio_variance + turnover
        )
    else:
        raise PortfolioOptimizationError(
            f"Unknown regime: '{regime}'."
        )

    return objective


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_returns_input(returns: pd.DataFrame) -> None:
    """Validate that returns data is suitable for portfolio optimization."""
    if not isinstance(returns, pd.DataFrame):
        raise PortfolioOptimizationError(
            f"returns must be a pd.DataFrame, got {type(returns).__name__}."
        )

    if returns.empty:
        raise PortfolioOptimizationError("Returns DataFrame is empty.")

    required = set(_ASSET_ORDER)
    present = set(returns.columns)
    missing = required - present
    if missing:
        raise PortfolioOptimizationError(
            f"Returns DataFrame is missing required columns: {sorted(missing)}. "
            f"Expected exactly {_ASSET_ORDER}."
        )

    # Check for extra columns (log warning but don't fail)
    extra = present - required
    if extra:
        logger.warning(
            "Returns DataFrame has extra columns that will be ignored: %s",
            sorted(extra),
        )

    # Check for NaN/inf only in required columns
    required_data = returns[_ASSET_ORDER]
    if required_data.isna().any().any():
        nan_cols = required_data.columns[required_data.isna().any()].tolist()
        raise PortfolioOptimizationError(
            f"Returns contain NaN in columns: {nan_cols}."
        )

    if not np.all(np.isfinite(required_data.values)):
        raise PortfolioOptimizationError(
            "Returns contain non-finite values (inf or -inf)."
        )


def _validate_previous_weights(
    previous_weights: Optional[Dict[str, float]],
) -> None:
    """Validate previous weights if provided."""
    if previous_weights is None:
        return

    required = set(_ASSET_ORDER)
    present = set(previous_weights.keys())
    missing = required - present
    if missing:
        raise PortfolioOptimizationError(
            f"previous_weights is missing required assets: {sorted(missing)}. "
            f"Expected all of {_ASSET_ORDER}."
        )

    for asset, weight in previous_weights.items():
        if not np.isfinite(weight):
            raise PortfolioOptimizationError(
                f"previous_weights['{asset}'] is not finite: {weight}."
            )
