"""Frozen, causal portfolio-risk experiments; no model selection or downloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from regime_shift.research.evaluation import StrategyRun, metric_row
from regime_shift.research.spec import ANNUALIZATION_FACTOR
from regime_shift.research.strategies import ResearchContext, TargetFn, min_variance

COST_BPS = (5, 10, 20, 50)
HISTORICAL_WINDOWS = (
    ("euro_crisis_2011", "2011-07-01", "2011-12-31"),
    ("covid_2020", "2020-02-01", "2020-04-30"),
    ("inflation_2022", "2022-01-01", "2022-12-31"),
)


@dataclass(frozen=True)
class RiskLabConfig:
    """One prespecified setting per variant, rather than a parameter search."""

    max_weight: float = 0.80
    turnover_penalty: float = 0.01  # annual variance units per L1 weight change
    target_volatility: float = 0.10
    contribution_tolerance: float = 0.002  # absolute share of portfolio variance

    def validate(self, n: int) -> None:
        values = (self.max_weight, self.turnover_penalty, self.target_volatility,
                  self.contribution_tolerance)
        if not np.isfinite(values).all():
            raise ValueError("Risk lab parameters must be finite.")
        if n < 2 or not 1 / n <= self.max_weight <= 1:
            raise ValueError("Infeasible maximum weight.")
        if self.turnover_penalty < 0 or self.target_volatility <= 0:
            raise ValueError("Penalty must be nonnegative and target volatility positive.")
        if not 0 < self.contribution_tolerance <= 0.01:
            raise ValueError("Contribution tolerance must be in (0, 0.01].")


def _covariance(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.isfinite(cov).all():
        raise ValueError("Covariance must be a finite square matrix.")
    if not np.allclose(cov, cov.T, atol=1e-12, rtol=1e-10):
        raise ValueError("Covariance must be symmetric.")
    if np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("Covariance must be positive definite (apply a stated ridge).")
    return cov


def _weights(weights, n: int, cap: float = 1.0) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if (w.shape != (n,) or not np.isfinite(w).all() or w.min() < -1e-8
            or abs(w.sum() - 1) > 1e-7 or w.max() > cap + 1e-7):
        raise ValueError("Invalid long-only, fully invested weights or cap violation.")
    w = np.maximum(w, 0)
    w /= w.sum()
    if w.max() > cap + 1e-7:
        raise ValueError("Normalized weights violate cap.")
    return w


def risk_contributions(weights, covariance: np.ndarray) -> np.ndarray:
    """Signed Euler contributions to variance, divided by total variance."""
    cov = _covariance(covariance)
    w = _weights(weights, len(cov))
    return w * (cov @ w) / float(w @ cov @ w)


def equal_risk_contribution(covariance: np.ndarray, tolerance: float = 0.002) -> np.ndarray:
    """Solve convex risk budgeting in positive x, then normalize and validate ERC.

    min 0.5*x' C x - sum(log(x))/n. Scaling C does not change normalized w.
    A weight cap would generally prevent exact ERC; this variant has only the
    natural long-only [0, 1] bounds. No clipped or approximate parity fallback.
    """
    cov = _covariance(covariance)
    if not np.isfinite(tolerance) or not 0 < tolerance <= 0.01:
        raise ValueError("Invalid contribution tolerance.")
    n = len(cov)
    scaled = cov / np.trace(cov) * n
    x = cp.Variable(n, pos=True)
    problem = cp.Problem(cp.Minimize(
        0.5 * cp.quad_form(x, cp.psd_wrap(scaled)) - cp.sum(cp.log(x)) / n))
    errors = []
    for solver in (cp.CLARABEL, cp.SCS):
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status not in ("optimal", "optimal_inaccurate") or x.value is None:
                raise ValueError(f"status={problem.status}")
            if not np.isfinite(x.value).all() or np.min(x.value) <= 0:
                raise ValueError("Risk budget solution is not positive and finite.")
            w = _weights(x.value / x.value.sum(), n)
            if np.max(np.abs(risk_contributions(w, cov) - 1 / n)) > tolerance:
                raise ValueError("Risk contribution validation failed.")
            return w
        except (cp.error.SolverError, ValueError) as exc:
            errors.append(f"{solver}: {exc}")
    raise RuntimeError("ERC failed: " + "; ".join(errors))


def minimum_variance(covariance: np.ndarray, cap: float = 0.80,
                     previous: Optional[np.ndarray] = None, penalty: float = 0.0) -> np.ndarray:
    """Annual variance + penalty * L1 distance from signal-close drifted holdings."""
    cov = _covariance(covariance)
    n = len(cov)
    if not np.isfinite(cap) or not 1 / n <= cap <= 1:
        raise ValueError("Infeasible maximum weight.")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Invalid turnover penalty.")
    prev = None if previous is None else _weights(previous, n)
    w = cp.Variable(n)
    objective = ANNUALIZATION_FACTOR * cp.quad_form(w, cp.psd_wrap(cov))
    if prev is not None:
        objective += penalty * cp.norm1(w - prev)
    problem = cp.Problem(cp.Minimize(objective), [w >= 0, w <= cap, cp.sum(w) == 1])
    errors = []
    for solver in (cp.CLARABEL, cp.OSQP, cp.SCS):
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
                raise ValueError(f"status={problem.status}")
            return _weights(w.value, n, cap)
        except (cp.error.SolverError, ValueError) as exc:
            errors.append(f"{solver}: {exc}")
    raise RuntimeError("Minimum variance failed: " + "; ".join(errors))


def volatility_target_weights(covariance: np.ndarray, bond_index: int,
                             target: float = 0.10) -> np.ndarray:
    """Maximum unlevered blend alpha*equal_weight + (1-alpha)*bond below target.

    Uses full blend variance, including bond covariance. Fails if no feasible
    blend exists; an unusually volatile bond proxy is not treated as cash.
    """
    cov = _covariance(covariance) * ANNUALIZATION_FACTOR
    n = len(cov)
    if n < 2 or not 0 <= bond_index < n or not np.isfinite(target) or target <= 0:
        raise ValueError("Invalid bond index or volatility target.")
    bond = np.eye(n)[bond_index]
    direction = np.full(n, 1 / n) - bond
    a = float(direction @ cov @ direction)
    b = float(2 * bond @ cov @ direction)
    c = float(bond @ cov @ bond - target ** 2)
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        raise ValueError("Volatility target infeasible without leverage or shorting.")
    lower, upper = sorted(((-b - np.sqrt(discriminant)) / (2 * a),
                           (-b + np.sqrt(discriminant)) / (2 * a)))
    alpha = min(1.0, upper)
    if alpha < max(0.0, lower) - 1e-12:
        raise ValueError("Volatility target infeasible without leverage or shorting.")
    w = _weights(bond + np.clip(alpha, 0, 1) * direction, n)
    if float(w @ cov @ w) > target ** 2 + 1e-10:
        raise ValueError("Volatility target validation failed.")
    return w


def estimate_covariance(window: pd.DataFrame, ridge: float, shrink: bool = False) -> np.ndarray:
    values = window.to_numpy(dtype=float)
    if (len(window) < 2 or not np.isfinite(values).all()
            or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("Need finite observations and a positive covariance ridge.")
    cov = LedoitWolf().fit(values).covariance_ if shrink else window.cov().to_numpy()
    return _covariance((cov + cov.T) / 2 + ridge * np.eye(values.shape[1]))


def risk_lab_targets(ctx: ResearchContext, settings: Optional[RiskLabConfig] = None) -> dict[str, TargetFn]:
    settings = settings or RiskLabConfig()
    settings.validate(len(ctx.assets))
    pc = ctx.config.portfolio_config

    def target(kind):
        def fn(date, features, returns, previous):
            cov = estimate_covariance(returns[ctx.assets].tail(pc.estimation_lookback),
                                      pc.covariance_ridge, shrink=kind == "lw_min_variance")
            if kind == "equal_risk_contribution":
                return equal_risk_contribution(cov, settings.contribution_tolerance)
            if kind == "vol_target_equal_weight":
                return volatility_target_weights(cov, ctx.assets.index(ctx.config.bond_col),
                                                 settings.target_volatility)
            prev = None if previous is None else np.array([previous[a] for a in ctx.assets])
            penalty = settings.turnover_penalty if kind == "turnover_min_variance" else 0.0
            return minimum_variance(cov, settings.max_weight, prev, penalty)
        return fn

    targets = {name: target(name) for name in (
        "equal_risk_contribution", "lw_min_variance", "turnover_min_variance",
        "vol_target_equal_weight")}
    targets["min_variance"] = min_variance(ctx, max_weight=settings.max_weight)
    targets["static_6040"] = lambda date, features, returns, previous: [
        {ctx.config.equity_col: 0.6, ctx.config.gold_col: 0.0, ctx.config.bond_col: 0.4}[a]
        for a in ctx.assets]
    targets["equal_weight"] = lambda date, features, returns, previous: np.full(len(ctx.assets), 1 / len(ctx.assets))
    return targets


def trade_deltas(run: StrategyRun, asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Actual signed close-fill trades reconstructed from post-trade holdings.

    The first row starts from cash. Later pre-fill holdings are previous
    post-trade holdings drifted through today's realized return, even on nonfills.
    """
    if run.weights is None or not run.index.isin(asset_returns.index).all():
        raise ValueError("Weights and complete asset returns required for trade accounting.")
    returns = asset_returns.loc[run.index, run.weights.columns]
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Asset returns must be finite.")
    pre = run.weights.shift(1, fill_value=0.0)
    gross = (pre * returns).sum(axis=1)
    if not np.allclose(gross, run.gross, atol=1e-12, rtol=0):
        raise ValueError("Holdings do not reproduce gross returns (initial cash required).")
    drifted = pre.mul(1 + returns).div(1 + gross, axis=0)
    deltas = run.weights - drifted
    traded = deltas.abs().sum(axis=1)
    expected = 2 * run.turnover.copy()
    expected.iloc[0] = run.turnover.iloc[0]
    if not np.allclose(traded, expected, atol=1e-10, rtol=0):
        raise ValueError("Trade deltas do not reconcile to engine turnover.")
    return deltas


def historical_replay(run: StrategyRun, cash: pd.Series) -> pd.DataFrame:
    """Calendar attribution of realized returns; never restart a crisis portfolio."""
    rows = []
    for label, start, end in HISTORICAL_WINDOWS:
        available = run.index[(run.index >= start) & (run.index <= end)]
        if len(available) < 2:
            rows.append({"strategy": run.strategy_id, "scenario": label,
                         "evaluation_type": "historical_replay", "status": "unavailable"})
        else:
            sliced = run.window(start, end)
            rows.append({**metric_row(sliced, cash), "scenario": label,
                         "evaluation_type": "historical_replay", "status": "available",
                         "window_return": float((1 + sliced.net).prod() - 1)})
    return pd.DataFrame(rows)


def hypothetical_stress(weights, covariance: np.ndarray, asset_shocks,
                        correlation: float = 0.90, concentration: float = 0.25) -> pd.DataFrame:
    """Terminal fixed-holdings diagnostics, not historical returns or forecasts.

    Correlation stress blends the original correlation matrix halfway toward
    equicorrelation rho. Concentration transfers fraction q of the whole book
    into its largest holding. Both transformations preserve feasibility/PSD.
    """
    cov = _covariance(covariance)
    w = _weights(weights, len(cov))
    shocks = np.asarray(asset_shocks, dtype=float)
    if (shocks.shape != w.shape or not np.isfinite(shocks).all() or (shocks < -1).any()
            or not np.isfinite([correlation, concentration]).all()
            or not 0 <= correlation <= 1 or not 0 <= concentration <= 1):
        raise ValueError("Invalid hypothetical shock assumptions.")
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    shocked_corr = 0.5 * corr + 0.5 * ((1 - correlation) * np.eye(len(w)) + correlation)
    shocked_cov = np.outer(sd, sd) * shocked_corr
    concentrated = (1 - concentration) * w
    concentrated[np.argmax(w)] += concentration
    rows = []
    for label, holdings, matrix in (("reference", w, cov), ("correlation_shock", w, shocked_cov),
                                     ("concentration_shock", concentrated, cov),
                                     ("combined_shock", concentrated, shocked_cov)):
        rows.append({"evaluation_type": "hypothetical_terminal_snapshot", "scenario": label,
                     "ann_vol": float(np.sqrt(holdings @ matrix @ holdings * ANNUALIZATION_FACTOR)),
                     "hhi": float(holdings @ holdings), "max_weight": float(holdings.max()),
                     "assumed_one_period_return": float(holdings @ shocks)})
    return pd.DataFrame(rows)
