"""Strategy variants run through the one official lagged-fill engine.

Every variant is gated by the same eligibility rule as RegimeShift (enough
history for the HMM and the covariance window), so all variants send their
first signal on the same close, fill on the same next close and pay the same
initial-allocation cost.  Their accounting indexes are therefore identical by
construction; :func:`require_common_index` still verifies it.

``TargetFn(date, features_through_t, returns_through_t, previous) -> weights | None``
sees data through close ``t`` only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from regime_shift.backtest import (
    BacktestResult,
    Signal,
    _cost_model,
    prepare_walk_forward_inputs,
    run_benchmark,
    run_walk_forward_backtest,
    simulate_lagged_execution,
)
from regime_shift.benchmarks import equal_weight_weights, static_60_40_weights
from regime_shift.config import RegimeShiftConfig
from regime_shift.features import fit_feature_scaler, transform_features
from regime_shift.portfolio import optimize_portfolio
from regime_shift.regime_model import fit_hmm, predict_current_state
from regime_shift.research.evaluation import StrategyRun, require_common_index
from regime_shift.research.spec import ANNUALIZATION_FACTOR

TargetFn = Callable[[pd.Timestamp, pd.DataFrame, pd.DataFrame, Optional[Dict[str, float]]], Optional[Sequence[float]]]

# Frozen protocol-v2 regime weights (ablation D and the volatility rule share them).
REGIME_FIXED_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Bull": {"equity": 0.65, "gold": 0.20, "bond": 0.15},
    "Bear": {"equity": 0.35, "gold": 0.30, "bond": 0.35},
    "Crisis": {"equity": 0.10, "gold": 0.30, "bond": 0.60},
}


@dataclass
class ResearchContext:
    prices: pd.DataFrame
    config: RegimeShiftConfig
    raw_features: pd.DataFrame
    asset_returns: pd.DataFrame
    rebalance_positions: set

    @property
    def assets(self) -> List[str]:
        return self.config.core_assets

    @property
    def min_history(self) -> int:
        """Rows through t required before RegimeShift may signal (its own gating rule)."""
        return max(self.config.minimum_training_observations,
                   self.config.portfolio_config.minimum_estimation_observations)

    @property
    def cash_returns(self) -> pd.Series:
        """LIQUIDBEES 'bond' leg, used as the cash proxy for excess-return Sharpe."""
        return self.asset_returns[self.config.bond_col]


def build_context(prices: pd.DataFrame, config: RegimeShiftConfig) -> ResearchContext:
    config = copy.deepcopy(config)
    config.validate()
    raw_features, asset_returns, rebalance_positions = prepare_walk_forward_inputs(prices, config)
    return ResearchContext(prices, config, raw_features, asset_returns, rebalance_positions)


def run_target_strategy(ctx: ResearchContext, strategy_id: str, target_fn: TargetFn) -> StrategyRun:
    """Run ``target_fn`` through ``simulate_lagged_execution`` with RegimeShift's gating."""
    assets = ctx.assets

    def signal_fn(pos, date, weights, allocated):
        if pos + 1 < ctx.min_history:
            return None
        previous = dict(zip(assets, weights)) if allocated else None
        target = target_fn(date, ctx.raw_features.iloc[:pos + 1], ctx.asset_returns.iloc[:pos + 1], previous)
        if target is None:
            return None
        target = np.asarray(target, dtype=float)
        if target.shape != (len(assets),) or np.any(target < -1e-9) or not np.isclose(target.sum(), 1.0, atol=1e-6):
            raise ValueError(f"{strategy_id}: invalid target {target} on {date.date()}.")
        return Signal(np.clip(target, 0.0, None) / np.clip(target, 0.0, None).sum())

    trace = simulate_lagged_execution(ctx.asset_returns, ctx.rebalance_positions, signal_fn,
                                      _cost_model(ctx.config, None))
    return StrategyRun.from_result(strategy_id, trace)


def run_regimeshift(ctx: ResearchContext) -> tuple[StrategyRun, BacktestResult]:
    result = run_walk_forward_backtest(ctx.prices, ctx.config, risk_free_rate=0.0)
    return StrategyRun.from_result("regimeshift", result), result


def run_fixed_benchmark(ctx: ResearchContext, strategy_id: str, weights: Dict[str, float],
                        reference: BacktestResult) -> StrategyRun:
    """Fixed weights traded on the reference strategy's fill dates, from its first fill."""
    result = run_benchmark(ctx.prices, weights, ctx.config, rebalance_flags=reference.rebalance_flags,
                           start_date=reference.net_returns.index[0])
    return StrategyRun.from_result(strategy_id, result)


# ---------------------------------------------------------------------------
# Target functions (frozen protocol-v2 definitions)
# ---------------------------------------------------------------------------

def no_regime_optimizer(ctx: ResearchContext) -> TargetFn:
    pc = ctx.config.portfolio_config

    def fn(date, features, returns, previous):
        portfolio = optimize_portfolio("Bull", returns.tail(pc.estimation_lookback), previous, config=pc)
        return [portfolio.weights[a] for a in ctx.assets]
    return fn


def hmm_regime(ctx: ResearchContext, features: pd.DataFrame) -> str:
    train = features.tail(ctx.config.train_window)
    scaler = fit_feature_scaler(train)
    hmm, _, _ = fit_hmm(transform_features(scaler, train), train, config=ctx.config.hmm_config)
    return predict_current_state(hmm, scaler, train, train, config=ctx.config.hmm_config).regime


def hmm_fixed(ctx: ResearchContext) -> TargetFn:
    def fn(date, features, returns, previous):
        regime = hmm_regime(ctx, features)
        return [REGIME_FIXED_WEIGHTS[regime][a] for a in ctx.assets]
    return fn


def vol_rule(ctx: ResearchContext, window: int = 63, high_vol: float = 0.20, crisis_vol: float = 0.30) -> TargetFn:
    def fn(date, features, returns, previous):
        vol = float(returns[ctx.config.equity_col].tail(window).std(ddof=1)) * np.sqrt(ANNUALIZATION_FACTOR)
        regime = "Crisis" if vol > crisis_vol else "Bear" if vol > high_vol else "Bull"
        return [REGIME_FIXED_WEIGHTS[regime][a] for a in ctx.assets]
    return fn


def psd_covariance(window: pd.DataFrame, ridge: float) -> np.ndarray:
    cov = window.cov().to_numpy()
    cov = (cov + cov.T) / 2.0
    vals, vecs = np.linalg.eigh(cov)
    cov = vecs @ np.diag(np.clip(vals, 0.0, None) + ridge) @ vecs.T
    return (cov + cov.T) / 2.0


def solve_long_only(objective_builder, n: int, max_weight: float) -> Optional[np.ndarray]:
    w = cp.Variable(n)
    problem = cp.Problem(cp.Minimize(objective_builder(w)), [cp.sum(w) == 1.0, w >= 0.0, w <= max_weight])
    for solver in (cp.CLARABEL, cp.OSQP, cp.SCS):
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception:  # solver unavailable or numerical failure: try the next one
            continue
        if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            target = np.clip(w.value, 0.0, None)
            return target / target.sum()
    return None


def min_variance(ctx: ResearchContext, max_weight: float = 0.80) -> TargetFn:
    pc = ctx.config.portfolio_config

    def fn(date, features, returns, previous):
        cov = psd_covariance(returns.tail(pc.estimation_lookback), pc.covariance_ridge)
        target = solve_long_only(lambda w: cp.quad_form(w, cp.psd_wrap(cov)), len(ctx.assets), max_weight)
        if target is None:
            raise RuntimeError(f"min_variance: all solvers failed on {date.date()}.")
        return target
    return fn


ABLATION_TARGETS: Dict[str, Callable[[ResearchContext], TargetFn]] = {
    "no_regime_optimizer": no_regime_optimizer,
    "vol_rule": vol_rule,
    "hmm_fixed": hmm_fixed,
    "min_variance": min_variance,
}


def run_core_suite(prices: pd.DataFrame, config: RegimeShiftConfig,
                   variants: Optional[Sequence[str]] = None) -> tuple[Dict[str, StrategyRun], BacktestResult, ResearchContext]:
    """RegimeShift, the frozen ablations and both static benchmarks on one verified index."""
    ctx = build_context(prices, config)
    rs, rs_result = run_regimeshift(ctx)
    runs: Dict[str, StrategyRun] = {"regimeshift": rs}
    for strategy_id in (variants if variants is not None else ABLATION_TARGETS):
        runs[strategy_id] = run_target_strategy(ctx, strategy_id, ABLATION_TARGETS[strategy_id](ctx))
    runs["static_6040"] = run_fixed_benchmark(ctx, "static_6040", static_60_40_weights(), rs_result)
    runs["equal_weight"] = run_fixed_benchmark(ctx, "equal_weight", equal_weight_weights(), rs_result)
    require_common_index(runs.values())
    return runs, rs_result, ctx
