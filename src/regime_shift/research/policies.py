"""Target-weight policies for controlled studies.

Policies may use information available at a rebalance, but never calculate
portfolio returns.  ``run_walk_forward_backtest`` owns that accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import pandas as pd


ASSETS = ("equity", "gold", "bond")
FIXED_REGIME_WEIGHTS = {
    "Bull": {"equity": 0.65, "gold": 0.20, "bond": 0.15},
    "Bear": {"equity": 0.35, "gold": 0.30, "bond": 0.35},
    "Crisis": {"equity": 0.10, "gold": 0.30, "bond": 0.60},
}


@dataclass
class PolicyDecision:
    weights: Dict[str, float]
    regime: str
    probabilities: Optional[np.ndarray] = None
    transition_matrix: Optional[pd.DataFrame] = None


class NoRegimeOptimizer:
    """One fixed mean-variance objective with long-only 80% caps."""

    name = "no_regime_optimizer"
    requires_hmm = False

    def select(self, *, returns: pd.DataFrame, **_: object) -> PolicyDecision:
        values = returns.loc[:, ASSETS].tail(252)
        mean = values.mean().to_numpy()
        cov = values.cov().to_numpy() + np.eye(len(ASSETS)) * 1e-6
        weights = cp.Variable(len(ASSETS))
        objective = cp.Maximize(mean @ weights - cp.quad_form(weights, cp.psd_wrap(cov)))
        problem = cp.Problem(objective, [cp.sum(weights) == 1, weights >= 0, weights <= 0.8])
        problem.solve(solver=cp.CLARABEL, verbose=False)
        result = np.maximum(np.asarray(weights.value).reshape(-1), 0.0)
        result /= result.sum()
        return PolicyDecision(dict(zip(ASSETS, result)), "No-regime")


class VolatilityRule:
    """Transparent 63-day equity-volatility allocation rule.

    The threshold is the 70th percentile of historical realised volatility
    available through the decision date, never a full-sample percentile.
    """

    name = "volatility_rule"
    requires_hmm = False
    low_vol_weights = FIXED_REGIME_WEIGHTS["Bull"]
    high_vol_weights = FIXED_REGIME_WEIGHTS["Bear"]

    def select(self, *, returns: pd.DataFrame, **_: object) -> PolicyDecision:
        equity = returns["equity"]
        rolling = equity.rolling(63, min_periods=63).std()
        available = rolling.dropna()
        current = float(available.iloc[-1])
        threshold = float(available.expanding(min_periods=1).quantile(0.70).iloc[-1])
        high = current >= threshold
        return PolicyDecision(dict(self.high_vol_weights if high else self.low_vol_weights), "HighVol" if high else "LowVol")


class HMMFixedAllocation:
    """The same HMM state selection, with predeclared fixed allocations."""

    name = "hmm_fixed_allocations"
    requires_hmm = True

    def select(self, *, hmm_solution, **_: object) -> PolicyDecision:
        if hmm_solution is None:
            raise ValueError("HMMFixedAllocation requires a fitted HMM solution.")
        return PolicyDecision(
            dict(FIXED_REGIME_WEIGHTS[hmm_solution.regime]),
            hmm_solution.regime,
            hmm_solution.probabilities.reindex(["Bull", "Bear", "Crisis"]).to_numpy(),
            hmm_solution.transition_matrix,
        )


class MinimumVariance:
    """Covariance-only minimum-variance allocation; no return estimate."""

    name = "minimum_variance"
    requires_hmm = False

    def select(self, *, returns: pd.DataFrame, **_: object) -> PolicyDecision:
        cov = returns.loc[:, ASSETS].tail(252).cov().to_numpy() + np.eye(len(ASSETS)) * 1e-6
        weights = cp.Variable(len(ASSETS))
        problem = cp.Problem(cp.Minimize(cp.quad_form(weights, cp.psd_wrap(cov))), [cp.sum(weights) == 1, weights >= 0, weights <= 0.8])
        problem.solve(solver=cp.CLARABEL, verbose=False)
        result = np.maximum(np.asarray(weights.value).reshape(-1), 0.0)
        result /= result.sum()
        return PolicyDecision(dict(zip(ASSETS, result)), "MinVariance")
