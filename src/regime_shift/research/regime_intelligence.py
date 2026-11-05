"""Bounded, signal-close HMM diagnostics and fixed-weight research controls.

Posteriors describe latent states at the current close, never future crashes.
All portfolio variants use the existing gated NEXT_CLOSE execution engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from regime_shift.benchmarks import equal_weight_weights, static_60_40_weights
from regime_shift.features import fit_feature_scaler, transform_features
from regime_shift.regime_model import fit_hmm, predict_current_state
from regime_shift.research.evaluation import StrategyRun, require_common_index
from regime_shift.research.spec import ANNUALIZATION_FACTOR
from regime_shift.research.strategies import (
    REGIME_FIXED_WEIGHTS,
    ResearchContext,
    run_target_strategy,
)

EXPERIMENT_ID = "regime_intelligence_v1"
REGIMES = ("Bull", "Bear", "Crisis")
NEGATIVE_CONTROL_SEED = 1729
ENTROPY_ALERT_THRESHOLD = 0.80
MAX_POSTERIOR_ALERT_THRESHOLD = 0.60
BASELINE_WINDOW = 63
BASELINE_HIGH_VOL = 0.20
BASELINE_CRISIS_VOL = 0.30
MAX_PRICE_ROWS = 6000
MAX_SIGNAL_CLOSES = 256
STRATEGY_IDS = ("hmm_fixed", "momentum_volatility", "negative_control", "static_6040", "equal_weight")


def _check_index(index: pd.Index) -> None:
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans
            or index.has_duplicates or not index.is_monotonic_increasing):
        raise ValueError("Expected a sorted, unique DatetimeIndex without NaT.")


def normalized_entropy(probabilities) -> float:
    """Shannon entropy / log(K); exact zeros contribute zero, without clipping."""
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)
            or not np.isclose(p.sum(), 1.0, atol=1e-8, rtol=0.0)):
        raise ValueError("Expected a finite probability vector summing to one (K >= 2).")
    positive = p[p > 0]
    return float(np.clip(-(positive * np.log(positive)).sum() / np.log(len(p)), 0.0, 1.0))


def uncertainty_diagnostics(probabilities) -> Dict:
    """Exploratory alerts only: thresholds are fixed before observing results."""
    entropy = normalized_entropy(probabilities)
    maximum = float(np.max(np.asarray(probabilities, dtype=float)))
    high_entropy = entropy >= ENTROPY_ALERT_THRESHOLD
    low_maximum = maximum <= MAX_POSTERIOR_ALERT_THRESHOLD
    return {
        "normalized_entropy": entropy, "max_posterior": maximum,
        "high_entropy_alert": high_entropy, "low_max_posterior_alert": low_maximum,
        "uncertainty_alert": high_entropy or low_maximum,
    }


def transition_persistence(transition: pd.DataFrame) -> pd.DataFrame:
    """Geometric expected duration 1/(1-p_ii), in training feature observations."""
    values = transition.to_numpy(dtype=float)
    if (transition.empty or not transition.index.is_unique or not transition.columns.is_unique
            or not transition.index.equals(transition.columns)
            or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
            or not np.allclose(values.sum(axis=1), 1.0, atol=1e-8, rtol=0.0)):
        raise ValueError("Expected a labeled, finite, row-stochastic transition matrix.")
    diagonal = np.diag(values)
    durations = np.full(len(diagonal), np.inf)
    np.divide(1.0, 1.0 - diagonal, out=durations, where=diagonal < 1.0)
    return pd.DataFrame({
        "regime": transition.index, "self_transition": diagonal,
        "expected_duration_feature_rows": durations, "absorbing": diagonal == 1.0,
    })


def overlap_state_stability(previous: pd.Series, current: pd.Series) -> Dict:
    """Compare decoded partitions on shared train dates, invariant to state IDs.

    Neither fit is rerun on the overlap alone. Each partition retains the
    sequence context of its own training window, as known at that refit close.
    """
    for states in (previous, current):
        _check_index(states.index)
        if states.isna().any():
            raise ValueError("Training state assignments contain missing values.")
    overlap = previous.index.intersection(current.index)
    old, new = previous.loc[overlap], current.loc[overlap]
    old_count, new_count = int(old.nunique()), int(new.nunique())
    reason = "ok"
    score = float("nan")
    if len(overlap) < 2:
        reason = "insufficient_overlap"
    elif min(old_count, new_count) < 2:
        reason = "degenerate_partition"
    else:
        score = float(adjusted_rand_score(old.to_numpy(), new.to_numpy()))
    return {
        "n_overlap": len(overlap), "overlap_start": overlap[0] if len(overlap) else pd.NaT,
        "overlap_end": overlap[-1] if len(overlap) else pd.NaT,
        "previous_states_on_overlap": old_count, "current_states_on_overlap": new_count,
        "adjusted_rand_index": score, "stability_status": reason,
    }


def empirical_regime_spells(regimes: pd.Series, observation_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Spells of latest signal label carried through subsequent observed closes.

    This carries labels, never posteriors or returns. Final spell is always
    right-censored; first is left-censored because its true onset is unknown.
    Counts include the start and exclude the next switching signal close.
    """
    _check_index(regimes.index)
    _check_index(observation_index)
    if regimes.empty or not regimes.isin(REGIMES).all():
        raise ValueError("Supply nonempty, known signal-close regime labels.")
    if not regimes.index.isin(observation_index).all():
        raise ValueError("Every signal close must exist in the observation index.")
    starts = np.flatnonzero(regimes.ne(regimes.shift()).to_numpy())
    rows = []
    for number, start in enumerate(starts):
        next_start = starts[number + 1] if number + 1 < len(starts) else len(regimes)
        signal_date = regimes.index[start]
        next_switch = regimes.index[next_start] if next_start < len(regimes) else pd.NaT
        observed = observation_index[observation_index >= signal_date]
        if pd.notna(next_switch):
            observed = observed[observed < next_switch]
        rows.append({
            "regime": regimes.iloc[start], "start_signal_close": signal_date,
            "last_observed_close": observed[-1], "next_switch_signal_close": next_switch,
            "n_signal_closes": next_start - start, "duration_trading_closes": len(observed),
            "left_censored": number == 0, "right_censored": next_start == len(regimes),
        })
    return pd.DataFrame(rows)


def momentum_volatility_regime(equity_returns: pd.Series) -> str:
    """Fixed 63-close rule: crisis vol >30%; bear vol >20% or momentum <=0."""
    window = equity_returns.tail(BASELINE_WINDOW).to_numpy(dtype=float)
    if len(window) != BASELINE_WINDOW or not np.isfinite(window).all() or np.any(window <= -1):
        raise ValueError("Momentum/volatility baseline needs 63 finite simple returns > -1.")
    momentum = float(np.prod(1.0 + window) - 1.0)
    vol = float(np.std(window, ddof=1) * np.sqrt(ANNUALIZATION_FACTOR))
    if vol > BASELINE_CRISIS_VOL:
        return "Crisis"
    return "Bear" if vol > BASELINE_HIGH_VOL or momentum <= 0.0 else "Bull"


def negative_control_assignments(index: pd.DatetimeIndex) -> pd.Series:
    """One IID uniform label per signal, PCG64 seed 1729; no data-dependent shuffle."""
    _check_index(index)
    rng = np.random.Generator(np.random.PCG64(NEGATIVE_CONTROL_SEED))
    labels = np.asarray(REGIMES)[rng.integers(0, len(REGIMES), size=len(index))]
    return pd.Series(labels, index=index, name="regime")


@dataclass
class RegimeIntelligenceResult:
    runs: Dict[str, StrategyRun]
    signals: pd.DataFrame
    diagnostics: pd.DataFrame
    transitions: pd.DataFrame
    persistence: pd.DataFrame
    stability: pd.DataFrame
    feature_means: pd.DataFrame
    spells: pd.DataFrame
    restarts: pd.DataFrame


def run_regime_intelligence(
    ctx: ResearchContext, *, on_strategy_start: Optional[Callable[[str], None]] = None,
) -> RegimeIntelligenceResult:
    """One bounded five-strategy comparison, with no search or return-driven choices."""
    cfg = ctx.config
    if len(ctx.prices) > MAX_PRICE_ROWS:
        raise ValueError(f"Experiment bound exceeded: at most {MAX_PRICE_ROWS} price rows.")
    if (cfg.train_window < max(ctx.min_history, cfg.hmm_config.minimum_training_observations)
            or ctx.min_history < max(BASELINE_WINDOW, cfg.hmm_config.minimum_training_observations)):
        raise ValueError("Training/gating history must support both the HMM and 63-row baseline.")
    positions = sorted(p for p in ctx.rebalance_positions if p + 1 >= ctx.min_history)
    if not positions or positions[0] >= len(ctx.asset_returns) - 1:
        raise ValueError("No eligible signal close with a following fill close.")
    if len(positions) > MAX_SIGNAL_CLOSES:
        raise ValueError(f"Experiment bound exceeded: at most {MAX_SIGNAL_CLOSES} HMM fits.")
    signal_index = pd.DatetimeIndex(ctx.asset_returns.index[positions])
    control = negative_control_assignments(signal_index)
    diagnostics, transitions, persistence, stability, means, restarts, signals = [], [], [], [], [], [], []
    previous_states = pd.Series(dtype=int, index=pd.DatetimeIndex([]))
    previous_date = pd.NaT

    def hmm_target(date, features, returns, previous):
        nonlocal previous_states, previous_date
        train = features.tail(cfg.train_window)
        scaler = fit_feature_scaler(train)
        model, hidden, _ = fit_hmm(transform_features(scaler, train), train, config=cfg.hmm_config)
        solution = predict_current_state(model, scaler, train, train, config=cfg.hmm_config)
        probabilities = solution.probabilities.reindex(REGIMES)
        diagnostics.append({
            "signal_close": date, "train_start": train.index[0], "train_end": train.index[-1],
            "n_train": len(train), "regime": solution.regime,
            **{f"posterior_{r}": float(probabilities[r]) for r in REGIMES},
            **uncertainty_diagnostics(probabilities), "converged": solution.convergence,
            "n_iter": solution.n_iter, "log_likelihood": solution.log_likelihood,
            "warning": solution.warning,
        })
        current = pd.Series(hidden, index=train.index)
        stability.append({"signal_close": date, "previous_signal_close": previous_date,
                          **overlap_state_stability(previous_states, current)})
        previous_states, previous_date = current, date
        matrix = solution.transition_matrix
        for source in matrix.index:
            for destination in matrix.columns:
                transitions.append({"signal_close": date, "from_regime": source,
                                    "to_regime": destination, "probability": float(matrix.loc[source, destination])})
        persistence.extend(transition_persistence(matrix).assign(signal_close=date).to_dict("records"))
        for state in solution.state_statistics:
            for feature, value in state.mean_features.items():
                means.append({"signal_close": date, "train_start": train.index[0],
                              "train_end": train.index[-1], "state_id_within_fit": state.state_id,
                              "regime": state.regime_label, "n_training_observations": state.n_observations,
                              "feature": feature, "mean_raw_feature": value})
        for restart in getattr(model, "_restart_diagnostics", []):
            restarts.append({**restart, "signal_close": date})
        return solution.regime

    def target_for(strategy_id, regime_fn=None, fixed=None):
        def target(date, features, returns, previous):
            regime = regime_fn(date, features, returns, previous) if regime_fn is not None else None
            weights = REGIME_FIXED_WEIGHTS[regime] if regime is not None else fixed
            row = {"strategy": strategy_id, "signal_close": date, "regime": regime}
            pos = ctx.asset_returns.index.get_loc(date)
            row["fill_close"] = ctx.asset_returns.index[pos + 1] if pos + 1 < len(ctx.asset_returns) else pd.NaT
            row["filled"] = pd.notna(row["fill_close"])
            row.update({f"target_{asset}": weights[asset] for asset in ctx.assets})
            signals.append(row)
            return [weights[asset] for asset in ctx.assets]
        return target

    targets = {
        "hmm_fixed": target_for("hmm_fixed", hmm_target),
        "momentum_volatility": target_for(
            "momentum_volatility",
            lambda date, features, returns, previous: momentum_volatility_regime(returns[cfg.equity_col]),
        ),
        "negative_control": target_for(
            "negative_control", lambda date, features, returns, previous: control.loc[date],
        ),
        "static_6040": target_for("static_6040", fixed=static_60_40_weights()),
        "equal_weight": target_for("equal_weight", fixed=equal_weight_weights()),
    }
    runs = {}
    for strategy_id in STRATEGY_IDS:
        if on_strategy_start is not None:
            on_strategy_start(strategy_id)
        runs[strategy_id] = run_target_strategy(ctx, strategy_id, targets[strategy_id])
    require_common_index(runs.values())
    diag = pd.DataFrame(diagnostics)
    labels = diag.set_index("signal_close")["regime"]
    return RegimeIntelligenceResult(
        runs=runs, signals=pd.DataFrame(signals), diagnostics=diag,
        transitions=pd.DataFrame(transitions), persistence=pd.DataFrame(persistence),
        stability=pd.DataFrame(stability), feature_means=pd.DataFrame(means),
        spells=empirical_regime_spells(labels, ctx.asset_returns.index),
        restarts=pd.DataFrame(restarts),
    )
