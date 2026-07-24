"""
Gaussian Hidden Markov Model for market regime detection.

Fits a 3-state Gaussian HMM (hmmlearn) on scaled feature data and maps the
arbitrary numeric states to interpretable regime labels:

    Bull  — positive momentum, low volatility
    Bear  — negative momentum, elevated volatility
    Crisis — highest volatility and stress profile

Sequential inference is leakage-safe: the model is fit once on a training
window, and current-state inference uses only data through the current date.
No future observations are ever used for fitting or inference.

State interpretation uses only training-period raw feature statistics and is
deterministic given a fixed random seed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from regime_shift.config import HMMConfig
from regime_shift.exceptions import RegimeDetectionError
from regime_shift.features import transform_features

logger = logging.getLogger(__name__)

# Deterministic regime label ordering
_REGIME_LABELS = ["Bull", "Bear", "Crisis"]


@dataclass
class StateStatistics:
    """
    Per-state feature statistics from the training period.

    Attributes:
        state_id: Numeric HMM state (0, 1, 2).
        regime_label: Mapped regime name (Bull, Bear, or Crisis).
        mean_features: Mean raw feature values for this state.
        n_observations: Number of training observations assigned to this state.
    """

    state_id: int
    regime_label: str
    mean_features: pd.Series
    n_observations: int


@dataclass
class RegimeSolution:
    """
    Result of regime detection.

    Attributes:
        regime: Current regime label (Bull, Bear, or Crisis).
        probabilities: Posterior probabilities for each regime.
        transition_matrix: Labeled transition matrix DataFrame.
        state_statistics: Per-state training statistics.
        convergence: Whether the HMM converged during fitting.
        n_iter: Number of EM iterations performed.
        log_likelihood: Final log-likelihood of the fitted model.
        warning: Optional warning message.
    """

    regime: str
    probabilities: pd.Series
    transition_matrix: pd.DataFrame
    state_statistics: List[StateStatistics]
    convergence: bool
    n_iter: int
    log_likelihood: Optional[float] = None
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_hmm(
    scaled_features: pd.DataFrame,
    raw_features: pd.DataFrame,
    config: Optional[HMMConfig] = None,
) -> Tuple[GaussianHMM, np.ndarray, pd.DataFrame]:
    """
    Fit a 3-state Gaussian HMM on scaled feature data.

    This function is called only during the training phase.  It does NOT
    perform inference — use predict_current_state() for that.

    Args:
        scaled_features: Scaled feature DataFrame (training window only).
            Must have no NaN or infinite values.
        raw_features: Unscaled feature DataFrame (same rows) for state
            interpretation.
        config: HMMConfig instance.  Defaults to HMMConfig() if None.

    Returns:
        (fitted_hmm, hidden_states, labeled_transition_matrix)

    Raises:
        RegimeDetectionError: If input is invalid or HMM fails to converge.
    """
    config = config or HMMConfig()
    config.validate()

    _validate_hmm_input(scaled_features, "scaled_features")
    if len(scaled_features) < config.minimum_training_observations:
        raise RegimeDetectionError(
            f"Insufficient training observations: {len(scaled_features)} rows, "
            f"need at least {config.minimum_training_observations}."
        )

    # Verify raw_features has matching rows and columns
    _validate_hmm_input(raw_features, "raw_features")
    if len(raw_features) != len(scaled_features):
        raise RegimeDetectionError(
            f"raw_features has {len(raw_features)} rows but scaled_features has "
            f"{len(scaled_features)}. They must have the same number of observations."
        )
    # Verify column consistency
    if list(scaled_features.columns) != list(raw_features.columns):
        raise RegimeDetectionError(
            f"Column mismatch: scaled_features has columns {list(scaled_features.columns)} "
            f"but raw_features has {list(raw_features.columns)}."
        )

    # Fit HMM
    hmm = GaussianHMM(
        n_components=config.n_components,
        covariance_type=config.covariance_type,
        n_iter=config.n_iter,
        tol=config.tolerance,
        random_state=config.random_state,
        min_covar=config.min_covar,
        verbose=False,
    )

    try:
        hmm.fit(scaled_features.values)
    except Exception as exc:
        raise RegimeDetectionError(
            f"HMM fitting failed: {exc}"
        ) from exc

    # Check convergence
    if not hmm.monitor_.converged:
        log_ll = hmm.score(scaled_features.values)
        logger.warning(
            "HMM did not converge after %d iterations (log-likelihood: %.4f). "
            "Consider increasing n_iter or checking data quality.",
            hmm.monitor_.n_iter,
            log_ll,
        )

    # Predict hidden states on training data
    try:
        hidden_states = hmm.predict(scaled_features.values)
    except Exception as exc:
        raise RegimeDetectionError(
            f"HMM state prediction failed: {exc}"
        ) from exc

    # Map numeric states to regime labels
    state_map = _interpret_states(hidden_states, raw_features, config)

    # Compute state statistics
    state_stats = _compute_state_statistics(
        hidden_states, raw_features, state_map, config
    )

    # Build labeled transition matrix
    trans_mat = _build_labeled_transition_matrix(
        hmm.transmat_, state_map
    )

    # Store state interpretation on the model for later inference
    hmm._state_map = state_map  # type: ignore[attr-defined]
    hmm._state_statistics = state_stats  # type: ignore[attr-defined]

    log_ll = float(hmm.score(scaled_features.values))
    logger.info(
        "HMM fitted: %d iterations, converged=%s, log-likelihood=%.2f",
        hmm.monitor_.n_iter,
        hmm.monitor_.converged,
        log_ll,
    )

    return hmm, hidden_states, trans_mat


# ---------------------------------------------------------------------------
# Sequential inference
# ---------------------------------------------------------------------------

def predict_current_state(
    hmm: GaussianHMM,
    scaler: StandardScaler,
    raw_features_through_current: pd.DataFrame,
    raw_training: pd.DataFrame,
    config: Optional[HMMConfig] = None,
) -> RegimeSolution:
    """
    Predict the current market regime using data through the current date.

    This function performs inference only — the HMM and scaler are NOT refitted.
    The scaler was fitted only on training data, and the HMM was fit only on
    scaled training data.  The current observation is transformed using the
    existing scaler and passed to the already-fitted HMM.

    Args:
        hmm: Fitted GaussianHMM from fit_hmm().
        scaler: Fitted StandardScaler from fit_feature_scaler().
        raw_features_through_current: Raw features from training start through
            the current date (inclusive).  The current date is the last row.
        raw_training: Raw features used for the original training window.
            Used for state interpretation (must not include future data).
        config: HMMConfig instance.  Defaults to HMMConfig() if None.

    Returns:
        RegimeSolution with current regime, probabilities, and diagnostics.

    Raises:
        RegimeDetectionError: If inputs are invalid or model is not fitted.
    """
    config = config or HMMConfig()
    config.validate()

    _validate_hmm_input(raw_features_through_current, "raw_features_through_current")
    if raw_features_through_current.empty:
        raise RegimeDetectionError(
            "Cannot infer current state from empty feature data."
        )

    # Verify column consistency with training
    _check_column_consistency(raw_features_through_current, raw_training, "raw_features_through_current")

    # Extract the last observation (current date)
    current_raw = raw_features_through_current.iloc[-1:]
    current_scaled = transform_features(scaler, current_raw)

    # Posterior probability for the current observation
    try:
        posteriors = hmm.predict_proba(current_scaled.values)
    except Exception as exc:
        raise RegimeDetectionError(
            f"HMM posterior prediction failed: {exc}"
        ) from exc

    current_posterior = posteriors[-1]

    # Verify probabilities
    if not np.all(np.isfinite(current_posterior)):
        raise RegimeDetectionError(
            "HMM produced non-finite posterior probabilities."
        )
    prob_sum = float(current_posterior.sum())
    if not np.isclose(prob_sum, 1.0, atol=1e-6):
        raise RegimeDetectionError(
            f"HMM posterior probabilities do not sum to 1 (sum={prob_sum:.6f})."
        )

    # Use state interpretation from training (stored on model)
    state_map = getattr(hmm, "_state_map", None)
    if state_map is None:
        raise RegimeDetectionError(
            "HMM has not been properly fitted. "
            "Call fit_hmm() before predict_current_state()."
        )
    state_stats = getattr(hmm, "_state_statistics", [])

    # Build labeled transition matrix
    trans_mat = _build_labeled_transition_matrix(hmm.transmat_, state_map)

    # Determine most probable regime
    best_state = int(np.argmax(current_posterior))
    regime = state_map[best_state]

    # Build probability series with deterministic regime ordering
    prob_series = pd.Series(
        [float(current_posterior[s]) for s in range(len(current_posterior))],
        index=[state_map[s] for s in range(len(current_posterior))],
    )
    # Sort by regime label for deterministic output
    prob_series = prob_series.sort_index()

    log_ll = float(hmm.score(raw_training.values))
    warning = None
    if not hmm.monitor_.converged:
        warning = (
            f"HMM did not converge after {hmm.monitor_.n_iter} iterations. "
            "Results may be unreliable."
        )

    solution = RegimeSolution(
        regime=regime,
        probabilities=prob_series,
        transition_matrix=trans_mat,
        state_statistics=state_stats,
        convergence=bool(hmm.monitor_.converged),
        n_iter=int(hmm.monitor_.n_iter),
        log_likelihood=log_ll,
        warning=warning,
    )

    logger.info(
        "Current regime: %s (probabilities: %s)",
        regime,
        {k: f"{v:.3f}" for k, v in prob_series.items()},
    )

    return solution


# ---------------------------------------------------------------------------
# State interpretation
# ---------------------------------------------------------------------------

def _interpret_states(
    hidden_states: np.ndarray,
    raw_features: pd.DataFrame,
    config: HMMConfig,
) -> dict:
    """Map numeric HMM states to regime labels based on training statistics."""
    return _interpret_states_from_training(raw_features, config, hidden_states)


def _interpret_states_from_training(
    raw_training: pd.DataFrame,
    config: HMMConfig,
    hidden_states: Optional[np.ndarray] = None,
) -> dict:
    """
    Determine the numeric-state-to-regime mapping using training data only.

    Scoring rule (deterministic, based on training-period raw features):
        1. Crisis: state with the highest crisis score:
           score = volatility_weight * mean(equity_volatility_21d)
                 + vix_weight * mean(vix_level)      [if VIX exists]
                 + momentum_weight * mean(equity_momentum_63d)

        2. Bull: among the two remaining states, the one with highest
           mean(equity_momentum_63d).

        3. Bear: the remaining state.
    """
    if hidden_states is None:
        raise RegimeDetectionError(
            "hidden_states must be provided for state interpretation."
        )

    raw_arr = raw_training.values
    feature_cols = list(raw_training.columns)

    def _col_idx(name: str) -> int:
        if name not in feature_cols:
            return -1
        return feature_cols.index(name)

    vol_col = _col_idx("equity_volatility_21d")
    vix_col = _col_idx("vix_level")
    mom_col = _col_idx("equity_momentum_63d")

    # Compute mean feature values per state
    state_scores = {}
    for s in range(config.n_components):
        mask = hidden_states == s
        if mask.sum() == 0:
            # State never visited — assign lowest crisis priority
            state_scores[s] = -np.inf
            continue

        state_data = raw_arr[mask]
        vol_mean = float(np.mean(state_data[:, vol_col])) if vol_col >= 0 else 0.0
        vix_mean = float(np.mean(state_data[:, vix_col])) if vix_col >= 0 else 0.0
        mom_mean = float(np.mean(state_data[:, mom_col])) if mom_col >= 0 else 0.0

        score = (
            config.crisis_volatility_weight * vol_mean
            + config.crisis_vix_weight * vix_mean
            + config.crisis_momentum_weight * mom_mean
        )
        state_scores[s] = score

    # Step 1: Crisis = highest crisis score
    crisis_state = max(range(config.n_components), key=lambda s: state_scores[s])

    # Step 2: Bull = highest momentum_63d among remaining states
    remaining = [s for s in range(config.n_components) if s != crisis_state]
    bull_state = max(remaining, key=lambda s: float(np.mean(raw_arr[hidden_states == s, mom_col])) if mom_col >= 0 else 0.0)

    # Step 3: Bear = remaining state
    bear_state = [s for s in range(config.n_components) if s not in (crisis_state, bull_state)][0]

    mapping = {crisis_state: "Crisis", bull_state: "Bull", bear_state: "Bear"}

    logger.info(
        "State mapping: %s → Crisis (score=%.4f), "
        "%s → Bull, %s → Bear",
        crisis_state, state_scores[crisis_state],
        bull_state, bear_state,
    )

    return mapping


def _build_labeled_transition_matrix(
    transmat: np.ndarray,
    state_map: dict,
) -> pd.DataFrame:
    """Build a regime-labeled transition matrix DataFrame with deterministic ordering."""
    # Sort by regime label order for deterministic output
    regime_order = ["Bull", "Bear", "Crisis"]
    numeric_order = sorted(state_map.keys(), key=lambda s: regime_order.index(state_map[s]))

    labels = [state_map[s] for s in numeric_order]
    labeled = transmat[numeric_order][:, numeric_order]

    return pd.DataFrame(
        labeled,
        index=labels,
        columns=labels,
    )


def _compute_state_statistics(
    hidden_states: np.ndarray,
    raw_features: pd.DataFrame,
    state_map: dict,
    config: HMMConfig,
) -> List[StateStatistics]:
    """Compute per-state statistics from training data."""
    stats = []
    raw_arr = raw_features.values
    feature_names = list(raw_features.columns)

    for numeric_state in range(config.n_components):
        mask = hidden_states == numeric_state
        n_obs = int(mask.sum())
        if n_obs > 0:
            mean_vals = raw_arr[mask].mean(axis=0)
        else:
            mean_vals = np.full(len(feature_names), np.nan)
        mean_series = pd.Series(mean_vals, index=feature_names)

        stats.append(StateStatistics(
            state_id=numeric_state,
            regime_label=state_map[numeric_state],
            mean_features=mean_series,
            n_observations=n_obs,
        ))

    return stats


def _check_column_consistency(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    name: str,
) -> None:
    """Verify that two DataFrames have the same feature columns."""
    current_cols = list(current.columns)
    ref_cols = list(reference.columns)
    if current_cols != ref_cols:
        raise RegimeDetectionError(
            f"{name} has columns {current_cols} but training data has {ref_cols}. "
            "Feature columns must match exactly."
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_hmm_input(data: pd.DataFrame, name: str) -> None:
    """Validate that feature data is suitable for HMM fitting or inference."""
    if not isinstance(data, pd.DataFrame):
        raise RegimeDetectionError(
            f"{name} must be a pd.DataFrame, got {type(data).__name__}."
        )

    if data.empty:
        raise RegimeDetectionError(f"{name} is empty.")

    if data.isna().any().any():
        nan_cols = data.columns[data.isna().any()].tolist()
        raise RegimeDetectionError(
            f"{name} contains NaN in columns: {nan_cols}. "
            "Call drop_feature_warmup() before using."
        )

    if not np.all(np.isfinite(data.values)):
        raise RegimeDetectionError(
            f"{name} contains non-finite values (inf or -inf)."
        )


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

def get_transition_matrix(solution: RegimeSolution) -> pd.DataFrame:
    """Return the labeled transition matrix from a RegimeSolution."""
    return solution.transition_matrix


def get_state_statistics(solution: RegimeSolution) -> List[StateStatistics]:
    """Return per-state training statistics from a RegimeSolution."""
    return solution.state_statistics


def get_regime(solution: RegimeSolution) -> str:
    """Return the current regime label from a RegimeSolution."""
    return solution.regime


def get_probabilities(solution: RegimeSolution) -> pd.Series:
    """Return posterior regime probabilities from a RegimeSolution."""
    return solution.probabilities
