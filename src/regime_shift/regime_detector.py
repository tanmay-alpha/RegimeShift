"""
regime_detector.py -- Student-t Hidden Markov Model for market regime detection.

Replaces Gaussian emissions with Student-t to handle the fat tails
present in financial return distributions. Uses Dirichlet prior on
transition matrix to enforce realistic regime persistence (expected
duration ~50 days vs ~3 days for uniform transitions).

Mathematical foundations:
  - Student-t emission density: t_nu(x | mu_k, Sigma_k)
      log p(x|z=k) = lgamma((nu+d)/2) - lgamma(nu/2) - (d/2)log(nupi)
                   - 0.5log|Sigma_k| - ((nu+d)/2)log(1 + (1/nu)delta_k(x))
  - Baum-Welch EM: Baum et al.
  - Forward-backward in log-space with logsumexp normalization
  - Viterbi decoding: Viterbi
  - Dirichlet prior on transitions: A_kj = (N_kj + alpha_kj - 1) / Sigma_l(...)

References:
  - Baum, L.E., Petrie, T., Soules, G. & Weiss, N.. A maximization
    technique occurring in the statistical analysis of probabilistic functions
    of Markov chains. Annals of Mathematical Statistics, 41(1), 164-171.
  - Viterbi, A.J.. Error bounds for convolutional codes and an
    asymptotically optimum decoding algorithm. IEEE Transactions on IT, 13(2).
  - Hamilton, J.D.. A new approach to the economic analysis of
    nonstationary time series and the business cycle. Econometrica, 57(2).
  - Peel, D. & McLachlan, G.J.. Robust mixture modelling using
    the t distribution. Statistics and Computing, 10(4), 339-348.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .regime_signal import RegimeSignal

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Log-Gamma Implementation (Lanczos Approximation)
# -----------------------------------------------------------------------------

# Lanczos coefficients for g=7 (accurate to ~15 decimal places for positive reals)
_LANCZOS_G = 7
_LANCZOS_COEFFS = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
]

def _log_gamma(z: np.ndarray) -> np.ndarray:
    """
    Compute log(Gamma(z)) using Lanczos approximation.

    Uses the Lanczos approximation with g=7 coefficients for
    high accuracy across the positive real line. For z < 0.5,
    uses the reflection formula: Gamma(z) = pi / (sin(pi*z) * Gamma(1-z)).

    Reference: Lanczos, "A Precision Approximation of the Gamma Function"

    Args:
        z: Array of positive real values

    Returns:
        log(Gamma(z)) with high precision for each element in z
    """
    z = np.asarray(z, dtype=np.float64)

    g = _LANCZOS_G  # 7.0

    # Compute A_g(z) = c_0 + c_1/z + c_2/(z+1) + ... + c_g/(z+g-1)
    series = _LANCZOS_COEFFS[0]
    for i in range(1, len(_LANCZOS_COEFFS)):
        series = series + _LANCZOS_COEFFS[i] / (z + i - 1)

    # Lanczos: log Gamma(z) = 0.5*log(2*pi) + (z-0.5)*log(z+g-0.5) - (z+g-0.5) + log(A_g(z))
    log_sqrt_2pi = 0.5 * np.log(2.0 * np.pi)
    result = (log_sqrt_2pi + (z - 0.5) * np.log(z + g - 0.5)
              - (z + g - 0.5) + np.log(series + 1e-300))

    # For z < 0.5, use reflection: Gamma(z) = pi / (sin(pi*z) * Gamma(1-z))
    small = z < 0.5
    if np.any(small):
        z_small = z[small]
        log_sin = np.log(np.sin(np.pi * z_small) + 1e-300)
        log_gamma_1mz = _log_gamma(1.0 - z_small)
        result[small] = np.log(np.pi) - log_sin - log_gamma_1mz

    return result


#: Dirichlet prior concentration for self-transitions (regime persistence)
DIRICHLET_SELF_ALPHA: float = 50.0
#: Dirichlet prior concentration for cross-transitions
OFF_DIAG_ALPHA: float = 1.0

# Numerical stability
EPS: float = 1e-12
COV_JITTER: float = 1e-6

#: Default Student-t degrees of freedom (typical 3-8 for equities)
DEFAULT_NU: float = 5.0
#: Maximum EM iterations
MAX_EM_ITER: int = 30
#: Convergence tolerance for parameter changes
CONVERGENCE_TOL: float = 1e-4
#: Number of consecutive iterations below tolerance to declare convergence
CONVERGENCE_WINDOW: int = 5

#: Default regime labels for n_states
REGIME_LABELS: dict = {
    2: ["Bear", "Bull"],
    3: ["Crisis", "Bear", "Bull"],
    4: ["Extreme_Crisis", "Crisis", "Bear", "Bull"],
}


def _log_gamma_scalar(z: float) -> float:
    """
    Compute log(Gamma(z)) for a scalar using the Lanczos approximation.

    Args:
        z: Positive real value

    Returns:
        log(Gamma(z)) as Python float
    """
    g = float(_LANCZOS_G)

    # Compute A_g(z)
    series = float(_LANCZOS_COEFFS[0])
    for i in range(1, len(_LANCZOS_COEFFS)):
        series = series + _LANCZOS_COEFFS[i] / (z + i - 1)

    log_sqrt_2pi = 0.5 * math.log(2.0 * math.pi)
    result = (log_sqrt_2pi + (z - 0.5) * math.log(z + g - 0.5)
              - (z + g - 0.5) + math.log(series + 1e-300))

    if z < 0.5:
        result = math.log(math.pi) - math.log(math.sin(math.pi * z) + 1e-300) - _log_gamma_scalar(1.0 - z)

    return result


def _logsumexp(a: np.ndarray, axis: Optional[int] = None,
               keepdims: bool = False) -> np.ndarray:
    """
    Compute log(sum(exp(a))) in a numerically stable way.

    log(sum(exp(a))) = max(a) + log(sum(exp(a - max(a))))

    Always returns a numpy scalar or ndarray (never a Python float).

    Args:
        a: Input array
        axis: Axis along which to compute (None = all elements)
        keepdims: Whether to keep reduced dimensions

    Returns:
        log(sum(exp(a))) computed in a numerically stable way
        as numpy scalar (if axis=None) or numpy ndarray
    """
    a_max = np.amax(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)

    result = a_max + np.log(
        np.sum(np.exp(a - a_max), axis=axis, keepdims=True) + EPS
    )

    if not keepdims:
        if axis is None:
            # Reduce all dimensions -> scalar
            result = result.reshape(())
        else:
            # Reduce only the specified axis
            result = np.squeeze(result, axis=axis)

    return result


# -----------------------------------------------------------------------------
# Student-t Emission Density
# -----------------------------------------------------------------------------

def _student_t_log_density(X: np.ndarray, mu: np.ndarray,
                            cov: np.ndarray, nu: float) -> np.ndarray:
    """
    Compute log-density of multivariate Student-t distribution.

    t_nu(x | mu, Sigma) = Gamma((nu+2)/2) / (Gamma(nu/2) * nu^(d/2) * pi^(d/2) * |Sigma|^(1/2))
                    * [1 + (1/nu) * (x-mu)^T Sigma^{-1} (x-mu)]^(-(nu+d)/2)

    In log-space:
    log p(x) = lgamma((nu+d)/2) - lgamma(nu/2) - (d/2)log(nu*pi)
             - 0.5*log|Sigma| - ((nu+d)/2) * log(1 + (1/nu) * delta(x))

    where delta(x) = (x-mu)^T Sigma^{-1} (x-mu) is the squared Mahalanobis distance.

    Args:
        X: (n, d) observation matrix
        mu: (d,) mean vector
        cov: (d, d) covariance matrix (must be PSD)
        nu: Degrees of freedom (nu > 0)

    Returns:
        (n,) log-density values
    """
    n, d = X.shape

    # Regularize covariance for numerical stability
    cov_reg = cov + COV_JITTER * np.eye(d)

    # Compute log determinant
    sign, log_det = np.linalg.slogdet(cov_reg)
    if sign <= 0:
        # Covariance is not positive definite -- use identity
        logger.warning("Covariance not PSD (sign=%d), using identity fallback", sign)
        cov_reg = np.eye(d) * np.trace(cov_reg) / d + COV_JITTER * np.eye(d)
        sign, log_det = np.linalg.slogdet(cov_reg)

    # Compute inverse
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        logger.warning("Covariance inversion failed, using pseudo-inverse")
        cov_inv = np.linalg.pinv(cov_reg)

    # Squared Mahalanobis distance: delta = (x-mu)^T Sigma^{-1} (x-mu)
    diff = X - mu[np.newaxis, :]  # (n, d)
    # Vectorized: delta_i = sum_j diff[i,j] * sum_k cov_inv[j,k] * diff[i,k]
    mahal = np.sum(diff @ cov_inv * diff, axis=1)  # (n,)

    # Log-density
    # Constant term (independent of x)
    nu_d2 = (nu + d) / 2.0
    nu2 = nu / 2.0
    const = (_log_gamma_scalar(nu_d2)
             - _log_gamma_scalar(nu2)
             - (d / 2.0) * np.log(nu * np.pi)
             - 0.5 * log_det)

    # Data-dependent term
    log_density = const - ((nu + d) / 2.0) * np.log(1.0 + (1.0 / nu) * mahal + EPS)

    return log_density


# -----------------------------------------------------------------------------
# Silhouette Score
# -----------------------------------------------------------------------------

def _silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute mean silhouette score across all samples.

    s(i) = [b(i) - a(i)] / max(a(i), b(i))

    a(i) = mean distance from i to all other points in same cluster
    b(i) = min over other clusters of mean distance from i to that cluster

    Interpretation:
      s > 0.5:  good separation
      s > 0.7:  excellent separation
      s < 0.2:  poor separation - flag warning

    Args:
        features: (n, d) feature matrix
        labels: (n,) integer cluster labels

    Returns:
        Mean silhouette score in [-1, 1]
    """
    n = len(features)
    if n < 2:
        return 0.0

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0  # Single cluster -> silhouette undefined

    # Compute pairwise squared Euclidean distances using broadcasting
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i * x_j
    sq_norms = np.sum(features ** 2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (features @ features.T)
    dist_sq = np.maximum(dist_sq, 0.0)  # Clamp negative values from numerical error
    np.fill_diagonal(dist_sq, np.inf)  # Exclude self-distance

    silhouettes = np.zeros(n)

    for i in range(n):
        same_mask = (labels == labels[i]) & (np.arange(n) != i)  # exclude self
        other_mask = labels != labels[i]
        n_same = same_mask.sum()  # Already excludes self

        if n_same == 0:
            silhouettes[i] = 0.0
            continue

        # a(i): mean distance to points in same cluster (excluding self)
        a_i = dist_sq[i, same_mask].mean()
        a_i = np.sqrt(a_i)  # Convert back to distance (not squared)

        # b(i): min mean distance to points in other clusters
        b_i = np.inf
        for label in unique_labels:
            if label == labels[i]:
                continue
            mask = labels == label
            if mask.sum() > 0:
                mean_dist = np.sqrt(dist_sq[i, mask].mean())
                b_i = min(b_i, mean_dist)

        if b_i == np.inf:
            silhouettes[i] = 0.0
        else:
            max_ab = max(a_i, b_i)
            if max_ab > EPS:
                silhouettes[i] = (b_i - a_i) / max_ab
            else:
                silhouettes[i] = 0.0

    return float(np.nanmean(silhouettes))


# -----------------------------------------------------------------------------
# Regime Labeler
# -----------------------------------------------------------------------------

class RegimeLabeler:
    """
    Maps raw HMM state indices to Bull / Bear / Crisis labels.

    Labeling is based on the equity return feature: the state with
    highest mean equity return is 'Bull', lowest is 'Crisis', middle is 'Bear'.
    This is economically interpretable and consistent across fits.
    """

    def __init__(self, n_states: int = 3) -> None:
        self.n_states: int = n_states
        self.state_means: Optional[np.ndarray] = None
        self.labels: Optional[list[str]] = None
        self.label_to_idx: Optional[Dict[str, int]] = None

    def fit(self, features: np.ndarray, states: np.ndarray) -> list[str]:
        """
        Assign human-readable labels to each HMM state.

        Algorithm:
        1. Compute mean feature vector for each state
        2. Rank states by equity return feature (feature index 0)
        3. Assign: highest = "Bull", lowest = "Crisis", middle = "Bear"
        4. For 4 states: add "Extreme_Crisis" for lowest

        Args:
            features: (T, d) feature matrix
            states: (T,) integer state assignments from HMM

        Returns:
            List of label strings, one per state index [0..n_states-1]
        """
        k = self.n_states
        d = features.shape[1]

        # Compute mean feature vector for each state
        self.state_means = np.zeros((k, d))
        for s in range(k):
            mask = states == s
            if mask.sum() > 0:
                self.state_means[s] = features[mask].mean(axis=0)
            else:
                # Empty state -- use zeros (will be labeled appropriately)
                self.state_means[s] = np.zeros(d)

        # Rank states by first feature (equity return)
        equity_scores = self.state_means[:, 0]
        sorted_idx = np.argsort(equity_scores)  # ascending: lowest first

        label_map: Dict[int, str] = {}
        if k == 2:
            label_map[sorted_idx[0]] = "Bear"
            label_map[sorted_idx[1]] = "Bull"
        elif k == 3:
            label_map[sorted_idx[0]] = "Crisis"
            label_map[sorted_idx[1]] = "Bear"
            label_map[sorted_idx[2]] = "Bull"
        elif k == 4:
            label_map[sorted_idx[0]] = "Extreme_Crisis"
            label_map[sorted_idx[1]] = "Crisis"
            label_map[sorted_idx[2]] = "Bear"
            label_map[sorted_idx[3]] = "Bull"
        else:
            # Generic: highest = Bull, lowest = Bear, rest = Neutral_N
            label_map[sorted_idx[-1]] = "Bull"
            label_map[sorted_idx[0]] = "Bear"
            for rank, idx in enumerate(sorted_idx[1:-1]):
                label_map[idx] = f"Neutral_{rank}"

        self.labels = [label_map[i] for i in range(k)]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        logger.debug("State labels: %s", dict(zip(range(k), self.labels)))
        return self.labels


# -----------------------------------------------------------------------------
# Regime Detector -- Student-t HMM
# -----------------------------------------------------------------------------

class RegimeDetector:
    """
    Student-t Hidden Markov Model for market regime detection.

    Replaces Gaussian emissions with Student-t to handle the fat tails
    present in financial return distributions. Uses Dirichlet prior on
    transition matrix to enforce realistic regime persistence.

    Outputs rich RegimeSignal objects with confidence scores via
    predict_signal(), while maintaining backward compatibility through
    fit_predict() which returns a pd.Series of named regime labels.

    Attributes:
        n_states: Number of regime states (default 3: Bull, Bear, Crisis)
        lookback: Rolling window size for training (trading days)
        retrain_freq: Retrain every N days
        nu: Student-t degrees of freedom
        feature_engineer: RegimeFeatureEngineer instance
        labeler: RegimeLabeler instance
        transition_matrix: Learned transition matrix A[i,j] = P(z_t=j|z_{t-1}=i)
        silhouette_score: Last computed silhouette score (-1 to 1)
    """

    def __init__(self, n_states: int = 3, n_iter: int = MAX_EM_ITER,
                 random_state: int = 42, nu: float = DEFAULT_NU,
                 lookback: int = 252, retrain_freq: int = 21) -> None:
        """
        Args:
            n_states: Number of hidden regimes (2, 3, or 4)
            n_iter: Maximum EM iterations (Baum-Welch)
            random_state: RNG seed for reproducibility
            nu: Student-t degrees of freedom (fixed; typical 3-8 for equities)
            lookback: Training window size in trading days (for feature engineering)
            retrain_freq: Retrain frequency in trading days
        """
        self.n_states: int = n_states
        self.n_iter: int = n_iter
        self.random_state: int = random_state
        self.nu: float = nu
        self.lookback: int = lookback
        self.retrain_freq: int = retrain_freq

        # Learned parameters (set after fit)
        self.mu_: Optional[np.ndarray] = None      # (k, d) mean vectors
        self.cov_: Optional[np.ndarray] = None      # (k, d, d) covariance matrices
        self.trans_: Optional[np.ndarray] = None     # (k, k) transition matrix
        self.weights_: Optional[np.ndarray] = None   # (k,) initial state distribution
        self.n_features_: Optional[int] = None

        # State label mapping (set after fit via _label_states)
        self._label_map: Dict[int, str] = {}

        # Feature engineer and labeler
        from .regime_features import RegimeFeatureEngineer
        self.feature_engineer: RegimeFeatureEngineer = RegimeFeatureEngineer(
            lookback_window=lookback
        )
        self.labeler: RegimeLabeler = RegimeLabeler(n_states)

        # Diagnostics
        self.silhouette_score: float = 0.0
        self._em_iterations: int = 0
        self._log_likelihood: float = -np.inf

    # --------------------------------------------------------------------------
    # Public API -- Backward Compatible
    # --------------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fit (has learned parameters)."""
        return (self.mu_ is not None and self.cov_ is not None
                and self.trans_ is not None and self.weights_ is not None)

    def fit_predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Fit Student-t HMM on features and return regime labels.

        This is the main entry point used by WalkForwardBacktest.
        Maintains backward compatibility with the original Gaussian HMM API.

        Args:
            features: DataFrame of features (n_samples, n_features)

        Returns:
            pd.Series of named regime labels (Bull/Bear/Crisis) indexed by date
        """
        if features is None or len(features) == 0:
            return pd.Series(dtype=object)

        X = features.dropna().values.astype(np.float64)
        if len(X) < self.n_states:
            return pd.Series(
                "Bull", index=features.dropna().index, dtype=object
            )

        self.n_features_ = X.shape[1]
        self._fit(X)
        states = self._viterbi(X)

        # Label states semantically
        self._label_states(X, states)

        valid_idx = features.dropna().index
        named_states = [self._label_map.get(s, f"State_{s}") for s in states]
        return pd.Series(named_states, index=valid_idx[:len(states)], dtype=object)

    def predict_signal(self, prices: pd.DataFrame,
                       date: Optional[pd.Timestamp] = None) -> RegimeSignal:
        """
        Get a rich regime signal with confidence scores.

        This is the production interface. Returns a RegimeSignal with
        confidence scores, posteriors, and transition flags.

        Args:
            prices: DataFrame of daily prices up to current date
            date: Optional date for the prediction (not used, for API compat)

        Returns:
            RegimeSignal with full confidence information
        """
        features = self.feature_engineer.fit_transform(prices)
        if len(features) < self.n_states:
            # Not enough data -- return default signal
            default_labels = REGIME_LABELS.get(self.n_states, ["Crisis", "Bear", "Bull"])
            return RegimeSignal(
                label=default_labels[-1],
                confidence=0.5,
                posteriors={l: 1.0 / len(default_labels) for l in default_labels},
                regime_duration=0,
                is_transition=False,
            )

        X = features.values.astype(np.float64)
        self.n_features_ = X.shape[1]
        self._fit(X)
        states = self._viterbi(X)

        # Label states
        self._label_states(X, states)

        # Compute posteriors
        log_gamma = self._compute_log_posteriors(X)
        gamma = np.exp(log_gamma)

        # Get current regime (last time step)
        t = len(X) - 1
        current_state = states[t]
        current_label = self._label_map.get(current_state, f"State_{current_state}")

        # Build posteriors dict with labels
        posteriors: Dict[str, float] = {}
        for k in range(self.n_states):
            label = self._label_map.get(k, f"State_{k}")
            posteriors[label] = float(gamma[t, k])

        confidence = float(np.max(gamma[t]))

        # Compute regime duration and transition flag
        is_transition = False
        transition_from = None
        regime_duration = 1

        if t > 0:
            prev_state = states[t - 1]
            prev_label = self._label_map.get(prev_state, f"State_{prev_state}")
            if prev_state != current_state:
                is_transition = True
                transition_from = prev_label
                regime_duration = 1
            else:
                regime_duration = 2  # At least 2 days (today + yesterday)

            # Count backward for actual duration
            for s in range(t - 1, -1, -1):
                if states[s] == current_state:
                    regime_duration += 1
                else:
                    break

        # Expected duration from transition matrix
        expected_duration = None
        if self.trans_ is not None and current_state < len(self.trans_):
            p_stay = self.trans_[current_state, current_state]
            if p_stay < 1.0 - EPS:
                expected_duration = 1.0 / (1.0 - p_stay)

        return RegimeSignal(
            label=current_label,
            confidence=confidence,
            posteriors=posteriors,
            regime_duration=regime_duration,
            is_transition=is_transition,
            transition_from=transition_from,
            expected_duration=expected_duration,
        )

    def predict_signal_for_states(self, features: pd.DataFrame) -> pd.Series:
        """
        Fit and predict using a features DataFrame directly (no prices).

        Args:
            features: DataFrame of features

        Returns:
            pd.Series of named regime labels
        """
        return self.fit_predict(features)

    def get_regime_metrics(self) -> Dict:
        """
        Return diagnostic metrics from last fit.

        Returns:
            Dict with:
              - silhouette_score: regime separation quality
              - transition_matrix: learned transition probabilities
              - state_means: mean feature vector per regime
              - state_covs: covariance matrix per regime
              - regime_durations: expected duration per regime
              - n_iter: number of EM iterations
              - log_likelihood: final log-likelihood
        """
        if self.mu_ is None or self.cov_ is None or self.trans_ is None:
            return {}

        regime_durations = {}
        if self.trans_ is not None:
            for k in range(self.n_states):
                p_stay = self.trans_[k, k]
                if p_stay < 1.0 - EPS:
                    regime_durations[k] = float(1.0 / (1.0 - p_stay))
                else:
                    regime_durations[k] = float("inf")

        return {
            "silhouette_score": self.silhouette_score,
            "transition_matrix": self.trans_.tolist() if self.trans_ is not None else None,
            "state_means": self.mu_.tolist() if self.mu_ is not None else None,
            "state_covs": [c.tolist() for c in self.cov_] if self.cov_ is not None else None,
            "regime_durations": regime_durations,
            "n_iter": self._em_iterations,
            "log_likelihood": float(self._log_likelihood),
            "nu": self.nu,
            "n_states": self.n_states,
        }

    def get_state_name(self, state_idx: int) -> str:
        """Return human-readable name for a state index."""
        return self._label_map.get(state_idx, f"State_{state_idx}")

    def get_transition_matrix(self) -> Optional[np.ndarray]:
        """Return the learned (n_states x n_states) transition matrix."""
        return self.trans_

    # --------------------------------------------------------------------------
    # EM -- Baum-Welch Algorithm for Student-t HMM
    # --------------------------------------------------------------------------

    def _fit(self, X: np.ndarray) -> float:
        """
        Fit Student-t HMM via Baum-Welch (EM) and return log-likelihood.

        Args:
            X: (n, d) feature matrix

        Returns:
            Final log-likelihood
        """
        n, d = X.shape
        k = self.n_states
        rng = np.random.RandomState(self.random_state)

        # -- Initialization via quantile-based seeding --
        # Split feature 0 (equity return) into k quantile groups
        q_steps = np.linspace(10, 90, k)
        perc_vals = np.percentile(X[:, 0], q_steps)
        self.mu_ = np.zeros((k, d))
        for s in range(k):
            closest_idx = np.argsort(np.abs(X[:, 0] - perc_vals[s]))[: max(10, n // k)]
            self.mu_[s] = X[closest_idx].mean(axis=0)

        # Initialize covariances to empirical covariance (scaled)
        base_cov = np.cov(X.T)
        if d == 1:
            base_cov = np.array([[base_cov]])
        base_cov += COV_JITTER * np.eye(d)
        self.cov_ = np.array([base_cov.copy() for _ in range(k)])

        # Transition matrix: high self-transition (0.9) as initialization
        self.trans_ = np.eye(k) * 0.9 + 0.05
        self.trans_ /= self.trans_.sum(axis=1, keepdims=True)
        self.weights_ = np.ones(k, dtype=np.float64) / k

        # -- Dirichlet prior for transition matrix --
        self._dirichlet_alpha = np.full((k, k), OFF_DIAG_ALPHA)
        np.fill_diagonal(self._dirichlet_alpha, DIRICHLET_SELF_ALPHA)

        # -- EM iterations --
        prev_ll = -np.inf
        tolerance_history: list[float] = []
        self._em_iterations = 0

        for iteration in range(self.n_iter):
            # E-step
            log_emit = self._compute_log_emissions(X)  # (n, k)
            log_alpha, log_beta, log_gamma, xi = self._forward_backward_log(X, log_emit)

            # M-step
            gamma = np.exp(log_gamma - _logsumexp(log_gamma, axis=1, keepdims=True))
            self._m_step(X, gamma, xi)

            # Log-likelihood = sum_t log p(x_t | x_1:t-1) = sum_t log p(x_1:t) - log p(x_1:t-1)
            # Standard implementation: ll = sum_t log p(x_t | x_1..t) where the last alpha gives log p(X)
            # p(X) = sum_k alpha_T[k]
            ll = float(_logsumexp(log_alpha[-1]))
            self._log_likelihood = ll

            # Convergence check: relative improvement in log-likelihood
            if iteration > 0:
                rel_change = abs(ll - prev_ll) / (abs(prev_ll) + EPS)
                tolerance_history.append(rel_change)

                # Check if tolerance met for N consecutive iterations
                if len(tolerance_history) >= CONVERGENCE_WINDOW:
                    recent = tolerance_history[-CONVERGENCE_WINDOW:]
                    if all(t < CONVERGENCE_TOL for t in recent):
                        logger.debug("EM converged at iteration %d (rel_change=%.2e)",
                                     iteration, rel_change)
                        break

            prev_ll = ll
            self._em_iterations = iteration + 1

        # Apply Dirichlet prior to final transition matrix
        self._apply_dirichlet_prior()

        # Compute silhouette score
        states = self._viterbi(X)
        self.silhouette_score = _silhouette_score(X, states)

        logger.info(
            "Student-t HMM fitted: %d states, %d features, %d samples, "
            "%d EM iterations, log-likelihood=%.2f, silhouette=%.3f",
            k, d, n, self._em_iterations, self._log_likelihood,
            self.silhouette_score,
        )

        return self._log_likelihood

    # --------------------------------------------------------------------------
    # E-step: Forward-Backward in Log-Space
    # --------------------------------------------------------------------------

    def _compute_log_emissions(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log emission probabilities for all states.

        log p(x_t | z_t = k) = Student-t log-density

        Args:
            X: (n, d) observation matrix

        Returns:
            (n, k) log emission probabilities (raw log-densities, NOT normalized)
        """
        n, d = X.shape
        k = self.n_states
        log_emit = np.zeros((n, k))

        for s in range(k):
            log_emit[:, s] = _student_t_log_density(X, self.mu_[s], self.cov_[s], self.nu)

        # Clip to prevent extreme values that could cause numerical issues
        # (Student-t can give very low log-densities for outliers)
        log_emit = np.clip(log_emit, -1e10, 1e10)

        return log_emit

    def _forward_backward_log(self, X: np.ndarray,
                               log_emit: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                               np.ndarray, np.ndarray]:
        """
        Forward-backward algorithm entirely in log-space.

        Args:
            X: (n, d) observation matrix
            log_emit: (n, k) log emission probabilities

        Returns:
            log_alpha: (n, k) forward variables in log-space
            log_beta: (n, k) backward variables in log-space
            log_gamma: (n, k) state posteriors in log-space
            xi: (n-1, k, k) joint posterior xi_t(i,j)
        """
        n, k = len(X), self.n_states
        log_A = np.log(self.trans_ + EPS)
        log_pi = np.log(self.weights_ + EPS)

        # -- Forward pass (vectorized) --
        log_alpha = np.full((n, k), -np.inf)
        log_alpha[0] = log_pi + log_emit[0]

        for t in range(1, n):
            # log_alpha[t,j] = log_emit[t,j] + logsumexp(log_alpha[t-1,:] + log_A[:,j])
            # Vectorized: (k, k) matrix where [i,j] = log_alpha[t-1,i] + log_A[i,j]
            log_alpha[t, :] = log_emit[t, :] + _logsumexp(
                log_alpha[t - 1, :, None] + log_A, axis=0
            )

        # -- Backward pass (vectorized) --
        log_beta = np.full((n, k), -np.inf)
        log_beta[-1] = 0.0  # log(1) = 0

        for t in range(n - 2, -1, -1):
            # log_beta[t,i] = logsumexp(log_A[i,:] + log_emit[t+1,:] + log_beta[t+1,:])
            # Vectorized: (k, k) matrix where [i,j] = log_A[i,j] + log_emit[t+1,j] + log_beta[t+1,j]
            log_beta[t, :] = _logsumexp(
                log_A + log_emit[t + 1, None, :] + log_beta[t + 1, None, :], axis=1
            )

        # -- Posterior: log gamma --
        log_gamma = log_alpha + log_beta
        # Normalize
        log_gamma = log_gamma - _logsumexp(log_gamma, axis=1, keepdims=True)

        # -- Joint posterior xi_t(i,j) (vectorized) --
        xi = np.zeros((n - 1, k, k))
        if n > 1:
            # (n-1, k, k): xi[t, i, j] = alpha[t,i] + A[i,j] + emit[t+1,j] + beta[t+1,j]
            joint = log_alpha[:-1, :, None] + log_A[None, :, :] + \
                    log_emit[1:, None, :] + log_beta[1:, None, :]
            # Normalize each time step: xi[t] = exp(joint[t] - logsumexp(joint[t]))
            log_norm = _logsumexp(joint, axis=(1, 2), keepdims=True)
            xi = np.exp(joint - log_norm)

        return log_alpha, log_beta, log_gamma, xi

    # --------------------------------------------------------------------------
    # M-step: Parameter Updates
    # --------------------------------------------------------------------------

    def _m_step(self, X: np.ndarray, gamma: np.ndarray,
                xi: np.ndarray) -> None:
        """
        Update HMM parameters from sufficient statistics.

        Uses Student-t corrected covariance update:
        Sigma_k = (1/(nu+d)) * Sigma_t gamma_t(k) * (x_t-mu_k)(x_t-mu_k)^T / N_k
            + epsilon * I

        Args:
            X: (n, d) observation matrix
            gamma: (n, k) state responsibilities
            xi: (n-1, k, k) joint transition posteriors
        """
        n, d = X.shape
        k = self.n_states
        nu = self.nu

        # Effective number of points per state
        N_k = gamma.sum(axis=0)  # (k,)
        N_k = np.maximum(N_k, EPS)

        # -- Initial distribution pi --
        self.weights_ = gamma[0].copy()
        self.weights_ = np.maximum(self.weights_, EPS)
        self.weights_ /= self.weights_.sum()

        # -- Transition matrix A (correct Baum-Welch with xi) --
        # A_{ij} = Sigma_t xi_t(i,j) / Sigma_t gamma_t(i)
        new_trans = np.zeros((k, k))
        for i in range(k):
            denom = gamma[:-1, i].sum()
            if denom > EPS:
                for j in range(k):
                    new_trans[i, j] = xi[:, i, j].sum() / denom
            new_trans[i] = np.maximum(new_trans[i], 0.01)
            row_sum = new_trans[i].sum()
            if row_sum > EPS:
                new_trans[i] /= row_sum

        self.trans_ = new_trans

        # -- Means mu_k --
        for s in range(k):
            total_w = N_k[s]
            if total_w > EPS:
                self.mu_[s] = (gamma[:, s:s+1] * X).sum(axis=0) / total_w

        # -- Covariances Sigma_k (Student-t corrected) --
        for s in range(k):
            total_w = N_k[s]
            if total_w > EPS:
                diff = X - self.mu_[s][np.newaxis, :]  # (n, d)
                # Student-t correction: scale by (nu+d) instead of just dividing
                cov = np.einsum("t,ti,tj->ij", gamma[:, s], diff, diff) / total_w
                # Scale correction factor for Student-t
                scale = nu / (nu + d)
                self.cov_[s] = scale * cov + COV_JITTER * np.eye(d)

            # Ensure PSD
            self._ensure_psd(s)

    def _ensure_psd(self, state_idx: int) -> None:
        """Ensure covariance matrix is positive semi-definite."""
        cov = self.cov_[state_idx]
        d = cov.shape[0]

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        min_eigval = eigvals[0]

        if min_eigval < COV_JITTER:
            # Add jitter to make PSD
            jitter = COV_JITTER - min_eigval
            self.cov_[state_idx] = cov + jitter * np.eye(d) + COV_JITTER * np.eye(d)

        # Final safety: ensure diagonal dominance
        self.cov_[state_idx] += 1e-8 * np.eye(d)

    # --------------------------------------------------------------------------
    # Dirichlet Prior on Transition Matrix
    # --------------------------------------------------------------------------

    def _apply_dirichlet_prior(self) -> None:
        """
        Apply Dirichlet prior to the transition matrix.

        A_kj = (N_kj + alpha_kj - 1) / Sigma_l(N_kl + alpha_kl - 1)

        With alpha_kk=50 (self-transition) and alpha_kj=1 (cross-transition),
        this enforces realistic regime persistence (expected duration ~50 days).
        """
        if self.trans_ is None or not hasattr(self, '_dirichlet_alpha'):
            return

        k = self.n_states
        counts = self.trans_ * 100  # Scale to pseudo-counts (proportional to observed freq)
        alpha = self._dirichlet_alpha

        # Posterior: counts + alpha - 1
        posterior = counts + alpha - 1.0
        posterior = np.maximum(posterior, 0.01)  # Floor to prevent zero probabilities

        # Normalize rows
        row_sums = posterior.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, EPS)
        self.trans_ = posterior / row_sums

        logger.debug("Transition matrix after Dirichlet prior:\n%s", self.trans_)

    # --------------------------------------------------------------------------
    # Posterior Probability Computation
    # --------------------------------------------------------------------------

    def _compute_log_posteriors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities p(z_t=k | X) using forward-backward.

        Args:
            X: (n, d) observation matrix

        Returns:
            log_gamma: (n, k) log posterior probabilities
        """
        log_emit = self._compute_log_emissions(X)
        log_alpha, log_beta, log_gamma, _ = self._forward_backward_log(X, log_emit)
        return log_gamma

    # --------------------------------------------------------------------------
    # Viterbi -- MAP State Sequence
    # --------------------------------------------------------------------------

    def _viterbi(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm for globally optimal state sequence.

        Finds the single path maximizing p(z_1:T | X), using log-space
        to avoid underflow.

        Reference: Viterbi.

        Args:
            X: (n, d) observation matrix

        Returns:
            ndarray, shape (n,) -- integer state labels
        """
        log_A = np.log(self.trans_ + EPS)
        log_pi = np.log(self.weights_ + EPS)
        log_emit = self._compute_log_emissions(X)

        k, n = self.n_states, len(X)
        delta = np.full((n, k), -np.inf)
        psi = np.zeros((n, k), dtype=int)

        # Initialization
        delta[0] = log_pi + log_emit[0]

        # Recursion
        for t in range(1, n):
            for j in range(k):
                scores = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = log_emit[t, j] + scores[psi[t, j]]

        # Traceback
        states = np.zeros(n, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    # --------------------------------------------------------------------------
    # State Labeling
    # --------------------------------------------------------------------------

    def _label_states(self, X: np.ndarray, states: np.ndarray) -> None:
        """
        Assign semantic names to states based on their statistical properties.

        Strategy (for 3 states):
          - Sort states by the FIRST feature (equity return)
          - Highest mean return  -> "Bull"
          - Lowest mean return   -> "Crisis"
          - Middle               -> "Bear"

        Args:
            X: (T, d) feature matrix
            states: (T,) integer state assignments from HMM
        """
        labels = self.labeler.fit(X, states)

        # Build _label_map as state_idx -> label_name (the correct direction)
        self._label_map = {i: label for i, label in enumerate(labels)}

    # --------------------------------------------------------------------------
    # BIC Model Selection
    # --------------------------------------------------------------------------

    def select_n_states(self, features: pd.DataFrame,
                        candidates: Optional[List[int]] = None) -> int:
        """
        Select optimal number of states via BIC.

        Reference: Schwarz.

        Args:
            features: DataFrame of features
            candidates: List of n_states values to evaluate

        Returns:
            Optimal number of states
        """
        if candidates is None:
            candidates = [2, 3, 4, 5]

        X = features.dropna().values.astype(np.float64)
        if len(X) < max(candidates):
            return candidates[0]

        n_samples, n_dims = X.shape
        best_bic = float("inf")
        best_k = candidates[0]

        saved_k = self.n_states
        saved_nu = self.nu
        saved_iter = self.n_iter

        for k in candidates:
            if len(X) < k:
                continue

            sub = X[: min(len(X), 2000)]  # Subsample for speed
            self.n_states = k
            self.n_features_ = n_dims

            try:
                ll = self._fit(sub)
                if ll is None or not np.isfinite(ll):
                    continue

                # BIC parameter count:
                # k * d (means)
                # + k * d * (d+1) / 2 (covariance upper triangle)
                # + k * (k-1) (transition matrix, k-1 free per row)
                # + k - 1 (initial distribution)
                n_params = (k * n_dims
                            + k * n_dims * (n_dims + 1) // 2
                            + k * (k - 1)
                            + k - 1)
                bic = -2.0 * ll + n_params * np.log(n_samples)

                logger.debug("BIC for k=%d: %.2f (params=%d, ll=%.2f)",
                             k, bic, n_params, ll)

                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            except Exception as e:
                logger.warning("BIC evaluation failed for k=%d: %s", k, e)
                continue

        self.n_states = saved_k
        self.nu = saved_nu
        self.n_iter = saved_iter

        logger.info("BIC selected n_states=%d (BIC=%.1f)", best_k, best_bic)
        return best_k

    def __repr__(self) -> str:
        return (
            f"RegimeDetector(n_states={self.n_states}, nu={self.nu}, "
            f"lookback={self.lookback}, silhouette={self.silhouette_score:.3f})"
        )
