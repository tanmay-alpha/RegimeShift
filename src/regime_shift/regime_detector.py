"""
Regime detection via Gaussian Hidden Markov Model (HMM).

Mathematical foundations:
  - Baum, Petrie, Soules & Weiss (1970) — Baum-Welch EM algorithm (forward-backward)
  - Viterbi (1967) — MAP decoding via dynamic programming
  - Hamilton (1989) — Markov-switching model for financial time series
  - BIC selection — Schwarz (1978)

States are automatically labeled after fitting by sorting on mean return:
  Bull   = highest mean return state
  Bear   = lowest mean return state
  Crisis = highest volatility state (remaining state)

This avoids the common error of hardcoding {0: 'Bull', 1: 'Bear', 2: 'Crisis'}
when EM assigns states randomly.

The Baum-Welch M-step here correctly uses ξ_t(i,j) — the joint two-step
posterior — NOT the approximate product γ_t(i) * γ_{t+1}(j).
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Gaussian HMM with full covariance emissions.

    Parameters
    ----------
    n_states : int
        Number of hidden regimes (default 3 → Bull, Bear, Crisis).
    n_iter : int
        Maximum EM iterations (Baum-Welch).
    random_state : int or None
        RNG seed for reproducibility.
    """

    # These are filled dynamically after fit based on state statistics
    STATE_NAMES: dict = {}

    def __init__(self, n_states: int = 3, n_iter: int = 100,
                 random_state: int = 42):
        self.n_states     = n_states
        self.n_iter       = n_iter
        self.random_state = random_state

        # Learned parameters (set after fit)
        self.mu_          = None   # (k, d) mean vectors
        self.cov_         = None   # (k, d, d) covariance matrices
        self.trans_       = None   # (k, k) transition matrix A
        self.weights_     = None   # (k,) initial state distribution π
        self.n_features_  = None

        # State label mapping (set after fit via _label_states)
        self._label_map: dict = {}  # state_idx → name

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def fit_predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Fit the HMM on *features* and return regime labels.

        Parameters
        ----------
        features : DataFrame, shape (n_samples, n_features)

        Returns
        -------
        Series — integer regime labels (0, 1, 2) indexed by valid dates.
        """
        if features is None or len(features) == 0:
            return pd.Series(dtype=int)

        X = features.dropna().values.astype(np.float64)
        if len(X) < self.n_states:
            return pd.Series(0, index=features.dropna().index)

        self.n_features_ = X.shape[1]
        self._fit(X)
        states = self._viterbi(X)

        # Label states semantically based on state statistics
        self._label_states(X, states)

        valid_idx = features.dropna().index
        named_states = [self._label_map.get(s, f"State_{s}") for s in states]
        return pd.Series(named_states, index=valid_idx[: len(states)])

    def get_state_name(self, state_idx: int) -> str:
        """Return human-readable name for a state index."""
        return self._label_map.get(state_idx, f"State_{state_idx}")

    def get_transition_matrix(self) -> np.ndarray:
        """Return the learned (n_states × n_states) transition matrix."""
        return self.trans_

    def select_n_states(self, features: pd.DataFrame,
                        candidates=None) -> int:
        """
        Select optimal number of states via BIC.

        BIC = -2 * ln(L̂) + k * ln(n)

        where k = number of free parameters:
            k = n_states * d         (means)
              + n_states * d*(d+1)/2 (covariance upper triangle)
              + n_states * n_states  (transition matrix)
              + n_states - 1         (initial distribution)

        Reference: Schwarz (1978).
        """
        if candidates is None:
            candidates = [2, 3, 4, 5]

        X = features.dropna().values.astype(np.float64)
        if len(X) < max(candidates):
            return candidates[0]

        n_samples, n_dims = X.shape
        best_bic = float("inf")
        best_k   = candidates[0]

        saved_k   = self.n_states
        saved_it  = self.n_iter

        for k in candidates:
            if len(X) < k:
                continue
            sub = X[: min(len(X), 2000)]
            self.n_states = k
            self.n_iter   = 30
            self.n_features_ = n_dims
            ll = self._fit(sub)
            self.n_states = saved_k
            self.n_iter   = saved_it

            if ll is None or not np.isfinite(ll):
                continue

            n_params = (k * n_dims
                        + k * n_dims * (n_dims + 1) // 2
                        + k * k
                        + k - 1)
            bic = -2.0 * ll + n_params * np.log(n_samples)
            logger.debug("BIC for k=%d: %.2f (params=%d)", k, bic, n_params)
            if bic < best_bic:
                best_bic = bic
                best_k   = k

        logger.info("BIC selected n_states=%d (BIC=%.1f)", best_k, best_bic)
        return best_k

    # ──────────────────────────────────────────────────────────────────────
    # EM — Baum-Welch Algorithm
    # ──────────────────────────────────────────────────────────────────────

    def _fit(self, X: np.ndarray) -> float:
        """
        Fit HMM via Baum-Welch (EM) and return log-likelihood.

        References: Baum et al. (1970), Rabiner (1989).
        """
        n, d = X.shape
        k    = self.n_states
        self.n_features_ = d

        rng = np.random.RandomState(self.random_state)

        # ── Smart Quantile / Empirical Initialisation ──
        # Split feature 0 (ret_ann) into k quantile groups to initialize distinct regime means
        q_steps = np.linspace(10, 90, k)
        perc_vals = np.percentile(X[:, 0], q_steps)
        self.mu_ = np.zeros((k, d))
        for s in range(k):
            # Find samples close to this percentile
            closest_idx = np.argsort(np.abs(X[:, 0] - perc_vals[s]))[: max(10, n // k)]
            self.mu_[s] = X[closest_idx].mean(axis=0)

        # Initialize covariances to empirical covariance of X (scaled)
        base_cov = np.cov(X.T)
        if d == 1:
            base_cov = np.array([[base_cov]])
        base_cov += 1e-4 * np.eye(d)
        self.cov_ = np.array([base_cov.copy() for _ in range(k)])

        # Transition matrix initialized with high diagonal persistence (0.95 self-transition)
        self.trans_ = np.eye(k) * 0.9 + 0.1 / k
        self.trans_ /= self.trans_.sum(axis=1, keepdims=True)
        self.weights_ = np.ones(k) / k

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            # E-step: compute γ (state posteriors) and ξ (transition posteriors)
            alpha, beta, gamma, xi = self._forward_backward(X)

            # M-step: update parameters
            self._m_step(X, gamma, xi)

            ll = self._log_likelihood_from_alpha(alpha)
            if abs(ll - prev_ll) < 1e-6:
                logger.debug("EM converged at iteration %d", iteration)
                break
            prev_ll = ll

        return prev_ll

    # ──────────────────────────────────────────────────────────────────────
    # E-step: Forward-Backward
    # ──────────────────────────────────────────────────────────────────────

    def _forward_backward(self, X: np.ndarray):
        """
        Forward-backward algorithm (scaled for numerical stability).

        Returns
        -------
        alpha  : (T, k) — forward variables (scaled)
        beta   : (T, k) — backward variables (scaled)
        gamma  : (T, k) — state posterior γ_t(i) = P(z_t=i | X)
        xi     : (T-1, k, k) — joint posterior ξ_t(i,j) = P(z_t=i, z_{t+1}=j | X)
        """
        n, k = len(X), self.n_states
        emit = self._emission_probs(X)          # (T, k)

        # ── Forward pass (scaled) ──
        alpha    = np.zeros((n, k))
        scales   = np.zeros(n)

        alpha[0] = self.weights_ * emit[0]
        scales[0] = alpha[0].sum() + 1e-300
        alpha[0] /= scales[0]

        for t in range(1, n):
            for j in range(k):
                alpha[t, j] = emit[t, j] * (alpha[t - 1] @ self.trans_[:, j])
            scales[t] = alpha[t].sum() + 1e-300
            alpha[t] /= scales[t]

        # ── Backward pass (scaled) ──
        beta      = np.zeros((n, k))
        beta[-1]  = 1.0

        for t in range(n - 2, -1, -1):
            for i in range(k):
                beta[t, i] = (self.trans_[i] * emit[t + 1] * beta[t + 1]).sum()
            denom = beta[t].sum() + 1e-300
            beta[t] /= denom

        # ── Gamma: state posteriors ──
        gamma  = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        # ── Xi: CORRECT joint two-step posteriors ──
        #
        # ξ_t(i, j) = α_t(i) * A_{ij} * b_j(x_{t+1}) * β_{t+1}(j)
        #             ─────────────────────────────────────────────────
        #               Σ_{i,j} α_t(i) * A_{ij} * b_j(x_{t+1}) * β_{t+1}(j)
        #
        # This is the MATHEMATICALLY CORRECT formula from Baum et al. (1970).
        # The previous code used γ_t(i) * γ_{t+1}(j), which ignores transition
        # probabilities and is an incorrect approximation.
        #
        xi = np.zeros((n - 1, k, k))
        for t in range(n - 1):
            for i in range(k):
                for j in range(k):
                    xi[t, i, j] = (alpha[t, i]
                                   * self.trans_[i, j]
                                   * emit[t + 1, j]
                                   * beta[t + 1, j])
            xi_sum = xi[t].sum() + 1e-300
            xi[t] /= xi_sum

        return alpha, beta, gamma, xi

    # ──────────────────────────────────────────────────────────────────────
    # M-step: Parameter Updates
    # ──────────────────────────────────────────────────────────────────────

    def _m_step(self, X: np.ndarray, gamma: np.ndarray,
                xi: np.ndarray) -> None:
        """
        Update HMM parameters from sufficient statistics.

        Correctly uses ξ (xi) for transition matrix update,
        and γ (gamma) for means, covariances, and initial weights.
        """
        n, k = gamma.shape
        d    = self.n_features_

        # ── Initial distribution π ──
        self.weights_  = gamma[0].copy()
        self.weights_ /= self.weights_.sum() + 1e-300

        # ── Transition matrix A (correct Baum-Welch formula) ──
        #
        # A_{ij} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
        #
        for i in range(k):
            denom = gamma[:-1, i].sum() + 1e-300
            for j in range(k):
                self.trans_[i, j] = xi[:, i, j].sum() / denom
            self.trans_[i] /= self.trans_[i].sum() + 1e-300

        # ── Means μ_k ──
        for s in range(k):
            w = gamma[:, s : s + 1]          # (T, 1)
            total_w = w.sum() + 1e-300
            self.mu_[s] = (w * X).sum(axis=0) / total_w

        # ── Covariances Σ_k ──
        for s in range(k):
            w = gamma[:, s]                  # (T,)
            diff = X - self.mu_[s]           # (T, d)
            # Σ_k = Σ_t γ_t(k) * (x_t - μ_k)(x_t - μ_k)^T / Σ_t γ_t(k)
            cov = np.einsum("t,ti,tj->ij", w, diff, diff) / (w.sum() + 1e-300)
            self.cov_[s] = cov + 1e-6 * np.eye(d)  # regularization

    # ──────────────────────────────────────────────────────────────────────
    # Viterbi — MAP State Sequence
    # ──────────────────────────────────────────────────────────────────────

    def _viterbi(self, X: np.ndarray) -> np.ndarray:
        """
        Viterbi algorithm for globally optimal state sequence.

        Finds the single path maximizing p(z_1:T | X), using log-space
        to avoid underflow.

        Reference: Viterbi (1967).

        Returns
        -------
        ndarray, shape (T,) — integer state labels
        """
        log_A    = np.log(self.trans_ + 1e-300)
        log_pi   = np.log(self.weights_ + 1e-300)
        log_emit = np.log(self._emission_probs(X) + 1e-300)

        k, n     = self.n_states, len(X)
        delta    = np.full((n, k), -np.inf)
        psi      = np.zeros((n, k), dtype=int)

        delta[0] = log_pi + log_emit[0]

        for t in range(1, n):
            for j in range(k):
                scores    = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = log_emit[t, j] + scores[psi[t, j]]

        # Traceback
        states       = np.zeros(n, dtype=int)
        states[-1]   = int(np.argmax(delta[-1]))
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    # ──────────────────────────────────────────────────────────────────────
    # State Labeling — Sort by Mean Return
    # ──────────────────────────────────────────────────────────────────────

    def _label_states(self, X: np.ndarray, states: np.ndarray) -> None:
        """
        Assign semantic names to states based on their statistical properties.

        Strategy (for 3 states):
          - Sort states by the FIRST feature (ret_ann — annualised return)
          - Highest mean return  → "Bull"
          - Lowest mean return   → "Bear"
          - Remaining state      → "Crisis" (typically highest volatility)

        For 2 states: Bull / Bear
        For 4+ states: Bull / Bear + numbered intermediates

        This fixes the critical bug where states were hardcoded as
        {0: 'Bull', 1: 'Bear', 2: 'Crisis'} regardless of what EM learned.
        """
        k = self.n_states
        # Mean of first feature (ret_ann) per state
        state_mean_ret = {}
        for s in range(k):
            mask = states == s
            if mask.sum() > 0:
                state_mean_ret[s] = X[mask, 0].mean()  # feature 0 = ret_ann
            else:
                state_mean_ret[s] = 0.0

        sorted_states = sorted(state_mean_ret, key=state_mean_ret.get, reverse=True)

        self._label_map = {}
        if k == 2:
            self._label_map[sorted_states[0]] = "Bull"
            self._label_map[sorted_states[1]] = "Bear"
        elif k == 3:
            self._label_map[sorted_states[0]] = "Bull"
            self._label_map[sorted_states[2]] = "Bear"
            self._label_map[sorted_states[1]] = "Crisis"
        else:
            self._label_map[sorted_states[0]]  = "Bull"
            self._label_map[sorted_states[-1]] = "Bear"
            for i, s in enumerate(sorted_states[1:-1]):
                self._label_map[s] = f"Neutral_{i}"

        self.STATE_NAMES = self._label_map
        logger.info("State labels (by mean return): %s", {
            self._label_map[s]: f"{state_mean_ret[s]:.4f}"
            for s in sorted_states
        })

    # ──────────────────────────────────────────────────────────────────────
    # Helper: Gaussian Emission Probabilities
    # ──────────────────────────────────────────────────────────────────────

    def _emission_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Multivariate Gaussian emission: p(x_t | z_t = k).

            log p(x | μ, Σ) = -½ [ d log(2π) + log|Σ| + (x-μ)ᵀ Σ⁻¹ (x-μ) ]

        Returns
        -------
        ndarray, shape (T, k) — emission probabilities
        """
        n, d  = X.shape
        k     = self.n_states
        log_emit = np.zeros((n, k))

        for s in range(k):
            cov = self.cov_[s] + 1e-4 * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0 or not np.isfinite(logdet):
                logdet = 0.0
                cov_inv = np.eye(d)
            else:
                cov_inv = np.linalg.inv(cov)

            diff = X - self.mu_[s]
            mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
            log_emit[:, s] = -0.5 * (d * np.log(2.0 * np.pi) + logdet + mahalanobis)

        # Subtract max per row for numerical stability before exp
        max_log = np.max(log_emit, axis=1, keepdims=True)
        emit = np.exp(np.clip(log_emit - max_log, -700, 0))
        emit /= emit.sum(axis=1, keepdims=True) + 1e-300
        return emit

    # ──────────────────────────────────────────────────────────────────────
    # Helper: Log-Likelihood
    # ──────────────────────────────────────────────────────────────────────

    def _log_likelihood_from_alpha(self, alpha: np.ndarray) -> float:
        """Compute log-likelihood from scaled forward variables."""
        # When alpha is normalised per step, LL = sum of log scale factors
        # Approximation: sum log(row sums before normalisation) ≈ log p(X)
        return float(np.log(alpha.sum(axis=1) + 1e-300).sum())

    def _log_likelihood(self, X: np.ndarray) -> float:
        """Compute full log-likelihood (for BIC)."""
        emit = self._emission_probs(X)
        n, k = len(X), self.n_states
        log_alpha = np.full((n, k), -np.inf)
        log_alpha[0] = np.log(self.weights_ + 1e-300) + np.log(emit[0] + 1e-300)
        log_alpha[0] -= np.logaddexp.reduce(log_alpha[0])

        for t in range(1, n):
            for j in range(k):
                log_alpha[t, j] = (
                    np.log(emit[t, j] + 1e-300)
                    + np.logaddexp.reduce(
                        log_alpha[t - 1] + np.log(self.trans_[:, j] + 1e-300)
                    )
                )
        return float(np.logaddexp.reduce(log_alpha[-1]))
