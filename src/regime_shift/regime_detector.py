"""
Regime detection via Gaussian Hidden Markov Model (HMM).

States: Bull (0), Bear (1), Crisis (2)

Fitting uses the Expectation-Maximization (EM) algorithm.  State
assignment after convergence uses the Viterbi algorithm (not
point-wise MAP) so that consecutive days respect the learned
transition matrix — producing persistent regime runs of 5–10 days
rather than daily flickering.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Gaussian HMM with full covariance emissions.

    Parameters
    ----------
    n_states : int
        Number of hidden regimes (default 3 → Bull, Bear, Crisis).
    n_iter : int
        Maximum EM iterations.
    random_state : int or None
        RNG seed for reproducibility.
    """

    STATE_NAMES = {0: "Bull", 1: "Bear", 2: "Crisis"}

    def __init__(self, n_states=3, n_iter=50, random_state=42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        # Learned parameters (set after fit)
        self.mu_ = None
        self.cov_ = None
        self.trans_ = None
        self.weights_ = None
        self.n_features_ = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(self, features):
        """Fit the HMM on *features* and return regime labels.

        Parameters
        ----------
        features : DataFrame, shape (n_samples, n_features)
            Feature matrix indexed by date.

        Returns
        -------
        Series
            Integer regime labels (0, 1, 2) indexed by the valid dates
            from *features*.
        """
        if features is None or len(features) == 0:
            return pd.Series(dtype=int)

        X = features.dropna().values
        if len(X) < self.n_states:
            return pd.Series("Bull", index=features.dropna().index)

        self.n_features_ = X.shape[1]
        self._fit_fallback(X)  # sets self.mu_, self.cov_, self.trans_, self.weights_
        states = self._viterbi(X)

        valid_idx = features.dropna().index
        return pd.Series(states, index=valid_idx[: len(states)])

    def get_transition_matrix(self):
        """Return the learned (n_states × n_states) transition matrix."""
        return self.trans_

    def get_state_name(self, state_idx):
        """Return human-readable name for a state index."""
        return self.STATE_NAMES.get(state_idx, f"State_{state_idx}")

    def select_n_states(self, features, candidates=None):
        """Select optimal number of states via BIC.

        Fits the HMM for each candidate ``n_states`` and returns the
        value with the lowest Bayesian Information Criterion.

        Parameters
        ----------
        features : DataFrame, shape (n_samples, n_features)
            Feature matrix indexed by date.
        candidates : list[int], optional
            State counts to evaluate.  Defaults to [2, 3, 4, 5].

        Returns
        -------
        int — the optimal number of states.
        """
        if candidates is None:
            candidates = [2, 3, 4, 5]

        X = features.dropna().values.astype(np.float64)
        if len(X) < candidates[0]:
            return candidates[0]

        n_samples, n_dims = X.shape
        best_bic = float("inf")
        best_k = candidates[0]

        for k in candidates:
            if len(X) < k:
                continue
            # Subsample for speed on large datasets
            sub = X[: min(len(X), 2000)]
            saved_n = self.n_states
            saved_iter = self.n_iter
            self.n_states = k
            self.n_iter = 20  # fewer iterations for BIC search
            ll = self._fit_fallback(sub)
            self.n_states = saved_n
            self.n_iter = saved_iter
            if ll is None or not np.isfinite(ll):
                continue

            # Parameter count: k means + k covariances + k transitions + k initial
            n_params = k * n_dims + k * n_dims * (n_dims + 1) // 2 + k * k + k - 1
            bic = -2.0 * ll + n_params * np.log(n_samples)

            logger.debug("BIC for k=%d: %.2f (params=%d)", k, bic, n_params)
            if bic < best_bic:
                best_bic = bic
                best_k = k

        logger.info("BIC selected n_states=%d (BIC=%.1f)", best_k, best_bic)
        return best_k

    # ------------------------------------------------------------------
    # Core: EM + Viterbi
    # ------------------------------------------------------------------

    def _fit_fallback(self, features):
        """Fit HMM parameters via EM and return log-likelihood."""
        n, d = features.shape
        k = self.n_states
        self.n_features_ = d

        rng = np.random.RandomState(self.random_state)

        # ---- Random initialization ----
        init_idx = rng.choice(n, k, replace=False)
        self.mu_ = features[init_idx].copy()
        self.cov_ = np.array([np.eye(d) for _ in range(k)])
        self.trans_ = rng.dirichlet(np.ones(k), size=k)
        self.weights_ = rng.dirichlet(np.ones(k))

        # ---- EM loop ----
        prev_log_lik = -np.inf
        for _ in range(self.n_iter):
            resp = self._e_step(features)
            self._m_step(features, resp)
            log_lik = self._log_likelihood(features)
            if abs(log_lik - prev_log_lik) < 1e-6:
                break
            prev_log_lik = log_lik

        return log_lik

    # ------------------------------------------------------------------
    # E-step: Forward-Backward
    # ------------------------------------------------------------------

    def _e_step(self, features):
        """Compute posterior state responsibilities via forward-backward."""
        n = len(features)
        k = self.n_states
        d = self.n_features_

        # Emission probabilities p(x_t | state=s)
        emit = self._emission_probs(features)

        # Forward pass
        alpha = np.zeros((n, k))
        alpha[0] = self.weights_ * emit[0]
        alpha[0] /= alpha[0].sum() + 1e-12
        for t in range(1, n):
            for s in range(k):
                alpha[t, s] = emit[t, s] * (alpha[t - 1] * self.trans_[:, s]).sum()
            alpha[t] /= alpha[t].sum() + 1e-12

        # Backward pass
        beta = np.zeros((n, k))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            for s in range(k):
                beta[t, s] = (self.trans_[s] * beta[t + 1] * emit[t + 1]).sum()

        # Posterior: p(s_t | X) ∝ alpha[t,s] * beta[t,s]
        resp = alpha * beta
        resp /= resp.sum(axis=1, keepdims=True) + 1e-12
        return resp

    # ------------------------------------------------------------------
    # M-step: Parameter updates
    # ------------------------------------------------------------------

    def _m_step(self, features, resp):
        """Update HMM parameters from posterior responsibilities."""
        n, d = features.shape
        k = self.n_states

        # Update initial weights
        self.weights_ = resp[0].copy()
        self.weights_ /= self.weights_.sum() + 1e-12

        # Update means and covariances
        for s in range(k):
            w = resp[:, s : s + 1]
            total_w = w.sum() + 1e-12
            self.mu_[s] = (w * features).sum(axis=0) / total_w
            diff = features - self.mu_[s]
            self.cov_[s] = (w * diff).T @ diff / total_w
            self.cov_[s] += 1e-6 * np.eye(d)

        # Update transition matrix
        for i in range(k):
            for j in range(k):
                num = (resp[:-1, i] * resp[1:, j]).sum()
                den = resp[:-1, i].sum() + 1e-12
                self.trans_[i, j] = num / den
            self.trans_[i] /= self.trans_[i].sum() + 1e-12

    # ------------------------------------------------------------------
    # Viterbi: Global most-likely state sequence
    # ------------------------------------------------------------------

    def _viterbi(self, features):
        """Viterbi algorithm for globally optimal state sequence.

        Unlike point-wise MAP (argmax per time step), Viterbi finds
        the single path that maximizes the joint probability
        p(state_sequence | observations), respecting transitions.

        Parameters
        ----------
        features : ndarray, shape (n, d)

        Returns
        -------
        ndarray of shape (n,) — integer state labels.
        """
        log_A = np.log(self.trans_ + 1e-12)
        log_pi = np.log(self.weights_ + 1e-12)
        log_emit = np.log(self._emission_probs(features) + 1e-12)

        k = self.n_states
        n = len(features)

        # Forward pass — store backpointers
        delta = np.zeros((n, k))
        psi = np.zeros((n, k), dtype=int)
        delta[0] = log_pi + log_emit[0]
        for t in range(1, n):
            for s in range(k):
                scores = delta[t - 1] + log_A[:, s]
                psi[t, s] = int(np.argmax(scores))
                delta[t, s] = log_emit[t, s] + scores[psi[t, s]]

        # Backward pass — traceback
        states = np.zeros(n, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emission_probs(self, features):
        """Compute Gaussian emission probabilities for all states.

        Returns
        -------
        ndarray, shape (n, k) — p(x_t | state=s)
        """
        n = len(features)
        k = self.n_states
        d = self.n_features_
        emit = np.zeros((n, k))

        for s in range(k):
            diff = features - self.mu_[s]
            cov = self.cov_[s] + 1e-6 * np.eye(d)
            cov_inv = np.linalg.inv(cov)
            cov_det = max(np.linalg.det(cov), 1e-12)
            norm_const = 0.5 * d * np.log(2.0 * np.pi) + 0.5 * np.log(cov_det)
            emit[:, s] = np.exp(
                -norm_const - 0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            )
        return emit

    def _log_likelihood(self, features):
        """Compute log-likelihood of data under current parameters."""
        emit = self._emission_probs(features)
        n = len(features)
        k = self.n_states

        log_alpha = np.zeros((n, k))
        log_alpha[0] = np.log(self.weights_ + 1e-12) + np.log(emit[0] + 1e-12)
        # Log-space normalization (log-sum-exp)
        log_alpha[0] -= np.logaddexp.reduce(log_alpha[0])

        for t in range(1, n):
            for s in range(k):
                log_alpha[t, s] = (
                    np.log(emit[t, s] + 1e-12)
                    + np.logaddexp.reduce(
                        log_alpha[t - 1] + np.log(self.trans_[:, s] + 1e-12)
                    )
                )

        return np.logaddexp.reduce(log_alpha[-1])
