import numpy as np
from .regime_signal import RegimeSignal

# Regime-conditioned target weights (default values)
# These map regime labels to position preferences for each asset
REGIME_TARGETS: dict = {
    "Bull":       np.array([0.6, 0.3, 0.1]),   # Heavy equity, light defensive
    "Bear":       np.array([0.1, 0.2, 0.7]),   # Heavy bonds, light equity
    "Crisis":     np.array([0.3, 0.5, 0.2]),   # Gold-heavy, defensive
    "Extreme_Crisis": np.array([0.1, 0.6, 0.3]), # Very defensive
}


class PortfolioOptimizer:
    """Mean-variance optimizer with pure-numpy fallback (no scipy needed)."""

    def __init__(self, n_assets, gamma=1.0):
        self.n_assets = n_assets
        self.gamma = gamma
        self._regime_weights: dict[str, np.ndarray] = {}
        self._compute_regime_targets()

    def _compute_regime_targets(self) -> None:
        """Pre-compute regime-specific target weights for each supported regime."""
        for regime, target in REGIME_TARGETS.items():
            if len(target) == self.n_assets:
                self._regime_weights[regime] = target / (target.sum() + 1e-12)

    def solve(self, mu, cov, lb, ub, max_turnover=None, current_weights=None):
        """Solve the mean-variance optimization problem.

        Parameters
        ----------
        mu : ndarray, shape (n_assets,)
            Expected returns vector.
        cov : ndarray, shape (n_assets, n_assets)
            Covariance matrix.
        lb, ub : ndarray, shape (n_assets,)
            Lower/upper weight bounds.
        max_turnover : float or None
            Maximum total turnover from current_weights (L1 constraint).
        current_weights : ndarray or None
            Current portfolio weights (used to limit turnover).

        Returns
        -------
        ndarray — optimal weights summing to 1.
        """
        w = self._optimize_fallback(mu, cov, lb, ub, self.gamma,
                                    max_turnover, current_weights)

        # Enforce turnover constraint if specified
        if max_turnover is not None and current_weights is not None:
            delta = np.abs(w - current_weights).sum()
            if delta > max_turnover:
                scale = max_turnover / (delta + 1e-12)
                w = current_weights + scale * (w - current_weights)
                w = np.clip(w, lb, ub)
                w = w / (w.sum() + 1e-12)

        return w.astype(np.float64)

    def optimize_with_signal(self, expected_returns, cov_matrix,
                             signal: RegimeSignal) -> np.ndarray:
        """
        Compute confidence-weighted portfolio weights from a RegimeSignal.

        For each regime k with posterior P(z=k|X):
          w = Σ_k P(z=k|X) * w_k*

        where w_k* is the optimal weight vector for regime k.

        When confidence is low (< 0.7), blend toward risk-parity to
        reduce exposure during uncertain periods.

        Parameters
        ----------
        expected_returns : ndarray, shape (n_assets,)
            Expected returns vector.
        cov_matrix : ndarray, shape (n_assets, n_assets)
            Covariance matrix.
        signal : RegimeSignal
            Rich regime signal with confidence and posteriors.

        Returns
        -------
        ndarray — confidence-weighted weights summing to 1.0
        """
        n = self.n_assets
        lb = np.zeros(n)
        ub = np.ones(n)

        # Confidence-weighted blend of regime targets
        blended = np.zeros(n, dtype=np.float64)
        matched = 0.0

        for regime, prob in signal.posteriors.items():
            if regime in self._regime_weights:
                blended += prob * self._regime_weights[regime]
                matched += prob

        if matched < 0.5:
            # Low confidence — use the dominant regime
            dominant_regime = max(signal.posteriors, key=signal.posteriors.get)
            if dominant_regime in self._regime_weights:
                blended = self._regime_weights[dominant_regime].copy()

        # Normalize
        total = blended.sum()
        if total > 1e-12:
            blended = blended / total
        else:
            blended = np.ones(n, dtype=np.float64) / n

        # If confidence is low, blend toward risk parity
        if signal.confidence < 0.7:
            rp_weights = self._risk_parity_weights(cov_matrix)
            blend_factor = max(0.0, (0.7 - signal.confidence) / 0.7)
            blended = (1.0 - blend_factor) * blended + blend_factor * rp_weights

        # Run optimizer from blended starting point
        final = self._optimize_fallback(
            expected_returns, cov_matrix, lb, ub,
            self.gamma, None, blended
        )

        return final

    def _risk_parity_weights(self, cov: np.ndarray) -> np.ndarray:
        """
        Compute inverse-volatility (risk parity) weights.

        w_i ∝ 1 / σ_i

        This provides a safe default when model confidence is low.
        """
        n = self.n_assets
        cov_reg = cov + 1e-4 * np.eye(n)
        vols = np.sqrt(np.maximum(np.diag(cov_reg), 1e-12))
        inv_vol = 1.0 / (vols + 1e-12)
        w = inv_vol / inv_vol.sum()
        return w.astype(np.float64)

    def _optimize_fallback(self, mu, cov, lb, ub, gamma,
                           max_turnover, current_weights):
        """Projected gradient descent — no external dependencies."""
        n = self.n_assets
        cov_reg = cov + 1e-4 * np.eye(n)  # regularize for numerical stability

        # Initialize with risk-parity (inverse-vol weights)
        inv_vol = 1.0 / (np.sqrt(np.diag(cov_reg)) + 1e-12)
        w = inv_vol / inv_vol.sum()
        w = np.clip(w, lb, ub)
        w = w / (w.sum() + 1e-12)

        # If we have current weights, start closer to them (reduces turnover)
        if current_weights is not None and len(current_weights) == n:
            w = 0.5 * w + 0.5 * current_weights
            w = np.clip(w, lb, ub)
            w = w / (w.sum() + 1e-12)

        lr = 0.02  # learning rate
        for _ in range(200):
            # Gradient of (w @ mu - 0.5 * gamma * w @ cov @ w)
            # subject to sum(w)=1 via Lagrange multiplier
            grad = mu - gamma * (cov_reg @ w)
            # Projected gradient step
            w = w + lr * grad
            # Enforce box constraints
            w = np.clip(w, lb, ub)
            # Re-normalize to sum to 1
            w = w / (w.sum() + 1e-12)
            # Check convergence
            if np.linalg.norm(grad) < 1e-6:
                break

        # Enforce turnover constraint — project onto L1 ball
        if current_weights is not None and max_turnover is not None:
            delta = w - current_weights
            turnover = np.abs(delta).sum() / 2.0
            if turnover > max_turnover:
                scale = max_turnover / (turnover + 1e-12)
                delta = delta * min(scale, 1.0)
                w = current_weights + delta
                w = np.clip(w, lb, ub)
                w = w / (w.sum() + 1e-12)

        return w.astype(np.float64)
