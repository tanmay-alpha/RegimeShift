import numpy as np


class PortfolioOptimizer:
    """Mean-variance optimizer with pure-numpy fallback (no scipy needed)."""

    def __init__(self, n_assets, gamma=1.0):
        self.n_assets = n_assets
        self.gamma = gamma

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
                # Scale the change down to respect turnover limit
                scale = max_turnover / (delta + 1e-12)
                w = current_weights + scale * (w - current_weights)
                w = np.clip(w, lb, ub)
                w = w / (w.sum() + 1e-12)

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
        if current_weights is not None:
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
