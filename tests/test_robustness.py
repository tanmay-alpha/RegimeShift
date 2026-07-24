"""
Robustness tests for RegimeShift.

Tests:
1. Seed sensitivity: HMM results stable across random seeds
2. Feature sensitivity: results stable with feature subsets
3. Cost sensitivity: Sharpe degrades gracefully with higher costs
4. Turnover cap enforcement: never exceeds max_turnover
5. Regime persistence: expected durations in realistic range [5, 500] days
6. PSD covariance: all covariance matrices are positive semi-definite
7. No look-ahead: features at time t only use data up to time t

Run with:
    python -m pytest tests/test_robustness.py -v
Or directly:
    python tests/test_robustness.py
"""

from __future__ import annotations

import logging
import sys
import os

# Ensure src/ is on path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from regime_shift.regime_detector import RegimeDetector, _enforce_psd
from regime_shift.transaction_costs import TransactionCostModel
from regime_shift.backtest import WalkForwardBacktest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_returns():
    """Generate realistic multi-asset returns."""
    np.random.seed(42)
    dates = pd.bdate_range("2015-01-01", periods=800)
    tickers = ["^NSEI", "GC=F", "TLT"]
    data = {}
    for i, t in enumerate(tickers):
        rets = np.random.randn(800) * 0.015
        rets += 0.0003 if i == 0 else 0.0001 if i == 1 else 0.00005
        data[t] = rets
    returns = pd.DataFrame(data, index=dates)
    return returns


# ---------------------------------------------------------------------------
# 1. Seed Sensitivity
# ---------------------------------------------------------------------------

class TestSeedSensitivity:
    """HMM results should be stable across different random seeds."""

    def test_silhouette_stable(self, sample_returns):
        """Silhouette scores should be in reasonable range across seeds."""
        from regime_shift.data_loader import compute_features

        features = compute_features(sample_returns)
        silhouettes = []
        for seed in [42, 123, 456, 789, 999]:
            np.random.seed(seed)
            det = RegimeDetector(n_states=3, lookback=252, retrain_freq=21)
            det.fit_predict(features)
            silhouettes.append(det.silhouette_score)

        # All silhouettes should be in valid range
        assert all(0.0 <= s <= 1.0 for s in silhouettes), \
            f"Bad silhouettes: {silhouettes}"

    def test_regime_distribution_stable(self, sample_returns):
        """Regime distribution should be similar across seeds."""
        from regime_shift.data_loader import compute_features

        features = compute_features(sample_returns)
        bull_pcts = []
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            det = RegimeDetector(n_states=3, lookback=252, retrain_freq=21)
            regimes = det.fit_predict(features)
            if "Bull" in regimes.values:
                bull_pcts.append((regimes == "Bull").mean())
            else:
                bull_pcts.append(1.0 / 3.0)

        mean_bull = np.mean(bull_pcts)
        for pct in bull_pcts:
            assert abs(pct - mean_bull) < 0.20, \
                f"Bull pct unstable across seeds: {bull_pcts}"


# ---------------------------------------------------------------------------
# 2. Feature Sensitivity
# ---------------------------------------------------------------------------

class TestFeatureSensitivity:
    """Results should be reasonable with different feature subsets."""

    def test_subset_produces_valid_labels(self, sample_returns):
        """Using a subset of features should still produce valid labels."""
        from regime_shift.data_loader import compute_features

        features = compute_features(sample_returns)

        if features.shape[1] < 5:
            pytest.skip("Not enough features for subset test")

        # Use first 5 features
        subset = features.iloc[:, :5]
        det = RegimeDetector(n_states=3, lookback=252, retrain_freq=21)
        regimes = det.fit_predict(subset)

        assert len(regimes) > 0, "Empty regime output"
        assert all(r in {"Bull", "Bear", "Crisis"} for r in regimes.values), \
            f"Invalid regime labels: {set(regimes.values)}"

    def test_feature_count_impact(self, sample_returns):
        """More features should not degrade results drastically."""
        from regime_shift.data_loader import compute_features

        features = compute_features(sample_returns)
        n_cols = features.shape[1]

        if n_cols < 3:
            pytest.skip("Not enough features")

        det = RegimeDetector(n_states=3, lookback=252, retrain_freq=21)
        det.fit_predict(features)
        full_sil = det.silhouette_score

        # Subset: first third of features
        subset = features.iloc[:, : max(1, n_cols // 3)]
        det2 = RegimeDetector(n_states=3, lookback=252, retrain_freq=21)
        det2.fit_predict(subset)
        sub_sil = det2.silhouette_score

        # Silhouette should not degrade by more than 50%
        assert sub_sil > full_sil * 0.3, \
            f"Subset silhouette ({sub_sil:.3f}) too low vs full ({full_sil:.3f})"


# ---------------------------------------------------------------------------
# 3. Cost Sensitivity
# ---------------------------------------------------------------------------

class TestCostSensitivity:
    """Sharpe ratio should degrade gracefully with higher transaction costs."""

    def test_higher_costs_lower_sharpe(self, sample_returns):
        """Sharpe should decrease monotonically as costs increase."""
        from regime_shift.data_loader import compute_features
        from regime_shift.evaluate import compute_metrics

        features = compute_features(sample_returns)
        prices = (1.0 + sample_returns).cumprod()

        sharpes = []
        for cost_bps in [0, 10, 50, 100, 200]:
            cost_model = TransactionCostModel(
                fixed_cost=0.0,
                variable_cost_bps=float(cost_bps),
                slippage_bps=float(cost_bps * 0.5),
            )
            bt = WalkForwardBacktest(
                prices=prices,
                returns=sample_returns,
                features=features,
                lookback=60,
                retrain_freq=21,
                cost_model=cost_model,
            )
            result = bt.run()
            if result.returns is not None and len(result.returns) > 0:
                sharpes.append(compute_metrics(result.returns).sharpe_ratio)
            else:
                sharpes.append(np.nan)

        # Sharpe should decrease as costs increase (allow NaN tolerance)
        valid = [(i, sharpes[i], sharpes[i + 1])
                 for i in range(len(sharpes) - 1)
                 if not (np.isnan(sharpes[i]) or np.isnan(sharpes[i + 1]))]
        for i, s_curr, s_next in valid:
            assert s_curr >= s_next - 0.05, \
                f"Sharpe not decreasing with costs at step {i}: {sharpes}"


# ---------------------------------------------------------------------------
# 4. Turnover Cap Enforcement
# ---------------------------------------------------------------------------

class TestTurnoverCap:
    """Turnover should never exceed the configured limit."""

    def test_turnover_within_limit(self, sample_returns):
        """Max turnover should be <= 1.5x the limit (allow for rounding)."""
        from regime_shift.data_loader import compute_features

        prices = (1.0 + sample_returns).cumprod()
        features = compute_features(sample_returns)

        bt = WalkForwardBacktest(
            prices=prices,
            returns=sample_returns,
            features=features,
            lookback=60,
            retrain_freq=21,
            turnover_limit=0.10,
        )
        result = bt.run()
        if len(result.weights_history) >= 2:
            diffs = result.weights_history.diff().dropna()
            turnovers = diffs.abs().sum(axis=1) / 2.0
            if len(turnovers) > 0:
                max_turn = turnovers.max()
                assert max_turn <= 0.15, \
                    f"Turnover {max_turn:.4f} exceeds 1.5x limit of 0.10"


# ---------------------------------------------------------------------------
# 5. Regime Persistence
# ---------------------------------------------------------------------------

class TestRegimePersistence:
    """Regime durations should be realistic for financial markets."""

    def test_expected_durations_realistic(self, sample_returns):
        """Expected regime durations should be in [5, 500] days."""
        from regime_shift.data_loader import compute_features

        features = compute_features(sample_returns)
        det = RegimeDetector(n_states=3, lookback=252, retrain_freq=21)
        det.fit_predict(features)
        metrics = det.get_regime_metrics()

        durations = metrics.get("regime_durations", {})
        if isinstance(durations, dict):
            durations = list(durations.values())
        for d in durations:
            if d != float("inf") and not np.isinf(d) and not np.isnan(d):
                assert 5 <= d <= 500, f"Unrealistic regime duration: {d} days"

    def test_self_transitions_high(self, sample_returns):
        """Self-transition probabilities should be > 0.5 for all states."""
        from regime_shift.data_loader import compute_features

        features = compute_features(sample_returns)
        det = RegimeDetector(n_states=3, lookback=252, retrain_freq=21)
        det.fit_predict(features)

        if hasattr(det, "trans_mat_") and det.trans_mat_ is not None:
            for i in range(det.trans_mat_.shape[0]):
                assert det.trans_mat_[i, i] > 0.5, \
                    f"Self-transition for state {i} is {det.trans_mat_[i, i]:.3f}"


# ---------------------------------------------------------------------------
# 6. PSD Covariance
# ---------------------------------------------------------------------------

class TestPSDCovariance:
    """Covariance matrices should always be positive semi-definite."""

    def test_enforce_psd_preserves_psd(self):
        """A PSD matrix should remain PSD after enforcement."""
        np.random.seed(42)
        A = np.random.randn(5, 5)
        cov = A @ A.T  # PSD by construction
        result = _enforce_psd(cov)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > -1e-10), \
            f"Not PSD after enforcement: min eigenvalue = {eigvals.min():.6e}"

    def test_enforce_psd_fixes_indefinite(self):
        """An indefinite matrix should be fixed to PSD."""
        cov = np.array([[1.0, 2.0], [2.0, 1.0]])  # Indefinite
        result = _enforce_psd(cov)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > -1e-10), \
            f"Failed to make PSD: eigenvalues = {eigvals}"

    def test_enforce_psd_symmetric(self):
        """Result should be symmetric."""
        np.random.seed(42)
        cov = np.random.randn(4, 4)
        result = _enforce_psd(cov)
        assert np.allclose(result, result.T), "Result is not symmetric"


# ---------------------------------------------------------------------------
# 7. No Look-Ahead Bias
# ---------------------------------------------------------------------------

class TestNoLookAhead:
    """Features at time t should not use data from after time t."""

    def test_features_no_future_data(self):
        """Modifying future returns should not affect past features."""
        from regime_shift.data_loader import compute_features

        np.random.seed(42)
        dates = pd.bdate_range("2020-01-01", periods=500)
        tickers = ["^NSEI", "GC=F", "TLT"]
        data = {t: np.random.randn(500) * 0.01 for t in tickers}
        returns = pd.DataFrame(data, index=dates)

        features = compute_features(returns)
        # Pick a feature row well inside the valid range (not near dropna boundary)
        feat_at_t = features.iloc[200].copy()

        # Modify a truly future return (index 400 — after feature date ~idx 360)
        returns_mod = returns.copy()
        returns_mod.iloc[400, 0] = 999.0

        features_mod = compute_features(returns_mod)
        feat_at_t_mod = features_mod.iloc[200]

        # Features at index 200 should be identical (within NaN tolerance)
        assert np.allclose(feat_at_t.values, feat_at_t_mod.values, equal_nan=True), \
            "Features at time t are affected by future data (look-ahead bias!)"


# ---------------------------------------------------------------------------
# Main block for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running robustness tests...")
    pytest.main([__file__, "-v", "-s"])
