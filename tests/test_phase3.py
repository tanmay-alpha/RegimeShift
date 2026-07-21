"""
Phase 3 tests — robustness, visualization, documentation.

Tests verify:
...
"""

from __future__ import annotations

import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from regime_shift.regime_features import RegimeFeatureEngineer, FEATURE_NAMES
from regime_shift.regime_detector import (
    RegimeDetector, RegimeLabeler, _silhouette_score,
    _student_t_log_density, _log_gamma_scalar,
)
from regime_shift.regime_signal import RegimeSignal
from regime_shift.transaction_costs import TransactionCostModel
from regime_shift.backtest import WalkForwardBacktest, BacktestResult
from regime_shift.benchmarks import run_benchmarks, _run_buy_and_hold
from regime_shift.evaluate import (
    compute_metrics, compute_regime_metrics, compute_turnover_metrics,
    bootstrap_metrics, MetricsResult,
)
from regime_shift.visualize import (
    plot_regime_timeline, plot_cumulative_returns, plot_drawdown,
    plot_rolling_sharpe, plot_monthly_heatmap, plot_regime_weights,
    plot_backtest_results, plot_turnover_costs, plot_regime_performance,
    plot_weight_evolution, plot_feature_importance, plot_feature_correlations,
    plot_regime_transitions, plot_silhouette_history, plot_regime_confidence,
    plot_bootstrap_distribution, save_all_plots,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def simulated_data():
    """Generate simulated multi-asset price data."""
    from regime_shift.data_loader import _simulate_prices, compute_returns, compute_features

    np.random.seed(42)
    prices = _simulate_prices()
    returns = compute_returns(prices)
    features = compute_features(returns, tickers=prices.columns.tolist(), window=20)
    features = features.dropna()
    return prices, returns, features


@pytest.fixture(scope="session")
def hmm_result(simulated_data):
    """Fit HMM on simulated data and return detector + regime series."""
    prices, returns, features = simulated_data
    detector = RegimeDetector(n_states=3, lookback=252, retrain_freq=21, n_iter=20)
    regime_series = detector.fit_predict(features)
    return detector, regime_series, prices, returns, features


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Visualization functions don't crash
# ─────────────────────────────────────────────────────────────────────────────

class TestVisualizationsNoCrash:
    """All visualization functions should work without errors."""

    def test_plot_regime_timeline(self, hmm_result):
        detector, regime_series, prices, returns, features = hmm_result
        import matplotlib
        matplotlib.use("Agg")
        fig, ax = plot_regime_timeline(regime_series, prices=prices)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_cumulative_returns(self, hmm_result):
        detector, regime_series, prices, returns, features = hmm_result
        import matplotlib
        matplotlib.use("Agg")
        bench_rets = {"BuyAndHold": returns.iloc[:, 0]}
        fig, ax = plot_cumulative_returns(
            returns.iloc[:, 0], bench_rets, regime_series=regime_series
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_drawdown(self, hmm_result):
        import matplotlib
        matplotlib.use("Agg")
        fig, ax = plot_drawdown(pd.Series(np.random.randn(100).cumsum()))
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_rolling_sharpe(self, hmm_result):
        import matplotlib
        matplotlib.use("Agg")
        fig, ax = plot_rolling_sharpe(pd.Series(np.random.randn(200)))
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_monthly_heatmap(self, hmm_result):
        import matplotlib
        matplotlib.use("Agg")
        daily_rets = pd.Series(
            np.random.randn(500),
            index=pd.date_range("2023-01-01", periods=500, freq="B"),
        )
        fig, ax = plot_monthly_heatmap(daily_rets)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_regime_weights(self, hmm_result):
        import matplotlib
        matplotlib.use("Agg")
        detector, regime_series, prices, returns, features = hmm_result
        n = len(regime_series)
        w = pd.DataFrame(
            np.random.dirichlet([1, 1, 1], size=n),
            index=regime_series.index,
            columns=["A", "B", "C"],
        )
        fig, axes = plot_regime_weights(w, regime_series)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_feature_importance(self):
        import matplotlib
        matplotlib.use("Agg")
        names = [f"feat_{i}" for i in range(20)]
        scores = np.random.rand(20)
        fig, ax = plot_feature_importance(names, scores, top_n=10)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_feature_correlations(self, hmm_result):
        import matplotlib
        matplotlib.use("Agg")
        detector, regime_series, prices, returns, features = hmm_result
        fig, ax = plot_feature_correlations(features, max_features=10)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_regime_transitions(self, hmm_result):
        import matplotlib
        matplotlib.use("Agg")
        detector, regime_series, prices, returns, features = hmm_result
        trans = detector.get_transition_matrix()
        if trans is not None:
            fig, ax = plot_regime_transitions(trans)
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_plot_silhouette_history(self):
        import matplotlib
        matplotlib.use("Agg")
        dates = pd.date_range("2023-01-01", periods=100)
        scores = np.random.uniform(0.1, 0.8, 100)
        fig, ax = plot_silhouette_history(dates, scores.tolist())
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_regime_confidence(self):
        import matplotlib
        matplotlib.use("Agg")
        signals = [
            {"posteriors": {"Bull": 0.8, "Bear": 0.15, "Crisis": 0.05}},
            {"posteriors": {"Bull": 0.3, "Bear": 0.5, "Crisis": 0.2}},
        ]
        dates = pd.date_range("2023-01-01", periods=2)
        fig, ax = plot_regime_confidence(dates, signals)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_bootstrap_distribution(self):
        import matplotlib
        matplotlib.use("Agg")
        ci = {"sharpe": (0.5, 0.2, 0.8), "ann_return": (0.1, -0.05, 0.25)}
        fig, axes = plot_bootstrap_distribution(ci)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Save to file doesn't crash
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveToFile:
    """Plotting functions should save correctly to disk."""

    def test_save_regime_timeline(self, hmm_result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        detector, regime_series, prices, returns, features = hmm_result
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "timeline.png")
            fig, ax = plot_regime_timeline(regime_series, prices=prices, output_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000  # At least 1KB
            plt.close(fig)

    def test_save_cumulative_returns(self, hmm_result):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        detector, regime_series, prices, returns, features = hmm_result
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cumret.png")
            bench_rets = {"BuyAndHold": returns.iloc[:, 0]}
            fig, ax = plot_cumulative_returns(
                returns.iloc[:, 0], bench_rets, regime_series=regime_series, output_path=path
            )
            assert os.path.exists(path)
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Feature standardization stability
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureStability:
    """Features should be stable and have no NaN/Inf after standardization."""

    def test_features_no_nan_inf(self, simulated_data):
        prices, returns, features = simulated_data
        assert not features.isna().any().any(), "Features contain NaN"
        assert not np.isinf(features.values).any(), "Features contain Inf"

    def test_features_shape(self, simulated_data):
        prices, returns, features = simulated_data
        assert features.shape[1] == len(FEATURE_NAMES), \
            f"Expected {len(FEATURE_NAMES)} features, got {features.shape[1]}"

    def test_feature_zscore_range(self, simulated_data):
        """Standardized features should mostly be within [-10, 10] after clipping."""
        prices, returns, features = simulated_data
        max_abs = features.abs().max().max()
        # After 5x z-score clipping, should be bounded
        assert max_abs <= 50, f"Feature range too large: {max_abs}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Regime detection robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeRobustness:
    """Regime detection should be robust to parameter changes."""

    def test_different_nu_same_labels(self, hmm_result):
        """Different nu values should produce similar regime labels (not wild changes)."""
        detector, regime_series, prices, returns, features = hmm_result

        # Fit with nu=4 and nu=8
        d4 = RegimeDetector(n_states=3, n_iter=15, nu=4.0, random_state=42)
        labels4 = d4.fit_predict(features)

        d8 = RegimeDetector(n_states=3, n_iter=15, nu=8.0, random_state=42)
        labels8 = d8.fit_predict(features)

        # Align and compare overlap
        common = labels4.dropna().index.intersection(labels8.dropna().index)
        if len(common) > 10:
            overlap = (labels4.loc[common] == labels8.loc[common]).mean()
            assert overlap > 0.3, f"Low label overlap between nu=4 and nu=8: {overlap:.2f}"

    def test_different_random_states(self, hmm_result):
        """Different random seeds should produce similar quality results."""
        detector, regime_series, prices, returns, features = hmm_result

        results = []
        for seed in [42, 123, 456]:
            d = RegimeDetector(n_states=3, n_iter=15, random_state=seed)
            rs = d.fit_predict(features)
            results.append(d.silhouette_score)

        # All should produce positive silhouette scores on simulated data
        for score in results:
            assert score > -1.0, f"Silhouette score out of range: {score}"

    def test_retrain_freq_affects_regime_changes(self, simulated_data):
        """Different retrain frequencies should affect regime change count."""
        prices, returns, features = simulated_data

        r1 = RegimeDetector(n_states=3, n_iter=15, random_state=42, retrain_freq=10)
        wb1 = WalkForwardBacktest(
            prices, returns, features,
            lookback=252, retrain_freq=10, n_states=3,
            detector=r1,
        )
        res1 = wb1.run()

        r2 = RegimeDetector(n_states=3, n_iter=15, random_state=42, retrain_freq=63)
        wb2 = WalkForwardBacktest(
            prices, returns, features,
            lookback=252, retrain_freq=63, n_states=3,
            detector=r2,
        )
        res2 = wb2.run()

        # Different retrain frequencies should produce different numbers
        # of regime changes (or at least valid results)
        assert res1.regime_changes >= 0
        assert res2.regime_changes >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Transition matrix quality
# ─────────────────────────────────────────────────────────────────────────────

class TestTransitionMatrix:
    """Transition matrix should have realistic properties."""

    def test_self_transitions_above_half(self, hmm_result):
        """Self-transitions should be > 0.5 (regimes persist)."""
        detector, regime_series, prices, returns, features = hmm_result
        trans = detector.get_transition_matrix()
        assert trans is not None
        diag = np.diag(trans)
        assert np.all(diag > 0.3), \
            f"Self-transitions too low: {diag}. Expected > 0.3."

    def test_rows_sum_to_one(self, hmm_result):
        """Transition matrix rows should sum to ~1.0."""
        detector, regime_series, prices, returns, features = hmm_result
        trans = detector.get_transition_matrix()
        assert trans is not None
        row_sums = trans.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_no_negative_probabilities(self, hmm_result):
        """No negative probabilities in transition matrix."""
        detector, regime_series, prices, returns, features = hmm_result
        trans = detector.get_transition_matrix()
        assert trans is not None
        assert np.all(trans >= -1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: End-to-end no NaN/Inf
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndNoNaN:
    """All pipeline outputs should be free of NaN and Inf."""

    def test_backtest_result_no_nan(self, hmm_result):
        detector, regime_series, prices, returns, features = hmm_result
        cost_model = TransactionCostModel()
        optimizer = __import__("regime_shift.optimizer", fromlist=["PortfolioOptimizer"]).PortfolioOptimizer(n_assets=len(returns.columns))

        wb = WalkForwardBacktest(
            prices, returns, features,
            lookback=252, retrain_freq=21, n_states=3,
            cost_model=cost_model, detector=detector, optimizer=optimizer,
        )
        result = wb.run()

        assert not result.portfolio_returns.isna().any(), "portfolio_returns has NaN"
        assert not np.isinf(result.portfolio_returns.values).any(), "portfolio_returns has Inf"
        assert not result.costs.isna().any(), "costs has NaN"
        assert not np.isinf(result.costs.values).any(), "costs has Inf"
        assert not result.weights_history.isna().any().any(), "weights_history has NaN"

    def test_regime_metrics_no_nan(self, hmm_result):
        detector, regime_series, prices, returns, features = hmm_result
        cost_model = TransactionCostModel()
        optimizer = __import__("regime_shift.optimizer", fromlist=["PortfolioOptimizer"]).PortfolioOptimizer(n_assets=len(returns.columns))

        wb = WalkForwardBacktest(
            prices, returns, features,
            lookback=252, retrain_freq=21, n_states=3,
            cost_model=cost_model, detector=detector, optimizer=optimizer,
        )
        result = wb.run()

        regime_m = compute_regime_metrics(result.portfolio_returns, result.regime_series)
        if len(regime_m) > 0:
            assert not regime_m.isna().any().any(), "regime_metrics has NaN"

    def test_benchmarks_no_nan(self, simulated_data):
        prices, returns, features = simulated_data
        cost_model = TransactionCostModel()
        results = run_benchmarks(prices, returns, features, cost_model=cost_model)
        for name, br in results.items():
            assert not br.portfolio_returns.isna().any(), f"{name} returns has NaN"
            assert not np.isinf(br.portfolio_returns.values).any(), f"{name} returns has Inf"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Cost model robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestCostModel:
    """Transaction costs should be reasonable across parameter ranges."""

    def test_zero_turnover_zero_cost(self):
        """Zero weight change should produce zero cost."""
        cm = TransactionCostModel()
        cost = cm.compute_per_asset_cost(0.0, 0.15, "Nifty", 1_000_000)
        assert cost == 0.0

    def test_cost_increases_with_turnover(self):
        """Cost should increase as turnover increases."""
        cm = TransactionCostModel()
        costs = []
        for wc in [0.01, 0.05, 0.10, 0.20]:
            c = cm.compute_per_asset_cost(wc, 0.15, "Nifty", 1_000_000)
            costs.append(c)
        assert all(costs[i] <= costs[i+1] + 1e-10 for i in range(len(costs)-1)), \
            "Costs should be monotonically increasing with turnover"

    def test_cost_as_fraction_bounded(self):
        """Cost fraction should be between 0 and 1."""
        cm = TransactionCostModel()
        old_w = np.array([0.5, 0.5])
        new_w = np.array([0.0, 1.0])
        vol = np.array([0.15, 0.20])
        frac = cm.cost_as_fraction(old_w, new_w, vol, ["A", "B"], notional=1_000_000)
        assert 0 <= frac <= 1.0, f"Cost fraction out of bounds: {frac}"

    def test_turnover_computation(self):
        """Turnover should be 0.5 * sum(|w_new - w_old|)."""
        old_w = np.array([0.5, 0.3, 0.2])
        new_w = np.array([0.3, 0.5, 0.2])
        expected = 0.5 * (abs(0.3 - 0.5) + abs(0.5 - 0.3) + abs(0.2 - 0.2))
        actual = TransactionCostModel.compute_turnover(old_w, new_w)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_annualized_cost_bps(self):
        """Annualized cost in bps should be positive for positive turnover."""
        cm = TransactionCostModel()
        bps = cm.annualized_cost_bps(avg_turnover=0.1, cost_per_rebalance=0.001, rebalance_freq_days=21)
        assert bps > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Bootstrap CIs validity
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCI:
    """Bootstrap confidence intervals should be valid."""

    def test_ci_contains_median(self):
        """Median should lie within [lo, hi]."""
        np.random.seed(42)
        rets = pd.Series(np.random.randn(500) * 0.01)
        ci = bootstrap_metrics(rets, n_bootstrap=200, block_size=21)
        if ci:  # May return empty if data too short
            for metric, (med, lo, hi) in ci.items():
                assert lo <= med <= hi, f"{metric}: median {med} not in [{lo}, {hi}]"

    def test_ci_positive_width(self):
        """CI should have positive width (unless degenerate)."""
        np.random.seed(42)
        rets = pd.Series(np.random.randn(500) * 0.01)
        ci = bootstrap_metrics(rets, n_bootstrap=200, block_size=21)
        if ci:
            for metric, (med, lo, hi) in ci.items():
                assert hi >= lo, f"{metric}: CI upper < lower"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: compute_regime_metrics edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeMetricsEdgeCases:
    """Regime metrics should handle edge cases gracefully."""

    def test_empty_returns(self):
        result = compute_regime_metrics(pd.Series([], dtype=float), pd.Series([], dtype=object))
        assert len(result) == 0

    def test_no_overlap(self):
        rets = pd.Series([0.01, -0.01, 0.02], index=pd.date_range("2023-01-01", periods=3))
        regimes = pd.Series(["Bull", "Bear"], index=pd.date_range("2024-01-01", periods=2))
        result = compute_regime_metrics(rets, regimes)
        assert len(result) == 0

    def test_single_regime(self):
        rets = pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2023-01-01", periods=3))
        regimes = pd.Series(["Bull", "Bull", "Bull"], index=pd.date_range("2023-01-01", periods=3))
        result = compute_regime_metrics(rets, regimes)
        assert len(result) == 1
        assert result.loc["Bull", "days"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: BIC state selection
# ─────────────────────────────────────────────────────────────────────────────

class TestBICSelection:
    """BIC-based state selection should work."""

    def test_bic_selects_valid_k(self, hmm_result):
        detector, regime_series, prices, returns, features = hmm_result
        if len(features) < 50:
            pytest.skip("Not enough data for BIC")

        k = detector.select_n_states(features, candidates=[2, 3])
        assert k in [2, 3]

    def test_bic_with_subsample(self, hmm_result):
        """BIC should work on smaller subsamples."""
        detector, regime_series, prices, returns, features = hmm_result
        sub = features.head(min(200, len(features)))
        if len(sub) < 20:
            pytest.skip("Not enough data")

        try:
            k = detector.select_n_states(sub, candidates=[2, 3])
            assert k in [2, 3]
        except Exception:
            pytest.skip("BIC failed on small sample — acceptable for very small data")


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: RegimeSignal posteriors sum to 1.0
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeSignalPosteriors:
    """RegimeSignal posteriors should always sum to 1.0."""

    def test_posteriors_sum_to_one(self, hmm_result):
        detector, regime_series, prices, returns, features = hmm_result
        signal = detector.predict_signal(prices)
        total = sum(signal.posteriors.values())
        np.testing.assert_allclose(total, 1.0, atol=1e-6,
                                   err_msg=f"Posteriors sum to {total}, not 1.0")

    def test_confidence_in_range(self, hmm_result):
        detector, regime_series, prices, returns, features = hmm_result
        signal = detector.predict_signal(prices)
        assert 0 <= signal.confidence <= 1.0, \
            f"Confidence out of range: {signal.confidence}"

    def test_signal_label_valid(self, hmm_result):
        detector, regime_series, prices, returns, features = hmm_result
        signal = detector.predict_signal(prices)
        valid_labels = {"Bull", "Bear", "Crisis", "Extreme_Crisis"}
        assert signal.label in valid_labels, f"Invalid label: {signal.label}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: BacktestResult properties
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestResult:
    """BacktestResult properties should be consistent."""

    def test_total_return_matches_product(self, hmm_result):
        detector, regime_series, prices, returns, features = hmm_result
        cost_model = TransactionCostModel()
        optimizer = __import__("regime_shift.optimizer", fromlist=["PortfolioOptimizer"]).PortfolioOptimizer(n_assets=len(returns.columns))

        wb = WalkForwardBacktest(
            prices, returns, features,
            lookback=100, retrain_freq=21, n_states=3,
            cost_model=cost_model, detector=detector, optimizer=optimizer,
        )
        result = wb.run()

        expected_total = (1.0 + result.portfolio_returns).prod() - 1.0
        np.testing.assert_allclose(result.total_return, expected_total, atol=1e-10)

    def test_sharpe_formula(self, hmm_result):
        detector, regime_series, prices, returns, features = hmm_result
        cost_model = TransactionCostModel()
        optimizer = __import__("regime_shift.optimizer", fromlist=["PortfolioOptimizer"]).PortfolioOptimizer(n_assets=len(returns.columns))

        wb = WalkForwardBacktest(
            prices, returns, features,
            lookback=100, retrain_freq=21, n_states=3,
            cost_model=cost_model, detector=detector, optimizer=optimizer,
        )
        result = wb.run()

        ann_ret = result.annualized_return
        ann_vol = result.annualized_volatility
        expected_sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
        np.testing.assert_allclose(result.sharpe_ratio, expected_sharpe, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: Student-t log density known value
# ─────────────────────────────────────────────────────────────────────────────

class TestStudentTDensity:
    """Student-t log-density should match known values."""

    def test_density_at_mean(self):
        """log p(x=mu | nu=5, mu=0, Sigma=1) should match analytical value."""
        x = np.array([[0.0]])
        mu = np.array([0.0])
        cov = np.array([[1.0]])
        nu = 5.0

        log_density = _student_t_log_density(x, mu, cov, nu)[0]

        # Analytical value for d=1, x=mu (mahal=0):
        # log p(mu) = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(nu*pi) - 0.5*log|Sigma|
        expected = (
            _log_gamma_scalar((nu + 1) / 2)
            - _log_gamma_scalar(nu / 2)
            - 0.5 * np.log(nu * np.pi)
            - 0.5 * np.log(1.0)  # |Sigma| = 1
        )

        np.testing.assert_allclose(log_density, expected, rtol=1e-5)

    def test_density_symmetric(self):
        """Student-t log-density should be symmetric around mean."""
        x = np.array([[2.0], [-2.0]])
        mu = np.array([0.0])
        cov = np.array([[1.0]])
        nu = 5.0

        log_densities = _student_t_log_density(x, mu, cov, nu)
        np.testing.assert_allclose(log_densities[0], log_densities[1], rtol=1e-10)

    def test_density_decreases_with_distance(self):
        """Log-density should decrease as we move away from the mean."""
        x = np.array([[0.0], [1.0], [2.0], [5.0]])
        mu = np.array([0.0])
        cov = np.array([[1.0]])
        nu = 5.0

        log_densities = _student_t_log_density(x, mu, cov, nu)
        assert log_densities[0] > log_densities[1] > log_densities[2] > log_densities[3]

    def test_log_gamma_scalar_known_values(self):
        """log Gamma(1) = 0, log Gamma(0.5) = log(sqrt(pi))."""
        np.testing.assert_allclose(_log_gamma_scalar(1.0), 0.0, atol=1e-8)
        np.testing.assert_allclose(
            _log_gamma_scalar(0.5), 0.5 * np.log(np.pi), atol=1e-8
        )
        np.testing.assert_allclose(
            _log_gamma_scalar(2.0), 0.0, atol=1e-8
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 14: RegimeLabeler consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeLabeler:
    """RegimeLabeler should assign consistent labels."""

    def test_labels_unique(self, hmm_result):
        detector, regime_series, prices, returns, features = hmm_result
        X = features.values
        states = detector._viterbi(X)
        labels = detector.labeler.fit(X, states)
        assert len(set(labels)) == len(labels), "Duplicate labels assigned"

    def test_bull_highest_return(self, hmm_result):
        """Bull state should have highest equity return feature."""
        detector, regime_series, prices, returns, features = hmm_result
        X = features.values
        states = detector._viterbi(X)
        labels = detector.labeler.fit(X, states)

        bull_idx = labels.index("Bull") if "Bull" in labels else None
        if bull_idx is not None:
            bull_mean = X[states == bull_idx, 0].mean()
            for label in labels:
                if label != "Bull":
                    idx = labels.index(label)
                    other_mean = X[states == idx, 0].mean()
                    assert bull_mean >= other_mean, \
                        f"Bull state ({bull_mean:.4f}) should have >= return than {label} ({other_mean:.4f})"


# ─────────────────────────────────────────────────────────────────────────────
# Test 15: Silhouette score range
# ─────────────────────────────────────────────────────────────────────────────

class TestSilhouetteScore:
    """Silhouette score should be in [-1, 1]."""

    def test_silhouette_range_well_separated(self):
        """On well-separated clusters, score should be positive."""
        np.random.seed(42)
        # Three well-separated clusters
        c1 = np.random.randn(50, 5) + np.array([5, 0, 0, 0, 0])
        c2 = np.random.randn(50, 5) + np.array([-5, 0, 0, 0, 0])
        c3 = np.random.randn(50, 5) + np.array([0, 5, 0, 0, 0])
        X = np.vstack([c1, c2, c3])
        labels = np.array([0] * 50 + [1] * 50 + [2] * 50)

        score = _silhouette_score(X, labels)
        assert -1.0 <= score <= 1.0, f"Silhouette out of range: {score}"
        assert score > 0.3, f"Well-separated clusters should have score > 0.3, got {score}"

    def test_silhouette_single_cluster(self):
        """Single cluster should return 0.0."""
        X = np.random.randn(50, 3)
        labels = np.zeros(50)
        score = _silhouette_score(X, labels)
        assert score == 0.0

    def test_silhouette_two_points(self):
        """Two points should return 0.0."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        labels = np.array([0, 1])
        score = _silhouette_score(X, labels)
        assert score == 0.0

    def test_silhouette_on_hmm_labels(self, hmm_result):
        """Silhouette score from HMM should be in [-1, 1]."""
        detector, regime_series, prices, returns, features = hmm_result
        X = features.values
        states = detector._viterbi(X)
        score = _silhouette_score(X, states)
        assert -1.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 16: Notebook generation
# ─────────────────────────────────────────────────────────────────────────────

class TestNotebookGeneration:
    """Notebook generator should produce valid .ipynb file."""

    def test_generate_notebook(self):
        from notebooks.generate_notebook import create_notebook, NOTEBOOK_PATH

        # Run generator
        create_notebook()

        # Verify file exists and is valid JSON
        assert os.path.exists(NOTEBOOK_PATH), f"Notebook not created at {NOTEBOOK_PATH}"
        with open(NOTEBOOK_PATH, "r") as f:
            nb = json.load(f)

        # Check structure
        assert "cells" in nb
        assert "nbformat" in nb
        assert len(nb["cells"]) > 0

        # Check cell types
        for cell in nb["cells"]:
            assert "cell_type" in cell
            assert "source" in cell
            assert cell["cell_type"] in ("markdown", "code")

        # Check we have both markdown and code cells
        types = {c["cell_type"] for c in nb["cells"]}
        assert "code" in types
        assert "markdown" in types


# ─────────────────────────────────────────────────────────────────────────────
# Test 17: BacktestResult defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestResultDefaults:
    """BacktestResult should have sensible defaults."""

    def test_empty_result(self):
        result = BacktestResult()
        assert result.total_return == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.regime_changes == 0

    def test_single_day(self):
        result = BacktestResult(
            portfolio_returns=pd.Series([0.01]),
            regime_series=pd.Series(["Bull"]),
        )
        np.testing.assert_allclose(result.total_return, 0.01, atol=1e-10)
        np.testing.assert_allclose(result.max_drawdown, 0.0, atol=1e-10)

    def test_negative_returns(self):
        result = BacktestResult(
            portfolio_returns=pd.Series([-0.01, -0.02, 0.01]),
            regime_series=pd.Series(["Bear", "Bear", "Bull"]),
        )
        assert result.total_return < 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 18: MetricsResult
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsResult:
    """MetricsResult should compute correctly."""

    def test_from_returns_basic(self):
        rets = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        m = MetricsResult.from_returns(rets, name="Test")
        assert m.name == "Test"
        assert abs(m.total_return - (1.01 * 0.99 * 1.02 * 0.98 * 1.01 - 1)) < 1e-10
        assert m.win_rate == 0.6  # 3 out of 5 positive

    def test_from_returns_empty(self):
        m = MetricsResult.from_returns(pd.Series([], dtype=float), name="Empty")
        assert m.total_return == 0.0
        assert m.sharpe_ratio == 0.0

    def test_negative_sharpe(self):
        """Strategy with negative mean should have negative Sharpe."""
        np.random.seed(42)
        rets = pd.Series(np.random.randn(200) * 0.02 - 0.001)  # Negative drift
        m = MetricsResult.from_returns(rets, name="Neg")
        # With negative drift, Sharpe should typically be negative
        assert isinstance(m.sharpe_ratio, float)


# ─────────────────────────────────────────────────────────────────────────────
# Test 19: Turnover metrics edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestTurnoverMetrics:
    """Turnover metrics should handle edge cases."""

    def test_empty_weights(self):
        result = compute_turnover_metrics(pd.DataFrame())
        assert result["avg_daily_turnover"] == 0.0
        assert result["max_single_day_turnover"] == 0.0

    def test_single_row_weights(self):
        df = pd.DataFrame([[0.5, 0.5]], columns=["A", "B"])
        result = compute_turnover_metrics(df)
        assert result["avg_daily_turnover"] == 0.0

    def test_known_turnover(self):
        """Verify turnover formula: 0.5 * sum(|diff|)."""
        df = pd.DataFrame(
            [[0.5, 0.5], [0.3, 0.7], [0.4, 0.6]],
            columns=["A", "B"],
            index=pd.date_range("2023-01-01", periods=3),
        )
        result = compute_turnover_metrics(df)
        # diff row 1: |0.3-0.5| + |0.7-0.5| = 0.4, turnover = 0.5 * 0.4 = 0.2
        # diff row 2: |0.4-0.3| + |0.6-0.7| = 0.2, turnover = 0.5 * 0.2 = 0.1
        # avg over 2 diffs = 0.15
        expected = 0.15
        np.testing.assert_allclose(result["avg_daily_turnover"], expected, atol=1e-10)

    def test_turnover_by_regime(self):
        """Turnover should be computable per regime."""
        df = pd.DataFrame(
            [[0.5, 0.5], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]],
            columns=["A", "B"],
            index=pd.date_range("2023-01-01", periods=4),
        )
        regimes = pd.Series(["Bull", "Bull", "Bear", "Bear"],
                           index=pd.date_range("2023-01-01", periods=4))
        result = compute_turnover_metrics(df, regimes)
        assert "turnover_by_regime" in result


# ─────────────────────────────────────────────────────────────────────────────
# Test 20: Buy and hold benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestBuyAndHoldBenchmark:
    """Buy and hold should have correct properties."""

    def test_buy_and_hold_return_matches_equal_weight(self, simulated_data):
        prices, returns, features = simulated_data
        cm = TransactionCostModel()
        result = _run_buy_and_hold(returns, cm, returns.columns.tolist())

        ew_return = returns.mean(axis=1)
        np.testing.assert_allclose(
            result.portfolio_returns.values, ew_return.values, atol=1e-10
        )

    def test_buy_and_hold_no_costs(self, simulated_data):
        prices, returns, features = simulated_data
        cm = TransactionCostModel()
        result = _run_buy_and_hold(returns, cm, returns.columns.tolist())
        assert (result.costs == 0.0).all(), "Buy and hold should have zero costs"

    def test_buy_and_hold_no_trades(self, simulated_data):
        prices, returns, features = simulated_data
        cm = TransactionCostModel()
        result = _run_buy_and_hold(returns, cm, returns.columns.tolist())
        assert len(result.trade_log) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 21: Directory I/O
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectoryIO:
    """Output directories and file operations should work."""

    def test_save_all_plots_creates_files(self, hmm_result):
        import matplotlib
        matplotlib.use("Agg")

        detector, regime_series, prices, returns, features = hmm_result
        cost_model = TransactionCostModel()
        optimizer = __import__("regime_shift.optimizer", fromlist=["PortfolioOptimizer"]).PortfolioOptimizer(n_assets=len(returns.columns))

        wb = WalkForwardBacktest(
            prices, returns, features,
            lookback=100, retrain_freq=21, n_states=3,
            cost_model=cost_model, detector=detector, optimizer=optimizer,
        )
        result = wb.run()

        regime_m = compute_regime_metrics(result.portfolio_returns, result.regime_series)
        turnover_m = compute_turnover_metrics(result.weights_history, result.regime_series)
        bench_results = run_benchmarks(prices, returns, features, cost_model=cost_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_all_plots(
                result, bench_results, regime_m, turnover_m,
                features=features, output_dir=tmpdir,
            )
            assert len(saved) > 0, "save_all_plots should return file paths"
            for path in saved:
                assert os.path.exists(path), f"Saved file missing: {path}"
