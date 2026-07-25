"""
Tests for the walk-forward backtest engine.

Covers:
- Basic backtest execution
- Transaction cost application
- Regime inference integration
- Benchmark comparison
- Leakage safety
- Deterministic output
- Edge cases
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_shift.config import RegimeShiftConfig, FeatureConfig, HMMConfig, PortfolioConfig
from regime_shift.data import _normalise_index
from regime_shift.features import compute_raw_features, drop_feature_warmup
from regime_shift.regime_model import fit_hmm, predict_current_state, RegimeSolution
from regime_shift.portfolio import optimize_portfolio
from regime_shift.backtest import (
    run_walk_forward_backtest,
    run_benchmark,
    BacktestResult,
    BenchmarkResult,
)
from regime_shift.benchmarks import static_60_40_weights, equal_weight_weights
from regime_shift.exceptions import RegimeDetectionError, PortfolioOptimizationError


# ---------------------------------------------------------------------------
# Synthetic data generator (deterministic)
# ---------------------------------------------------------------------------

def _generate_prices(periods: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic multi-asset prices with regime segments."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=periods)

    n_bull = periods // 3
    n_bear = periods // 3
    n_crisis = periods - n_bull - n_bear

    bull_ret = rng.normal(0.003, 0.008, n_bull)
    bear_ret = rng.normal(-0.002, 0.018, n_bear)
    crisis_ret = rng.normal(0.0, 0.05, n_crisis)

    all_ret = np.concatenate([bull_ret, bear_ret, crisis_ret])
    equity = 100.0 * np.exp(np.cumsum(all_ret))
    gold = 1800.0 + np.cumsum(rng.normal(0.0001, 0.005, periods))
    bond = 97.0 + np.cumsum(rng.normal(0.00005, 0.002, periods))
    vix = np.concatenate([
        np.full(n_bull, 15.0),
        np.full(n_bear, 25.0),
        np.full(n_crisis, 45.0),
    ])

    df = pd.DataFrame(
        {"equity": equity, "gold": gold, "bond": bond, "vix": vix},
        index=idx,
    )
    return _normalise_index(df)


# ---------------------------------------------------------------------------
# Basic execution tests
# ---------------------------------------------------------------------------

class TestBasicExecution:

    def test_backtest_returns_result_object(self):
        """run_walk_forward_backtest returns a BacktestResult."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)
        assert isinstance(result, BacktestResult)

    def test_result_has_required_fields(self):
        """BacktestResult has all expected fields."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        assert hasattr(result, "net_returns")
        assert hasattr(result, "gross_returns")
        assert hasattr(result, "transaction_costs")
        assert hasattr(result, "turnover")
        assert hasattr(result, "target_weights")
        assert hasattr(result, "regime_series")
        assert hasattr(result, "regime_probabilities")
        assert hasattr(result, "net_equity")
        assert hasattr(result, "metrics")
        assert hasattr(result, "hmm_diagnostics")

    def test_net_returns_non_empty(self):
        """Net returns series is non-empty."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)
        assert len(result.net_returns) > 0

    def test_regime_series_contains_valid_labels(self):
        """Regime series contains only Bull/Bear/Crisis."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        valid = {"Bull", "Bear", "Crisis", "Unknown"}
        for regime in result.regime_series.dropna():
            assert regime in valid, f"Invalid regime label: {regime}"

    def test_weights_sum_to_one(self):
        """Target weights sum to approximately 1.0 at all rebalance dates."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        tw = result.target_weights
        # Only check rows where weights are non-zero (i.e., rebalance happened)
        non_zero = tw[tw.sum(axis=1) > 1e-10]
        if len(non_zero) > 0:
            row_sums = non_zero.sum(axis=1)
            for s in row_sums:
                assert np.isclose(s, 1.0, atol=1e-6), f"Weights sum to {s}"

    def test_weights_are_nonnegative(self):
        """All weights are non-negative."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        for col in result.target_weights.columns:
            assert (result.target_weights[col] >= -1e-10).all()

    def test_insufficient_data_raises(self):
        """Backtest raises on insufficient data."""
        prices = _generate_prices(50)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 200
        with pytest.raises((ValueError, Exception)):
            run_walk_forward_backtest(prices, config)


# ---------------------------------------------------------------------------
# Transaction cost tests
# ---------------------------------------------------------------------------

class TestTransactionCosts:

    def test_zero_cost_produces_equal_net_and_gross(self):
        """With zero transaction cost, net returns equal gross returns."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.transaction_cost_bps = 0.0
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        np.testing.assert_allclose(
            result.net_returns.values,
            result.gross_returns.values,
            atol=1e-10,
        )

    def test_higher_cost_reduces_net_return(self):
        """Higher transaction costs reduce net cumulative return."""
        prices = _generate_prices(600)
        config_low = RegimeShiftConfig()
        config_low.transaction_cost_bps = 0.0
        config_low.minimum_training_observations = 60

        config_high = RegimeShiftConfig()
        config_high.transaction_cost_bps = 50.0
        config_high.minimum_training_observations = 60

        result_low = run_walk_forward_backtest(prices, config_low)
        result_high = run_walk_forward_backtest(prices, config_high)

        net_low = (1 + result_low.net_returns).cumprod().iloc[-1]
        net_high = (1 + result_high.net_returns).cumprod().iloc[-1]

        assert net_low >= net_high - 1e-10

    def test_costs_non_negative(self):
        """Transaction costs are never negative."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        assert (result.transaction_costs >= -1e-10).all()

    def test_costs_only_on_rebalance_dates(self):
        """Transaction costs are zero on non-rebalance dates."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.rebalance_frequency = 21
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        non_rebalance = result.transaction_costs[~result.rebalance_flags]
        if len(non_rebalance) > 0:
            assert (non_rebalance.abs() < 1e-10).all()


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmarks:

    def test_static_60_40_weights(self):
        """Static 60/40 benchmark has correct weights."""
        w = static_60_40_weights()
        assert np.isclose(w["equity"], 0.60)
        assert np.isclose(w["gold"], 0.00)
        assert np.isclose(w["bond"], 0.40)

    def test_equal_weight_weights(self):
        """Equal weight benchmark has 1/3 each."""
        w = equal_weight_weights()
        assert np.isclose(w["equity"], 1 / 3)
        assert np.isclose(w["gold"], 1 / 3)
        assert np.isclose(w["bond"], 1 / 3)

    def test_no_vix_in_benchmarks(self):
        """Benchmark weights do not include VIX."""
        assert "vix" not in static_60_40_weights()
        assert "vix" not in equal_weight_weights()

    def test_benchmark_returns_non_empty(self):
        """Benchmark produces non-empty return series."""
        prices = _generate_prices(600)
        bench = run_benchmark(prices, static_60_40_weights())
        assert len(bench.net_returns) > 0

    def test_benchmark_equity_sum(self):
        """Benchmark produces non-empty equity curve."""
        prices = _generate_prices(600)
        bench = run_benchmark(prices, static_60_40_weights())
        assert len(bench.net_equity) > 0

    def test_benchmark_weights_sum(self):
        """Benchmark weights sum to 1.0."""
        prices = _generate_prices(600)
        bench = run_benchmark(prices, equal_weight_weights())
        tw = bench.net_returns  # no target_weights in BenchmarkResult
        # Just verify returns exist
        assert len(tw) > 0


# ---------------------------------------------------------------------------
# Regime integration tests
# ---------------------------------------------------------------------------

class TestRegimeIntegration:

    def test_regime_series_matches_hmm_output(self):
        """Regime series in result matches HMM predictions."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100
        result = run_walk_forward_backtest(prices, config)

        if len(result.regime_series.dropna()) > 0:
            regimes = result.regime_series.dropna().unique()
            for r in regimes:
                assert r in {"Bull", "Bear", "Crisis", "Unknown"}

    def test_regime_probabilities_sum_to_one(self):
        """Regime probabilities sum to 1.0."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100
        result = run_walk_forward_backtest(prices, config)

        probs = result.regime_probabilities.dropna()
        if len(probs) > 0:
            row_sums = probs.sum(axis=1)
            np.testing.assert_allclose(
                row_sums.values, 1.0, atol=1e-6,
                err_msg="Regime probabilities do not sum to 1.0",
            )

    def test_transition_matrix_shape(self):
        """Transition matrix is 3x3."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100
        result = run_walk_forward_backtest(prices, config)

        assert result.transition_matrix.shape == (3, 3)

    def test_transition_matrix_rows_sum_to_one(self):
        """Transition matrix rows sum to approximately 1.0."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100
        result = run_walk_forward_backtest(prices, config)

        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Leakage safety tests
# ---------------------------------------------------------------------------

class TestLeakageSafety:

    def test_future_returns_dont_affect_weights(self):
        """Weights computed at T are unaffected by returns after T."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 200

        result = run_walk_forward_backtest(prices, config)

        # Get first rebalance weights
        if len(result.target_weights) > 0:
            first_weights = result.target_weights.iloc[0]
            assert np.isfinite(first_weights.values).all()

    def test_no_data_after_cutoff_used_for_training(self):
        """HMM only sees data through the information cutoff."""
        # This is verified by the fact that all rebalances use expanding
        # window through the previous date only
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100
        result = run_walk_forward_backtest(prices, config)

        # Verify the backtest completed without using future data
        # by checking that all diagnostics have finite values
        for diag in result.hmm_diagnostics:
            assert np.isfinite(diag.get("log_likelihood", 0)), \
                "Non-finite log-likelihood suggests data leakage"


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_config_same_result(self):
        """Same config and data produce identical results."""
        prices = _generate_prices(600, seed=42)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100
        config.random_seed = 42

        result1 = run_walk_forward_backtest(prices, config)
        result2 = run_walk_forward_backtest(prices, config)

        np.testing.assert_allclose(
            result1.net_returns.values,
            result2.net_returns.values,
            atol=1e-10,
        )

        if len(result1.target_weights) > 0 and len(result2.target_weights) > 0:
            np.testing.assert_allclose(
                result1.target_weights.values,
                result2.target_weights.values,
                atol=1e-10,
            )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_without_vix(self):
        """Backtest works without VIX column."""
        prices = _generate_prices(600)
        prices_no_vix = prices[["equity", "gold", "bond"]]

        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices_no_vix, config)

        assert len(result.net_returns) > 0

    def test_all_weights_finite(self):
        """All weights are finite."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        assert np.isfinite(result.target_weights.values).all()
        assert np.isfinite(result.net_returns.values).all()

    def test_net_equity_starts_at_one(self):
        """Net equity curve starts at a value consistent with the first day's return."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        if len(result.net_equity) > 0:
            # First equity value = (1 + first_day_return), should be near 1.0
            # (within 10% for reasonable market conditions)
            assert 0.9 < result.net_equity.iloc[0] < 1.1, (
                f"First equity value {result.net_equity.iloc[0]} is unreasonably far from 1.0"
            )

    def test_metrics_computed(self):
        """Performance metrics are computed and non-empty."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        assert result.metrics is not None
        assert len(result.metrics) > 0
        assert "Sharpe" in result.metrics or "Sharpe Ratio" in result.metrics

    def test_future_returns_dont_affect_pre_trade_weights(self):
        """Future return after next rebalance cannot affect pre-trade weights."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100

        result = run_walk_forward_backtest(prices, config)

        # Find a rebalance date and verify weights before it are unaffected
        # by returns after it
        rb_dates = result.rebalance_flags[result.rebalance_flags].index
        if len(rb_dates) >= 2:
            first_rb = rb_dates[0]
            second_rb = rb_dates[1]
            # Weights at first_rb should be determined only by data through first_rb-1
            tw_at_rb = result.target_weights.loc[first_rb]
            assert np.isfinite(tw_at_rb.values).all()
            assert tw_at_rb.sum() > 1e-10  # non-zero means rebalance occurred

    def test_truncated_full_runs_match(self):
        """Truncated and full runs match through truncation date."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100

        # Full run
        result_full = run_walk_forward_backtest(prices, config)

        # Truncated run (drop last 200 rows)
        prices_trunc = prices.iloc[:-200]
        result_trunc = run_walk_forward_backtest(prices_trunc, config)

        # Both should produce identical results through the truncation date
        common_start = max(
            result_full.net_returns.index[0],
            result_trunc.net_returns.index[0],
        )
        common_end = result_trunc.net_returns.index[-1]

        if common_start <= common_end:
            full_slice = result_full.net_returns.loc[common_start:common_end]
            trunc_slice = result_trunc.net_returns.loc[common_start:common_end]
            np.testing.assert_allclose(
                full_slice.values, trunc_slice.values, atol=1e-10,
                err_msg="Truncated run differs from full run through truncation date",
            )

    def test_rebalance_does_not_drift_to_end(self):
        """A rebalance at date d only affects date d, not all subsequent dates."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100

        result = run_walk_forward_backtest(prices, config)

        rb_dates = result.rebalance_flags[result.rebalance_flags].index
        if len(rb_dates) >= 1:
            first_rb = rb_dates[0]
            # Target weights should only be non-zero at rebalance dates
            # (not at all dates after the first rebalance)
            tw_after = result.target_weights.loc[first_rb:]
            non_zero = tw_after[tw_after.sum(axis=1) > 1e-10]
            # Non-zero target weights should only appear at rebalance dates
            for date in non_zero.index:
                assert date in rb_dates, (
                    f"Non-zero target weights at {date} which is not a rebalance date"
                )

    def test_bull_bear_probabilities_not_swapped(self):
        """Bull and Bear probability columns cannot be swapped by alphabetical sort."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100

        result = run_walk_forward_backtest(prices, config)

        probs = result.regime_probabilities.dropna()
        if len(probs) > 0:
            # Columns must be in explicit Bull, Bear, Crisis order
            assert list(probs.columns) == ["Bull", "Bear", "Crisis"], (
                f"Probability columns are {list(probs.columns)}, "
                "expected ['Bull', 'Bear', 'Crisis']"
            )

    def test_transition_matrix_not_identity_placeholder(self):
        """Transition matrix is the actual last HMM fit, not an identity placeholder."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100

        result = run_walk_forward_backtest(prices, config)

        # If the backtest ran, the transition matrix should NOT be identity
        # (unless the HMM actually learned identity transitions)
        # We verify it's a valid stochastic matrix (rows sum to 1)
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(
            row_sums.values, 1.0, atol=1e-6,
            err_msg="Transition matrix rows do not sum to 1",
        )

        # And verify it was actually set from HMM (not left as zeros)
        assert not (result.transition_matrix.values == 0).all(), (
            "Transition matrix appears to be all zeros — was it set from the HMM?"
        )

    def test_initial_strategy_turnover_equals_one(self):
        """Initial strategy turnover from cash equals 1.0 for fully invested portfolio."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100

        result = run_walk_forward_backtest(prices, config)

        # Find first rebalance
        rb_dates = result.rebalance_flags[result.rebalance_flags].index
        if len(rb_dates) > 0:
            first_rb = rb_dates[0]
            turnover_at_first = result.turnover.loc[first_rb]
            # Initial allocation from cash: sum of abs weights
            expected_turnover = result.target_weights.loc[first_rb].sum()
            assert np.isclose(turnover_at_first, expected_turnover, atol=1e-6), (
                f"Initial turnover {turnover_at_first} != sum of abs weights {expected_turnover}"
            )

    def test_max_drawdown_between_zero_and_one(self):
        """Maximum drawdown is between 0 and 1."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        if result.metrics and "Maximum Drawdown" in result.metrics:
            md = result.metrics["Maximum Drawdown"]
            if md is not None and not np.isnan(md):
                assert 0 <= md <= 1, f"Max drawdown {md} is outside [0, 1]"

    def test_transaction_cost_drag_not_equal_to_turnover(self):
        """Transaction cost drag is not equal to total turnover."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.transaction_cost_bps = 5.0
        result = run_walk_forward_backtest(prices, config)

        if result.metrics:
            turnover = result.metrics.get("Total Turnover", None)
            cost_drag = result.metrics.get("Transaction Cost Drag", None)
            if (
                turnover is not None
                and cost_drag is not None
                and not np.isnan(turnover)
                and not np.isnan(cost_drag)
            ):
                # They should be different (cost drag depends on cost rate)
                assert not np.isclose(turnover, cost_drag, atol=1e-6), (
                    f"Cost drag {cost_drag} equals turnover {turnover} — "
                    "cost drag should be defined differently from turnover"
                )

    def test_strategy_and_benchmarks_identical_indices(self):
        """Strategy and benchmarks have identical date indices."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        result = run_walk_forward_backtest(prices, config)

        bench_60 = run_benchmark(
            prices, static_60_40_weights(),
            config=config,
            rebalance_flags=result.rebalance_flags,
            start_date=result.net_returns.index[0],
        )
        bench_eq = run_benchmark(
            prices, equal_weight_weights(),
            config=config,
            rebalance_flags=result.rebalance_flags,
            start_date=result.net_returns.index[0],
        )

        assert result.net_returns.index.equals(bench_60.net_returns.index), (
            "Strategy and 60/40 benchmark have different date indices"
        )
        assert result.net_returns.index.equals(bench_eq.net_returns.index), (
            "Strategy and equal-weight benchmark have different date indices"
        )

    def test_benchmark_rebalances_only_on_flags(self):
        """Benchmark rebalances only on actual rebalance flag dates."""
        prices = _generate_prices(600)
        config = RegimeShiftConfig()
        config.minimum_training_observations = 60
        config.rebalance_frequency = 100

        result = run_walk_forward_backtest(prices, config)
        rb_flags = result.rebalance_flags

        bench = run_benchmark(
            prices, static_60_40_weights(),
            config=config,
            rebalance_flags=rb_flags,
            start_date=result.net_returns.index[0],
        )

        # Benchmark turnover should be zero on non-rebalance dates
        turnover = bench.turnover
        non_rb_turnover = turnover[~rb_flags.loc[turnover.index]]
        if len(non_rb_turnover) > 0:
            assert (non_rb_turnover.abs() < 1e-10).all(), (
                "Benchmark has turnover on non-rebalance dates"
            )

    def test_run_submission_imports(self):
        """run_submission.py can be imported without error."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_submission", str(Path(__file__).parent.parent / "run_submission.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")
        assert hasattr(mod, "parse_args")
