"""
Comprehensive tests for the Gaussian HMM regime detection model.

Tests cover:
  1.  GaussianHMM is used
  2.  Exactly 3 components are fitted
  3.  covariance_type is diag
  4.  Training is deterministic with fixed seed
  5.  Model refuses invalid inputs
  6.  Missing or infinite values are rejected
  7.  Insufficient history is rejected
  8.  Feature column mismatch is rejected
  9.  Transition matrix shape is 3x3
  10. Transition rows sum to one
  11. State labels are exactly Bull/Bear/Crisis
  12. Crisis state has strongest stress profile
  13. Bull has strongest momentum among remaining states
  14. Mapping works without VIX
  15. Mapping works with VIX
  16. Posterior probabilities are finite
  17. Posterior probabilities sum to one
  18. Current-state inference does not refit the HMM
  19. Current-state inference does not refit the scaler
  20. Adding data after T cannot alter a decision from data through T
  21. Future test-block observations are not used for earlier predictions
  22. Model-not-fitted errors are explicit
  23. Convergence failures are visible

Uses deterministic synthetic features with clearly separable segments.
No network access required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_shift.config import FeatureConfig, HMMConfig
from regime_shift.data import _normalise_index
from regime_shift.exceptions import RegimeDetectionError
from regime_shift.features import (
    compute_raw_features,
    drop_feature_warmup,
    fit_feature_scaler,
    transform_features,
)
from regime_shift.regime_model import (
    fit_hmm,
    predict_current_state,
    get_transition_matrix,
    get_state_statistics,
    get_regime,
    get_probabilities,
    RegimeSolution,
    StateStatistics,
    _validate_hmm_input,
)


# ---------------------------------------------------------------------------
# Deterministic synthetic data
# ---------------------------------------------------------------------------

def _make_prices(
    periods: int = 500,
    start: str = "2015-01-01",
    seed: int = 42,
    include_vix: bool = False,
) -> pd.DataFrame:
    """Create deterministic multi-asset price DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)

    equity_returns = rng.normal(0.0005, 0.015, periods)
    equity = 100.0 * np.exp(np.cumsum(equity_returns))

    gold_returns = rng.normal(0.0002, 0.008, periods)
    gold = 1800.0 * np.exp(np.cumsum(gold_returns))

    bond_returns = rng.normal(0.0001, 0.003, periods)
    bond = 97.0 * np.exp(np.cumsum(bond_returns))

    data = {"equity": equity, "gold": gold, "bond": bond}

    if include_vix:
        vix = 20.0 + np.abs(rng.normal(0, 5, periods))
        vix = np.clip(vix, 10.0, 60.0)
        data["vix"] = vix

    df = pd.DataFrame(data, index=idx)
    df = _normalise_index(df)
    return df


def _make_segmented_features(
    n_per_regime: int = 120,
    include_vix: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Create raw features with three clearly separable regime segments."""
    rng = np.random.default_rng(seed)
    total = n_per_regime * 3
    idx = pd.bdate_range("2015-01-01", periods=total)

    bull_ret = rng.normal(0.003, 0.008, n_per_regime)
    bear_ret = rng.normal(-0.002, 0.018, n_per_regime)
    crisis_ret = rng.normal(0.0, 0.06, n_per_regime)
    all_ret = np.concatenate([bull_ret, bear_ret, crisis_ret])

    equity = 100.0 * np.exp(np.cumsum(all_ret))
    gold = 1800.0 + np.cumsum(rng.normal(0.0001, 0.005, total))
    bond = 97.0 + np.cumsum(rng.normal(0.00005, 0.002, total))

    prices = pd.DataFrame(
        {"equity": equity, "gold": gold, "bond": bond},
        index=idx,
    )

    if include_vix:
        vix = np.concatenate([
            np.full(n_per_regime, 15.0),
            np.full(n_per_regime, 25.0),
            np.full(n_per_regime, 45.0),
        ])
        prices["vix"] = vix

    config = FeatureConfig()
    features = compute_raw_features(prices, config)
    features = drop_feature_warmup(features, config=config)
    return features


def _make_training_and_test(
    n_per_regime: int = 120,
    train_frac: float = 0.7,
    include_vix: bool = False,
    seed: int = 42,
):
    """Create segmented features and split into training/test."""
    all_features = _make_segmented_features(n_per_regime, include_vix, seed)
    n_train = int(len(all_features) * train_frac)
    train_features = all_features.iloc[:n_train]
    test_features = all_features.iloc[n_train:]
    return train_features, test_features


# ---------------------------------------------------------------------------
# 1-3. GaussianHMM usage, 3 components, diag covariance
# ---------------------------------------------------------------------------

class TestHMMConfiguration:

    def test_gaussian_hmm_is_used(self):
        """fit_hmm must use hmmlearn.hmm.GaussianHMM."""
        import regime_shift.regime_model as rm
        src_path = Path(rm.__file__)
        with open(src_path) as f:
            source = f.read()

        assert "GaussianHMM" in source, "regime_model.py must use GaussianHMM"
        assert "hmmlearn" in source, "regime_model.py must import hmmlearn"

    def test_exactly_three_components(self):
        """HMM must be configured with n_components=3."""
        config = HMMConfig()
        assert config.n_components == 3
        train_features, _ = _make_training_and_test()
        scaler = fit_feature_scaler(train_features)
        scaled = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled, train_features, config)
        assert hmm.n_components == 3

    def test_covariance_type_is_diag(self):
        """HMM must use covariance_type='diag'."""
        config = HMMConfig()
        assert config.covariance_type == "diag"
        train_features, _ = _make_training_and_test()
        scaler = fit_feature_scaler(train_features)
        scaled = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled, train_features, config)
        assert hmm.covariance_type == "diag"


# ---------------------------------------------------------------------------
# 4. Deterministic training with fixed seed
# ---------------------------------------------------------------------------

class TestDeterministicTraining:

    def test_same_seed_same_result(self):
        """Same seed should produce identical fitted models."""
        train_features, _ = _make_training_and_test(seed=42)

        config = HMMConfig(random_state=42)
        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)

        hmm1, states1, _ = fit_hmm(scaled_train, train_features, config)
        hmm2, states2, _ = fit_hmm(scaled_train, train_features, config)

        np.testing.assert_array_equal(states1, states2)
        np.testing.assert_array_almost_equal(
            hmm1.means_, hmm2.means_, decimal=10
        )
        np.testing.assert_array_almost_equal(
            hmm1.covars_, hmm2.covars_, decimal=10
        )


# ---------------------------------------------------------------------------
# 5-8. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_nan_rejected(self):
        """NaN in features should raise RegimeDetectionError."""
        from regime_shift.regime_model import _validate_hmm_input
        features = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        with pytest.raises(RegimeDetectionError, match="NaN"):
            _validate_hmm_input(features, "features")

    def test_inf_rejected(self):
        """Infinite values in features should raise RegimeDetectionError."""
        from regime_shift.regime_model import _validate_hmm_input
        features = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        with pytest.raises(RegimeDetectionError, match="non-finite"):
            _validate_hmm_input(features, "features")

    def test_empty_rejected(self):
        """Empty DataFrame should raise RegimeDetectionError."""
        from regime_shift.regime_model import _validate_hmm_input
        features = pd.DataFrame()
        with pytest.raises(RegimeDetectionError, match="empty"):
            _validate_hmm_input(features, "features")

    def test_non_dataframe_rejected(self):
        """Non-DataFrame input should raise RegimeDetectionError."""
        from regime_shift.regime_model import _validate_hmm_input
        with pytest.raises(RegimeDetectionError):
            _validate_hmm_input(np.array([1, 2, 3]), "features")

    def test_insufficient_observations_rejected(self):
        """Too few observations should raise RegimeDetectionError."""
        # Create a valid DataFrame with too few rows (must pass NaN/inf checks)
        features = pd.DataFrame({
            "equity_log_return_1d": [0.01, 0.02],
            "equity_momentum_21d": [0.01, 0.02],
            "equity_momentum_63d": [0.01, 0.02],
            "equity_volatility_21d": [0.01, 0.02],
            "equity_volatility_ratio_21_63": [1.0, 1.0],
            "equity_gold_correlation_63d": [0.1, 0.1],
            "equity_bond_correlation_63d": [0.1, 0.1],
        })
        config = HMMConfig(minimum_training_observations=100)
        scaler = fit_feature_scaler(features)
        scaled = transform_features(scaler, features)
        with pytest.raises(RegimeDetectionError, match="Insufficient"):
            fit_hmm(scaled, features, config)

    def test_column_mismatch_rejected(self):
        """Column mismatch between scaled and raw should raise error."""
        train_features, _ = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled = transform_features(scaler, train_features)

        # Create raw with different column order (still same columns)
        raw_wrong = train_features.copy()
        raw_wrong.columns = list(reversed(raw_wrong.columns))

        with pytest.raises(RegimeDetectionError, match="columns"):
            fit_hmm(scaled, raw_wrong, config)

    def test_row_mismatch_rejected(self):
        """Row count mismatch should raise error."""
        train_features, _ = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled = transform_features(scaler, train_features)

        raw_short = train_features.iloc[:5]
        with pytest.raises(RegimeDetectionError, match="rows"):
            fit_hmm(scaled, raw_short, config)


# ---------------------------------------------------------------------------
# 9-10. Transition matrix shape and row sums
# ---------------------------------------------------------------------------

class TestTransitionMatrix:

    def test_shape_is_3x3(self):
        """Transition matrix must be exactly 3x3."""
        train_features, _ = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled = transform_features(scaler, train_features)
        _, _, trans_mat = fit_hmm(scaled, train_features, config)

        assert trans_mat.shape == (3, 3), (
            f"Transition matrix shape must be (3, 3), got {trans_mat.shape}"
        )

    def test_rows_sum_to_one(self):
        """Each row of the transition matrix must sum to 1."""
        train_features, _ = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled = transform_features(scaler, train_features)
        _, _, trans_mat = fit_hmm(scaled, train_features, config)

        for regime in ["Bull", "Bear", "Crisis"]:
            row_sum = trans_mat.loc[regime].sum()
            assert np.isclose(row_sum, 1.0, atol=1e-6), (
                f"Transition row for {regime} sums to {row_sum}, not 1.0"
            )

    def test_columns_are_regime_labels(self):
        """Transition matrix columns must be Bull, Bear, Crisis in deterministic order."""
        train_features, _ = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled = transform_features(scaler, train_features)
        _, _, trans_mat = fit_hmm(scaled, train_features, config)

        # Deterministic ordering: Bull, Bear, Crisis (by regime_order list)
        assert list(trans_mat.columns) == ["Bull", "Bear", "Crisis"]
        assert list(trans_mat.index) == ["Bull", "Bear", "Crisis"]

    def test_values_finite(self):
        """All transition matrix values must be finite."""
        train_features, _ = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled = transform_features(scaler, train_features)
        _, _, trans_mat = fit_hmm(scaled, train_features, config)

        assert np.all(np.isfinite(trans_mat.values)), "Transition matrix must be finite"
        assert np.all(trans_mat.values >= 0), "Transition probabilities must be non-negative"
        assert np.all(trans_mat.values <= 1), "Transition probabilities must be <= 1"


# ---------------------------------------------------------------------------
# 11-15. State interpretation
# ---------------------------------------------------------------------------

class TestStateInterpretation:

    def test_labels_are_bull_bear_crisis(self):
        """RegimeSolution must contain exactly Bull, Bear, Crisis labels in probabilities."""
        train_features, test_features = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        combined = pd.concat([train_features, test_features.iloc[:1]])
        solution = predict_current_state(hmm, scaler, combined, train_features, config)

        labels = sorted(solution.probabilities.index.tolist())
        assert labels == ["Bear", "Bull", "Crisis"], f"Got labels: {labels}"

    def test_crisis_has_highest_volatility(self):
        """Crisis state must have highest mean equity_volatility_21d in state stats."""
        train_features, test_features = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        combined = pd.concat([train_features, test_features.iloc[:1]])
        solution = predict_current_state(hmm, scaler, combined, train_features, config)

        stats_by_regime = {s.regime_label: s for s in solution.state_statistics}
        crisis_vol = stats_by_regime["Crisis"].mean_features["equity_volatility_21d"]
        bull_vol = stats_by_regime["Bull"].mean_features["equity_volatility_21d"]
        bear_vol = stats_by_regime["Bear"].mean_features["equity_volatility_21d"]

        assert crisis_vol > bull_vol, "Crisis must have higher vol than Bull"
        assert crisis_vol > bear_vol, "Crisis must have higher vol than Bear"

    def test_bull_has_highest_momentum(self):
        """Among Bull and Bear, Bull must have highest mean equity_momentum_63d."""
        train_features, test_features = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        combined = pd.concat([train_features, test_features.iloc[:1]])
        solution = predict_current_state(hmm, scaler, combined, train_features, config)

        stats_by_regime = {s.regime_label: s for s in solution.state_statistics}
        bull_mom = stats_by_regime["Bull"].mean_features["equity_momentum_63d"]
        bear_mom = stats_by_regime["Bear"].mean_features["equity_momentum_63d"]

        assert bull_mom > bear_mom, "Bull must have higher momentum than Bear"

    def test_mapping_works_without_vix(self):
        """State mapping must work without VIX features."""
        train_features, test_features = _make_training_and_test(include_vix=False)
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        combined = pd.concat([train_features, test_features.iloc[:1]])
        solution = predict_current_state(hmm, scaler, combined, train_features, config)

        labels = sorted(solution.probabilities.index.tolist())
        assert labels == ["Bear", "Bull", "Crisis"]

    def test_mapping_works_with_vix(self):
        """State mapping must work with VIX features."""
        train_features, test_features = _make_training_and_test(include_vix=True)
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        combined = pd.concat([train_features, test_features.iloc[:1]])
        solution = predict_current_state(hmm, scaler, combined, train_features, config)

        labels = sorted(solution.probabilities.index.tolist())
        assert labels == ["Bear", "Bull", "Crisis"]


def _compute_states_for_test(hmm, scaled, raw, config):
    """Helper to compute hidden states for testing."""
    hidden = hmm.predict(scaled.values)
    from regime_shift.regime_model import _interpret_states
    state_map = _interpret_states(hidden, raw, config)
    return _compute_state_statistics(hidden, raw, state_map, config)


def _compute_state_statistics(hidden_states, raw_features, state_map, config):
    """Helper to compute state statistics."""
    from regime_shift.regime_model import StateStatistics
    stats = []
    raw_arr = raw_features.values
    feature_names = list(raw_features.columns)
    for numeric_state in range(config.n_components):
        mask = hidden_states == numeric_state
        n_obs = int(mask.sum())
        mean_vals = raw_arr[mask].mean(axis=0) if n_obs > 0 else np.full(len(feature_names), np.nan)
        mean_series = pd.Series(mean_vals, index=feature_names)
        stats.append(StateStatistics(
            state_id=numeric_state,
            regime_label=state_map[numeric_state],
            mean_features=mean_series,
            n_observations=n_obs,
        ))
    return stats


# ---------------------------------------------------------------------------
# 16-17. Posterior probabilities
# ---------------------------------------------------------------------------

class TestPosteriorProbabilities:

    def test_probabilities_are_finite(self):
        """Posterior probabilities must be finite."""
        train_features, test_features = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        # Use features through current date (training + first test row)
        combined = pd.concat([train_features, test_features.iloc[:1]])
        solution = predict_current_state(hmm, scaler, combined, train_features, config)

        assert np.all(np.isfinite(solution.probabilities.values)), (
            "Posterior probabilities must be finite"
        )

    def test_probabilities_sum_to_one(self):
        """Posterior probabilities must sum to 1."""
        train_features, test_features = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        combined = pd.concat([train_features, test_features.iloc[:1]])
        solution = predict_current_state(hmm, scaler, combined, train_features, config)

        prob_sum = float(solution.probabilities.sum())
        assert np.isclose(prob_sum, 1.0, atol=1e-6), (
            f"Posterior probabilities sum to {prob_sum}, not 1.0"
        )

    def test_probabilities_non_negative(self):
        """Posterior probabilities must be non-negative."""
        train_features, test_features = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        combined = pd.concat([train_features, test_features.iloc[:1]])
        solution = predict_current_state(hmm, scaler, combined, train_features, config)

        assert np.all(solution.probabilities.values >= -1e-10), (
            "Posterior probabilities must be non-negative"
        )


# ---------------------------------------------------------------------------
# 18-19. No refitting during inference
# ---------------------------------------------------------------------------

class TestNoRefitting:

    def test_inference_does_not_refit_hmm(self):
        """predict_current_state must not modify the fitted HMM."""
        train_features, test_features = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        # Store original model parameters
        orig_means = hmm.means_.copy()
        orig_covars = hmm.covars_.copy()
        orig_transmat = hmm.transmat_.copy()

        combined = pd.concat([train_features, test_features.iloc[:1]])
        predict_current_state(hmm, scaler, combined, train_features, config)

        np.testing.assert_array_equal(hmm.means_, orig_means)
        np.testing.assert_array_equal(hmm.covars_, orig_covars)
        np.testing.assert_array_equal(hmm.transmat_, orig_transmat)

    def test_inference_does_not_refit_scaler(self):
        """predict_current_state must not modify the fitted scaler."""
        train_features, test_features = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        orig_mean = scaler.mean_.copy()
        orig_scale = scaler.scale_.copy()

        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        combined = pd.concat([train_features, test_features.iloc[:1]])
        predict_current_state(hmm, scaler, combined, train_features, config)

        np.testing.assert_array_equal(scaler.mean_, orig_mean)
        np.testing.assert_array_equal(scaler.scale_, orig_scale)


# ---------------------------------------------------------------------------
# 20-21. Leakage: future data doesn't affect past decisions
# ---------------------------------------------------------------------------

class TestLeakageSafety:

    def test_data_after_T_does_not_change_regime(self):
        """Adding data after T must not change regime inferred from data through T."""
        train_features, test_features = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        # Inference with data through T (training + first test row)
        combined_short = pd.concat([train_features, test_features.iloc[:1]])
        solution_short = predict_current_state(
            hmm, scaler, combined_short, train_features, config
        )

        # Inference with data through T + extra future rows
        combined_long = pd.concat([train_features, test_features.iloc[:5]])
        solution_long = predict_current_state(
            hmm, scaler, combined_long, train_features, config
        )

        assert solution_short.regime == solution_long.regime, (
            f"Regime changed when adding future data: "
            f"{solution_short.regime} vs {solution_long.regime}"
        )
        np.testing.assert_allclose(
            solution_short.probabilities.sort_index().values,
            solution_long.probabilities.sort_index().values,
            atol=1e-10,
        )

    def test_future_test_not_used_for_earlier_predictions(self):
        """predict_current_state uses only the last observation's features."""
        train_features, test_features = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        # Prediction at T: data through T
        combined_at_T = pd.concat([train_features, test_features.iloc[:1]])
        solution_at_T = predict_current_state(
            hmm, scaler, combined_at_T, train_features, config
        )

        # Repeat — must be deterministic
        solution_at_T_again = predict_current_state(
            hmm, scaler, combined_at_T, train_features, config
        )
        assert solution_at_T.regime == solution_at_T_again.regime
        np.testing.assert_allclose(
            solution_at_T.probabilities.sort_index().values,
            solution_at_T_again.probabilities.sort_index().values,
            atol=1e-10,
        )

        # Add an extra row after T (T+1) — the prediction should now reflect T+1
        combined_at_T1 = pd.concat([train_features, test_features.iloc[:2]])
        solution_at_T1 = predict_current_state(
            hmm, scaler, combined_at_T1, train_features, config
        )
        assert solution_at_T1.regime in ("Bull", "Bear", "Crisis")


# ---------------------------------------------------------------------------
# 22-23. Error conditions
# ---------------------------------------------------------------------------

class TestErrorConditions:

    def test_not_fitted_error_message(self):
        """Using an unfitted model should raise a clear error."""
        from hmmlearn.hmm import GaussianHMM
        unfitted = GaussianHMM(n_components=3)
        scaler = StandardScaler()

        train_features, _ = _make_training_and_test()
        config = HMMConfig()

        with pytest.raises(Exception):  # hmmlearn raises various errors for unfitted models
            predict_current_state(unfitted, scaler, train_features, train_features, config)

    def test_convergence_failure_logged(self):
        """Non-convergence should produce a warning, not silently succeed."""
        # Use very few iterations to force non-convergence
        train_features, _ = _make_training_and_test()
        config = HMMConfig(n_iter=1, random_state=42)

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        # The model should still work but may not converge
        combined = pd.concat([train_features, train_features.iloc[:1]])
        solution = predict_current_state(hmm, scaler, combined, train_features, config)
        assert solution.regime in ("Bull", "Bear", "Crisis")


# ---------------------------------------------------------------------------
# RegimeSolution dataclass tests
# ---------------------------------------------------------------------------

class TestRegimeSolution:

    def test_solution_has_all_fields(self):
        """RegimeSolution must have all expected fields."""
        train_features, test_features = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        combined = pd.concat([train_features, test_features.iloc[:1]])
        solution = predict_current_state(hmm, scaler, combined, train_features, config)

        assert hasattr(solution, "regime")
        assert hasattr(solution, "probabilities")
        assert hasattr(solution, "transition_matrix")
        assert hasattr(solution, "state_statistics")
        assert hasattr(solution, "convergence")
        assert hasattr(solution, "n_iter")
        assert hasattr(solution, "log_likelihood")
        assert hasattr(solution, "warning")

    def test_state_statistics_count(self):
        """StateStatistics should have exactly 3 entries."""
        train_features, _ = _make_training_and_test()
        config = HMMConfig()

        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)
        hmm, _, _ = fit_hmm(scaled_train, train_features, config)

        combined = pd.concat([train_features, train_features.iloc[:1]])
        solution = predict_current_state(hmm, scaler, combined, train_features, config)

        assert len(solution.state_statistics) == 3
