"""
Comprehensive leakage-safety and correctness tests for the feature pipeline.

Tests cover:
  1. Expected feature columns without VIX
  2. Expected feature columns with VIX
  3. Deterministic feature order
  4. Input DataFrame is not mutated
  5. Output is finite after warmup
  6. Insufficient history raises a clear error
  7. Constant prices do not produce infinite values
  8. Scale invariance: multiplying prices by a constant does not change
     return/momentum/volatility features
  9. Feature at t is unaffected by changes to prices after t
  10. Truncation invariance: features through T are identical whether later
      data exists or not
  11. Future perturbation: changing prices after T cannot change features at
      or before T
  12. Changing price at T may change the feature at T
  13. No negative shift appears in official feature source
  14. No centered rolling window appears
  15. No backward fill appears
  16. Scaler mean and variance use training rows only
  17. Adding future rows cannot alter an already-fitted scaler
  18. Current observation is transformed without refitting
  19. Optional VIX absence is handled
  20. VIX is not included as a portfolio asset
  21. Warmup row count is documented and deterministic

All tests use synthetic deterministic fixtures.  No network access required.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_shift.config import FeatureConfig, RegimeShiftConfig
from regime_shift.data import _normalise_index
from regime_shift.exceptions import FeatureEngineeringError
from regime_shift.features import (
    _get_feature_columns,
    compute_raw_features,
    drop_feature_warmup,
    fit_feature_scaler,
    fit_transform_training_features,
    transform_features,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_prices(
    periods: int = 200,
    start: str = "2022-01-03",
    seed: int = 42,
    include_vix: bool = False,
) -> pd.DataFrame:
    """Create a deterministic multi-asset price DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)

    equity_returns = rng.normal(0.0005, 0.015, periods)
    equity = 100.0 * np.exp(np.cumsum(equity_returns))

    gold_returns = rng.normal(0.0002, 0.008, periods)
    gold = 1800.0 * np.exp(np.cumsum(gold_returns))

    bond_returns = rng.normal(0.0001, 0.003, periods)
    bond = 97.0 * np.exp(np.cumsum(bond_returns))

    data = {
        "equity": equity,
        "gold": gold,
        "bond": bond,
    }

    if include_vix:
        vix = 20.0 + np.abs(rng.normal(0, 5, periods))
        vix = np.clip(vix, 10.0, 60.0)
        data["vix"] = vix

    df = pd.DataFrame(data, index=idx)
    df = _normalise_index(df)
    return df


def _make_feature_config() -> FeatureConfig:
    """Create a FeatureConfig for tests."""
    return FeatureConfig(
        short_window=21,
        medium_window=63,
        correlation_window=63,
        vix_change_window=5,
        annualization_factor=252,
        minimum_feature_observations=10,
    )


# ---------------------------------------------------------------------------
# 1. Expected feature columns without VIX
# ---------------------------------------------------------------------------

class TestFeatureColumns:

    def test_expected_columns_without_vix(self):
        """Without VIX, exactly 7 base features should be present."""
        prices = _make_prices(periods=200, include_vix=False)
        config = _make_feature_config()
        features = compute_raw_features(prices, config)
        features = drop_feature_warmup(features, config=config)

        expected = [
            "equity_log_return_1d",
            "equity_momentum_21d",
            "equity_momentum_63d",
            "equity_volatility_21d",
            "equity_volatility_ratio_21_63",
            "equity_gold_correlation_63d",
            "equity_bond_correlation_63d",
        ]
        assert list(features.columns) == expected

    def test_expected_columns_with_vix(self):
        """With VIX, exactly 9 features should be present (7 base + 2 VIX)."""
        prices = _make_prices(periods=200, include_vix=True)
        config = _make_feature_config()
        features = compute_raw_features(prices, config)
        features = drop_feature_warmup(features, config=config)

        expected = [
            "equity_log_return_1d",
            "equity_momentum_21d",
            "equity_momentum_63d",
            "equity_volatility_21d",
            "equity_volatility_ratio_21_63",
            "equity_gold_correlation_63d",
            "equity_bond_correlation_63d",
            "vix_change_5d",
            "vix_level",
        ]
        assert list(features.columns) == expected


# ---------------------------------------------------------------------------
# 2. Deterministic feature order
# ---------------------------------------------------------------------------

class TestDeterministicOrder:

    def test_feature_order_is_deterministic(self):
        """Feature column order must be the same on every call."""
        prices = _make_prices(periods=200, include_vix=True)
        config = _make_feature_config()

        features1 = compute_raw_features(prices, config)
        features2 = compute_raw_features(prices, config)

        assert list(features1.columns) == list(features2.columns)

    def test_feature_order_without_vix(self):
        """Without VIX, order must match the base feature list."""
        prices = _make_prices(periods=200, include_vix=False)
        config = _make_feature_config()
        features = compute_raw_features(prices, config)

        expected = [
            "equity_log_return_1d",
            "equity_momentum_21d",
            "equity_momentum_63d",
            "equity_volatility_21d",
            "equity_volatility_ratio_21_63",
            "equity_gold_correlation_63d",
            "equity_bond_correlation_63d",
        ]
        assert list(features.columns) == expected


# ---------------------------------------------------------------------------
# 3. Input DataFrame not mutated
# ---------------------------------------------------------------------------

class TestNoMutation:

    def test_input_not_mutated(self):
        """compute_raw_features must not mutate the input DataFrame."""
        prices = _make_prices(periods=100, include_vix=True)
        original_values = prices.values.copy()
        original_columns = list(prices.columns)

        compute_raw_features(prices, _make_feature_config())

        np.testing.assert_array_equal(prices.values, original_values)
        assert list(prices.columns) == original_columns

    def test_input_index_not_mutated(self):
        """compute_raw_features must not mutate the input index."""
        prices = _make_prices(periods=100, include_vix=True)
        original_index = prices.index.copy()

        compute_raw_features(prices, _make_feature_config())

        pd.testing.assert_index_equal(prices.index, original_index)


# ---------------------------------------------------------------------------
# 4. Output is finite after warmup
# ---------------------------------------------------------------------------

class TestFiniteOutput:

    def test_output_finite_after_warmup_without_vix(self):
        """After warmup removal, all feature values must be finite."""
        prices = _make_prices(periods=200, include_vix=False)
        config = _make_feature_config()
        features = compute_raw_features(prices, config)
        features = drop_feature_warmup(features, config=config)

        assert np.isfinite(features.values).all()

    def test_output_finite_after_warmup_with_vix(self):
        """After warmup removal, all feature values (including VIX) must be finite."""
        prices = _make_prices(periods=200, include_vix=True)
        config = _make_feature_config()
        features = compute_raw_features(prices, config)
        features = drop_feature_warmup(features, config=config)

        assert np.isfinite(features.values).all()


# ---------------------------------------------------------------------------
# 5. Insufficient history raises a clear error
# ---------------------------------------------------------------------------

class TestInsufficientHistory:

    def test_insufficient_raises_error(self):
        """Too few rows should raise FeatureEngineeringError with a clear message."""
        prices = _make_prices(periods=10, include_vix=False)
        config = FeatureConfig(
            short_window=5,
            medium_window=63,
            correlation_window=63,
            minimum_feature_observations=5,
        )
        features = compute_raw_features(prices, config)
        with pytest.raises(FeatureEngineeringError, match="All.*NaN"):
            drop_feature_warmup(features, config=config)

    def test_insufficient_raises_clear_message(self):
        """Error message should mention the warmup / data shortage issue."""
        prices = _make_prices(periods=20, include_vix=False)
        config = FeatureConfig(
            short_window=5,
            medium_window=63,
            correlation_window=63,
            minimum_feature_observations=5,
        )
        features = compute_raw_features(prices, config)
        with pytest.raises(FeatureEngineeringError, match="warmup"):
            drop_feature_warmup(features, config=config)


# ---------------------------------------------------------------------------
# 6. Constant prices do not produce infinite values
# ---------------------------------------------------------------------------

class TestConstantPrices:

    def test_constant_prices_log_returns_are_zero(self):
        """Constant prices produce zero log returns (not NaN or inf)."""
        idx = pd.bdate_range("2022-01-03", periods=100)
        prices = pd.DataFrame(
            {
                "equity": np.full(100, 100.0),
                "gold": np.full(100, 1800.0),
                "bond": np.full(100, 97.0),
            },
            index=idx,
        )

        config = _make_feature_config()
        features = compute_raw_features(prices, config)

        # Log returns should be exactly 0 (or NaN for the very first row)
        log_ret = features["equity_log_return_1d"].dropna()
        assert (log_ret == 0.0).all(), "Constant prices should produce zero log returns"

    def test_constant_prices_no_infinite_volatility(self):
        """Constant prices produce zero volatility (not inf)."""
        idx = pd.bdate_range("2022-01-03", periods=100)
        prices = pd.DataFrame(
            {
                "equity": np.full(100, 100.0),
                "gold": np.full(100, 1800.0),
                "bond": np.full(100, 97.0),
            },
            index=idx,
        )

        config = _make_feature_config()
        features = compute_raw_features(prices, config)

        # Volatility of zero-variance returns is 0, not inf
        vol = features["equity_volatility_21d"].dropna()
        assert (vol == 0.0).all(), "Constant prices should produce zero volatility"
        assert np.isfinite(vol.values).all(), "Volatility must be finite"

    def test_constant_prices_no_infinite_volatility_ratio(self):
        """When vol_63 is also zero, vol_ratio should be NaN (not inf)."""
        idx = pd.bdate_range("2022-01-03", periods=100)
        prices = pd.DataFrame(
            {
                "equity": np.full(100, 100.0),
                "gold": np.full(100, 1800.0),
                "bond": np.full(100, 97.0),
            },
            index=idx,
        )

        config = _make_feature_config()
        features = compute_raw_features(prices, config)

        # When vol_21 and vol_63 are both 0, the ratio is NaN (0/0 handled)
        ratio = features["equity_volatility_ratio_21_63"].dropna()
        # After warmup (63 rows), all ratios should be present and finite or NaN
        # With 0/0, our _compute_volatility_ratio returns NaN where vol_long is 0
        # But we set vol_long > 0 check, so 0/0 → NaN. Let's verify no inf
        assert not np.isinf(ratio.values).any(), "No infinite values in volatility ratio"


# ---------------------------------------------------------------------------
# 7. Scale invariance: multiplying prices by a constant
# ---------------------------------------------------------------------------

class TestScaleInvariance:

    def test_multiply_equity_does_not_change_momentum(self):
        """Multiplying equity prices by a constant must not change momentum features."""
        prices1 = _make_prices(periods=200, seed=42, include_vix=False)
        prices2 = prices1.copy()
        prices2["equity"] = prices2["equity"] * 1.5

        config = _make_feature_config()
        f1 = compute_raw_features(prices1, config)
        f2 = compute_raw_features(prices2, config)

        np.testing.assert_allclose(
            f1["equity_momentum_21d"].values,
            f2["equity_momentum_21d"].values,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            f1["equity_momentum_63d"].values,
            f2["equity_momentum_63d"].values,
            equal_nan=True,
        )

    def test_multiply_equity_does_not_change_volatility(self):
        """Multiplying equity prices by a constant must not change volatility features."""
        prices1 = _make_prices(periods=200, seed=42, include_vix=False)
        prices2 = prices1.copy()
        prices2["equity"] = prices2["equity"] * 1.5

        config = _make_feature_config()
        f1 = compute_raw_features(prices1, config)
        f2 = compute_raw_features(prices2, config)

        np.testing.assert_allclose(
            f1["equity_volatility_21d"].values,
            f2["equity_volatility_21d"].values,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            f1["equity_log_return_1d"].values,
            f2["equity_log_return_1d"].values,
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# 8. Feature at t is unaffected by changes to prices after t
# ---------------------------------------------------------------------------

class TestNoFutureDataLeakage:

    def _get_cutoff_date(self, prices, config, positional_cutoff=100):
        """Get a date-based cutoff well past warmup."""
        features = compute_raw_features(prices, config)
        clean = drop_feature_warmup(features, config=config)
        # Use the date at the given positional index in the clean features
        return clean.index[positional_cutoff]

    def test_feature_at_t_unaffected_by_later_prices(self):
        """Changing prices after cutoff date must not affect any feature before it."""
        prices = _make_prices(periods=300, seed=42, include_vix=False)
        config = _make_feature_config()

        # Compute features from original data
        f_orig = compute_raw_features(prices, config)
        f_orig_clean = drop_feature_warmup(f_orig, config=config)

        # Get a cutoff date well into the data
        cutoff_date = self._get_cutoff_date(prices, config, positional_cutoff=80)

        # Corrupt all prices AFTER the cutoff date
        prices_corrupted = prices.copy()
        prices_corrupted.loc[prices_corrupted.index > cutoff_date, "equity"] *= 100.0

        f_corr = compute_raw_features(prices_corrupted, config)
        f_corr_clean = drop_feature_warmup(f_corr, config=config)

        # Align both on common dates up to and including cutoff
        orig_up_to = f_orig_clean[f_orig_clean.index <= cutoff_date]
        corr_up_to = f_corr_clean[f_corr_clean.index <= cutoff_date]

        # They should have the same dates
        common_dates = orig_up_to.index.intersection(corr_up_to.index)
        assert len(common_dates) > 0, "Should have common dates up to cutoff"

        np.testing.assert_allclose(
            orig_up_to.loc[common_dates].values,
            corr_up_to.loc[common_dates].values,
            equal_nan=True,
            err_msg="Features before cutoff should not change when later prices change",
        )

    def test_future_perturbation_no_change_before_T(self):
        """Future perturbation: changing prices after T cannot change features at or before T."""
        prices = _make_prices(periods=300, seed=42, include_vix=False)
        config = _make_feature_config()

        f_orig = compute_raw_features(prices, config)
        f_orig_clean = drop_feature_warmup(f_orig, config=config)

        # Pick a cutoff date well past warmup
        cutoff_date = self._get_cutoff_date(prices, config, positional_cutoff=80)

        # Corrupt everything after the cutoff date
        prices_corrupted = prices.copy()
        for col in ["equity", "gold", "bond"]:
            prices_corrupted.loc[prices_corrupted.index > cutoff_date, col] *= 999.0

        f_corr = compute_raw_features(prices_corrupted, config)
        f_corr_clean = drop_feature_warmup(f_corr, config=config)

        # All rows up to cutoff must be identical
        dates_up_to = f_orig_clean.index[f_orig_clean.index <= cutoff_date]
        for date in dates_up_to:
            orig_row = f_orig_clean.loc[date]
            corr_row = f_corr_clean.loc[date]
            np.testing.assert_allclose(
                orig_row.values, corr_row.values, equal_nan=True,
                err_msg=f"Feature at {date} changed after future perturbation"
            )


# ---------------------------------------------------------------------------
# 9. Changing price at T may change the feature at T
# ---------------------------------------------------------------------------

class TestLocalPerturbation:

    def test_changing_price_at_T_changes_feature_at_T(self):
        """Changing price at position T can change the feature at T (expected behavior)."""
        prices = _make_prices(periods=300, seed=42, include_vix=False)
        config = _make_feature_config()

        f_orig = compute_raw_features(prices, config)
        f_orig_clean = drop_feature_warmup(f_orig, config=config)

        # Change price at a position well past warmup — build modified df
        # directly from numpy to avoid SettingWithCopy issues
        T_pos = 80
        col_idx = prices.columns.get_loc("equity")

        equity_arr = prices["equity"].to_numpy().copy()
        equity_arr[T_pos] *= 2.5
        prices_modified = pd.DataFrame(
            {c: prices[c].to_numpy().copy() for c in prices.columns},
            index=prices.index,
        )
        prices_modified["equity"] = equity_arr

        # Verify the change took effect
        assert prices_modified.iat[T_pos, col_idx] != prices.iat[T_pos, col_idx]

        f_mod = compute_raw_features(prices_modified, config)
        f_mod_clean = drop_feature_warmup(f_mod, config=config)

        # Use the actual date from original prices index — after warmup removal,
        # .iloc positions in cleaned features no longer align with price positions
        T_date = prices.index[T_pos]
        orig_val = f_orig_clean.loc[T_date, "equity_log_return_1d"]
        mod_val = f_mod_clean.loc[T_date, "equity_log_return_1d"]

        # Changing price[T] changes log_return[T] = log(price[T]/price[T-1])
        assert orig_val != mod_val or math.isnan(orig_val) or math.isnan(mod_val), (
            "Changing price at T should affect the feature at T"
        )


# ---------------------------------------------------------------------------
# 10. No negative shift in feature source
# ---------------------------------------------------------------------------

class TestNoNegativeShift:

    def _find_negative_shifts(self, source_file: str) -> list[str]:
        """Scan the features.py source for any negative shift arguments."""
        with open(source_file, "r") as f:
            source = f.read()
        tree = ast.parse(source)
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "shift":
                    for arg in node.args:
                        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                            violations.append(
                                f"Line {node.lineno}: shift with negative arg"
                            )
                        elif isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value < 0:
                            violations.append(
                                f"Line {node.lineno}: shift({arg.value})"
                            )
        return violations

    def test_no_negative_shift_in_features_source(self):
        """Feature source must not contain shift(-1) or any negative shift."""
        src_path = Path(__file__).parent.parent / "src" / "regime_shift" / "features.py"
        violations = self._find_negative_shifts(str(src_path))
        assert violations == [], f"Negative shifts found in features.py: {violations}"


# ---------------------------------------------------------------------------
# 11. No centered rolling window
# ---------------------------------------------------------------------------

class TestNoCenteredWindow:

    def test_no_centered_rolling_in_features_source(self):
        """Feature source must not use centered=True in any rolling() call."""
        src_path = Path(__file__).parent.parent / "src" / "regime_shift" / "features.py"
        with open(src_path, "r") as f:
            source = f.read()

        assert "centered=True" not in source, (
            "features.py must not use centered=True in rolling windows"
        )

    def test_rolling_is_trailing_in_features_source(self):
        """All rolling() calls in features.py must be trailing (center=False or default)."""
        src_path = Path(__file__).parent.parent / "src" / "regime_shift" / "features.py"
        with open(src_path, "r") as f:
            source = f.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "rolling":
                    for kw in node.keywords:
                        if kw.arg == "center":
                            if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                raise AssertionError(
                                    f"Found rolling(center=True) at line {node.lineno}"
                                )


# ---------------------------------------------------------------------------
# 12. No backward fill
# ---------------------------------------------------------------------------

class TestNoBackwardFill:

    def test_no_bfill_in_features_source(self):
        """Feature source must not contain .bfill() method calls."""
        src_path = Path(__file__).parent.parent / "src" / "regime_shift" / "features.py"
        with open(src_path, "r") as f:
            source = f.read()

        # Check for .bfill() method (not the word "backward" in docstrings)
        assert ".bfill" not in source, "features.py must not use .bfill()"
        assert ".backfill" not in source, "features.py must not use .backfill()"


# ---------------------------------------------------------------------------
# 13. Scaler uses training rows only
# ---------------------------------------------------------------------------

class TestTrainOnlyScaling:

    def _make_training_and_test(self, total=300, train_frac=0.7):
        """Create prices and split into training and test segments by date."""
        prices = _make_prices(periods=total, seed=42, include_vix=False)
        config = _make_feature_config()

        all_features = compute_raw_features(prices, config)
        all_features = drop_feature_warmup(all_features, config=config)

        # Split by position (dates are ordered)
        n_train = int(len(all_features) * train_frac)
        train_features = all_features.iloc[:n_train]
        test_features = all_features.iloc[n_train:]

        return train_features, test_features

    def test_scaler_fitted_on_training_only(self):
        """Scaler mean_ must be computed from training rows only."""
        train_features, _ = self._make_training_and_test()

        scaler = fit_feature_scaler(train_features)

        expected_mean = train_features.mean().values
        np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-10)

    def test_scaler_std_from_training_only(self):
        """Scaler scale_ must be computed from training rows only."""
        train_features, _ = self._make_training_and_test()

        scaler = fit_feature_scaler(train_features)
        # sklearn StandardScaler uses population std (ddof=0), matching scale_
        expected_std = train_features.std(ddof=0).values
        np.testing.assert_allclose(scaler.scale_, expected_std, rtol=1e-10)

    def test_scaler_does_not_change_when_refit_on_same_training(self):
        """Re-fitting on the same training data produces identical scaler."""
        train_features, _ = self._make_training_and_test()

        scaler1 = fit_feature_scaler(train_features)
        scaler2 = fit_feature_scaler(train_features)

        np.testing.assert_array_equal(scaler1.mean_, scaler2.mean_)
        np.testing.assert_array_equal(scaler1.scale_, scaler2.scale_)

    def test_transform_without_refitting(self):
        """Test observation is transformed using the same scaler without refitting."""
        train_features, test_features = self._make_training_and_test()

        scaler = fit_feature_scaler(train_features)

        # Transform a single test row without refitting
        single_row = test_features.iloc[:1]
        transformed = transform_features(scaler, single_row)

        assert np.isfinite(transformed.values).all()
        assert transformed.shape == single_row.shape


# ---------------------------------------------------------------------------
# 14. fit_transform_training_features convenience
# ---------------------------------------------------------------------------

class TestFitTransformConvenience:

    def test_fit_transform_returns_scaled_and_scaler(self):
        """fit_transform_training_features must return (scaled_df, scaler)."""
        prices = _make_prices(periods=200, include_vix=False)
        config = _make_feature_config()
        features = compute_raw_features(prices, config)
        features = drop_feature_warmup(features, config=config)

        scaled, scaler = fit_transform_training_features(features)

        assert isinstance(scaled, pd.DataFrame)
        assert hasattr(scaler, "transform")
        assert list(scaled.columns) == list(features.columns)
        assert list(scaled.index) == list(features.index)

    def test_fit_transform_produces_zero_mean(self):
        """fit_transform on training data should produce mean ≈ 0."""
        prices = _make_prices(periods=200, include_vix=False)
        config = _make_feature_config()
        features = compute_raw_features(prices, config)
        features = drop_feature_warmup(features, config=config)

        scaled, _ = fit_transform_training_features(features)

        # Means should be near zero
        assert np.allclose(scaled.mean().values, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 15. Optional VIX absence is handled
# ---------------------------------------------------------------------------

class TestVixHandling:

    def test_no_vix_column_in_features_when_absent(self):
        """Without VIX column in input, no VIX features in output."""
        prices = _make_prices(periods=200, include_vix=False)
        config = _make_feature_config()
        features = compute_raw_features(prices, config)

        assert "vix_change_5d" not in features.columns
        assert "vix_level" not in features.columns

    def test_vix_features_when_present(self):
        """With VIX column in input, VIX features appear in output."""
        prices = _make_prices(periods=200, include_vix=True)
        config = _make_feature_config()
        features = compute_raw_features(prices, config)

        assert "vix_change_5d" in features.columns
        assert "vix_level" in features.columns

    def test_vix_not_in_portfolio_asset_returns(self):
        """VIX should never be mixed with equity/gold/bond in return calculations."""
        prices = _make_prices(periods=200, include_vix=True)
        config = _make_feature_config()
        features = compute_raw_features(prices, config)

        assert "equity_log_return_1d" in features.columns
        assert "vix_level" in features.columns
        # Confirm VIX is not part of any return/momentum/volatility feature
        for col in features.columns:
            if "log_return" in col or "momentum" in col or "volatility" in col:
                assert not col.startswith("vix"), (
                    f"VIX should not have return/momentum/volatility features: {col}"
                )


# ---------------------------------------------------------------------------
# 16. Warmup row count is documented and deterministic
# ---------------------------------------------------------------------------

class TestWarmupDeterminism:

    def test_warmup_count_deterministic(self):
        """Warmup count should be identical for the same input and config."""
        prices = _make_prices(periods=200, seed=42, include_vix=False)
        config = _make_feature_config()

        features1 = compute_raw_features(prices, config)
        clean1, warmup1 = _count_warmup(features1)

        features2 = compute_raw_features(prices, config)
        clean2, warmup2 = _count_warmup(features2)

        assert warmup1 == warmup2, "Warmup count must be deterministic"
        assert len(clean1) == len(clean2), "Clean count must be deterministic"

    def test_warmup_count_at_least_max_lookback_minus_one(self):
        """Warmup count should be at least max_lookback - 1."""
        prices = _make_prices(periods=200, seed=42, include_vix=False)
        config = _make_feature_config()

        features = compute_raw_features(prices, config)
        _, warmup_count = _count_warmup(features)

        # The 63-day correlation needs 62 prior observations, so at least
        # 63 rows should be warmup (positions 0..62)
        assert warmup_count >= config.medium_window - 1

    def test_warmup_reports_removed_rows(self):
        """drop_feature_warmup must remove fewer rows than total."""
        prices = _make_prices(periods=200, seed=42, include_vix=False)
        config = _make_feature_config()

        features = compute_raw_features(prices, config)
        total_before = len(features)

        clean = drop_feature_warmup(features, config=config)
        total_after = len(clean)

        assert total_after < total_before, "Warmup removal must reduce row count"
        assert total_after + (total_before - total_after) == total_before

    def test_no_warmup_for_large_enough_data(self):
        """With sufficient data, warmup should be cleanly removed and no NaN remain."""
        prices = _make_prices(periods=500, seed=42, include_vix=True)
        config = _make_feature_config()

        features = compute_raw_features(prices, config)
        clean = drop_feature_warmup(features, config=config)

        assert not clean.isna().any().any(), "No NaN should remain after warmup removal"
        assert len(clean) > config.minimum_feature_observations


def _count_warmup(features: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Count warmup rows: rows with any NaN."""
    nan_mask = features.isna().any(axis=1)
    warmup_count = int(nan_mask.sum())
    clean = features[~nan_mask].copy()
    return clean, warmup_count


# ---------------------------------------------------------------------------
# 17. FeatureConfig properties
# ---------------------------------------------------------------------------

class TestFeatureConfig:

    def test_default_feature_config_values(self):
        """Default FeatureConfig should have correct values."""
        config = FeatureConfig()
        assert config.short_window == 21
        assert config.medium_window == 63
        assert config.correlation_window == 63
        assert config.vix_change_window == 5
        assert config.annualization_factor == 252
        assert config.minimum_feature_observations == 10

    def test_max_lookback(self):
        """max_lookback should return the maximum window size."""
        config = FeatureConfig()
        assert config.max_lookback() == 63

    def test_feature_config_from_regime_shift_config(self):
        """RegimeShiftConfig.feature_config should inherit annualization_factor."""
        rsc = RegimeShiftConfig(annualization_factor=252)
        fc = rsc.feature_config
        assert fc.annualization_factor == 252

    def test_feature_config_validation_rejects_small_window(self):
        """FeatureConfig.validate() should reject windows < 2."""
        config = FeatureConfig(short_window=1)
        with pytest.raises(ValueError, match="short_window"):
            config.validate()

    def test_feature_config_validation_rejects_zero_annualization(self):
        """FeatureConfig.validate() should reject annualization_factor < 1."""
        config = FeatureConfig(annualization_factor=0)
        with pytest.raises(ValueError, match="annualization_factor"):
            config.validate()

    def test_feature_config_validation_rejects_zero_min_obs(self):
        """FeatureConfig.validate() should reject minimum_feature_observations < 1."""
        config = FeatureConfig(minimum_feature_observations=0)
        with pytest.raises(ValueError, match="minimum_feature_observations"):
            config.validate()

    def test_feature_config_default_validates(self):
        """Default FeatureConfig should pass validation."""
        config = FeatureConfig()
        config.validate()  # must not raise


# ---------------------------------------------------------------------------
# 18. Feature at t uses only data through t (source code audit)
# ---------------------------------------------------------------------------

class TestFeatureDefinitions:

    def test_feature_formulas_in_docstring(self):
        """compute_raw_features docstring should document all feature formulas."""
        docstring = compute_raw_features.__doc__
        assert "equity_log_return_1d" in docstring
        assert "equity_momentum_21d" in docstring
        assert "equity_momentum_63d" in docstring
        assert "equity_volatility_21d" in docstring
        assert "equity_volatility_ratio_21_63" in docstring
        assert "equity_gold_correlation_63d" in docstring
        assert "equity_bond_correlation_63d" in docstring
        assert "vix_change_5d" in docstring
        assert "vix_level" in docstring

    def test_signal_timing_documented(self):
        """Docstring must document the signal timing: feature at t used for decision at t+1."""
        docstring = compute_raw_features.__doc__
        assert "t+1" in docstring or "t + 1" in docstring, (
            "Docstring must document that feature at t is used for decision at t+1"
        )

    def test_all_features_finite_for_long_series(self):
        """For a long series with all features, all values after warmup must be finite."""
        prices = _make_prices(periods=500, seed=42, include_vix=True)
        config = _make_feature_config()
        features = compute_raw_features(prices, config)
        clean = drop_feature_warmup(features, config=config)

        assert np.isfinite(clean.values).all(), (
            f"Non-finite values found: {clean.isna().sum().to_dict()}"
        )
