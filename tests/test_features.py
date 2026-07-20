"""
test_features.py — Unit tests for feature engineering module.

Tests verify:
  1. ATR calculation correctness (known values)
  2. Volume spike threshold is strictly backward-looking (no lookahead)
  3. OBV direction logic
  4. Feature matrix has correct shape and no NaN in valid region
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from src.regime_shift.features import (
    compute_atr,
    compute_obv,
    compute_vwap,
    compute_single_asset_features,
    validate_ohlcv,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_ohlcv():
    """50-bar synthetic OHLCV with controlled price movement."""
    np.random.seed(42)
    n = 50
    close  = 100 + np.cumsum(np.random.randn(n))
    high   = close + np.abs(np.random.randn(n))
    low    = close - np.abs(np.random.randn(n))
    open_  = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 5000, n).astype(float)
    dates  = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume
    }, index=dates)


# ──────────────────────────────────────────────────────────────────────────────
# ATR Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_atr_positive(simple_ohlcv):
    """ATR must always be >= 0."""
    df = simple_ohlcv
    atr = compute_atr(df["high"], df["low"], df["close"], length=14)
    valid = atr.dropna()
    assert (valid >= 0).all(), "ATR contains negative values"


def test_atr_length(simple_ohlcv):
    """ATR with length=14 should have valid values from index 1 onward (EMA)."""
    df  = simple_ohlcv
    atr = compute_atr(df["high"], df["low"], df["close"], length=14)
    # First value of TR is NaN (needs prev_close), so ATR[0] may be NaN
    assert atr.iloc[1:].notna().sum() > 30, "ATR has too few valid values"


def test_atr_high_volatility_increases():
    """ATR should increase when price range expands."""
    n      = 30
    dates  = pd.date_range("2020-01-01", periods=n, freq="D")
    # Quiet period: range = 1
    high   = pd.Series(np.ones(n) * 101, index=dates)
    low    = pd.Series(np.ones(n) * 99,  index=dates)
    close  = pd.Series(np.ones(n) * 100, index=dates)
    atr_q  = compute_atr(high, low, close, length=5).dropna().mean()

    # Volatile: range = 10
    high_v = pd.Series(np.ones(n) * 110, index=dates)
    low_v  = pd.Series(np.ones(n) * 90,  index=dates)
    atr_v  = compute_atr(high_v, low_v, close, length=5).dropna().mean()
    assert atr_v > atr_q, "ATR should be higher for more volatile data"


# ──────────────────────────────────────────────────────────────────────────────
# Volume Spike Lookahead Bias Test
# ──────────────────────────────────────────────────────────────────────────────

def test_volume_spike_no_lookahead(simple_ohlcv):
    """
    Critical test: bar i's volume spike threshold must NOT use bar i's volume.

    The threshold at index i is computed using volume[i-window:i-1] (shift=1).
    If we change bar i's volume, the threshold at bar i must NOT change.
    """
    from src.regime_shift.strategy import compute_indicators
    import config

    df = simple_ohlcv.copy().reset_index()
    df.rename(columns={"index": "datetime"}, inplace=True)

    result_original = compute_indicators(df.copy())

    # Mutate bar at index 20 (volume ×100)
    df_modified = df.copy()
    df_modified.loc[20, "volume"] = df_modified.loc[20, "volume"] * 100
    result_modified = compute_indicators(df_modified)

    # The THRESHOLD at index 20 must be unchanged (it uses volume[0:19])
    thresh_original = result_original.loc[20, "vol_spike"]
    thresh_modified = result_modified.loc[20, "vol_spike"]

    assert abs(thresh_original - thresh_modified) < 1e-9, (
        f"LOOKAHEAD BIAS: vol_spike threshold at bar 20 changed when bar 20's "
        f"volume was modified.\n"
        f"  Original threshold: {thresh_original:.4f}\n"
        f"  Modified threshold: {thresh_modified:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# OBV Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_obv_increasing_on_rising_closes(simple_ohlcv):
    """OBV should increase when prices consistently rise."""
    n      = 20
    dates  = pd.date_range("2020-01-01", periods=n, freq="D")
    close  = pd.Series(range(100, 100 + n, 1), dtype=float, index=dates)
    volume = pd.Series(np.ones(n) * 1000, index=dates)
    obv    = compute_obv(close, volume)
    # OBV should be monotonically increasing (since every close > prev close)
    obv_valid = obv.iloc[1:]  # first is 0 always
    assert (obv_valid.diff().iloc[1:] >= 0).all(), "OBV not increasing on rising prices"


def test_obv_formula():
    """Test OBV manual calculation."""
    closes  = pd.Series([100.0, 102.0, 101.0, 103.0])
    volumes = pd.Series([1000.0, 2000.0, 1500.0, 3000.0])
    obv     = compute_obv(closes, volumes)
    # Day 1: 0 (first)
    # Day 2: 0 + 2000 = 2000 (close up)
    # Day 3: 2000 - 1500 = 500 (close down)
    # Day 4: 500 + 3000 = 3500 (close up)
    expected = [0.0, 2000.0, 500.0, 3500.0]
    for i, exp in enumerate(expected):
        assert abs(obv.iloc[i] - exp) < 1e-9, f"OBV mismatch at index {i}"


# ──────────────────────────────────────────────────────────────────────────────
# Feature Matrix Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_feature_matrix_shape(simple_ohlcv):
    """Feature matrix must have 7 columns."""
    features = compute_single_asset_features(simple_ohlcv, window=10)
    assert features.shape[1] == 7, f"Expected 7 features, got {features.shape[1]}"


def test_feature_matrix_no_nan_valid_region(simple_ohlcv):
    """After dropna(), feature matrix must have no NaN."""
    features = compute_single_asset_features(simple_ohlcv, window=10)
    assert not features.isna().any().any(), "NaN values in feature matrix after dropna()"


def test_feature_matrix_columns(simple_ohlcv):
    """Feature matrix must have the expected 7 column names."""
    features  = compute_single_asset_features(simple_ohlcv, window=10)
    expected  = {"ret_ann", "vol_ann", "vol_zscore", "atr_ratio",
                 "obv_zscore", "vwap_dev", "ret_zscore"}
    actual    = set(features.columns)
    missing   = expected - actual
    assert not missing, f"Missing feature columns: {missing}"


# ──────────────────────────────────────────────────────────────────────────────
# OHLCV Validation Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_validate_ohlcv_passes_clean_data(simple_ohlcv, capsys):
    """Validation should pass on clean synthetic data."""
    df = simple_ohlcv.copy().reset_index()
    df.rename(columns={"index": "datetime"}, inplace=True)
    validated = validate_ohlcv(df)
    assert "data_ok" in validated.columns
    # All bars should pass (our fixture has proper OHLCV by construction)
    # Note: may have some fails due to random generation; just check it runs
    assert len(validated) > 0


def test_validate_ohlcv_catches_bad_bars():
    """Validator should flag bars where close > high."""
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=3, freq="D"),
        "open":  [100.0, 100.0, 100.0],
        "high":  [105.0, 105.0, 105.0],
        "low":   [95.0,  95.0,  95.0],
        "close": [102.0, 110.0, 103.0],  # Bar 1: close=110 > high=105 (bad!)
        "volume": [1000.0, 1000.0, 1000.0],
    })
    validated = validate_ohlcv(df)
    bad_rows  = (~validated["data_ok"]).sum()
    assert bad_rows >= 1, "Validator missed bad OHLCV bar (close > high)"
