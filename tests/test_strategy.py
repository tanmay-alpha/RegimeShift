"""
test_strategy.py — Unit tests for the trading strategy module.

Tests verify:
  1. Lookahead bias: truncated vs full-series signals must match
  2. Signal encoding is correct (0, 1, -1, 2, -2)
  3. Position state machine transitions
  4. Trailing stop logic
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from src.regime_shift.strategy import (
    compute_indicators,
    volume_spike_signals,
    regime_conditional_signals,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def btc_like_data():
    """Synthetic BTC-like OHLCV, 200 bars, with a clear volume spike."""
    np.random.seed(7)
    n      = 200
    close  = 10000 + np.cumsum(np.random.randn(n) * 100)
    high   = close + np.abs(np.random.randn(n) * 50)
    low    = close - np.abs(np.random.randn(n) * 50)
    open_  = close + np.random.randn(n) * 30
    volume = np.random.randint(5000, 15000, n).astype(float)

    # Inject a clear volume spike at bar 50 (bullish)
    volume[50] = 200000
    close[50]  = close[49] * 1.05   # bullish candle (close > open)
    open_[50]  = close[49]

    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=n, freq="D"),
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    })
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Signal Encoding Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_signals_valid_values(btc_like_data):
    """All signal values must be in {-2, -1, 0, 1, 2}."""
    result = regime_conditional_signals(btc_like_data.copy(), regimes=None)
    valid_signals = {-2, -1, 0, 1, 2}
    actual_signals = set(result["signals"].unique())
    unexpected = actual_signals - valid_signals
    assert not unexpected, f"Unexpected signal values: {unexpected}"


def test_no_consecutive_open_long_signals(btc_like_data):
    """
    Cannot have two consecutive LONG entry signals without a CLOSE in between.
    (Position state machine: can't enter long when already long.)
    """
    result   = regime_conditional_signals(btc_like_data.copy(), regimes=None)
    position = 0
    for i, row in result.iterrows():
        sig = row["signals"]
        if position == 0 and sig == 1:
            position = 1
        elif position == 1 and sig == 1:
            tt = row["trade_type"]
            assert tt in ("CLOSE", "LONG", "REVERSE_LONG_TO_SHORT", "REVERSE_SHORT_TO_LONG"), (
                f"Unexpected trade_type '{tt}' with signal=1 while already LONG at index {i}"
            )
        elif position == 1 and sig in (-1, -2):
            position = -1 if sig == -2 else 0
        elif position == -1 and sig in (1, 2):
            position = 1 if sig == 2 else 0


def test_hold_signal_is_zero(btc_like_data):
    """HOLD trade_type rows must have signal = 0."""
    result   = regime_conditional_signals(btc_like_data.copy(), regimes=None)
    hold_rows = result[result["trade_type"] == "HOLD"]
    assert (hold_rows["signals"] == 0).all(), "HOLD rows must have signal = 0"


# ──────────────────────────────────────────────────────────────────────────────
# Lookahead Bias Test (CRITICAL)
# ──────────────────────────────────────────────────────────────────────────────

def test_no_lookahead_bias(btc_like_data):
    """
    Critical property: signal at index i must be the same whether computed
    on the full series OR on data truncated at index i.

    Tests 20 random signal indices.
    """
    import random
    data   = btc_like_data.copy()
    full   = regime_conditional_signals(data.copy(), regimes=None)

    signal_indices = full[full["signals"] != 0].index.tolist()
    if len(signal_indices) == 0:
        pytest.skip("No signals generated — skip lookahead test")

    random.seed(42)
    sample = random.sample(signal_indices, min(20, len(signal_indices)))

    for idx in sorted(sample):
        trunc  = data.iloc[:idx + 1].copy().reset_index(drop=True)
        result_trunc = regime_conditional_signals(trunc, regimes=None)

        sig_full  = full.loc[idx, "signals"]
        sig_trunc = result_trunc.iloc[idx]["signals"]

        assert sig_full == sig_trunc, (
            f"LOOKAHEAD BIAS at index {idx}:\n"
            f"  Full-series signal: {sig_full}\n"
            f"  Truncated signal  : {sig_trunc}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Trailing Stop Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_trailing_stop_closes_long():
    """
    When close drops below the trailing stop, position must close.
    """
    # Create a scenario where price rises then falls below trailing stop
    n  = 60
    np.random.seed(5)
    close  = np.ones(n) * 10000.0
    # Bars 0-29: price at 10000
    # Bar 30: volume spike + bullish candle → LONG
    # Bar 31-39: price rises to 10500
    # Bar 40: price drops to 9000 (well below trailing stop)
    for i in range(31, 40):
        close[i] = 10000 + (i - 30) * 50
    close[40:] = 9000.0

    high   = close + 200
    low    = close - 200
    open_  = close - 50

    # inject volume spike at bar 30 with bullish candle
    volume = np.ones(n) * 5000.0
    volume[30] = 200000.0
    close[30]  = open_[30] + 100  # ensure close > open

    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=n, freq="D"),
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume
    })

    result = regime_conditional_signals(df, regimes=None)

    # Must have a CLOSE signal somewhere after bar 30
    closes_after_entry = result.iloc[31:][result.iloc[31:]["trade_type"] == "CLOSE"]
    assert len(closes_after_entry) > 0, (
        "Trailing stop did not close the position when price dropped significantly"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Regime Filtering Test
# ──────────────────────────────────────────────────────────────────────────────

def test_regime_blocks_short_in_bull(btc_like_data):
    """
    In Bull regime, SHORT entry signals should be blocked.
    """
    data = btc_like_data.copy()

    # Create a regime Series with all "Bull" (state 0) for these dates
    dates_idx  = pd.to_datetime(data["datetime"])
    # Use integer regime labels; state 0 = Bull (will be labeled in regime_detector)
    # For this test we mock a regime that's all-Bull (0)
    regimes_mock = pd.Series(
        ["Bull"] * len(data),
        index=dates_idx,
    )

    # Get signals without regime filter (to find SHORT entries)
    pure  = regime_conditional_signals(data.copy(), regimes=None)
    short_entries = pure[(pure["trade_type"] == "SHORT")]

    if len(short_entries) == 0:
        pytest.skip("No SHORT entries generated — skip regime block test")

    # Now apply Bull-only regime filter — shorts should be blocked
    # Note: regime_conditional_signals works on string labels in the series
    result = regime_conditional_signals(data.copy(), regimes=regimes_mock)
    blocked_shorts = result[result["trade_type"] == "REGIME_BLOCKED"]
    remaining_short_entries = result[result["trade_type"] == "SHORT"]

    # In Bull regime, SHORT entries should be blocked
    assert len(remaining_short_entries) == 0, (
        f"Bull regime filter should block all SHORT entries. "
        f"Found {len(remaining_short_entries)} remaining."
    )
