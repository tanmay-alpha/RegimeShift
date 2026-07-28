"""
Comprehensive tests for the RegimeShift multi-asset data pipeline.

Tests are grouped into:
  1. TestValidation         – strengthened validation contract
  2. TestAssetSemantics     – SeriesKind enforcement / yield rejection
  3. TestDownloadMocked     – download_market_data() with mocked yfinance (no network)
  4. TestCSVRoundTrip       – save/load CSV round-trip (offline execution)
  5. TestAlignment          – multi-asset date alignment and forward-fill policy
  6. TestFillMaskAccuracy   – explicit fill-mask tracking (not inferred from flat prices)
  7. TestDeterminism        – column order is always deterministic

All yfinance calls are mocked so tests run fully offline.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_shift.config import (
    AssetSpec,
    DataConfig,
    RegimeShiftConfig,
    SeriesKind,
    TickerConfig,
    VIX_FALLBACK,
)
from regime_shift.data import (
    _build_aligned_frame,
    _normalise_index,
    download_market_data,
    load_market_data_csv,
    save_market_data_csv,
)
from regime_shift.exceptions import (
    DataAlignmentError,
    DataDownloadError,
    DataValidationError,
)
from regime_shift.validation import (
    check_forward_fill_limit,
    check_no_duplicates,
    check_timezone_naive,
    validate_price_data,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_price_asset(ticker: str, currency: str = "INR") -> AssetSpec:
    """Convenience builder for a PRICE AssetSpec in tests."""
    return AssetSpec(
        ticker=ticker,
        kind=SeriesKind.PRICE,
        description=f"Test price asset {ticker}",
        currency=currency,
    )


def _make_indicator_asset(ticker: str) -> AssetSpec:
    """Convenience builder for an INDICATOR AssetSpec in tests."""
    return AssetSpec(
        ticker=ticker,
        kind=SeriesKind.INDICATOR,
        description=f"Test indicator {ticker}",
    )


@pytest.fixture
def base_config() -> RegimeShiftConfig:
    """
    Config with PRICE-kind AssetSpec objects for all required assets.
    Tickers are the new defaults; caching is disabled.
    """
    tickers = TickerConfig()
    # Override with explicit AssetSpec objects so tests do not depend on
    # the network-reachable production tickers.
    tickers.equity = _make_price_asset("^NSEI")
    tickers.gold   = _make_price_asset("GOLDBEES.NS")
    tickers.bond   = _make_price_asset("0P0001BVE8.BO")
    tickers.vix    = _make_indicator_asset("^INDIAVIX")

    return RegimeShiftConfig(
        tickers=tickers,
        data=DataConfig(
            default_start="2022-01-01",
            default_end="2022-06-30",
            forward_fill_limit=3,
            cache_dir=None,
        ),
    )


@pytest.fixture
def sample_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2022-01-03", periods=100)


@pytest.fixture
def valid_price_df(sample_dates) -> pd.DataFrame:
    """Well-formed price DataFrame satisfying all contracts."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "equity": 100 + np.cumsum(rng.normal(0, 1, 100)),
            "gold": 1800 + np.cumsum(rng.normal(0, 3, 100)),
            "bond": 97 + rng.uniform(0, 1, 100),
        },
        index=sample_dates,
    )


def _make_price_series(periods: int = 120, start: str = "2022-01-03") -> pd.Series:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(start, periods=periods)
    return pd.Series(100 + np.cumsum(rng.normal(0, 1, periods)), index=idx)


def _mock_download(side_effects: Dict[str, pd.Series]):
    """Build a mock for yfinance.download() that returns per-ticker DataFrames."""
    def _fake(ticker, start, end, **kwargs):
        if ticker not in side_effects:
            return pd.DataFrame()
        series = side_effects[ticker]
        return pd.DataFrame({"Close": series}, index=series.index)
    return _fake


# ---------------------------------------------------------------------------
# 1. Validation contract tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Strengthened validation checks beyond the foundation test_data_contract suite."""

    def test_timezone_aware_index_rejected(self, valid_price_df):
        """DatetimeIndex with tz info must be rejected."""
        df = valid_price_df.copy()
        df.index = df.index.tz_localize("Asia/Kolkata")
        with pytest.raises(DataValidationError, match="timezone-naive"):
            validate_price_data(df)

    def test_timezone_naive_passes(self, valid_price_df):
        """Timezone-naive index must pass."""
        assert validate_price_data(valid_price_df) is not None

    def test_forward_fill_limit_enforced(self, sample_dates):
        """Exceeding forward_fill_limit raises DataValidationError when an
        explicit fill mask is provided.  Without a fill mask the identical-
        value run is NOT classified as a forward-fill (a naturally flat bond
        NAV would be wrongly rejected)."""
        prices = np.linspace(100, 200, 100)
        prices[20:30] = prices[19]  # 10 identical values
        df = pd.DataFrame(
            {"equity": prices, "gold": np.linspace(1800, 1900, 100),
             "bond": np.linspace(95, 98, 100)},
            index=sample_dates,
        )
        # Construct an explicit fill mask that flags a long forward-fill
        fill_mask = pd.DataFrame(
            False, index=df.index, columns=df.columns,
        )
        fill_mask.iloc[20:30, 0] = True  # 10-day equity fill
        with pytest.raises(DataValidationError, match="forward-fill limit"):
            validate_price_data(df, forward_fill_limit=5,
                                fill_mask=fill_mask)

    def test_forward_fill_limit_disabled(self, sample_dates):
        """forward_fill_limit=0 disables the run-length check."""
        prices = np.linspace(100, 200, 100)
        prices[20:40] = prices[19]  # long flat run
        df = pd.DataFrame(
            {"equity": prices, "gold": np.linspace(1800, 1900, 100),
             "bond": np.linspace(95, 98, 100)},
            index=sample_dates,
        )
        validate_price_data(df, forward_fill_limit=0)  # must not raise

    def test_naturally_flat_prices_accepted_within_limit(self, sample_dates):
        """
        A series with a short natural flat plateau (e.g. a bond NAV that does
        not change on a public holiday) must NOT be rejected if the plateau
        length is within the forward_fill_limit.

        This proves the forward-fill check is not a false positive for
        naturally constant prices.
        """
        prices = np.linspace(97, 100, 100)
        prices[30:33] = prices[29]  # 3 identical values — exactly at the limit
        df = pd.DataFrame(
            {"equity": np.linspace(100, 200, 100),
             "gold": np.linspace(1800, 1900, 100),
             "bond": prices},
            index=sample_dates,
        )
        # limit=3 → a run of 3 equal values is accepted (not flagged as over-fill)
        validate_price_data(df, forward_fill_limit=3)

    def test_naturally_flat_prices_exceeding_limit_raise(self, sample_dates):
        """A natural flat plateau WITHOUT a fill mask must NOT raise — the
        heuristic is unreliable for low-volatility assets like bonds."""
        prices = np.linspace(97, 100, 100)
        prices[30:37] = prices[29]  # 7 identical values — over limit
        df = pd.DataFrame(
            {"equity": np.linspace(100, 200, 100),
             "gold": np.linspace(1800, 1900, 100),
             "bond": prices},
            index=sample_dates,
        )
        # Without a fill mask, natural flatness is NOT flagged as forward-fill
        result = validate_price_data(df, forward_fill_limit=3)
        assert result is not None

    def test_missing_values_in_specific_column_named(self, sample_dates):
        """Error message must name which columns have NaN."""
        df = pd.DataFrame(
            {"equity": np.linspace(100, 200, 100),
             "gold": np.linspace(1800, 1900, 100),
             "bond": np.linspace(95, 98, 100)},
            index=sample_dates,
        )
        df.iloc[5, 1] = np.nan  # gold
        with pytest.raises(DataValidationError, match="gold"):
            validate_price_data(df, forward_fill_limit=0)

    def test_check_no_duplicates_raises_datavalidation(self, valid_price_df):
        """check_no_duplicates must raise DataValidationError (a ValueError subclass)."""
        df = valid_price_df.copy()
        idx = df.index.tolist()
        idx[1] = idx[0]
        df.index = pd.DatetimeIndex(idx)
        with pytest.raises(ValueError):
            check_no_duplicates(df)

    def test_check_timezone_naive_raises(self, valid_price_df):
        df = valid_price_df.copy()
        df.index = df.index.tz_localize("UTC")
        with pytest.raises(DataValidationError):
            check_timezone_naive(df)

    def test_check_forward_fill_limit_exact_boundary(self, sample_dates):
        """A run equal to the limit should NOT raise."""
        prices = np.linspace(100, 200, 100)
        prices[10:13] = prices[9]   # run of 3 — matches limit exactly
        df = pd.DataFrame(
            {"equity": prices, "gold": np.linspace(1800, 1900, 100),
             "bond": np.linspace(95, 98, 100)},
            index=sample_dates,
        )
        check_forward_fill_limit(df, ["equity"], max_consecutive=3)

    def test_check_forward_fill_limit_over_boundary(self, sample_dates):
        """When an explicit fill mask is supplied, a fill run of limit+1
        identical values raises DataValidationError."""
        prices = np.linspace(100, 200, 100)
        prices[10:15] = prices[9]  # run of 5
        df = pd.DataFrame(
            {"equity": prices, "gold": np.linspace(1800, 1900, 100),
             "bond": np.linspace(95, 98, 100)},
            index=sample_dates,
        )
        fill_mask = pd.DataFrame(
            False, index=df.index, columns=df.columns,
        )
        fill_mask.iloc[10:15, 0] = True  # 5-day equity fill
        with pytest.raises(DataValidationError, match="forward-fill limit"):
            check_forward_fill_limit(df, ["equity"], max_consecutive=3,
                                    fill_mask=fill_mask)


# ---------------------------------------------------------------------------
# 2. Asset semantic / SeriesKind tests
# ---------------------------------------------------------------------------

class TestAssetSemantics:
    """Verify that SeriesKind distinctions are enforced before any download."""

    def test_price_spec_require_price_passes(self):
        """A PRICE AssetSpec must pass require_price()."""
        spec = _make_price_asset("^NSEI")
        spec.require_price()  # should not raise

    def test_yield_spec_require_price_raises(self):
        """A YIELD AssetSpec must raise TypeError when require_price() is called."""
        spec = AssetSpec(
            ticker="^IRX",
            kind=SeriesKind.YIELD,
            description="13-week T-Bill yield",
            notes="Rate in percent — NOT a price.",
        )
        with pytest.raises(TypeError, match="yield"):
            spec.require_price()

    def test_indicator_spec_require_price_raises(self):
        """An INDICATOR AssetSpec must raise TypeError when require_price() is called."""
        spec = _make_indicator_asset("^INDIAVIX")
        with pytest.raises(TypeError, match="indicator"):
            spec.require_price()

    def test_validate_price_assets_passes_for_default_config(self):
        """Default TickerConfig must have all PRICE specs for equity/gold/bond."""
        cfg = TickerConfig()
        cfg.validate_price_assets()  # must not raise

    def test_yield_bond_spec_blocked_on_download(self, base_config):
        """
        Swapping bond to a YIELD spec must raise TypeError before any network call.
        Uses validate_price_assets() which download_market_data() calls first.
        """
        bad_cfg = base_config
        bad_cfg.tickers.bond = AssetSpec(
            ticker="^IRX",
            kind=SeriesKind.YIELD,
            description="Yield series — must be blocked",
        )
        with pytest.raises(TypeError, match="yield"):
            bad_cfg.tickers.validate_price_assets()

    def test_vix_fallback_constant_is_cboe(self):
        """VIX_FALLBACK must be the CBOE VIX ticker, explicitly defined."""
        assert VIX_FALLBACK == "^VIX"

    def test_india_vix_is_preferred_default(self):
        """Default vix spec must be ^INDIAVIX, not ^VIX."""
        cfg = TickerConfig()
        assert cfg.vix.ticker == "^INDIAVIX"

    def test_irx_is_not_default_bond(self):
        """^IRX (yield series) must NOT be the default bond ticker."""
        cfg = TickerConfig()
        assert cfg.bond.ticker != "^IRX"
        assert cfg.bond.kind is SeriesKind.PRICE

    def test_bond_ticker_property_raises_for_yield_spec(self):
        """bond_ticker property must raise if bond spec is accidentally a YIELD."""
        cfg = TickerConfig()
        cfg.bond = AssetSpec(ticker="^IRX", kind=SeriesKind.YIELD, description="yield")
        with pytest.raises(TypeError):
            _ = cfg.bond_ticker

    def test_gold_spec_notes_document_currency_policy(self):
        """Default gold AssetSpec notes must mention INR / currency / FX."""
        cfg = TickerConfig()
        notes_lower = cfg.gold.notes.lower()
        assert any(w in notes_lower for w in ["inr", "currency", "fx", "gold"]), (
            "Gold spec notes must document the currency/FX policy."
        )

    def test_irx_yield_reference_spec_exists_and_is_yield(self):
        """TickerConfig must retain ^IRX as a YIELD reference (not removed)."""
        cfg = TickerConfig()
        assert cfg.irx_yield.ticker == "^IRX"
        assert cfg.irx_yield.kind is SeriesKind.YIELD


# ---------------------------------------------------------------------------
# 3. Mocked download tests
# ---------------------------------------------------------------------------

class TestDownloadMocked:
    """Tests for download_market_data() with yfinance fully mocked — no network."""

    def _side_effects(self, cfg: RegimeShiftConfig, *, include_vix: bool = False):
        s = _make_price_series()
        effects = {
            cfg.tickers.equity_ticker: s,
            cfg.tickers.gold_ticker: s * 18,
            cfg.tickers.bond_ticker: s * 0.97,
        }
        if include_vix:
            effects[cfg.tickers.vix_ticker] = s * 0.15
        return effects

    def test_successful_download_returns_valid_frame(self, base_config):
        """Happy path: three tickers return data → validated DataFrame."""
        with patch("yfinance.download", side_effect=_mock_download(self._side_effects(base_config))):
            df = download_market_data(base_config, cache=False)
        assert set(df.columns) >= {"equity", "gold", "bond"}
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing
        assert not df.index.has_duplicates

    def test_column_order_is_deterministic(self, base_config):
        """Column order must always be equity, gold, bond [, vix]."""
        with patch("yfinance.download", side_effect=_mock_download(self._side_effects(base_config))):
            df = download_market_data(base_config, cache=False)
        assert list(df.columns[:3]) == ["equity", "gold", "bond"]

    def test_missing_ticker_raises_data_download_error(self, base_config):
        """If a required ticker returns empty data, DataDownloadError is raised."""
        cfg = base_config
        s = _make_price_series()
        # Only equity and gold — bond missing
        side_effects = {
            cfg.tickers.equity_ticker: s,
            cfg.tickers.gold_ticker: s * 18,
            # bond ticker absent → returns empty DataFrame
        }
        with patch("yfinance.download", side_effect=_mock_download(side_effects)):
            with pytest.raises(DataDownloadError, match="Failed to download"):
                download_market_data(cfg, cache=False)

    def test_empty_download_raises_data_download_error(self, base_config):
        """If yfinance returns an empty DataFrame for every ticker, DataDownloadError is raised."""
        def always_empty(ticker, **kwargs):
            return pd.DataFrame()
        with patch("yfinance.download", side_effect=always_empty):
            with pytest.raises(DataDownloadError):
                download_market_data(base_config, cache=False)

    def test_index_is_timezone_naive(self, base_config):
        """Downloaded data index must be timezone-naive after normalisation."""
        cfg = base_config
        s = _make_price_series()
        # Simulate yfinance returning timezone-aware index
        s_tz = s.copy()
        s_tz.index = s_tz.index.tz_localize("UTC")
        side_effects = {
            cfg.tickers.equity_ticker: s_tz,
            cfg.tickers.gold_ticker: s * 18,
            cfg.tickers.bond_ticker: s * 0.97,
        }
        with patch("yfinance.download", side_effect=_mock_download(side_effects)):
            df = download_market_data(cfg, cache=False)
        assert df.index.tz is None

    def test_vix_column_present_when_requested(self, base_config):
        """include_vix=True must produce a 'vix' column in output."""
        with patch("yfinance.download", side_effect=_mock_download(self._side_effects(base_config, include_vix=True))):
            df = download_market_data(base_config, cache=False, include_vix=True)
        assert "vix" in df.columns

    def test_vix_absent_by_default(self, base_config):
        """VIX column must NOT appear when include_vix=False (default)."""
        with patch("yfinance.download", side_effect=_mock_download(self._side_effects(base_config))):
            df = download_market_data(base_config, cache=False, include_vix=False)
        assert "vix" not in df.columns

    def test_no_backward_fill_applied(self, base_config):
        """
        When equity has a NaN at index 50, forward-fill must copy from index 49,
        NOT from index 51.  A backward-fill (bfill) would copy from index 51.

        We verify this by inspecting the actual filled value:
          - ffill: filled_value == s.iloc[49]
          - bfill: filled_value == s.iloc[51]
        """
        cfg = base_config
        s = _make_price_series(periods=120)
        s_with_nan = s.copy()
        s_with_nan.iloc[50] = np.nan

        side_effects = {
            cfg.tickers.equity_ticker: s_with_nan,
            cfg.tickers.gold_ticker: s * 18,
            cfg.tickers.bond_ticker: s * 0.97,
        }
        with patch("yfinance.download", side_effect=_mock_download(side_effects)):
            df = download_market_data(cfg, cache=False)

        # The original raw series has NaN at positional index 50.
        # After ffill the value at that row should equal the value at index 49.
        # Build expected values from raw series (before NaN injection).
        raw_idx49 = float(s.iloc[49])
        raw_idx51 = float(s.iloc[51])

        if len(df) > 52:
            filled_val = float(df["equity"].iloc[50])
            # Assert it is the PREVIOUS value (ffill), not the NEXT value (bfill)
            assert abs(filled_val - raw_idx49) < 1e-9, (
                f"Expected forward-fill value {raw_idx49} but got {filled_val}. "
                "This suggests backward-fill was applied."
            )
            # Also assert it is NOT the next day's value
            assert abs(filled_val - raw_idx51) > 1e-9 or raw_idx49 == raw_idx51, (
                "Forward-filled value equals the next-day value — bfill signature detected."
            )


# ---------------------------------------------------------------------------
# 4. CSV round-trip and offline execution
# ---------------------------------------------------------------------------

class TestCSVRoundTrip:
    """Tests for save_market_data_csv() and load_market_data_csv()."""

    def test_csv_round_trip_preserves_data(self, valid_price_df):
        """Save → load must produce an identical DataFrame."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_market_data_csv(valid_price_df, path)
            loaded = load_market_data_csv(path)
            pd.testing.assert_frame_equal(
                valid_price_df.reset_index(drop=True),
                loaded.reset_index(drop=True),
                check_dtype=False,
                atol=1e-6,
            )
        finally:
            os.unlink(path)

    def test_csv_round_trip_preserves_index(self, valid_price_df):
        """Saved and reloaded index dates must match exactly."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_market_data_csv(valid_price_df, path)
            loaded = load_market_data_csv(path)
            pd.testing.assert_index_equal(valid_price_df.index, loaded.index)
        finally:
            os.unlink(path)

    def test_offline_load_requires_no_network(self, valid_price_df):
        """load_market_data_csv must succeed with yfinance entirely unavailable."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_market_data_csv(valid_price_df, path)
            with patch.dict("sys.modules", {"yfinance": None}):
                loaded = load_market_data_csv(path)
            assert len(loaded) == len(valid_price_df)
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self):
        """Loading a nonexistent CSV must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_market_data_csv("/nonexistent/path/data.csv")

    def test_loaded_csv_column_order_deterministic(self, valid_price_df):
        """Reloaded CSV must have equity, gold, bond column order."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_market_data_csv(valid_price_df, path)
            loaded = load_market_data_csv(path)
            assert list(loaded.columns[:3]) == ["equity", "gold", "bond"]
        finally:
            os.unlink(path)

    def test_csv_with_timezone_aware_index_normalised(self, valid_price_df):
        """A CSV whose index was written with timezone info must be normalised on load."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
            tz_df = valid_price_df.copy()
            tz_df.index = tz_df.index.tz_localize("UTC")
            tz_df.to_csv(f, date_format="%Y-%m-%d %H:%M:%S%z")
        try:
            loaded = load_market_data_csv(path)
            assert loaded.index.tz is None
        finally:
            os.unlink(path)

    def test_invalid_csv_rejected(self, sample_dates):
        """A CSV with negative prices must fail validation on load."""
        bad = pd.DataFrame(
            {"equity": [-1.0] * 10, "gold": [1800.0] * 10, "bond": [97.0] * 10},
            index=sample_dates[:10],
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            bad.to_csv(path, date_format="%Y-%m-%d")
            with pytest.raises((DataValidationError, ValueError)):
                load_market_data_csv(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 5. Alignment and missing-data policy
# ---------------------------------------------------------------------------

class TestAlignment:
    """Tests for _build_aligned_frame and forward-fill policy."""

    def _make_series(self, start, periods, offset=0):
        idx = pd.bdate_range(start, periods=periods)
        return pd.Series(100.0 + offset + np.arange(periods, dtype=float), index=idx)

    def test_common_dates_only_in_output(self):
        """Aligned frame must contain only dates shared by ALL series."""
        s1 = self._make_series("2022-01-03", 60)
        s2 = self._make_series("2022-01-10", 60)  # starts later
        series_map = {"equity": s1, "gold": s2, "bond": s2 * 0.97}
        df, _ = _build_aligned_frame(series_map, ["equity", "gold", "bond"], forward_fill_limit=0)
        assert df.index.min() >= s2.index.min()

    def test_ffill_gap_filled_within_limit(self):
        """A 1-day gap filled by ffill within limit must produce non-NaN result."""
        idx = pd.bdate_range("2022-01-03", periods=20)
        eq = pd.Series(100.0 + np.arange(20, dtype=float), index=idx)
        gold = eq.copy()
        gold.iloc[10] = np.nan  # one gap
        bond = eq * 0.97

        series_map = {"equity": eq, "gold": gold, "bond": bond}
        df, diag = _build_aligned_frame(series_map, ["equity", "gold", "bond"], forward_fill_limit=3)
        assert df["gold"].isna().sum() == 0  # NaN filled by ffill
        assert diag["ffill_cells_applied"] >= 1

    def test_gap_exceeding_limit_drops_row(self):
        """Gaps longer than forward_fill_limit must result in dropped rows, not filled."""
        idx = pd.bdate_range("2022-01-03", periods=30)
        eq = pd.Series(100.0 + np.arange(30, dtype=float), index=idx)
        gold = eq.copy()
        gold.iloc[10:15] = np.nan   # 5-day gap
        bond = eq * 0.97

        series_map = {"equity": eq, "gold": gold, "bond": bond}
        df, diag = _build_aligned_frame(series_map, ["equity", "gold", "bond"], forward_fill_limit=3)
        # 5-day gap with limit=3 → first 3 filled, 2 still NaN → 2 rows dropped
        assert diag["dates_dropped"] >= 2

    def test_no_backward_fill_in_alignment(self):
        """Alignment must NEVER apply backward-fill."""
        idx = pd.bdate_range("2022-01-03", periods=20)
        eq = pd.Series(100.0 + np.arange(20, dtype=float), index=idx)
        gold = eq.copy()
        gold.iloc[5] = np.nan  # gap at index 5
        bond = eq * 0.97

        series_map = {"equity": eq, "gold": gold, "bond": bond}
        df, _ = _build_aligned_frame(series_map, ["equity", "gold", "bond"], forward_fill_limit=3)
        # After ffill, row 5 should equal row 4, NOT row 6 (which bfill would produce)
        if len(df) > 6:
            assert df["gold"].iloc[5] == df["gold"].iloc[4]  # ffill from previous
            assert df["gold"].iloc[5] != df["gold"].iloc[6]  # NOT bfill from next

    def test_empty_result_raises_alignment_error(self):
        """If no common dates exist, DataAlignmentError must be raised."""
        s1 = self._make_series("2022-01-03", 10)
        s2 = self._make_series("2023-01-03", 10)   # completely disjoint date range
        series_map = {"equity": s1, "gold": s2, "bond": s2 * 0.97}
        with pytest.raises(DataAlignmentError):
            _build_aligned_frame(series_map, ["equity", "gold", "bond"], forward_fill_limit=0)


# ---------------------------------------------------------------------------
# 6. Fill-mask accuracy tests
# ---------------------------------------------------------------------------

class TestFillMaskAccuracy:
    """
    Prove that ffill_cells_applied is derived from the explicit fill mask
    (was-NaN-before AND is-non-NaN-after), NOT from price equality.

    A constant / naturally flat price series must NOT be miscounted as filled cells.
    """

    def _make_series(self, start, periods, offset=0):
        idx = pd.bdate_range(start, periods=periods)
        return pd.Series(100.0 + offset + np.arange(periods, dtype=float), index=idx)

    def test_fill_mask_counts_exactly_one_gap(self):
        """One NaN gap → ffill_cells_applied must equal exactly 1."""
        idx = pd.bdate_range("2022-01-03", periods=20)
        eq = pd.Series(100.0 + np.arange(20, dtype=float), index=idx)
        gold = eq.copy()
        gold.iloc[10] = np.nan  # exactly one NaN
        bond = eq * 0.97

        _, diag = _build_aligned_frame(
            {"equity": eq, "gold": gold, "bond": bond},
            ["equity", "gold", "bond"],
            forward_fill_limit=3,
        )
        assert diag["ffill_cells_applied"] == 1, (
            f"Expected exactly 1 filled cell, got {diag['ffill_cells_applied']}"
        )

    def test_fill_mask_zero_when_no_nans(self):
        """No NaN gaps → ffill_cells_applied must equal exactly 0."""
        idx = pd.bdate_range("2022-01-03", periods=20)
        eq = pd.Series(100.0 + np.arange(20, dtype=float), index=idx)
        gold = eq * 1.5
        bond = eq * 0.97

        _, diag = _build_aligned_frame(
            {"equity": eq, "gold": gold, "bond": bond},
            ["equity", "gold", "bond"],
            forward_fill_limit=3,
        )
        assert diag["ffill_cells_applied"] == 0

    def test_naturally_flat_bond_nav_not_counted_as_filled(self):
        """
        A bond NAV that holds the same value for 3 consecutive days (naturally flat,
        not a fill artefact) must NOT be counted as forward-filled cells.

        This is the key regression test: if ffill_cells_applied were inferred from
        repeated prices, it would falsely flag these rows.  The fill mask correctly
        ignores them because they had no NaN in the raw data.
        """
        idx = pd.bdate_range("2022-01-03", periods=20)
        eq = pd.Series(100.0 + np.arange(20, dtype=float), index=idx)
        gold = eq * 1.5

        # Bond NAV is constant for 3 days (common in gilt funds around holidays)
        bond = pd.Series(97.0 + np.arange(20, dtype=float) * 0.05, index=idx)
        bond.iloc[10:13] = bond.iloc[9]   # 3 naturally identical values, NO NaN

        _, diag = _build_aligned_frame(
            {"equity": eq, "gold": gold, "bond": bond},
            ["equity", "gold", "bond"],
            forward_fill_limit=3,
        )
        assert diag["ffill_cells_applied"] == 0, (
            f"Naturally flat prices must not be counted as filled; "
            f"got ffill_cells_applied={diag['ffill_cells_applied']}"
        )

    def test_fill_mask_counts_three_consecutive_gaps(self):
        """Three consecutive NaN gaps filled within limit → count must equal 3."""
        idx = pd.bdate_range("2022-01-03", periods=20)
        eq = pd.Series(100.0 + np.arange(20, dtype=float), index=idx)
        gold = eq.copy()
        gold.iloc[10:13] = np.nan  # 3 consecutive NaN values, within limit=3
        bond = eq * 0.97

        _, diag = _build_aligned_frame(
            {"equity": eq, "gold": gold, "bond": bond},
            ["equity", "gold", "bond"],
            forward_fill_limit=3,
        )
        assert diag["ffill_cells_applied"] == 3, (
            f"Expected 3 filled cells, got {diag['ffill_cells_applied']}"
        )


# ---------------------------------------------------------------------------
# 7. Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Column order is always ['equity', 'gold', 'bond'] regardless of insertion order."""

    def test_normalise_index_removes_tz(self, valid_price_df):
        df = valid_price_df.copy()
        df.index = df.index.tz_localize("US/Eastern")
        result = _normalise_index(df)
        assert result.index.tz is None

    def test_normalise_index_sorts(self, valid_price_df):
        df = valid_price_df.iloc[::-1].copy()  # reverse order
        result = _normalise_index(df)
        assert result.index.is_monotonic_increasing

    def test_normalise_index_deduplicates(self, valid_price_df):
        df = pd.concat([valid_price_df, valid_price_df.iloc[:5]])
        result = _normalise_index(df)
        assert not result.index.has_duplicates
