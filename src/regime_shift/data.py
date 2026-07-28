"""
Multi-asset market data pipeline for RegimeShift.

Provides two ingestion paths:

  1. download_market_data(...)
       Fetches live price history from Yahoo Finance via yfinance.
       Requires an internet connection.

  2. load_market_data_csv(path, ...)
       Reads a locally cached CSV file produced by save_market_data_csv().
       Fully offline — suitable for unseen-data evaluation.

Both paths return an identical, validated pd.DataFrame:

    Index  : DatetimeIndex — timezone-naive, sorted ascending, unique dates.
    Columns: ['equity', 'gold', 'bond']        ← always present
             ['vix']                            ← present only when include_vix=True

Column names are controlled entirely by RegimeShiftConfig; no ticker string
appears in any function other than _download_single() and the ticker_to_col map.

Missing-data policy
-------------------
- Yield/indicator series passed where a price is expected raise DataDownloadError
  immediately (before any network call).
- Assets that return empty data raise DataDownloadError immediately.
- After outer-joining on all dates, remaining NaN values in required columns are
  forward-filled up to config.data.forward_fill_limit days.  The number of filled
  cells is tracked via an explicit fill mask (not inferred from repeated prices).
- Residual NaN values after forward-filling raise DataValidationError.
- Backward-fill is NEVER applied.

Caching
-------
save_market_data_csv(df, path) and load_market_data_csv(path) provide a
round-trippable CSV cache.  No database is used.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from regime_shift.config import AssetSpec, RegimeShiftConfig, SeriesKind
from regime_shift.exceptions import DataAlignmentError, DataDownloadError, DataValidationError
from regime_shift.validation import validate_price_data

logger = logging.getLogger(__name__)

# Deterministic output column order so consumers can rely on positional indexing.
_CORE_COLS = ["equity", "gold", "bond"]
_OPTIONAL_COLS = ["vix"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    return date.today().isoformat()


def _download_single(
    ticker: str,
    start: str,
    end: str,
    price_field: str = "Close",
) -> pd.Series:
    """
    Download a single ticker from Yahoo Finance and return a named Series.

    Raises:
        DataDownloadError: If the download returns empty data or raises an
            exception (network error, invalid ticker, etc.).
    """
    try:
        import yfinance as yf  # lazy import — not required for offline path
    except ImportError as exc:
        raise DataDownloadError(
            ticker, start, end,
            "yfinance is not installed. Run: pip install yfinance"
        ) from exc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                actions=False,
            )
    except Exception as exc:
        raise DataDownloadError(ticker, start, end, str(exc)) from exc

    if raw is None or raw.empty:
        raise DataDownloadError(
            ticker, start, end,
            "yfinance returned an empty DataFrame. "
            "Check ticker symbol, date range, or network access."
        )

    # yfinance may return MultiIndex columns — flatten
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    if price_field not in raw.columns:
        available = raw.columns.tolist()
        raise DataDownloadError(
            ticker, start, end,
            f"Price field '{price_field}' not found in downloaded columns: {available}"
        )

    series = raw[price_field].copy()

    # Remove timezone info and normalise to date-only index
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series.index = pd.DatetimeIndex(series.index.normalize())

    # Drop any NaN rows that yfinance sometimes appends at boundaries
    series = series.dropna()

    if series.empty:
        raise DataDownloadError(
            ticker, start, end,
            f"All rows were NaN after dropping missing values for price field '{price_field}'."
        )

    return series


def _build_aligned_frame(
    series_map: Dict[str, pd.Series],
    col_order: List[str],
    forward_fill_limit: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Align asset series to common valid trading dates and forward-fill sparingly.

    Forward-fill cell count is tracked via an explicit boolean fill mask
    (positions that were NaN before ffill and non-NaN after).  This is exact
    and does NOT infer fills from repeated identical price values, which would
    produce false positives for naturally flat prices.

    Policy:
      - Outer-join on all dates to preserve the full universe.
      - Forward-fill up to forward_fill_limit days per gap.
      - Fill count reported from fill mask, not from price equality.
      - Drop rows where any required column is still NaN after fill.
      - Backward-fill is NEVER applied.

    Args:
        series_map: mapping of output_col_name → pd.Series.
        col_order: Desired deterministic column output order.
        forward_fill_limit: Max consecutive days to forward-fill (0 = disabled).

    Returns:
        (aligned_df, diagnostics_dict)
    """
    # Build raw frame — outer join preserves all dates initially
    raw = pd.DataFrame({col: series_map[col] for col in col_order if col in series_map})
    total_dates_pre = len(raw)

    # Forward-fill with strict limit BEFORE dropping.
    # Track filled cells via explicit mask — not from price equality, so
    # naturally flat prices (e.g. a constant bond NAV) are never miscounted.
    if forward_fill_limit > 0:
        was_nan_before = raw.isna()
        filled = raw.ffill(limit=forward_fill_limit)
        fill_mask = was_nan_before & filled.notna()   # True only at cells actually filled
        ffill_applied = int(fill_mask.sum().sum())
        logger.info("Forward-filled %d NaN observations (limit=%d days).", ffill_applied, forward_fill_limit)
    else:
        filled = raw
        fill_mask = pd.DataFrame(False, index=raw.index, columns=raw.columns)
        ffill_applied = 0

    # Drop rows where any required column is still NaN
    clean = filled.dropna(subset=[c for c in col_order if c in series_map])
    dropped = total_dates_pre - len(clean)
    if dropped > 0:
        logger.info(
            "Dropped %d dates due to missing values in required columns after "
            "forward-fill (limit=%d).",
            dropped, forward_fill_limit,
        )

    if clean.empty:
        raise DataAlignmentError(
            "No common valid trading dates remain after alignment. "
            "Check that all tickers have overlapping date ranges."
        )

    diagnostics: Dict[str, int] = {
        "total_dates_before_alignment": total_dates_pre,
        "dates_dropped": dropped,
        "ffill_cells_applied": ffill_applied,
        "ffill_mask": fill_mask,
    }

    return clean, diagnostics


def _normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timezone-naive, date-normalised, sorted, deduplicated index."""
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        idx = idx.normalize()
    else:
        idx = pd.DatetimeIndex(idx).normalize()

    df = df.copy()
    df.index = idx
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# Public API — Online path
# ---------------------------------------------------------------------------

def download_market_data(
    config: Optional[RegimeShiftConfig] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    include_vix: bool = False,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download multi-asset price history from Yahoo Finance.

    Requires internet access.  For offline / evaluation use, see load_market_data_csv().

    Args:
        config: RegimeShiftConfig instance (creates default if None).
        start: Start date (YYYY-MM-DD).  Defaults to config.data.default_start.
        end: End date (YYYY-MM-DD).  Defaults to today.
        include_vix: If True, also download the VIX series and include a 'vix' column.
        cache: If True and config.data.cache_dir is set, save result to cache.

    Returns:
        Validated pd.DataFrame with columns ['equity','gold','bond'] (+ 'vix' if requested).
        Index: timezone-naive DatetimeIndex sorted ascending.

    Raises:
        DataDownloadError: If any required asset fails to download.
        DataAlignmentError: If aligned DataFrame is empty after common-date join.
        DataValidationError: If the assembled DataFrame violates data contracts.
    """
    cfg = config or RegimeShiftConfig()
    start = start or cfg.data.default_start
    end = end or _today()
    pf = cfg.data.price_field

    logger.info("Downloading market data: %s → %s", start, end)

    # Guard: ensure all required asset specs are PRICE series before any network call.
    # This prevents accidentally downloading a yield series and using it as a price.
    cfg.tickers.validate_price_assets()

    # Determine which tickers to download
    required_tickers = {
        cfg.equity_col: cfg.tickers.equity_ticker,
        cfg.gold_col:   cfg.tickers.gold_ticker,
        cfg.bond_col:   cfg.tickers.bond_ticker,
    }
    if include_vix:
        required_tickers[cfg.vix_col] = cfg.tickers.vix_ticker

    series_map: Dict[str, pd.Series] = {}
    for col, ticker in required_tickers.items():
        logger.info("  Downloading %-8s (%s)…", col, ticker)
        series = _download_single(ticker, start, end, price_field=pf)
        series.name = col
        series_map[col] = series
        logger.info("  %-8s: %d observations  [%s → %s]",
                    col, len(series), series.index[0].date(), series.index[-1].date())

    # Build deterministic column order
    col_order = [c for c in _CORE_COLS + _OPTIONAL_COLS if c in series_map]

    aligned, diagnostics = _build_aligned_frame(
        series_map, col_order, cfg.data.forward_fill_limit
    )
    aligned = _normalise_index(aligned)

    logger.info(
        "Alignment complete: %d rows retained, %d dropped, %d ffill cells.",
        len(aligned), diagnostics["dates_dropped"], diagnostics["ffill_cells_applied"],
    )

    # Final validation
    validate_price_data(
        aligned,
        required_cols=cfg.core_assets,
        allow_missing=False,
        forward_fill_limit=cfg.data.forward_fill_limit,
        fill_mask=diagnostics.get("ffill_mask"),
    )

    if cache and cfg.data.cache_dir:
        _write_cache(aligned, cfg.data.cache_dir, start, end, include_vix)

    # Store diagnostics on DataFrame attrs so run_metadata can pick them up.
    mask = diagnostics.get("ffill_mask")
    per_asset = {c: int(mask[c].sum()) for c in mask.columns} if mask is not None else {}
    aligned.attrs["data_diagnostics"] = {
        "ffill_cells_total": diagnostics.get("ffill_cells_applied", 0),
        "ffill_cells_per_asset": per_asset,
        "dates_dropped": diagnostics.get("dates_dropped", 0),
        "rows_after_load": len(aligned),
    }

    return aligned


# ---------------------------------------------------------------------------
# Public API — Offline path
# ---------------------------------------------------------------------------

def load_market_data_csv(
    path: str,
    config: Optional[RegimeShiftConfig] = None,
    forward_fill_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a market data CSV file produced by save_market_data_csv().

    This function requires NO internet access and is suitable for unseen
    evaluation runs where a pre-built CSV is supplied.

    Args:
        path: Absolute or relative path to the CSV file.
        config: RegimeShiftConfig instance (creates default if None).
        forward_fill_limit: Override for maximum forward-fill days.
            Defaults to config.data.forward_fill_limit.

    Returns:
        Validated pd.DataFrame with required asset columns and a
        ``data_diagnostics`` attr containing fill counts per asset and
        dates dropped.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        DataValidationError: If the loaded data violates contracts.
    """
    cfg = config or RegimeShiftConfig()
    ffl = forward_fill_limit if forward_fill_limit is not None else cfg.data.forward_fill_limit
    p = Path(path)
    # If the path does not exist as-is (e.g. notebook CWD is notebooks/),
    # also try ../<path> and ../../<path> so that "data/foo.csv" works
    # from notebooks/, scripts/, and the project root.
    candidates = [p]
    if not p.is_absolute():
        candidates += [Path("..") / p, Path("../..") / p]
    resolved = next((c for c in candidates if c.exists()), None)
    if resolved is None:
        tried = ", ".join(str(c.resolve()) for c in candidates)
        raise FileNotFoundError(
            f"Market data CSV not found: {path} "
            f"(tried: {tried})"
        )
    p = resolved

    logger.info("Loading market data from CSV: %s", p)
    df = pd.read_csv(p, index_col=0, parse_dates=True)

    # Normalise the index (handles tz-aware exports)
    df = _normalise_index(df)

    # Re-apply the canonical forward-fill of up to ffl days so that a CSV
    # recorded with raw NaNs is still bridged conservatively and the cell-
    # count diagnostic reflects the same policy the runtime uses.
    raw_len = len(df)
    if ffl > 0:
        was_nan_before = df.isna()
        filled = df.ffill(limit=ffl)
        fill_mask = was_nan_before & filled.notna()
        ffill_cells = int(fill_mask.sum().sum())
        # per-asset fill counts
        per_asset = {c: int(fill_mask[c].sum()) for c in df.columns}
        # Drop residual NaNs in required columns
        aligned = filled.dropna(subset=[c for c in _CORE_COLS if c in filled.columns])
        dropped = raw_len - len(aligned)
    else:
        ffill_cells = 0
        per_asset = {c: 0 for c in df.columns}
        aligned = df
        dropped = 0

    df = aligned

    # Ensure deterministic column order (only include columns that exist)
    ordered = [c for c in _CORE_COLS + _OPTIONAL_COLS if c in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    df = df[ordered + extra]

    validate_price_data(
        df,
        required_cols=cfg.core_assets,
        allow_missing=False,
        forward_fill_limit=ffl,
        fill_mask=fill_mask if ffl > 0 else None,
    )

    # Diagnostics for run_metadata.json
    df.attrs["data_diagnostics"] = {
        "data_path": str(p.resolve()),
        "ffill_limit": ffl,
        "ffill_cells_total": ffill_cells,
        "ffill_cells_per_asset": per_asset,
        "dates_dropped": dropped,
        "rows_after_load": len(df),
    }

    logger.info("Loaded %d rows from %s.", len(df), p.name)
    return df


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def save_market_data_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save a validated market data DataFrame to a CSV file.

    The CSV can be reloaded via load_market_data_csv() for offline runs.

    Args:
        df: Validated DataFrame (index = DatetimeIndex).
        path: Destination file path (parent directories created if absent).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, date_format="%Y-%m-%d")
    logger.info("Saved market data to %s (%d rows).", p, len(df))


def _write_cache(
    df: pd.DataFrame,
    cache_dir: str,
    start: str,
    end: str,
    include_vix: bool,
) -> None:
    """Write aligned data to a date-stamped cache CSV (internal use)."""
    vix_tag = "_vix" if include_vix else ""
    filename = f"market_data_{start}_{end}{vix_tag}.csv"
    dest = Path(cache_dir) / filename
    try:
        save_market_data_csv(df, str(dest))
        logger.info("Cached market data → %s", dest)
    except OSError as exc:
        logger.warning("Could not write cache file %s: %s", dest, exc)
