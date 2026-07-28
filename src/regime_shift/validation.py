"""
Data contract and integrity validation routines for multi-asset market data.

Enforces strict input data guarantees required for leakage-safe regime detection
and portfolio optimization.

Rules enforced (see validate_price_data docstring for full contract):
  - DatetimeIndex, sorted ascending, no duplicates, timezone-naive UTC-normalised.
  - Required columns present.
  - All price values numeric, strictly positive, non-infinite.
  - No NaN values in required columns unless allow_missing=True.
  - No backward-fill evidence (detected heuristically via zero-return run check).
  - Forward-fill runs bounded by configurable limit.

A naturally constant price series (e.g. an ETF with low-volatility NAV) is
NOT considered a forward-fill artifact. The ``fill_mask`` recorded by
``data.py`` is the authoritative source for what was forward-filled and
what was naturally flat. ``check_forward_fill_limit`` therefore inspects
the recorded fill mask rather than heuristically inspecting identical-value
runs.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from regime_shift.exceptions import DataValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Raise DataValidationError if any required asset columns are absent."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required asset columns in DataFrame: {missing}"
        )


def check_monotonic_index(df: pd.DataFrame) -> None:
    """Raise DataValidationError if index is not a monotonic-ascending DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            "DataFrame index must be a pandas DatetimeIndex."
        )
    if not df.index.is_monotonic_increasing:
        raise DataValidationError(
            "DataFrame DatetimeIndex must be sorted in strictly ascending "
            "chronological order."
        )


def check_no_duplicates(df: pd.DataFrame) -> None:
    """Raise DataValidationError if any duplicate timestamps exist."""
    if df.index.has_duplicates:
        duplicates = df.index[df.index.duplicated()].unique()
        raise DataValidationError(
            f"Duplicate dates found in DatetimeIndex: {duplicates.tolist()}"
        )


def check_timezone_naive(df: pd.DataFrame) -> None:
    """Raise DataValidationError if the DatetimeIndex has timezone information."""
    if df.index.tz is not None:
        raise DataValidationError(
            f"DatetimeIndex must be timezone-naive (UTC-normalised). "
            f"Got tz='{df.index.tz}'. Call .tz_localize(None) first."
        )


def check_numeric_positive(df: pd.DataFrame, asset_cols: List[str]) -> None:
    """Raise DataValidationError if any price is non-numeric, non-positive, or infinite."""
    for col in asset_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' must be numeric.")
        if (df[col] <= 0).any():
            raise DataValidationError(
                f"Column '{col}' contains non-positive price values (<= 0)."
            )
        if np.isinf(df[col]).any():
            raise DataValidationError(
                f"Column '{col}' contains infinite values."
            )


def check_no_backward_fill(df: pd.DataFrame, asset_cols: List[str]) -> None:
    """
    Heuristically detect suspicious backward-fill patterns.

    A backward-fill creates *increasing* identical runs from the future.
    We detect columns where NaN positions in the raw data appear after
    non-NaN values that could only be filled from the future. Since we
    cannot observe the pre-fill state, we instead flag any column where
    more than 20 % of observations are exact duplicates of the *next*
    observation (which is what bfill produces).
    """
    for col in asset_cols:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        bfill_like = (series == series.shift(-1)).sum()
        frac = bfill_like / len(series)
        if frac > 0.20:
            logger.warning(
                "Column '%s' has %.0f%% values equal to their successor - "
                "possible backward-fill detected. Verify raw data source.",
                col, frac * 100,
            )


def check_forward_fill_limit(
    df: pd.DataFrame,
    asset_cols: List[str],
    max_consecutive: int,
    fill_mask: Optional[pd.DataFrame] = None,
) -> None:
    """
    Raise DataValidationError if any column has been forward-filled for a run
    longer than ``max_consecutive`` days.

    The authoritative check uses the explicit fill mask recorded by the data
    pipeline, NOT a heuristic on identical-value runs. A naturally constant
    series is permitted; only genuine forward-fill runs exceeding the limit
    trigger an error.

    Backward compatibility: if no fill mask is supplied, fall back to the
    identical-value heuristic with a warning that the result is unreliable
    for low-volatility assets.
    """
    if max_consecutive <= 0:
        return

    if fill_mask is None:
        logger.warning(
            "check_forward_fill_limit called without a fill mask - falling "
            "back to identical-value heuristic. Pass fill_mask from the data "
            "pipeline for accurate results."
        )
        for col in asset_cols:
            series = df[col].dropna()
            if len(series) < 2:
                continue
            is_same = series == series.shift(1)
            run_id = (~is_same).cumsum()
            run_lengths = is_same.groupby(run_id).sum()
            max_run = int(run_lengths.max()) if len(run_lengths) > 0 else 0
            if max_run > max_consecutive:
                raise DataValidationError(
                    f"Column '{col}' has a constant-price run of {max_run} "
                    f"days, exceeding the forward-fill limit of "
                    f"{max_consecutive}. Check data source or reduce "
                    "forward_fill_limit."
                )
        return

    for col in asset_cols:
        if col not in fill_mask.columns:
            continue
        col_mask = fill_mask[col].astype(bool).values
        if not col_mask.any():
            continue
        max_run = 0
        cur = 0
        for v in col_mask:
            if v:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        if max_run > max_consecutive:
            raise DataValidationError(
                f"Column '{col}' has a forward-fill run of {max_run} days, "
                f"exceeding the forward-fill limit of {max_consecutive}. "
                "Check data source or reduce forward_fill_limit."
            )


# ---------------------------------------------------------------------------
# Primary validation entry-point
# ---------------------------------------------------------------------------

def validate_price_data(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    allow_missing: bool = False,
    forward_fill_limit: int = 3,
    check_bfill: bool = True,
    fill_mask: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Validate a multi-asset price DataFrame against the RegimeShift data contract.

    Contract rules:
      1. Index is a timezone-naive DatetimeIndex.
      2. Index is monotonically ascending.
      3. Index has no duplicate dates.
      4. Required columns ['equity', 'gold', 'bond'] are present.
      5. All required column values are numeric, strictly positive, and finite.
      6. No NaN values in required columns (unless allow_missing=True).
      7. No backward-fill evidence (heuristic warning, not hard error).
      8. Forward-fill runs do not exceed forward_fill_limit days (uses the
         explicit fill mask if provided).

    Args:
        df: Input price DataFrame.
        required_cols: Required asset column names; defaults to ['equity','gold','bond'].
        allow_missing: If False (default), raise on any NaN in required columns.
        forward_fill_limit: Max consecutive forward-fill days (0 = disabled).
        check_bfill: Whether to run the backward-fill heuristic check.
        fill_mask: Optional explicit fill mask recorded by the data pipeline.

    Returns:
        The validated DataFrame (unchanged).

    Raises:
        DataValidationError: When any contract rule is violated.
    """
    if required_cols is None:
        required_cols = ["equity", "gold", "bond"]

    if df.empty:
        raise DataValidationError("Price DataFrame is empty.")

    check_timezone_naive(df)
    check_monotonic_index(df)
    check_no_duplicates(df)
    check_required_columns(df, required_cols)
    check_numeric_positive(df, required_cols)

    if not allow_missing and df[required_cols].isna().any().any():
        nan_cols = df[required_cols].columns[df[required_cols].isna().any()].tolist()
        raise DataValidationError(
            f"Price DataFrame contains missing (NaN) values in columns: {nan_cols}"
        )

    if check_bfill:
        check_no_backward_fill(df, required_cols)

    if forward_fill_limit > 0:
        check_forward_fill_limit(
            df, required_cols, forward_fill_limit, fill_mask=fill_mask
        )

    return df