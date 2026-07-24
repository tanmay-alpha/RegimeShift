"""
Data contract and integrity validation routines for multi-asset market data.

Enforces strict input data guarantees required for leakage-safe regime detection
and portfolio optimization.
"""

from typing import List, Optional
import pandas as pd
import numpy as np


def check_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """Verify that all required asset columns are present in the DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required asset columns in DataFrame: {missing}")


def check_monotonic_index(df: pd.DataFrame) -> None:
    """Verify that DataFrame index is a DatetimeIndex and strictly monotonic ascending."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame DatetimeIndex must be sorted in strictly ascending chronological order.")


def check_no_duplicates(df: pd.DataFrame) -> None:
    """Verify that there are no duplicate dates in the DatetimeIndex."""
    if df.index.has_duplicates:
        duplicates = df.index[df.index.duplicated()].unique()
        raise ValueError(f"Duplicate dates found in DatetimeIndex: {duplicates}")


def check_numeric_positive(df: pd.DataFrame, asset_cols: List[str]) -> None:
    """Verify that asset price series contain strictly positive numeric values."""
    for col in asset_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric.")
        if (df[col] <= 0).any():
            raise ValueError(f"Column '{col}' contains non-positive price values (<= 0).")
        if np.isinf(df[col]).any():
            raise ValueError(f"Column '{col}' contains infinite values.")


def validate_price_data(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    allow_missing: bool = False
) -> pd.DataFrame:
    """
    Validate multi-asset price DataFrame against the official RegimeShift data contract.

    Args:
        df: Input price DataFrame.
        required_cols: List of required asset column names. Defaults to ['equity', 'gold', 'bond'].
        allow_missing: If False, raises error when missing NaN values are detected.

    Returns:
        Validated DataFrame.

    Raises:
        ValueError, TypeError if data contract specifications are violated.
    """
    if required_cols is None:
        required_cols = ["equity", "gold", "bond"]

    if df.empty:
        raise ValueError("Price DataFrame is empty.")

    check_monotonic_index(df)
    check_no_duplicates(df)
    check_required_columns(df, required_cols)
    check_numeric_positive(df, required_cols)

    if not allow_missing and df[required_cols].isna().any().any():
        raise ValueError("Price DataFrame contains missing (NaN) values.")

    return df
