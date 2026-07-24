"""
Feature engineering interface for regime detection.

Calculates leakage-safe rolling statistical indicators (returns, volatility,
cross-asset correlations) for Hidden Markov Model fitting.

Signal timing
-------------
A feature value at date t is computed from data available *through* date t.
The future backtester will use the feature observed at t to make a decision for t+1.
No feature at t ever uses data from t+1 or later.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from regime_shift.config import FeatureConfig
from regime_shift.exceptions import FeatureEngineeringError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic feature column ordering
# ---------------------------------------------------------------------------

# Base features (always present when equity, gold, bond columns exist)
_BASE_FEATURE_COLUMNS: list[str] = [
    "equity_log_return_1d",
    "equity_momentum_21d",
    "equity_momentum_63d",
    "equity_volatility_21d",
    "equity_volatility_ratio_21_63",
    "equity_gold_correlation_63d",
    "equity_bond_correlation_63d",
]

# VIX features (appended only when vix column is present)
_VIX_FEATURE_COLUMNS: list[str] = [
    "vix_change_5d",
    "vix_level",
]


def _get_feature_columns(has_vix: bool) -> list[str]:
    """Return the deterministic feature column order for the given input."""
    cols = list(_BASE_FEATURE_COLUMNS)
    if has_vix:
        cols.extend(_VIX_FEATURE_COLUMNS)
    return cols


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log(price_t / price_{t-1}) for each t.

    Uses shift(1) to get t-1 prices — looks backward only, no future data.
    """
    return np.log(prices / prices.shift(1))


def _compute_momentum(prices: pd.Series, window: int) -> pd.Series:
    """Compute price_t / price_{t-window} - 1.

    Uses shift(window) to get t-window prices — looks backward only.
    """
    return prices / prices.shift(window) - 1


def _compute_annualized_volatility(
    log_returns: pd.Series,
    window: int,
    annual_factor: int,
) -> pd.Series:
    """Compute rolling annualized volatility using a trailing window.

    Uses rolling(window).std() which includes observations [t-window+1, t].
    No centered windows, no future data.
    """
    return log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(
        annual_factor
    )


def _compute_volatility_ratio(
    vol_short: pd.Series,
    vol_long: pd.Series,
) -> pd.Series:
    """Compute vol_short / vol_long.

    Returns NaN where vol_long is zero (division by zero — mathematically undefined).
    """
    result = vol_short / vol_long
    result = result.where(vol_long > 0, np.nan)
    return result


def _compute_rolling_correlation(
    log_returns_x: pd.Series,
    log_returns_y: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling correlation of two return series using a trailing window.

    Uses rolling(window).corr() which includes observations [t-window+1, t].
    No centered windows, no future data.
    """
    return log_returns_x.rolling(window=window, min_periods=window).corr(log_returns_y)


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_raw_features(
    prices: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Compute leakage-safe raw features from validated price data.

    All features use trailing windows only.  No future data is ever accessed.
    The input DataFrame is NOT mutated.

    Args:
        prices: Validated price DataFrame with DatetimeIndex and at least
            columns ['equity', 'gold', 'bond'].  Optional 'vix' column.
        config: FeatureConfig instance.  Defaults to FeatureConfig() if None.

    Returns:
        pd.DataFrame with feature columns and same DatetimeIndex as input
        (before warmup removal).  Column order is deterministic.

    Raises:
        FeatureEngineeringError: If required price columns are missing,
            or if the config is invalid.

    Feature definitions (feature value at date t):

        equity_log_return_1d[t]          = log(equity[t] / equity[t-1])
        equity_momentum_21d[t]           = equity[t] / equity[t-21] - 1
        equity_momentum_63d[t]           = equity[t] / equity[t-63] - 1
        equity_volatility_21d[t]         = std(log_return[t-20 : t+1]) * sqrt(252)
        equity_volatility_ratio_21_63[t] = vol_21d[t] / vol_63d[t]
        equity_gold_correlation_63d[t]   = corr(log_ret_eq, log_ret_gold)[t-62 : t+1]
        equity_bond_correlation_63d[t]   = corr(log_ret_eq, log_ret_bond)[t-62 : t+1]
        vix_change_5d[t]                 = vix[t] / vix[t-5] - 1       (optional)
        vix_level[t]                     = vix[t]                        (optional)

    Signal timing: feature observed at t is used for decision at t+1.
    """
    cfg = config or FeatureConfig()
    cfg.validate()

    required_cols = ["equity", "gold", "bond"]
    missing = [c for c in required_cols if c not in prices.columns]
    if missing:
        raise FeatureEngineeringError(
            f"Input prices DataFrame is missing required columns: {missing}. "
            f"Expected at least {required_cols}."
        )

    # Work on a copy to guarantee no mutation of input
    prices = prices.copy()

    has_vix = "vix" in prices.columns

    # Compute log returns for all price assets
    # shift(1) looks backward — no future data
    equity_log_ret = _compute_log_returns(prices["equity"])
    gold_log_ret = _compute_log_returns(prices["gold"])
    bond_log_ret = _compute_log_returns(prices["bond"])

    # Build feature DataFrame
    features = pd.DataFrame(index=prices.index)

    # 1. equity_log_return_1d
    features["equity_log_return_1d"] = equity_log_ret

    # 2. equity_momentum_21d
    features["equity_momentum_21d"] = _compute_momentum(
        prices["equity"], cfg.short_window
    )

    # 3. equity_momentum_63d
    features["equity_momentum_63d"] = _compute_momentum(
        prices["equity"], cfg.medium_window
    )

    # 4. equity_volatility_21d
    features["equity_volatility_21d"] = _compute_annualized_volatility(
        equity_log_ret, cfg.short_window, cfg.annualization_factor
    )

    # 5. equity_volatility_ratio_21_63
    vol_21 = features["equity_volatility_21d"]
    vol_63 = _compute_annualized_volatility(
        equity_log_ret, cfg.medium_window, cfg.annualization_factor
    )
    features["equity_volatility_ratio_21_63"] = _compute_volatility_ratio(vol_21, vol_63)

    # 6. equity_gold_correlation_63d
    features["equity_gold_correlation_63d"] = _compute_rolling_correlation(
        equity_log_ret, gold_log_ret, cfg.correlation_window
    )

    # 7. equity_bond_correlation_63d
    features["equity_bond_correlation_63d"] = _compute_rolling_correlation(
        equity_log_ret, bond_log_ret, cfg.correlation_window
    )

    # 8-9. VIX features (optional)
    if has_vix:
        features["vix_change_5d"] = _compute_momentum(
            prices["vix"], cfg.vix_change_window
        )
        features["vix_level"] = prices["vix"]

    # Enforce deterministic column order
    features = features[_get_feature_columns(has_vix)]

    return features


# ---------------------------------------------------------------------------
# Warmup removal
# ---------------------------------------------------------------------------

def drop_feature_warmup(
    features: pd.DataFrame,
    minimum_observations: Optional[int] = None,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Remove warmup rows that contain NaN due to insufficient lookback.

    This function:
    - Drops only rows that cannot yet satisfy the required lookback (NaN in
      any column)
    - Reports how many rows were removed
    - Raises a clear error if no usable rows remain
    - Never fills feature NaNs using future data
    - Never silently replaces NaN with zero

    Args:
        features: Feature DataFrame from compute_raw_features().
        minimum_observations: Minimum usable rows required.  Defaults to
            config.minimum_feature_observations if config is provided, else 10.
        config: FeatureConfig instance for default minimum_observations.

    Returns:
        pd.DataFrame with NaN warmup rows removed.  Same columns, same
        DatetimeIndex dtype.

    Raises:
        FeatureEngineeringError: If no usable rows remain after warmup
            removal, or if fewer than minimum_observations rows remain.
    """
    if minimum_observations is None:
        if config is not None:
            minimum_observations = config.minimum_feature_observations
        else:
            minimum_observations = 10

    if features.empty:
        raise FeatureEngineeringError(
            "Feature DataFrame is empty. Cannot drop warmup rows."
        )

    # Identify warmup rows: any NaN in any feature column
    nan_mask = features.isna().any(axis=1)
    warmup_count = int(nan_mask.sum())

    logger.info("Dropping %d warmup rows with NaN features.", warmup_count)

    # Remove warmup rows
    clean = features[~nan_mask].copy()

    if clean.empty:
        raise FeatureEngineeringError(
            f"All {len(features)} feature rows contain NaN after warmup removal. "
            "This indicates the input data is too short for the configured windows. "
            f"Required minimum lookback: {minimum_observations} rows."
        )

    if len(clean) < minimum_observations:
        raise FeatureEngineeringError(
            f"Only {len(clean)} usable feature rows remain after warmup removal, "
            f"which is below the minimum required ({minimum_observations}). "
            f"Input had {len(features)} rows, {warmup_count} were warmup."
        )

    logger.info(
        "Warmup removal complete: %d rows removed, %d rows remain (%.1f%% yield).",
        warmup_count,
        len(clean),
        100.0 * len(clean) / len(features),
    )

    return clean


# ---------------------------------------------------------------------------
# Train-only scaling interfaces
# ---------------------------------------------------------------------------

def fit_feature_scaler(training_features: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler ONLY on the supplied training feature rows.

    This function must NEVER be called on the full feature dataset or on data
    that includes future observations relative to any inference point.

    The future walk-forward HMM will:
    1. Select training data ending at t-1
    2. Call fit_feature_scaler() on that training data
    3. Call transform_features() on training data (for HMM fitting)
    4. Call transform_features() on the current observation (for HMM inference)

    Args:
        training_features: Feature DataFrame from compute_raw_features() with
            warmup already removed.  Must not contain NaN.

    Returns:
        A fitted sklearn StandardScaler instance.

    Raises:
        FeatureEngineeringError: If training_features is empty or contains NaN.
    """
    if training_features.empty:
        raise FeatureEngineeringError(
            "Cannot fit scaler on empty feature DataFrame."
        )
    if training_features.isna().any().any():
        raise FeatureEngineeringError(
            "Cannot fit scaler on features containing NaN. "
            "Call drop_feature_warmup() first."
        )

    scaler = StandardScaler()
    scaler.fit(training_features)
    return scaler


def transform_features(
    scaler: StandardScaler,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform feature rows using a previously fitted scaler.

    Does NOT refit the scaler.  Uses the same scaler that was fit on training
    data only.

    Args:
        scaler: A StandardScaler previously fitted via fit_feature_scaler().
        features: Feature DataFrame to transform.  Must have the same columns
            as the training data used to fit the scaler.

    Returns:
        pd.DataFrame of scaled features with same index and columns as input.

    Raises:
        FeatureEngineeringError: If features is empty, contains NaN, or has
            mismatched columns.
    """
    if features.empty:
        raise FeatureEngineeringError(
            "Cannot transform empty feature DataFrame."
        )
    if features.isna().any().any():
        raise FeatureEngineeringError(
            "Cannot transform features containing NaN. "
            "Call drop_feature_warmup() first."
        )

    # Verify column alignment with scaler's training columns
    expected_cols = list(features.columns)
    if hasattr(scaler, "feature_names_in_"):
        scaler_cols = list(scaler.feature_names_in_)
        if expected_cols != scaler_cols:
            raise FeatureEngineeringError(
                f"Feature column mismatch: expected {scaler_cols}, "
                f"got {expected_cols}. "
                "The scaler was fit on different columns."
            )

    scaled_values = scaler.transform(features)
    scaled_df = pd.DataFrame(
        scaled_values,
        index=features.index,
        columns=features.columns,
    )
    return scaled_df


def fit_transform_training_features(
    training_features: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on training data and return (scaled_features, scaler).

    Convenience function that calls fit_feature_scaler() then transform_features().
    Only suitable for training data — never call on the full dataset.

    Args:
        training_features: Training feature DataFrame with warmup removed.

    Returns:
        (scaled_features, fitted_scaler) tuple.
    """
    scaler = fit_feature_scaler(training_features)
    scaled = transform_features(scaler, training_features)
    return scaled, scaler
