"""
Feature engineering interface for regime detection.

Calculates leakage-safe rolling statistical indicators (returns, volatility,
cross-asset correlations) for Hidden Markov Model fitting.
"""

import pandas as pd


def compute_regime_features(
    prices: pd.DataFrame,
    lookback: int = 252
) -> pd.DataFrame:
    """
    Compute leakage-free rolling features from asset prices for HMM regime detection.

    Args:
        prices: Validated price DataFrame with DatetimeIndex.
        lookback: Rolling window size in trading days.

    Returns:
        pd.DataFrame of standardized features aligned with price index.

    Raises:
        NotImplementedError: Feature engineering engine will be implemented in the next phase.
    """
    raise NotImplementedError(
        "Leakage-safe feature engineering pipeline will be implemented in the next phase."
    )
