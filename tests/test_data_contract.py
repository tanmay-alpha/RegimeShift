"""
Test data contract validation functions.
"""

import pytest
import pandas as pd
import numpy as np
from regime_shift.validation import validate_price_data


@pytest.fixture
def valid_price_df() -> pd.DataFrame:
    """Create a valid sample multi-asset price DataFrame."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    data = {
        "equity": np.linspace(100, 110, 100),
        "gold": np.linspace(200, 205, 100),
        "bond": np.linspace(50, 52, 100),
    }
    return pd.DataFrame(data, index=dates)


def test_valid_data_passes(valid_price_df):
    """Valid price data should pass validation without error."""
    result = validate_price_data(valid_price_df)
    assert len(result) == 100


def test_rejects_missing_required_asset(valid_price_df):
    """Should reject DataFrame missing required asset columns."""
    df_missing = valid_price_df.drop(columns=["bond"])
    with pytest.raises(ValueError, match="Missing required asset columns"):
        validate_price_data(df_missing)


def test_rejects_unsorted_dates(valid_price_df):
    """Should reject DataFrame with unsorted DatetimeIndex."""
    df_unsorted = valid_price_df.iloc[::-1]
    with pytest.raises(ValueError, match="sorted in strictly ascending chronological order"):
        validate_price_data(df_unsorted)


def test_rejects_duplicate_dates(valid_price_df):
    """Should reject DataFrame containing duplicate dates."""
    dates = valid_price_df.index.to_list()
    dates[1] = dates[0]
    df_dups = valid_price_df.copy()
    df_dups.index = pd.DatetimeIndex(dates)
    with pytest.raises(ValueError, match="Duplicate dates found"):
        validate_price_data(df_dups)


def test_rejects_negative_prices(valid_price_df):
    """Should reject DataFrame containing negative or zero prices."""
    df_neg = valid_price_df.copy()
    df_neg.iloc[5, 0] = -10.0
    with pytest.raises(ValueError, match="non-positive price values"):
        validate_price_data(df_neg)


def test_rejects_infinite_values(valid_price_df):
    """Should reject DataFrame containing infinite values."""
    df_inf = valid_price_df.copy()
    df_inf.iloc[10, 1] = np.inf
    with pytest.raises(ValueError, match="infinite values"):
        validate_price_data(df_inf)
