"""
RegimeShift: Market-Regime Identification & Dynamic Asset Allocation Framework.

Designed for IIT Bombay Summer Quant 2026 assignment.
"""

from regime_shift.config import (
    AssetSpec,
    DataConfig,
    FeatureConfig,
    RegimeShiftConfig,
    SeriesKind,
    TickerConfig,
    VIX_FALLBACK,
)
from regime_shift.exceptions import (
    RegimeShiftError,
    DataDownloadError,
    DataAlignmentError,
    DataValidationError,
    FeatureEngineeringError,
)
from regime_shift.validation import validate_price_data, check_monotonic_index, check_no_duplicates
from regime_shift.data import download_market_data, load_market_data_csv, save_market_data_csv
from regime_shift.features import (
    compute_raw_features,
    drop_feature_warmup,
    fit_feature_scaler,
    transform_features,
    fit_transform_training_features,
)

__version__ = "0.4.0-feature-pipeline"

__all__ = [
    # Config
    "RegimeShiftConfig",
    "TickerConfig",
    "DataConfig",
    "FeatureConfig",
    "AssetSpec",
    "SeriesKind",
    "VIX_FALLBACK",
    # Exceptions
    "RegimeShiftError",
    "DataDownloadError",
    "DataAlignmentError",
    "DataValidationError",
    "FeatureEngineeringError",
    # Validation
    "validate_price_data",
    "check_monotonic_index",
    "check_no_duplicates",
    # Data pipeline
    "download_market_data",
    "load_market_data_csv",
    "save_market_data_csv",
    # Feature pipeline
    "compute_raw_features",
    "drop_feature_warmup",
    "fit_feature_scaler",
    "transform_features",
    "fit_transform_training_features",
]
