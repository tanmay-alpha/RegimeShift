"""
RegimeShift: Market-Regime Identification & Dynamic Asset Allocation Framework.

Designed for IIT Bombay Summer Quant 2026 assignment.
"""

from regime_shift.config import RegimeShiftConfig, TickerConfig, DataConfig
from regime_shift.exceptions import (
    RegimeShiftError,
    DataDownloadError,
    DataAlignmentError,
    DataValidationError,
)
from regime_shift.validation import validate_price_data, check_monotonic_index, check_no_duplicates
from regime_shift.data import download_market_data, load_market_data_csv, save_market_data_csv

__version__ = "0.3.0-data-pipeline"

__all__ = [
    # Config
    "RegimeShiftConfig",
    "TickerConfig",
    "DataConfig",
    # Exceptions
    "RegimeShiftError",
    "DataDownloadError",
    "DataAlignmentError",
    "DataValidationError",
    # Validation
    "validate_price_data",
    "check_monotonic_index",
    "check_no_duplicates",
    # Data pipeline
    "download_market_data",
    "load_market_data_csv",
    "save_market_data_csv",
]
