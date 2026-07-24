"""
RegimeShift: Market-Regime Identification & Dynamic Asset Allocation Framework.

Designed for IIT Bombay Summer Quant 2026 assignment.
"""

from regime_shift.config import RegimeShiftConfig
from regime_shift.validation import validate_price_data, check_monotonic_index, check_no_duplicates

__version__ = "0.2.0-foundation"

__all__ = [
    "RegimeShiftConfig",
    "validate_price_data",
    "check_monotonic_index",
    "check_no_duplicates",
]
