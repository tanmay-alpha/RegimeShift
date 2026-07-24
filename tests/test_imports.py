"""
Test that all clean official submission modules import successfully.
"""

import pytest


def test_official_modules_import():
    """Verify that all core regime_shift modules can be imported without errors."""
    import regime_shift
    import regime_shift.config
    import regime_shift.data
    import regime_shift.features
    import regime_shift.regime_model
    import regime_shift.portfolio
    import regime_shift.backtest
    import regime_shift.benchmarks
    import regime_shift.metrics
    import regime_shift.validation

    assert regime_shift.__version__ is not None
    assert regime_shift.RegimeShiftConfig is not None
    assert regime_shift.validate_price_data is not None
