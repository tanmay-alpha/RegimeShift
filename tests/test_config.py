"""
Test configuration parameters and defaults.
"""

from regime_shift.config import RegimeShiftConfig


def test_config_defaults():
    """Verify that configuration parameters meet official IITB assignment requirements."""
    config = RegimeShiftConfig()

    assert config.annualization_factor == 252, (
        f"Annualization factor must be 252 for Indian equity/bond markets, got {config.annualization_factor}"
    )
    assert 5.0 <= config.transaction_cost_bps <= 10.0, (
        f"Transaction costs must be between 5 and 10 bps, got {config.transaction_cost_bps}"
    )
    assert config.number_of_regimes == 3, (
        f"Number of regimes must be 3 (Bull, Bear, Crisis), got {config.number_of_regimes}"
    )
    assert config.equity_col == "equity"
    assert config.gold_col == "gold"
    assert config.bond_col == "bond"
