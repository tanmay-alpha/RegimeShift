"""
Configuration module for the RegimeShift market-regime asset allocation framework.

Contains neutral configuration parameters and dataclasses for walk-forward
regime detection, portfolio optimization, and backtesting on multi-asset market data.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RegimeShiftConfig:
    """
    Configuration parameters for the RegimeShift pipeline.

    Attributes:
        random_seed: Seed for random number generators ensuring reproducibility.
        number_of_regimes: Number of market regimes to fit (e.g. 3 for Bull, Bear, Crisis).
        train_window: Rolling window size in trading days for walk-forward training.
        rebalance_frequency: Portfolio rebalancing frequency in trading days.
        transaction_cost_bps: Fixed transaction cost in basis points (1 bps = 0.0001).
        annualization_factor: Number of trading days per year for Indian equity/bond markets (252).
        minimum_training_observations: Minimum required observations before first regime inference.
        equity_col: Column name for equity asset (e.g., NSE NIFTY 50).
        gold_col: Column name for gold asset.
        bond_col: Column name for sovereign bond asset.
        vix_col: Optional column name for market volatility index (India VIX).
    """

    random_seed: int = 42
    number_of_regimes: int = 3
    train_window: int = 252
    rebalance_frequency: int = 21
    transaction_cost_bps: float = 5.0
    annualization_factor: int = 252
    minimum_training_observations: int = 126

    equity_col: str = "equity"
    gold_col: str = "gold"
    bond_col: str = "bond"
    vix_col: str = "vix"

    @property
    def core_assets(self) -> List[str]:
        """Return list of mandatory asset column names."""
        return [self.equity_col, self.gold_col, self.bond_col]
