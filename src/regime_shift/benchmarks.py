"""
Benchmark portfolio computation interface.

Calculates static 60/40 (Equity/Bond) and Equal-Weight benchmark performance
series under identical rebalancing dates and transaction cost rules.
"""

from typing import Dict, List, Optional
import pandas as pd


def compute_static_60_40(
    prices: pd.DataFrame,
    equity_col: str = "equity",
    bond_col: str = "bond",
    rebalance_freq: int = 21,
    cost_bps: float = 5.0
) -> pd.Series:
    """
    Compute cumulative return series for static 60% Equity / 40% Bond benchmark.

    Raises:
        NotImplementedError: Static 60/40 benchmark will be implemented in the next phase.
    """
    raise NotImplementedError(
        "Static 60/40 benchmark calculation will be implemented in the next phase."
    )


def compute_equal_weight(
    prices: pd.DataFrame,
    asset_cols: Optional[List[str]] = None,
    rebalance_freq: int = 21,
    cost_bps: float = 5.0
) -> pd.Series:
    """
    Compute cumulative return series for Equal-Weight multi-asset benchmark.

    Raises:
        NotImplementedError: Equal-weight benchmark will be implemented in the next phase.
    """
    raise NotImplementedError(
        "Equal-weight benchmark calculation will be implemented in the next phase."
    )
