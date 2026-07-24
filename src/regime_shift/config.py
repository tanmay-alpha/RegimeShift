"""
Configuration module for the RegimeShift market-regime asset allocation framework.

Contains neutral configuration parameters and dataclasses for walk-forward
regime detection, portfolio optimization, and backtesting on multi-asset market data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TickerConfig:
    """
    Centralised mapping from logical asset names to yfinance ticker symbols.

    All ticker choices are documented here so they never appear hardcoded
    across the wider codebase.

    Ticker selection rationale:
      equity : ^NSEI  – NIFTY 50 Index (NSE benchmark; daily adjusted by exchange)
      gold   : GC=F   – COMEX Gold Futures (liquid USD-priced proxy; widely used
                         for Indian gold exposure in academic studies)
      bond   : IGLT.L – iShares UK Gilts ETF used as a sovereign-bond proxy.
                         For a purely India-specific pipeline, replace with
                         "0P00009OSF.BO" (SBI Magnum Gilt) or a direct NSE bond
                         index once yfinance coverage is confirmed.
      vix    : ^VIX   – CBOE VIX used as a volatility regime indicator when
                         India VIX (^INDIAVIX) download is unavailable offline.
                         Swap to "^INDIAVIX" when internet access is available.

    All values are overridable at runtime via environment or config injection.
    """

    equity_ticker: str = "^NSEI"
    gold_ticker: str = "GC=F"
    bond_ticker: str = "^IRX"   # 13-week T-Bill rate as yield proxy; swap as needed
    vix_ticker: str = "^VIX"


@dataclass
class DataConfig:
    """
    Configuration parameters for market data acquisition and validation.

    Attributes:
        default_start: Default start date string (YYYY-MM-DD) for history downloads.
        default_end: Default end date string or None (uses today).
        forward_fill_limit: Maximum consecutive trading days to forward-fill prices.
            Set to 0 to disable forward-filling entirely.
        cache_dir: Directory for local CSV/Parquet caches. None disables caching.
        price_field: yfinance column to use as the canonical price series.
    """

    default_start: str = "2010-01-01"
    default_end: Optional[str] = None
    forward_fill_limit: int = 3
    cache_dir: Optional[str] = "data/cache"
    price_field: str = "Close"


@dataclass
class RegimeShiftConfig:
    """
    Configuration parameters for the RegimeShift pipeline.

    Attributes:
        random_seed: Seed for random number generators ensuring reproducibility.
        number_of_regimes: Number of market regimes (3: Bull, Bear, Crisis).
        train_window: Rolling window size in trading days for walk-forward training.
        rebalance_frequency: Portfolio rebalancing frequency in trading days.
        transaction_cost_bps: Fixed transaction cost in basis points (1 bps = 0.0001).
        annualization_factor: Trading days per year for Indian equity/bond markets (252).
        minimum_training_observations: Minimum observations before first HMM inference.
        equity_col: Output column name for equity prices.
        gold_col: Output column name for gold prices.
        bond_col: Output column name for bond prices.
        vix_col: Output column name for VIX (optional).
        tickers: Centralised ticker configuration.
        data: Data acquisition configuration.
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

    tickers: TickerConfig = field(default_factory=TickerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @property
    def core_assets(self) -> List[str]:
        """Return list of mandatory output column names."""
        return [self.equity_col, self.gold_col, self.bond_col]

    @property
    def ticker_to_col(self) -> Dict[str, str]:
        """Return mapping of yfinance ticker → output column name."""
        return {
            self.tickers.equity_ticker: self.equity_col,
            self.tickers.gold_ticker: self.gold_col,
            self.tickers.bond_ticker: self.bond_col,
            self.tickers.vix_ticker: self.vix_col,
        }

    @property
    def col_to_ticker(self) -> Dict[str, str]:
        """Return mapping of output column name → yfinance ticker."""
        return {v: k for k, v in self.ticker_to_col.items()}
