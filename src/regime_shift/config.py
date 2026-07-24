"""
Configuration module for the RegimeShift market-regime asset allocation framework.

Contains neutral configuration parameters and dataclasses for walk-forward
regime detection, portfolio optimization, and backtesting on multi-asset market data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class SeriesKind(str, Enum):
    """
    Semantic category of a time series used in the pipeline.

    Distinguishes price/NAV series (suitable for direct portfolio use) from
    yield series (rates expressed as percentages) and indicator series (like VIX)
    that require separate handling and must NOT be passed where a price is expected.

    Values
    ------
    PRICE     : Adjusted close / NAV expressed in currency units.
                Positive, mean-reverting around trends, directly comparable as
                portfolio weights denominator. Examples: NIFTY 50, Gold ETF, Gilt ETF.
    YIELD     : Interest rate expressed in percent (e.g. 0–15 %).
                Cannot be used as a portfolio price. A synthetic bond price must
                be computed from yields before entering any P&L calculation.
    INDICATOR : Dimensionless or volatility index (e.g. VIX, INDIAVIX).
                Used as a regime feature only — not a tradeable price.
    """

    PRICE = "price"
    YIELD = "yield"
    INDICATOR = "indicator"


@dataclass
class AssetSpec:
    """
    Full specification for a single market data series.

    Attributes:
        ticker:      yfinance ticker symbol.
        kind:        Semantic category (PRICE / YIELD / INDICATOR).
        description: Human-readable description for documentation.
        currency:    ISO 4217 currency code for PRICE series; None for others.
        notes:       Any caveats, alternatives, or data-quality warnings.
    """

    ticker: str
    kind: SeriesKind
    description: str
    currency: Optional[str] = None
    notes: str = ""

    def require_price(self) -> None:
        """
        Assert that this series is a PRICE series.

        Raises:
            TypeError: If the series is a YIELD or INDICATOR — these must not be
                used directly in portfolio P&L calculations or as a price denominator.
        """
        if self.kind is not SeriesKind.PRICE:
            raise TypeError(
                f"Ticker '{self.ticker}' is a {self.kind.value} series, not a price. "
                "Portfolio calculations require a PRICE series (NAV / adjusted close). "
                f"Notes: {self.notes}"
            )


@dataclass
class TickerConfig:
    """
    Centralised mapping from logical asset roles to fully documented AssetSpec objects.

    Every ticker choice is explained here.  No raw ticker string may appear in any
    module outside this class and _download_single().

    Ticker selection rationale
    --------------------------

    equity
        ^NSEI — NIFTY 50 Index.
        Kind   : PRICE (exchange-adjusted daily close, INR).
        Notes  : The most widely used Indian equity benchmark.  yfinance coverage
                 is reliable from ~2000 onwards.

    gold
        GOLDBEES.NS — Nippon India ETF Gold Bees (NSE-listed).
        Kind    : PRICE (NAV in INR).
        Currency: INR.
        Notes   : Directly INR-denominated; avoids the USD/INR conversion issue of
                 COMEX Gold Futures (GC=F).  Tracks domestic gold prices (MCX).
                 Alternative: SGOLD.NS (SBI Gold ETF) or HDFCMFGETF.NS.
                 IMPORTANT: GC=F is USD-denominated — using it without FX conversion
                 introduces a currency mismatch with INR equity returns.  The GC=F
                 default was retained in Phase 1 as a placeholder; GOLDBEES.NS is
                 the preferred default from Phase 1 hardening onwards.

    bond
        0P0001BVE8.BO — SBI Magnum Gilt Fund – Regular Plan (BSE mutual fund NAV).
        Kind    : PRICE (NAV in INR, adjusted for distributions).
        Currency: INR.
        Notes   : A liquid Indian government gilt mutual fund with long yfinance
                 history.  Alternative government-gilt ETFs: GSEC10RBETF.NS (Mirae),
                 CPSEETF.NS, or Bharat Bond ETF series.
                 ^IRX (13-week T-Bill yield) was used in Phase 1 but is a YIELD
                 series — it CANNOT be used as a portfolio price.  It is retained in
                 AssetSpec with kind=YIELD for reference.

    vix
        ^INDIAVIX — NSE India Volatility Index (India VIX).
        Kind    : INDICATOR.
        Notes   : Preferred volatility indicator for Indian markets.  yfinance
                 coverage exists from ~2008.  If a download fails (e.g. offline
                 or restricted access), the explicit fallback is ^VIX (CBOE).
                 Callers MUST explicitly pass vix_ticker=VIX_FALLBACK to switch;
                 there is NO silent automatic fallback.

    Gold currency policy (documented here; no feature conversion implemented yet)
    ------------------------------------------------------------------------------
    If GC=F (USD) is used instead of GOLDBEES.NS (INR), the pipeline must apply
    an INR/USD FX rate before computing returns.  This conversion is NOT yet
    implemented.  Using GC=F without FX conversion will produce cross-currency
    contamination in correlation and return features.  This is flagged as a
    known limitation until the feature-engineering phase implements FX adjustment.
    """

    # Preferred defaults — all INR-denominated price series
    equity: AssetSpec = field(default_factory=lambda: AssetSpec(
        ticker="^NSEI",
        kind=SeriesKind.PRICE,
        description="NIFTY 50 Index — NSE large-cap benchmark",
        currency="INR",
        notes="Exchange-adjusted daily close. yfinance history from ~2000.",
    ))

    gold: AssetSpec = field(default_factory=lambda: AssetSpec(
        ticker="GOLDBEES.NS",
        kind=SeriesKind.PRICE,
        description="Nippon India ETF Gold Bees — INR-denominated gold ETF (NSE)",
        currency="INR",
        notes=(
            "Tracks MCX gold prices in INR. Preferred over GC=F (USD) to avoid "
            "FX mismatch with INR equity returns. "
            "Gold currency policy: GC=F requires explicit USD→INR FX conversion "
            "before computing returns; not yet implemented."
        ),
    ))

    bond: AssetSpec = field(default_factory=lambda: AssetSpec(
        ticker="0P0001BVE8.BO",
        kind=SeriesKind.PRICE,
        description="SBI Magnum Gilt Fund – Regular Plan (BSE NAV, INR)",
        currency="INR",
        notes=(
            "Liquid Indian government gilt mutual fund. NAV series suitable as "
            "portfolio bond proxy. Alternative: GSEC10RBETF.NS (Mirae ETF). "
            "Do NOT substitute with ^IRX (T-Bill yield) — that is a YIELD series."
        ),
    ))

    vix: AssetSpec = field(default_factory=lambda: AssetSpec(
        ticker="^INDIAVIX",
        kind=SeriesKind.INDICATOR,
        description="India VIX — NSE implied volatility index (preferred)",
        notes=(
            "Preferred VIX for Indian market regime detection. Coverage from ~2008. "
            "Explicit fallback: use VIX_FALLBACK constant or override vix_ticker. "
            "There is NO silent automatic fallback to CBOE VIX."
        ),
    ))

    # Reference-only: yield series — NOT suitable as portfolio price
    irx_yield: AssetSpec = field(default_factory=lambda: AssetSpec(
        ticker="^IRX",
        kind=SeriesKind.YIELD,
        description="13-Week US T-Bill Yield (%) — reference only, NOT a price series",
        notes=(
            "A yield expressed in percent (0–10 range). Must NOT be used as a "
            "portfolio price. Retained for reference; was incorrectly used as "
            "bond_ticker in Phase 1 draft."
        ),
    ))

    # Explicit VIX fallback — must be set intentionally, never silently
    cboe_vix_fallback: AssetSpec = field(default_factory=lambda: AssetSpec(
        ticker="^VIX",
        kind=SeriesKind.INDICATOR,
        description="CBOE VIX — US volatility index (explicit fallback only)",
        notes=(
            "Use only when ^INDIAVIX is unavailable and an explicit fallback is "
            "required. Must be set via vix_ticker override — never used automatically."
        ),
    ))

    # Convenience accessors
    @property
    def equity_ticker(self) -> str:
        return self.equity.ticker

    @property
    def gold_ticker(self) -> str:
        return self.gold.ticker

    @property
    def bond_ticker(self) -> str:
        self.bond.require_price()  # guard: raises if bond spec is swapped to a yield
        return self.bond.ticker

    @property
    def vix_ticker(self) -> str:
        return self.vix.ticker

    def get_required_price_specs(self) -> Dict[str, AssetSpec]:
        """Return AssetSpec objects for the three required price assets."""
        return {
            "equity": self.equity,
            "gold": self.gold,
            "bond": self.bond,
        }

    def validate_price_assets(self) -> None:
        """
        Verify that all required asset specs are PRICE series.

        Raises:
            TypeError: If any required asset is a YIELD or INDICATOR spec.
        """
        for col, spec in self.get_required_price_specs().items():
            spec.require_price()


# Sentinel constant: explicit CBOE VIX fallback ticker
VIX_FALLBACK: str = "^VIX"


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
        tickers: Centralised ticker/asset-spec configuration.
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
            self.tickers.gold_ticker:   self.gold_col,
            self.tickers.bond_ticker:   self.bond_col,
            self.tickers.vix_ticker:    self.vix_col,
        }

    @property
    def col_to_ticker(self) -> Dict[str, str]:
        """Return mapping of output column name → yfinance ticker."""
        return {v: k for k, v in self.ticker_to_col.items()}

    def validate(self) -> None:
        """
        Validate the entire config, including asset-spec semantics.

        Raises:
            TypeError: If any required asset spec is not a PRICE series.
        """
        self.tickers.validate_price_assets()
