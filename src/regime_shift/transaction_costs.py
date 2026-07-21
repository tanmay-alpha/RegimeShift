"""
transaction_costs.py — Realistic transaction cost model for Indian equity/ETF trading.

Uses Almgren-Chriss market impact model with slippage and commission components.
Costs are computed per rebalance event based on turnover.

References:
  - Almgren, R. & Chriss, N.. Optimal execution of portfolio transactions.
    Journal of Risk, 3(2), 5-39.
  - Square-root law: Impact ∝ (participation_rate)^0.5
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """
    Models realistic transaction costs for Indian equity/ETF trading.

    Uses Almgren-Chriss market impact model with slippage and commission.
    Costs are computed per rebalance event based on turnover.

    Attributes:
        commission_rate: Brokerage rate (fraction of notional)
        fixed_commission: Per-trade fixed cost (INR)
        slippage_factor: Empirical slippage constant
        impact_factor: Market impact constant
        impact_exponent: Square-root law exponent (0.5)
        adv_estimates: Estimated ADV per asset (shares/day)
        participation_rate: Assumed participation rate for backtest
    """

    COMMISSION_RATE: float = 0.0003
    FIXED_COMMISSION: float = 20.0
    SLIPPAGE_FACTOR: float = 0.1
    IMPACT_FACTOR: float = 0.1
    IMPACT_EXPONENT: float = 0.5

    ADV_ESTIMATES: dict = {
        "Nifty": 1_000_000,
        "Gold": 500_000,
        "Bond": 200_000,
    }

    ASSET_LABELS: dict = {
        "^NSEI": "Nifty",
        "NIFTY": "Nifty",
        "GC=F": "Gold",
        "GOLD": "Gold",
        "TLT": "Bond",
        "BOND": "Bond",
    }

    def __init__(
        self,
        commission_rate: Optional[float] = None,
        fixed_commission: Optional[float] = None,
        slippage_factor: Optional[float] = None,
        impact_factor: Optional[float] = None,
        impact_exponent: Optional[float] = None,
        adv_overrides: Optional[dict] = None,
    ) -> None:
        self.commission_rate = commission_rate if commission_rate is not None else self.COMMISSION_RATE
        self.fixed_commission = fixed_commission if fixed_commission is not None else self.FIXED_COMMISSION
        self.slippage_factor = slippage_factor if slippage_factor is not None else self.SLIPPAGE_FACTOR
        self.impact_factor = impact_factor if impact_factor is not None else self.IMPACT_FACTOR
        self.impact_exponent = impact_exponent if impact_exponent is not None else self.IMPACT_EXPONENT
        self.adv_estimates = dict(self.ADV_ESTIMATES)
        if adv_overrides:
            self.adv_estimates.update(adv_overrides)
        self.participation_rate = 0.05

    def compute_per_asset_cost(
        self,
        weight_change: float,
        volatility: float,
        asset_label: str,
        notional: float,
    ) -> float:
        """
        Compute cost for a single asset trade.

        Total cost = Commission + Slippage + MarketImpact

        Args:
            weight_change: Absolute weight change (|w_new - w_old|)
            volatility: Annualized volatility (fraction, e.g. 0.15)
            asset_label: Asset identifier for ADV lookup
            notional: Total portfolio notional (INR)

        Returns:
            Cost in INR (always >= 0)
        """
        if weight_change < 1e-8:
            return 0.0

        trade_notional = weight_change * notional

        # -- Commission --
        commission = self.fixed_commission + self.commission_rate * trade_notional
        commission = min(commission, trade_notional * 0.01)  # Cap at 1%

        # -- Slippage --
        participation = self.participation_rate
        slippage = self.slippage_factor * volatility * np.sqrt(participation) * trade_notional
        slippage = min(slippage, trade_notional * 0.005)  # Cap at 0.5%

        # -- Market Impact (Almgren-Chriss square-root law) --
        impact = self.impact_factor * np.power(participation, self.impact_exponent) * trade_notional
        impact = min(impact, trade_notional * 0.01)  # Cap at 1%

        total_cost = commission + slippage + impact
        return max(total_cost, 0.0)

    def compute_turnover_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        volatilities: np.ndarray,
        tickers: Optional[list] = None,
        notional: float = 1_000_000,
    ) -> float:
        """
        Compute total transaction cost for a portfolio rebalance.

        Turnover = 0.5 * sum(|w_new[i] - w_old[i]|)

        Args:
            old_weights: (d,) current portfolio weights
            new_weights: (d,) target portfolio weights
            volatilities: (d,) annualized volatility per asset
            tickers: (d,) asset ticker symbols for ADV lookup
            notional: Total portfolio notional (INR)

        Returns:
            Total transaction cost in INR (always >= 0)
        """
        old_weights = np.asarray(old_weights, dtype=np.float64)
        new_weights = np.asarray(new_weights, dtype=np.float64)
        volatilities = np.asarray(volatilities, dtype=np.float64)
        n = len(old_weights)

        total_cost = 0.0
        for i in range(n):
            weight_change = abs(new_weights[i] - old_weights[i])
            label = tickers[i] if tickers and i < len(tickers) else f"asset_{i}"
            asset_label = self.ASSET_LABELS.get(label, label)
            vol = volatilities[i] if i < len(volatilities) else 0.15
            total_cost += self.compute_per_asset_cost(weight_change, vol, asset_label, notional)

        return max(total_cost, 0.0)

    def cost_as_fraction(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        volatilities: np.ndarray,
        tickers: Optional[list] = None,
        notional: float = 1_000_000,
    ) -> float:
        """
        Transaction cost as fraction of portfolio value.

        Args:
            old_weights: (d,) current portfolio weights
            new_weights: (d,) target portfolio weights
            volatilities: (d,) annualized volatility per asset
            tickers: (d,) asset ticker symbols
            notional: Total portfolio notional (INR)

        Returns:
            Cost as fraction (e.g. 0.001 = 10 bps = 0.1%)
        """
        cost_inr = self.compute_turnover_cost(
            old_weights, new_weights, volatilities, tickers, notional
        )
        return cost_inr / notional

    @staticmethod
    def compute_turnover(old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """
        Portfolio turnover: 0.5 * sum(|w_new - w_old|).

        Each trade is counted once (not both buy and sell).

        Args:
            old_weights: (d,) current weights
            new_weights: (d,) target weights

        Returns:
            Turnover as a fraction (0 to 1)
        """
        return float(0.5 * np.abs(new_weights - old_weights).sum())

    def annualized_cost_bps(
        self,
        avg_turnover: float,
        cost_per_rebalance: float,
        rebalance_freq_days: int = 21,
    ) -> float:
        """
        Annualize transaction costs.

        Args:
            avg_turnover: Average portfolio turnover per rebalance (0-1)
            cost_per_rebalance: Average cost per rebalance as fraction of NAV
            rebalance_freq_days: Days between rebalances

        Returns:
            Annualized cost in basis points
        """
        rebalances_per_year = 252.0 / max(rebalance_freq_days, 1)
        annual_cost = cost_per_rebalance * rebalances_per_year
        return annual_cost * 10000  # Convert to bps
