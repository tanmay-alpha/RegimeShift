"""
Performance metrics for portfolio evaluation.

Computes standard risk-adjusted return and risk metrics (Sharpe ratio, Sortino ratio,
Maximum Drawdown, Calmar ratio, portfolio turnover, transaction cost drag).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """
    Container for portfolio performance metrics.

    All return metrics use daily arithmetic returns.  Annualization uses 252
    trading days unless overridden.
    """

    # Return metrics
    total_return: Optional[float] = None
    cagr: Optional[float] = None
    annualized_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    maximum_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None

    # Turnover and costs
    total_turnover: Optional[float] = None
    annualized_turnover: Optional[float] = None
    total_transaction_cost_drag: Optional[float] = None

    # Metadata
    n_observations: int = 0
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    risk_free_rate: float = 0.0
    annualization_factor: int = 252

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert metrics to a flat dictionary."""
        return {
            "Total Return": _fmt(self.total_return),
            "CAGR": _fmt(self.cagr),
            "Annualised Volatility": _fmt(self.annualized_volatility),
            "Sharpe": _fmt(self.sharpe_ratio),
            "Sortino": _fmt(self.sortino_ratio),
            "Maximum Drawdown": _fmt(self.maximum_drawdown),
            "Calmar": _fmt(self.calmar_ratio),
            "Total Turnover": _fmt(self.total_turnover),
            "Annualised Turnover": _fmt(self.annualized_turnover),
            "Transaction Cost Drag": _fmt(self.total_transaction_cost_drag),
        }


def _fmt(val: Optional[float]) -> Optional[float]:
    """Format a metric value: round to 4 decimal places, or None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 4)


def _annualize_daily_rate(daily_rate: float, annual_factor: int) -> float:
    """Convert a daily rate to annualized using compounding."""
    return (1 + daily_rate) ** annual_factor - 1


def compute_performance_metrics(
    returns: pd.Series,
    turnover: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> PerformanceMetrics:
    """
    Compute comprehensive portfolio performance metrics.

    Args:
        returns: Daily arithmetic return series (net of transaction costs).
        turnover: Optional daily turnover series (used for total/ann. turnover
            and transaction cost drag).  If None, turnover metrics are NaN.
        risk_free_rate: Annualized risk-free rate (default 0.0).
        annualization_factor: Trading days per year (default 252).

    Returns:
        PerformanceMetrics dataclass with all computed metrics.

    Raises:
        ValueError: If returns is empty or contains all NaN values.
    """
    # Input validation
    if returns is None or len(returns) == 0:
        raise ValueError("returns series is empty.")
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        raise ValueError("returns series contains only NaN values.")

    # Clean and sort
    returns = returns.sort_index()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) == 0:
        raise ValueError("returns series contains only infinite values.")

    n = len(returns)
    daily_rf = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    excess = returns - daily_rf

    # Equity curve
    equity_curve = (1 + returns).cumprod()

    # Total return
    total_return = float(equity_curve.iloc[-1] - 1)

    # CAGR
    if total_return <= -1.0:
        cagr = float("-inf")
    else:
        cagr = float(equity_curve.iloc[-1] ** (annualization_factor / n) - 1)

    # Annualized volatility
    std_dev = float(returns.std(ddof=1))
    ann_vol = std_dev * np.sqrt(annualization_factor)

    # Sharpe ratio
    mean_excess = float(excess.mean())
    std_excess = float(excess.std(ddof=1))
    if std_excess > 0:
        sharpe = mean_excess / std_excess * np.sqrt(annualization_factor)
    else:
        sharpe = float("nan")

    # Sortino ratio (downside deviation)
    downside = returns[returns < 0]
    if len(downside) > 0:
        downside_dev = float(np.sqrt((downside ** 2).mean()))
        if downside_dev > 0:
            sortino = mean_excess / downside_dev * np.sqrt(annualization_factor)
        else:
            sortino = float("nan")
    else:
        sortino = float("nan")

    # Maximum drawdown
    cummax = equity_curve.cummax()
    drawdown = equity_curve / cummax - 1
    max_dd = float(drawdown.min())  # negative value

    # Calmar ratio
    if max_dd < 0:
        calmar = cagr / abs(max_dd) if not np.isinf(cagr) else float("nan")
    else:
        calmar = float("nan")

    # Turnover metrics
    total_turnover = float("nan")
    ann_turnover = float("nan")
    total_cost_drag = float("nan")
    if turnover is not None:
        turnover = pd.Series(turnover).fillna(0.0)
        total_turnover = float(turnover.sum())
        ann_turnover = total_turnover * annualization_factor / n
        total_cost_drag = float(turnover.sum())  # in turnover units

    start_date = returns.index[0] if len(returns) > 0 else None
    end_date = returns.index[-1] if len(returns) > 0 else None

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        maximum_drawdown=max_dd,
        calmar_ratio=calmar,
        total_turnover=total_turnover,
        annualized_turnover=ann_turnover,
        total_transaction_cost_drag=total_cost_drag,
        n_observations=n,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
    )


def compute_benchmark_returns(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    rebalance_frequency: int = 21,
    transaction_cost_bps: float = 5.0,
    annualization_factor: int = 252,
) -> pd.Series:
    """
    Compute benchmark returns with periodic rebalancing and transaction costs.

    Args:
        prices: DataFrame of asset prices with DatetimeIndex.
        weights: Dict mapping asset column names to target weights.
        rebalance_frequency: Days between rebalances.
        transaction_cost_bps: Transaction cost in basis points.
        annualization_factor: Trading days per year.

    Returns:
        pd.Series of net daily returns.
    """
    assets = list(weights.keys())
    w = np.array([weights[a] for a in assets])
    price_subset = prices[assets].copy()
    returns = price_subset.pct_change().iloc[1:]

    dates = returns.index
    n = len(dates)
    current_weights = w.copy()
    net_returns = np.zeros(n)

    cost_rate = transaction_cost_bps / 10000.0

    for i in range(n):
        gross_ret = float(np.dot(current_weights, returns.iloc[i].values))

        if i % rebalance_frequency == 0:
            turnover = 0.5 * np.sum(np.abs(current_weights - w))
            cost = turnover * cost_rate
            net_returns[i] = (1 - cost) * (1 + gross_ret) - 1
            current_weights = w * (1 + returns.iloc[i].values)
            current_weights = current_weights / current_weights.sum()
        else:
            net_returns[i] = gross_ret
            current_weights = current_weights * (1 + returns.iloc[i].values)
            current_weights = current_weights / current_weights.sum()

    return pd.Series(net_returns, index=dates)
