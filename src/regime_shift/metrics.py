"""
Performance metrics for portfolio evaluation.

Computes standard risk-adjusted return and risk metrics (Sharpe ratio, Sortino ratio,
Maximum Drawdown, Calmar ratio, portfolio turnover, transaction cost drag).

Transaction cost drag definition
---------------------------------
    transaction_cost_drag = gross_final - net_final
where
    gross_final = (1 + gross_returns).prod()
    net_final   = (1 + net_returns).prod()

This is the exact compounding-based drag (in wealth terms), not a proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for portfolio performance metrics."""

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
            "n_observations": _fmt_value(self.n_observations),
            "risk_free_rate": _fmt_value(self.risk_free_rate),
            "annualization_factor": _fmt_value(self.annualization_factor),
        }


def _fmt(val: Optional[float]) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 4)


def _fmt_value(val) -> object:
    """Round floats; preserve ints and other types unchanged."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float):
        return round(float(val), 4)
    return val


def compute_performance_metrics(
    returns: pd.Series,
    gross_returns: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
    transaction_costs: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> PerformanceMetrics:
    """
    Compute comprehensive portfolio performance metrics.

    Args:
        returns: Daily arithmetic net return series (after transaction costs).
        gross_returns: Daily gross return series (before transaction costs).
            Required for transaction-cost drag computation.  Must be aligned
            to returns.
        turnover: Optional daily turnover series aligned to returns.
        transaction_costs: Optional daily transaction-cost series aligned to
            returns.
        risk_free_rate: Annualized risk-free rate (default 0.0).
        annualization_factor: Trading days per year (default 252).

    Returns:
        PerformanceMetrics dataclass.

    Raises:
        ValueError: If returns is empty, contains NaN, inf, or returns <= -1.
        ValueError: If turnover or transaction_costs indices don't align with
            returns.
        ValueError: If gross_returns is required but not provided for cost-drag.
    """
    # Convert inputs
    returns = pd.Series(returns).sort_index()

    if len(returns) == 0:
        raise ValueError("returns series is empty.")

    # Coerce to numeric so empty/object dtypes don't break np.isfinite below.
    returns = returns.astype(float)

    if returns.isna().any():
        raise ValueError(
            "returns series contains NaN. Clean or drop missing values first."
        )

    has_nonfinite = ~np.isfinite(returns.values)
    if has_nonfinite.any():
        idx = returns.index[has_nonfinite]
        raise ValueError(
            f"returns series contains non-finite values at {list(idx[:5])}. "
            "Check for infinite or missing returns."
        )

    if (returns <= -1.0).any():
        raise ValueError(
            "returns series contains values <= -1, which would produce "
            "non-positive portfolio wealth. Check return calculation."
        )

    # Index alignment checks
    for name, series in [("turnover", turnover), ("transaction_costs", transaction_costs), ("gross_returns", gross_returns)]:
        if series is not None:
            s = pd.Series(series).sort_index()
            if not s.index.equals(returns.index):
                raise ValueError(
                    f"{name} index does not align with returns index. "
                    "All series must cover the same dates."
                )

    n = len(returns)
    daily_rf = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    excess = returns - daily_rf

    # Equity curve (always (1+returns).cumprod, never returns.cumprod)
    equity_curve = (1 + returns).cumprod()

    # Total return
    total_return = float(equity_curve.iloc[-1] - 1)

    # CAGR
    cagr = float(equity_curve.iloc[-1] ** (annualization_factor / n) - 1)

    # Annualized volatility
    std_dev = float(returns.std(ddof=1))
    ann_vol = std_dev * np.sqrt(annualization_factor)

    # Sharpe ratio.  A constant positive excess-return sequence does not have
    # a defined ratio; returning zero would silently misstate it as no reward.
    epsilon = 1e-12
    mean_excess = float(excess.mean())
    std_excess = float(excess.std(ddof=1))
    if not np.isfinite(std_excess):
        sharpe = float("nan")
    elif abs(std_excess) <= epsilon:
        sharpe = 0.0 if abs(mean_excess) <= epsilon else float("nan")
    else:
        sharpe = mean_excess / std_excess * np.sqrt(annualization_factor)

    # Sortino ratio (downside deviation of excess returns)
    downside_excess = excess[excess < 0]
    if len(downside_excess) > 0:
        downside_dev = float(np.sqrt((downside_excess ** 2).mean()))
        if downside_dev <= epsilon:
            sortino = 0.0 if abs(mean_excess) <= epsilon else float("nan")
        else:
            sortino = mean_excess / downside_dev * np.sqrt(annualization_factor)
    else:
        sortino = 0.0 if abs(mean_excess) <= epsilon else float("nan")

    # Maximum drawdown (positive 0â€“1 value)
    cummax = equity_curve.cummax()
    drawdown = equity_curve / cummax - 1
    max_dd = float(abs(drawdown.min()))
    max_dd = max(0.0, min(1.0, max_dd))

    # Calmar ratio
    calmar = float("nan")
    if max_dd > 1e-12 and not np.isinf(cagr):
        calmar = cagr / max_dd

    # Turnover
    total_turnover = float("nan")
    ann_turnover = float("nan")
    if turnover is not None:
        turnover_s = pd.Series(turnover).fillna(0.0)
        total_turnover = float(turnover_s.sum())
        ann_turnover = total_turnover * annualization_factor / n

    # Transaction cost drag = gross_final - net_final
    total_cost_drag = float("nan")
    if gross_returns is not None:
        gross_s = pd.Series(gross_returns).sort_index()
        gross_final = float((1.0 + gross_s).prod())
        net_final = float(equity_curve.iloc[-1])
        total_cost_drag = gross_final - net_final

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
