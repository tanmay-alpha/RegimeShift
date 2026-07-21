"""
Benchmark strategies for comparing against RegimeShift.

Each benchmark uses the SAME transaction cost model for fair comparison.
Includes: Buy-and-Hold, Equal-Weight, Risk Parity, Simple Momentum.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import BacktestResult, WalkForwardBacktest
from .transaction_costs import TransactionCostModel

logger = logging.getLogger(__name__)


def run_benchmarks(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    cost_model: Optional[TransactionCostModel] = None,
    rebalance_freq: int = 21,
    lookback: int = 252,
) -> dict[str, BacktestResult]:
    """
    Run all benchmark strategies.

    Returns dict: strategy_name -> BacktestResult
    """
    cost_model = cost_model or TransactionCostModel()
    tickers = returns.columns.tolist()

    results: dict[str, BacktestResult] = {}

    # 1. Buy and Hold
    results["BuyAndHold"] = _run_buy_and_hold(returns, cost_model, tickers)

    # 2. Equal Weight (periodic rebalance)
    results["EqualWeight"] = _run_equal_weight(returns, cost_model, tickers, rebalance_freq)

    # 3. Risk Parity (inverse-vol, periodic rebalance)
    results["RiskParity"] = _run_risk_parity(returns, cost_model, tickers, rebalance_freq, lookback)

    # 4. Simple Momentum (long top 2 by 3-month momentum)
    results["Momentum"] = _run_momentum(returns, cost_model, tickers, rebalance_freq, lookback)

    return results


def _run_buy_and_hold(
    returns: pd.DataFrame,
    cost_model: TransactionCostModel,
    tickers: list,
) -> BacktestResult:
    """Buy and hold: equal weight, never rebalance."""
    n_assets = len(tickers)
    weights = np.ones(n_assets) / n_assets
    weights_history = pd.DataFrame(
        np.tile(weights, (len(returns), 1)),
        index=returns.index,
        columns=tickers,
    )
    port_rets = (returns.values @ weights)
    costs = pd.Series(0.0, index=returns.index, name="transaction_cost")

    return BacktestResult(
        portfolio_returns=pd.Series(port_rets, index=returns.index, name="portfolio_return"),
        regime_series=pd.Series(["N/A"] * len(returns), index=returns.index, name="regime"),
        weights_history=weights_history,
        costs=costs,
        trade_log=[],
        regime_changes=0,
    )


def _run_equal_weight(
    returns: pd.DataFrame,
    cost_model: TransactionCostModel,
    tickers: list,
    rebalance_freq: int,
) -> BacktestResult:
    """Equal weight, rebalanced every rebalance_freq days."""
    n_assets = len(tickers)
    target = np.ones(n_assets) / n_assets
    current = target.copy()
    weights_history = []
    port_rets = []
    costs = []
    trade_log = []

    for t in range(len(returns)):
        weights_history.append(current.copy())
        port_rets.append(float(np.dot(current, returns.iloc[t].values)))

        if t > 0 and t % rebalance_freq == 0:
            turnover = TransactionCostModel.compute_turnover(current, target)
            if turnover > 1e-6:
                vol = returns.iloc[max(0, t - 63):t].std().values * np.sqrt(252)
                cost_frac = cost_model.cost_as_fraction(current, target, vol, tickers)
                cost_frac = min(cost_frac, 0.02)
                costs.append(cost_frac)
                trade_log.append({
                    "date": str(returns.index[t]),
                    "turnover": float(turnover),
                    "cost": float(cost_frac),
                })
                current = target.copy()
            else:
                costs.append(0.0)
        else:
            costs.append(0.0)

    return BacktestResult(
        portfolio_returns=pd.Series(port_rets, index=returns.index, name="portfolio_return"),
        regime_series=pd.Series(["N/A"] * len(returns), index=returns.index, name="regime"),
        weights_history=pd.DataFrame(weights_history, index=returns.index, columns=tickers),
        costs=pd.Series(costs, index=returns.index, name="transaction_cost"),
        trade_log=trade_log,
        regime_changes=0,
    )


def _run_risk_parity(
    returns: pd.DataFrame,
    cost_model: TransactionCostModel,
    tickers: list,
    rebalance_freq: int,
    lookback: int,
) -> BacktestResult:
    """Inverse-volatility weighted, rebalanced periodically."""
    n_assets = len(tickers)
    current = np.ones(n_assets) / n_assets
    weights_history = []
    port_rets = []
    costs = []
    trade_log = []

    for t in range(len(returns)):
        weights_history.append(current.copy())
        port_rets.append(float(np.dot(current, returns.iloc[t].values)))

        if t > 0 and t % rebalance_freq == 0 and t >= 63:
            window = returns.iloc[max(0, t - lookback):t]
            vol = window.std().values * np.sqrt(252)
            vol = np.maximum(vol, 0.01)
            inv_vol = 1.0 / vol
            target = inv_vol / inv_vol.sum()

            turnover = TransactionCostModel.compute_turnover(current, target)
            if turnover > 1e-6:
                cost_frac = cost_model.cost_as_fraction(current, target, vol, tickers)
                cost_frac = min(cost_frac, 0.02)
                costs.append(cost_frac)
                trade_log.append({
                    "date": str(returns.index[t]),
                    "turnover": float(turnover),
                    "cost": float(cost_frac),
                })
                current = target.copy()
            else:
                costs.append(0.0)
        else:
            costs.append(0.0)

    return BacktestResult(
        portfolio_returns=pd.Series(port_rets, index=returns.index, name="portfolio_return"),
        regime_series=pd.Series(["N/A"] * len(returns), index=returns.index, name="regime"),
        weights_history=pd.DataFrame(weights_history, index=returns.index, columns=tickers),
        costs=pd.Series(costs, index=returns.index, name="transaction_cost"),
        trade_log=trade_log,
        regime_changes=0,
    )


def _run_momentum(
    returns: pd.DataFrame,
    cost_model: TransactionCostModel,
    tickers: list,
    rebalance_freq: int,
    lookback: int,
) -> BacktestResult:
    """Simple momentum: long top 2 assets by 3-month momentum, equal weight."""
    n_assets = len(tickers)
    current = np.ones(n_assets) / n_assets
    weights_history = []
    port_rets = []
    costs = []
    trade_log = []

    for t in range(len(returns)):
        weights_history.append(current.copy())
        port_rets.append(float(np.dot(current, returns.iloc[t].values)))

        if t > 0 and t % rebalance_freq == 0 and t >= 63:
            window = returns.iloc[max(0, t - 63):t]
            momentum = (1 + window).prod() - 1
            target = np.zeros(n_assets)
            top = np.argsort(momentum.values)[-2:]
            target[top] = 0.5

            turnover = TransactionCostModel.compute_turnover(current, target)
            if turnover > 1e-6:
                vol = returns.iloc[max(0, t - 63):t].std().values * np.sqrt(252)
                cost_frac = cost_model.cost_as_fraction(current, target, vol, tickers)
                cost_frac = min(cost_frac, 0.02)
                costs.append(cost_frac)
                trade_log.append({
                    "date": str(returns.index[t]),
                    "turnover": float(turnover),
                    "cost": float(cost_frac),
                })
                current = target.copy()
            else:
                costs.append(0.0)
        else:
            costs.append(0.0)

    return BacktestResult(
        portfolio_returns=pd.Series(port_rets, index=returns.index, name="portfolio_return"),
        regime_series=pd.Series(["N/A"] * len(returns), index=returns.index, name="regime"),
        weights_history=pd.DataFrame(weights_history, index=returns.index, columns=tickers),
        costs=pd.Series(costs, index=returns.index, name="transaction_cost"),
        trade_log=trade_log,
        regime_changes=0,
    )
