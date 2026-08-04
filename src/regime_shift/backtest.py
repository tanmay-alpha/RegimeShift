"""Event-driven, causally valid close-to-close walk-forward backtests.

At close ``t`` the portfolio that was installed after close ``t-1`` first earns
the ``t-1 -> t`` return.  Only then may a model observe close ``t``, submit a
target, pay execution costs, and install weights that first earn ``t -> t+1``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from regime_shift.benchmarks import equal_weight_weights, static_60_40_weights
from regime_shift.config import RegimeShiftConfig
from regime_shift.execution import ExecutionCostModel, ExecutionModel
from regime_shift.features import compute_raw_features, drop_feature_warmup, fit_feature_scaler, transform_features
from regime_shift.metrics import compute_performance_metrics
from regime_shift.portfolio import optimize_portfolio
from regime_shift.regime_model import fit_hmm, predict_current_state

logger = logging.getLogger(__name__)
_ASSET_ORDER = ["equity", "gold", "bond"]
_REGIME_ORDER = ["Bull", "Bear", "Crisis"]


@dataclass
class BacktestResult:
    gross_returns: pd.Series = field(default_factory=pd.Series, repr=False)
    net_returns: pd.Series = field(default_factory=pd.Series, repr=False)
    transaction_costs: pd.Series = field(default_factory=pd.Series, repr=False)
    turnover: pd.Series = field(default_factory=pd.Series, repr=False)
    target_weights: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    daily_drifted_weights: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    pre_return_weights: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    post_return_pre_trade_weights: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    end_of_day_post_trade_weights: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    regime_series: pd.Series = field(default_factory=pd.Series, repr=False)
    regime_probabilities: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    transition_matrix: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    gross_equity: pd.Series = field(default_factory=pd.Series, repr=False)
    net_equity: pd.Series = field(default_factory=pd.Series, repr=False)
    rebalance_flags: pd.Series = field(default_factory=pd.Series, repr=False)
    event_log: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    metrics: Optional[Dict] = None
    hmm_diagnostics: List[Dict] = field(default_factory=list)
    config_metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    name: str
    gross_returns: pd.Series = field(default_factory=pd.Series, repr=False)
    net_returns: pd.Series = field(default_factory=pd.Series, repr=False)
    gross_equity: pd.Series = field(default_factory=pd.Series, repr=False)
    net_equity: pd.Series = field(default_factory=pd.Series, repr=False)
    turnover: pd.Series = field(default_factory=pd.Series, repr=False)
    transaction_costs: pd.Series = field(default_factory=pd.Series, repr=False)
    rebalance_flags: pd.Series = field(default_factory=pd.Series, repr=False)
    daily_drifted_weights: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    target_weights: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    event_log: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    metrics: Optional[Dict] = None


def _cost_model(config: RegimeShiftConfig, override_bps: Optional[float]) -> ExecutionCostModel:
    if override_bps is not None:
        return ExecutionCostModel.flat(override_bps, config.core_assets, name="cli_override")
    if config.execution_cost_model is not None:
        return config.execution_cost_model
    return ExecutionCostModel.flat(config.transaction_cost_bps, config.core_assets, name="legacy_flat_assumption")


def _drift(weights: np.ndarray, asset_return: np.ndarray, date: pd.Timestamp) -> tuple[float, np.ndarray]:
    gross = float(np.dot(weights, asset_return))
    if not np.any(weights):
        return gross, weights.copy()
    wealth = 1.0 + gross
    if wealth <= 0.0:
        raise ValueError(f"Portfolio wealth became non-positive on {date}.")
    return gross, weights * (1.0 + asset_return) / wealth


def _event_row(date: pd.Timestamp, rebalance: bool, pre_trade: np.ndarray, target: np.ndarray, post_trade: np.ndarray, post_return: np.ndarray) -> Dict:
    row: Dict = {
        "signal_date": date if rebalance else pd.NaT,
        "execution_date": date if rebalance else pd.NaT,
        "return_date": date,
        "rebalance_flag": bool(rebalance),
    }
    for prefix, values in (("pre_return", pre_trade), ("post_return_pre_trade", post_return), ("target", target), ("end_of_day_post_trade", post_trade)):
        for asset, value in zip(_ASSET_ORDER, values):
            row[f"{prefix}_{asset}_weight"] = float(value)
    return row


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    config: Optional[RegimeShiftConfig] = None,
    transaction_cost_bps: Optional[float] = None,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Run the official ``NEXT_CLOSE`` event sequence without same-bar fills."""
    config = copy.deepcopy(config or RegimeShiftConfig())
    config.validate()
    if config.execution_model is not ExecutionModel.NEXT_CLOSE:
        raise ValueError("NEXT_OPEN cannot be simulated from close-only price data.")
    assets = config.core_assets
    price_data = prices[assets].copy()
    asset_returns = price_data.pct_change().iloc[1:]
    feature_prices = price_data.copy()
    has_vix = config.vix_col in prices.columns
    if has_vix:
        feature_prices[config.vix_col] = prices[config.vix_col]
    raw_features = drop_feature_warmup(compute_raw_features(feature_prices, config.feature_config), config=config.feature_config)
    common = raw_features.index.intersection(asset_returns.index)
    raw_features, asset_returns = raw_features.loc[common], asset_returns.loc[common]
    dates = pd.DatetimeIndex(asset_returns.index)
    minimum = config.minimum_training_observations + config.rebalance_frequency
    if len(dates) < minimum:
        raise ValueError(f"Insufficient data after feature warmup: have {len(dates)}, need at least {minimum}.")
    rebalance_positions = set(range(config.rebalance_frequency, len(dates), config.rebalance_frequency))
    if not rebalance_positions:
        rebalance_positions = {len(dates) - 1}
    cost_model = _cost_model(config, transaction_cost_bps)

    gross: List[float] = []
    net: List[float] = []
    costs: List[float] = []
    turns: List[float] = []
    flags: List[bool] = []
    targets: List[np.ndarray] = []
    drifts: List[np.ndarray] = []
    pre_returns: List[np.ndarray] = []
    end_of_days: List[np.ndarray] = []
    regimes: List[str] = []
    probs: List[np.ndarray] = []
    rows: List[Dict] = []
    diag: List[Dict] = []
    current = np.zeros(len(assets))
    allocated = False
    regime = "Unknown"
    probability = np.full(len(_REGIME_ORDER), np.nan)
    transition = pd.DataFrame(np.eye(3), index=_REGIME_ORDER, columns=_REGIME_ORDER)
    first_allocation: Optional[int] = None

    for pos, date in enumerate(dates):
        # 1-3: existing post-close weights earn the just-completed close return and drift.
        pre_trade = current.copy()
        day_return = asset_returns.iloc[pos].to_numpy(dtype=float)
        gross_return, post_return = _drift(pre_trade, day_return, date)
        current = post_return.copy()
        rebalance = pos in rebalance_positions
        target = np.zeros(len(assets))
        cost = turnover = 0.0
        post_trade = current.copy()

        if rebalance:
            # 4-6: close t is known; every fit/estimate is cut off at t.
            train = raw_features.loc[:date].tail(config.train_window)
            returns_for_estimation = asset_returns.loc[:date].tail(config.portfolio_config.estimation_lookback)
            if len(train) >= config.minimum_training_observations and len(returns_for_estimation) >= config.portfolio_config.minimum_estimation_observations:
                scaler = fit_feature_scaler(train)
                scaled = transform_features(scaler, train)
                hmm, _, _ = fit_hmm(scaled, train, config=config.hmm_config)
                solution = predict_current_state(hmm, scaler, train, train, config=config.hmm_config)
                regime = solution.regime
                probability = solution.probabilities.reindex(_REGIME_ORDER).to_numpy(dtype=float)
                transition = solution.transition_matrix.copy()
                previous = dict(zip(assets, current)) if allocated else None
                portfolio = optimize_portfolio(regime, returns_for_estimation, previous, config=config.portfolio_config)
                target = np.array([portfolio.weights[asset] for asset in assets], dtype=float)
                # 7-8: cost is charged at close t and target becomes active only after that close.
                turnover = float(np.abs(target).sum()) if not allocated else 0.5 * float(np.abs(target - current).sum())
                cost = cost_model.cost_for_trade(target, current, assets, initial=not allocated)
                post_trade = target.copy()
                current = target.copy()
                allocated = True
                if first_allocation is None:
                    first_allocation = pos
                diag.append({
                    "date": date, "regime": regime, "converged": solution.convergence,
                    "n_iter": solution.n_iter, "log_likelihood": solution.log_likelihood,
                    "turnover": turnover, "cost": cost,
                    "restarts": getattr(hmm, "_restart_diagnostics", []),
                })
            else:
                rebalance = False

        # Gross return belongs exclusively to holdings present before close t. Cost is paid after it.
        net_return = (1.0 + gross_return) * (1.0 - cost) - 1.0
        gross.append(gross_return); net.append(net_return); costs.append(cost); turns.append(turnover); flags.append(rebalance)
        targets.append(target); drifts.append(post_return); regimes.append(regime); probs.append(probability.copy())
        pre_returns.append(pre_trade); end_of_days.append(post_trade)
        rows.append(_event_row(date, rebalance, pre_trade, target, post_trade, post_return))

    if first_allocation is None:
        raise ValueError("No successful portfolio allocation occurred.")
    start = first_allocation
    index = dates[start:]
    def series(values: List[float], name: str) -> pd.Series: return pd.Series(values[start:], index=index, name=name)
    target_df = pd.DataFrame(targets[start:], index=index, columns=assets)
    drifted_df = pd.DataFrame(drifts[start:], index=index, columns=assets)
    pre_return_df = pd.DataFrame(pre_returns[start:], index=index, columns=assets)
    end_of_day_df = pd.DataFrame(end_of_days[start:], index=index, columns=assets)
    gross_s, net_s = series(gross, "gross_return"), series(net, "net_return")
    costs_s, turns_s, flags_s = series(costs, "transaction_cost"), series(turns, "turnover"), pd.Series(flags[start:], index=index, name="rebalance")
    result = BacktestResult(
        gross_returns=gross_s, net_returns=net_s, transaction_costs=costs_s, turnover=turns_s,
        target_weights=target_df, daily_drifted_weights=end_of_day_df,
        pre_return_weights=pre_return_df, post_return_pre_trade_weights=drifted_df,
        end_of_day_post_trade_weights=end_of_day_df,
        regime_series=pd.Series(regimes[start:], index=index, name="regime"),
        regime_probabilities=pd.DataFrame(probs[start:], index=index, columns=_REGIME_ORDER),
        transition_matrix=transition, gross_equity=(1 + gross_s).cumprod(), net_equity=(1 + net_s).cumprod(),
        rebalance_flags=flags_s, event_log=pd.DataFrame(rows[start:], index=index),
        hmm_diagnostics=diag,
        config_metadata=_build_metadata(config, prices, has_vix, index, risk_free_rate, cost_model),
    )
    result.metrics = compute_performance_metrics(net_s, gross_returns=gross_s, turnover=turns_s, transaction_costs=costs_s, risk_free_rate=risk_free_rate, annualization_factor=config.annualization_factor).to_dict()
    return result


def run_benchmark(prices: pd.DataFrame, weights: Dict[str, float], config: Optional[RegimeShiftConfig] = None, transaction_cost_bps: Optional[float] = None, rebalance_flags: Optional[pd.Series] = None, start_date: Optional[pd.Timestamp] = None, risk_free_rate: float = 0.0) -> BenchmarkResult:
    """Run a benchmark with the exact same return-then-close-execution ordering."""
    config = copy.deepcopy(config or RegimeShiftConfig()); config.validate()
    assets = config.core_assets
    returns = prices[assets].pct_change().iloc[1:]
    if start_date is not None:
        returns = returns.loc[returns.index >= pd.Timestamp(start_date)]
    dates = pd.DatetimeIndex(returns.index)
    flags = rebalance_flags.reindex(dates, fill_value=False).astype(bool) if rebalance_flags is not None else pd.Series([i % config.rebalance_frequency == 0 for i in range(len(dates))], index=dates)
    target_fixed = np.array([weights[asset] for asset in assets], dtype=float)
    if not np.isclose(target_fixed.sum(), 1.0):
        raise ValueError("Benchmark target weights must sum to one.")
    model = _cost_model(config, transaction_cost_bps)
    current = np.zeros(len(assets)); allocated = False
    values: Dict[str, List] = {k: [] for k in ("gross", "net", "cost", "turn")}
    target_rows: List[np.ndarray] = []; drift_rows: List[np.ndarray] = []; records: List[Dict] = []
    for pos, date in enumerate(dates):
        pre = current.copy(); g, post_return = _drift(pre, returns.iloc[pos].to_numpy(dtype=float), date); current = post_return.copy()
        rebalance = bool(flags.iloc[pos]); target = np.zeros(len(assets)); cost = turn = 0.0; post_trade = current.copy()
        if rebalance:
            target = target_fixed.copy(); turn = float(np.abs(target).sum()) if not allocated else 0.5 * float(np.abs(target - current).sum())
            cost = model.cost_for_trade(target, current, assets, initial=not allocated); current = target.copy(); post_trade = target.copy(); allocated = True
        values["gross"].append(g); values["net"].append((1 + g) * (1 - cost) - 1); values["cost"].append(cost); values["turn"].append(turn)
        target_rows.append(target); drift_rows.append(post_return); records.append(_event_row(date, rebalance, pre, target, post_trade, post_return))
    gross_s = pd.Series(values["gross"], index=dates, name="gross_return"); net_s = pd.Series(values["net"], index=dates, name="net_return")
    costs_s = pd.Series(values["cost"], index=dates, name="transaction_cost"); turns_s = pd.Series(values["turn"], index=dates, name="turnover")
    metrics = compute_performance_metrics(net_s, gross_returns=gross_s, turnover=turns_s, transaction_costs=costs_s, risk_free_rate=risk_free_rate, annualization_factor=config.annualization_factor).to_dict()
    return BenchmarkResult("benchmark", gross_s, net_s, (1 + gross_s).cumprod(), (1 + net_s).cumprod(), turns_s, costs_s, flags, pd.DataFrame(drift_rows, index=dates, columns=assets), pd.DataFrame(target_rows, index=dates, columns=assets), pd.DataFrame(records, index=dates), metrics)


def _override_transaction_cost(config: RegimeShiftConfig, bps: float) -> RegimeShiftConfig:
    result = copy.deepcopy(config); result.transaction_cost_bps = bps; result.execution_cost_model = None; return result


def _get_rebalance_dates(dates: pd.DatetimeIndex, frequency: int) -> pd.DatetimeIndex:
    return dates[list(range(frequency, len(dates), frequency))] if len(dates) > frequency else pd.DatetimeIndex([])


def _build_metadata(config: RegimeShiftConfig, prices: pd.DataFrame, has_vix: bool, result_dates: pd.DatetimeIndex, risk_free_rate: float, costs: ExecutionCostModel) -> Dict:
    diagnostics = getattr(prices, "attrs", {}).get("data_diagnostics", {})
    return {
        "execution_model": ExecutionModel.NEXT_CLOSE.value,
        "execution_timing": "Return t-1-to-t accrues before close-t signal, execution, and cost; close-t target first earns t-to-t+1.",
        "tradability_label": "Index-level allocation simulation: equity uses ^NSEI, which is a signal/benchmark series and not a guaranteed executable fill.",
        "market_impact_and_capacity": "Not modeled because executable volume/ADV data is not included.",
        "cost_model": {"name": costs.name, "assets": {asset: cost.total_bps for asset, cost in costs.assets.items()}, "units": "one-way assumed bps"},
        "date_range": {"start": str(result_dates[0]), "end": str(result_dates[-1]), "n_observations": len(result_dates), "original_start": str(prices.index[0]), "original_end": str(prices.index[-1])},
        "configured_tickers": {"equity": config.tickers.equity_ticker, "gold": config.tickers.gold_ticker, "bond": config.tickers.bond_ticker},
        "vix_included": has_vix, "train_window": config.train_window,
        "rebalance_frequency": config.rebalance_frequency, "hmm_n_restarts": config.hmm_config.n_restarts,
        "data_file": diagnostics.get("data_path", ""), "data_sha256": diagnostics.get("sha256", ""), "random_seed": config.random_seed,
        "risk_free_rate": risk_free_rate, "annualization_factor": config.annualization_factor,
    }
