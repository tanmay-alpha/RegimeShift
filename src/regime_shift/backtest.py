"""Event-driven, causally valid close-to-close walk-forward backtests.

Signal at close ``t`` (data through ``t`` only); fill at close ``t+1``.  At each
close the portfolio installed after the previous close first earns the
just-completed return.  A pending signal from the prior close is then traded,
costs are paid, and the new weights first earn the following close-to-close
return.  A target is never filled on the bar whose close produced it.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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


def _event_row(date: pd.Timestamp, signal_date: Optional[pd.Timestamp], executed: bool, pre_trade: np.ndarray, target: np.ndarray, post_trade: np.ndarray, post_return: np.ndarray) -> Dict:
    row: Dict = {
        "signal_date": signal_date if executed else pd.NaT,
        "execution_date": date if executed else pd.NaT,
        "return_date": date,
        "rebalance_flag": bool(executed),
    }
    for prefix, values in (("pre_return", pre_trade), ("post_return_pre_trade", post_return), ("target", target), ("end_of_day_post_trade", post_trade)):
        for asset, value in zip(_ASSET_ORDER, values):
            row[f"{prefix}_{asset}_weight"] = float(value)
    return row


@dataclass
class Signal:
    """Target weights decided at a close, plus model metadata to report."""
    target: np.ndarray
    info: Dict = field(default_factory=dict)


# (position, signal date, end-of-day post-trade weights, already allocated) -> Signal or None
SignalFn = Callable[[int, pd.Timestamp, np.ndarray, bool], Optional[Signal]]


@dataclass
class ExecutionTrace:
    """Per-date output of :func:`simulate_lagged_execution`, trimmed to the first fill."""
    gross: pd.Series
    net: pd.Series
    costs: pd.Series
    turnover: pd.Series
    executed: pd.Series
    targets: pd.DataFrame
    pre_trade: pd.DataFrame
    post_return: pd.DataFrame
    post_trade: pd.DataFrame
    events: pd.DataFrame
    signals: Dict[pd.Timestamp, Signal]
    fills: Dict[pd.Timestamp, pd.Timestamp]  # signal date -> execution date


def simulate_lagged_execution(
    asset_returns: pd.DataFrame,
    rebalance_positions: set,
    signal_fn: SignalFn,
    cost_model: ExecutionCostModel,
) -> ExecutionTrace:
    """Run the shared signal-at-close-t, fill-at-close-t+1 event loop.

    ``signal_fn`` is called only on ``rebalance_positions`` and only after the
    day's return and any pending fill are booked, so it can see data through
    close ``t`` but its target is not traded until close ``t+1``.  A signal on
    the final row has no next close and is never filled.
    """
    assets = list(asset_returns.columns)
    dates = pd.DatetimeIndex(asset_returns.index)
    returns = asset_returns.to_numpy(dtype=float)
    n_assets = len(assets)
    current = np.zeros(n_assets)
    allocated = False
    pending: Optional[tuple] = None
    first_fill: Optional[int] = None
    cols: Dict[str, List] = {k: [] for k in ("gross", "net", "cost", "turn", "exec", "target", "pre", "drift", "post", "event")}
    signals: Dict[pd.Timestamp, Signal] = {}
    fills: Dict[pd.Timestamp, pd.Timestamp] = {}

    for pos, date in enumerate(dates):
        pre_trade = current.copy()
        gross_return, post_return = _drift(pre_trade, returns[pos], date)
        current = post_return.copy()
        target = np.zeros(n_assets)
        cost = turnover = 0.0
        signal_date = None
        executed = pending is not None
        if executed:
            signal_date, signal = pending
            pending = None
            target = signal.target.copy()
            turnover = float(np.abs(target).sum()) if not allocated else 0.5 * float(np.abs(target - current).sum())
            cost = cost_model.cost_for_trade(target, current, assets, initial=not allocated)
            current = target.copy()
            allocated = True
            fills[signal_date] = date
            if first_fill is None:
                first_fill = pos
        post_trade = current.copy()

        if pos in rebalance_positions:
            signal = signal_fn(pos, date, post_trade.copy(), allocated)
            if signal is not None:
                signals[date] = signal
                pending = (date, signal)

        # Gross return belongs exclusively to holdings present before close t. Cost is paid after it.
        cols["gross"].append(gross_return); cols["net"].append((1.0 + gross_return) * (1.0 - cost) - 1.0)
        cols["cost"].append(cost); cols["turn"].append(turnover); cols["exec"].append(executed)
        cols["target"].append(target); cols["pre"].append(pre_trade); cols["drift"].append(post_return); cols["post"].append(post_trade)
        cols["event"].append(_event_row(date, signal_date, executed, pre_trade, target, post_trade, post_return))

    if first_fill is None:
        raise ValueError("No successful portfolio allocation occurred.")
    index = dates[first_fill:]

    def series(key: str, name: str) -> pd.Series:
        return pd.Series(cols[key][first_fill:], index=index, name=name)

    def frame(key: str) -> pd.DataFrame:
        return pd.DataFrame(cols[key][first_fill:], index=index, columns=assets)

    return ExecutionTrace(
        gross=series("gross", "gross_return"), net=series("net", "net_return"),
        costs=series("cost", "transaction_cost"), turnover=series("turn", "turnover"),
        executed=series("exec", "rebalance").astype(bool),
        targets=frame("target"), pre_trade=frame("pre"), post_return=frame("drift"), post_trade=frame("post"),
        events=pd.DataFrame(cols["event"][first_fill:], index=index), signals=signals, fills=fills,
    )


def prepare_walk_forward_inputs(prices: pd.DataFrame, config: RegimeShiftConfig) -> tuple:
    """Return (raw_features, asset_returns, rebalance_positions) on a shared date index."""
    assets = config.core_assets
    asset_returns = prices[assets].pct_change().iloc[1:]
    feature_prices = prices[assets].copy()
    if config.vix_col in prices.columns:
        feature_prices[config.vix_col] = prices[config.vix_col]
    raw_features = drop_feature_warmup(compute_raw_features(feature_prices, config.feature_config), config=config.feature_config)
    common = raw_features.index.intersection(asset_returns.index)
    raw_features, asset_returns = raw_features.loc[common], asset_returns.loc[common]
    minimum = config.minimum_training_observations + config.rebalance_frequency
    if len(common) < minimum:
        raise ValueError(f"Insufficient data after feature warmup: have {len(common)}, need at least {minimum}.")
    rebalance_positions = set(range(config.rebalance_frequency, len(common), config.rebalance_frequency))
    if not rebalance_positions:
        rebalance_positions = {len(common) - 2}
    return raw_features, asset_returns, rebalance_positions


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    config: Optional[RegimeShiftConfig] = None,
    transaction_cost_bps: Optional[float] = None,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Run the official ``NEXT_CLOSE`` sequence: signal at close t, fill at close t+1."""
    config = copy.deepcopy(config or RegimeShiftConfig())
    config.validate()
    if config.execution_model is not ExecutionModel.NEXT_CLOSE:
        raise ValueError("NEXT_OPEN cannot be simulated from close-only price data.")
    assets = config.core_assets
    has_vix = config.vix_col in prices.columns
    raw_features, asset_returns, rebalance_positions = prepare_walk_forward_inputs(prices, config)
    cost_model = _cost_model(config, transaction_cost_bps)

    def signal_fn(pos: int, date: pd.Timestamp, weights: np.ndarray, allocated: bool) -> Optional[Signal]:
        # Every fit/estimate is cut off at close t.
        train = raw_features.loc[:date].tail(config.train_window)
        returns_for_estimation = asset_returns.loc[:date].tail(config.portfolio_config.estimation_lookback)
        if len(train) < config.minimum_training_observations or len(returns_for_estimation) < config.portfolio_config.minimum_estimation_observations:
            return None
        scaler = fit_feature_scaler(train)
        hmm, _, _ = fit_hmm(transform_features(scaler, train), train, config=config.hmm_config)
        solution = predict_current_state(hmm, scaler, train, train, config=config.hmm_config)
        previous = dict(zip(assets, weights)) if allocated else None
        portfolio = optimize_portfolio(solution.regime, returns_for_estimation, previous, config=config.portfolio_config)
        return Signal(
            target=np.array([portfolio.weights[asset] for asset in assets], dtype=float),
            info={
                "regime": solution.regime,
                "probabilities": solution.probabilities.reindex(_REGIME_ORDER).to_numpy(dtype=float),
                "transition": solution.transition_matrix.copy(),
                "converged": solution.convergence, "n_iter": solution.n_iter,
                "log_likelihood": solution.log_likelihood,
                "restarts": getattr(hmm, "_restart_diagnostics", []),
            },
        )

    trace = simulate_lagged_execution(asset_returns, rebalance_positions, signal_fn, cost_model)
    index = trace.net.index

    # Regime state as known at each close: the latest signal dated on or before it.
    signal_dates = sorted(trace.signals)
    known = pd.DataFrame(
        [[trace.signals[d].info["regime"], *trace.signals[d].info["probabilities"]] for d in signal_dates],
        index=pd.DatetimeIndex(signal_dates), columns=["regime", *_REGIME_ORDER],
    ).reindex(index.union(signal_dates)).ffill().reindex(index)
    regime_series = known["regime"].fillna("Unknown").rename("regime")
    regime_probabilities = known[_REGIME_ORDER].astype(float)
    transition = trace.signals[signal_dates[-1]].info["transition"]

    diag = []
    for signal_date in signal_dates:
        info = trace.signals[signal_date].info
        execution_date = trace.fills.get(signal_date)
        diag.append({
            "date": signal_date, "execution_date": execution_date, "regime": info["regime"],
            "converged": info["converged"], "n_iter": info["n_iter"], "log_likelihood": info["log_likelihood"],
            "turnover": float(trace.turnover.get(execution_date, 0.0)) if execution_date is not None else 0.0,
            "cost": float(trace.costs.get(execution_date, 0.0)) if execution_date is not None else 0.0,
            "restarts": info["restarts"],
        })

    result = BacktestResult(
        gross_returns=trace.gross, net_returns=trace.net, transaction_costs=trace.costs, turnover=trace.turnover,
        target_weights=trace.targets, daily_drifted_weights=trace.post_trade,
        pre_return_weights=trace.pre_trade, post_return_pre_trade_weights=trace.post_return,
        end_of_day_post_trade_weights=trace.post_trade,
        regime_series=regime_series, regime_probabilities=regime_probabilities,
        transition_matrix=transition, gross_equity=(1 + trace.gross).cumprod(), net_equity=(1 + trace.net).cumprod(),
        rebalance_flags=trace.executed, event_log=trace.events,
        hmm_diagnostics=diag,
        config_metadata=_build_metadata(config, prices, has_vix, index, risk_free_rate, cost_model),
    )
    result.metrics = compute_performance_metrics(trace.net, gross_returns=trace.gross, turnover=trace.turnover, transaction_costs=trace.costs, risk_free_rate=risk_free_rate, annualization_factor=config.annualization_factor).to_dict()
    return result


def run_benchmark(prices: pd.DataFrame, weights: Dict[str, float], config: Optional[RegimeShiftConfig] = None, transaction_cost_bps: Optional[float] = None, rebalance_flags: Optional[pd.Series] = None, start_date: Optional[pd.Timestamp] = None, risk_free_rate: float = 0.0) -> BenchmarkResult:
    """Run a fixed-weight benchmark with the same return-then-close-execution ordering.

    Pass the strategy's ``rebalance_flags`` (its fill dates) to trade on identical
    dates.  Fixed weights carry no information, so signal and fill dates coincide.
    """
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
        target_rows.append(target); drift_rows.append(post_return); records.append(_event_row(date, date, rebalance, pre, target, post_trade, post_return))
    gross_s = pd.Series(values["gross"], index=dates, name="gross_return"); net_s = pd.Series(values["net"], index=dates, name="net_return")
    costs_s = pd.Series(values["cost"], index=dates, name="transaction_cost"); turns_s = pd.Series(values["turn"], index=dates, name="turnover")
    metrics = compute_performance_metrics(net_s, gross_returns=gross_s, turnover=turns_s, transaction_costs=costs_s, risk_free_rate=risk_free_rate, annualization_factor=config.annualization_factor).to_dict()
    return BenchmarkResult("benchmark", gross_s, net_s, (1 + gross_s).cumprod(), (1 + net_s).cumprod(), turns_s, costs_s, flags, pd.DataFrame(drift_rows, index=dates, columns=assets), pd.DataFrame(target_rows, index=dates, columns=assets), pd.DataFrame(records, index=dates), metrics)


def _build_metadata(config: RegimeShiftConfig, prices: pd.DataFrame, has_vix: bool, result_dates: pd.DatetimeIndex, risk_free_rate: float, costs: ExecutionCostModel) -> Dict:
    diagnostics = getattr(prices, "attrs", {}).get("data_diagnostics", {})
    return {
        "execution_model": ExecutionModel.NEXT_CLOSE.value,
        "execution_timing": "Signal at close t from data through t; fill and cost at close t+1 after the t-to-t+1 return accrues on prior holdings; new target first earns t+1-to-t+2.",
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
