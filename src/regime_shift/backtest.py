"""
Leakage-safe walk-forward backtesting engine.

Implements the complete RegimeShift backtest pipeline:

    Real prices + optional VIX
        ↓
    Leakage-safe features
        ↓
    Train-only scaling
        ↓
    3-state Gaussian HMM
        ↓
    Regime inference
        ↓
    CVXPY portfolio optimization
        ↓
    Weight drift + transaction costs
        ↓
    Net returns

All information used at decision date d is restricted to data through d-1.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from regime_shift.config import (
    RegimeShiftConfig,
    FeatureConfig,
)
from regime_shift.exceptions import (
    RegimeDetectionError,
    PortfolioOptimizationError,
    FeatureEngineeringError,
)
from regime_shift.features import (
    compute_raw_features,
    drop_feature_warmup,
    fit_feature_scaler,
    transform_features,
)
from regime_shift.regime_model import (
    fit_hmm,
    predict_current_state,
    RegimeSolution,
)
from regime_shift.portfolio import (
    optimize_portfolio,
    PortfolioSolution,
)
from regime_shift.benchmarks import (
    static_60_40_weights,
    equal_weight_weights,
)
from regime_shift.metrics import compute_performance_metrics

logger = logging.getLogger(__name__)

# Deterministic asset column ordering
_ASSET_ORDER = ["equity", "gold", "bond"]


@dataclass
class BacktestResult:
    """
    Complete backtest result container.

    All series have a DatetimeIndex aligned to trading days in the backtest period.
    """
    # Core return series
    gross_returns: pd.Series = field(default_factory=pd.Series, repr=False)
    net_returns: pd.Series = field(default_factory=pd.Series, repr=False)
    transaction_costs: pd.Series = field(default_factory=pd.Series, repr=False)
    turnover: pd.Series = field(default_factory=pd.Series, repr=False)

    # Weight history
    target_weights: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    daily_drifted_weights: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # Regime information
    regime_series: pd.Series = field(default_factory=pd.Series, repr=False)
    regime_probabilities: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    transition_matrix: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    # Equity curves
    gross_equity: pd.Series = field(default_factory=pd.Series, repr=False)
    net_equity: pd.Series = field(default_factory=pd.Series, repr=False)

    # Rebalance flags
    rebalance_flags: pd.Series = field(default_factory=pd.Series, repr=False)

    # Performance metrics
    metrics: Optional[Dict] = None

    # HMM diagnostics per rebalance
    hmm_diagnostics: List[Dict] = field(default_factory=list)

    # Configuration metadata
    config_metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Benchmark return series and metrics."""
    name: str
    gross_returns: pd.Series = field(default_factory=pd.Series, repr=False)
    net_returns: pd.Series = field(default_factory=pd.Series, repr=False)
    gross_equity: pd.Series = field(default_factory=pd.Series, repr=False)
    net_equity: pd.Series = field(default_factory=pd.Series, repr=False)
    turnover: pd.Series = field(default_factory=pd.Series, repr=False)
    metrics: Optional[Dict] = None


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    config: Optional[RegimeShiftConfig] = None,
    transaction_cost_bps: Optional[float] = None,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """
    Execute a leakage-safe walk-forward backtest.

    At each rebalance date d, ALL information used for the portfolio decision
    is restricted to data through the previous trading date (d-1).  The
    resulting weights are applied beginning on date d.

    Args:
        prices: Validated price DataFrame with columns [equity, gold, bond]
            and optional [vix]. DatetimeIndex must be sorted and unique.
        config: RegimeShiftConfig.  Defaults to RegimeShiftConfig() if None.
        transaction_cost_bps: Override transaction cost (default from config).
        risk_free_rate: Annualized risk-free rate for Sharpe/Sortino (default 0.0).

    Returns:
        BacktestResult with all return series, weights, regimes, and metrics.

    Raises:
        ValueError: If prices are insufficient for the configured windows.
        RegimeDetectionError: If HMM fails during any rebalance.
        PortfolioOptimizationError: If optimization fails during any rebalance.
    """
    config = config or RegimeShiftConfig()
    if transaction_cost_bps is not None:
        config = _override_transaction_cost(config, transaction_cost_bps)
    config.validate()

    # Extract price columns
    core_assets = config.core_assets
    price_data = prices[core_assets].copy()
    has_vix = config.vix_col in prices.columns

    # Compute log returns for portfolio calculations
    asset_returns = price_data.pct_change().iloc[1:]

    # Compute raw features
    feature_prices = price_data.copy()
    if has_vix:
        feature_prices[config.vix_col] = prices[config.vix_col]

    raw_features = compute_raw_features(feature_prices, config.feature_config)
    raw_features = drop_feature_warmup(raw_features, config.feature_config)

    if len(raw_features) == 0:
        raise ValueError(
            "No features remain after warmup removal. "
            "Supply more historical data."
        )

    # Align features and returns
    common_idx = raw_features.index.intersection(asset_returns.index)
    raw_features = raw_features.loc[common_idx]
    asset_returns = asset_returns.loc[common_idx]

    # Determine rebalance dates
    rebalance_dates = _get_rebalance_dates(
        raw_features.index, config.rebalance_frequency
    )

    if len(rebalance_dates) == 0:
        raise ValueError(
            "Not enough data for even one rebalance. "
            f"Need at least {config.minimum_training_observations} training observations "
            f"plus {config.rebalance_frequency} days for the first rebalance."
        )

    # Minimum data check
    min_required = (
        config.feature_config.max_lookback()
        + config.minimum_training_observations
        + config.rebalance_frequency
    )
    if len(raw_features) < min_required:
        raise ValueError(
            f"Insufficient data: have {len(raw_features)} rows after warmup, "
            f"need at least {min_required} for warmup + training + first rebalance."
        )

    # Initialize tracking
    dates = asset_returns.index
    n_dates = len(dates)

    current_weights = np.zeros(len(core_assets))
    target_w = np.zeros(len(core_assets))

    gross_returns_arr = np.zeros(n_dates)
    net_returns_arr = np.zeros(n_dates)
    costs_arr = np.zeros(n_dates)
    turnover_arr = np.zeros(n_dates)
    rebalance_arr = np.zeros(n_dates, dtype=bool)

    target_weights_list = []
    drifted_weights_list = []
    regime_list = []
    prob_list = []
    hmm_diag = []

    cost_rate = config.transaction_cost_bps / 10000.0

    # Map dates to positions for efficient lookup
    date_to_pos = {d: i for i, d in enumerate(dates)}

    # Track the last rebalance position
    last_rebalance_pos = -1

    for rebalance_date in rebalance_dates:
        if rebalance_date not in date_to_pos:
            continue
        rebalance_pos = date_to_pos[rebalance_date]

        # information_cutoff = previous trading day
        if rebalance_pos == 0:
            continue
        info_cutoff = dates[rebalance_pos - 1]

        # ---- Training window ----
        # Use all available data through info_cutoff for training (expanding window)
        train_features = raw_features.loc[:info_cutoff]

        if len(train_features) < config.minimum_training_observations:
            logger.warning(
                "Skipping rebalance on %s: only %d training observations, "
                "need %d.",
                rebalance_date, len(train_features),
                config.minimum_training_observations,
            )
            # Continue with previous weights
            for pos in range(last_rebalance_pos + 1, rebalance_pos + 1):
                if pos >= n_dates:
                    break
                gross_returns_arr[pos] = np.dot(
                    current_weights, asset_returns.iloc[pos].values
                )
                current_weights = current_weights * (1 + asset_returns.iloc[pos].values)
                current_weights = current_weights / current_weights.sum()
                net_returns_arr[pos] = gross_returns_arr[pos]
                drifted_weights_list.append(
                    (dates[pos], pd.Series(current_weights, index=core_assets))
                )
                regime_list.append((dates[pos], "Unknown"))
                prob_list.append(
                    (dates[pos], pd.Series(np.nan, index=["Bull", "Bear", "Crisis"]))
                )
            last_rebalance_pos = rebalance_pos
            continue

        # ---- Fit scaler on training data ----
        scaler = fit_feature_scaler(train_features)
        scaled_train = transform_features(scaler, train_features)

        # ---- Fit HMM ----
        try:
            hmm, hidden_states, trans_mat = fit_hmm(
                scaled_train, train_features
            )
        except RegimeDetectionError as exc:
            logger.error("HMM fitting failed on %s: %s", info_cutoff, exc)
            raise

        # ---- Infer regime at info_cutoff ----
        features_through_cutoff = raw_features.loc[:info_cutoff]
        regime_solution = predict_current_state(
            hmm, scaler, features_through_cutoff, train_features
        )

        regime = regime_solution.regime
        probs = regime_solution.probabilities

        # ---- Optimize portfolio ----
        # Use returns through info_cutoff for estimation
        returns_for_est = asset_returns.loc[:info_cutoff].tail(
            config.portfolio_config.estimation_lookback
        )

        if len(returns_for_est) < config.portfolio_config.minimum_estimation_observations:
            logger.warning(
                "Skipping rebalance on %s: only %d return observations, "
                "need %d.",
                rebalance_date, len(returns_for_est),
                config.portfolio_config.minimum_estimation_observations,
            )
            for pos in range(last_rebalance_pos + 1, rebalance_pos + 1):
                if pos >= n_dates:
                    break
                gross_returns_arr[pos] = np.dot(
                    current_weights, asset_returns.iloc[pos].values
                )
                current_weights = current_weights * (1 + asset_returns.iloc[pos].values)
                current_weights = current_weights / current_weights.sum()
                net_returns_arr[pos] = gross_returns_arr[pos]
                drifted_weights_list.append(
                    (dates[pos], pd.Series(current_weights, index=core_assets))
                )
                regime_list.append((dates[pos], regime))
                prob_list.append((dates[pos], probs))
            last_rebalance_pos = rebalance_pos
            continue

        # Previous weights for turnover penalty
        prev_w = pd.Series(current_weights, index=core_assets).to_dict()

        try:
            port_solution = optimize_portfolio(
                regime=regime,
                returns=returns_for_est,
                previous_weights=prev_w if np.sum(current_weights) > 0 else None,
                config=config.portfolio_config,
            )
        except PortfolioOptimizationError as exc:
            logger.error("Portfolio optimization failed on %s: %s", rebalance_date, exc)
            raise

        target_w = np.array([port_solution.weights[a] for a in core_assets])

        # ---- Apply transaction costs and update weights ----
        # For the rebalance date itself:
        pre_trade_weights = current_weights.copy()
        turnover = 0.5 * np.sum(np.abs(target_w - pre_trade_weights))
        cost = turnover * cost_rate

        # Apply rebalance: replace drifted weights with target weights, then apply returns
        gross_ret = float(np.dot(target_w, asset_returns.iloc[rebalance_pos].values))
        net_ret = (1 - cost) * (1 + gross_ret) - 1

        gross_returns_arr[rebalance_pos] = gross_ret
        net_returns_arr[rebalance_pos] = net_ret
        costs_arr[rebalance_pos] = cost
        turnover_arr[rebalance_pos] = turnover
        rebalance_arr[rebalance_pos] = True

        # Update current weights to target * (1 + return)
        current_weights = target_w * (1 + asset_returns.iloc[rebalance_pos].values)
        current_weights = current_weights / current_weights.sum()

        # Record target weights
        target_weights_list.append(
            (rebalance_date, pd.Series(target_w, index=core_assets))
        )
        drifted_weights_list.append(
            (rebalance_date, pd.Series(current_weights.copy(), index=core_assets))
        )
        regime_list.append((rebalance_date, regime))
        prob_list.append((rebalance_date, probs))

        # Record diagnostics
        hmm_diag.append({
            "date": rebalance_date,
            "regime": regime,
            "converged": regime_solution.convergence,
            "n_iter": regime_solution.n_iter,
            "log_likelihood": regime_solution.log_likelihood,
            "turnover": turnover,
            "cost": cost,
        })

        last_rebalance_pos = rebalance_pos

        # ---- Drift weights between rebalances ----
        for pos in range(rebalance_pos + 1, n_dates):
            gross_ret = float(np.dot(
                current_weights, asset_returns.iloc[pos].values
            ))
            gross_returns_arr[pos] = gross_ret
            net_returns_arr[pos] = gross_ret  # no cost on non-rebalance dates
            costs_arr[pos] = 0.0
            turnover_arr[pos] = 0.0

            # Drift weights
            current_weights = current_weights * (1 + asset_returns.iloc[pos].values)
            current_weights = current_weights / current_weights.sum()

            drifted_weights_list.append(
                (dates[pos], pd.Series(current_weights.copy(), index=core_assets))
            )
            regime_list.append((dates[pos], regime))
            prob_list.append((dates[pos], probs))

    # ---- Build result DataFrames ----
    dates_series = pd.DatetimeIndex(dates)

    gross_returns_s = pd.Series(gross_returns_arr, index=dates_series, name="gross_return")
    net_returns_s = pd.Series(net_returns_arr, index=dates_series, name="net_return")
    costs_s = pd.Series(costs_arr, index=dates_series, name="transaction_cost")
    turnover_s = pd.Series(turnover_arr, index=dates_series, name="turnover")
    rebalance_s = pd.Series(rebalance_arr, index=dates_series, name="rebalance")

    # Target weights (rebalance dates only)
    if target_weights_list:
        tw_idx, tw_vals = zip(*target_weights_list)
        target_weights_df = pd.DataFrame(tw_vals, index=pd.DatetimeIndex(tw_idx))
        target_weights_df = target_weights_df.reindex(dates_series, method="ffill")
    else:
        target_weights_df = pd.DataFrame(
            np.zeros((n_dates, len(core_assets))),
            index=dates_series,
            columns=core_assets,
        )

    # Drifted weights
    if drifted_weights_list:
        dw_idx, dw_vals = zip(*drifted_weights_list)
        drifted_weights_df = pd.DataFrame(dw_vals, index=pd.DatetimeIndex(dw_idx))
        drifted_weights_df = drifted_weights_df.reindex(dates_series, method="ffill")
    else:
        drifted_weights_df = pd.DataFrame(
            np.zeros((n_dates, len(core_assets))),
            index=dates_series,
            columns=core_assets,
        )

    # Regime series
    if regime_list:
        rg_idx, rg_vals = zip(*regime_list)
        regime_s = pd.Series(rg_vals, index=pd.DatetimeIndex(rg_idx), name="regime")
        regime_s = regime_s.reindex(dates_series, method="ffill")
    else:
        regime_s = pd.Series(index=dates_series, dtype=str, name="regime")

    # Regime probabilities
    if prob_list:
        pb_idx, pb_vals = zip(*prob_list)
        pb_df = pd.DataFrame(pb_vals.tolist(), index=pd.DatetimeIndex(pb_idx))
        pb_df = pb_df.reindex(dates_series, method="ffill")
    else:
        pb_df = pd.DataFrame(
            np.zeros((n_dates, 3)),
            index=dates_series,
            columns=["Bull", "Bear", "Crisis"],
        )

    # Equity curves
    gross_equity_s = (1 + gross_returns_s).cumprod()
    net_equity_s = (1 + net_returns_s).cumprod()

    # Transition matrix from last HMM fit
    final_trans_mat = pd.DataFrame(
        np.eye(3),
        index=["Bull", "Bear", "Crisis"],
        columns=["Bull", "Bear", "Crisis"],
    )
    if hmm_diag:
        last_diag = hmm_diag[-1]
        # The transition matrix is stored in the last fit_hmm result
        # We'll capture it separately

    # Compute metrics
    metrics = compute_performance_metrics(
        net_returns_s, turnover=turnover_s, risk_free_rate=risk_free_rate
    )

    result = BacktestResult(
        gross_returns=gross_returns_s,
        net_returns=net_returns_s,
        transaction_costs=costs_s,
        turnover=turnover_s,
        target_weights=target_weights_df,
        daily_drifted_weights=drifted_weights_df,
        regime_series=regime_s,
        regime_probabilities=pb_df,
        transition_matrix=final_trans_mat,
        gross_equity=gross_equity_s,
        net_equity=net_equity_s,
        rebalance_flags=rebalance_s,
        metrics=metrics.to_dict() if metrics else None,
        hmm_diagnostics=hmm_diag,
        config_metadata=_build_metadata(config, prices, has_vix),
    )

    return result


def run_benchmark(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    config: Optional[RegimeShiftConfig] = None,
    transaction_cost_bps: Optional[float] = None,
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
) -> BenchmarkResult:
    """
    Run a benchmark strategy with the same timing and cost conventions.

    Args:
        prices: Price DataFrame with equity, gold, bond columns.
        weights: Target weight dictionary.
        config: RegimeShiftConfig.
        transaction_cost_bps: Override transaction cost.
        rebalance_dates: Pre-computed rebalance dates (must match strategy).

    Returns:
        BenchmarkResult with returns and equity curves.
    """
    config = config or RegimeShiftConfig()
    if transaction_cost_bps is not None:
        config = _override_transaction_cost(config, transaction_cost_bps)

    core_assets = config.core_assets
    price_data = prices[core_assets].copy()
    asset_returns = price_data.pct_change().iloc[1:]

    dates = asset_returns.index
    n = len(dates)

    # Default rebalance dates
    if rebalance_dates is None:
        rebalance_dates = _get_rebalance_dates(dates, config.rebalance_frequency)

    date_to_pos = {d: i for i, d in enumerate(dates)}
    rebalance_positions = [
        date_to_pos[d] for d in rebalance_dates if d in date_to_pos
    ]

    w = np.array([weights[a] for a in core_assets])
    current_weights = w.copy()

    cost_rate = config.transaction_cost_bps / 10000.0
    gross_arr = np.zeros(n)
    net_arr = np.zeros(n)
    costs_arr = np.zeros(n)
    turnover_arr = np.zeros(n)

    for pos in range(n):
        gross_ret = float(np.dot(current_weights, asset_returns.iloc[pos].values))

        if pos in rebalance_positions:
            turnover = 0.5 * np.sum(np.abs(current_weights - w))
            cost = turnover * cost_rate
            net_ret = (1 - cost) * (1 + gross_ret) - 1
            turnover_arr[pos] = turnover
            costs_arr[pos] = cost

            # Update to target weights then apply return
            current_weights = w * (1 + asset_returns.iloc[pos].values)
            current_weights = current_weights / current_weights.sum()
        else:
            net_ret = gross_ret
            current_weights = current_weights * (1 + asset_returns.iloc[pos].values)
            current_weights = current_weights / current_weights.sum()

        gross_arr[pos] = gross_ret
        net_arr[pos] = net_ret

    gross_s = pd.Series(gross_arr, index=dates, name="gross_return")
    net_s = pd.Series(net_arr, index=dates, name="net_return")
    turnover_s = pd.Series(turnover_arr, index=dates, name="turnover")
    gross_equity = (1 + gross_s).cumprod()
    net_equity = (1 + net_s).cumprod()

    metrics = compute_performance_metrics(
        net_s, turnover=turnover_s, risk_free_rate=0.0
    )

    return BenchmarkResult(
        name="benchmark",
        gross_returns=gross_s,
        net_returns=net_s,
        gross_equity=gross_equity,
        net_equity=net_equity,
        turnover=turnover_s,
        metrics=metrics.to_dict() if metrics else None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _override_transaction_cost(
    config: RegimeShiftConfig,
    bps: float,
) -> RegimeShiftConfig:
    """Return a copy of config with overridden transaction cost."""
    import copy
    new_config = copy.deepcopy(config)
    new_config.transaction_cost_bps = bps
    return new_config


def _get_rebalance_dates(
    dates: pd.DatetimeIndex,
    frequency: int,
) -> pd.DatetimeIndex:
    """
    Return rebalance dates at the specified frequency.

    The first rebalance date is after enough data has accumulated for:
    feature warmup + minimum training + one rebalance period.
    """
    if len(dates) < frequency:
        return pd.DatetimeIndex([])

    # Start from frequency to allow initial drift period
    indices = list(range(frequency, len(dates), frequency))
    if not indices:
        indices = [len(dates) - 1]

    return dates[indices]


def _build_metadata(
    config: RegimeShiftConfig,
    prices: pd.DataFrame,
    has_vix: bool,
) -> Dict:
    """Build run metadata dictionary."""
    return {
        "date_range": {
            "start": str(prices.index[0]),
            "end": str(prices.index[-1]),
            "n_observations": len(prices),
        },
        "tickers": {
            "equity": config.tickers.equity.ticker,
            "gold": config.tickers.gold.ticker,
            "bond": config.tickers.bond.ticker,
            "vix": config.tickers.vix.ticker if has_vix else "omitted",
        },
        "vix_included": has_vix,
        "transaction_cost_bps": config.transaction_cost_bps,
        "train_window": config.train_window,
        "rebalance_frequency": config.rebalance_frequency,
        "hmm_config": {
            "n_components": config.hmm_config.n_components,
            "covariance_type": config.hmm_config.covariance_type,
            "n_iter": config.hmm_config.n_iter,
            "random_state": config.hmm_config.random_state,
        },
        "feature_list": [
            "equity_log_return_1d",
            "equity_momentum_21d",
            "equity_momentum_63d",
            "equity_volatility_21d",
            "equity_volatility_ratio_21_63",
            "equity_gold_correlation_63d",
            "equity_bond_correlation_63d",
        ]
        + (["vix_change_5d", "vix_level"] if has_vix else []),
        "package_version": "0.5.0-hmm-portfolio",
        "random_seed": config.random_seed,
    }
