"""
Leakage-safe walk-forward backtesting engine.

Implements the complete RegimeShift backtest pipeline:

    Real prices + optional VIX
        â†“
    Leakage-safe features
        â†“
    Train-only scaling
    â†“
    3-state Gaussian HMM
        â†“
    Regime inference
        â†“
    CVXPY portfolio optimization
        â†“
    Weight drift + transaction costs
        â†“
    Net returns

All information used at decision date d is restricted to data through d-1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from regime_shift.config import RegimeShiftConfig
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

# Deterministic regime label ordering
_REGIME_ORDER = ["Bull", "Bear", "Crisis"]


@dataclass
class BacktestResult:
    """
    Complete backtest result container.

    All series have a DatetimeIndex aligned to trading days in the backtest period.
    The index starts at the first successful allocation date (not the first price date).
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
    """
    Benchmark return series and metrics.

    All series have identical, aligned DatetimeIndex. Target weights track
    the configured weights on rebalance dates; daily_drifted_weights track
    the actual weights after returns drift between rebalances.
    """
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
    metrics: Optional[Dict] = None


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    config: Optional[RegimeShiftConfig] = None,
    transaction_cost_bps: Optional[float] = None,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """
    Execute a leakage-safe walk-forward backtest.

    Uses a single chronological loop over trading dates. At each date d:

    1. current_weights represents the drifted portfolio before trading on d.
    2. If d is a rebalance date, fit scaler/HMM/optimizer using data through d-1.
    3. Calculate return on d using the (possibly new) weights.
    4. Deduct transaction cost on d if rebalancing.
    5. Drift weights by the return on d.

    The reported results start at the first successful allocation date.

    Args:
        prices: Validated price DataFrame with columns [equity, gold, bond]
            and optional [vix]. DatetimeIndex must be sorted and unique.
        config: RegimeShiftConfig.  Defaults to RegimeShiftConfig() if None.
        transaction_cost_bps: Override transaction cost (default from config).
        risk_free_rate: Annualized risk-free rate for Sharpe/Sortino (default 0.0).

    Returns:
        BacktestResult with all return series, weights, regimes, and metrics.
        The result index starts at the first successful allocation date.

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
    raw_features = drop_feature_warmup(
        raw_features, config=config.feature_config
    )

    if len(raw_features) == 0:
        raise ValueError(
            "No features remain after warmup removal. "
            "Supply more historical data."
        )

    # Align features and returns
    common_idx = raw_features.index.intersection(asset_returns.index)
    raw_features = raw_features.loc[common_idx]
    asset_returns = asset_returns.loc[common_idx]

    dates = asset_returns.index
    n_dates = len(dates)

    if n_dates == 0:
        raise ValueError("No aligned return dates after feature warmup.")

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

    # Determine rebalance positions (every rebalance_frequency days)
    rebalance_positions = list(
        range(config.rebalance_frequency, n_dates, config.rebalance_frequency)
    )
    if not rebalance_positions:
        rebalance_positions = [n_dates - 1]

    cost_rate = config.transaction_cost_bps / 10000.0

    # Output arrays â€” sized to full date range
    # We'll trim to the first successful allocation date later
    gross_returns_arr = np.zeros(n_dates)
    net_returns_arr = np.zeros(n_dates)
    costs_arr = np.zeros(n_dates)
    turnover_arr = np.zeros(n_dates)
    rebalance_arr = np.zeros(n_dates, dtype=bool)

    all_target_w = np.zeros((n_dates, len(core_assets)))
    all_drifted_w = np.zeros((n_dates, len(core_assets)))
    all_regimes = np.full(n_dates, "Unknown", dtype=object)
    all_probs = np.zeros((n_dates, len(_REGIME_ORDER)))
    all_probs[:] = np.nan

    hmm_diag = []
    first_allocation_pos = None
    final_trans_mat = pd.DataFrame(
        np.eye(3), index=_REGIME_ORDER, columns=_REGIME_ORDER
    )

    # ---- Single chronological loop ----
    current_weights = np.zeros(len(core_assets))
    has_allocation = False
    current_regime = "Unknown"
    current_probs = np.full(len(_REGIME_ORDER), np.nan)

    for pos in range(n_dates):
        date = dates[pos]
        rebalance_today = pos in rebalance_positions

        if rebalance_today and pos > 0:
            info_cutoff = dates[pos - 1]

            # Training window: rolling window of config.train_window observations
            # through the information cutoff
            train_features = raw_features.loc[:info_cutoff].tail(
                config.train_window
            )

            if len(train_features) < config.minimum_training_observations:
                logger.warning(
                    "Skipping rebalance on %s: only %d training observations, "
                    "need %d.",
                    date, len(train_features),
                    config.minimum_training_observations,
                )
                # Continue with previous weights (no rebalance)
            else:
                # ---- Fit scaler on training data only ----
                scaler = fit_feature_scaler(train_features)
                scaled_train = transform_features(scaler, train_features)

                # ---- Fit HMM on scaled training data ----
                try:
                    hmm, hidden_states, _trans_mat = fit_hmm(
                        scaled_train, train_features,
                        config=config.hmm_config,
                    )

                    # Infer current regime using ONLY the rolling training window
                    # (features_through_cutoff = train_features).
                    # This prevents the scaler from seeing future data.
                    regime_solution = predict_current_state(
                        hmm, scaler, train_features, train_features,
                        config=config.hmm_config,
                    )
                    # Store the actual transition matrix from the last fit
                    final_trans_mat = regime_solution.transition_matrix.copy()
                except RegimeDetectionError as exc:
                    logger.error("HMM fitting failed on %s: %s", info_cutoff, exc)
                    raise

                current_regime = regime_solution.regime
                # Explicitly reindex probabilities to Bull, Bear, Crisis order
                current_probs = regime_solution.probabilities.reindex(
                    _REGIME_ORDER
                ).values

                # ---- Optimize portfolio ----
                # Use returns through info_cutoff for estimation
                returns_for_est = asset_returns.loc[:info_cutoff].tail(
                    config.portfolio_config.estimation_lookback
                )

                if len(returns_for_est) < config.portfolio_config.minimum_estimation_observations:
                    logger.warning(
                        "Skipping rebalance on %s: only %d return observations, "
                        "need %d.",
                        date, len(returns_for_est),
                        config.portfolio_config.minimum_estimation_observations,
                    )
                else:
                    # Previous weights for turnover calculation
                    prev_w = pd.Series(
                        current_weights, index=core_assets
                    ).to_dict()

                    try:
                        port_solution = optimize_portfolio(
                            regime=current_regime,
                            returns_through_date=returns_for_est,
                            previous_weights=(
                                prev_w if has_allocation else None
                            ),
                            config=config.portfolio_config,
                        )
                    except PortfolioOptimizationError as exc:
                        logger.error(
                            "Portfolio optimization failed on %s: %s",
                            date, exc
                        )
                        raise

                    target_w = np.array([
                        port_solution.weights[a] for a in core_assets
                    ])

                    # ---- Calculate turnover ----
                    if not has_allocation:
                        # Initial allocation from cash: turnover = sum(abs(target))
                        turnover = float(np.sum(np.abs(target_w)))
                    else:
                        turnover = 0.5 * float(
                            np.sum(np.abs(target_w - current_weights))
                        )

                    cost = turnover * cost_rate

                    # ---- Apply return using target weights ----
                    day_return = asset_returns.iloc[pos].values
                    gross_ret = float(np.dot(target_w, day_return))
                    net_ret = (1 - cost) * (1 + gross_ret) - 1

                    gross_returns_arr[pos] = gross_ret
                    net_returns_arr[pos] = net_ret
                    costs_arr[pos] = cost
                    turnover_arr[pos] = turnover
                    rebalance_arr[pos] = True

                    # Drift weights by the return
                    current_weights = target_w * (1 + day_return)
                    s = current_weights.sum()
                    if s > 1e-12:
                        current_weights = current_weights / s

                    all_target_w[pos] = target_w
                    all_drifted_w[pos] = current_weights.copy()
                    all_regimes[pos] = current_regime
                    all_probs[pos] = current_probs

                    if first_allocation_pos is None:
                        first_allocation_pos = pos

                    has_allocation = True

                    # Record diagnostics
                    hmm_diag.append({
                        "date": date,
                        "regime": current_regime,
                        "converged": regime_solution.convergence,
                        "n_iter": regime_solution.n_iter,
                        "log_likelihood": regime_solution.log_likelihood,
                        "turnover": turnover,
                        "cost": cost,
                    })
                    continue  # Skip the drift section below; already drifted

        # ---- Drift weights (non-rebalance dates, or skipped rebalances) ----
        if has_allocation:
            day_return = asset_returns.iloc[pos].values
            gross_ret = float(np.dot(current_weights, day_return))
            gross_returns_arr[pos] = gross_ret
            net_returns_arr[pos] = gross_ret  # no cost on non-rebalance dates
            costs_arr[pos] = 0.0
            turnover_arr[pos] = 0.0

            # Drift weights by the return
            current_weights = current_weights * (1 + day_return)
            s = current_weights.sum()
            if s > 1e-12:
                current_weights = current_weights / s

            all_drifted_w[pos] = current_weights
            all_regimes[pos] = current_regime
            all_probs[pos] = current_probs

            if first_allocation_pos is None:
                first_allocation_pos = pos

    # ---- Trim to first successful allocation date ----
    if first_allocation_pos is None:
        raise ValueError(
            "No successful portfolio allocation occurred. "
            "Check training data sufficiency and HMM convergence."
        )

    trim_start = first_allocation_pos

    dates_series = pd.DatetimeIndex(dates[trim_start:])

    gross_returns_s = pd.Series(
        gross_returns_arr[trim_start:], index=dates_series, name="gross_return"
    )
    net_returns_s = pd.Series(
        net_returns_arr[trim_start:], index=dates_series, name="net_return"
    )
    costs_s = pd.Series(
        costs_arr[trim_start:], index=dates_series, name="transaction_cost"
    )
    turnover_s = pd.Series(
        turnover_arr[trim_start:], index=dates_series, name="turnover"
    )
    rebalance_s = pd.Series(
        rebalance_arr[trim_start:], index=dates_series, name="rebalance"
    )

    target_weights_df = pd.DataFrame(
        all_target_w[trim_start:], index=dates_series, columns=core_assets
    )
    drifted_weights_df = pd.DataFrame(
        all_drifted_w[trim_start:], index=dates_series, columns=core_assets
    )
    regime_s = pd.Series(
        all_regimes[trim_start:], index=dates_series, name="regime"
    )
    pb_df = pd.DataFrame(
        all_probs[trim_start:], index=dates_series, columns=_REGIME_ORDER
    )

    # Equity curves
    gross_equity_s = (1 + gross_returns_s).cumprod()
    net_equity_s = (1 + net_returns_s).cumprod()

    # Compute metrics with gross returns for cost-drag
    metrics = compute_performance_metrics(
        net_returns_s,
        gross_returns=gross_returns_s,
        turnover=turnover_s,
        transaction_costs=costs_s,
        risk_free_rate=risk_free_rate,
        annualization_factor=config.annualization_factor,
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
        config_metadata=_build_metadata(config, prices, has_vix, dates_series, risk_free_rate),
    )

    return result


def run_benchmark(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    config: Optional[RegimeShiftConfig] = None,
    transaction_cost_bps: Optional[float] = None,
    rebalance_flags: Optional[pd.Series] = None,
    start_date: Optional[pd.Timestamp] = None,
    risk_free_rate: float = 0.0,
) -> BenchmarkResult:
    """
    Run a benchmark using the strategy's chronological execution convention.

    On each date, pre-trade weights are first used to calculate turnover.  On
    rebalance dates the target is installed before calculating that day's gross
    return, so the return is earned by the post-trade portfolio.  Transaction
    costs are then deducted multiplicatively and the portfolio is drifted after
    the return.
    """
    config = config or RegimeShiftConfig()
    if transaction_cost_bps is not None:
        config = _override_transaction_cost(config, transaction_cost_bps)
    config.validate()

    core_assets = config.core_assets
    price_data = prices[core_assets].copy()
    asset_returns = price_data.pct_change().iloc[1:]
    dates = asset_returns.index

    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
        asset_returns = asset_returns.loc[dates]

    if rebalance_flags is not None:
        rb_flags = rebalance_flags.reindex(dates, fill_value=False).astype(bool)
    else:
        rb_flags = pd.Series(False, index=dates, name="rebalance")
        if len(dates):
            rb_flags.iloc[0] = True
            rb_flags.iloc[config.rebalance_frequency::config.rebalance_frequency] = True

    target = np.array([weights[a] for a in core_assets], dtype=float)
    if not np.isclose(target.sum(), 1.0):
        raise ValueError("Benchmark target weights must sum to one.")

    cost_rate = config.transaction_cost_bps / 10000.0
    n = len(dates)
    gross_arr = np.zeros(n)
    net_arr = np.zeros(n)
    costs_arr = np.zeros(n)
    turnover_arr = np.zeros(n)
    drifted_arr = np.zeros((n, len(core_assets)))
    target_arr = np.zeros((n, len(core_assets)))

    current_weights = np.zeros(len(core_assets), dtype=float)
    has_allocation = False

    for pos, date in enumerate(dates):
        day_return = asset_returns.iloc[pos].to_numpy(dtype=float)
        rebalance_today = bool(rb_flags.iloc[pos])
        turnover = 0.0
        cost = 0.0

        if rebalance_today:
            if not has_allocation:
                turnover = float(np.abs(target).sum())
            else:
                turnover = 0.5 * float(np.abs(target - current_weights).sum())
            cost = turnover * cost_rate
            current_weights = target.copy()
            has_allocation = True
            target_arr[pos] = target

        # Rebalance-day returns use newly installed targets; non-rebalance days
        # use the pre-trade drifted weights.
        gross_ret = float(np.dot(current_weights, day_return))
        net_ret = (1.0 - cost) * (1.0 + gross_ret) - 1.0 if rebalance_today else gross_ret

        gross_arr[pos] = gross_ret
        net_arr[pos] = net_ret
        costs_arr[pos] = cost
        turnover_arr[pos] = turnover

        denominator = 1.0 + gross_ret
        if denominator <= 0.0:
            raise ValueError(f"Benchmark wealth became non-positive on {date}.")
        current_weights = current_weights * (1.0 + day_return) / denominator
        drifted_arr[pos] = current_weights

    gross_s = pd.Series(gross_arr, index=dates, name="gross_return")
    net_s = pd.Series(net_arr, index=dates, name="net_return")
    turnover_s = pd.Series(turnover_arr, index=dates, name="turnover")
    costs_s = pd.Series(costs_arr, index=dates, name="transaction_cost")
    rebalance_s = pd.Series(rb_flags.to_numpy(), index=dates, name="rebalance")
    drifted_df = pd.DataFrame(drifted_arr, index=dates, columns=core_assets)
    target_df = pd.DataFrame(target_arr, index=dates, columns=core_assets)

    metrics = compute_performance_metrics(
        net_s,
        gross_returns=gross_s,
        turnover=turnover_s,
        transaction_costs=costs_s,
        risk_free_rate=risk_free_rate,
        annualization_factor=config.annualization_factor,
    )

    return BenchmarkResult(
        name="benchmark",
        gross_returns=gross_s,
        net_returns=net_s,
        gross_equity=(1.0 + gross_s).cumprod(),
        net_equity=(1.0 + net_s).cumprod(),
        turnover=turnover_s,
        transaction_costs=costs_s,
        rebalance_flags=rebalance_s,
        daily_drifted_weights=drifted_df,
        target_weights=target_df,
        metrics=metrics.to_dict(),
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

    The first rebalance date is at `frequency` trading days from the start.
    """
    if len(dates) < frequency:
        return pd.DatetimeIndex([])

    indices = list(range(frequency, len(dates), frequency))
    if not indices:
        indices = [len(dates) - 1]

    return dates[indices]


def _build_metadata(
    config: RegimeShiftConfig,
    prices: pd.DataFrame,
    has_vix: bool,
    result_dates: pd.DatetimeIndex,
    risk_free_rate: float = 0.0,
) -> Dict:
    """Build run metadata dictionary."""
    diag = getattr(prices, "attrs", {}).get("data_diagnostics", {})
    hmm_conf = config.hmm_config
    # ``result_dates`` is the backtest horizon (may be shorter than the
    # full price history due to warmup / pre-roll).  ``rows_after_load``
    # captures the raw row count of the loaded CSV; both are reported
    # explicitly so downstream consumers can distinguish them.
    # Compute actual feature list used in this run
    features_used = [
        "equity_log_return_1d",
        "equity_momentum_21d",
        "equity_momentum_63d",
        "equity_volatility_21d",
        "equity_volatility_ratio_21_63",
        "equity_gold_correlation_63d",
        "equity_bond_correlation_63d",
    ]
    if has_vix:
        features_used += ["vix_change_5d", "vix_level"]

    # Tickers actually present in the dataset (filter by column presence)
    configured_tickers = {
        "equity": config.tickers.equity_ticker,
        "gold": config.tickers.gold_ticker,
        "bond": config.tickers.bond_ticker,
        "optional_vix": config.tickers.vix_ticker,
    }
    dataset_tickers = {
        col: tk for col, tk in zip(
            [config.equity_col, config.gold_col, config.bond_col, config.vix_col],
            [config.tickers.equity_ticker, config.tickers.gold_ticker,
             config.tickers.bond_ticker, config.tickers.vix_ticker],
        )
        if col in prices.columns
    }

    return {
        "date_range": {
            "start": str(result_dates[0]),
            "end": str(result_dates[-1]),
            "n_observations": len(result_dates),
            "original_start": str(prices.index[0]),
            "original_end": str(prices.index[-1]),
        },
        "row_counts": {
            "backtest_observations": len(result_dates),
            "rows_after_load": diag.get("rows_after_load", len(prices)),
        },
        "configured_tickers": configured_tickers,
        "dataset_tickers": dataset_tickers,
        "vix_included": has_vix,
        "transaction_cost_bps": config.transaction_cost_bps,
        "train_window": config.train_window,
        "rebalance_frequency": config.rebalance_frequency,
        "hmm_config": {
            "n_components": config.number_of_regimes,
            "covariance_type": "diag",
            "n_iter": hmm_conf.n_iter,
            "random_state": config.random_seed,
        },
        "features_used": features_used,
        "package_version": "1.0.0",
        "random_seed": config.random_seed,
        "risk_free_rate": risk_free_rate,
        "annualization_factor": config.annualization_factor,
        # Data diagnostics — populated by data.py
        "data_file": diag.get("data_path", ""),
        "data_sha256": diag.get("sha256", ""),
        "forward_fill_limit": config.data.forward_fill_limit,
        "ffill_cells_total": diag.get("ffill_cells_total", 0),
        "ffill_cells_per_asset": diag.get("ffill_cells_per_asset", {}),
        "dates_dropped": diag.get("dates_dropped", 0),
    }
