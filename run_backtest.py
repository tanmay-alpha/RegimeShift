#!/usr/bin/env python3
"""
Main entry point for RegimeShift backtest.

Usage:
    python run_backtest.py                    # Simulated data
    python run_backtest.py --export           # Export CSV + plots
    python run_backtest.py --lookback 300     # Custom lookback
    python run_backtest.py --n-states 4       # Specify number of HMM regimes
    python run_backtest.py --simulate         # Simulated multi-asset data
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np

import config
from src.regime_shift.data_loader import (
    load_btc_data, compute_features_btc,
    _simulate_prices, compute_features, compute_returns, load_prices,
)
from src.regime_shift.regime_detector import RegimeDetector
from src.regime_shift.optimizer import PortfolioOptimizer
from src.regime_shift.transaction_costs import TransactionCostModel
from src.regime_shift.backtest import WalkForwardBacktest, BacktestResult
from src.regime_shift.benchmarks import run_benchmarks
from src.regime_shift.evaluate import compute_metrics, compute_regime_metrics, compute_turnover_metrics
from src.regime_shift.visualize import (
    plot_backtest_results, plot_turnover_costs,
    plot_regime_performance, plot_weight_evolution,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="RegimeShift walk-forward backtest")
    p.add_argument("--prices",        type=str, default=None,
                   help="Path to OHLCV CSV (default: btc_18_22_1d.csv)")
    p.add_argument("--simulate",      action="store_true",
                   help="Use simulated multi-asset data instead of BTC")
    p.add_argument("--select-nstates",action="store_true",
                   help="Auto-select n_states via BIC before backtest")
    p.add_argument("--n-states",      type=int, default=config.N_REGIMES,
                   help=f"Number of HMM states (default: {config.N_REGIMES})")
    p.add_argument("--persistence",   type=int, default=config.REGIME_PERSISTENCE,
                   help="Consecutive days to confirm regime change")
    p.add_argument("--rebalance",     type=str, default="1ME",
                   help="Rebalance frequency (pandas offset, e.g. 1ME, 1W, 5D)")
    p.add_argument("--window",        type=int, default=20,
                   help="Rolling window for feature computation")
    p.add_argument("--cost",          type=float, default=config.TRANSACTION_FEE,
                   help="Transaction cost per trade (proportional)")
    p.add_argument("--bootstrap",     action="store_true",
                   help="Run block bootstrap confidence intervals after backtest")
    p.add_argument("--export",        action="store_true",
                   help="Export CSV + plots to results/")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load data ──
    if args.simulate:
        logger.info("Using simulated multi-asset prices")
        prices = _simulate_prices()
        returns = compute_returns(prices)
        tickers = prices.columns.tolist()
    else:
        data_path = args.prices or config.DATA_PATH
        logger.info("Loading BTC data from %s", data_path)
        btc_df  = load_btc_data(data_path)
        btc_df["datetime"] = pd.to_datetime(btc_df["datetime"])
        prices  = btc_df.set_index("datetime")[["close"]].rename(columns={"close": "BTC"})
        returns = prices.pct_change().dropna()
        tickers = ["BTC"]

    # ── BIC state selection ──
    n_states = args.n_states
    if args.select_nstates:
        logger.info("Running BIC state selection...")
        if args.simulate:
            features = compute_features(returns, tickers, window=args.window)
        else:
            features = compute_features_btc(
                btc_df.reset_index() if "datetime" not in btc_df.columns else btc_df,
                window=args.window
            )
        det      = RegimeDetector(n_states=5, n_iter=20)
        n_states = det.select_n_states(features)
        logger.info("BIC selected n_states=%d", n_states)

    # ── Build features ──
    features = compute_features(returns, tickers, window=args.window)
    features = features.dropna()
    if len(features) == 0:
        logger.error("No valid features computed. Check data and window size.")
        sys.exit(1)

    # ── Walk-forward backtest ──
    logger.info(
        "Running walk-forward backtest (n_states=%d, persistence=%d, rebalance=%s)",
        n_states, args.persistence, args.rebalance,
    )
    detector = RegimeDetector(
        n_states=n_states,
        lookback=args.window,
        retrain_freq=21,
    )
    optimizer = PortfolioOptimizer(n_assets=len(tickers))
    cost_model = TransactionCostModel()

    wb = WalkForwardBacktest(
        prices=prices,
        returns=returns,
        features=features,
        lookback=args.window,
        retrain_freq=21,
        n_states=n_states,
        cost_model=cost_model,
        detector=detector,
        optimizer=optimizer,
        turnover_limit=0.20,
    )
    result = wb.run()

    # ── Compute metrics ──
    strategy_metrics = compute_metrics(result.portfolio_returns, name="RegimeShift")
    regime_metrics = compute_regime_metrics(result.portfolio_returns, result.regime_series)
    turnover_metrics = compute_turnover_metrics(result.weights_history, result.regime_series)

    # ── Run benchmarks ──
    logger.info("Running benchmarks...")
    bench_results = run_benchmarks(
        prices, returns, features,
        cost_model=cost_model,
        rebalance_freq=21,
        lookback=args.window,
    )

    # ── Print results ──
    print("\n" + "=" * 70)
    print("  REGIMESHIFT — WALK-FORWARD BACKTEST RESULTS")
    print("=" * 70)

    print(f"\n{'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Turnover':>10} {'Costs':>8}")
    print("-" * 70)

    all_results: dict[str, BacktestResult] = {"RegimeShift": result}
    all_results.update(bench_results)

    for name, res in all_results.items():
        m = compute_metrics(res.portfolio_returns, name=name)
        cost_drag = res.cost_drag if hasattr(res, 'cost_drag') else 0.0
        print(f"{name:<20} {m.total_return:>9.1%} {m.sharpe_ratio:>8.2f} "
              f"{m.max_drawdown:>9.1%} {res.turnover:>9.1f}x {cost_drag:>7.2%}")

    print("\n--- Regime Breakdown ---")
    if len(regime_metrics) > 0:
        print(regime_metrics.to_string())

    print(f"\n--- Turnover Analysis ---")
    print(f"  Avg daily turnover: {turnover_metrics['avg_daily_turnover']:.4f}")
    print(f"  Avg annual turnover: {turnover_metrics['avg_annual_turnover']:.1f}x")

    print(f"\n--- Regime Detection ---")
    print(f"  Regime changes: {result.regime_changes}")
    if len(result.regime_series) > 0:
        regime_counts = result.regime_series.value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(result.regime_series) * 100
            print(f"  {regime}: {count} days ({pct:.0f}%)")

    # ── Bootstrap CIs ──
    if args.bootstrap:
        from src.regime_shift.evaluate import print_confidence_intervals
        print()
        print_confidence_intervals(result)

    # ── Export ──
    if args.export:
        os.makedirs("results", exist_ok=True)
        result.portfolio_returns.to_csv("results/portfolio_returns.csv")
        result.weights_history.to_csv("results/weights_history.csv")
        result.regime_series.to_csv("results/regime_series.csv")
        result.costs.to_csv("results/transaction_costs.csv")

        plot_backtest_results(result, bench_results, output_path="results/backtest_chart.png")
        plot_turnover_costs(result, output_path="results/turnover_chart.png")
        plot_regime_performance(regime_metrics, output_path="results/regime_performance.png")
        plot_weight_evolution(result.weights_history, output_path="results/weights.png")
        logger.info("Results exported to results/")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
