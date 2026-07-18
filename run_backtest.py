#!/usr/bin/env python3
"""
RegimeShift — run a walk-forward backtest with optional BIC state selection.

Usage:
    python run_backtest.py                          # default: simulated data
    python run_backtest.py --prices path/to.csv     # real price data
    python run_backtest.py --select-nstates          # auto-select n_states via BIC
    python run_backtest.py --persistence 5           # require 5 consecutive days
"""

import argparse
import logging
import sys
import os

import numpy as np
import pandas as pd

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from regime_shift.data_loader import _simulate_prices, compute_returns, compute_features
from regime_shift.regime_detector import RegimeDetector
from regime_shift.backtest import WalkForwardBacktest
from regime_shift.optimizer import PortfolioOptimizer
from regime_shift.benchmarks import run_benchmarks, compute_sharpe, compute_total_return, BenchmarkResult
from regime_shift.evaluate import print_confidence_intervals

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="RegimeShift walk-forward backtest")
    p.add_argument("--prices", type=str, default=None,
                   help="Path to CSV with OHLCV data (datetime, open, high, low, close, volume)")
    p.add_argument("--select-nstates", action="store_true",
                   help="Auto-select number of HMM states via BIC before backtest")
    p.add_argument("--n-states", type=int, default=3,
                   help="Number of HMM states (ignored if --select-nstates)")
    p.add_argument("--persistence", type=int, default=3,
                   help="Consecutive days required to confirm a regime change")
    p.add_argument("--rebalance", type=str, default="1M",
                   help="Rebalance frequency (pandas offset, e.g. 1M, 1W, 5D)")
    p.add_argument("--window", type=int, default=20,
                   help="Rolling window for feature computation")
    p.add_argument("--cost", type=float, default=0.0015,
                   help="Transaction cost per trade (proportional)")
    p.add_argument("--bootstrap", action="store_true",
                   help="Run bootstrap confidence intervals after backtest")
    return p.parse_args()


def main():
    args = parse_args()

    # -- Load data --
    if args.prices:
        logger.info("Loading prices from %s", args.prices)
        prices = pd.read_csv(args.prices, parse_dates=["datetime"], index_col="datetime")
    else:
        logger.info("Using simulated multi-asset prices")
        prices = _simulate_prices()

    returns = compute_returns(prices) if "return" not in prices.columns.str.lower() else prices

    # -- Select n_states via BIC if requested --
    n_states = args.n_states
    if args.select_nstates:
        logger.info("Running BIC state selection...")
        features = compute_features(returns, returns.columns.tolist(), window=args.window)
        det = RegimeDetector(n_states=5, n_iter=20)
        n_states = det.select_n_states(features)
        logger.info("BIC selected n_states=%d", n_states)

    # -- Run walk-forward backtest --
    logger.info("Running walk-forward backtest (n_states=%d, persistence=%d, rebalance=%s)",
                n_states, args.persistence, args.rebalance)
    wb = WalkForwardBacktest(
        prices=prices,
        tickers=returns.columns.tolist(),
        rebalance_freq=args.rebalance,
        window_size=args.window,
        n_regimes=n_states,
        transaction_cost=args.cost,
        regime_persistence=args.persistence,
    )
    result = wb.run(returns)

    # -- Strategy metrics --
    strategy_sharpe = wb.sharpe_ratio(result["equity_curve"])
    strategy_total = float((1.0 + result["returns"].sum(axis=1)).prod() - 1.0)

    print("\n" + "=" * 50)
    print("           REGIME SHIFT STRATEGY")
    print("=" * 50)
    print(f"  Sharpe Ratio:          {strategy_sharpe:.3f}")
    print(f"  Total Return:          {strategy_total * 100:.2f}%")
    print(f"  Avg Turnover:          {result['turnover'].mean():.4f}")
    print(f"  Max Turnover:          {result['turnover'].max():.4f}")

    regime_stats = wb.regime_statistics(result["regimes"])
    for k, v in regime_stats.items():
        print(f"  {v['name']:>8}: freq={v['freq_pct']:.1f}%, "
              f"avg_duration={v['avg_duration_days']:.1f}d")

    # -- Benchmarks --
    bench_results = run_benchmarks(prices, returns)

    # Align benchmarks to the strategy's backtest period
    strategy_start = result["returns"].index[0]
    strategy_end = result["returns"].index[-1]
    aligned_bench = {}
    for name, bench in bench_results.items():
        aligned = BenchmarkResult(
            name=bench.name,
            returns=bench.returns.loc[strategy_start:strategy_end],
        )
        aligned_bench[name] = aligned

    print("\n" + "=" * 50)
    print("           BENCHMARK COMPARISON")
    print("=" * 50)
    print(f"  {'Metric':<30} {'Strategy':>10} {'Buy&Hold':>10} {'60/40':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    metrics = {
        "Sharpe Ratio": lambda r: compute_sharpe(r),
        "Total Return": lambda r: compute_total_return(r),
    }
    for metric_name, fn in metrics.items():
        vals = [fn(result)]
        for b in aligned_bench.values():
            vals.append(fn(b))
        print(f"  {metric_name:<30} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f}")

    # -- Bootstrap CIs --
    if args.bootstrap:
        print()
        print_confidence_intervals(result)

    print("=" * 50)


if __name__ == "__main__":
    main()
