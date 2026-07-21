#!/usr/bin/env python3
"""
RegimeShift — HMM Walk-Forward Backtest Runner

Runs the full HMM-based regime detection + walk-forward backtest.
By default operates on real BTC/USD data. Falls back to simulated multi-asset
data for algorithm testing.

Usage:
    python run_backtest.py                            # BTC data, 3-state HMM
    python run_backtest.py --simulate                 # Simulated multi-asset data
    python run_backtest.py --select-nstates           # Auto-select n_states via BIC
    python run_backtest.py --n-states 4               # Specify number of HMM states
    python run_backtest.py --bootstrap                # Bootstrap confidence intervals
"""

import argparse
import logging
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np

import config
from src.regime_shift.data_loader import (
    load_btc_data, compute_features_btc,
    _simulate_prices, compute_features, compute_returns,
)
from src.regime_shift.regime_detector import RegimeDetector
from src.regime_shift.backtest         import WalkForwardBacktest
from src.regime_shift.optimizer        import PortfolioOptimizer
from src.regime_shift.benchmarks       import run_benchmarks, compute_sharpe, compute_total_return, BenchmarkResult
from src.regime_shift.evaluate         import print_confidence_intervals

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
        # For walk-forward, treat BTC close as single-asset "price"
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

    # ── Walk-forward backtest ──
    logger.info(
        "Running walk-forward backtest (n_states=%d, persistence=%d, rebalance=%s)",
        n_states, args.persistence, args.rebalance,
    )
    wb = WalkForwardBacktest(
        prices=prices,
        tickers=tickers,
        rebalance_freq=args.rebalance,
        window_size=args.window,
        n_regimes=n_states,
        transaction_cost=args.cost,
        regime_persistence=args.persistence,
    )
    result = wb.run(returns)

    # ── Strategy metrics ──
    strategy_sharpe = wb.sharpe_ratio(result["equity_curve"])
    strategy_total  = float((1.0 + result["returns"].sum(axis=1)).prod() - 1.0)
    avg_turnover    = result["turnover"].mean()

    print("\n" + "=" * 55)
    print("           REGIMESHIFT — WALK-FORWARD BACKTEST")
    print("=" * 55)
    print(f"  Sharpe Ratio          : {strategy_sharpe:.3f}")
    print(f"  Total Return          : {strategy_total * 100:.2f}%")
    print(f"  Avg Daily Turnover    : {avg_turnover:.4f}")
    print(f"  Max Daily Turnover    : {result['turnover'].max():.4f}")

    regime_stats = wb.regime_statistics(result["regimes"])
    print("\n  Regime Breakdown:")
    for k, v in regime_stats.items():
        print(f"    {v['name']:8s}: freq={v['freq_pct']:.1f}%  "
              f"avg_duration={v['avg_duration_days']:.1f}d  "
              f"max_duration={v['max_duration_days']:.0f}d")

    # ── Benchmark comparison ──
    bench_results = run_benchmarks(prices, returns)
    strategy_start = result["returns"].index[0]
    strategy_end   = result["returns"].index[-1]

    aligned_bench = {}
    for name, bench in bench_results.items():
        aligned = BenchmarkResult(
            name=bench.name,
            returns=bench.returns.loc[strategy_start:strategy_end],
        )
        aligned_bench[name] = aligned

    print("\n" + "=" * 55)
    print("              BENCHMARK COMPARISON")
    print("=" * 55)
    print(f"  {'Metric':<30} {'Strategy':>10} {'Buy&Hold':>10} {'60/40':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    # Wrap strategy result in BenchmarkResult so the comparison
    # functions (which call .returns) work on both strategy and benchmarks.
    # Compute portfolio returns (multi-asset returns × weights) for benchmark
    # comparison. result["returns"] is a multi-asset DataFrame; we want the
    # scalar portfolio return series here.
    portfolio_returns = (result["returns"].values * result["weights"].values).sum(axis=1)
    portfolio_returns = pd.Series(
        portfolio_returns,
        index=result["returns"].index,
        name="strategy",
    )
    strategy_result = BenchmarkResult(
        name="Strategy",
        returns=portfolio_returns,
    )

    for metric_name, fn in [
        ("Sharpe Ratio",  compute_sharpe),
        ("Total Return",  compute_total_return),
    ]:
        vals = [fn(strategy_result)]
        for b in aligned_bench.values():
            vals.append(fn(b))
        print(f"  {metric_name:<30} {vals[0]:>10.3f} {vals[1]:>10.3f} {vals[2]:>10.3f}")

    # ── Bootstrap CIs ──
    if args.bootstrap:
        print()
        print_confidence_intervals(result)

    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
