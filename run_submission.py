#!/usr/bin/env python
"""
Official submission entrypoint for IIT Bombay Summer Quant 2026 — RegimeShift.

Executes the full pipeline:

    Real NSE equity + gold + bond (+ optional VIX) prices
        ↓
    Leakage-safe feature engineering
        ↓
    Train-only StandardScaler
        ↓
    3-state Gaussian HMM (Bull / Bear / Crisis)
        ↓
    Sequential regime inference
        ↓
    CVXPY regime-conditioned portfolio optimization
        ↓
    Walk-forward backtest with transaction costs
        ↓
    Performance metrics + charts

Usage:
    python run_submission.py                    # default config
    python run_submission.py --start 2015-01-01 # custom start
    python run_submission.py --no-vix            # skip VIX
    python run_submission.py --output results/   # custom output dir
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure src/ is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from regime_shift.config import RegimeShiftConfig
from regime_shift.data import load_multi_asset_data
from regime_shift.features import compute_raw_features, drop_feature_warmup
from regime_shift.regime_model import fit_hmm, predict_current_state
from regime_shift.portfolio import optimize_portfolio
from regime_shift.benchmarks import static_60_40_weights, equal_weight_weights
from regime_shift.backtest import (
    run_walk_forward_backtest,
    run_benchmark,
)
from regime_shift.plots import generate_all_charts
from regime_shift.metrics import compute_performance_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="RegimeShift official submission runner (IIT Bombay Summer Quant 2026)"
    )
    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="Start date for data download (YYYY-MM-DD). Default: 2010-01-01",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date for data download (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--no-vix",
        action="store_true",
        help="Exclude VIX from features and pipeline.",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for charts and reports. Default: results/",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=None,
        help="Override transaction cost in basis points. Default: from config (5.0 bps).",
    )
    parser.add_argument(
        "--rebalance-freq",
        type=int,
        default=None,
        help="Rebalance frequency in trading days. Default: 21 (monthly).",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annualized risk-free rate for Sharpe/Sortino. Default: 0.0.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Execute the full RegimeShift pipeline.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("IIT Bombay Summer Quant 2026 — RegimeShift")
    logger.info("=" * 60)

    # ---- Configuration ----
    config = RegimeShiftConfig()

    if args.no_vix:
        logger.info("VIX excluded per --no-vix flag.")
        # VIX is optional; pipeline handles absence automatically
        # We signal absence by not requesting it in data loading

    if args.cost_bps is not None:
        config = _override(config, transaction_cost_bps=args.cost_bps)

    if args.rebalance_freq is not None:
        config = _override(config, rebalance_frequency=args.rebalance_freq)

    config.validate()
    logger.info("Configuration validated.")

    # ---- Step 1: Load data ----
    logger.info("Loading multi-asset price data...")
    try:
        prices = load_multi_asset_data(
            start=args.start,
            end=args.end,
            include_vix=not args.no_vix,
            config=config,
        )
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        return 1

    logger.info(
        "Loaded %d rows from %s to %s.",
        len(prices),
        prices.index[0].strftime("%Y-%m-%d"),
        prices.index[-1].strftime("%Y-%m-%d"),
    )

    # ---- Step 2: Run strategy backtest ----
    logger.info("Running walk-forward backtest...")
    try:
        strategy_result = run_walk_forward_backtest(
            prices=prices,
            config=config,
            transaction_cost_bps=args.cost_bps,
            risk_free_rate=args.risk_free_rate,
        )
    except Exception as exc:
        logger.error("Backtest failed: %s", exc, exc_info=True)
        return 1

    logger.info(
        "Strategy complete. Net CAGR: %.2f%%, Sharpe: %.2f, Max DD: %.2f%%",
        _pct(strategy_result.metrics.get("CAGR", 0)) if strategy_result.metrics else 0,
        strategy_result.metrics.get("Sharpe", 0) if strategy_result.metrics else 0,
        _pct(strategy_result.metrics.get("Maximum Drawdown", 0)) if strategy_result.metrics else 0,
    )

    # ---- Step 3: Run benchmarks ----
    logger.info("Running benchmarks...")
    bench_60_40 = run_benchmark(
        prices=prices,
        weights=static_60_40_weights(),
        config=config,
        transaction_cost_bps=args.cost_bps,
        rebalance_dates=strategy_result.target_weights.index,
    )

    bench_eq = run_benchmark(
        prices=prices,
        weights=equal_weight_weights(),
        config=config,
        transaction_cost_bps=args.cost_bps,
        rebalance_dates=strategy_result.target_weights.index,
    )

    benchmarks = {
        "Static 60/40": bench_60_40,
        "Equal Weight": bench_eq,
    }

    for name, bench in benchmarks.items():
        if bench.metrics:
            logger.info(
                "%s — CAGR: %.2f%%, Sharpe: %.2f, Max DD: %.2f%%",
                name,
                _pct(bench.metrics.get("CAGR", 0)),
                bench.metrics.get("Sharpe", 0),
                _pct(bench.metrics.get("Maximum Drawdown", 0)),
            )

    # ---- Step 4: Generate charts ----
    logger.info("Generating charts...")
    os.makedirs(args.output, exist_ok=True)
    try:
        saved_paths = generate_all_charts(
            result=strategy_result,
            benchmarks=benchmarks,
            prices=prices,
            output_dir=args.output,
        )
        for p in saved_paths:
            logger.info("Saved chart: %s", p)
    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)

    # ---- Step 5: Save metrics CSV ----
    _save_metrics(strategy_result, benchmarks, args.output)

    # ---- Step 6: Print summary ----
    _print_summary(strategy_result, benchmarks)

    logger.info("=" * 60)
    logger.info("RegimeShift pipeline complete.")
    logger.info("Results saved to: %s", args.output)
    return 0


def _override(config: RegimeShiftConfig, **kwargs) -> RegimeShiftConfig:
    """Return a config copy with overridden fields."""
    import copy
    new_config = copy.deepcopy(config)
    for key, value in kwargs.items():
        if hasattr(new_config, key):
            setattr(new_config, key, value)
    return new_config


def _pct(value: float) -> float:
    """Convert decimal to percentage for logging."""
    return float(value) * 100.0 if value is not None else 0.0


def _save_metrics(
    strategy_result,
    benchmarks: dict,
    output_dir: str,
) -> None:
    """Save performance metrics to CSV."""
    rows = []

    if strategy_result.metrics:
        row = {"Strategy": "RegimeShift"}
        row.update(strategy_result.metrics)
        rows.append(row)

    for name, bench in benchmarks.items():
        if bench.metrics:
            row = {"Strategy": name}
            row.update(bench.metrics)
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, "performance_metrics.csv")
        df.to_csv(path, index=False)
        logger.info("Saved metrics to %s", path)


def _print_summary(strategy_result, benchmarks: dict) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'RegimeShift':>12} {'60/40':>12} {'Equal Wt':>12}")
    print("-" * 70)

    all_metrics = {}
    if strategy_result.metrics:
        all_metrics["RegimeShift"] = strategy_result.metrics
    for name, bench in benchmarks.items():
        if bench.metrics:
            all_metrics[name] = bench.metrics

    metric_keys = [
        "Total Return", "CAGR", "Annualised Volatility",
        "Sharpe", "Sortino", "Maximum Drawdown", "Calmar",
        "Total Turnover",
    ]

    for key in metric_keys:
        vals = []
        for label in ["RegimeShift", "Static 60/40", "Equal Weight"]:
            v = all_metrics.get(label, {}).get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(f"{v:>12.4f}")
            else:
                vals.append(f"{'N/A':>12}")
        print(f"{key:<30} {vals[0]} {vals[1]} {vals[2]}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    sys.exit(main())
