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
    python run_submission.py --data-path data/prices.csv   # offline mode
    python run_submission.py --start 2015-01-01 --end 2024-12-31  # online mode
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
from regime_shift.data import download_market_data, load_market_data_csv
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
        "--data-path",
        default=None,
        help="Path to CSV file with pre-downloaded price data (offline mode). "
             "When provided, yfinance is never called.",
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
        "--include-vix",
        action="store_true",
        help="Include VIX in features and pipeline.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable data caching.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=5.0,
        help="Transaction cost in basis points. Official runs: 5-10 bps. Default: 5.0",
    )
    parser.add_argument(
        "--rebalance-freq",
        type=int,
        default=21,
        help="Rebalance frequency in trading days. Default: 21 (monthly).",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annualized risk-free rate for Sharpe/Sortino. Default: 0.0.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for charts and reports. Default: results/",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.transaction_cost_bps < 5 or args.transaction_cost_bps > 10:
        logger.warning(
            "Transaction cost %.1f bps is outside the official range [5, 10] bps. "
            "Official submissions should use 5-10 bps.",
            args.transaction_cost_bps,
        )


def main() -> int:
    """
    Execute the full RegimeShift pipeline.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    args = parse_args()
    validate_args(args)

    logger.info("=" * 60)
    logger.info("IIT Bombay Summer Quant 2026 — RegimeShift")
    logger.info("=" * 60)

    # ---- Configuration ----
    config = RegimeShiftConfig()
    config.rebalance_frequency = args.rebalance_freq
    config.transaction_cost_bps = args.transaction_cost_bps
    config.validate()
    logger.info("Configuration validated.")

    # ---- Step 1: Load data ----
    logger.info("Loading multi-asset price data...")
    try:
        if args.data_path is not None:
            # Offline mode: load from CSV, never call yfinance
            logger.info("Offline mode: loading from %s", args.data_path)
            prices = load_market_data_csv(path=args.data_path, config=config)
        else:
            # Online mode: download from yfinance
            logger.info(
                "Online mode: downloading %s → %s", args.start, args.end or "today"
            )
            prices = download_market_data(
                config=config,
                start=args.start,
                end=args.end,
                include_vix=args.include_vix,
                cache=not args.no_cache,
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
            transaction_cost_bps=args.transaction_cost_bps,
            risk_free_rate=args.risk_free_rate,
        )
    except Exception as exc:
        logger.error("Backtest failed: %s", exc, exc_info=True)
        return 1

    if strategy_result.metrics:
        logger.info(
            "Strategy complete. Net CAGR: %.2f%%, Sharpe: %.2f, Max DD: %.2f%%",
            _pct(strategy_result.metrics.get("CAGR", 0)),
            strategy_result.metrics.get("Sharpe", 0) or 0,
            _pct(strategy_result.metrics.get("Maximum Drawdown", 0)),
        )

    # ---- Step 3: Run benchmarks ----
    logger.info("Running benchmarks...")

    # Benchmarks use the strategy's rebalance flags and start date
    strategy_rb_flags = strategy_result.rebalance_flags
    strategy_start = strategy_result.net_returns.index[0]

    bench_60_40 = run_benchmark(
        prices=prices,
        weights=static_60_40_weights(),
        config=config,
        transaction_cost_bps=args.transaction_cost_bps,
        rebalance_flags=strategy_rb_flags,
        start_date=strategy_start,
    )

    bench_eq = run_benchmark(
        prices=prices,
        weights=equal_weight_weights(),
        config=config,
        transaction_cost_bps=args.transaction_cost_bps,
        rebalance_flags=strategy_rb_flags,
        start_date=strategy_start,
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
                bench.metrics.get("Sharpe", 0) or 0,
                _pct(bench.metrics.get("Maximum Drawdown", 0)),
            )

    # ---- Step 4: Generate charts ----
    logger.info("Generating charts...")
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        saved_paths = generate_all_charts(
            result=strategy_result,
            benchmarks=benchmarks,
            prices=prices,
            output_dir=args.output_dir,
        )
        for p in saved_paths:
            logger.info("Saved chart: %s", p)
    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)

    # ---- Step 5: Save results ----
    _save_results(strategy_result, benchmarks, args.output_dir)

    # ---- Step 6: Print summary ----
    _print_summary(strategy_result, benchmarks)

    logger.info("=" * 60)
    logger.info("RegimeShift pipeline complete.")
    logger.info("Results saved to: %s", args.output_dir)
    return 0


def _override(config: RegimeShiftConfig, **kwargs) -> RegimeShiftConfig:
    """Return a config copy with overridden fields."""
    import copy
    new_config = copy.deepcopy(config)
    for key, value in kwargs.items():
        if hasattr(new_config, key):
            setattr(new_config, key, value)
    return new_config


def _pct(value: Optional[float]) -> float:
    """Convert decimal to percentage for logging."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value) * 100.0


def _save_results(
    strategy_result: "BacktestResult",
    benchmarks: dict,
    output_dir: str,
) -> None:
    """Save all result files to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Daily results
    daily = pd.DataFrame({
        "gross_return": strategy_result.gross_returns,
        "net_return": strategy_result.net_returns,
        "transaction_cost": strategy_result.transaction_costs,
        "turnover": strategy_result.turnover,
        "regime": strategy_result.regime_series,
        "rebalance": strategy_result.rebalance_flags,
    })
    daily.to_csv(os.path.join(output_dir, "daily_results.csv"))

    # Weights
    strategy_result.target_weights.to_csv(os.path.join(output_dir, "weights.csv"))

    # Regimes
    regime_df = pd.DataFrame({
        "regime": strategy_result.regime_series,
        **strategy_result.regime_probabilities.to_dict(),
    })
    regime_df.to_csv(os.path.join(output_dir, "regimes.csv"))

    # Transition matrix
    strategy_result.transition_matrix.to_csv(os.path.join(output_dir, "transition_matrix.csv"))

    # Performance summary (six rows)
    rows = []
    for label, is_gross in [
        ("RegimeShift Gross", True),
        ("RegimeShift Net", False),
        ("Static 60/40 Gross", True),
        ("Static 60/40 Net", False),
        ("Equal Weight Gross", True),
        ("Equal Weight Net", False),
    ]:
        row = {"Strategy": label}
        if label.startswith("RegimeShift"):
            if is_gross:
                # Compute gross metrics from strategy
                gross_metrics = compute_performance_metrics(
                    strategy_result.gross_returns,
                    turnover=strategy_result.turnover,
                    risk_free_rate=0.0,
                )
                row.update(gross_metrics.to_dict())
            else:
                row.update(strategy_result.metrics or {})
        else:
            bench_name = label.replace(" Gross", "").replace(" Net", "")
            bench = benchmarks.get(bench_name)
            if bench:
                if is_gross:
                    gross_metrics = compute_performance_metrics(
                        bench.gross_returns,
                        turnover=bench.turnover,
                        risk_free_rate=0.0,
                    )
                    row.update(gross_metrics.to_dict())
                else:
                    row.update(bench.metrics or {})
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "performance_summary.csv"), index=False
    )

    # Run metadata
    metadata = strategy_result.config_metadata
    pd.Series(metadata).to_json(
        os.path.join(output_dir, "run_metadata.json"), indent=2, default=str
    )

    logger.info("Saved all results to %s", output_dir)


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
