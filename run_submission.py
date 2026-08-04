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
    python run_submission.py --data-path data/submission_market_data.csv --transaction-cost-bps 5 --output-dir results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

# Ensure src/ is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from regime_shift.config import RegimeShiftConfig
from regime_shift.data import load_market_data_csv
from regime_shift.benchmarks import static_60_40_weights, equal_weight_weights
from regime_shift.backtest import (
    run_walk_forward_backtest,
    run_benchmark,
)
from regime_shift.plots import generate_all_charts
from regime_shift.metrics import compute_performance_metrics
from regime_shift.execution import ExecutionCostModel
from regime_shift.experiment import sha256_file, write_manifest

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
        required=True,
        help="Path to CSV file with pre-downloaded price data. "
             "Required for submission runs. Example: data/submission_market_data.csv",
    )
    parser.add_argument(
        "--include-vix",
        action="store_true",
        help="Include VIX in features and pipeline.",
    )
    cost_group = parser.add_mutually_exclusive_group()
    cost_group.add_argument(
        "--transaction-cost-bps", type=float, default=None,
        help="Flat per-asset one-way cost override in basis points.",
    )
    cost_group.add_argument(
        "--cost-scenario",
        choices=["optimistic", "base", "stressed"],
        default=None,
        help="Named one-way asset-level assumption: 5, 10, or 20 bps.",
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


def sha256_of_file(path: str) -> str:
    """Backward-compatible alias for the canonical manifest hash policy."""
    return sha256_file(path)


def main() -> int:
    """Execute the full RegimeShift pipeline."""
    args = parse_args()

    if args.transaction_cost_bps is not None and args.transaction_cost_bps < 0:
        logger.warning(
            "Transaction cost %.1f bps is negative.",
            args.transaction_cost_bps,
        )

    logger.info("=" * 60)
    logger.info("IIT Bombay Summer Quant 2026 - RegimeShift")
    logger.info("=" * 60)

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path.resolve())
        return 1

    sha = sha256_of_file(str(data_path))
    logger.info("Data file: %s", data_path.name)
    logger.info("Data SHA-256: %s", sha)

    # ---- Configuration ----
    config = RegimeShiftConfig()
    config.rebalance_frequency = args.rebalance_freq
    cost_override = args.transaction_cost_bps
    scenario = args.cost_scenario or "base"
    if cost_override is None:
        config.cost_scenario = scenario
        config.execution_cost_model = ExecutionCostModel.scenario(
            scenario, config.core_assets
        )
    else:
        config.transaction_cost_bps = cost_override
    config.validate()
    logger.info("Configuration validated.")

    # ---- Step 1: Load data ----
    logger.info("Loading multi-asset price data...")
    try:
        prices = load_market_data_csv(path=str(data_path), config=config)
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        return 1
    if args.include_vix and config.vix_col not in prices.columns:
        logger.error("--include-vix requires a valid 'vix' column in the supplied data.")
        return 1
    if not args.include_vix and config.vix_col in prices.columns:
        prices = prices.drop(columns=[config.vix_col])

    # Annotate diagnostics with SHA-256
    diag = prices.attrs.get("data_diagnostics", {})
    diag["sha256"] = sha
    prices.attrs["data_diagnostics"] = diag

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
            transaction_cost_bps=cost_override,
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
    strategy_rb_flags = strategy_result.rebalance_flags
    strategy_start = strategy_result.net_returns.index[0]

    bench_60_40 = run_benchmark(
        prices=prices,
        weights=static_60_40_weights(),
        config=config,
        transaction_cost_bps=cost_override,
        rebalance_flags=strategy_rb_flags,
        start_date=strategy_start,
        risk_free_rate=args.risk_free_rate,
    )
    bench_eq = run_benchmark(
        prices=prices,
        weights=equal_weight_weights(),
        config=config,
        transaction_cost_bps=cost_override,
        rebalance_flags=strategy_rb_flags,
        start_date=strategy_start,
        risk_free_rate=args.risk_free_rate,
    )

    benchmarks = {
        "Static 60/40": bench_60_40,
        "Equal Weight": bench_eq,
    }
    # The common strategy horizon is the sole evaluation horizon.  A benchmark
    # must never be allowed to pick up an extra pre-allocation history.
    strategy_index = strategy_result.net_returns.index
    for benchmark in benchmarks.values():
        for attr in ("gross_returns", "net_returns", "transaction_costs", "turnover", "rebalance_flags"):
            series = getattr(benchmark, attr).reindex(strategy_index)
            if series.isna().any():
                raise ValueError(f"Benchmark {attr} has missing observations on the strategy evaluation index.")
            setattr(benchmark, attr, series)
        benchmark.gross_equity = (1.0 + benchmark.gross_returns).cumprod()
        benchmark.net_equity = (1.0 + benchmark.net_returns).cumprod()
    assert strategy_index.equals(bench_60_40.net_returns.index)
    assert strategy_index.equals(bench_eq.net_returns.index)

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
        logger.error("Chart generation failed: %s", exc, exc_info=True)
        return 1

    # ---- Step 5: Save results ----
    _save_results(
        strategy_result=strategy_result,
        benchmarks=benchmarks,
        output_dir=args.output_dir,
        risk_free_rate=args.risk_free_rate,
        annualization_factor=config.annualization_factor,
        data_path=str(data_path),
        data_sha256=sha,
    )

    _print_summary(strategy_result, benchmarks)

    logger.info("=" * 60)
    logger.info("RegimeShift pipeline complete.")
    logger.info("Results saved to: %s", args.output_dir)
    return 0


def _pct(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value) * 100.0


def _save_results(
    strategy_result,
    benchmarks: dict,
    output_dir: str,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    data_path: str = "",
    data_sha256: str = "",
) -> None:
    """Save all result files to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    daily = pd.DataFrame({
        "strategy_gross_return": strategy_result.gross_returns,
        "strategy_net_return": strategy_result.net_returns,
        "strategy_transaction_cost": strategy_result.transaction_costs,
        "strategy_turnover": strategy_result.turnover,
        "strategy_regime": strategy_result.regime_series,
        "strategy_rebalance_flag": strategy_result.rebalance_flags,
    })
    if "Static 60/40" in benchmarks:
        b = benchmarks["Static 60/40"]
        daily["static_60_40_gross_return"] = b.gross_returns
        daily["static_60_40_net_return"] = b.net_returns
    if "Equal Weight" in benchmarks:
        b = benchmarks["Equal Weight"]
        daily["equal_weight_gross_return"] = b.gross_returns
        daily["equal_weight_net_return"] = b.net_returns
    daily.to_csv(os.path.join(output_dir, "daily_results.csv"))
    strategy_result.event_log.to_csv(
        os.path.join(output_dir, "execution_timeline.csv")
    )

    strategy_result.target_weights.to_csv(os.path.join(output_dir, "weights.csv"))

    regime_df = pd.DataFrame({
        "regime": strategy_result.regime_series,
        **strategy_result.regime_probabilities.to_dict(),
    })
    regime_df.to_csv(os.path.join(output_dir, "regimes.csv"))

    strategy_result.transition_matrix.to_csv(
        os.path.join(output_dir, "transition_matrix.csv")
    )

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
            returns = strategy_result.gross_returns if is_gross else strategy_result.net_returns
            metrics = compute_performance_metrics(
                returns,
                gross_returns=strategy_result.gross_returns,
                turnover=strategy_result.turnover,
                transaction_costs=strategy_result.transaction_costs,
                risk_free_rate=risk_free_rate,
                annualization_factor=annualization_factor,
            )
            row.update(metrics.to_dict())
        else:
            bench_name = label.replace(" Gross", "").replace(" Net", "")
            bench = benchmarks.get(bench_name)
            if bench is not None:
                returns = bench.gross_returns if is_gross else bench.net_returns
                metrics = compute_performance_metrics(
                    returns,
                    gross_returns=bench.gross_returns,
                    turnover=bench.turnover,
                    transaction_costs=bench.transaction_costs,
                    risk_free_rate=risk_free_rate,
                    annualization_factor=annualization_factor,
                )
                row.update(metrics.to_dict())
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "performance_summary.csv"), index=False
    )

    metadata = dict(strategy_result.config_metadata or {})
    metadata["risk_free_rate"] = float(risk_free_rate)
    metadata["annualization_factor"] = int(annualization_factor)
    metadata["data_file"] = data_path
    metadata["dataset_path"] = str(Path(data_path).as_posix())
    metadata["dataset_sha256_canonical"] = data_sha256
    metadata["dataset_hash_policy"] = "CRLF and LF normalized to LF before hashing"
    metadata.pop("data_sha256", None)
    metadata["rebalance_count"] = int(strategy_result.rebalance_flags.sum())
    metadata["regime_counts"] = (
        strategy_result.regime_series.value_counts().to_dict()
    )

    def _json_default(obj):
        return str(obj)

    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=_json_default)

    write_manifest(
        output_dir,
        dataset_path=data_path,
        config={
            "execution_model": metadata.get("execution_model"),
            "transaction_cost_bps": metadata.get("transaction_cost_bps"),
            "cost_model": metadata.get("cost_model"),
            "train_window": metadata.get("train_window"),
            "rebalance_frequency": metadata.get("rebalance_frequency"),
            "risk_free_rate": risk_free_rate,
        },
        metadata=metadata,
    )

    logger.info("Saved all results to %s", output_dir)


def _print_summary(strategy_result, benchmarks: dict) -> None:
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
