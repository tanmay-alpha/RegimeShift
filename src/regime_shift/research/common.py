"""Shared loading, execution, and serialisation helpers for research runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from regime_shift.backtest import BacktestResult, BenchmarkResult, run_benchmark, run_walk_forward_backtest
from regime_shift.benchmarks import equal_weight_weights, static_60_40_weights
from regime_shift.config import RegimeShiftConfig
from regime_shift.data import load_market_data_csv
from regime_shift.execution import ExecutionCostModel
from regime_shift.metrics import compute_performance_metrics


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data" / "submission_market_data.csv"


@dataclass
class ResearchRun:
    strategy: BacktestResult
    benchmarks: Dict[str, BenchmarkResult]


def base_config() -> RegimeShiftConfig:
    config = RegimeShiftConfig()
    config.cost_scenario = "base"
    config.execution_cost_model = ExecutionCostModel.scenario("base", config.core_assets)
    config.validate()
    return config


def load_canonical_prices(path: Path = DATASET) -> pd.DataFrame:
    return load_market_data_csv(str(path), config=base_config())


def run_full_and_benchmarks(prices: pd.DataFrame, policy: Optional[object] = None) -> ResearchRun:
    """Run one policy and common-index benchmarks through causal engine code."""
    config = base_config()
    strategy = run_walk_forward_backtest(prices, config=config, policy=policy)
    flags, start = strategy.rebalance_flags, strategy.net_returns.index[0]
    benchmarks = {
        "Static 60/40": run_benchmark(prices, static_60_40_weights(), config=config, rebalance_flags=flags, start_date=start),
        "Equal Weight": run_benchmark(prices, equal_weight_weights(), config=config, rebalance_flags=flags, start_date=start),
    }
    index = strategy.net_returns.index
    for benchmark in benchmarks.values():
        for attr in ("gross_returns", "net_returns", "transaction_costs", "turnover", "rebalance_flags"):
            setattr(benchmark, attr, getattr(benchmark, attr).reindex(index))
        benchmark.gross_equity = (1 + benchmark.gross_returns).cumprod()
        benchmark.net_equity = (1 + benchmark.net_returns).cumprod()
        if benchmark.net_returns.isna().any():
            raise ValueError("Benchmark did not align to the strategy evaluation index.")
        benchmark.metrics = compute_performance_metrics(
            benchmark.net_returns,
            gross_returns=benchmark.gross_returns,
            turnover=benchmark.turnover,
            transaction_costs=benchmark.transaction_costs,
            annualization_factor=config.annualization_factor,
        ).to_dict()
    return ResearchRun(strategy, benchmarks)
