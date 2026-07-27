"""Cost sensitivity sweep: rerun the full walk-forward backtest at 5 bps and 10 bps.

Loads the cached CSV price data from data/cache/, or downloads fresh if missing.
Then re-runs the full walk-forward backtest at the two transaction-cost levels.
Writes a comparison table to results/cost_sensitivity.csv.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

from regime_shift.config import RegimeShiftConfig
from regime_shift.data import download_market_data, load_market_data_csv
from regime_shift.backtest import run_walk_forward_backtest
from regime_shift.metrics import compute_performance_metrics


def _load_prices(cfg: RegimeShiftConfig) -> pd.DataFrame:
    # 1. Try cached CSV files in data/cache/
    cache_dir = Path(cfg.data.cache_dir) if cfg.data.cache_dir else None
    cached = sorted(glob.glob(str(cache_dir / "market_data_*.csv"))) if cache_dir else []
    if cached:
        print(f"Loading cached data from {cached[-1]} ...")
        return load_market_data_csv(cached[-1], config=cfg)
    # 2. Fallback: download online
    print("No cached file found — downloading online ...")
    return download_market_data(config=cfg, cache=True)


def main() -> int:
    cfg = RegimeShiftConfig()

    print(f"Loading price data ...")
    prices = _load_prices(cfg)
    print(f"Loaded {len(prices)} rows: {prices.index[0].date()} -> {prices.index[-1].date()}")

    rows = []
    for cost_bps in (5, 10):
        print(f"\n=== Cost: {cost_bps} bps ===")

        result = run_walk_forward_backtest(
            prices=prices,
            transaction_cost_bps=float(cost_bps),
        )

        # Use Net returns (post-cost) — attribute names are snake_case
        metrics = compute_performance_metrics(
            returns=result.net_returns,
            gross_returns=result.gross_returns,
            turnover=result.turnover,
        )

        rows.append({
            "Cost (bps)": cost_bps,
            "Sharpe": round(metrics.sharpe_ratio, 3),
            "Sortino": round(metrics.sortino_ratio, 3),
            "CAGR": round(metrics.cagr, 4),
            "Max Drawdown": round(metrics.maximum_drawdown, 4),
            "Calmar": round(metrics.calmar_ratio, 3),
            "Annualised Vol": round(metrics.annualized_volatility, 4),
            "Total Return": round(metrics.total_return, 4),
            "Total Turnover": round(metrics.total_turnover, 3),
            "Annualised Turnover": round(metrics.annualized_turnover, 3),
            "Cost Drag": round(metrics.total_transaction_cost_drag, 4),
        })
        print(f"  Sharpe={metrics.sharpe_ratio:.3f}  CAGR={metrics.cagr:.4f}  "
              f"MaxDD={metrics.maximum_drawdown:.4f}  CostDrag={metrics.total_transaction_cost_drag:.4f}")

    df = pd.DataFrame(rows)
    print("\n=== Cost Sensitivity Summary ===")
    print(df.to_string(index=False))

    out = Path("results/cost_sensitivity.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())