"""Cost sensitivity sweep: rerun the full walk-forward backtest at 5 bps and 10 bps.

Requires the user to supply the same exact real-market dataset used for the
official run, via --data-path.  This script never auto-selects files via glob
or downloads synthetic data.

Usage:
    python scripts/cost_sensitivity.py --data-path data/submission_market_data.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

from regime_shift.config import RegimeShiftConfig
from regime_shift.data import load_market_data_csv
from regime_shift.backtest import run_walk_forward_backtest
from regime_shift.metrics import compute_performance_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cost sensitivity sweep at 5 and 10 bps"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the SAME real-market dataset used for the official run "
             "(e.g. data/submission_market_data.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for the sensitivity table.",
    )
    return parser.parse_args()


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read()
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()

    if not Path(args.data_path).exists():
        print(f"ERROR: data file not found: {args.data_path}", file=sys.stderr)
        return 1

    sha = sha256_of_file(args.data_path)
    print(f"Data file: {args.data_path}")
    print(f"Data SHA-256: {sha}")

    cfg = RegimeShiftConfig()
    print(f"Loading price data ...")
    prices = load_market_data_csv(path=args.data_path, config=cfg)
    print(f"Loaded {len(prices)} rows: {prices.index[0].date()} -> {prices.index[-1].date()}")

    rows = []
    for cost_bps in (5, 10):
        print(f"\n=== Cost: {cost_bps} bps ===")

        result = run_walk_forward_backtest(
            prices=prices,
            config=cfg,
            transaction_cost_bps=float(cost_bps),
        )

        metrics = compute_performance_metrics(
            returns=result.net_returns,
            gross_returns=result.gross_returns,
            turnover=result.turnover,
            transaction_costs=result.transaction_costs,
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
              f"MaxDD={metrics.maximum_drawdown:.4f}  "
              f"CostDrag={metrics.total_transaction_cost_drag:.4f}")

    df = pd.DataFrame(rows)
    print("\n=== Cost Sensitivity Summary ===")
    print(df.to_string(index=False))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "cost_sensitivity.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")

    # Also save a metadata sidecar for auditability
    sidecar = out_dir / "cost_sensitivity_metadata.json"
    diag = prices.attrs.get("data_diagnostics", {})
    meta = {
        "data_file": args.data_path,
        "data_sha256": sha,
        "first_date": str(prices.index[0].date()),
        "last_date": str(prices.index[-1].date()),
        "row_count": len(prices),
        "tickers": list(cfg.col_to_ticker.values()),
        "vix_included": False,
        "ffill_cells_per_asset": diag.get("ffill_cells_per_asset", {}),
        "dates_dropped": diag.get("dates_dropped", 0),
        "sensitivity_rows": rows,
    }
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"Wrote {sidecar}")

    return 0


if __name__ == "__main__":
    sys.exit(main())