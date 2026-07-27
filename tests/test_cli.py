"""
Tests for run_submission.py CLI behaviour.

Covers:
- Offline mode never imports or calls yfinance.
- Chart-generation failure returns nonzero exit code.
- Required output files are produced.
- performance_summary.csv has exactly six required rows.
"""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_submission import main as _cli_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_offline_csv(tmp_path: Path) -> Path:
    """Write a deterministic multi-asset CSV for offline mode."""
    idx = pd.bdate_range("2010-01-01", periods=600)
    rng = np.random.default_rng(7)
    eq = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, 600)))
    gold = 1800.0 + np.cumsum(rng.normal(0.0001, 0.006, 600))
    bond = 97.0 + np.cumsum(rng.normal(0.00005, 0.002, 600))
    df = pd.DataFrame({"equity": eq, "gold": gold, "bond": bond}, index=idx)
    csv_path = tmp_path / "prices.csv"
    df.to_csv(csv_path)
    return csv_path


def _run_cli_offline(tmp_path: Path) -> int:
    csv_path = _make_offline_csv(tmp_path)
    out_dir = tmp_path / "results"
    argv = [
        "run_submission.py",
        "--data-path", str(csv_path),
        "--transaction-cost-bps", "5.0",
        "--output-dir", str(out_dir),
        "--rebalance-freq", "21",
    ]
    with patch.object(sys, "argv", argv):
        return _cli_main()


# ---------------------------------------------------------------------------
# Offline mode must not call yfinance
# ---------------------------------------------------------------------------

class TestOfflineMode:

    def test_offline_never_calls_yfinance(self, tmp_path):
        """yfinance must not be imported when --data-path is given."""
        # Remove yfinance from sys.modules so any import raises ImportError.
        original = sys.modules.pop("yfinance", None)
        original_download = sys.modules.pop("yfinance.download", None)
        try:
            exit_code = _run_cli_offline(tmp_path)
            assert exit_code == 0, "Offline CLI should exit 0"
        except ImportError as exc:
            pytest.fail(f"yfinance was imported in offline mode: {exc}")
        finally:
            if original is not None:
                sys.modules["yfinance"] = original
            if original_download is not None:
                sys.modules["yfinance.download"] = original_download


# ---------------------------------------------------------------------------
# Required output files
# ---------------------------------------------------------------------------

class TestOutputFiles:

    def test_required_files_produced(self, tmp_path):
        exit_code = _run_cli_offline(tmp_path)
        assert exit_code == 0
        out = tmp_path / "results"
        required = [
            "performance_summary.csv",
            "daily_results.csv",
            "weights.csv",
            "regimes.csv",
            "transition_matrix.csv",
            "run_metadata.json",
            "regime_price_chart.png",
            "transition_matrix.png",
            "equity_curves.png",
            "drawdowns.png",
            "portfolio_weights.png",
            "regime_probabilities.png",
        ]
        for fname in required:
            assert (out / fname).exists(), f"Missing required file: {fname}"
            assert (out / fname).stat().st_size > 0, f"Empty file: {fname}"


# ---------------------------------------------------------------------------
# Performance summary row count
# ---------------------------------------------------------------------------

class TestPerformanceSummary:

    def test_exactly_six_rows(self, tmp_path):
        exit_code = _run_cli_offline(tmp_path)
        assert exit_code == 0
        csv_path = tmp_path / "results" / "performance_summary.csv"
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 6, (
            f"performance_summary.csv has {len(rows)} rows, expected 6"
        )

    def test_row_labels_match_required_six(self, tmp_path):
        exit_code = _run_cli_offline(tmp_path)
        assert exit_code == 0
        csv_path = tmp_path / "results" / "performance_summary.csv"
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        expected_labels = {
            "RegimeShift Gross",
            "RegimeShift Net",
            "Static 60/40 Gross",
            "Static 60/40 Net",
            "Equal Weight Gross",
            "Equal Weight Net",
        }
        actual_labels = {r.get("Strategy", "") for r in rows}
        assert actual_labels == expected_labels, (
            f"Row labels mismatch.\nExpected: {expected_labels}\nGot: {actual_labels}"
        )

    def test_daily_results_has_benchmark_columns(self, tmp_path):
        exit_code = _run_cli_offline(tmp_path)
        assert exit_code == 0
        csv_path = tmp_path / "results" / "daily_results.csv"
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        required_cols = [
            "strategy_gross_return",
            "strategy_net_return",
            "strategy_transaction_cost",
            "strategy_turnover",
            "strategy_regime",
            "strategy_rebalance_flag",
            "static_60_40_gross_return",
            "static_60_40_net_return",
            "equal_weight_gross_return",
            "equal_weight_net_return",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column in daily_results.csv: {col}"


# ---------------------------------------------------------------------------
# CLI exit codes
# ---------------------------------------------------------------------------

class TestExitCodes:

    def test_chart_failure_returns_nonzero(self, tmp_path):
        """If generate_all_charts raises, main() must return 1."""
        csv_path = _make_offline_csv(tmp_path)
        argv = [
            "run_submission.py",
            "--data-path", str(csv_path),
            "--transaction-cost-bps", "5.0",
            "--output-dir", str(tmp_path / "results_fail"),
            "--rebalance-freq", "21",
        ]
        # Reference the already-loaded run_submission module.
        rs_mod = sys.modules["run_submission"]
        with patch.object(sys, "argv", argv):
            with patch.object(rs_mod, "generate_all_charts",
                              side_effect=RuntimeError("chart boom")):
                rc = _cli_main()
        assert rc == 1, "CLI must return nonzero exit code on chart failure"
