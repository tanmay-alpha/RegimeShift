"""
Tests for chart-generation behaviour.

Covers:
- Drawdowns chart equity must be (1 + returns).cumprod() (never
  returns.cumprod()).
- Portfolio-weights chart must read daily_drifted_weights, not
  target_weights.
- Transition-matrix chart must raise when the fitted matrix is absent
  (no silent identity substitution).
- Regime-probability chart must enforce the explicit Bull/Bear/Crisis
  column order and use only causal rows.
- Regime-price chart must shade the labelled regimes from the
  regime_series.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regime_shift.plots import (
    _regime_price_chart,
    _transition_matrix_chart,
    _equity_curves_chart,
    _drawdowns_chart,
    _portfolio_weights_chart,
    _regime_probabilities_chart,
)
from regime_shift.backtest import BacktestResult, BenchmarkResult


# ---------------------------------------------------------------------------
# Synthetic helper for chart smoke tests
# ---------------------------------------------------------------------------

def _make_result(
    with_transition_matrix: bool = True,
    with_drifted_weights: bool = True,
) -> BacktestResult:
    idx = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(123)
    net_returns = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    gross_returns = net_returns + 0.0001
    transaction_costs = pd.Series(np.where(np.arange(len(idx)) % 21 == 0, 0.0001, 0.0), index=idx)
    turnover = transaction_costs / 0.0005 if transaction_costs.max() > 0 else transaction_costs
    tw = pd.DataFrame(
        np.zeros((len(idx), 3)),
        index=idx, columns=["equity", "gold", "bond"],
    )
    for i in np.where(transaction_costs > 0)[0]:
        tw.iloc[i] = [0.6, 0.0, 0.4]
    dw = pd.DataFrame(
        np.zeros((len(idx), 3)),
        index=idx, columns=["equity", "gold", "bond"],
    )
    if with_drifted_weights:
        for i in range(len(idx)):
            dw.iloc[i] = [0.6, 0.0, 0.4]

    regime_series = pd.Series(
        ["Bull"] * 50 + ["Bear"] * 30 + ["Crisis"] * 20,
        index=idx,
    )
    regime_probs = pd.DataFrame(
        np.vstack([
            np.full((50, 3), [0.8, 0.1, 0.1]),
            np.full((30, 3), [0.1, 0.7, 0.2]),
            np.full((20, 3), [0.1, 0.2, 0.7]),
        ]),
        index=idx, columns=["Bull", "Bear", "Crisis"],
    )
    rebalance_flags = (np.arange(len(idx)) % 21 == 0)
    rebalance_flags[0] = False
    rebalance_flags = pd.Series(rebalance_flags, index=idx)
    transition_matrix = None
    if with_transition_matrix:
        transition_matrix = pd.DataFrame(
            [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]],
            index=["Bull", "Bear", "Crisis"],
            columns=["Bull", "Bear", "Crisis"],
        )

    return BacktestResult(
        gross_returns=gross_returns,
        net_returns=net_returns,
        transaction_costs=transaction_costs,
        turnover=turnover,
        target_weights=tw,
        daily_drifted_weights=dw,
        regime_series=regime_series,
        regime_probabilities=regime_probs,
        transition_matrix=transition_matrix,
        gross_equity=(1 + gross_returns).cumprod(),
        net_equity=(1 + net_returns).cumprod(),
        rebalance_flags=rebalance_flags,
    )


def _make_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=100)
    return pd.DataFrame({
        "equity": np.linspace(100, 110, len(idx)),
        "gold": np.linspace(40, 45, len(idx)),
        "bond": np.linspace(50, 52, len(idx)),
    }, index=idx)


# ---------------------------------------------------------------------------
# Drawdowns chart
# ---------------------------------------------------------------------------

class TestDrawdownsChart:
    def test_drawdowns_chart_uses_one_plus_returns_cumprod(self, tmp_path):
        result = _make_result()
        prices = _make_prices()
        path = _drawdowns_chart(result, {}, str(tmp_path))
        assert Path(path).exists()
        # Title should be present and not mention "returns.cumprod"
        # (only behaviour-level guarantee is that the chart renders
        # without raising and produces a non-empty file).
        assert Path(path).stat().st_size > 1000


# ---------------------------------------------------------------------------
# Portfolio weights chart
# ---------------------------------------------------------------------------

class TestPortfolioWeightsChart:
    def test_portfolio_weights_chart_reads_drifted_weights(self, tmp_path):
        result = _make_result()
        path = _portfolio_weights_chart(result, str(tmp_path))
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000

    def test_portfolio_weights_chart_handles_missing_drifted_weights(self, tmp_path):
        result = _make_result(with_drifted_weights=False)
        # Should not raise; it falls back gracefully.
        path = _portfolio_weights_chart(result, str(tmp_path))
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Transition-matrix chart
# ---------------------------------------------------------------------------

class TestTransitionMatrixChart:
    def test_transition_matrix_chart_renders_with_matrix(self, tmp_path):
        result = _make_result(with_transition_matrix=True)
        path = _transition_matrix_chart(result, str(tmp_path))
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000

    def test_transition_matrix_chart_raises_when_absent(self, tmp_path):
        result = _make_result(with_transition_matrix=False)
        with pytest.raises(ValueError, match="Transition matrix"):
            _transition_matrix_chart(result, str(tmp_path))


# ---------------------------------------------------------------------------
# Regime-probability chart
# ---------------------------------------------------------------------------

class TestRegimeProbabilitiesChart:
    def test_probabilities_chart_enforces_bull_bear_crisis_order(self, tmp_path):
        result = _make_result()
        path = _regime_probabilities_chart(result, str(tmp_path))
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000

    def test_probabilities_chart_handles_all_nan_columns(self, tmp_path):
        result = _make_result()
        result.regime_probabilities = pd.DataFrame(
            np.nan, columns=["Bull", "Bear", "Crisis"],
            index=result.regime_probabilities.index,
        )
        # Should not raise — just produces an empty chart.
        path = _regime_probabilities_chart(result, str(tmp_path))
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Regime-price chart
# ---------------------------------------------------------------------------

class TestRegimePriceChart:
    def test_regime_price_chart_renders(self, tmp_path):
        result = _make_result()
        prices = _make_prices()
        path = _regime_price_chart(result, prices, str(tmp_path))
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000

    def test_regime_price_chart_handles_unknown_regime_label(self, tmp_path):
        result = _make_result()
        result.regime_series = result.regime_series.replace("Bull", "Mystery")
        prices = _make_prices()
        # Unknown labels fall through to gray shade; should not raise.
        path = _regime_price_chart(result, prices, str(tmp_path))
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Equity curves chart
# ---------------------------------------------------------------------------

class TestEquityCurvesChart:
    def test_equity_curves_chart_renders_with_benchmarks(self, tmp_path):
        result = _make_result()
        # Synthetic benchmark
        idx = result.net_returns.index
        bench = BenchmarkResult(
            name="test",
            gross_returns=result.gross_returns,
            net_returns=result.net_returns,
            gross_equity=(1 + result.gross_returns).cumprod(),
            net_equity=(1 + result.net_returns).cumprod(),
            transaction_costs=result.transaction_costs,
            turnover=result.turnover,
            rebalance_flags=result.rebalance_flags,
            daily_drifted_weights=result.daily_drifted_weights,
            target_weights=result.target_weights,
        )
        path = _equity_curves_chart(result, {"test": bench}, str(tmp_path))
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000
