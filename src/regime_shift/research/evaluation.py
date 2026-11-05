"""Strategy runs with full accounting and strict paired-sample alignment.

A :class:`StrategyRun` always carries gross returns, net returns, transaction
costs and turnover on one index.  Comparisons go through
:func:`require_common_index`, which raises instead of reindexing or filling,
so a paired statistic can never be computed on manufactured observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from regime_shift.metrics import compute_performance_metrics
from regime_shift.research.spec import ANNUALIZATION_FACTOR


class MisalignedSamplesError(ValueError):
    """Raised when strategies that must be compared do not share one date index."""


_SERIES_FIELDS = ("gross", "net", "costs", "turnover")


@dataclass(frozen=True)
class StrategyRun:
    """Daily accounting for one strategy. All series share one DatetimeIndex."""

    strategy_id: str
    gross: pd.Series
    net: pd.Series
    costs: pd.Series
    turnover: pd.Series
    weights: Optional[pd.DataFrame] = field(default=None, repr=False)  # end-of-day post-trade
    info: Dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        index = self.net.index
        if not isinstance(index, pd.DatetimeIndex) or not index.is_monotonic_increasing or index.has_duplicates:
            raise ValueError(f"{self.strategy_id}: index must be a sorted, unique DatetimeIndex.")
        for name in _SERIES_FIELDS:
            series = getattr(self, name)
            if not series.index.equals(index):
                raise MisalignedSamplesError(f"{self.strategy_id}: {name} index differs from net index.")
            if not np.all(np.isfinite(series.to_numpy(dtype=float))):
                raise ValueError(f"{self.strategy_id}: {name} contains NaN or inf.")
        if self.weights is not None and not self.weights.index.equals(index):
            raise MisalignedSamplesError(f"{self.strategy_id}: weights index differs from net index.")
        implied = (1.0 + self.gross) * (1.0 - self.costs) - 1.0
        if not np.allclose(implied.to_numpy(), self.net.to_numpy(), atol=1e-12, rtol=0.0):
            raise ValueError(f"{self.strategy_id}: net != (1 + gross) * (1 - cost) - 1.")

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.net.index)

    def window(self, start, end) -> "StrategyRun":
        """Calendar slice of this continuous run (chronological attribution, not a re-fit)."""
        mask = (self.index >= pd.Timestamp(start)) & (self.index <= pd.Timestamp(end))
        if not mask.any():
            raise ValueError(f"{self.strategy_id}: no observations in [{start}, {end}].")
        return StrategyRun(
            self.strategy_id,
            *(getattr(self, name)[mask] for name in _SERIES_FIELDS),
            weights=None if self.weights is None else self.weights[mask],
            info=self.info,
        )

    def on_index(self, index: pd.DatetimeIndex) -> "StrategyRun":
        """Restrict to ``index``; every requested date must already exist (no filling)."""
        missing = pd.DatetimeIndex(index).difference(self.index)
        if len(missing):
            raise MisalignedSamplesError(
                f"{self.strategy_id}: {len(missing)} requested dates absent, first {list(missing[:3])}."
            )
        return StrategyRun(
            self.strategy_id,
            *(getattr(self, name).loc[index] for name in _SERIES_FIELDS),
            weights=None if self.weights is None else self.weights.loc[index],
            info=self.info,
        )

    @classmethod
    def from_result(cls, strategy_id: str, result, info: Optional[Dict] = None) -> "StrategyRun":
        """Build from a ``BacktestResult``, ``BenchmarkResult`` or ``ExecutionTrace``."""
        if hasattr(result, "net_returns"):
            gross, net, costs = result.gross_returns, result.net_returns, result.transaction_costs
            events = result.event_log
            cols = [c for c in events.columns if c.startswith("end_of_day_post_trade_")]
            weights = events[cols].rename(columns=lambda c: c[len("end_of_day_post_trade_"):-len("_weight")])
        else:
            gross, net, costs, weights = result.gross, result.net, result.costs, result.post_trade
        return cls(
            strategy_id,
            gross.astype(float).rename("gross_return"), net.astype(float).rename("net_return"),
            costs.astype(float).rename("transaction_cost"), result.turnover.astype(float).rename("turnover"),
            weights=weights, info=dict(info or {}),
        )


def require_common_index(runs: Iterable[StrategyRun]) -> pd.DatetimeIndex:
    """Return the shared index of ``runs`` or raise :class:`MisalignedSamplesError`."""
    runs = list(runs)
    if not runs:
        raise ValueError("No strategy runs supplied.")
    reference = runs[0]
    for run in runs[1:]:
        if not run.index.equals(reference.index):
            only_ref = reference.index.difference(run.index)
            only_run = run.index.difference(reference.index)
            raise MisalignedSamplesError(
                f"{run.strategy_id} ({len(run.index)} obs, {run.index[0].date()}..{run.index[-1].date()}) "
                f"vs {reference.strategy_id} ({len(reference.index)} obs, "
                f"{reference.index[0].date()}..{reference.index[-1].date()}): "
                f"{len(only_ref)} dates only in {reference.strategy_id}, {len(only_run)} only in {run.strategy_id}."
            )
    return reference.index


def net_return_frame(runs: Sequence[StrategyRun]) -> pd.DataFrame:
    """Aligned net-return matrix (dates x strategies); raises on any misalignment."""
    index = require_common_index(runs)
    return pd.DataFrame({run.strategy_id: run.net.to_numpy() for run in runs}, index=index)


def metric_row(run: StrategyRun, cash_returns: Optional[pd.Series] = None,
               annualization_factor: int = ANNUALIZATION_FACTOR) -> Dict:
    """Metrics defined in :mod:`regime_shift.research.spec` for one run."""
    m = compute_performance_metrics(
        run.net, gross_returns=run.gross, turnover=run.turnover, transaction_costs=run.costs,
        risk_free_rate=0.0, annualization_factor=annualization_factor,
    )
    n = len(run.net)
    row = {
        "strategy": run.strategy_id, "n_obs": n,
        "start": str(run.index[0].date()), "end": str(run.index[-1].date()),
        "cagr": m.cagr, "ann_vol": m.annualized_volatility, "sharpe": m.sharpe_ratio,
        "sharpe_vs_cash": float("nan"), "sortino": m.sortino_ratio,
        "max_drawdown": m.maximum_drawdown, "calmar": m.calmar_ratio,
        "ann_turnover": m.annualized_turnover,
        "ann_cost": float(run.costs.sum()) * annualization_factor / n,
        "cost_drag": m.total_transaction_cost_drag,
    }
    if cash_returns is not None:
        cash = cash_returns.reindex(run.index)
        if cash.isna().any():
            raise MisalignedSamplesError(f"{run.strategy_id}: cash proxy missing on evaluation dates.")
        excess = run.net - cash
        sd = float(excess.std(ddof=1))
        row["sharpe_vs_cash"] = float(excess.mean()) / sd * np.sqrt(annualization_factor) if sd > 1e-12 else float("nan")
    return row


def metrics_table(runs: Sequence[StrategyRun], cash_returns: Optional[pd.Series] = None) -> pd.DataFrame:
    require_common_index(runs)
    return pd.DataFrame([metric_row(run, cash_returns) for run in runs])


def window_table(runs: Sequence[StrategyRun], windows: Sequence[Tuple[str, str, str]],
                 cash_returns: Optional[pd.Series] = None, min_obs: int = 20) -> pd.DataFrame:
    """Chronological attribution: every strategy sliced to the same calendar windows.

    Benchmarks keep their genuine gross returns, turnover and costs inside each window.
    """
    require_common_index(runs)
    rows: List[Dict] = []
    for label, start, end in windows:
        sliced = [run.window(start, end) for run in runs]
        if len(sliced[0].index) < min_obs:
            continue
        for run in sliced:
            rows.append({"window": label, "evaluation_type": "chronological_attribution",
                         **metric_row(run, cash_returns)})
    return pd.DataFrame(rows)
