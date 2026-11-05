"""Walk-forward model selection with explicit train / validation / evaluation boundaries.

Each candidate strategy is an online, causal backtest (its models are re-fitted
on trailing *training* windows inside the engine).  At the last close before
each evaluation year, a selection rule fixed in advance ranks candidates on
their *validation* window (the trailing ``lookback_years`` of net returns that
ended at that close) and the winner is held for the *evaluation* year.

Switching follows the engine's timing: the choice made at close ``t`` is
filled at close ``t+1`` (the first session of the year).  That day earns the
outgoing strategy's gross return, then the portfolio trades from the outgoing
holdings to the incoming strategy's post-trade weights and pays the cost.
From the next day it tracks the incoming strategy exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from regime_shift.execution import ExecutionCostModel
from regime_shift.research.evaluation import StrategyRun, require_common_index
from regime_shift.research.spec import ANNUALIZATION_FACTOR


@dataclass
class WalkForwardResult:
    run: StrategyRun
    selections: pd.DataFrame  # one row per evaluation year


def _sharpe(r: pd.Series) -> float:
    sd = float(r.std(ddof=1))
    return float(r.mean()) / sd * np.sqrt(ANNUALIZATION_FACTOR) if sd > 1e-12 else float("nan")


def walk_forward_select(
    runs: Dict[str, StrategyRun],
    asset_returns: pd.DataFrame,
    cost_model: ExecutionCostModel,
    lookback_years: int = 3,
    strategy_id: str = "walk_forward_selected",
    cash_returns: Optional[pd.Series] = None,
) -> WalkForwardResult:
    """Select the trailing-Sharpe leader each calendar year and account for switches.

    With ``cash_returns`` the score is the Sharpe of ``net - cash`` (excess over the
    cash proxy); otherwise it is the rf = 0 Sharpe of net returns.
    """
    index = require_common_index(runs.values())
    if not asset_returns.index.equals(index):
        asset_returns = asset_returns.loc[index]
    assets = list(asset_returns.columns)
    for run in runs.values():
        if run.weights is None or list(run.weights.columns) != assets:
            raise ValueError(f"{run.strategy_id}: post-trade weights for {assets} are required.")

    years = sorted(set(index.year))
    first_full_year = index[0].year + (0 if (index[0].month, index[0].day) == (1, 1) else 1)
    eval_years = [y for y in years if y - lookback_years >= first_full_year]
    if not eval_years:
        raise ValueError("Sample too short for the requested validation lookback.")

    columns = {k: [] for k in ("gross", "net", "costs", "turnover")}
    out_index: List[pd.Timestamp] = []
    weights_rows: List[np.ndarray] = []
    selections = []
    held: str = ""
    for year in eval_years:
        days = index[index.year == year]
        decision_close = index[index < days[0]][-1]
        validation = index[(index > decision_close - pd.DateOffset(years=lookback_years)) & (index <= decision_close)]

        def score(run: StrategyRun) -> float:
            r = run.net.loc[validation]
            return _sharpe(r - cash_returns.loc[validation] if cash_returns is not None else r)

        scores = {sid: score(run) for sid, run in runs.items()}
        chosen = max(scores, key=lambda s: -np.inf if not np.isfinite(scores[s]) else scores[s])
        selections.append({
            "evaluation_year": year, "decision_close": decision_close.date(), "fill_close": days[0].date(),
            "validation_start": validation[0].date(), "validation_end": validation[-1].date(),
            "selected": chosen, **{f"val_sharpe_{sid}": s for sid, s in scores.items()},
        })
        new = runs[chosen]
        for pos, day in enumerate(days):
            if pos == 0 and held and held != chosen:
                old = runs[held]
                prev_day = index[index.get_loc(day) - 1]
                pre = old.weights.loc[prev_day].to_numpy(dtype=float)
                r = asset_returns.loc[day].to_numpy(dtype=float)
                gross = float(pre @ r)
                drifted = pre * (1.0 + r) / (1.0 + gross)
                target = new.weights.loc[day].to_numpy(dtype=float)
                cost = cost_model.cost_for_trade(target, drifted, assets)
                turnover = 0.5 * float(np.abs(target - drifted).sum())
            elif pos == 0 and not held:
                # Start with the incoming strategy's own already-held book (no extra switch).
                gross, cost, turnover = float(new.gross[day]), float(new.costs[day]), float(new.turnover[day])
            else:
                gross, cost, turnover = float(new.gross[day]), float(new.costs[day]), float(new.turnover[day])
            columns["gross"].append(gross)
            columns["costs"].append(cost)
            columns["net"].append((1.0 + gross) * (1.0 - cost) - 1.0)
            columns["turnover"].append(turnover)
            weights_rows.append(new.weights.loc[day].to_numpy(dtype=float))
            out_index.append(day)
        held = chosen

    idx = pd.DatetimeIndex(out_index)
    run = StrategyRun(
        strategy_id,
        pd.Series(columns["gross"], index=idx, name="gross_return"),
        pd.Series(columns["net"], index=idx, name="net_return"),
        pd.Series(columns["costs"], index=idx, name="transaction_cost"),
        pd.Series(columns["turnover"], index=idx, name="turnover"),
        weights=pd.DataFrame(weights_rows, index=idx, columns=assets),
        info={"rule": f"max trailing {lookback_years}y net Sharpe (rf=0), chosen at last close of prior year"},
    )
    return WalkForwardResult(run, pd.DataFrame(selections))
