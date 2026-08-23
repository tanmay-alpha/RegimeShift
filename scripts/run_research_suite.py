"""Research validation suite for RegimeShift.

Executes five research analyses required for quant-resume rigour:
  1. Controlled ablation study (7 strategies)
  2. Chronological / subperiod analysis
  3. Rolling-origin evaluation
  4. Paired moving-block bootstrap (2,000 samples, block=21, seed=42)
  5. HMM stability diagnostics

All variants use:
  - Identical NEXT_CLOSE causal execution
  - Identical cost model (base = 10 bps one-way)
  - Identical dates (strategy evaluation window)
  - Identical assets and annualisation factor
  - Frozen RegimeShiftConfig (no parameter changes)

Usage:
    python scripts/run_research_suite.py \\
        --data-path data/submission_market_data.csv \\
        --output-dir results/research
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from regime_shift.config import RegimeShiftConfig
from regime_shift.data import load_market_data_csv
from regime_shift.backtest import run_walk_forward_backtest, run_benchmark, _drift, _cost_model
from regime_shift.benchmarks import static_60_40_weights, equal_weight_weights
from regime_shift.metrics import compute_performance_metrics
from regime_shift.execution import ExecutionCostModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_ASSETS = ["equity", "gold", "bond"]
_RF = 0.0
_ANN = 252

# HMM-fixed allocations for ablation variant D
_HMM_FIXED = {
    "Bull":   {"equity": 0.65, "gold": 0.20, "bond": 0.15},
    "Bear":   {"equity": 0.35, "gold": 0.30, "bond": 0.35},
    "Crisis": {"equity": 0.10, "gold": 0.30, "bond": 0.60},
}

BLOCK_LEN = 21
N_BOOT = 2_000
BOOT_SEED = 42

SUBPERIODS = [
    ("Development", "2010-10-05", "2018-12-31",
     "Retrospective research. Full-sample data was inspected before this protocol. Not a pristine holdout."),
    ("Validation", "2019-01-01", "2021-12-31",
     "Retrospective research. Full-sample data was inspected before this protocol. Not a pristine holdout."),
    ("Retrospective Evaluation", "2022-01-01", "2026-07-27",
     "Retrospective evaluation. Not a pristine holdout because full historical sample was inspected during earlier project development."),
]

FOLDS = [
    ("2013-2015", "2013-01-01", "2015-12-31"),
    ("2016-2018", "2016-01-01", "2018-12-31"),
    ("2019-2021", "2019-01-01", "2021-12-31"),
    ("2022-2024", "2022-01-01", "2024-12-31"),
    ("2025-2026", "2025-01-01", "2026-07-27"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics_row(label: str, net: pd.Series, gross: pd.Series,
                 turnover: pd.Series, costs: pd.Series) -> dict:
    m = compute_performance_metrics(
        net, gross_returns=gross, turnover=turnover, transaction_costs=costs,
        risk_free_rate=_RF, annualization_factor=_ANN,
    )
    return {
        "strategy": label,
        "cagr": m.cagr,
        "ann_vol": m.annualized_volatility,
        "sharpe": m.sharpe_ratio,
        "sortino": m.sortino_ratio,
        "max_drawdown": m.maximum_drawdown,
        "calmar": m.calmar_ratio,
        "ann_turnover": m.annualized_turnover,
        "cost_drag": m.total_transaction_cost_drag,
        "n_obs": m.n_observations,
        "start": str(net.index[0].date()) if len(net) else "",
        "end": str(net.index[-1].date()) if len(net) else "",
    }


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Ablation variant runners
# ---------------------------------------------------------------------------

def _run_no_regime_optimizer(prices: pd.DataFrame, config: RegimeShiftConfig) -> tuple:
    """Always optimise as if Bull regime — no HMM switching."""
    from regime_shift.features import (compute_raw_features, drop_feature_warmup,
                                        fit_feature_scaler, transform_features)
    from regime_shift.portfolio import optimize_portfolio

    cfg = copy.deepcopy(config)
    assets = cfg.core_assets
    price_data = prices[assets].copy()
    asset_returns = price_data.pct_change().iloc[1:]
    raw_features = drop_feature_warmup(compute_raw_features(price_data, cfg.feature_config),
                                        config=cfg.feature_config)
    common = raw_features.index.intersection(asset_returns.index)
    raw_features, asset_returns = raw_features.loc[common], asset_returns.loc[common]
    dates = pd.DatetimeIndex(asset_returns.index)
    minimum = cfg.minimum_training_observations + cfg.rebalance_frequency
    if len(dates) < minimum:
        raise ValueError("Insufficient data for no-regime optimizer ablation.")
    rebalance_positions = set(range(cfg.rebalance_frequency, len(dates), cfg.rebalance_frequency))
    cost_mdl = _cost_model(cfg, None)

    gross_l, net_l, costs_l, turns_l = [], [], [], []
    current = np.zeros(len(assets))
    allocated = False
    first_allocation = None

    for pos, date in enumerate(dates):
        pre_trade = current.copy()
        day_return = asset_returns.iloc[pos].to_numpy(dtype=float)
        gross_return, post_return = _drift(pre_trade, day_return, date)
        current = post_return.copy()
        rebalance = pos in rebalance_positions
        cost = turnover = 0.0

        if rebalance:
            train = raw_features.loc[:date].tail(cfg.train_window)
            returns_for_estimation = asset_returns.loc[:date].tail(
                cfg.portfolio_config.estimation_lookback)
            if (len(train) >= cfg.minimum_training_observations and
                    len(returns_for_estimation) >= cfg.portfolio_config.minimum_estimation_observations):
                previous = dict(zip(assets, current)) if allocated else None
                portfolio = optimize_portfolio("Bull", returns_for_estimation, previous,
                                               config=cfg.portfolio_config)
                target = np.array([portfolio.weights[a] for a in assets], dtype=float)
                turnover = (float(np.abs(target).sum()) if not allocated
                            else 0.5 * float(np.abs(target - current).sum()))
                cost = cost_mdl.cost_for_trade(target, current, assets, initial=not allocated)
                current = target.copy()
                allocated = True
                if first_allocation is None:
                    first_allocation = pos
            else:
                rebalance = False

        net_return = (1.0 + gross_return) * (1.0 - cost) - 1.0
        gross_l.append(gross_return); net_l.append(net_return)
        costs_l.append(cost); turns_l.append(turnover)

    if first_allocation is None:
        raise ValueError("No-regime optimizer: no allocation occurred.")
    start = first_allocation
    index = dates[start:]
    return (pd.Series(gross_l[start:], index=index),
            pd.Series(net_l[start:], index=index),
            pd.Series(costs_l[start:], index=index),
            pd.Series(turns_l[start:], index=index))


def _run_hmm_fixed(prices: pd.DataFrame, config: RegimeShiftConfig) -> tuple:
    """HMM regime detection but fixed per-regime weights — no CVXPY."""
    from regime_shift.features import (compute_raw_features, drop_feature_warmup,
                                        fit_feature_scaler, transform_features)
    from regime_shift.regime_model import fit_hmm, predict_current_state

    cfg = copy.deepcopy(config)
    assets = cfg.core_assets
    price_data = prices[assets].copy()
    asset_returns = price_data.pct_change().iloc[1:]
    raw_features = drop_feature_warmup(compute_raw_features(price_data, cfg.feature_config),
                                        config=cfg.feature_config)
    common = raw_features.index.intersection(asset_returns.index)
    raw_features, asset_returns = raw_features.loc[common], asset_returns.loc[common]
    dates = pd.DatetimeIndex(asset_returns.index)
    minimum = cfg.minimum_training_observations + cfg.rebalance_frequency
    if len(dates) < minimum:
        raise ValueError("Insufficient data for HMM-fixed ablation.")
    rebalance_positions = set(range(cfg.rebalance_frequency, len(dates), cfg.rebalance_frequency))
    cost_mdl = _cost_model(cfg, None)

    gross_l, net_l, costs_l, turns_l = [], [], [], []
    current = np.zeros(len(assets))
    allocated = False
    first_allocation = None

    for pos, date in enumerate(dates):
        pre_trade = current.copy()
        day_return = asset_returns.iloc[pos].to_numpy(dtype=float)
        gross_return, post_return = _drift(pre_trade, day_return, date)
        current = post_return.copy()
        rebalance = pos in rebalance_positions
        cost = turnover = 0.0

        if rebalance:
            train = raw_features.loc[:date].tail(cfg.train_window)
            if len(train) >= cfg.minimum_training_observations:
                scaler = fit_feature_scaler(train)
                scaled = transform_features(scaler, train)
                hmm, _, _ = fit_hmm(scaled, train, config=cfg.hmm_config)
                solution = predict_current_state(hmm, scaler, train, train, config=cfg.hmm_config)
                regime = solution.regime
                target_dict = _HMM_FIXED[regime]
                target = np.array([target_dict[a] for a in assets], dtype=float)
                turnover = (float(np.abs(target).sum()) if not allocated
                            else 0.5 * float(np.abs(target - current).sum()))
                cost = cost_mdl.cost_for_trade(target, current, assets, initial=not allocated)
                current = target.copy()
                allocated = True
                if first_allocation is None:
                    first_allocation = pos
            else:
                rebalance = False

        net_return = (1.0 + gross_return) * (1.0 - cost) - 1.0
        gross_l.append(gross_return); net_l.append(net_return)
        costs_l.append(cost); turns_l.append(turnover)

    if first_allocation is None:
        raise ValueError("HMM-fixed: no allocation occurred.")
    start = first_allocation
    index = dates[start:]
    return (pd.Series(gross_l[start:], index=index),
            pd.Series(net_l[start:], index=index),
            pd.Series(costs_l[start:], index=index),
            pd.Series(turns_l[start:], index=index))


def _run_min_variance(prices: pd.DataFrame, config: RegimeShiftConfig) -> tuple:
    """Covariance-only minimum-variance portfolio. No expected returns."""
    import cvxpy as cp
    from regime_shift.features import compute_raw_features, drop_feature_warmup

    cfg = copy.deepcopy(config)
    assets = cfg.core_assets
    price_data = prices[assets].copy()
    asset_returns = price_data.pct_change().iloc[1:]
    raw_features = drop_feature_warmup(compute_raw_features(price_data, cfg.feature_config),
                                        config=cfg.feature_config)
    common = raw_features.index.intersection(asset_returns.index)
    raw_features, asset_returns = raw_features.loc[common], asset_returns.loc[common]
    dates = pd.DatetimeIndex(asset_returns.index)
    minimum = cfg.minimum_training_observations + cfg.rebalance_frequency
    if len(dates) < minimum:
        raise ValueError("Insufficient data for min-variance ablation.")
    rebalance_positions = set(range(cfg.rebalance_frequency, len(dates), cfg.rebalance_frequency))
    cost_mdl = _cost_model(cfg, None)
    LOOKBACK = cfg.portfolio_config.estimation_lookback
    MIN_OBS = cfg.portfolio_config.minimum_estimation_observations
    RIDGE = cfg.portfolio_config.covariance_ridge

    gross_l, net_l, costs_l, turns_l = [], [], [], []
    current = np.zeros(len(assets))
    allocated = False
    first_allocation = None

    for pos, date in enumerate(dates):
        pre_trade = current.copy()
        day_return = asset_returns.iloc[pos].to_numpy(dtype=float)
        gross_return, post_return = _drift(pre_trade, day_return, date)
        current = post_return.copy()
        rebalance = pos in rebalance_positions
        cost = turnover = 0.0

        if rebalance:
            returns_for_estimation = asset_returns.loc[:date].tail(LOOKBACK)
            if len(returns_for_estimation) >= MIN_OBS:
                cov = returns_for_estimation.cov().values
                cov = (cov + cov.T) / 2.0
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.where(eigvals < 0, 0.0, eigvals) + RIDGE
                cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
                cov = (cov + cov.T) / 2.0
                w = cp.Variable(len(assets))
                prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))),
                                  [cp.sum(w) == 1.0, w >= 0.0, w <= 0.80])
                solved = False
                for solver in [cp.CLARABEL, cp.OSQP, cp.SCS]:
                    try:
                        prob.solve(solver=solver, verbose=False)
                        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                            solved = True
                            break
                    except Exception:
                        continue
                if solved and w.value is not None:
                    target = np.maximum(w.value, 0.0)
                    ws = target.sum()
                    if ws > 0:
                        target = target / ws
                    turnover = (float(np.abs(target).sum()) if not allocated
                                else 0.5 * float(np.abs(target - current).sum()))
                    cost = cost_mdl.cost_for_trade(target, current, assets, initial=not allocated)
                    current = target.copy()
                    allocated = True
                    if first_allocation is None:
                        first_allocation = pos
                else:
                    rebalance = False
            else:
                rebalance = False

        net_return = (1.0 + gross_return) * (1.0 - cost) - 1.0
        gross_l.append(gross_return); net_l.append(net_return)
        costs_l.append(cost); turns_l.append(turnover)

    if first_allocation is None:
        raise ValueError("Min-variance: no allocation occurred.")
    start = first_allocation
    index = dates[start:]
    return (pd.Series(gross_l[start:], index=index),
            pd.Series(net_l[start:], index=index),
            pd.Series(costs_l[start:], index=index),
            pd.Series(turns_l[start:], index=index))


def _run_volatility_rule(prices: pd.DataFrame, config: RegimeShiftConfig,
                          start_date: pd.Timestamp) -> tuple:
    """Simple rule: trailing 63-day realized vol threshold. No HMM."""
    assets = config.core_assets
    returns = prices[assets].pct_change().iloc[1:]
    returns = returns.loc[returns.index >= start_date]
    dates = pd.DatetimeIndex(returns.index)
    eq_returns = prices["equity"].pct_change().iloc[1:]
    cost_mdl = ExecutionCostModel.scenario("base", assets)

    gross_l, net_l, costs_l, turns_l = [], [], [], []
    current = np.zeros(len(assets))
    allocated = False
    WINDOW = 63
    HIGH_VOL = 0.20
    CRISIS_VOL = 0.30

    for pos, date in enumerate(dates):
        pre = current.copy()
        day_r = returns.iloc[pos].to_numpy(dtype=float)
        gross = float(np.dot(pre, day_r))
        if np.any(pre):
            wealth = 1.0 + gross
            current = pre * (1.0 + day_r) / max(wealth, 1e-10)
        else:
            current = pre.copy()

        rebalance = pos % config.rebalance_frequency == 0 and pos > 0
        cost = turn = 0.0

        if rebalance:
            hist = eq_returns.loc[:date].tail(WINDOW)
            if len(hist) >= WINDOW // 2:
                ann_vol = float(hist.std(ddof=1)) * np.sqrt(_ANN)
                if ann_vol > CRISIS_VOL:
                    target = np.array([0.10, 0.30, 0.60])
                elif ann_vol > HIGH_VOL:
                    target = np.array([0.35, 0.30, 0.35])
                else:
                    target = np.array([0.65, 0.20, 0.15])
                turn = (float(np.abs(target).sum()) if not allocated
                        else 0.5 * float(np.abs(target - current).sum()))
                cost = cost_mdl.cost_for_trade(target, current, assets, initial=not allocated)
                current = target.copy()
                allocated = True

        net_r = (1.0 + gross) * (1.0 - cost) - 1.0
        gross_l.append(gross); net_l.append(net_r)
        costs_l.append(cost); turns_l.append(turn)

    index = dates
    return (pd.Series(gross_l, index=index),
            pd.Series(net_l, index=index),
            pd.Series(costs_l, index=index),
            pd.Series(turns_l, index=index))


# ===========================================================================
# Section 1: Ablation Study
# ===========================================================================

def run_ablation(prices: pd.DataFrame, config: RegimeShiftConfig,
                 output_dir: Path, rs_result=None) -> pd.DataFrame:
    logger.info("=== Ablation Study ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    if rs_result is None:
        logger.info("Running A: Full RegimeShift (HMM + CVXPY)...")
        rs_result = run_walk_forward_backtest(prices=prices, config=config, risk_free_rate=_RF)

    rs = rs_result
    eval_start = rs.net_returns.index[0]
    eval_end = rs.net_returns.index[-1]
    logger.info("Evaluation window: %s to %s (%d obs)",
                eval_start.date(), eval_end.date(), len(rs.net_returns))

    rows = [_metrics_row("A_RegimeShift_Full", rs.net_returns, rs.gross_returns,
                          rs.turnover, rs.transaction_costs)]
    all_net = {"A_RegimeShift_Full": rs.net_returns}

    # B. No-regime optimizer (always Bull)
    logger.info("Running B: No-regime optimizer (always Bull)...")
    try:
        g, n, c, t = _run_no_regime_optimizer(prices, config)
        n_aligned = n.reindex(rs.net_returns.index).dropna()
        g_a = g.reindex(n_aligned.index).fillna(0)
        c_a = c.reindex(n_aligned.index).fillna(0)
        t_a = t.reindex(n_aligned.index).fillna(0)
        rows.append(_metrics_row("B_NoRegime_Optimizer", n_aligned, g_a, t_a, c_a))
        all_net["B_NoRegime_Optimizer"] = n_aligned
    except Exception as exc:
        logger.warning("B failed: %s", exc)

    # C. Volatility-rule allocation
    logger.info("Running C: Volatility-rule allocation...")
    try:
        g, n, c, t = _run_volatility_rule(prices, config, eval_start)
        n_aligned = n.reindex(rs.net_returns.index).dropna()
        g_a = g.reindex(n_aligned.index).fillna(0)
        c_a = c.reindex(n_aligned.index).fillna(0)
        t_a = t.reindex(n_aligned.index).fillna(0)
        rows.append(_metrics_row("C_VolRule", n_aligned, g_a, t_a, c_a))
        all_net["C_VolRule"] = n_aligned
    except Exception as exc:
        logger.warning("C failed: %s", exc)

    # D. HMM + fixed allocations
    logger.info("Running D: HMM + fixed allocations (no optimizer)...")
    try:
        g, n, c, t = _run_hmm_fixed(prices, config)
        n_aligned = n.reindex(rs.net_returns.index).dropna()
        g_a = g.reindex(n_aligned.index).fillna(0)
        c_a = c.reindex(n_aligned.index).fillna(0)
        t_a = t.reindex(n_aligned.index).fillna(0)
        rows.append(_metrics_row("D_HMM_Fixed", n_aligned, g_a, t_a, c_a))
        all_net["D_HMM_Fixed"] = n_aligned
    except Exception as exc:
        logger.warning("D failed: %s", exc)

    # E. Minimum variance
    logger.info("Running E: Minimum variance (covariance-only)...")
    try:
        g, n, c, t = _run_min_variance(prices, config)
        n_aligned = n.reindex(rs.net_returns.index).dropna()
        g_a = g.reindex(n_aligned.index).fillna(0)
        c_a = c.reindex(n_aligned.index).fillna(0)
        t_a = t.reindex(n_aligned.index).fillna(0)
        rows.append(_metrics_row("E_MinVariance", n_aligned, g_a, t_a, c_a))
        all_net["E_MinVariance"] = n_aligned
    except Exception as exc:
        logger.warning("E failed: %s", exc)

    # F. Static 60/40
    logger.info("Running F: Static 60/40...")
    b6040 = run_benchmark(prices=prices, weights=static_60_40_weights(), config=config,
                          rebalance_flags=rs.rebalance_flags, start_date=eval_start)
    rows.append(_metrics_row("F_Static_6040", b6040.net_returns, b6040.gross_returns,
                              b6040.turnover, b6040.transaction_costs))
    all_net["F_Static_6040"] = b6040.net_returns

    # G. Equal Weight
    logger.info("Running G: Equal Weight...")
    bew = run_benchmark(prices=prices, weights=equal_weight_weights(), config=config,
                        rebalance_flags=rs.rebalance_flags, start_date=eval_start)
    rows.append(_metrics_row("G_EqualWeight", bew.net_returns, bew.gross_returns,
                              bew.turnover, bew.transaction_costs))
    all_net["G_EqualWeight"] = bew.net_returns

    # Save CSV
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    logger.info("Saved ablation_summary.csv")

    daily = pd.DataFrame(all_net)
    daily.to_csv(output_dir / "ablation_daily_returns.csv")
    logger.info("Saved ablation_daily_returns.csv")

    # Equity curves + drawdown chart
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    ax_eq, ax_dd = axes
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for i, (label, net) in enumerate(all_net.items()):
        eq = (1 + net).cumprod()
        dd = eq / eq.cummax() - 1
        eq.plot(ax=ax_eq, label=label, color=palette[i % len(palette)], linewidth=1.5)
        dd.plot(ax=ax_dd, label=label, color=palette[i % len(palette)], linewidth=1.5)
    ax_eq.set_title("Ablation Study — Cumulative Net Returns (base 10 bps)", fontsize=12, fontweight="bold")
    ax_eq.set_ylabel("Cumulative Return"); ax_eq.legend(loc="upper left", fontsize=7)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax_dd.set_title("Ablation Study — Drawdowns"); ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_dd.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    _save_fig(fig, output_dir / "ablation_equity_curves.png")

    # Sharpe / Turnover scatter
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    for _, row in summary.iterrows():
        ax2.scatter(row["ann_turnover"], row["sharpe"], s=80, zorder=5)
        ax2.annotate(row["strategy"].split("_", 1)[0],
                     (row["ann_turnover"], row["sharpe"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Annualised Turnover"); ax2.set_ylabel("Net Sharpe Ratio")
    ax2.set_title("Ablation — Sharpe vs Turnover")
    fig2.tight_layout()
    _save_fig(fig2, output_dir / "ablation_sharpe_turnover.png")

    meta = {
        "eval_start": str(eval_start.date()), "eval_end": str(eval_end.date()),
        "n_strategies": len(rows), "cost_scenario": "base",
        "cost_bps": 10, "execution_model": "NEXT_CLOSE",
    }
    (output_dir / "ablation_metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("Ablation complete: %d strategies", len(rows))
    return summary


# ===========================================================================
# Section 2: Subperiod / Chronological Analysis
# ===========================================================================

def run_subperiod(prices: pd.DataFrame, config: RegimeShiftConfig, output_dir: Path,
                  full_rs_net: pd.Series, full_rs_gross: pd.Series,
                  full_rs_turnover: pd.Series, full_rs_costs: pd.Series,
                  full_b6040_net: pd.Series, full_bew_net: pd.Series) -> pd.DataFrame:
    logger.info("=== Subperiod Analysis ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    zeros = pd.Series(0.0, index=full_b6040_net.index)
    for name, start, end, note in SUBPERIODS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        for label, net, gross, turn, costs in [
            ("RegimeShift", full_rs_net, full_rs_gross, full_rs_turnover, full_rs_costs),
            ("Static 60/40", full_b6040_net, full_b6040_net, zeros, zeros),
            ("Equal Weight", full_bew_net, full_bew_net, zeros, zeros),
        ]:
            sub = net[(net.index >= s) & (net.index <= e)]
            if len(sub) < 20:
                continue
            sub_g = gross.reindex(sub.index).fillna(0)
            sub_t = turn.reindex(sub.index).fillna(0)
            sub_c = costs.reindex(sub.index).fillna(0)
            m = compute_performance_metrics(sub, gross_returns=sub_g, turnover=sub_t,
                                            transaction_costs=sub_c,
                                            risk_free_rate=_RF, annualization_factor=_ANN)
            rows.append({
                "period": name, "strategy": label, "period_note": note,
                "n_obs": len(sub), "start": str(sub.index[0].date()),
                "end": str(sub.index[-1].date()),
                "cagr": m.cagr, "ann_vol": m.annualized_volatility,
                "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
                "max_drawdown": m.maximum_drawdown, "calmar": m.calmar_ratio,
                "ann_turnover": m.annualized_turnover, "cost_drag": m.total_transaction_cost_drag,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "subperiod_summary.csv", index=False)
    logger.info("Saved subperiod_summary.csv (%d rows)", len(df))

    periods = [p for p, _, _, _ in SUBPERIODS]
    strategies = df["strategy"].unique()
    colors = {"RegimeShift": "#1f77b4", "Static 60/40": "#ff7f0e", "Equal Weight": "#2ca02c"}

    # Sharpe chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(periods))
    width = 0.25
    for i, strat in enumerate(strategies):
        sharpes = []
        for p in periods:
            sub_df = df[(df["period"] == p) & (df["strategy"] == strat)]
            sharpes.append(sub_df["sharpe"].values[0] if len(sub_df) > 0 else np.nan)
        ax.bar(x + i * width, sharpes, width, label=strat,
               color=colors.get(strat, f"C{i}"), alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width); ax.set_xticklabels(periods, rotation=10)
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Subperiod Net Sharpe Ratio\n(all periods retrospective — no pristine holdout)")
    ax.legend(); fig.tight_layout()
    _save_fig(fig, output_dir / "subperiod_sharpe.png")

    # Drawdown per-period chart
    fig2, axes2 = plt.subplots(1, len(periods), figsize=(14, 4), sharey=True)
    for i, (pname, start, end, _) in enumerate(SUBPERIODS):
        ax2 = axes2[i]
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        for label, net in [("RegimeShift", full_rs_net),
                           ("Static 60/40", full_b6040_net),
                           ("Equal Weight", full_bew_net)]:
            sub = net[(net.index >= s) & (net.index <= e)]
            if len(sub) < 5:
                continue
            eq = (1 + sub).cumprod()
            (eq / eq.cummax() - 1).plot(ax=ax2, label=label, linewidth=1.2,
                                          color=colors.get(label))
        ax2.set_title(pname, fontsize=9)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    axes2[0].set_ylabel("Drawdown")
    axes2[-1].legend(loc="lower right", fontsize=7)
    fig2.suptitle("Drawdowns by Subperiod", fontsize=11, fontweight="bold")
    fig2.tight_layout()
    _save_fig(fig2, output_dir / "subperiod_drawdowns.png")

    logger.info("Subperiod analysis complete.")
    return df


# ===========================================================================
# Section 3: Rolling-Origin Analysis
# ===========================================================================

def run_rolling_origin(prices: pd.DataFrame, config: RegimeShiftConfig, output_dir: Path,
                       full_rs_net: pd.Series, full_rs_gross: pd.Series,
                       full_rs_turnover: pd.Series, full_rs_costs: pd.Series,
                       full_b6040_net: pd.Series, full_bew_net: pd.Series) -> pd.DataFrame:
    logger.info("=== Rolling-Origin Analysis ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    zeros_rs = pd.Series(0.0, index=full_rs_net.index)
    zeros_b = pd.Series(0.0, index=full_b6040_net.index)

    for fold_name, fold_start, fold_end in FOLDS:
        fs, fe = pd.Timestamp(fold_start), pd.Timestamp(fold_end)
        for label, net, gross, turn, costs in [
            ("RegimeShift", full_rs_net, full_rs_gross, full_rs_turnover, full_rs_costs),
            ("Static 60/40", full_b6040_net, full_b6040_net, zeros_b, zeros_b),
            ("Equal Weight", full_bew_net, full_bew_net, zeros_b, zeros_b),
        ]:
            sub = net[(net.index >= fs) & (net.index <= fe)]
            if len(sub) < 20:
                continue
            sub_g = gross.reindex(sub.index).fillna(0)
            sub_t = turn.reindex(sub.index).fillna(0)
            sub_c = costs.reindex(sub.index).fillna(0)
            m = compute_performance_metrics(sub, gross_returns=sub_g, turnover=sub_t,
                                            transaction_costs=sub_c,
                                            risk_free_rate=_RF, annualization_factor=_ANN)
            rows.append({
                "fold": fold_name, "strategy": label,
                "n_obs": len(sub), "start": str(sub.index[0].date()),
                "end": str(sub.index[-1].date()),
                "cagr": m.cagr, "ann_vol": m.annualized_volatility,
                "sharpe": m.sharpe_ratio, "sortino": m.sortino_ratio,
                "max_drawdown": m.maximum_drawdown, "calmar": m.calmar_ratio,
                "ann_turnover": m.annualized_turnover, "cost_drag": m.total_transaction_cost_drag,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "rolling_origin_summary.csv", index=False)

    rs_rows = df[df["strategy"] == "RegimeShift"]
    b6040_rows = df[df["strategy"] == "Static 60/40"]
    bew_rows = df[df["strategy"] == "Equal Weight"]
    pos_sharpe = int((rs_rows["sharpe"] > 0).sum())
    beat_6040 = int((rs_rows["sharpe"].values > b6040_rows["sharpe"].values[:len(rs_rows)]).sum()) if len(b6040_rows) >= len(rs_rows) else 0
    beat_ew = int((rs_rows["sharpe"].values > bew_rows["sharpe"].values[:len(rs_rows)]).sum()) if len(bew_rows) >= len(rs_rows) else 0
    median_sharpe = float(rs_rows["sharpe"].median())
    worst_dd = float(rs_rows["max_drawdown"].max())
    logger.info("Rolling-origin: %d folds | pos_Sharpe=%d | beat_6040=%d | beat_EW=%d | median=%.3f | worst_DD=%.3f",
                len(rs_rows), pos_sharpe, beat_6040, beat_ew, median_sharpe, worst_dd)

    folds = [f for f, _, _ in FOLDS]
    strategies = df["strategy"].unique()
    colors = {"RegimeShift": "#1f77b4", "Static 60/40": "#ff7f0e", "Equal Weight": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(folds))
    width = 0.25
    for i, strat in enumerate(strategies):
        sharpes = []
        for f in folds:
            sub_df = df[(df["fold"] == f) & (df["strategy"] == strat)]
            sharpes.append(sub_df["sharpe"].values[0] if len(sub_df) > 0 else np.nan)
        ax.bar(x + i * width, sharpes, width, label=strat, color=colors.get(strat, f"C{i}"), alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width); ax.set_xticklabels(folds, rotation=15, fontsize=8)
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title(f"Rolling-Origin Net Sharpe by Fold\n"
                 f"RegimeShift: {pos_sharpe}/{len(rs_rows)} positive, "
                 f"{beat_6040}/{len(rs_rows)} beat 60/40, median={median_sharpe:.3f}")
    ax.legend(); fig.tight_layout()
    _save_fig(fig, output_dir / "rolling_origin_sharpe.png")

    fig2, ax2 = plt.subplots(figsize=(13, 5))
    for label, net in [("RegimeShift", full_rs_net),
                       ("Static 60/40", full_b6040_net),
                       ("Equal Weight", full_bew_net)]:
        eq = (1 + net).cumprod()
        (eq / eq.cummax() - 1).plot(ax=ax2, label=label, linewidth=1.5, color=colors.get(label))
    for _, fs, fe in FOLDS:
        ax2.axvspan(pd.Timestamp(fs), pd.Timestamp(fe), alpha=0.06, color="gray")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.set_title("Rolling-Origin Drawdowns (shaded = fold evaluation windows)")
    ax2.legend(); fig2.tight_layout()
    _save_fig(fig2, output_dir / "rolling_origin_drawdowns.png")

    logger.info("Rolling-origin complete.")
    return df


# ===========================================================================
# Section 4: Paired Moving-Block Bootstrap
# ===========================================================================

def _bootstrap_indices(n: int, block_len: int, n_boot: int,
                        rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_len))
    max_start = max(1, n - block_len + 1)
    starts = rng.integers(0, max_start, size=(n_boot, n_blocks))
    indices = np.zeros((n_boot, n_blocks * block_len), dtype=int)
    for b in range(n_blocks):
        for j in range(block_len):
            indices[:, b * block_len + j] = np.minimum(starts[:, b] + j, n - 1)
    return indices[:, :n]


def _boot_metrics_array(returns_array: np.ndarray, ann: int = 252) -> list:
    results = []
    for r in returns_array:
        r = r.astype(float)
        n = len(r)
        eq = np.cumprod(1 + r)
        cagr = eq[-1] ** (ann / n) - 1
        vol = float(np.std(r, ddof=1)) * np.sqrt(ann)
        excess = r - ((1 + _RF) ** (1 / ann) - 1)
        std_exc = float(np.std(excess, ddof=1))
        sharpe = float(excess.mean() / std_exc) * np.sqrt(ann) if std_exc > 1e-12 else 0.0
        cummax = np.maximum.accumulate(eq)
        max_dd = float(np.max(1 - eq / cummax))
        calmar = cagr / max_dd if max_dd > 1e-12 else np.nan
        results.append({"cagr": cagr, "vol": vol, "sharpe": sharpe,
                         "max_drawdown": max_dd, "calmar": calmar})
    return results


def run_bootstrap(full_rs_net: pd.Series, full_b6040_net: pd.Series,
                  full_bew_net: pd.Series, output_dir: Path) -> pd.DataFrame:
    logger.info("=== Paired Moving-Block Bootstrap (n=%d, block=%d, seed=%d) ===",
                N_BOOT, BLOCK_LEN, BOOT_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    idx = full_rs_net.index
    rs_arr = full_rs_net.values.astype(float)
    b6040_arr = full_b6040_net.reindex(idx).fillna(0).values.astype(float)
    bew_arr = full_bew_net.reindex(idx).fillna(0).values.astype(float)
    n = len(rs_arr)

    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = _bootstrap_indices(n, BLOCK_LEN, N_BOOT, rng)

    rs_metrics = _boot_metrics_array(rs_arr[boot_idx])
    b6040_metrics = _boot_metrics_array(b6040_arr[boot_idx])
    bew_metrics = _boot_metrics_array(bew_arr[boot_idx])

    metrics_keys = ["cagr", "vol", "sharpe", "max_drawdown", "calmar"]

    # Confidence intervals
    ci_rows = []
    for strat_name, mlist in [("RegimeShift", rs_metrics),
                                ("Static_6040", b6040_metrics),
                                ("Equal_Weight", bew_metrics)]:
        for key in metrics_keys:
            vals = np.array([m[key] for m in mlist])
            vals = vals[np.isfinite(vals)]
            if len(vals) < 10:
                continue
            ci_rows.append({
                "strategy": strat_name, "metric": key,
                "mean": float(np.mean(vals)), "median": float(np.median(vals)),
                "ci_lower_95": float(np.percentile(vals, 2.5)),
                "ci_upper_95": float(np.percentile(vals, 97.5)),
                "n_valid": len(vals),
            })
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)
    logger.info("Saved bootstrap_confidence_intervals.csv")

    # Paired differences
    diff_rows = []
    for key in ["sharpe", "cagr", "max_drawdown"]:
        for vs_name, vs_metrics in [("vs_6040", b6040_metrics), ("vs_EW", bew_metrics)]:
            diffs = np.array([r[key] - b[key] for r, b in zip(rs_metrics, vs_metrics)])
            diffs = diffs[np.isfinite(diffs)]
            if len(diffs) < 10:
                continue
            ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
            zero_inside = ci_lo <= 0 <= ci_hi
            diff_rows.append({
                "metric": f"RegimeShift_{key}_{vs_name}",
                "mean_diff": float(np.mean(diffs)),
                "ci_lower_95": ci_lo, "ci_upper_95": ci_hi,
                "zero_inside_ci": zero_inside,
                "statistically_significant": not zero_inside,
                "n_valid": len(diffs),
                "note": ("No significance claimed when zero lies inside CI"
                         if zero_inside else "Zero outside CI; causal retrospective data only"),
            })
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(output_dir / "bootstrap_benchmark_differences.csv", index=False)
    logger.info("Saved bootstrap_benchmark_differences.csv")

    # Charts
    rs_sharpes = [m["sharpe"] for m in rs_metrics]
    b6040_sharpes = [m["sharpe"] for m in b6040_metrics]
    bew_sharpes = [m["sharpe"] for m in bew_metrics]
    diff_rs_6040 = [r["sharpe"] - b["sharpe"] for r, b in zip(rs_metrics, b6040_metrics)]
    diff_rs_ew = [r["sharpe"] - b["sharpe"] for r, b in zip(rs_metrics, bew_metrics)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_s, ax_d = axes
    ax_s.hist(rs_sharpes, bins=60, alpha=0.6, label="RegimeShift", color="#1f77b4")
    ax_s.hist(b6040_sharpes, bins=60, alpha=0.6, label="Static 60/40", color="#ff7f0e")
    ax_s.hist(bew_sharpes, bins=60, alpha=0.6, label="Equal Weight", color="#2ca02c")
    ax_s.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax_s.set_title(f"Bootstrap Sharpe Distributions\n(n={N_BOOT}, block={BLOCK_LEN})")
    ax_s.set_xlabel("Bootstrapped Sharpe"); ax_s.legend(fontsize=9)

    lo_6040 = np.percentile(diff_rs_6040, 2.5); hi_6040 = np.percentile(diff_rs_6040, 97.5)
    lo_ew = np.percentile(diff_rs_ew, 2.5); hi_ew = np.percentile(diff_rs_ew, 97.5)
    ax_d.hist(diff_rs_6040, bins=60, alpha=0.7, label=f"RS - 60/40 [{lo_6040:.3f},{hi_6040:.3f}]", color="#8c564b")
    ax_d.hist(diff_rs_ew, bins=60, alpha=0.7, label=f"RS - EW [{lo_ew:.3f},{hi_ew:.3f}]", color="#e377c2")
    ax_d.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Zero")
    ax_d.axvspan(lo_6040, hi_6040, alpha=0.1, color="#8c564b")
    ax_d.axvspan(lo_ew, hi_ew, alpha=0.1, color="#e377c2")
    ax_d.set_title("Paired Sharpe Differences (95% CI shaded)")
    ax_d.set_xlabel("Sharpe Difference"); ax_d.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, output_dir / "bootstrap_sharpe_distributions.png")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for label, diffs, color in [("RS - 60/40 Sharpe", diff_rs_6040, "#8c564b"),
                                  ("RS - EW Sharpe", diff_rs_ew, "#e377c2")]:
        lo = np.percentile(diffs, 2.5); hi = np.percentile(diffs, 97.5)
        ax2.hist(diffs, bins=60, alpha=0.6,
                 label=f"{label}\n95% CI: [{lo:.3f}, {hi:.3f}]", color=color)
        ax2.axvspan(lo, hi, alpha=0.08, color=color)
    ax2.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Zero")
    ax2.set_title("Bootstrap Paired Sharpe Difference Distributions")
    ax2.set_xlabel("Sharpe Difference (RegimeShift - Benchmark)"); ax2.legend(fontsize=9)
    fig2.tight_layout()
    _save_fig(fig2, output_dir / "bootstrap_difference_distributions.png")

    logger.info("Bootstrap complete.")
    return diff_df


# ===========================================================================
# Section 5: HMM Diagnostics
# ===========================================================================

def run_hmm_diagnostics(backtest_result, output_dir: Path) -> pd.DataFrame:
    logger.info("=== HMM Diagnostics ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_records = backtest_result.hmm_diagnostics

    if not diag_records:
        logger.warning("No HMM diagnostics recorded in backtest result.")
        pd.DataFrame().to_csv(output_dir / "hmm_diagnostics.csv", index=False)
        return pd.DataFrame()

    rows = []
    for rec in diag_records:
        restarts = rec.get("restarts", [])
        n_restarts = len(restarts) if restarts else 1
        converged_restarts = (sum(1 for r in restarts if r.get("converged", False))
                              if restarts else (1 if rec.get("converged") else 0))
        best_seed = restarts[-1].get("seed", None) if restarts else None
        rows.append({
            "date": str(rec.get("date", ""))[:10] if rec.get("date") else "",
            "regime": rec.get("regime", ""),
            "converged": bool(rec.get("converged", False)),
            "n_iter": int(rec.get("n_iter", 0)) if rec.get("n_iter") is not None else None,
            "log_likelihood": float(rec.get("log_likelihood", np.nan)),
            "turnover": float(rec.get("turnover", np.nan)),
            "cost": float(rec.get("cost", np.nan)),
            "n_restarts_attempted": n_restarts,
            "n_restarts_converged": converged_restarts,
            "best_seed": best_seed,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "hmm_diagnostics.csv", index=False)

    # State occupancy
    if "regime" in df.columns and len(df) > 0:
        occ = df["regime"].value_counts().reset_index()
        occ.columns = ["regime", "count"]
        occ["fraction"] = occ["count"] / occ["count"].sum()
        occ.to_csv(output_dir / "hmm_state_occupancy.csv", index=False)
        logger.info("State occupancy: %s",
                    occ.set_index("regime")["fraction"].to_dict())

    total_fits = len(df)
    conv_rate = float(df["converged"].mean()) if total_fits > 0 else np.nan
    logger.info("Total HMM fits: %d | Convergence rate: %.1f%%",
                total_fits, conv_rate * 100)

    # Posterior confidence chart
    probs_df = backtest_result.regime_probabilities
    if not probs_df.empty:
        max_prob = probs_df.max(axis=1)
        entropy_arr = -np.sum(
            probs_df.clip(lower=1e-12).values *
            np.log(probs_df.clip(lower=1e-12).values),
            axis=1,
        )
        fig, axes = plt.subplots(2, 1, figsize=(13, 7))
        ax_conf, ax_ent = axes
        max_prob.plot(ax=ax_conf, linewidth=0.8, color="#1f77b4", alpha=0.7)
        ax_conf.axhline(0.5, color="red", linewidth=0.8, linestyle="--", label="50%")
        ax_conf.set_ylabel("Max Posterior Probability")
        ax_conf.set_title("HMM Posterior Confidence Over Time")
        ax_conf.set_ylim(0, 1); ax_conf.legend()
        pd.Series(entropy_arr, index=probs_df.index).plot(
            ax=ax_ent, linewidth=0.8, color="#d62728", alpha=0.7)
        ax_ent.set_ylabel("Posterior Entropy")
        ax_ent.set_title("HMM Posterior Entropy (higher = more regime uncertainty)")
        fig.tight_layout()
        _save_fig(fig, output_dir / "hmm_posterior_confidence.png")

    logger.info("HMM diagnostics complete.")
    return df


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RegimeShift research validation suite")
    p.add_argument("--data-path", default="data/submission_market_data.csv")
    p.add_argument("--output-dir", default="results/research")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return 1

    logger.info("Loading canonical dataset: %s", data_path)
    config = RegimeShiftConfig()
    config.execution_cost_model = ExecutionCostModel.scenario("base", config.core_assets)
    prices = load_market_data_csv(path=str(data_path), config=config)
    logger.info("Loaded %d rows: %s to %s",
                len(prices), prices.index[0].date(), prices.index[-1].date())

    logger.info("Running full official backtest (base 10 bps, NEXT_CLOSE)...")
    rs = run_walk_forward_backtest(prices=prices, config=config, risk_free_rate=_RF)
    eval_start = rs.net_returns.index[0]

    b6040 = run_benchmark(prices=prices, weights=static_60_40_weights(), config=config,
                          rebalance_flags=rs.rebalance_flags, start_date=eval_start)
    bew = run_benchmark(prices=prices, weights=equal_weight_weights(), config=config,
                        rebalance_flags=rs.rebalance_flags, start_date=eval_start)

    logger.info("Official backtest: %s to %s, %d obs",
                eval_start.date(), rs.net_returns.index[-1].date(), len(rs.net_returns))

    run_ablation(prices, config, output_dir, rs_result=rs)
    run_subperiod(prices, config, output_dir, rs.net_returns, rs.gross_returns,
                  rs.turnover, rs.transaction_costs, b6040.net_returns, bew.net_returns)
    run_rolling_origin(prices, config, output_dir, rs.net_returns, rs.gross_returns,
                       rs.turnover, rs.transaction_costs, b6040.net_returns, bew.net_returns)
    run_bootstrap(rs.net_returns, b6040.net_returns, bew.net_returns, output_dir)
    run_hmm_diagnostics(rs, output_dir)

    logger.info("=== Research suite complete. Output: %s ===", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
