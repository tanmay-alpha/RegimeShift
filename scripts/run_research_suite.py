"""RegimeShift research validation suite (protocol v3).

All strategies run through the one lagged-fill engine on one verified index
(``require_common_index`` raises on any mismatch; nothing is filled).

Outputs (``--output-dir``):
  ablation_summary.csv              full-sample metrics, 7 strategies
  daily_accounting.csv              date x strategy gross/net/cost/turnover
  subperiod_summary.csv             chronological attribution, protocol splits
  calendar_attribution_summary.csv  chronological attribution, 3-year calendar slices
  walk_forward_selections.csv       yearly selection log (validation -> evaluation)
  walk_forward_summary.csv          selected strategy vs every fixed candidate, same years
  bootstrap_paired_differences.csv  paired moving-block bootstrap, Holm-adjusted
  bootstrap_block_sensitivity.csv   Sharpe-difference CI for block lengths 10/21/63
  selection_bias.json               Deflated Sharpe Ratio and CSCV PBO over the trial set
  hmm_diagnostics.csv, hmm_state_occupancy.csv
  research_metadata.json
  *.png                             charts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from regime_shift.backtest import _cost_model  # noqa: E402
from regime_shift.config import RegimeShiftConfig  # noqa: E402
from regime_shift.data import load_market_data_csv  # noqa: E402
from regime_shift.execution import ExecutionCostModel  # noqa: E402
from regime_shift.research.evaluation import (  # noqa: E402
    metrics_table,
    net_return_frame,
    window_table,
)
from regime_shift.research.registry import register_attempt, write_experiment_manifest  # noqa: E402
from regime_shift.research.spec import EVALUATION_TYPES, STRATEGIES  # noqa: E402
from regime_shift.research.stats import (  # noqa: E402
    cscv_pbo,
    deflated_sharpe_ratio,
    holm_adjust,
    paired_block_bootstrap,
)
from regime_shift.research.strategies import run_core_suite  # noqa: E402
from regime_shift.research.walk_forward import walk_forward_select  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
logging.getLogger("regime_shift").setLevel(logging.WARNING)
logger = logging.getLogger("research_suite")

BLOCK_LEN = 21
BLOCK_SENSITIVITY = (10, 21, 63)
N_BOOT = 2_000
BOOT_SEED = 42
CSCV_SPLITS = 16
WALK_FORWARD_LOOKBACK_YEARS = 3

SUBPERIODS = [
    ("Development", "2010-01-01", "2018-12-31"),
    ("Validation", "2019-01-01", "2021-12-31"),
    ("Retrospective Evaluation", "2022-01-01", "2026-12-31"),
]
CALENDAR_WINDOWS = [
    ("2013-2015", "2013-01-01", "2015-12-31"),
    ("2016-2018", "2016-01-01", "2018-12-31"),
    ("2019-2021", "2019-01-01", "2021-12-31"),
    ("2022-2024", "2022-01-01", "2024-12-31"),
    ("2025-2026", "2025-01-01", "2026-12-31"),
]
# LIQUIDBEES distributions are present in the Yahoo adjusted series only for this span.
CASH_PROXY_COMPLETE = ("2013-01-01", "2025-09-30")


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _equity_drawdown_chart(frame: pd.DataFrame, path: Path) -> None:
    fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for column in frame.columns:
        wealth = (1 + frame[column]).cumprod()
        wealth.plot(ax=ax_eq, label=column, linewidth=1.3)
        (wealth / wealth.cummax().clip(lower=1.0) - 1).plot(ax=ax_dd, label=column, linewidth=1.0)
    ax_eq.set_title("Net wealth, identical dates and costs (base 10 bps, NEXT_CLOSE)")
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax_eq.legend(fontsize=8, loc="upper left")
    ax_dd.set_title("Drawdown")
    ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    fig.tight_layout()
    _save(fig, path)


def _window_sharpe_chart(table: pd.DataFrame, title: str, path: Path) -> None:
    pivot = table.pivot(index="window", columns="strategy", values="sharpe")
    pivot = pivot.loc[list(dict.fromkeys(table["window"]))]
    ax = pivot.plot.bar(figsize=(12, 5), width=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net Sharpe (rf = 0)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=4)
    plt.xticks(rotation=0)
    _save(ax.figure, path)


def _bootstrap_chart(draws: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for (a, b, metric), values in draws.items():
        if metric == "sharpe":
            ax.hist(values, bins=60, alpha=0.45, label=f"{a} - {b}")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title(f"Paired moving-block bootstrap: Sharpe difference (block={BLOCK_LEN}, n={N_BOOT})")
    ax.legend(fontsize=7)
    _save(fig, path)


def hmm_diagnostics(rs_result, output_dir: Path) -> pd.DataFrame:
    rows = []
    for rec in rs_result.hmm_diagnostics:
        restarts = rec.get("restarts") or []
        rows.append({
            "date": str(rec["date"])[:10], "regime": rec["regime"], "converged": bool(rec["converged"]),
            "n_iter": rec["n_iter"], "log_likelihood": rec["log_likelihood"],
            "turnover": rec["turnover"], "cost": rec["cost"],
            "n_restarts_attempted": len(restarts),
            "n_restarts_converged": sum(bool(r.get("converged")) for r in restarts),
            "best_seed": next((r.get("seed") for r in restarts if r.get("selected")), None),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "hmm_diagnostics.csv", index=False)
    occupancy = df["regime"].value_counts().rename_axis("regime").reset_index(name="count")
    occupancy["fraction"] = occupancy["count"] / occupancy["count"].sum()
    occupancy.to_csv(output_dir / "hmm_state_occupancy.csv", index=False)

    probs = rs_result.regime_probabilities.dropna()
    p = probs.clip(lower=1e-12).to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    probs.max(axis=1).plot(ax=ax1, linewidth=0.8)
    ax1.set_ylabel("max filtered posterior")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("HMM filtered state posterior at each signal close (latent-state, not a forecast probability)")
    pd.Series(-(p * np.log(p)).sum(axis=1), index=probs.index).plot(ax=ax2, linewidth=0.8, color="C3")
    ax2.set_ylabel("posterior entropy (nats)")
    _save(fig, output_dir / "hmm_posterior_confidence.png")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RegimeShift research validation suite (protocol v3)")
    p.add_argument("--data-path", default="data/submission_market_data.csv")
    p.add_argument("--output-dir", default="results/research")
    p.add_argument("--cost-scenario", default="base", choices=["optimistic", "base", "stressed"])
    return p.parse_args()


def run_suite(args, trials_started: list) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = RegimeShiftConfig()
    config.cost_scenario = args.cost_scenario
    config.execution_cost_model = ExecutionCostModel.scenario(args.cost_scenario, config.core_assets)
    prices = load_market_data_csv(path=args.data_path, config=config)
    if config.vix_col in prices.columns:
        prices = prices.drop(columns=[config.vix_col])

    logger.info("Running 7 strategies on one engine...")
    runs, rs_result, ctx = run_core_suite(prices, config, on_trial_start=trials_started.append)
    ordered = list(runs.values())
    cash = ctx.cash_returns

    # 1. Full-sample ablation
    summary = metrics_table(ordered, cash)
    summary.insert(1, "description", summary["strategy"].map(STRATEGIES))
    summary.to_csv(out / "ablation_summary.csv", index=False)
    accounting = pd.concat(
        {sid: pd.DataFrame({"gross": r.gross, "net": r.net, "cost": r.costs, "turnover": r.turnover})
         for sid, r in runs.items()}, names=["strategy", "date"],
    ).reset_index()
    accounting.to_csv(out / "daily_accounting.csv", index=False, float_format="%.10g")
    net = net_return_frame(ordered)
    _equity_drawdown_chart(net, out / "ablation_equity_curves.png")

    # 2-3. Chronological attribution (slices of the same continuous runs)
    sub = window_table(ordered, SUBPERIODS, cash)
    sub.to_csv(out / "subperiod_summary.csv", index=False)
    _window_sharpe_chart(sub, "Protocol splits: chronological attribution (all retrospective)", out / "subperiod_sharpe.png")
    cal = window_table(ordered, CALENDAR_WINDOWS, cash)
    cal.to_csv(out / "calendar_attribution_summary.csv", index=False)
    _window_sharpe_chart(cal, "Calendar slices of one online run (not independent re-fits)", out / "calendar_attribution_sharpe.png")
    cash_complete = window_table(ordered, [("cash_proxy_complete", *CASH_PROXY_COMPLETE)], cash)
    cash_complete.to_csv(out / "cash_proxy_complete_window_summary.csv", index=False)

    # 4. Walk-forward model selection.  Attempt A (rf = 0 Sharpe) was run first and is kept:
    # it selects the near-cash min-variance book every year.  Attempt B scores excess over cash.
    wf_rows, wf_logs, wf_runs = [], [], {}
    for basis, cash_arg in (("sharpe_rf0", None), ("sharpe_vs_cash", cash)):
        trials_started.append(f"walk_forward_{basis}")
        wf = walk_forward_select(runs, ctx.asset_returns, _cost_model(ctx.config, None),
                                 lookback_years=WALK_FORWARD_LOOKBACK_YEARS,
                                 strategy_id=f"walk_forward_{basis}", cash_returns=cash_arg)
        wf_runs[basis] = wf.run
        wf_logs.append(wf.selections.assign(selection_score=basis))
        from regime_shift.research.evaluation import restart_from_cash
        table = metrics_table(
            [wf.run] + [restart_from_cash(r, wf.run.index, _cost_model(ctx.config, None)) for r in ordered],
            cash,
        )
        wf_rows.append(table.assign(selection_score=basis, evaluation_type="walk_forward_selection"))
    pd.concat(wf_logs).to_csv(out / "walk_forward_selections.csv", index=False)
    wf_table = pd.concat(wf_rows, ignore_index=True)
    wf_table.to_csv(out / "walk_forward_summary.csv", index=False)
    wf_index = wf_runs["sharpe_vs_cash"].index

    # 5. Paired bootstrap: RegimeShift vs every alternative; Holm across ALL rows below.
    pairs = [("regimeshift", sid) for sid in runs if sid != "regimeshift"]
    excess = net.sub(cash.loc[net.index], axis=0)
    boot_net, draws = paired_block_bootstrap(net, pairs, block_len=BLOCK_LEN, n_boot=N_BOOT, seed=BOOT_SEED)
    boot_exc, _ = paired_block_bootstrap(excess, pairs, metrics=["sharpe"], block_len=BLOCK_LEN,
                                         n_boot=N_BOOT, seed=BOOT_SEED)
    boot = pd.concat([boot_net.assign(return_basis="net"), boot_exc.assign(return_basis="excess_over_cash")],
                     ignore_index=True)
    boot["p_value_holm"] = holm_adjust(boot["p_value"].to_numpy())
    boot["significant_holm_5pct"] = boot["p_value_holm"] < 0.05
    boot.to_csv(out / "bootstrap_paired_differences.csv", index=False)
    _bootstrap_chart(draws, out / "bootstrap_sharpe_differences.png")
    sens = pd.concat([
        paired_block_bootstrap(frame, pairs, metrics=["sharpe"], block_len=b, n_boot=N_BOOT, seed=BOOT_SEED)[0]
        .assign(return_basis=basis)
        for basis, frame in (("net", net), ("excess_over_cash", excess)) for b in BLOCK_SENSITIVITY
    ])
    sens.to_csv(out / "bootstrap_block_sensitivity.csv", index=False)

    # 6. Selection bias: DSR for RegimeShift over the registered trial set; CSCV PBO on both bases
    per_period_sr = (net.mean() / net.std(ddof=1)).to_dict()
    dsr = deflated_sharpe_ratio(net["regimeshift"].to_numpy(), list(per_period_sr.values()))
    dsr_excess = deflated_sharpe_ratio(excess["regimeshift"].to_numpy(),
                                       list((excess.mean() / excess.std(ddof=1)).values))
    pbo = cscv_pbo(net, n_splits=CSCV_SPLITS)
    pbo_excess = cscv_pbo(excess, n_splits=CSCV_SPLITS)
    selection = {
        "dsr_regimeshift": dsr,
        "dsr_regimeshift_excess_over_cash": dsr_excess,
        "trial_set": list(net.columns),
        "trial_set_note": ("Registered trials only. Unrecorded development iterations before protocol v2 "
                           "are not counted, so the DSR threshold is a lower bound."),
        "cscv_net": {k: v for k, v in pbo.items() if k not in ("logits", "is_oos_pairs")},
        "cscv_excess_over_cash": {k: v for k, v in pbo_excess.items() if k not in ("logits", "is_oos_pairs")},
        "cscv_note": ("CSCV estimates how often the in-sample best of these fixed trials ranks below the "
                      "median out-of-sample. It is not a chronological test. On rf = 0 Sharpe the near-cash "
                      "min-variance trial wins every split, so PBO = 0 there says nothing about skill."),
    }
    (out / "selection_bias.json").write_text(json.dumps(selection, indent=2, default=float))

    # 7. HMM diagnostics
    hmm_diagnostics(rs_result, out)

    extra = {
        "execution_model": "NEXT_CLOSE",
        "eval_start": str(net.index[0].date()), "eval_end": str(net.index[-1].date()), "n_obs": len(net),
        "strategies": STRATEGIES, "evaluation_types": EVALUATION_TYPES,
        "bootstrap": {"block_len": BLOCK_LEN, "n_boot": N_BOOT, "seed": BOOT_SEED,
                      "block_sensitivity": list(BLOCK_SENSITIVITY), "p_value": "centred, two-sided; Holm across family"},
        "walk_forward": {"lookback_years": WALK_FORWARD_LOOKBACK_YEARS, "eval_start": str(wf_index[0].date()),
                         "attempts": {"A_sharpe_rf0": "max trailing 3y rf=0 net Sharpe (run first, kept)",
                                      "B_sharpe_vs_cash": "max trailing 3y Sharpe of net minus cash proxy"}},
        "cash_proxy": "bond (LIQUIDBEES adj. close); distributions recorded only "
                      f"{CASH_PROXY_COMPLETE[0]}..{CASH_PROXY_COMPLETE[1]}",
    }
    write_experiment_manifest(out, experiment_id="research_suite", description=__doc__.splitlines()[0],
                              dataset_path=args.data_path, config={"cost_scenario": args.cost_scenario},
                              extra=extra)
    rs_row = summary.set_index("strategy").loc["regimeshift"]
    wf_sel = wf_table.set_index(["selection_score", "strategy"])
    register_attempt(experiment_id="research_suite", status="completed", n_trials=len(runs) + 2,
                     dataset_path=args.data_path,
                     summary={"regimeshift_sharpe": float(rs_row["sharpe"]),
                              "regimeshift_sharpe_vs_cash": float(rs_row["sharpe_vs_cash"]),
                              "dsr": dsr["dsr"], "pbo_net": pbo["pbo"], "pbo_excess": pbo_excess["pbo"],
                              "wf_A_sharpe": float(wf_sel.loc[("sharpe_rf0", "walk_forward_sharpe_rf0"), "sharpe"]),
                              "wf_B_sharpe_vs_cash": float(
                                  wf_sel.loc[("sharpe_vs_cash", "walk_forward_sharpe_vs_cash"), "sharpe_vs_cash"])})
    logger.info("Research suite complete: %s", out)
    return 0


def main() -> int:
    args = parse_args()
    trials_started: list = []
    try:
        return run_suite(args, trials_started)
    except Exception as exc:
        register_attempt(
            experiment_id="research_suite", status="failed", n_trials=len(trials_started),
            dataset_path=args.data_path, error=f"{type(exc).__name__}: {exc}",
            summary={"trials_started": trials_started},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
