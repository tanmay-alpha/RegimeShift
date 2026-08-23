"""Build the RegimeShift submission notebook as JSON and write it to disk.

This notebook is generated from the REAL market dataset used by the
official submission. No synthetic data is generated here.

The notebook reflects the final repository state:
  - NEXT_CLOSE causal execution
  - Base 10 bps one-way assumed transaction costs (cost-scenario base)
  - Official corrected results
  - Research validation summaries (ablation, subperiod, rolling-origin,
    bootstrap, HMM diagnostics)

All code cells are written with execution_count=null and outputs=[] so that
genuine kernel execution is preserved.

Usage:
    python scripts/build_notebook.py
    python -m jupyter nbconvert --to notebook --execute \\
        notebooks/RegimeShift_Submission.ipynb --inplace \\
        --ExecutePreprocessor.timeout=3600
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Notebook cell builders — execution_count=null until genuinely run
# ---------------------------------------------------------------------------

def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _to_md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-Flavored Markdown table (no tabulate required)."""
    cols = list(df.columns)
    rows = df.values.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body_lines = ["| " + " | ".join(_fmt_cell(v) for v in row) + " |" for row in rows]
    return "\n".join([header, sep] + body_lines)


def _fmt_cell(v) -> str:
    if isinstance(v, float):
        if abs(v) >= 1000 or abs(v) < 0.001 and v != 0:
            return f"{v:.4e}"
        return f"{v:.4f}"
    return str(v) if not (isinstance(v, float) and v != v) else "nan"


# ---------------------------------------------------------------------------
# Canonical SHA-256 (CRLF-normalised, matching experiment_manifest.json)
# ---------------------------------------------------------------------------

def _sha256_canonical(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        digest.update(fh.read().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Notebook builder
# ---------------------------------------------------------------------------

def _notebook_json(
    prices: pd.DataFrame,
    diag: dict,
    sha256: str,
    strategy,
    bench_6040,
    bench_ew,
    perf_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame | None,
    subperiod_summary: pd.DataFrame | None,
    rolling_origin_summary: pd.DataFrame | None,
    bootstrap_diff: pd.DataFrame | None,
    hmm_diag: pd.DataFrame | None,
    tickers: dict,
    vix_included: bool,
    config,
) -> str:
    cells = []

    first_date = prices.index[0].strftime("%Y-%m-%d")
    last_date = prices.index[-1].strftime("%Y-%m-%d")
    n_rows = len(prices)
    cols = list(prices.columns)
    regime_counts = strategy.regime_series.value_counts().to_dict()
    n_rebalances = int(strategy.rebalance_flags.sum())
    n_obs = len(strategy.net_returns)

    equity_tk = tickers.get("equity", "^NSEI")
    gold_tk = tickers.get("gold", "GOLDBEES.NS")
    bond_tk = tickers.get("bond", "LIQUIDBEES.NS")

    bond_desc = (
        "LIQUIDBEES.NS — Nippon India Liquid Bees, INR-denominated "
        "liquid-bond / cash-equivalent proxy (NOT a sovereign bond or "
        "10-year G-Sec; very short duration, no long-term rate hedge)"
    )

    ffill_total = diag.get("ffill_cells_total", 0)
    ffill_per_asset = diag.get("ffill_cells_per_asset", {})
    dates_dropped = diag.get("dates_dropped", 0)

    # ---- Title ----
    cells.append(_md(
        "# RegimeShift: Market-Regime Asset Allocation System\n"
        "## IIT Bombay Summer Quant 2026 — Final Submission\n\n"
        "This notebook reproduces the complete RegimeShift pipeline:\n"
        "data → features → HMM → portfolio → backtest → metrics → research validation.\n\n"
        "All outputs are computed from the **real market dataset** loaded at\n"
        "the top of this notebook.\n\n"
        f"**Dataset SHA-256 (canonical):** `{sha256}`\n\n"
        f"**Official cost model:** base scenario — 10 bps one-way per asset\n\n"
        f"**Execution model:** NEXT_CLOSE (no same-bar fills)"
    ))

    # ---- 1. Research Question ----
    cells.append(_md(
        "### 1. Research Question\n\n"
        "Does a 3-state Gaussian HMM improve risk-adjusted returns (Sharpe ratio) "
        "for a three-asset Indian multi-asset portfolio — NIFTY 50 equity, "
        "GOLDBEES.NS gold, LIQUIDBEES.NS bond — relative to static benchmarks, "
        "under causal NEXT_CLOSE execution and realistic transaction cost assumptions?"
    ))

    # ---- 2. Asset Universe ----
    cells.append(_md(
        "### 2. Asset Universe\n\n"
        "| Role | Instrument | Ticker | Notes |\n"
        "|---|---|---|---|\n"
        f"| Equity | NIFTY 50 Index | `{equity_tk}` | **Index-level simulation — not directly executable** |\n"
        f"| Gold | Nippon India ETF Gold Bees | `{gold_tk}` | INR-denominated gold ETF |\n"
        "| Defensive | Nippon India Liquid Bees | `LIQUIDBEES.NS` | Liquid-bond / cash-equivalent proxy; short duration |\n"
        "| Volatility feature | India VIX | `^INDIAVIX` | Optional feature only; omitted from official submitted CSV |\n\n"
        f"**Index-level tradability limitation:** `{equity_tk}` is an index series, "
        "not a directly tradeable instrument. Any real implementation requires a "
        "separately verified equity ETF (e.g. Nippon India NIFTY 50 BeES). "
        "This is an **index-level allocation simulation**, not an executable portfolio.\n\n"
        f"**Bond proxy:** {bond_desc}"
    ))

    # ---- 3. Dataset ----
    cells.append(_md(
        "### 3. Dataset and Canonical Hash\n\n"
        "| Property | Value |\n"
        "|---|---|\n"
        f"| File | `data/submission_market_data.csv` |\n"
        f"| SHA-256 (canonical) | `{sha256}` |\n"
        f"| First date | `{first_date}` |\n"
        f"| Last date | `{last_date}` |\n"
        f"| Row count | `{n_rows}` |\n"
        f"| Columns | `{', '.join(cols)}` |\n"
        f"| VIX included | `{vix_included}` |\n"
        f"| Total forward-filled cells | `{ffill_total}` |\n"
        f"| Dates dropped (residual NaNs) | `{dates_dropped}` |\n\n"
        "Fill provenance is recorded explicitly by the data pipeline before any "
        "forward-fill occurs. Naturally constant prices (e.g. LIQUIDBEES.NS "
        "liquid-bond NAV) are not considered fill artefacts.\n\n"
        "Third-party market data is included only for academic reproducibility; "
        "it remains subject to the terms of its original providers and exchanges."
    ))
    cells.append(_code(
        'from pathlib import Path\n'
        'import sys\n'
        'sys.path.insert(0, str(Path(".").resolve() / "src"))\n'
        '\n'
        'from regime_shift.config import RegimeShiftConfig\n'
        'from regime_shift.execution import ExecutionCostModel\n'
        'from regime_shift.data import load_market_data_csv\n'
        '\n'
        'config = RegimeShiftConfig()\n'
        'config.execution_cost_model = ExecutionCostModel.scenario("base", config.core_assets)\n'
        '\n'
        'DATA_PATH = "data/submission_market_data.csv"\n'
        'prices = load_market_data_csv(path=DATA_PATH, config=config)\n'
        'diag = prices.attrs.get("data_diagnostics", {})\n'
        '\n'
        'import hashlib\n'
        'with open(DATA_PATH, "rb") as _fh:\n'
        '    _raw = _fh.read().replace(b"\\r\\n", b"\\n")\n'
        'canonical_sha256 = hashlib.sha256(_raw).hexdigest()\n'
        '\n'
        'print(f"Price data: {len(prices)} rows")\n'
        'print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")\n'
        'print(f"Columns: {list(prices.columns)}")\n'
        'print(f"Canonical SHA-256: {canonical_sha256}")\n'
        'print(f"Forward-filled cells: {diag.get(\'ffill_cells_total\', 0)}")\n'
        'print(f"Dates dropped: {diag.get(\'dates_dropped\', 0)}")\n'
        'display(prices.head())\n'
        'display(prices.describe())\n'
    ))

    # ---- 4. Causal NEXT_CLOSE Timing ----
    cells.append(_md(
        "### 4. Causal NEXT_CLOSE Execution Timing\n\n"
        "At close **t**:\n\n"
        "1. Holdings established after close **t-1** earn the **t-1 → t** return.\n"
        "2. Close **t** price becomes known.\n"
        "3. Features, scaler, and HMM may use information strictly through **t**.\n"
        "4. Regime and portfolio target are generated at close **t**.\n"
        "5. Execution costs are paid at close **t** (applied to returned wealth).\n"
        "6. New holdings first earn the **t → t+1** return.\n\n"
        "This is the NEXT_CLOSE convention. No same-bar fills. No future data.\n\n"
        "```\n"
        "Prices\n"
        "  ↓\n"
        "Causal trailing features (rolling windows, no lookahead)\n"
        "  ↓\n"
        "Train-only StandardScaler (fit on rolling window ≤ t)\n"
        "  ↓\n"
        "Rolling 3-state Gaussian HMM (fit on rolling window ≤ t)\n"
        "  ↓\n"
        "Regime-conditioned CVXPY allocation\n"
        "  ↓\n"
        "NEXT_CLOSE execution (costs at close t, return starts t→t+1)\n"
        "  ↓\n"
        "Benchmark-aligned evaluation with identical cost model\n"
        "```"
    ))

    # ---- 5. Leakage-Safe Features ----
    cells.append(_md(
        "### 5. Leakage-Safe Features\n\n"
        "Seven base features computed from strictly trailing windows:\n"
        "log returns, realized volatility (21d, 63d), price-to-MA ratios, "
        "MACD signal, rolling drawdown, and cross-asset correlations.\n"
        "VIX features are optional and omitted when VIX is not in the dataset.\n\n"
        "The StandardScaler is fitted only on the rolling training window "
        "(252 days through close **t**). No future data reaches the scaler."
    ))
    cells.append(_code(
        'from regime_shift.features import (\n'
        '    compute_raw_features, drop_feature_warmup,\n'
        '    fit_feature_scaler, transform_features,\n'
        ')\n'
        '\n'
        'feature_cfg = config.feature_config\n'
        'raw_features = compute_raw_features(prices, config=feature_cfg)\n'
        'raw_features = drop_feature_warmup(raw_features, config=feature_cfg)\n'
        'print(f"Features after warmup: {len(raw_features)} rows, {len(raw_features.columns)} columns")\n'
        'print(f"Feature columns: {list(raw_features.columns)}")\n'
        'display(raw_features.head())\n'
    ))

    # ---- 6. Train-Only Scaler & HMM ----
    cells.append(_md(
        "### 6. Train-Only StandardScaler and 3-Restart Gaussian HMM\n\n"
        f"Training window: **{config.train_window} days** rolling.\n"
        f"HMM: 3 states, diagonal covariance, {config.hmm_config.n_restarts} "
        "deterministic restarts per fit.\n\n"
        "Restart selection: highest converged training log-likelihood. "
        "**No selection based on out-of-sample performance.**"
    ))
    cells.append(_code(
        'from regime_shift.regime_model import fit_hmm\n'
        '\n'
        'train_end = raw_features.index[min(config.train_window, len(raw_features) - 1)]\n'
        'train_features = raw_features.loc[:train_end].tail(config.train_window)\n'
        'scaler = fit_feature_scaler(train_features)\n'
        'scaled_train = transform_features(scaler, train_features)\n'
        'hmm, hidden_states, trans_mat = fit_hmm(scaled_train, train_features, config=config.hmm_config)\n'
        'print(f"HMM converged: {hmm.monitor_.converged}")\n'
        'print(f"Iterations: {hmm.monitor_.n_iter}")\n'
        'print(f"Log-likelihood: {hmm.score(scaled_train.values):.2f}")\n'
        'print("Transition matrix:")\n'
        'display(trans_mat)\n'
    ))

    # ---- 7. State Mapping ----
    cells.append(_md(
        "### 7. Bull / Bear / Crisis State Mapping\n\n"
        "Numeric HMM states are mapped to interpretable labels using "
        "training-period volatility, momentum, and VIX statistics only. "
        "**No out-of-sample performance is used for state labelling.**"
    ))
    cells.append(_code(
        'from regime_shift.regime_model import predict_current_state\n'
        'solution = predict_current_state(\n'
        '    hmm, scaler, train_features, train_features, config=config.hmm_config,\n'
        ')\n'
        'print(f"Current regime: {solution.regime}")\n'
        'print(f"Regime probabilities: {dict(solution.probabilities)}")\n'
        'print(f"Convergence: {solution.convergence}")\n'
        'print(f"Iterations: {solution.n_iter}")\n'
    ))

    # ---- 8. CVXPY Portfolio Construction ----
    cells.append(_md(
        "### 8. Regime-Conditioned CVXPY Portfolio Construction\n\n"
        "Each regime has a distinct convex objective and regime-specific weight constraints:\n\n"
        "| Regime | Objective | Key Constraints |\n"
        "|---|---|---|\n"
        "| Bull | Maximize return − risk_aversion × variance | Equity ≥ 45%, ≤ 80% |\n"
        "| Bear | Minimize variance − return_reward × return | Equity ≤ 40%; gold+bond ≥ 60% |\n"
        "| Crisis | Minimize variance | Equity ≤ 15%; gold ≥ 25%; bond ≥ 40% |\n\n"
        "Ridge regularization is applied to the covariance matrix for PSD guarantee.\n"
        "All allocations are long-only; no leverage."
    ))
    cells.append(_code(
        'from regime_shift.portfolio import optimize_portfolio\n'
        '\n'
        'asset_returns = prices[["equity", "gold", "bond"]].pct_change().iloc[1:]\n'
        'est_returns = asset_returns.loc[:train_end].tail(config.portfolio_config.estimation_lookback)\n'
        '\n'
        'port_sol = optimize_portfolio(\n'
        '    regime=solution.regime,\n'
        '    returns_through_date=est_returns,\n'
        '    previous_weights=None,\n'
        '    config=config.portfolio_config,\n'
        ')\n'
        'print(f"Regime: {port_sol.regime}")\n'
        'print(f"Solver: {port_sol.solver} | Status: {port_sol.status}")\n'
        'print(f"Weights: equity={port_sol.weights[\'equity\']:.4f}, "\n'
        '      f"gold={port_sol.weights[\'gold\']:.4f}, "\n'
        '      f"bond={port_sol.weights[\'bond\']:.4f}")\n'
    ))

    # ---- 9. Base 10 bps Transaction Costs ----
    cells.append(_md(
        "### 9. Transaction Costs — Base Scenario (10 bps one-way)\n\n"
        "Cost model: **base scenario** — 10 basis points one-way per asset.\n\n"
        "- **Initial allocation:** turnover = Σ |w_i| (full L1).\n"
        "- **Subsequent rebalance:** turnover = 0.5 × Σ |w_i − w_{prev}| (half-L1).\n"
        "- **Net return:** r_net = (1 + r_gross) × (1 − cost) − 1.\n"
        "- No cost on non-rebalance dates.\n\n"
        "**Limitation:** These are assumed-cost scenarios, not measured historical "
        "spreads or fills. Market impact and capacity are excluded because "
        "volume/ADV data is absent. `^NSEI` is not a directly executable fill.\n\n"
        "Alternative scenarios: optimistic (5 bps), stressed (20 bps)."
    ))

    # ---- 10. Walk-Forward Backtest ----
    cells.append(_md(
        "### 10. Walk-Forward Backtest with NEXT_CLOSE\n\n"
        f"- **Evaluation observations:** {n_obs}\n"
        f"- **Rebalance frequency:** {config.rebalance_frequency} trading days\n"
        f"- **Minimum training window:** {config.minimum_training_observations} observations\n"
        f"- **Regime counts (rebalance dates):** {regime_counts}\n"
        f"- **Total rebalances:** {n_rebalances}"
    ))
    cells.append(_code(
        'from regime_shift.backtest import run_walk_forward_backtest, run_benchmark\n'
        'from regime_shift.benchmarks import static_60_40_weights, equal_weight_weights\n'
        '\n'
        'strategy = run_walk_forward_backtest(\n'
        '    prices=prices, config=config, risk_free_rate=0.0,\n'
        ')\n'
        'print(f"Backtest complete.")\n'
        'print(f"Evaluation: {len(strategy.net_returns)} trading days")\n'
        'print(f"  {strategy.net_returns.index[0].date()} to {strategy.net_returns.index[-1].date()}")\n'
        'print(f"Rebalances: {strategy.rebalance_flags.sum()}")\n'
        'for r in ["Bull", "Bear", "Crisis"]:\n'
        '    print(f"  {r}: {strategy.regime_series.value_counts().get(r, 0)} days")\n'
    ))
    cells.append(_code(
        'bench_6040 = run_benchmark(\n'
        '    prices=prices, weights=static_60_40_weights(), config=config,\n'
        '    rebalance_flags=strategy.rebalance_flags,\n'
        '    start_date=strategy.net_returns.index[0],\n'
        ')\n'
        'bench_ew = run_benchmark(\n'
        '    prices=prices, weights=equal_weight_weights(), config=config,\n'
        '    rebalance_flags=strategy.rebalance_flags,\n'
        '    start_date=strategy.net_returns.index[0],\n'
        ')\n'
        'print("Benchmarks complete.")\n'
    ))

    # ---- 11. Official Results ----
    cells.append(_md(
        "### 11. Official Aligned Results\n\n"
        "All strategies share the same evaluation dates, NEXT_CLOSE execution, "
        "base 10 bps cost model, and risk-free rate of 0."
    ))
    cells.append(_code(
        'from regime_shift.metrics import compute_performance_metrics\n'
        'import pandas as pd\n'
        '\n'
        'def _row(label, net, gross, turn, costs):\n'
        '    m = compute_performance_metrics(\n'
        '        net, gross_returns=gross, turnover=turn, transaction_costs=costs, risk_free_rate=0.0)\n'
        '    d = m.to_dict(); d["Strategy"] = label; return d\n'
        '\n'
        'perf_rows = [\n'
        '    _row("RegimeShift Gross", strategy.gross_returns, strategy.gross_returns,\n'
        '         strategy.turnover, strategy.transaction_costs),\n'
        '    _row("RegimeShift Net",   strategy.net_returns,   strategy.gross_returns,\n'
        '         strategy.turnover, strategy.transaction_costs),\n'
        '    _row("Static 60/40 Gross", bench_6040.gross_returns, bench_6040.gross_returns,\n'
        '         bench_6040.turnover, bench_6040.transaction_costs),\n'
        '    _row("Static 60/40 Net",   bench_6040.net_returns,   bench_6040.gross_returns,\n'
        '         bench_6040.turnover, bench_6040.transaction_costs),\n'
        '    _row("Equal Weight Gross", bench_ew.gross_returns, bench_ew.gross_returns,\n'
        '         bench_ew.turnover, bench_ew.transaction_costs),\n'
        '    _row("Equal Weight Net",   bench_ew.net_returns,   bench_ew.gross_returns,\n'
        '         bench_ew.turnover, bench_ew.transaction_costs),\n'
        ']\n'
        'perf_df = pd.DataFrame(perf_rows)\n'
        'display(perf_df[["Strategy","CAGR","Sharpe","Sortino","Maximum Drawdown",\n'
        '                  "Calmar","Annualised Turnover","Transaction Cost Drag"]])\n'
    ))

    ablation_md = "### 12. Ablation Study Results\n\n"
    if ablation_summary is not None and not ablation_summary.empty:
        ab_cols = ["strategy", "sharpe", "cagr", "max_drawdown", "ann_turnover"]
        ablation_md += _to_md_table(ablation_summary[ab_cols]) + "\n\n"
    else:
        ablation_md += "_Run `python scripts/run_research_suite.py` to generate ablation results._\n\n"
    ablation_md += (
        "Ablation strategies: (A) Full RegimeShift HMM+CVXPY, (B) No-regime optimizer (always Bull), "
        "(C) Volatility-rule allocation, (D) HMM+fixed weights, (E) Minimum variance, "
        "(F) Static 60/40, (G) Equal Weight. All share identical dates, costs, and execution."
    )
    cells.append(_md(ablation_md))

    # ---- 13. Chronological Subperiod Results ----
    sub_md = "### 13. Chronological Subperiod Analysis\n\n"
    sub_md += (
        "> **Important:** The retrospective evaluation period (2022-2026) is **not** a pristine "
        "holdout because the full historical sample was inspected during earlier project development.\n\n"
    )
    if subperiod_summary is not None and not subperiod_summary.empty:
        rs_sub = subperiod_summary[subperiod_summary["strategy"] == "RegimeShift"][
            ["period", "n_obs", "cagr", "sharpe", "max_drawdown"]]
        sub_md += _to_md_table(rs_sub) + "\n"
    else:
        sub_md += "_Run `python scripts/run_research_suite.py` to generate subperiod results._\n"
    cells.append(_md(sub_md))

    # ---- 14. Rolling-Origin Results ----
    roll_md = "### 14. Rolling-Origin Evaluation\n\n"
    if rolling_origin_summary is not None and not rolling_origin_summary.empty:
        rs_roll = rolling_origin_summary[rolling_origin_summary["strategy"] == "RegimeShift"][
            ["fold", "n_obs", "cagr", "sharpe", "max_drawdown"]]
        roll_md += _to_md_table(rs_roll) + "\n"
    else:
        roll_md += "_Run `python scripts/run_research_suite.py` to generate rolling-origin results._\n"
    cells.append(_md(roll_md))

    # ---- 15. Bootstrap Uncertainty ----
    boot_md = "### 15. Bootstrap Uncertainty (Paired Moving-Block, n=2000, block=21)\n\n"
    if bootstrap_diff is not None and not bootstrap_diff.empty:
        boot_md += _to_md_table(bootstrap_diff) + "\n\n"
        boot_md += (
            "No significance is claimed when zero lies inside a confidence interval. "
            "All results are retrospective causal simulation data only."
        )
    else:
        boot_md += "_Run `python scripts/run_research_suite.py` to generate bootstrap results._\n"
    cells.append(_md(boot_md))

    # ---- 16. HMM Diagnostics ----
    hmm_md = "### 16. HMM Stability Diagnostics\n\n"
    if hmm_diag is not None and not hmm_diag.empty:
        total_fits = len(hmm_diag)
        conv_rate = float(hmm_diag["converged"].mean()) * 100 if total_fits > 0 else float("nan")
        hmm_md += f"- Total rolling HMM fits: **{total_fits}**\n"
        hmm_md += f"- Convergence rate: **{conv_rate:.1f}%**\n"
        if "log_likelihood" in hmm_diag.columns:
            hmm_md += f"- Median log-likelihood: **{hmm_diag['log_likelihood'].median():.2f}**\n"
        if "regime" in hmm_diag.columns:
            occ = hmm_diag["regime"].value_counts(normalize=True)
            for regime, frac in occ.items():
                hmm_md += f"- State occupancy ({regime}): **{frac:.1%}**\n"
    else:
        hmm_md += "_Run `python scripts/run_research_suite.py` to generate HMM diagnostics._\n"
    cells.append(_md(hmm_md))

    # ---- 17. Negative Findings ----
    cells.append(_md(
        "### 17. Negative Findings\n\n"
        "**RegimeShift did not outperform the simpler static benchmarks on headline "
        "risk-adjusted performance.**\n\n"
        "Under corrected NEXT_CLOSE causal timing and base 10 bps transaction costs:\n\n"
        "| Strategy | CAGR | Sharpe | Max Drawdown |\n"
        "|---|---|---|---|\n"
        f"| RegimeShift | ~6.0% | ~0.40 | ~35.8% |\n"
        f"| Static 60/40 | ~6.6% | ~0.70 | ~23.4% |\n"
        f"| Equal Weight | ~8.8% | ~0.57 | ~33.0% |\n\n"
        "(Read exact numbers from `results/submission/performance_summary.csv`.)\n\n"
        "**What this research demonstrated:**\n"
        "- The causal pipeline design (rolling train-only scaler+HMM, NEXT_CLOSE ordering) "
        "is leak-free and reproducible.\n"
        "- HMM regime detection produces interpretable Bull/Bear/Crisis states with "
        "reasonable transition persistence.\n"
        "- The added complexity of dynamic regime allocation does not generate alpha "
        "over static equal-weight or 60/40 portfolios in this Indian multi-asset dataset.\n"
        "- High turnover (3.3× annualised) relative to static benchmarks (~0.18×) "
        "creates significant cost drag that offsets any gross diversification benefit.\n"
        "- Bootstrap confidence intervals for paired Sharpe differences generally "
        "include zero, consistent with no statistically significant outperformance."
    ))

    # ---- 18. Limitations ----
    cells.append(_md(
        "### 18. Limitations\n\n"
        "1. **Index-level equity simulation:** `^NSEI` is an index, not a directly "
        "executable instrument. A real portfolio requires an equity ETF with verified "
        "common history, corporate-action treatment, and liquidity audit.\n"
        "2. **Assumed costs, not measured:** 5/10/20 bps are scenario assumptions. "
        "Market impact, bid-ask spreads, and capacity are not modelled.\n"
        "3. **Short-duration bond proxy:** LIQUIDBEES.NS provides no long-term "
        "duration hedge or inflation protection.\n"
        "4. **Retrospective evaluation:** All periods were visible during development. "
        "No segment is a pristine holdout.\n"
        "5. **Three assets only:** Broader diversification requires expanding the universe.\n"
        "6. **Gaussian HMM:** Student-t emissions may better capture fat tails.\n"
        "7. **No alpha proven:** Positive CAGR reflects the underlying market uptrend, "
        "not superior risk-adjusted returns."
    ))

    # ---- 19. Reproduction ----
    cells.append(_md(
        "### 19. Reproduction\n\n"
        "```bash\n"
        "pip install -e \".[dev]\"\n\n"
        "# Official experiment\n"
        "python run_submission.py \\\n"
        "    --data-path data/submission_market_data.csv \\\n"
        "    --cost-scenario base \\\n"
        "    --output-dir results/submission\n\n"
        "# Research validation suite\n"
        "python scripts/run_research_suite.py\n\n"
        "# Rebuild and execute this notebook\n"
        "python scripts/build_notebook.py\n"
        "python -m jupyter nbconvert --to notebook --execute \\\n"
        "    notebooks/RegimeShift_Submission.ipynb --inplace \\\n"
        "    --ExecutePreprocessor.timeout=3600\n\n"
        "# Tests\n"
        "python -m pytest -q\n"
        "python scripts/verify_release.py\n"
        "```"
    ))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))

    from regime_shift.config import RegimeShiftConfig
    from regime_shift.execution import ExecutionCostModel
    from regime_shift.data import load_market_data_csv
    from regime_shift.backtest import run_walk_forward_backtest, run_benchmark
    from regime_shift.benchmarks import static_60_40_weights, equal_weight_weights

    print("Building notebook from real market data (base 10 bps scenario)...")
    cfg = RegimeShiftConfig()
    cfg.execution_cost_model = ExecutionCostModel.scenario("base", cfg.core_assets)

    data_path = _Path("data/submission_market_data.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    prices = load_market_data_csv(path=str(data_path), config=cfg)
    sha = _sha256_canonical(data_path)
    diag = prices.attrs.get("data_diagnostics", {})

    print("Running official backtest for notebook data...")
    strategy = run_walk_forward_backtest(prices=prices, config=cfg, risk_free_rate=0.0)
    bench_6040 = run_benchmark(
        prices=prices, weights=static_60_40_weights(), config=cfg,
        rebalance_flags=strategy.rebalance_flags,
        start_date=strategy.net_returns.index[0],
    )
    bench_ew = run_benchmark(
        prices=prices, weights=equal_weight_weights(), config=cfg,
        rebalance_flags=strategy.rebalance_flags,
        start_date=strategy.net_returns.index[0],
    )

    # Load research results if available
    research_dir = _Path("results/research")
    ablation_summary = None
    subperiod_summary = None
    rolling_origin_summary = None
    bootstrap_diff = None
    hmm_diag = None

    for attr, fname in [
        ("ablation_summary", "ablation_summary.csv"),
        ("subperiod_summary", "subperiod_summary.csv"),
        ("rolling_origin_summary", "rolling_origin_summary.csv"),
        ("bootstrap_diff", "bootstrap_benchmark_differences.csv"),
        ("hmm_diag", "hmm_diagnostics.csv"),
    ]:
        fp = research_dir / fname
        if fp.exists():
            locals()[attr] = pd.read_csv(fp)
            print(f"  Loaded {fname}")
        else:
            print(f"  {fname} not found — run scripts/run_research_suite.py first")

    # hmm_diag was loaded from file above; no fallback import needed
    # (run `python scripts/run_research_suite.py` to generate hmm_diagnostics.csv)

    tickers = {
        "equity": cfg.tickers.equity_ticker,
        "gold": cfg.tickers.gold_ticker,
        "bond": cfg.tickers.bond_ticker,
        "vix": cfg.tickers.vix_ticker,
    }

    nb_json = _notebook_json(
        prices=prices,
        diag=diag,
        sha256=sha,
        strategy=strategy,
        bench_6040=bench_6040,
        bench_ew=bench_ew,
        perf_summary=None,
        ablation_summary=ablation_summary,
        subperiod_summary=subperiod_summary,
        rolling_origin_summary=rolling_origin_summary,
        bootstrap_diff=bootstrap_diff,
        hmm_diag=hmm_diag,
        tickers=tickers,
        vix_included=False,
        config=cfg,
    )

    out_path = _Path(__file__).parent.parent / "notebooks" / "RegimeShift_Submission.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(nb_json)
    print(f"Wrote {out_path}")
    print(f"  Cells: {len(json.loads(nb_json)['cells'])}")
    print("Next: python -m jupyter nbconvert --to notebook --execute "
          "notebooks/RegimeShift_Submission.ipynb --inplace "
          "--ExecutePreprocessor.timeout=3600")