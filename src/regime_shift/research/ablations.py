"""Controlled policy ablations using a single causal accounting engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from regime_shift.research.artefacts import plot_drawdowns, plot_equity, write_json
from regime_shift.research.common import load_canonical_prices, run_full_and_benchmarks
from regime_shift.research.policies import HMMFixedAllocation, MinimumVariance, NoRegimeOptimizer, VolatilityRule


def run_ablations(output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    prices = load_canonical_prices()
    variants = [("RegimeShift", None), ("No-regime optimizer", NoRegimeOptimizer()), ("Volatility rule", VolatilityRule()), ("HMM fixed allocations", HMMFixedAllocation()), ("Minimum variance", MinimumVariance())]
    returns, rows, turnovers, weights = {}, [], [], []
    full_run = None
    for label, policy in variants:
        run = run_full_and_benchmarks(prices, policy)
        full_run = full_run or run
        returns[label] = run.strategy.net_returns
        metric = dict(run.strategy.metrics); metric["Strategy"] = label; rows.append(metric)
        turnovers.append(pd.DataFrame({"date": run.strategy.turnover.index, "Strategy": label, "turnover": run.strategy.turnover.values}))
        weights.append(run.strategy.end_of_day_post_trade_weights.assign(Strategy=label).rename_axis("date").reset_index().melt(id_vars=["date", "Strategy"], var_name="asset", value_name="weight"))
    for label, benchmark in full_run.benchmarks.items():
        returns[label] = benchmark.net_returns
        metric = dict(benchmark.metrics); metric["Strategy"] = label; rows.append(metric)
        turnovers.append(pd.DataFrame({"date": benchmark.turnover.index, "Strategy": label, "turnover": benchmark.turnover.values}))
        weights.append(benchmark.daily_drifted_weights.assign(Strategy=label).rename_axis("date").reset_index().melt(id_vars=["date", "Strategy"], var_name="asset", value_name="weight"))
    daily = pd.DataFrame(returns)
    if daily.isna().any().any() or daily.apply(len).nunique() != 1:
        raise ValueError("Ablation policies must share one complete evaluation index.")
    summary = (
        pd.DataFrame(rows)
        .set_index("Strategy")
        .reindex(daily.columns)
        .rename_axis("Strategy")
        .reset_index()
    )
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    daily.to_csv(output_dir / "ablation_daily_returns.csv", index_label="date")
    pd.concat(turnovers, ignore_index=True).to_csv(output_dir / "ablation_turnover.csv", index=False)
    pd.concat(weights, ignore_index=True).to_csv(output_dir / "ablation_weights.csv", index=False)
    plot_equity(daily, output_dir / "ablation_equity_curves.png", "Controlled ablations: net equity")
    plot_drawdowns(daily, output_dir / "ablation_drawdowns.png", "Controlled ablations: drawdowns")
    ax = summary.set_index("Strategy")[["Sharpe", "Annualised Turnover"]].plot.bar(subplots=True, figsize=(10, 7), rot=30, legend=False, title=["Sharpe", "Annualised turnover"])
    ax[0].figure.tight_layout(); ax[0].figure.savefig(output_dir / "ablation_sharpe_turnover.png", dpi=180); ax[0].figure.clf()
    hmm_raw = [dict(item) for item in full_run.strategy.hmm_diagnostics]
    write_json(output_dir / "hmm_raw_diagnostics.json", {"diagnostics": hmm_raw})
    probabilities = full_run.strategy.regime_probabilities.copy()
    probabilities.to_csv(output_dir / "hmm_posterior_probabilities.csv", index_label="date")
    write_json(output_dir / "ablation_metadata.json", {
        "execution_model": "NEXT_CLOSE", "cost_scenario": "base", "effective_asset_cost_bps": 10,
        "n_observations": int(len(daily)), "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()),
        "variants": list(daily.columns), "return_accounting": "Shared run_walk_forward_backtest / run_benchmark engine",
    })
    return daily
