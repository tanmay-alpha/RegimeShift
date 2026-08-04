"""Build an unexecuted, reproducible RegimeShift research notebook.

The notebook deliberately starts with null execution counts and no outputs.
Use nbconvert to execute it; execution state is never fabricated here.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "RegimeShift_Submission.ipynb"


def markdown(title: str, text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n\n{text}\n"]}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [source + "\n"]}


def build() -> dict:
    cells = [
        markdown("Research question", "Can causal regime-aware allocation improve fixed multi-asset allocations after stated costs? This is retrospective research, not investment advice."),
        markdown("Objective and reproducibility", "Objective: evaluate a causal allocation protocol. Reproducibility comes from committed data, manifests, code, and executed artefacts."),
        markdown("Asset universe and data validation", "The asset universe has equity, gold, and defensive series. Data validation enforces positive prices, ordered dates, and no backward fill."),
        markdown("Data and index-level tradability", "The equity series is ^NSEI: an index-level signal and return proxy, not a directly executable fill. Gold is GOLDBEES and the defensive proxy is LIQUIDBEES."),
        markdown("Canonical data contract", "The canonical CSV hash is read from the official manifest; its CRLF/LF-normalization policy is preserved."),
        markdown("Retrospective research protocol", "2022-2026 is a retrospective evaluation period, not a pristine holdout, because the complete historical sample had been inspected during earlier development."),
        markdown("NEXT_CLOSE event timeline", "Installed weights earn first, then close-time information is observed, costs are paid, and targets apply to the next close-to-close return."),
        markdown("Walk-forward causal execution", "Walk-forward decisions use only observations available at each close; no same-bar execution is restored."),
        markdown("Leakage-safe features and train-only scaling", "Features, scaler fitting, and HMM estimation use information available through each rebalance date only."),
        markdown("Three-restart Gaussian HMM and state interpretation", "Three deterministic candidates are selected by likelihood, never later portfolio performance; Bull, Bear, and Crisis labels are training-data interpretations."),
        markdown("Bull/Bear/Crisis state mapping", "Bull, Bear, and Crisis are deterministic labels based on the fitted training-state characteristics."),
        markdown("CVXPY portfolio construction and asset-level costs", "Policies select targets only. The shared causal engine owns return accrual, drift, turnover, and the base 10 bps one-way cost assumption."),
        markdown("Transaction costs and 60/40 and equal weight", "The base scenario charges 10 bps one-way asset-level costs. Static 60/40 and equal weight use the same causal event ordering."),
        markdown("Official common-horizon results", "All official metrics use 3,732 aligned observations under NEXT_CLOSE and the base cost scenario."),
        markdown("Performance metrics and charts", "CAGR, Sharpe, drawdown, turnover, and cost drag are computed on common indices. The cells below display committed charts."),
        markdown("Controlled ablations", "No-regime, volatility-rule, fixed-HMM, and minimum-variance alternatives use the identical accounting engine."),
        markdown("Subperiod results and rolling-origin results", "Chronological summaries expose development, validation, and retrospective periods without calling any period untouched."),
        markdown("Paired bootstrap uncertainty", "A 21-day moving-block bootstrap, 2,000 samples, seed 42, and shared resampled block positions produces confidence intervals."),
        markdown("HMM diagnostics, negative findings, limitations, and reproducibility", "Negative benchmark and complexity findings remain visible. Capacity, impact, taxes, and executable-index fill quality are not modeled."),
        markdown("Conclusions", "The research does not establish outperformance. Simpler alternatives compare favorably, and uncertainty is reported rather than hidden."),
        code("""from pathlib import Path\nimport json, pandas as pd\nROOT = Path.cwd()\nif not (ROOT / 'results').exists(): ROOT = ROOT.parent\nsubmission = ROOT / 'results' / 'submission'\nresearch = ROOT / 'results' / 'research'\nmanifest = json.loads((submission / 'experiment_manifest.json').read_text())\nprint('Canonical hash:', manifest['dataset_sha256_canonical'])\nprint('Execution:', manifest['configuration']['execution_model'], '| Cost scenario:', manifest['configuration']['cost_scenario'])"""),
        code("""summary = pd.read_csv(submission / 'performance_summary.csv')\ndisplay(summary[['Strategy', 'CAGR', 'Sharpe', 'Maximum Drawdown', 'n_observations']])\nassert summary['n_observations'].nunique() == 1 and int(summary['n_observations'].iloc[0]) == 3732\nprint('Official result table matches committed CSV.')"""),
        code("""ablations = pd.read_csv(research / 'ablation_summary.csv')\nrolling = pd.read_csv(research / 'rolling_origin_summary.csv')\nbootstrap = pd.read_csv(research / 'bootstrap_benchmark_differences.csv')\ndisplay(ablations[['Strategy', 'Sharpe', 'Maximum Drawdown', 'Annualised Turnover']])\ndisplay(bootstrap)\nprint('Rolling-origin folds:', ', '.join(rolling['Fold'].unique()))"""),
        code("""from IPython.display import Image, display\nfor name in ['equity_curves.png', 'drawdowns.png', 'portfolio_weights.png', 'regime_price_chart.png', 'transition_matrix.png', 'regime_probabilities.png']:\n    print('submission artifact:', name)\nfor name in ['ablation_equity_curves.png', 'rolling_origin_sharpe.png', 'bootstrap_difference_distributions.png']:\n    print('research artifact:', name)\ndisplay(Image(filename=str(submission / 'equity_curves.png')))\ndisplay(Image(filename=str(research / 'ablation_equity_curves.png')))"""),
        code("""hmm = pd.read_csv(research / 'hmm_restart_summary.csv')\ndisplay(hmm)\nprint('Conclusion: negative benchmark and complexity findings remain visible.')"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"Wrote unexecuted notebook: {OUT}")
