# RegimeShift — Quantitative Regime & Portfolio Risk Laboratory

An offline Indian-market research laboratory for testing whether regime models and portfolio optimization improve risk control **after costs and against simple alternatives**. Python engine, reproducible CLI, executed notebook, optional Streamlit/Plotly researcher interface. No price-prediction product, brokerage integration or live trading.

**Finding:** complexity has not established an advantage. The Gaussian HMM allocation does not reliably beat equal weight or simpler regime rules. Negative controls and cash-relative metrics are first-class outputs, not hidden failures.

## Validated historical experiment

Three price proxies: NIFTY 50 (`^NSEI`), GOLDBEES, LIQUIDBEES. Signal at close t, fill at next session close t+1, first return t+1 → t+2. Base cost assumption: **10 bps per unit of absolute traded asset weight**, not a measured spread. All paired comparisons share exactly **3,900 dates, 2010-10-06 through 2026-07-27**; missing strategy returns are never filled with zero.

| Strategy | Net CAGR | Net Sharpe | Max Drawdown |
|---|---:|---:|---:|
| **RegimeShift** | 7.47% | 0.815 | 14.47% |
| Static 60/40 | 7.15% | 0.755 | 23.39% |
| Equal Weight | 8.92% | 1.179 | 14.95% |

Source: `results/submission/performance_summary.csv`; `scripts/verify_release.py` checks these rows. Sharpe uses rf=0 here. The research suite also reports Sharpe of daily return minus the cash proxy: **0.450 for RegimeShift versus 0.720 for equal weight**. The cash proxy itself has incomplete distribution history.

The corrected paired moving-block analysis uses 2,000 draws, seed 42, block sensitivity 10/21/63 sessions and Holm adjustment across 24 comparisons. RegimeShift's Sharpe difference against equal weight is **not significant after this adjustment** (adjusted p about 0.874). This supersedes the old marginal-interval significance claim. DSR and CSCV are exploratory selection-bias diagnostics with explicitly incomplete trial-history and independence assumptions, not evidence of alpha.

## What the laboratory does

- **Scientific accounting:** gross/net returns, actual modeled turnover and costs in every subperiod; one strict sample contract; initial capital included in drawdown; target-downside Sortino over all observations.
- **Regime intelligence:** filtered latent-state probabilities, normalized entropy, exploratory uncertainty alerts, transition persistence, censored regime spells, state-partition stability on overlapping refit windows, explanatory state feature means.
- **Fair model controls:** HMM + optimizer, no-regime optimizer, volatility rule, HMM fixed weights, momentum/volatility baseline and seeded IID regime negative control. No calibrated future-crash claim.
- **Portfolio laboratory:** constrained minimum variance, equal-risk contribution, Ledoit–Wolf covariance, turnover-penalized allocation and unlevered volatility targeting. Seven candidates × four cost assumptions (5/10/20/50 bps): **all 28 trials completed**.
- **Stress analysis:** historical replay with inherited holdings, distinct hypothetical correlation/concentration shocks, actual trade-delta cost sensitivity. No invented spreads, liquidity or overnight prices.
- **Research controls:** calendar attribution clearly separated from annual walk-forward strategy selection; decision/validation/evaluation ledger; paired bootstrap, exploratory DSR/CSCV; versioned attempts and hash-bound manifests, including failures.
- **Forward protocol:** immutable local source/config/data-prefix freeze, incremental new-row evaluation, mutation and overlap rejection. No local clock can certify prospective status; outputs explicitly decline that certification.
- **Researcher workflow:** validated JSON configurations; choose existing variants; inspect allocations, wealth, drawdown, turnover, costs and regime diagnostics; signal-linked allocation explanations; export generated artifacts without redistributing source prices.

### Negative findings worth discussing

The fixed-weight HMM achieved Sharpe 0.892; the momentum/volatility rule 0.987; a seeded random-regime control 1.025 over the same dates. One random seed is a diagnostic, not a formal null distribution, but the result undermines a strong claim that the HMM labels add value.

Minimum variance and risk parity largely exploit the near-cash leg. At 10 bps, equal-risk contribution had CAGR 5.14%, volatility 2.47% and Sharpe over cash 0.699, versus equal weight's CAGR 8.92%, volatility 7.48% and Sharpe over cash 0.720. Lower drawdown is not free excess return. The first walk-forward rule, scored on rf=0 Sharpe, chose minimum variance every year; that failed research design is retained alongside a separately labelled cash-relative selection attempt.

## Run it

Use a dedicated Python 3.11 virtual environment; the optional UI must not share an
incompatible FastAPI/Starlette installation. Exercised dependency versions are in
`config/reproducibility_constraints.txt`.

```bash
python -m venv .venv
# Activate .venv (Windows: .venv\Scripts\activate; POSIX: source .venv/bin/activate)
pip install -e ".[dev,app]" -c config/reproducibility_constraints.txt

# Explicit historical preparation; preserves the source and emits hash/repair diagnostics.
# Refuses to overwrite: use a new destination for another preparation attempt.
python -m regime_shift.cli prepare-data \
  --input data/submission_market_data.csv \
  --output data/cache/validated_market_data.csv

# Reproducible CLI; inspect/export never fit models.
python -m regime_shift.cli run --config config/laboratory.json
python -m regime_shift.cli inspect
python -m regime_shift.cli export results/experiments/YOUR_RUN --output reproduction.zip

# Local interface; models run only after explicit form submission.
streamlit run research_app.py
```

Core research reproductions:

```bash
python run_submission.py --data-path data/submission_market_data.csv \
  --cost-scenario base --output-dir results/submission
python scripts/run_research_suite.py
python scripts/run_regime_intelligence.py
python scripts/run_risk_lab.py
python scripts/build_notebook.py
python -m jupyter nbconvert --to notebook --execute notebooks/RegimeShift_Submission.ipynb \
  --inplace --ExecutePreprocessor.timeout=3600
python -m pytest -q
python scripts/verify_release.py
```

## Execution and data limitations

1. **Index-level, not executable performance.** `^NSEI` cannot be bought. Yahoo ETF history is not an independent corporate-action/liquidity audit; NIFTYBEES was investigated but has not replaced the research universe.
2. **Only conditional execution causality.** Models use history through signal close; fills occur on the following close. The legacy bad-print repair examines up to five later observations to identify reversals. Prepared data are therefore **ex-post cleaned**, not point-in-time. No end-to-end causal-data claim is made.
3. **Cash-proxy distributions are incomplete.** Canonical LIQUIDBEES values approximately match Yahoo adjusted close, not raw close. Vendor distribution records observed in the audit span 2013–September 2025; flat earlier/later stretches are material limitations. The suite reports that span separately, without certifying completeness.
4. **Costs are scenarios.** Market impact, capacity, measured spreads, dividends unavailable from source, executable opens and overnight-gap stress are not invented.
5. **Everything historical is retrospective.** Prior development inspected the complete sample. Post-dataset and post-release are not synonyms for unseen or prospective. No real trading profitability claim.
6. **Uncertainty is conditional.** Gaussian HMM state posteriors are not calibrated crash probabilities; annualized ratios and IID DSR approximations have dependence and selection limitations. Repeated experiments do not create independent evidence.

## Architecture and evidence

```text
src/regime_shift/             data, features, HMM, optimizer, causal event engine, metrics
src/regime_shift/research/    authoritative spec, strict accounting, comparisons, stats,
                             regime/risk diagnostics, workflow, registry, forward protocol
src/regime_shift/cli.py       UI-independent researcher commands
research_app.py              optional Streamlit/Plotly interface
config/                      protocol and explicit experiment configurations
results/submission/          validated original-universe experiment
results/research/            comparative and chronological research outputs
results/experiments/         versioned laboratory attempts and manifests
results/registry/            append-only attempt logs, including failures
results/archive/protocol_v2/  preserved historical evidence, not current validation
```

- [Scientific audit and reference decisions](docs/scientific_audit.md)
- [Regime diagnostics and negative controls](docs/regime_intelligence.md)
- [Portfolio risk and stress methodology](docs/risk_lab.md)
- [Research workflow and artifact schemas](docs/research_workflow.md)
- [Forward-evaluation integrity protocol](docs/forward_evaluation.md)
- [Technical walkthrough](docs/technical_walkthrough.md)

Design references: [Qlib](https://github.com/microsoft/qlib), [LEAN](https://github.com/QuantConnect/Lean), [vectorbt](https://github.com/polakowo/vectorbt), [Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf), [Deflated Sharpe Ratio](https://doi.org/10.3905/jpm.2014.40.5.094), [official NSE historical reports](https://www.nseindia.com/all-reports). These inform methodology; none was added as a platform dependency.

Originally developed for the IIT Bombay Summer Quant 2026 submission. Yahoo Finance/yfinance data remain subject to provider terms; code license does not grant market-data redistribution rights.
