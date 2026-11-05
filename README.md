# RegimeShift — Causal Regime-Allocation Research

RegimeShift is a reproducible, index-level Indian multi-asset allocation study.
It uses rolling train-only features, a three-state Gaussian HMM and a CVXPY
regime-conditioned constrained optimizer, evaluated under causal NEXT_CLOSE
execution with stated per-asset cost assumptions.

**Research question:** Does HMM regime detection improve risk-adjusted returns
versus static benchmarks for a three-asset Indian portfolio — NIFTY 50,
GOLDBEES.NS gold, LIQUIDBEES.NS liquid-bond — under realistic cost assumptions?

**Answer (corrected experiment):** Mixed. RegimeShift beats static 60/40 on Sharpe
(0.81 vs 0.76) and max drawdown (14.5% vs 23.4%), but loses to equal weight
(Sharpe 1.18), and the paired bootstrap shows that gap is statistically significant.
The edge over 60/40 is not significant. Simpler ablations (HMM + fixed weights,
minimum variance) match or beat the CVXPY optimizer.

---

## Results

Official output is in `results/submission/` and uses `NEXT_CLOSE` causal
execution (signal at close t, fill at close t+1), three deterministic HMM restarts,
and the **base scenario (10 bps one-way per asset)**. 3,900 trading days,
2010-10-06 → 2026-07-27, identical for strategy and benchmarks.

**`^NSEI` is not a directly executable fill — this is an index-level simulation.**

| Strategy | Net CAGR | Net Sharpe | Max Drawdown |
|---|---:|---:|---:|
| **RegimeShift** | 7.47% | 0.815 | 14.47% |
| Static 60/40 | 7.15% | 0.755 | 23.39% |
| Equal Weight | 8.92% | 1.179 | 14.95% |

_Source: `results/submission/performance_summary.csv` (`scripts/verify_release.py` fails if this table drifts). Full methodology in the notebook._

---

## Causal Execution Timing

```
At close t:
  1. Holdings in place before close t earn the t-1 → t return
  2. Any target signalled at close t-1 is traded now; costs paid at close t
  3. Close t price becomes known
  4. Features / HMM / optimizer fitted on data ≤ t  (no future data)
  5. New target signalled — NOT filled until close t+1
  6. Filled target first earns t+1 → t+2
```

The one-bar signal-to-fill gap means no target is ever traded on the close that
produced it. `tests/test_causal_execution.py` pins this with a price jump on the
bar between signal and fill that the strategy must not earn.

`results/submission/execution_timeline.csv` records signal, execution, return,
pre-trade, target, post-trade and drifted weights for every trading day.

---

## Data Quality Fixes

| Issue | Effect before fix | Fix |
|---|---|---|
| `GOLDBEES.NS` printed 0.3355 instead of ~33.6 on 2019-12-19/20 (~100× unit error) | −34% / +53% fake strategy days; set the 35.8% headline drawdown | `data._mask_transient_spikes` masks one-day >2× moves that revert within 5 rows, then the 3-day forward-fill bridges them. A non-reverting >2× move raises `DataValidationError`. |
| `LIQUIDBEES.NS` NAV flat through 2012 → 63d equity–bond correlation undefined | 169 dates silently dropped from strategy and benchmark returns | Correlation is 0 when either leg is constant over the window |
| hmmlearn reports `converged=True` when EM hits `n_iter` | Convergence filter was a no-op; logs showed `n_iter=200` | `regime_model.em_converged` requires log-likelihood gain < `tol`; actual EM iterations are logged (median 12) |
---

## Transaction Costs and Tradability

| Scenario | One-way cost per asset |
|---|---|
| optimistic | 5 bps |
| **base (official)** | **10 bps** |
| stressed | 20 bps |

These are **assumed-cost scenarios**, not sourced measurements of spreads or
fills. Market impact and capacity are excluded (volume/ADV data absent).

`^NSEI` is an index series. `GOLDBEES.NS` and `LIQUIDBEES.NS` are ETF proxies.
An executable equity claim requires a separately verified ETF dataset with
common history, corporate-action treatment, and liquidity audit.

---

## Asset Universe

| Role | Ticker | Description |
|---|---|---|
| Equity | `^NSEI` | NIFTY 50 — **index, not directly executable** |
| Gold | `GOLDBEES.NS` | Nippon India ETF Gold Bees |
| Defensive | `LIQUIDBEES.NS` | Nippon India Liquid Bees — short-duration cash proxy |
| VIX (feature only) | `^INDIAVIX` | Optional — omitted from submitted CSV |

---

## Research Validation

The research suite (`scripts/run_research_suite.py`) produces:

1. **Ablation study**: 7 strategies, all run through the same lagged-fill event engine
   (`backtest.simulate_lagged_execution`) on identical dates and costs.

   | Variant | Net Sharpe | Max DD | Turnover/yr |
   |---|---:|---:|---:|
   | Full RegimeShift (HMM + CVXPY) | 0.81 | 14.5% | 3.2× |
   | No-regime optimizer | 0.77 | 26.3% | 1.0× |
   | Volatility rule | 0.84 | 25.6% | 0.5× |
   | HMM + fixed weights | 0.89 | 15.2% | 1.7× |
   | Minimum variance | 2.12 | 4.4% | 0.1× |
   | Static 60/40 | 0.76 | 23.4% | 0.2× |
   | Equal weight | 1.18 | 15.0% | 0.2× |

   The HMM drives the drawdown reduction (no-regime optimizer: 26% DD). The CVXPY
   layer adds turnover without improving on fixed regime weights. Minimum variance's
   Sharpe is inflated by `rf = 0` and a near-cash `LIQUIDBEES` leg (2.3% vol).
2. **Chronological subperiod analysis**: Development (2010–18), Validation
   (2019–21), Retrospective Evaluation (2022–26).
3. **Calendar-fold slices**: 5 folds (2013–15, 2016–18, 2019–21, 2022–24, 2025–26),
   which are slices of one walk-forward run, not independent re-fits. RegimeShift
   beats 60/40 in 2 folds and equal weight in 1.
4. **Paired moving-block bootstrap**: 2,000 samples, block=21, seed=42. Sharpe
   difference vs 60/40 is +0.05 (95% CI −0.45 to +0.56, not significant); vs equal
   weight it is −0.37 (95% CI −0.72 to −0.02, significant underperformance).
5. **HMM stability diagnostics**: 186 rolling fits, all converged by tolerance;
   state occupancy Bull 35% / Bear 36% / Crisis 28%.

**Disclosure:** No historical split is a pristine holdout — the full sample was
inspected during development. HMM restarts are selected by training log-likelihood
only; no performance-driven parameter changes were made after the protocol was frozen.

---

## Reproduce

```bash
pip install -e ".[dev]"

# Official experiment (base 10 bps, NEXT_CLOSE)
python run_submission.py \
    --data-path data/submission_market_data.csv \
    --cost-scenario base \
    --output-dir results/submission

# Research validation suite
python scripts/run_research_suite.py \
    --data-path data/submission_market_data.csv \
    --output-dir results/research

# Rebuild submission notebook
python scripts/build_notebook.py

# Tests
python -m pytest -q
python scripts/verify_release.py
```

---

## Repository Layout

```
src/regime_shift/      Causal engine: features, HMM, portfolio, backtest, metrics
config/                Research protocol and frozen configuration YAML
data/                  Canonical market data CSV (yfinance-sourced, CRLF-normalised)
results/submission/    Official experiment artefacts (hashed, manifest-bound)
results/research/      Research validation outputs (generated by run_research_suite.py)
notebooks/             Submission notebook (generated by build_notebook.py)
scripts/               build_notebook.py, run_research_suite.py, cost_sensitivity.py, verify_release.py
tests/                 Timing, leakage, bad-print, convergence, data, HMM, metrics, cost-model tests
config/research_protocol.yaml  Experiment protocol v2 (hypothesis, splits, cost scenarios, v2 correctness changes)
```

---

## Integrity

- `results/submission/experiment_manifest.json` binds all outputs to:
  dataset SHA-256, config hash, git commit, and result hashes.
- Static benchmarks use the **identical** event ordering, cost model, and
  date range as RegimeShift — no benchmark-period cherry-picking.
- No strategy parameters were changed after inspecting the retrospective evaluation
  period. Protocol v2 changed only correctness (fill lag, bad prints, dropped dates,
  convergence test); see `protocol_changes_v2` in `config/research_protocol.yaml`.
- `scripts/verify_release.py` re-validates manifests and output consistency.

---

## Limitations

1. **Index-level simulation** — `^NSEI` results are not executable fills.
2. **Assumed costs** — not measured; market impact excluded.
3. **Short-duration bond** — LIQUIDBEES.NS provides no rate hedge.
4. **Retrospective evaluation** — no pristine holdout exists.
5. **Three assets only** — narrow universe limits diversification.
6. **Gaussian HMM** — Student-t tails may be more appropriate.
7. **No alpha proven** — positive CAGR reflects the underlying market uptrend.
8. **Risk-free rate is 0** — Sharpe ratios of cash-heavy portfolios (minimum
   variance, Crisis allocations) are overstated; `LIQUIDBEES` `Close` also omits
   most of its distributions.

---

_IIT Bombay Summer Quant 2026 submission. Data sourced from Yahoo Finance via yfinance; subject to original provider terms._
