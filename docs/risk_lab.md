# Portfolio risk and stress lab v1

Hypothesis: train-only covariance allocation and explicit risk control may change
drawdown, concentration and trading costs relative to sample minimum variance,
static 60/40 and equal weight. This is a bounded descriptive experiment, with four
new variants and three baselines. It does not select a winner or tune parameters
against full-sample returns. No numerical performance claim is made here.

The implementation uses `research.spec` metric definitions, including annualization
252 and both zero-risk-free and cash-proxy Sharpe. `research.evaluation` enforces
identical daily indexes. All seven strategies use `run_target_strategy`, the same
eligibility gate and scheduled signals, and the existing NEXT_CLOSE engine.
At close t holdings earn their return before a pending trade fills; targets using
data through t fill at t+1. All strategies start from cash on the same fill date,
and pay initial allocation costs. No benchmark is restarted inside a crisis window.

## Frozen definitions

All covariance estimation uses the trailing available 252 return rows, subject to
the shared engine's minimum history (126 feature-aligned rows by default). Early
windows can therefore have fewer than 252 observations. Sample covariance receives
the existing positive diagonal ridge of 1e-6 in daily variance units. There is no
expected-return forecast. Signals rebalance every 21 rows on the existing engine
schedule. Four new variants are:

- Equal risk contribution: solve the convex positive-variable risk budget problem
  `min 0.5 x' C x - sum(log(x))/3`, then normalize x. Scaling the covariance for
  numerical conditioning preserves normalized weights. Accept only finite positive
  weights whose Euler variance contributions `w_i (Cw)_i / (w'Cw)` differ from 1/3
  by at most 0.002. Failed validation tries the next solver, then raises. ERC has
  natural long-only bounds [0,1]; imposing the 80% cap could prevent exact parity,
  so that cap applies to minimum variance variants, not ERC.
- Ledoit-Wolf minimum variance: estimate shrinkage using only the trailing window,
  add the stated ridge, minimize variance subject to long-only weights summing to
  one and each weight at most 80%. Validate the solved weights and cap.
- Turnover-penalized minimum variance: sample covariance plus ridge; minimize
  `252 w'Cw + 0.01 ||w - previous||_1` under the same 80% cap. Previous means actual
  drifted holdings at the signal close, not the last target. There is no penalty
  before initial investment. This is a regularizer in annual variance units,
  not a transaction-cost estimate or a guarantee of lower realized turnover.
- Volatility-targeted equal weight: blend the full three-asset equal-weight book
  with residual bonds: `alpha * equal_weight + (1-alpha) * bond`. Choose the largest
  alpha in [0,1] with estimated annual volatility at most 10%, using the full blend
  covariance, including the bond leg. Never lever or short. If no feasible blend
  exists, fail explicitly. The target is an estimate, not a realized volatility cap.

Baselines are the existing sample minimum variance target (80% cap), 60% equity /
40% bond, and one-third each. Static strategies are periodically rebalanced through
the same target engine. The minimum variance baseline retains its shared solver
implementation; the new solvers validate their own results without changing it.

## Reproduce offline

From this worktree root in PowerShell, with project dependencies installed:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python scripts/run_risk_lab.py
python -m pytest tests/test_risk_lab.py -q
```

The script prioritizes this worktree's `src` and reads only the canonical local
`data/submission_market_data.csv` via the existing validator. It never downloads.
Canonical loader corrections and short-gap policy remain unchanged and their data
diagnostics are preserved. No spread observations, open prices, gaps or volume
series are manufactured. Optional VIX is excluded to match the canonical study.

Default output is `results/experiments/risk_lab_v1`. An occupied output directory
is rejected to prevent stale files entering a manifest; use a fresh `--output-dir`
for another attempt. The runner appends every attempt to
`results/registry/risk_lab_v1.jsonl`, including setup failures and partial failures.
There are 28 planned strategy/cost trials: seven definitions times four assumed
cost scenarios (5/10/20/50 bps one-way). Records distinguish planned, attempted,
completed and failed trials. These are seven allocation definitions, not 28
independent hypotheses. Rolling solves are estimation steps, not parameter trials.
Repeated invocations remain visible in the append-only registry.

Outputs:

- `daily_accounting.csv`: gross/net returns, costs and turnover per strategy/date/cost.
- `weights.csv`: end-of-day post-trade weights, including daily drift on nonfill days.
- `trade_deltas.csv`: signed target-minus-drifted-holdings trades, initial cash included.
  Actual engine costs must reconcile to `sum(abs(delta)) * bps / 10000`. Initial
  turnover is 1; subsequent turnover is half the absolute traded weight sum.
- `metrics.csv`: shared metric definitions and cash-relative Sharpe, common dates.
- `risk_contributions.csv`: ERC shares against each signal's estimation covariance;
  a final signal without a following close is identified and never treated as a fill.
- `historical_replay.csv`: calendar slices of the same continuous net-accounting runs
  for July-December 2011, February-April 2020 and calendar 2022. Actual observed
  dates are reported; unavailable windows are labelled rather than filled.
- `hypothetical_stress.csv`: terminal fixed-holdings diagnostics, explicitly separate
  from historical replay. Reference, correlation, concentration and combined cases.
- `trials.json` and `manifest.json`: failures, trial counts, configuration, source
  hashes for the two new executable files, canonical dataset hash, environment,
  git commit/dirty state and hashes of output files. A missing dataset cannot be
  hashed: its failure registry record uses a null hash; manifest creation fails visibly.

Cost scenarios rerun genuine engine accounting without changing the strategy
definitions. Uniform portfolio-level cost deductions do not change relative
holdings, so targets and gross returns should agree across costs. Net returns use
the engine identity `(1+gross)*(1-cost)-1`, not simple subtraction.

## Stress interpretation and limitations

Hypothetical diagnostics freeze final realized holdings and use trailing sample
covariance plus ridge at the snapshot date. The correlation case blends the original
correlation matrix halfway toward equicorrelation 0.90, preserving marginal
variances and positive semidefiniteness. This can reduce some already higher pairwise
correlations; it is a stated scenario, not a universal worst case. The concentration
case moves 25% of the whole book into its largest holding, retaining 75% of the
original weights. Ties use asset order. Concentration scenarios may exceed allocation
caps deliberately. Assumed single-period asset returns are equity -20%, gold -10%,
bond -2%. The table reports their weighted portfolio return, annual covariance
volatility, HHI and maximum weight. These are transparent hypothetical assumptions,
not observed shocks, stress probabilities, forecasts or synthetic historical paths.
No trading costs are inferred for the hypothetical concentration transfer.

Historical attribution is retrospective and carries positions formed before the
window. It is not an independent refit, holdout or counterfactual event replay.
The full canonical sample has already been inspected in this repository; there is
no pristine holdout. No full-sample parameter search is performed by this lab.
The three-asset universe is narrow; NIFTY is an index, not an executable security.
LIQUIDBEES is a low-volatility cash proxy, not a duration hedge; its price history
has distribution limitations and may inflate zero-risk-free Sharpe. Costs are
assumptions, not measured spreads or fills; capacity, market impact and execution
gaps remain unmodeled. Ridge and shrinkage can materially affect a near-cash asset's
risk budget. Neither solver success nor passing behavioral tests establishes
investment value or production readiness.

Acceptance requires the parent to run the focused tests and canonical experiment,
inspect all 28 trial outcomes, verify common dates, initial costs, contribution
tolerances and manifests. This implementation handoff does not claim those commands
have passed; no experiment results are invented.
