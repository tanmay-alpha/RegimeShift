# Regime intelligence v1

## Hypothesis and scope

Exploratory hypothesis: a three-state HMM may provide interpretable descriptions
of current market conditions and useful allocation timing beyond a simple
momentum/volatility rule. High posterior uncertainty or unstable refit partitions
may identify occasions when that description deserves less confidence. This
slice measures those properties and compares five fixed allocation rules. It
does not assume either hypothesis is true, and reports no metrics until an
experiment actually runs.

**Posterior probabilities describe latent states at a signal close. They are
not probabilities of a future crash, future return, or future drawdown.** The
names Bull, Bear and Crisis are training-feature interpretations, not observed
ground-truth market outcomes. No deep learning, external API, live download,
parameter search, threshold optimization, strategy selection, or alert-driven
trading is implemented here.

## Exact experiment boundaries

- Input is one existing local market-data CSV, loaded by the repository's
  `load_market_data_csv`. Relative paths resolve against this worktree, without
  the loader's parent-directory fallback. VIX is omitted to match the existing
  research suite. Data cleaning diagnostics are included in the manifest.
- The CLI uses `RegimeShiftConfig` defaults: rolling training window 252 usable
  feature rows, minimum training history 126 rows, rebalance every 21 rows,
  three diagonal Gaussian states, seeds 42/43/44, up to 200 EM iterations per
  restart, tolerance 0.001. The existing fitter chooses converged candidates by
  training likelihood; if none converges it uses the best finite likelihood.
  Convergence, warning and restart records are retained; nonconverged fits are
  not silently removed or reported as convergence.
- At most 6,000 cleaned price rows and 256 eligible signal closes (including a
  final unfilled signal) are permitted. Exceeding a bound fails before running
  a strategy; data are not silently truncated. There is one fixed five-strategy
  trial set per invocation. The CLI exposes only the input path and one cost
  scenario, with no search loop.
- All five rules run through `run_target_strategy`. Eligibility is the same
  maximum of HMM and portfolio minimum-history requirements used in the research
  layer, even for static rules that need no training. Every rule supplies a
  target at every eligible scheduled signal close. `require_common_index`
  rejects differing samples; comparison returns are never filled or reindexed
  onto invented observations.
- At signal close t, the scaler, HMM and features see data through t only.
  The target fills at the next **observed trading close** t+1 and first earns
  the return from t+1 to t+2. The first fill starts the accounting sample and
  includes the initial allocation from cash and its cost. A terminal signal
  with no following close is retained in diagnostics and marked unfilled.
- Base costs are the shared 10 bps one-way per asset assumptions. Optional
  `optimistic` and `stressed` runs use 5 and 20 bps respectively. The chosen
  scenario applies identically to all rules. The engine books
  `net = (1 + gross) * (1 - cost) - 1`, with costs at fill closes, drifted holdings
  and genuine turnover. Initial turnover is 1; subsequent one-way turnover is
  half the absolute change from drifted weights. Costs apply to the full
  absolute traded weight, following the shared specification.

The five frozen rules are:

| Strategy ID | Decision at each eligible signal close |
| --- | --- |
| `hmm_fixed` | Existing train-only HMM regime, using the existing fixed regime weight table; the same fit supplies diagnostics. |
| `momentum_volatility` | Last 63 simple equity returns: momentum = product(1+r)-1, volatility = sample standard deviation (ddof=1) times sqrt(252). Crisis if volatility > 30%; otherwise Bear if volatility > 20% or momentum <= 0; otherwise Bull. Same fixed regime weights as HMM. |
| `negative_control` | One IID uniform Bull/Bear/Crisis assignment per eligible signal, in chronological order, using a private NumPy PCG64 generator with predeclared seed **1729**. Same fixed regime weights. No shuffling of future HMM labels or occupancy fitting. |
| `static_6040` | 60% equity, 0% gold, 40% bond on the same signal/fill schedule. |
| `equal_weight` | One third each of equity, gold and bond on that schedule. |

Fixed weights in equity/gold/bond order are Bull 0.65/0.20/0.15,
Bear 0.35/0.30/0.35, Crisis 0.10/0.30/0.60. This isolates the assignment rule
from optimization. The full RegimeShift CVXPY optimizer is outside this slice.
The negative control is reproducible and prefix-stable for a fixed schedule;
it does not preserve HMM occupancy or persistence. One random realization is a
sanity check, not a null distribution or a significance test. Do not rerun seeds
and select a favorable one.

## Diagnostics and limitations

`diagnostics.csv` contains one row per actual chronological signal close,
including the exact training boundaries, current posterior vector, and fit
status. Posterior rows are not carried forward across days. The final row of
`predict_current_state` is used; training-sequence smoothed posterior rows are
not presented as historical as-of predictions.

- **Normalized entropy:** `-sum(p_i log(p_i))/log(3)`, with zero contributions
  handled exactly. Zero means concentrated; one means uniform. Invalid
  probability vectors fail instead of being silently normalized. Exploratory
  uncertainty alerts use **entropy >= 0.80 OR maximum posterior <= 0.60**.
  These thresholds are explicit, uncalibrated, and fixed before the run. The
  two component alerts and combined alert are saved. They measure model
  ambiguity, not calibrated forecast risk, and do not change portfolio weights.
- **Transition persistence:** the labeled fitted transition matrix and
  `1/(1-p_ii)` expected spell duration are saved for every state and fit. Duration
  is in **training feature observations**, approximately daily trading closes,
  under a fixed homogeneous Markov model. `p_ii=1` produces `inf`, with an
  explicit absorbing flag. These are in-window fitted quantities, not reliable
  forward holding-period forecasts, and refitting changes the model.
- **Empirical spells:** the latest signal label is carried across observed
  trading closes until the next switching signal. Each spell includes its
  starting close and excludes the next switching close. The final spell extends
  through the last available data close and is always **right-censored**; its
  observed length is a lower bound, even if the last signal itself is unfilled.
  The first spell is left-censored because its true onset is unknown. Durations
  count observed closes, not calendar days. Counts of sampled signal closes are
  also saved. This measures persistence of online, refitted, signal-sampled
  labels; it cannot detect within-rebalance switching and should not be equated
  with geometric durations from the daily transition matrix. Completed spells
  become identifiable only at their later switch; this retrospective table is
  never an input to trading. No uncensored duration mean or survival estimator
  is claimed.
- **Rolling refit stability:** adjusted Rand index compares the two fits'
  decoded training partitions **only on intersecting training dates**. Original
  sequence contexts are retained, and the index is invariant to arbitrary
  numeric state permutations. No numeric state-ID equality score is used.
  Boundaries, overlap size and state counts are saved. Less than two overlapping
  rows or a constant partition yields an undefined score with an explicit
  status; it does not manufacture perfect agreement. The first fit has no
  overlap. High agreement does not establish stable economic meaning, accurate
  forecasts, or out-of-sample skill. The later fit can revise earlier training
  assignments using data available at that later signal close.
- **Feature means:** raw per-state training feature means and occupancy counts
  use the existing decoded training statistics. State IDs are scoped to each
  fit. Unvisited states retain missing means and zero counts. These explain
  what observations the model grouped together; they are **not causal feature
  attribution, feature importance, marginal effects, or predictive evidence**.

All causality statements concern fitting/inference and trading on the supplied
offline dataset. The shared CSV loader detects transient spikes using subsequent
reversions and can revise past observations during preprocessing. This slice
inherits that cleaning policy; it does not establish point-in-time data
availability. Missing dates and source-price quality remain inherited concerns.

## Outputs and provenance

Each invocation creates a fresh directory under
`results/experiments/regime_intelligence_v1/<UTC-attempt-id>/`; existing attempts
are never overwritten. Outputs are:

- `attempt.json`: fixed specification, attempted strategy IDs, status and bounds.
- `strategy_metrics.csv`, `same_index_comparisons.csv`: shared metric definitions
  and descriptive differences versus HMM fixed, static 60/40 and equal weight.
  All use the same continuous full-sample index. Differences are strategy minus
  reference; a positive maximum-drawdown difference means a larger loss.
- `daily_accounting.csv`, `net_returns.csv`: genuine daily gross/net returns,
  costs, turnover and end-of-day post-trade holdings; aligned net-return matrix.
- `signals.csv`: targets, regime labels, signal/fill closes and filled flags
  for all rules, allowing schedule and last-signal auditing.
- `diagnostics.csv`, `transitions.csv`, `persistence.csv`, `stability.csv`,
  `feature_means.csv`, `spells.csv`, `restarts.csv`: the diagnostics above and
  the existing fitter's individual restart records.
- `manifest.json`: written by `write_experiment_manifest` after outputs exist;
  records source commit/dirty status, dataset hash, configuration, environment,
  output hashes and hashes of this slice's Python sources.

`register_attempt` appends completion/failure records to
`results/registry/regime_intelligence_v1.jsonl`. `n_trials` counts strategy runs
actually started (maximum five), not EM restarts or a claimed search budget.
Partial output from a failure is explicitly unsuccessful: `failure.json`
contains the error and started rules, and no successful manifest is retained.
Missing/unreadable inputs cannot be hashed by the shared registry helper; the
CLI retains a failed record with a null dataset hash and the registration error
instead of inventing a hash. A write-permission failure can prevent recording;
the CLI reports that explicitly to stderr and exits nonzero. Dependencies must
be installed for the entry point to import. Invalid CLI syntax exits through
argparse before an experiment attempt starts.

There are no inferential p-values, bootstrap intervals, holdout claims, or
performance-selected parameters. The historical sample was already inspected
by the broader project, so this is exploratory research. The inherited
index-level equity simulation, cash-proxy quality and assumed execution costs
still limit investability claims. Metrics over the cash proxy use the shared
bond-leg definition; they do not repair incomplete historical distributions.

## Run and acceptance

From this worktree in PowerShell, use the local source tree explicitly because
an editable installation may point at the parent checkout:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts/run_regime_intelligence.py --data-path data/submission_market_data.csv --cost-scenario base
```

For the parent verification pass, with the same `PYTHONPATH`:

```powershell
python -m pytest tests/test_regime_intelligence.py -q
```

Behavioral acceptance covers exact-zero/uniform/invalid posteriors, alert
boundaries, absorbing transitions, overlap-only permutation-invariant stability,
degenerate overlap, censoring, baseline precedence, reproducible controls,
identical gating/index/initial costs, next-close execution, terminal unfilled
signals, equivalence to the existing HMM-fixed target, future-mutation isolation,
experiment bounds and failure registration. Tests use small synthetic fixtures
and stub fitting to verify mechanics; they are not evidence of real HMM
convergence or investment performance. Test/build/lint/experiment execution is
left to the parent validation pass; no research results are claimed here.
