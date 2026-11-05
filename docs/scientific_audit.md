# Scientific audit — protocol v3

## Evidence and scope

Baseline revision: `66e9409`. Local baseline: 279 tests passed; release verification passed. GitHub access was verified with ADMIN permission and the latest main CI run succeeded. The audit request described an older state: at baseline, README net CAGR values already matched `results/submission/performance_summary.csv`, and all seven archived ablations already had 3,900 observations. Earlier sample counts must not be presented as current failures.

Historical evidence is preserved in `results/archive/protocol_v2/`, committed as `7482030`. Its files are evidence of the old methodology, not current validated outputs. No raw market data were replaced.

### Why the older README contradicted its cited CSV

Direct inspection of revision `28df753` confirms the reported discrepancy:

| Artifact at `28df753` | Static 60/40 CAGR | Equal-weight CAGR | Observations |
|---|---:|---:|---:|
| README headline | 7.20% | 8.95% | Not stated |
| Research ablation benchmark rows | 7.198% | 8.952% | 3,901 |
| Cited submission net rows | 6.60% | 8.75% | 3,732 |

The README quoted the research benchmarks' longer calendar while citing the submission
summary's shorter strategy-aligned horizon. Undefined correlation on the constant-price
bond leg removed strategy feature dates, whereas benchmark construction retained price
dates. Subsequent protocol-v2 fixes made those correlations zero, restored the missing
sessions, moved fills one session later and regenerated the headline table. The current
canonical source is the validated submission output, not either historical headline.
PR #3's description also explicitly records the old 3,732-observation horizon; PR #5
claimed identical research dates despite the unequal ablation CSV above.

## Confirmed defects and repairs

| Defect | Root cause | Repair and behavioral proof |
|---|---|---|
| Benchmark subperiod turnover and cost drag reported as zero | Research script passed benchmark net as gross and fabricated zero turnover/cost series | `StrategyRun` carries all four accounting series; `window_table` slices them together. Regression checks genuine benchmark turnover and wealth drag. |
| Unequal paired samples remained possible despite current equal counts | Per-variant `dropna`, `reindex` and `fillna(0)` plus warning-only failure handling | Shared eligibility gate and `require_common_index`; missing observations raise rather than being manufactured. |
| Early-start ablations omitted initial entry costs from the published horizon | Volatility and minimum-variance variants signalled before the HMM became eligible | All variants use RegimeShift's eligibility gate and first fill. Regression verifies initial cost and turnover for each variant. |
| Sortino denominator used only negative days | Conditional negative-return RMS substituted for target downside deviation | Denominator is `sqrt(mean(min(excess, 0)^2))` over every observation. Failing-before/passing-after regression. |
| Initial losses vanished from drawdown | Running peak began at the first post-return wealth rather than initial capital | Peak floored at 1.0 in metrics and bootstrap kernel. A first-day 20% loss must report 20% drawdown. |
| Calendar attribution described as rolling-origin evaluation | Slicing one historical online run was labelled as independent folds | Renamed calendar attribution; separate selection ledger records decision, validation and evaluation boundaries. |
| Selector inherited its first winner's holdings without paying entry | Initial selection copied a shadow portfolio's return and ordinary cost | Selector now starts in cash, earns no invested first-day return, pays entry at the first fill close. |
| Undefined statistical scores could imply zero PBO or significance | NaN CSCV ranks compare false; Holm accepted NaN; bootstrap percentile preceded draw validation | Undefined scores and p-values fail explicitly. Regressions cover constant-return candidates. |
| Bootstrap p-value centering differed from its stated procedure | Draw mean used instead of observed difference | Null-centred differences use `d_boot - d_observed`; finite Monte Carlo correction prevents reporting p=0. |
| Internal feature gaps could delete held-portfolio return sessions | Execution calendar intersected a feature frame after dropping all missing rows | Execution preparation rejects internal calendar gaps; leading warmup remains permitted. |

## Numerical interpretation

The first corrected seven-strategy run reproduced RegimeShift net CAGR 7.47%, rf=0 Sharpe 0.8145 and maximum drawdown 14.47% over 3,900 dates. These are retrospective index-level results, not executable trading evidence. Sortino changed from 0.7983 to 1.1731 when its denominator was corrected; returns did not change from that metric repair.

The initial multiple-comparison run found an unadjusted RegimeShift-minus-equal-weight Sharpe p-value around 0.038, but its Holm-adjusted p-value was around 0.616 across the 18 comparisons in that attempt. That attempt therefore does not support the old unconditional claim of significant underperformance. Current outputs include an expanded comparison family and corrected centering; consult regenerated CSVs rather than reusing these intermediate p-values.

The first rf=0 walk-forward selection picked minimum variance every year. This is a negative finding about the scoring criterion: cash-heavy portfolios dominate rf=0 Sharpe. The second, explicitly retrospective attempt scores excess return over the bond proxy. Both attempts are retained; the second is not a pristine holdout or pre-registered discovery.

## Data and causality limitations

1. `^NSEI` is an index, not a fillable security. Next-session close execution removes same-close signal lookahead conditional on the supplied price history; it does not establish instrument tradability or fill realism.
2. The bad-print detector looks ahead up to five observations to identify a reversal. Consequently the cleaned dataset is retrospectively repaired, not point-in-time available. Trailing-only model fits do not make this preprocessing causal. Results must say **causal signal/execution conditional on ex-post-cleaned data**, not end-to-end point-in-time execution.
3. A fresh Yahoo query with `auto_adjust=False` showed the canonical LIQUIDBEES values match the adjusted-close series approximately, not raw close: e.g. 2010 year-end raw close 1,000 versus adjusted close about 589.80. Recorded distributions run from 2013 through September 2025; this is not proof of complete economic total returns. Flat segments before 2013 and after the recorded distributions are material limitations.
4. Yahoo historical ETF volume contains zero values and open/close equality can be common in older spans. This is not verified liquidity. No spreads, ADV impact, fills or overnight gap performance have been invented.
5. Neither an ETF substitution nor a fully executable dataset is established by downloading a ticker. Corporate-action records and independent source verification are required before replacing the research universe.
6. DSR here is an IID/non-normality approximation conditional on the supplied correlated trial set and incomplete historical search ledger. It is not dependence-adjusted proof. CSCV is a retrospective rank-stability diagnostic, not chronological validation.
7. Every historical date was available during development. Post-dataset or post-release analysis is not automatically unseen, and prospective status requires an independently timestamped freeze before observations arrive.

## Reference decisions

- [Microsoft Qlib](https://github.com/microsoft/qlib): separate data, model, strategy, executor and experiment recorder; reuse this separation without adopting its dependencies.
- [QuantConnect LEAN](https://github.com/QuantConnect/Lean): explicit event ordering and execution/cost models; a signal timestamp is not a fill timestamp.
- [vectorbt](https://github.com/polakowo/vectorbt): composable comparisons and interactive diagnostics; fast parameter sweeps still require trial accounting.
- [Probability of Backtest Overfitting, author manuscript](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf): CSCV concerns strategy selection over a fixed trial set, not an independent chronological lockbox.
- [Deflated Sharpe Ratio](https://doi.org/10.3905/jpm.2014.40.5.094), [author manuscript](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf): distinguish per-period Sharpe units, non-normal moments, expected maximum across trials, and the independent-trial approximation.
- [Official NSE historical reports](https://www.nseindia.com/all-reports): appropriate starting point for exchange sessions, bhavcopy, volumes and corporate-action reconciliation; Yahoo downloads alone do not meet that standard.
