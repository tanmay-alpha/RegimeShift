# Verification evidence

## Baseline and regression coverage

Baseline `66e9409`: 279 tests passed, release verification passed, main GitHub CI successful. Historical mismatch evidence and confirmed defects are in [scientific_audit.md](scientific_audit.md).

The final integrated local full suite completed with **362 passed** in 106.33 seconds. All **23 forward-protocol tests** passed after the trial-ledger correction. The new chart regression also confirms both displayed wealth and drawdown retain initial losses. Notebook validation and numerical-output checks passed after genuine kernel execution. CI runs the full suite on Windows and Linux.

Falsifiable regressions cover first-day losses, target downside deviation, unequal paired indexes, genuine benchmark subperiod costs, common entry costs, absent execution sessions, undefined bootstrap/CSCV statistics, model convergence deterioration, initial selector funding, mutation of source/config/history, chronology, incremental holdings, failures and trial counts.

## Runtime and independent numerical reconciliation

| Surface | Exercised evidence |
|---|---|
| Original-universe experiment | Actual `run_submission.py` pipeline on canonical data, base assumed costs, 3,900 common dates |
| Research suite | Seven variants, chronological attribution, two explicitly retrospective selection rules, paired block sensitivity and selection-bias outputs |
| Regime diagnostics | Five common-period strategies, chronological diagnostics, momentum/volatility rule and one seeded negative control |
| Portfolio lab | All 28 prespecified strategy/cost paths completed; zero failed trials |
| Stored-result reconciliation | Independent NumPy calculation from saved daily accounting reproduced CAGR/Sharpe with maximum difference 1.83e-14; all net-return identities passed |
| ERC invariant | Maximum recorded risk-budget residual 3.95e-5, below the 0.002 acceptance tolerance |
| Standalone CLI | Explicit preparation, JSON-config run, inspection, checksum-verified ZIP export; 27 generated files checked with no source-price file in the ZIP |
| Browser interface | Actual Chromium view of same-period metrics and seven plots; drawdown tab; configuration form; successful registered experiment submission |
| Forward CLI | Historical freeze/evaluate replay from local prepared data; `historical`, `prospective_certified: false` |
| Failure handling | Intentional missing dataset exited nonzero and appended failed attempt, zero started trials and null input hash |
| Notebook | `nbconvert --execute` completed; embedded numerical results agree with validated submission outputs |
| Packaging | Python compilation, new research-code Ruff checks and wheel build passed |
| Artifact integrity | Release verifier checks canonical input hash, output hashes, headline metrics, common horizons, execution chronology and benchmark accounting |

## Scope of verification

The original repository had 139 Ruff findings outside the new research surface. CI lint is explicitly scoped to the laboratory code rather than silently claiming a full legacy lint cleanup. No full-repository static type-check success is claimed. Existing numerical tests and compilation cover the legacy engine.

The optional UI was exercised in an isolated environment because the workstation's existing FastAPI installation pinned an older Starlette incompatible with the installed Streamlit. No unrelated FastAPI upgrade was required. The notebook completed with Windows kernel transport/event-loop warnings, not execution errors.

A solver occasionally reported `optimal_inaccurate` during ERC solves; solutions were not accepted solely on solver status. Weight and risk-contribution residual checks passed. This does not establish performance robustness or instrument tradability.

No deployed service, independent prospective timestamp authority, verified instrument-level total-return dataset, measured liquidity or live trading was exercised or claimed. Historical repair remains ex-post. Current manifests identify their actual generating source revision and whether source inputs were dirty; intermediate attempts are retained rather than relabelled as clean final runs.
