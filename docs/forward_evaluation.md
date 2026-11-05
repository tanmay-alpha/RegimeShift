# Offline frozen forward evaluation

This is an **index-level research protocol**, not live trading, an execution simulator calibrated to broker fills, or a certificate that an experiment was genuinely prospective. It never downloads data. Both commands accept only user-supplied local CSV files, and write hashes and derived accounting, not copies of market prices.

## Freeze before evaluating

From the repository root, with the canonical CSV unchanged:

```console
python scripts/forward_evaluation.py freeze --experiment-dir results/forward/frozen_001 --dataset data/submission_market_data.csv --cutoff 2026-07-27
```

The cutoff above is the last row of the checked-in canonical dataset. If the dataset changes, use its actual final trading date and a **new experiment ID**. The historical input must end exactly at cutoff: a file containing later rows is rejected, not silently truncated. For a historical rehearsal, supply a separately prepared historical-prefix CSV and disclose that it is a rehearsal; do not change the canonical CSV in place.

Freeze is a distinct command. Evaluation cannot create a freeze on demand. A pre-existing experiment directory, including an incomplete or failed freeze, is never reused. The returned `freeze_sha256` should be recorded independently before new data is collected. Keep all original input files available at their recorded absolute paths.

By default, the frozen configuration is `RegimeShiftConfig()` with its effective execution-cost model materialized. The base cost scenario means **10 bps one way per unit of traded weight**, not the older `transaction_cost_bps` compatibility field. `--config path/to/config.json` accepts a **complete** dataclass-shaped configuration, including nested fields, rather than partial overrides. Configuration-file parsing and validation occur inside the registered freeze attempt. A configuration can be prepared using `dataclasses.asdict(RegimeShiftConfig())` and JSON serialization with enum values. Cost assumptions, HMM seeds/restarts, training window, minimum history, rebalance frequency, asset identifiers and every other dataclass field are frozen. The feature and portfolio settings derived by configuration properties are additionally bound by the code-source hashes; they are not separately adjustable at evaluation.

`--observed-at` optionally supplies a timezone-aware observation timestamp; otherwise the local UTC clock is used. `--release-at` optionally records a declared release timestamp. **Neither is an independent timestamp authority.** A timestamp is an assertion about when the input was observed, not an exchange close timestamp or a proof of when a researcher first saw the data.

## Append new observations

Prepare a CSV containing **only new dates**, with the same columns in the same order as the frozen history. Its first date must be strictly after cutoff and, for later batches, strictly after the last accepted date. Do not resend the historical prefix or an earlier evaluated batch.

```console
python scripts/forward_evaluation.py evaluate --experiment-dir results/forward/frozen_001 --dataset /private/new_observations_batch_001.csv
python scripts/forward_evaluation.py evaluate --experiment-dir results/forward/frozen_001 --dataset /private/new_observations_batch_002.csv
```

CSV format is a first-column trading-date index followed by `equity,gold,bond`, optionally the frozen VIX column. Asset names follow the frozen configuration. Dates must be unique, sorted, timezone-naive midnight dates. Prices must be finite and positive. Unknown columns, missing cells, intraday dates and schema changes are rejected. No sorting, duplicate removal, forward filling, spike masking, or download fallback occurs. This intentionally does **not** invoke the legacy cleaning CSV loader, whose normalization could hide chronology or historical revisions.

Dates cannot exceed the declared observation date in Asia/Kolkata. Observation timestamps must strictly increase beyond the freeze and **all earlier timestamped attempts, including failures**. A backdated attempt is rejected and still logged; choose a later timestamp for another attempt. Supplying an erroneously far-future observation timestamp can consequently prevent later attempts until that date. A lock serializes evaluation of an experiment; a lock collision is a failed, separately recorded attempt, not an overwrite. An interrupted process may leave a lock. Preserve the directory and audit it before manually addressing such a lock; the command does not automatically erase evidence or recover an unregistered completed result.

## What is frozen and checked

`freeze.json`, its digest sidecar, and the completed freeze's registry anchor bind:

- SHA-256 of every Python file under `src/regime_shift`, the forward command, and `pyproject.toml`/`requirements.txt` when present. Added, removed or changed files invalidate the freeze. Git commit identity alone is not used as a substitute for source hashes.
- The full serialized configuration, the research protocol version, and the explicit rolling-training, signal, fill, baseline and no-selection policy.
- The cutoff, observation timestamp, optional declared release timestamp, and explicit unverified-clock status.
- The canonical CSV hash using the shared helper's CRLF-to-LF normalization, the historical file path, schema and per-date row digests. All other byte-level input rewrites are rejected, even if numerical values appear equivalent.
- The shared environment record: Python and recorded dependency versions and platform.

Each accepted incremental file is similarly hashed and retained by reference. Before another batch, all historical and previously accepted inputs, prior request/failure records with registered hashes, accepted result manifests, and accounting-output hashes are checked. Missing or rewritten history cannot be silently substituted. Inputs and frozen code are rechecked after execution before accepting a result. This is not a transactional snapshot against a malicious process rewriting files between individual checks.

**Editing code, dependencies, protocol or configuration requires a new experiment and freeze.** Do not edit an existing manifest, repair prices in place, or replace prior accepted input files. The original files are needed because incremental evaluation replays from the frozen entry using the complete accepted prefix. No raw price snapshot is published or embedded in the output directory. Moving/deleting an input therefore makes the experiment unreplayable rather than causing a download or silent replacement.

The environment check records versions, not third-party wheel hashes, native solver binaries, BLAS implementation details or the operating system's full state. It cannot guarantee bitwise-identical numerical optimization across installations. The source set does not include notebooks, tests, documentation or arbitrary external scripts; those are not used by this command. No arbitrary user strategy plugins are supported.

## Training, entry and incremental accounting

At the **cutoff close**, all three strategies start from cash and RegimeShift forms its first target from historical features through that close only. The prefix must already be long enough for feature warmup, the HMM training minimum and covariance estimation. There is no historical portfolio whose carried holdings or entry costs are hidden outside the measured window.

The first target fills at the **next supplied close**, pays its initial allocation cost there, and first earns a return on the following close-to-close interval. On that first fill date, gross return is zero, initial turnover is 1, and net return includes the entry cost. The static 60/40 and equal-weight baselines have the identical first fill, cost assumptions, observation index and subsequent RegimeShift fill dates. Paired samples use the shared strict alignment helper; there is no filling of return dates.

Subsequent signals occur every frozen `rebalance_frequency` observed sessions after cutoff. At each signal, the feature scaler and HMM are refitted on the trailing `train_window` observations ending **at that signal close**, and the optimizer estimates from the correspondingly trailing available returns. Earlier accepted evaluation observations become legitimate training history for later signals; future observations never do. Parameters and the training rule remain fixed; fitted model parameters evolve. There is no tuning, candidate selection or favorable-window search inside this protocol.

Each later batch replays that continuous cash-entry experiment from the original cutoff and reports **only the new batch's dates**. Holdings are retained across batches. It does not reset wealth, charge a second initial allocation, or drop the first return of a later batch. A target formed at the preceding batch's last close can fill on the new batch's first close. Reported batch metrics describe that batch, not a newly initialized portfolio or automatically compounded aggregate report. Shared gross/net/cost/turnover definitions apply; net is `(1 + gross) * (1 - cost) - 1`.

## Artifacts and attempts

For the example experiment, outputs are:

```text
results/forward/frozen_001/freeze.json
results/forward/frozen_001/freeze.sha256.json
results/forward/frozen_001/attempts/<unique_attempt_id>/request.json
results/forward/frozen_001/attempts/<unique_attempt_id>/result.json
results/forward/frozen_001/attempts/<unique_attempt_id>/summary.csv
results/forward/frozen_001/attempts/<unique_attempt_id>/daily_accounting.csv
results/forward/frozen_001/attempts/<unique_attempt_id>/failure.json
results/forward/registry/frozen_001.jsonl
```

Successful attempts have `result.json` and the CSVs; failed attempts have `failure.json`, and may retain incomplete outputs for audit. Requests are saved when the observation timestamp parses, before integrity validation. Invalid timestamps and lock collisions still get a unique failure artifact and registry line. Registry uses **one append-only JSONL per experiment**, including freeze, rejected repeated freezes, completed evaluations and failed evaluations. Freeze has zero trials; evaluations count each strategy path as it starts (one HMM configuration plus two fixed baselines on completion). Rejected pre-model inputs count zero, and partial failures retain the attempted strategy IDs. A missing dataset is recorded with a null hash and its requested path.

Files are created exclusively and attempt IDs are never reused. A completed result that exists without its registry entry blocks further evaluation instead of silently evaluating the same period again. All attempt directories, including interrupted ones, must be retained. Calls that cannot identify an existing experiment (evaluation before freeze), cannot create their destination, or cannot write to the registry cannot promise a durable failure log.

## Historical, post-release and genuinely prospective are different

The result's descriptive `evaluation_type` is:

- **`historical`** when the first new trading date is on or before the freeze-observation trading date or, if supplied, the declared release trading date. A mixed batch crossing that boundary is conservatively classified historical as a whole.
- **`post_release_unverified`** when all new dates are later than both declared boundaries. Without `--release-at`, this means after the local freeze assertion only; it is not evidence of a public release.

Every result sets **`prospective_certified: false`** and `timestamp_authority: unverified_local_clock`. There is deliberately no flag that turns a user-entered timestamp into a true-prospective certificate. True prospective evidence would additionally require an independently witnessed pre-data freeze commitment, trustworthy first-availability/vintage provenance for each observation, and a complete independently preserved attempt history. Those prerequisites are not provided or verified by this offline tool. A post-release historical file supplied today remains retrospective evidence unless its genuine availability history is established outside this command.

Local immutability means exclusive creation, append-only command behavior and tamper detection against preserved digest/registry anchors. A filesystem owner can rewrite **all** local artifacts and the registry together, delete attempts, change their clock, or fabricate an input CSV; no local Python command can make that impossible. Independently preserve the freeze digest and attempt history to defend against that adversary. The registry itself is not a signed transparency log.

The bundled source CSV contains historical bad prints; the separate legacy preparation step repairs them ex post. Neither source nor prepared history establishes a point-in-time vintage. `^NSEI` is not tradable. Hashes establish identity, not economic truth or executable prices. There is no exchange calendar/volume feed: omitted sessions or falsely dated prices cannot be detected merely from a sorted local CSV. Costs are recorded explicit assumptions (the core default is 5 bps unless configured; use 10 bps explicitly for the official base study). No raw data are published and no orders are placed.
