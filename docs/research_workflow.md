# Independent offline researcher workflow

The command line and optional Streamlit/Plotly inspector both call
`regime_shift.research.workflow`. Neither interface contains a second backtest
engine. There is no vendor download, brokerage connection, order submission, or
live-trading mode. The CLI does not import Streamlit or Plotly.

## Configure a bounded experiment

Save the following as a JSON file, for example `experiment.json`:

```json
{
  "schema_version": 1,
  "description": "Compare the HMM allocation with frozen ablations and references",
  "dataset_path": "data/cache/validated_market_data.csv",
  "strategies": [
    "regimeshift", "no_regime_optimizer", "vol_rule", "hmm_fixed",
    "min_variance", "static_6040", "equal_weight"
  ],
  "evaluation": {
    "start": "2012-01-01",
    "end": "2025-12-31"
  },
  "engine": {
    "train_window": 252,
    "minimum_training_observations": 126,
    "rebalance_frequency": 21,
    "transaction_cost_bps": 10.0,
    "random_seed": 42,
    "hmm_n_iter": 200,
    "hmm_n_restarts": 3
  }
}
```

Only JSON data is accepted. Unknown fields, duplicate/unknown strategy IDs,
nonfinite costs, invalid dates, remote dataset URLs, and out-of-range settings
are rejected. Configurations cannot specify Python code, imports, expressions,
custom strategy callables, solver plug-ins, or an execution mode. A relative
CSV path is relative to the process working directory, **not** the JSON file.
Use an absolute local CSV path when running from elsewhere. The interfaces are
local research tools, not a multi-tenant filesystem sandbox.

The CSV must have a date column first, followed by `equity`, `gold`, `bond`, and
optionally `vix`. Dates must already be sorted, unique, daily, timezone-naive,
and nonmissing. Values must be numeric, positive, finite, complete, and pass the
shared price-data checks. The workflow does **not** silently repair prices,
sort dates, fill missing values, remove spikes, or intersect incompatible
strategy samples. Prepare historical source data explicitly before an experiment:

```console
python -m regime_shift.cli prepare-data --input data/submission_market_data.csv --output data/cache/validated_market_data.csv
```

Preparation uses the legacy retrospective spike/reversal detector and bounded fill,
and saves source/output hashes plus diagnostics beside the prepared CSV. It refuses
to overwrite existing outputs. This is not a point-in-time correction. The original
canonical source contains the bad prints; it is preserved unchanged. Raw or prepared
market data are not included in reproducibility ZIP exports.

Evaluation start/end are explicit inclusive calendar bounds within the dataset
span; weekends are permitted. Earlier dataset rows supply warmup and causal
training. Rows after the evaluation end are excluded before feature creation
and engine execution. Start must be on or after the first eligible fill; the
runner raises rather than silently shortening the requested evaluation span.
At least two common accounting observations are required.

The allowed computational envelope is 20,000 dataset rows, training windows of
60–2,520 rows, minimum training histories of 50–2,520 rows (no larger than the
training window), signal intervals of 1–252 rows, at most 2,000 scheduled
reference fits, 1–1,000 HMM iterations and 1–10 restarts. A zero-to-1,000 bps cost
range supports explicit sensitivity assumptions; 10 bps is the default, not a
calibrated estimate. The seed is a nonnegative 32-bit integer.

## Run, inspect, export

After installing the package and core research dependencies:

```console
python -m regime_shift.cli run --config experiment.json
python -m regime_shift.cli inspect
python -m regime_shift.cli inspect results/experiments/EXPERIMENT_ID
python -m regime_shift.cli export results/experiments/EXPERIMENT_ID --output reproduction.zip
```

`run` accepts `--results-dir` and `--registry-dir`; `inspect` without an experiment
accepts `--results-dir`. Run output identifies the versioned directory. A failed
attempt exits with status 1 and identifies its saved failure directory.
`inspect` and `export` read existing artifacts and never fit a model.

Python callers use the same functions:

```python
from regime_shift.research.workflow import (
    run_experiment, inspect_experiment, export_experiment_zip,
)

result = run_experiment("experiment.json")
manifest = inspect_experiment(result.output_dir)
export_experiment_zip(result.output_dir, "reproduction.zip")
```

The existing core suite always computes RegimeShift and both static references
in order to preserve its official common execution calendar. Only requested
ablations are evaluated, and only requested strategy outputs are published.
Thus requesting `vol_rule` alone still computes the reference HMM. This is
explicit in metadata and the attempt's computed-strategy count; hiding a
reference is not evidence it was never evaluated. The workflow does not rank,
optimize a parameter grid, select a winner, or claim a validation/test split.
For example, changing `train_window` changes the shared core estimation window,
including the minimum-variance ablation's covariance window.

## Versioned artifacts and failure accounting

Each call receives a unique protocol-versioned directory under
`results/experiments/`, never overwriting an earlier run. The runner uses
`write_experiment_manifest` and `register_attempt` from the shared registry.
The per-experiment JSONL registry remains append-only. Registration captures
failed configuration, dataset, alignment, or engine attempts as well as completed
runs; unavailable dataset hashes are null rather than invented. Engine-attempt
counts include hidden computed references. A failure before engine invocation
has zero trials; an engine-stage failure reports the requested computed suite
count, not a claim that every variant completed.

A successful directory contains:

| Artifact | Meaning |
| --- | --- |
| `config.json` | Validated configuration with explicit defaults |
| `manifest.json` | Protocol/workflow versions, dataset SHA-256, config, git commit/dirty flag, environment, output checksums, status and evaluation summary |
| `attempt.json` | Snapshot of the append-only registry record |
| `metrics.csv`, `metrics.json` | Same-period descriptive metrics; undefined JSON metrics are null |
| `metadata.json` | Units, execution timing, provenance caveats, computed/published strategy IDs, engine metadata |
| `regime_probabilities.csv` | Latest reference HMM Bull/Bear/Crisis probabilities at each accounting close, interpreted regime, entropy and maximum probability |
| `diagnostics.json` | Reference HMM convergence, restart diagnostics, signal and fill dates |
| `reference_events.csv` | Reference event log for the displayed accounting dates |
| `STRATEGY/accounting.csv` | Gross/net daily returns, paid costs, turnover, gross/net wealth, signed drawdown and sum of cost fractions |
| `STRATEGY/weights.csv` | Actual end-of-day post-trade accounting weights, including drift |
| `STRATEGY/explanations.csv` | Strategy rule, held-target signal date/regime where applicable, weight/probability semantics |

Failures retain `error.json`, the attempted config, `attempt.json`, a failed
manifest, and any partial generated artifacts. A manifest with failed status is
not a completed comparison. Unreadable or malformed config files may have an
empty attempted-config snapshot; the error records the reason.

The ZIP is a reproducibility **artifact bundle**, not a self-contained dataset
or software distribution. It includes allowlisted generated artifacts and the
manifest; it excludes raw market prices and unrelated files even if placed in
the experiment directory. Output checksums are verified before export;
modified outputs are rejected. Keep the original dataset separately and obtain
it through the project's normal data-provenance process; its manifest hash
identifies the required input. The ZIP does not bundle source code or a full
environment lock, and matching a seed does not guarantee bitwise results across
numerical library versions. ZIP output must be outside the immutable experiment
directory.

## Inspect-first optional interface

Install the optional UI dependencies supplied by the project, then launch:

```console
streamlit run research_app.py
```

The default screen lists saved manifests and inspects existing results. With no
saved results it shows instructions rather than starting computation. An
explicit form accepts a local dataset path, optional uploaded JSON config,
model/allocation variants, inclusive evaluation dates, training settings,
assumed costs and seed. **Only pressing “Run and register experiment” executes
the engine.** Changing charts, selected dates, or other inspection widgets never
refits an HMM. Uploaded JSON supplies form values; it is not executed as code.

The inspector offers same-period wealth, signed drawdown, turnover and cost
overlays; chronological reference regime probabilities and posterior entropy;
actual allocation histories; selected-close rule/constraint explanations;
convergence/restart diagnostics; the manifest; and the same reproducibility ZIP
as the CLI. Failed attempts remain inspectable and exportable. Engine, runner,
artifact-reading, and UI rendering functions are separate.

## Interpretation and scientific limits

- Signals use data through close `t`; execution and costs occur at close `t+1`,
  after the old holdings earn that close's return. New target holdings first
  earn `t+1` to `t+2`. Accounting dates are shared exactly, with no filling.
- Evaluation is **chronological attribution** of one continuous run. Holdings
  entering the window may have formed earlier. Window wealth is rebased to 1
  immediately before its first accounting row, not reset to cash.
- Wealth is cumulative compounded simple returns. Saved signed drawdown is
  `wealth / max(1, cumulative_max(wealth)) - 1`; it includes initial wealth and
  the first paid allocation cost. Shared metrics report maximum drawdown as a
  positive loss magnitude using the shared metric specification. Summed daily
  cost fractions are not compounded wealth drag; the metrics' cost drag is
  gross terminal wealth minus net terminal wealth.
- Probabilities describe the latest reference HMM signal known at each close,
  potentially newer than the signal that produced today's holdings. Allocation
  explanations link to the **executed** target's signal instead. Non-HMM
  strategies do not possess HMM posterior probabilities: the shared probability
  table is reference-model context, never a fabricated strategy-specific
  probability. Entropy is model-conditional uncertainty, not a calibrated
  forecast confidence interval.
- HMM-fixed explanations use the same-config reference signal regime, not a
  separately instrumented fit-diagnostics trace for that ablation. Volatility
  rules use their causal 63-row equity volatility at the decision close.
- Explanations document the rule and constraints, not a causal decomposition or
  an assertion that every constraint binds. Optimizer target constraints apply
  at decision/fill targets; subsequent drifted weights may exceed them. Static
  and volatility-rule allocations are not the HMM optimizer's constraint set.
- The canonical dataset is ex-post-cleaned, historical, index-level data.
  `^NSEI` is not directly tradable. The bond proxy is liquid/cash-like, not a
  long-duration sovereign bond hedge. Costs are assumptions, not execution
  calibration; taxes, impact, capacity and implementation shortfall are not
  established.
- Descriptive performance comparisons, repeated researcher trials and state
  probabilities do not establish statistical significance, causal effects,
  forecast skill, future profits, or investment suitability. This workflow
  emits no significance claim or winner declaration.

## Behavioral coverage

`tests/test_workflow.py` uses deterministic engine traces to exercise the real
artifact/manifest/registration paths without stochastic fits: selected variants,
end-date truncation before engine invocation, same-period accounting identities,
checksums and unique versioned attempts, engine/config/missing-dataset failure
registration, strict ingestion, invalid schema boundaries, refusal to silently
clip warmup or fill misaligned strategy samples, raw-data exclusion and tamper
rejection in ZIP export, and CLI run/inspect/export sharing existing artifacts.
UI rendering/visual smoke checks are separate from these runner behavior tests.
