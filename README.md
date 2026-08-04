# RegimeShift — causal regime-allocation research

RegimeShift is a reproducible, index-level Indian multi-asset allocation study. It uses rolling train-only features, a three-state Gaussian HMM and a constrained optimizer, evaluated with causal close-only execution and stated asset-level cost assumptions.

The original academic baseline demonstrated useful walk-forward engineering but did not outperform static benchmarks. Corrected execution timing and uncertainty work are reported without performance-driven retuning.

## What the corrected experiment does

```text
price data -> causal features -> train-only scaler/HMM -> constrained target
      ^                                                       |
      +---- existing weights earn current close-to-close bar <-+
                                      then costs are paid at close
```

At close `t`, the portfolio already earned the `t-1 -> t` return. Only then can it observe data through `t`, fit the HMM, trade the target and deduct costs. The target first earns `t -> t+1`. `results/submission/execution_timeline.csv` records signal, execution, return, pre-trade, target, post-trade and drifted weights.

## Corrected results

Official output is `results/submission/` and uses `NEXT_CLOSE`, three deterministic HMM restarts, and a 10 bps one-way per-asset assumed-cost scenario. The dataset ends 2026-07-27 and remains an **index-level allocation simulation**: `^NSEI` is not a guaranteed tradable fill.

| Net strategy | CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|
| RegimeShift | 6.02% | 0.400 | 35.77% |
| Static 60/40 | 7.20% | 0.760 | 23.39% |
| Equal Weight | 8.95% | 0.588 | 32.96% |

The former 7.16% / 0.459 RegimeShift result is retained only in `results/baseline_legacy/`; it used the invalid same-bar execution convention and is not a resume result.

## Research protocol

The protocol is in [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md) and frozen settings in `config/frozen_resume_v1.yaml`.

| Research period | Dates | Interpretation |
|---|---|---|
| Development | 2010–2018 | Retrospective research |
| Validation | 2019–2021 | Retrospective research |
| Evaluation | 2022–2026 | Retrospective evaluation, not a pristine holdout |
| Future lockbox | after dataset end | Frozen config only |

No historical split is called untouched because the full sample had been inspected before this protocol. HMM restarts are selected by training likelihood, never later portfolio performance.

## Costs and tradability

The cost model charges absolute per-asset traded notional at each close. `optimistic`, `base`, and `stressed` scenarios correspond to 5, 10, and 20 bps one-way assumed cost. These are stress assumptions, not sourced measurements of spreads or fills. Market impact, capacity, ETF tracking difference and NAV-versus-traded-price differences are excluded because volume/ADV and execution data are absent.

`^NSEI` may be used as a regime/benchmark series, but it is an index. GOLDBEES and LIQUIDBEES are the gold and defensive ETF/NAV proxies. A resume-facing executable ETF claim requires a separately verified common-history equity ETF dataset, corporate-action treatment and liquidity audit.

## Reproduce

```powershell
pip install -e ".[dev]"
python run_submission.py --data-path data/submission_market_data.csv --transaction-cost-bps 10 --output-dir results/submission
python -m pytest -q
```

Use `--include-vix` only when the input actually has a valid `vix` column; without the flag a supplied VIX column is deliberately ignored. `NEXT_OPEN` is rejected for this close-only dataset rather than being simulated from closes.

## Evidence and limitations

- `results/submission/experiment_manifest.json` binds outputs to a dataset hash, configuration hash, commit and result hashes.
- Static benchmarks use the identical event ordering and cost model.
- The HMM is a research hypothesis. Under corrected timing it underperforms both committed static benchmarks on headline Sharpe and drawdown.
- Historical output is retrospective. Bootstrap confidence, fold/ablation and capacity claims require the corresponding reproducible artefacts before they may be asserted.
- See [docs/QUANT_INTERVIEW_GUIDE.md](docs/QUANT_INTERVIEW_GUIDE.md) for concise code-matched answers.

## Layout

```text
src/regime_shift/   causal engine, costs, HMM, metrics, manifest
config/             protocol and frozen configuration
results/submission/ corrected, hashed official experiment
results/baseline_legacy/ isolated comparison-only legacy summary
tests/              timing, leakage, data, HMM and metric checks
```
