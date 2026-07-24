# Repository Cleanup Audit — RegimeShift (IIT Bombay 2026)

**Repository:** `tanmay-alpha/RegimeShift`  
**Audit Date:** July 24, 2026  
**Auditor:** Antigravity AI  
**Branch:** `cleanup/iitb-2026-foundation`  
**Legacy Preservation Branch:** `legacy/pre-cleanup`

---

## 1. Executive Summary

This repository audit evaluated all tracked and untracked files in `tanmay-alpha/RegimeShift` prior to structural cleanup. The audit identified substantial project contamination from legacy BTC/crypto trading code, broken backtester logic with look-ahead bias, unverified performance claims, duplicate HMM/feature implementations, and non-standard dependency overhead.

---

## 2. Pre-Cleanup Repository File Tree

```
RegimeShift/
├── .gitignore
├── MY_LEARNING_GUIDE.md
├── Problem_statement.pdf
├── README.md
├── backtester.py
├── btc_18_22_1d.csv
├── config.py
├── dashboard.py
├── final_data.csv
├── live_trader.log
├── live_trader_stub.py
├── main.py
├── quant_skills_and_manual_tasks.md
├── requirements.txt
├── run_backtest.py
├── stress_test.py
├── notebooks/
│   ├── analysis.ipynb
│   └── generate_notebook.py
├── src/
│   └── regime_shift/
│       ├── __init__.py
│       ├── backtest.py
│       ├── benchmarks.py
│       ├── data_loader.py
│       ├── evaluate.py
│       ├── features.py
│       ├── monte_carlo.py
│       ├── optimizer.py
│       ├── regime_detector.py
│       ├── regime_features.py
│       ├── regime_signal.py
│       ├── stats.py
│       ├── strategy.py
│       ├── transaction_costs.py
│       └── visualize.py
└── tests/
    ├── __init__.py
    ├── test_backtest.py
    ├── test_features.py
    ├── test_hmm.py
    ├── test_phase3.py
    ├── test_regime_detector.py
    ├── test_robustness.py
    ├── test_stats.py
    └── test_strategy.py
```

---

## 3. File Classification Matrix

| File / Component | Category | Description & Justification |
| :--- | :---: | :--- |
| `Problem_statement.pdf` | **A** | Official IIT Bombay assignment specification. Preserved in repository root. |
| `README.md` | **D/Replacement** | Contains misleading claims of "institutional-grade" crypto backtesting. Replaced with honest foundation README. |
| `config.py` | **C/Replacement** | Hardcoded Binance fees, BTC symbols, volume spike parameters. Replaced by `src/regime_shift/config.py`. |
| `requirements.txt` | **C/Replacement** | Contains unused crypto dependencies (`ccxt`, `streamlit`, `plotly`, `pandas-ta-classic`). Replaced with clean set. |
| `pyproject.toml` | **A** | Modern packaging setup created for clean dependency declaration. |
| `btc_18_22_1d.csv` | **C** | Legacy BTC daily price data (2018–2022). Removed from active submission branch. |
| `final_data.csv` | **C/E** | Mixed simulated/crypto historical data file. Removed from active submission branch. |
| `live_trader_stub.py` | **C** | Binance CCXT paper trading engine stub. Unrelated to asset allocation assignment. Removed. |
| `live_trader.log` | **E** | Execution log file from crypto stub. Removed. |
| `dashboard.py` | **C** | Streamlit visual interactive app for crypto trading. Removed. |
| `stress_test.py` | **B/C** | High-frequency stress test generator for crypto orderbooks. Removed. |
| `backtester.py` | **D** | Legacy root-level backtester script with look-ahead bias and uncalibrated costs. Removed. |
| `main.py` | **C/D** | Legacy runner script driving crypto/simulated pipelines. Removed. |
| `run_backtest.py` | **C/D** | Old CLI backtest invocation script. Removed. |
| `MY_LEARNING_GUIDE.md` | **E** | Informal notes markdown file. Removed. |
| `quant_skills_and_manual_tasks.md` | **E** | Internal task checklist markdown file. Removed. |
| `notebooks/analysis.ipynb` | **D** | Auto-generated notebook with hardcoded crypto outputs. Replaced by `notebooks/RegimeShift_Submission.ipynb`. |
| `notebooks/generate_notebook.py` | **E** | Script that generated `analysis.ipynb`. Removed. |
| `src/regime_shift/regime_detector.py` | **B** | Custom Student-t HMM (Baum-Welch, Viterbi, log-gamma, Dirichlet prior). Moved to `research/legacy_student_t_hmm/`. |
| `src/regime_shift/regime_features.py` | **B** | 54-dimensional feature extraction engine. Moved to `research/legacy_student_t_hmm/`. |
| `src/regime_shift/transaction_costs.py` | **B/D** | Almgren-Chriss market impact cost model and Binance cost model. Isolated under `research/legacy_student_t_hmm/`. |
| `src/regime_shift/monte_carlo.py` | **B** | Research bootstrap and Monte Carlo stress tools. Isolated under `research/legacy_student_t_hmm/`. |
| `src/regime_shift/visualize.py` | **B** | Research plotting utilities (15+ plots). Isolated under `research/legacy_student_t_hmm/`. |
| `src/regime_shift/backtest.py` | **D** | Walk-forward backtester with look-ahead leakage. Replaced with `NotImplementedError` stub. |
| `src/regime_shift/data_loader.py` | **C/D** | Data loader with BTC-only defaults (`compute_features_btc`). Replaced with `src/regime_shift/data.py` and `validation.py`. |
| `src/regime_shift/benchmarks.py` | **D** | Benchmark implementations with timing mismatches. Replaced with clean stub. |
| `src/regime_shift/optimizer.py` | **D** | Heuristic portfolio optimizer. Replaced with clean stub. |
| `src/regime_shift/strategy.py` | **D** | Signal-to-position strategy class. Replaced with clean stub. |
| `src/regime_shift/evaluate.py` | **D** | Metrics evaluation module. Replaced with `src/regime_shift/metrics.py`. |
| `src/regime_shift/stats.py` | **B** | Custom statistical utilities. Moved to `research/legacy_student_t_hmm/`. |
| `tests/*` | **D** | Old tests asserting against legacy/crypto modules. Replaced with foundation integrity tests in `tests/`. |

*Categories:*  
- **A**: Required for official submission  
- **B**: Potentially useful advanced research (isolated to `research/`)  
- **C**: Unrelated or contaminated legacy code (removed from active branch)  
- **D**: Broken or misleading logic (replaced with clean stubs/contracts)  
- **E**: Generated output / temporary artifact (removed from active branch)

---

## 4. Audit Findings & Issues Breakdown

### A. Look-Ahead Bias & Flawed Backtesting
1. **Same-day Return Leakage:** In legacy `src/regime_shift/backtest.py`, feature normalization and signal calculation for index `t` incorporated observation `t`, after which portfolio return was calculated against `returns.iloc[t]`, effectively allowing trades at prices already reflecting day `t` returns.
2. **Global Pre-processing:** Scalers (e.g. `StandardScaler` / z-score normalization in `data_loader.py` and `regime_features.py`) were fit over the entire dataset range prior to splitting into walk-forward slices.
3. **Full-sample HMM Viterbi Smoothing:** Viterbi state decoding was executed across the full dataset time-series rather than strictly using expanding window filtering ($P(S_t | X_{1:t})$).

### B. Project Contamination & Crypto Code
1. **Hardcoded BTC Defaults:** Default tickers across configuration, loaders, and features defaulted to BTC/USD and crypto pairs.
2. **Annualization Factor Error:** Returns were annualized using $365$ days instead of standard equity trading year factor $252$.
3. **Unused Crypto Integrations:** `live_trader_stub.py`, `dashboard.py`, `stress_test.py`, and `requirements.txt` included CCXT, Binance API keys, testnet configurations, and orderbook dynamics.

### C. Transaction Cost Disconnect
1. In `src/regime_shift/backtest.py`, transaction costs were calculated separately for reporting but were not deducted from net cumulative asset returns when iterating step-by-step.
2. Costs were specified in Binance spot fee conventions (0.1%) instead of standard basis points ($5\text{--}10\text{ bps}$) for equity/gold/bonds.

### D. Misleading Claims & Stale Dependencies
1. `README.md` claimed "Institutional-grade Student-t HMM Regime-Switching Strategy" with verified alpha, despite zero out-of-sample testing on real Indian market data (NSE equity, Gold, Indian Sovereign Bonds).
2. Dependencies included `streamlit`, `plotly`, `ccxt`, `pandas-ta-classic`, `seaborn` which are not required for core quantitative allocation and backtesting.

---

## 5. Action Plan & Decision Log

1. **Preserve Legacy State:** Created Git branch `legacy/pre-cleanup` preserving all original files, scripts, datasets, and history.
2. **Isolate Research Code:** Moved Student-t HMM, Viterbi, Baum-Welch, 54-feature engine, and Monte Carlo tools to `research/legacy_student_t_hmm/` with explicit non-import guarantees.
3. **Remove Contamination:** Deleted all BTC datasets, crypto trading stubs, Streamlit apps, orderbook stress testers, and unnecessary notes from active submission branch.
4. **Establish Data Contracts:** Defined strict data contracts in `src/regime_shift/validation.py` verifying positive prices, ascending `DatetimeIndex`, no duplicate dates, required columns (`equity`, `gold`, `bond`, optional `vix`), and no backward fill.
5. **Establish Neutral Configuration:** Created `src/regime_shift/config.py` with `annualization_factor=252`, `transaction_cost_bps=5.0`, and `n_regimes=3`.
6. **Replace Backtester & Models with Clean Stubs:** Installed explicit `NotImplementedError` stubs on all unfinished strategy/backtest interfaces to prevent partial execution or false claims.
7. **Write Clean Submission Notebook:** Built 14-section `notebooks/RegimeShift_Submission.ipynb` skeleton.
8. **Rewrite Documentation & Tests:** Created `README.md`, `CLEANUP_REPORT.md`, `pyproject.toml`, clean `requirements.txt`, and 4 foundation test suites in `tests/`.
