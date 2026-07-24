# RegimeShift — Market-Regime Dynamic Asset Allocation

**IIT Bombay Summer Quant 2026 Assignment**  
**Repository:** `tanmay-alpha/RegimeShift`  
**Current Status:** Foundation Cleanup Branch (`cleanup/iitb-2026-foundation`)

---

## 1. Assignment Objective

The goal of this project is to build a leakage-free market-regime identification system (Bull, Bear, Crisis) that dynamically rebalances multi-asset portfolios across Indian market assets to outperform static benchmarks on unseen out-of-sample data.

### Target Asset Universe
- **NSE Equity Index** (e.g., NIFTY 50 / `^NSEI`)
- **Gold** (MCX Gold / Gold ETF proxy)
- **Indian Sovereign Bonds** (10-Year G-Sec proxy)
- **Market Volatility Index (Optional)** (India VIX / `^INDIAVIX`)

---

## 2. Planned Methodology

1. **Sequential Data Processing:** Strict walk-forward expansion with zero future-data leakage or look-ahead bias.
2. **Hidden Markov Model (HMM):** Sequential Gaussian HMM fit on rolling historical windows to infer market regime states (Bull, Bear, Crisis).
3. **Regime-Conditioned Optimization:** Dynamic portfolio allocation using CVXPY.
4. **Transaction Cost Modeling:** Realistic deduction of $5\text{--}10\text{ bps}$ transaction friction on all portfolio rebalances.
5. **Rigorous Benchmarking:** Side-by-side evaluation against static 60/40 Equity/Bond and Equal-Weight benchmarks.

---

## 3. Current Status & Cleanup Summary

This repository has undergone a comprehensive foundation cleanup to remove legacy BTC/crypto code, uncalibrated high-frequency tools, and backtesting logic affected by look-ahead bias.

### What Has Been Removed / Isolated
- **Crypto & BTC Contamination:** Deleted BTC datasets, Binance fee models, CCXT exchange connectors, testnet trading stubs, and crypto 365-day annualization assumptions.
- **Experimental Student-t HMM:** Isolated custom Student-t HMM, Viterbi/Baum-Welch routines, and 54-feature pipelines under `research/legacy_student_t_hmm/`. These are not imported by official submission code.
- **Flawed Backtester:** Removed legacy backtesting logic that allowed same-day return leakage and uncosted portfolio returns. Replaced with explicit `NotImplementedError` stubs.

---

## 4. Repository Structure

```
RegimeShift/
├── README.md                           # Honest project README & status
├── CLEANUP_AUDIT.md                    # Full pre-cleanup audit log
├── CLEANUP_REPORT.md                   # Post-cleanup summary & Codex roadmap
├── pyproject.toml                      # Clean project packaging definition
├── requirements.txt                    # Pinned core dependencies
├── .gitignore                          # Standard git ignore file
├── run_submission.py                   # Submission CLI entrypoint
│
├── notebooks/
│   └── RegimeShift_Submission.ipynb   # 14-section submission notebook skeleton
│
├── src/
│   └── regime_shift/
│       ├── __init__.py                 # Package exports
│       ├── config.py                   # Dataclass configuration (annualization=252)
│       ├── data.py                     # Data loader interface
│       ├── features.py                 # Feature engineering interface
│       ├── regime_model.py             # Gaussian HMM model interface
│       ├── portfolio.py                # CVXPY portfolio optimizer interface
│       ├── backtest.py                 # Walk-forward backtester interface
│       ├── benchmarks.py               # Benchmark computation interface
│       ├── metrics.py                  # Quantitative metrics interface
│       └── validation.py               # Data contract validation functions
│
├── tests/
│   ├── test_imports.py                 # Package import integrity test
│   ├── test_config.py                  # Configuration parameter verification
│   ├── test_data_contract.py           # Data contract validation tests
│   └── test_no_legacy_imports.py       # Legacy code contamination scan
│
├── data/
│   └── README.md                       # Data formatting guidelines
│
├── results/
│   └── .gitkeep                        # Output directory placeholder
│
└── research/
    ├── README.md                       # Research directory overview
    └── legacy_student_t_hmm/           # Isolated legacy Student-t HMM code
```

---

## 5. Implementation Checklist

- [x] Real multi-asset data loader (Phase 1 + Phase 1 hardening)
- [x] Leakage-safe feature engineering (Phase 2)
- [ ] Walk-forward HMM
- [ ] Regime labelling
- [ ] CVXPY portfolio optimization
- [ ] 5–10 bps transaction costs
- [ ] 60/40 benchmark
- [ ] Equal-weight benchmark
- [ ] Gross and net performance
- [ ] Sharpe
- [ ] Sortino
- [ ] Maximum drawdown
- [ ] Calmar
- [ ] Turnover
- [ ] Truncation-invariance test
- [ ] Final executed notebook

### Feature Set

The feature pipeline produces a small, interpretable feature set for the future 3-state Gaussian HMM. All features use **trailing rolling windows only** — no centered windows, no backward fill, no future data.

| Feature | Formula | Window |
|---|---|---|
| `equity_log_return_1d` | `log(equity_t / equity_{t-1})` | 1 day |
| `equity_momentum_21d` | `equity_t / equity_{t-21} - 1` | 21 days |
| `equity_momentum_63d` | `equity_t / equity_{t-63} - 1` | 63 days |
| `equity_volatility_21d` | `rolling_std(log_return, 21) * sqrt(252)` | 21 days |
| `equity_volatility_ratio_21_63` | `vol_21d / vol_63d` | 21/63 days |
| `equity_gold_correlation_63d` | `rolling_corr(equity_log_ret, gold_log_ret, 63)` | 63 days |
| `equity_bond_correlation_63d` | `rolling_corr(equity_log_ret, bond_log_ret, 63)` | 63 days |
| `vix_change_5d` *(optional)* | `vix_t / vix_{t-5} - 1` | 5 days |
| `vix_level` *(optional)* | `vix_t` (raw level) | — |

**Signal timing:** Feature observed at `t` is used to make the decision for `t+1`.

**Scaling policy:** StandardScaler is fit on training rows only (not the full dataset). Walk-forward procedure: select training data ending at `t-1`, fit scaler, transform training + current observation.

**Warmup:** 62 rows removed (first 63-day window has insufficient data). Deterministic count — no filling.

**VIX:** Optional feature-only series. Never included in portfolio asset returns. Omitted cleanly if absent.

### Leakage Protections

- No `shift(-1)` or any negative shift
- No centered rolling windows
- No backward fill
- No global mean/standard deviation
- No full-sample normalization
- No future Viterbi states, returns, or regime labels
- No random train/test shuffling

---

## 6. Installation & Testing

### Installation
```bash
git clone https://github.com/tanmay-alpha/RegimeShift.git
cd RegimeShift
git checkout cleanup/iitb-2026-foundation
pip install -e .
```

### Running Foundation Tests
```bash
pytest tests/ -v
```

---

## 7. Disclaimers

> [!NOTE]
> **Academic & Research Disclaimer:** This project is developed solely for the IIT Bombay Summer Quant 2026 assignment. It is an academic codebase under active development and is NOT production-ready or institutional trading software.

> [!WARNING]
> **No Financial Advice:** Nothing in this repository constitutes financial, investment, or trading advice.
