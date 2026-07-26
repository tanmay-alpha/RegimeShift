# RegimeShift — Market-Regime Dynamic Asset Allocation

**IIT Bombay Summer Quant 2026 Assignment**  
**Repository:** [`tanmay-alpha/RegimeShift`](https://github.com/tanmay-alpha/RegimeShift)  
**Branch:** `feature/final-walkforward-submission`  
**Status:** Full Walk-Forward Engine Implemented & Verified (225/225 Passing Tests)

---

## 1. Executive Summary

**RegimeShift** is a quantitative asset allocation framework designed to dynamically adapt multi-asset portfolios across changing market regimes (Bull, Bear, and Crisis). Built specifically for the Indian market asset universe, RegimeShift utilizes a 3-state Gaussian Hidden Markov Model (HMM) combined with a CVXPY regime-conditioned portfolio optimizer within a strictly leakage-free, walk-forward backtesting pipeline.

### Target Asset Universe
- **NSE Equity Index:** NIFTY 50 (`^NSEI`)
- **Gold:** MCX Gold / Gold ETF (`GC=F`)
- **Indian Sovereign Bonds:** 10-Year G-Sec Proxy (`^SENSEX` / Bond Proxy)
- **Market Volatility Index (Optional):** India VIX (`^INDIAVIX`)

---

## 2. System Architecture & Pipeline

The pipeline executes chronologically in a strict single-pass walk-forward framework:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Multi-Asset Data Ingestion & Semantic Validation         │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 2. Leakage-Safe Trailing Feature Engineering                │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 3. Rolling Window Fit (Train-Only Scaler + Gaussian HMM)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 4. Online Posterior Regime State Inference p(S_t | X_1:t)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 5. CVXPY Regime-Conditioned Portfolio Optimization          │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 6. Walk-Forward Execution & Transaction Friction Deductions │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 7. Benchmarking (60/40, EW) & Performance Visualizations    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Key Methodological Features & Mathematical Principles

### A. Strict Zero-Leakage Architecture
To guarantee out-of-sample validity and eliminate look-ahead bias:
- **No Future Data:** Features use trailing rolling windows exclusively ($t-N$ to $t$). No negative shifts (`shift(-1)`), centered windows, or backward filling.
- **Train-Only Scaling:** `StandardScaler` is fitted solely on historical training windows ($1$ to $t-1$) and applied to observation $t$.
- **Causal Timing:** Signal generated at time $t$ determines portfolio weights executed for return at $t+1$.
- **No Full-Sample Normalization:** HMM parameters and regime probabilities are computed using rolling historical sequence history without future Viterbi smoothing.

### B. Leakage-Safe Feature Set
| Feature | Description / Formula | Window |
|---|---|---|
| `equity_log_return_1d` | Single-day log return: $\ln(P_t / P_{t-1})$ | 1 day |
| `equity_momentum_21d` | 1-month equity momentum: $P_t / P_{t-21} - 1$ | 21 days |
| `equity_momentum_63d` | 3-month equity momentum: $P_t / P_{t-63} - 1$ | 63 days |
| `equity_volatility_21d` | Annualized 1-month volatility: $\text{std}(\text{ret}_{21}) \times \sqrt{252}$ | 21 days |
| `equity_volatility_ratio_21_63` | Volatility regime indicator: $\text{vol}_{21\text{d}} / \text{vol}_{63\text{d}}$ | 21/63 days |
| `equity_gold_correlation_63d` | Equity-Gold diversification metric: $\text{corr}(\text{ret}_{\text{eq}}, \text{ret}_{\text{gold}}, 63)$ | 63 days |
| `equity_bond_correlation_63d` | Equity-Bond diversification metric: $\text{corr}(\text{ret}_{\text{eq}}, \text{ret}_{\text{bond}}, 63)$ | 63 days |
| `vix_change_5d` *(Optional)* | 5-day VIX rate of change: $VIX_t / VIX_{t-5} - 1$ | 5 days |
| `vix_level` *(Optional)* | Absolute VIX level | Raw $t$ |

### C. 3-State Gaussian HMM
Fits a 3-state Gaussian Hidden Markov Model with diagonal covariance matrices on rolling historical windows (e.g., 504 trading days). Regimes are dynamically sorted and labeled:
- **State 0 — Bull:** High expected equity return, low volatility.
- **State 1 — Bear:** Negative expected equity return, elevated volatility.
- **State 2 — Crisis:** Extreme negative expected return, severe volatility spike.

### D. CVXPY Regime-Conditioned Optimizer
Blends regime mean vector $\mu_k$ and covariance matrices $\Sigma_k$ weighted by online posterior regime probabilities $p_{t,k} = P(S_t = k \mid X_{1:t})$:
$$\mu_t = \sum_{k=0}^{2} p_{t,k} \mu_k, \quad \Sigma_t = \sum_{k=0}^{2} p_{t,k} \Sigma_k$$
Formulates a convex quadratic optimization problem:
$$\max_{w} \quad w^T \mu_t - \frac{\gamma}{2} w^T \Sigma_t w$$
$$\text{subject to} \quad \sum_{i=1}^{N} w_i = 1, \quad 0 \le w_i \le 1$$

### E. Realistic Transaction Cost Friction
Deducts transaction costs (default 5–10 bps) on every rebalance based on portfolio turnover:
$$\text{Net Return}_t = \text{Gross Return}_t - c \cdot \sum_{i=1}^N |w_{t,i} - w_{t-1,i}^{\text{unadj}}|$$

---

## 4. Repository Structure

```
RegimeShift/
├── README.md                           # Main project documentation
├── pyproject.toml                      # Python package packaging definition
├── requirements.txt                    # Pinned dependency requirements
├── run_submission.py                   # Main CLI entrypoint runner
├── CLEANUP_AUDIT.md                    # Pre-cleanup legacy code audit log
├── CLEANUP_REPORT.md                   # Post-cleanup refactoring summary
│
├── src/
│   └── regime_shift/                   # Core regime_shift package
│       ├── __init__.py                 # Package exports
│       ├── backtest.py                 # Walk-forward backtester engine
│       ├── benchmarks.py               # Static 60/40 & Equal-Weight benchmarks
│       ├── config.py                   # Dataclass configurations (annualization=252)
│       ├── data.py                     # Market data loader (yfinance & CSV)
│       ├── exceptions.py               # Custom domain exceptions
│       ├── features.py                 # Trailing-only feature pipeline
│       ├── metrics.py                  # Quantitative performance metrics
│       ├── plots.py                    # Visualizations & equity curve plotting
│       ├── portfolio.py                # CVXPY regime-conditioned optimizer
│       ├── regime_model.py             # 3-State Gaussian HMM model
│       └── validation.py               # Data contract validation functions
│
├── tests/                              # Comprehensive test suite (225 passing tests)
│   ├── test_backtest.py                # Walk-forward backtest & cost tests
│   ├── test_config.py                  # Configuration parameter verification
│   ├── test_data_contract.py           # Data schema & contract validation
│   ├── test_data_pipeline.py           # Data loading & alignment tests
│   ├── test_features.py                # Feature pipeline & leakage tests
│   ├── test_imports.py                 # Package import integrity
│   ├── test_integration.py             # End-to-end integration tests
│   ├── test_no_legacy_imports.py       # Legacy code contamination scanner
│   ├── test_portfolio.py               # CVXPY optimization & constraint tests
│   └── test_regime_model.py            # HMM fitting & posterior inference tests
│
├── notebooks/                          # Jupyter submission notebooks
│   └── RegimeShift_Submission.ipynb   # Executable submission notebook
│
├── data/                               # Local dataset folder & CSV format guides
│   └── README.md
│
├── research/                           # Research sandbox & legacy code isolation
│   └── legacy_student_t_hmm/           # Isolated legacy Student-t HMM code
│
└── results/                            # Generated performance metrics and plots
    └── .gitkeep
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.9+
- `pip` package manager

### Clone Repository & Install
```bash
git clone https://github.com/tanmay-alpha/RegimeShift.git
cd RegimeShift
git checkout feature/final-walkforward-submission
pip install -e .
```

---

## 6. Usage & Execution

### Running Official Submission CLI
Execute the end-to-end walk-forward pipeline using `run_submission.py`:

```bash
# Online mode (fetches market data via yfinance from 2010 to current date)
python run_submission.py --start 2010-01-01 --transaction-cost-bps 5.0

# Offline mode (using pre-downloaded CSV data)
python run_submission.py --data-path data/prices.csv --transaction-cost-bps 5.0

# With VIX feature included
python run_submission.py --include-vix --transaction-cost-bps 5.0
```

### Running Test Suite
Verify system integrity with the full unit test suite (225 tests covering leakage, HMM, portfolio optimizer, and backtest math):

```bash
pytest -v
```

---

## 7. Implementation Checklist

- [x] **Phase 1: Multi-Asset Data Pipeline** (`regime_shift.data`, `regime_shift.validation`)
- [x] **Phase 2: Leakage-Safe Feature Engineering** (`regime_shift.features`)
- [x] **Phase 3: Sequential 3-State Gaussian HMM** (`regime_shift.regime_model`)
- [x] **Phase 4: CVXPY Regime-Conditioned Portfolio Optimizer** (`regime_shift.portfolio`)
- [x] **Phase 5: Walk-Forward Backtesting Engine** (`regime_shift.backtest`)
- [x] **Phase 6: Static Benchmarks (60/40 Equity/Bond & Equal-Weight)** (`regime_shift.benchmarks`)
- [x] **Phase 7: Realistic Transaction Costs (5–10 bps)** (`regime_shift.backtest`)
- [x] **Phase 8: Comprehensive Metrics (Sharpe, Sortino, Drawdown, Calmar, Turnover, Net Returns)** (`regime_shift.metrics`)
- [x] **Phase 9: Truncation-Invariance & Contamination Verification** (`tests/`)
- [x] **Phase 10: Submission Entrypoint & Executable Notebook** (`run_submission.py`, `notebooks/RegimeShift_Submission.ipynb`)

---

## 8. Disclaimers

> [!NOTE]
> **Academic & Research Disclaimer:** Developed for the IIT Bombay Summer Quant 2026 assignment. This codebase is an academic implementation for quantitative research and performance benchmarking.

> [!WARNING]
> **No Financial Advice:** Nothing in this repository constitutes financial, investment, or trading advice.
