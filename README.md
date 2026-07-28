# RegimeShift — Market-Regime Dynamic Asset Allocation

**IIT Bombay Summer Quant 2026 Assignment**
**Repository:** [`tanmay-alpha/RegimeShift`](https://github.com/tanmay-alpha/RegimeShift)
**Branch:** `polish/final-submission`
**Status:** Final submission — all code executed and verified

---

## 1. Executive Summary

**RegimeShift** is a quantitative asset allocation framework that dynamically adapts multi-asset portfolios across changing market regimes (Bull, Bear, and Crisis). Built specifically for the Indian market asset universe, RegimeShift uses a **3-state Gaussian Hidden Markov Model** combined with a **CVXPY regime-conditioned portfolio optimizer** in a strictly leakage-free, walk-forward backtesting pipeline.

### Target Asset Universe
- **NSE Equity Index:** NIFTY 50 (`^NSEI`)
- **Gold:** GOLDBEES.NS (Nippon India ETF Gold Bees — INR-denominated)
- **Defensive Proxy:** LIQUIDBEES.NS (Nippon India Liquid Bees — INR-denominated liquid-bond / cash-equivalent proxy, NOT a sovereign bond or 10-year G-Sec)
- **Market Volatility Index (Optional):** ^INDIAVIX (NSE India VIX)

**Note:** LIQUIDBEES.NS tracks the Nifty Liquid Bond Index with very short effective duration — it is used as a defensive / cash-equivalent bucket, not as a duration hedge or inflation-protection instrument. This is a documented limitation. All assets are INR-denominated PRICE/NAV series. No currency conversion is required. The VIX is optional and used only as a feature — it is never allocated as a portfolio position.

---

## 2. System Architecture & Pipeline

The pipeline executes chronologically in a single-pass walk-forward framework:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Multi-Asset Data Ingestion & Semantic Validation         │
│    (yfinance online / CSV offline; forward-fill ≤ 3 days)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 2. Leakage-Safe Trailing Feature Engineering                │
│    (7 base features + 2 optional VIX features)              │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 3. Rolling Window Fit (252-day; Train-Only Scaler + HMM)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 4. Online Posterior Regime State Inference p(S_t | X_1:t)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 5. CVXPY Regime-Specific Portfolio Optimization             │
│    (one discrete inferred regime → one of three convex     │
│     objectives with regime-specific constraints)            │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│ 6. Walk-Forward Execution & Transaction Friction Deductions │
│    (initial: full-L1 turnover; rebalance: 0.5×L1)         │
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

### C. 3-State Gaussian HMM — Regime Mapping

The HMM produces arbitrary numeric states (0, 1, 2) on each fitting. These states are mapped to interpretable regime labels **deterministically** using training-period raw feature statistics:

1. **Crisis** = state with the highest crisis score:
   $$
   \text{score} = w_{\text{vol}} \cdot \text{mean}(\text{vol}_{21\text{d}}) + w_{\text{vix}} \cdot \text{mean}(\text{vix}_{\text{level}}) + w_{\text{mom}} \cdot \text{mean}(\text{mom}_{63\text{d}})
   $$
   (with default weights $w_{\text{vol}}=1.0$, $w_{\text{vix}}=0.5$, $w_{\text{mom}}=-0.3$)

2. **Bull** = among the two remaining states, the one with the highest mean(equity_momentum_63d)

3. **Bear** = the remaining state

**Important:** State 0 is **not** always Bull. The numeric-to-regime mapping is derived from training statistics and can differ across fitting windows. The mapping is computed fresh on each training window to ensure it reflects the local market context.

### D. CVXPY Regime-Conditioned Optimizer

For each rebalance, the **inferred regime** (one of Bull / Bear / Crisis) selects one of three convex quadratic objectives:

- **Bull:** $\max_w w^T \mu - \frac{\lambda}{2} w^T \Sigma w$ — maximize risk-adjusted return
- **Bear:** $\min_w \frac{1}{2} w^T \Sigma w - \kappa \cdot w^T \mu$ — minimize risk while rewarding return
- **Crisis:** $\min_w \frac{1}{2} w^T \Sigma w$ — minimum-risk allocation

Constraints (long-only, no leverage):
- Global: $\sum_i w_i = 1$, $0 \le w_i \le 0.80$
- Bull: equity $\in [0.45, 0.80]$, gold $\le 0.35$, bond $\le 0.45$
- Bear: equity $\le 0.40$, gold + bond $\ge 0.60$
- Crisis: equity $\le 0.15$, gold $\ge 0.25$, bond $\ge 0.40$

The optimiser tries solvers in order: CLARABEL → OSQP → SCS.

### E. Realistic Transaction Cost Friction

Two conventions for turnover:

1. **Initial allocation (from cash):**
   $$
   \text{turnover} = \sum_{i=1}^{N} |w_{i}|
   $$

2. **Subsequent rebalance:**
   $$
   \text{turnover} = \frac{1}{2} \sum_{i=1}^{N} |w_{i} - w_{i-1}^{\text{unadj}}|
   $$

3. **Cost and net return on rebalance dates:**
   $$
   \text{cost} = \text{turnover} \times \frac{\text{bps}}{10{,}000}, \qquad
   \text{Net Return}_t = (1 - \text{cost}) \times (1 + \text{Gross Return}_t) - 1
   $$

No transaction cost is deducted on non-rebalance dates.

### F. Transaction-Cost Sensitivity Conclusion

Doubling transaction costs from 5 to 10 bps had a limited effect on Sharpe
and CAGR in this sample, although cumulative cost drag increased materially.
This sensitivity result does not guarantee future robustness.

---

## 4. Repository Structure

```
RegimeShift/
├── README.md                           # Main project documentation
├── pyproject.toml                      # Python package definition (v1.0.0)
├── run_submission.py                   # Official CLI entrypoint
├── requirements.txt                    # Pinned dependencies
├── CLAUDE.md                           # Claude Code session memory
│
├── src/
│   └── regime_shift/                   # Core package
│       ├── __init__.py                 # Public API exports
│       ├── backtest.py                 # Walk-forward backtester engine
│       ├── benchmarks.py               # Static 60/40 & Equal-Weight benchmarks
│       ├── config.py                   # Dataclass configurations (annualization=252)
│       ├── data.py                     # Market data loader (yfinance & CSV)
│       ├── exceptions.py               # Custom domain exceptions
│       ├── features.py                 # Leakage-safe feature pipeline
│       ├── metrics.py                  # Performance metrics
│       ├── plots.py                    # Visualization charts
│       ├── portfolio.py                # CVXPY regime-conditioned optimizer
│       ├── regime_model.py             # 3-State Gaussian HMM
│       └── validation.py               # Data contract validation
│
├── tests/                              # Test suite (286 tests)
│   ├── test_backtest.py                # Backtest engine & benchmark tests
│   ├── test_cli.py                     # CLI behaviour tests
│   ├── test_config.py                  # Configuration tests
│   ├── test_features.py                # Feature pipeline tests
│   ├── test_imports.py                 # Import integrity tests
│   ├── test_integration.py             # End-to-end integration tests
│   ├── test_metrics.py                 # Metrics module tests
│   ├── test_no_legacy_imports.py       # Legacy contamination scanner
│   ├── test_notebook.py                # Notebook structure tests
│   ├── test_plots.py                   # Chart generation tests
│   ├── test_portfolio.py               # CVXPY optimizer tests
│   └── test_regime_model.py            # HMM tests
│
├── notebooks/
│   └── RegimeShift_Submission.ipynb    # Executable submission notebook
│
├── data/                               # Local datasets & cache
├── research/                           # Research sandbox (isolated)
└── results/                            # Generated outputs (PNG + CSV)
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.9+
- `pip` package manager

```bash
git clone https://github.com/tanmay-alpha/RegimeShift.git
cd RegimeShift
pip install -e .
```

---

## 6. Usage & Execution

### Running Official Submission CLI
```bash
# Official offline mode — always use the submitted dataset
python run_submission.py --data-path data/submission_market_data.csv --transaction-cost-bps 5 --output-dir results

# 10 bps sensitivity check
python run_submission.py --data-path data/submission_market_data.csv --transaction-cost-bps 10 --output-dir results_10bps

# With VIX feature included (omitted by default)
python run_submission.py --data-path data/submission_market_data.csv --include-vix --transaction-cost-bps 5
```

### Running Test Suite
```bash
pytest -v
# 286 tests covering leakage, HMM, portfolio optimizer, backtest math,
# charts, CLI, and notebook structure
```

---

## 7. Outputs

All results are saved to the `--output-dir` (default: `results/`):

| File | Description |
|---|---|
| `daily_results.csv` | Daily gross/net returns, costs, turnover, regime, rebalance flag + benchmark returns |
| `weights.csv` | Target weights at rebalance dates |
| `regimes.csv` | Regime labels and probabilities per date |
| `transition_matrix.csv` | Last HMM transition matrix (3×3) |
| `performance_summary.csv` | Six-row table: RegimeShift Gross/Net, 60/40 Gross/Net, EW Gross/Net |
| `run_metadata.json` | Date range, tickers, HMM config, package version |
| `regime_price_chart.png` | NIFTY price with regime background shading |
| `transition_matrix.png` | Regime transition heatmap |
| `equity_curves.png` | Normalized equity curves (strategy + benchmarks) |
| `drawdowns.png` | Drawdown chart via (1+r).cumprod() |
| `portfolio_weights.png` | Daily drifted portfolio weights |
| `regime_probabilities.png` | Bull/Bear/Crisis probability series |

---

## 8. Implementation Checklist

- [x] **Phase 1:** Multi-Asset Data Pipeline (`src/regime_shift/data`, `validation`)
- [x] **Phase 2:** Leakage-Safe Feature Engineering (`src/regime_shift/features`)
- [x] **Phase 3:** Sequential 3-State Gaussian HMM (`src/regime_shift/regime_model`)
- [x] **Phase 4:** CVXPY Regime-Specific Portfolio Optimizer (`src/regime_shift/portfolio`)
- [x] **Phase 5:** Walk-Forward Backtesting Engine (`src/regime_shift/backtest`)
- [x] **Phase 6:** Static Benchmarks (60/40, Equal-Weight) (`src/regime_shift/benchmarks`)
- [x] **Phase 7:** Transaction Costs (5–10 bps) (`src/regime_shift/backtest`)
- [x] **Phase 8:** Performance Metrics (Sharpe, Sortino, Drawdown, Calmar, Turnover) (`src/regime_shift/metrics`)
- [x] **Phase 9:** Charts & Visualizations (`src/regime_shift/plots`)
- [x] **Phase 10:** Submission Entrypoint & Executable Notebook (`run_submission.py`, `notebooks/RegimeShift_Submission.ipynb`)

---

## 9. Disclaimers

> [!NOTE]
> **Academic & Research Disclaimer:** Developed for the IIT Bombay Summer Quant 2026 assignment. This codebase is an academic implementation for quantitative research and performance benchmarking.

> [!WARNING]
> **No Financial Advice:** Nothing in this repository constitutes financial, investment, or trading advice.

---

*Package version: 1.0.0 | Generated: 2026-07-28 | IIT Bombay Summer Quant 2026*
