# RegimeShift 📈

> A research-grade quantitative trading framework for BTC/USD using **Hidden Markov Model regime detection** combined with **volume-spike momentum signals**.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧠 Core Thesis

**RegimeShift** detects latent market regimes (Bull / Bear / Crisis) in BTC daily data using a **Gaussian HMM (Hamilton 1989)**, then applies a **volume-spike entry strategy conditionally** based on the detected regime:

| Regime | Rule | Rationale |
|--------|------|-----------|
| **Bull** | LONG entries only (short spikes blocked) | Trend-following in uptrend |
| **Bear** | SHORT entries only (long spikes blocked) | Momentum in downtrend |
| **Crisis** | 50% position sizing | Volatility protection |

This is the "RegimeShift" innovation: the same volume spike has **regime-dependent edge**.

---

## 📁 Project Structure

```
RegimeShift/
├── main.py                      ← Full 9-step pipeline (entry point)
├── run_backtest.py              ← HMM walk-forward backtest runner
├── dashboard.py                 ← Streamlit interactive dashboard
├── backtester.py                ← Trade execution + PnL simulation engine
├── config.py                    ← All hyperparameters (edit here)
├── btc_18_22_1d.csv             ← BTC/USD OHLCV 2018–2022 (daily)
├── final_data.csv               ← Output: signals + indicators
├── requirements.txt
│
├── src/regime_shift/
│   ├── data_loader.py           ← Load BTC CSV + OHLCV validation
│   ├── features.py              ← 7 HMM features (all lookahead-safe)
│   ├── regime_detector.py       ← Gaussian HMM (Baum-Welch + Viterbi)
│   ├── strategy.py              ← Regime-conditional volume-spike signals
│   ├── backtest.py              ← Walk-forward backtest
│   ├── stats.py                 ← 15+ quant metrics
│   ├── optimizer.py             ← Mean-variance portfolio optimizer
│   ├── monte_carlo.py           ← Block bootstrap significance tests
│   ├── evaluate.py              ← Bootstrap confidence intervals
│   └── benchmarks.py            ← Buy-and-hold, 60/40 benchmarks
│
└── tests/
    ├── test_features.py         ← ATR, OBV, lookahead bias tests
    ├── test_stats.py            ← All metric formula tests
    ├── test_strategy.py         ← Signal encoding, regime filter tests
    └── test_hmm.py              ← HMM convergence, state labeling tests
```

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (9 steps)
python main.py

# Skip HMM (pure volume-spike only, faster)
python main.py --no-regime

# Skip Monte Carlo (faster)
python main.py --no-monte-carlo

# HMM walk-forward backtest
python run_backtest.py

# Auto-select n_states via BIC
python run_backtest.py --select-nstates

# Interactive dashboard
streamlit run dashboard.py

# Run all tests
pytest tests/ -v
```

---

## 📊 Strategy Details

### 1. Volume Spike Detection

Threshold computed using only **past data** (no lookahead):

$$\text{threshold}_t = \mu_{\text{vol}}[t-w:t-1] + k \cdot \sigma_{\text{vol}}[t-w:t-1]$$

where $w = 20$ (window), $k = 1.5$ (multiplier). Entry triggered when:
- $\text{volume}_t > \text{threshold}_t$ AND
- Candle is bullish (close > open) → LONG
- Candle is bearish (close < open) → SHORT

### 2. ATR Trailing Stop (Chandelier Exit — LeBeau 2000)

$$SL_{\text{long},t}  = \text{close}_t - 2.0 \times \text{ATR}_{14,t}$$
$$SL_{\text{short},t} = \text{close}_t + 2.0 \times \text{ATR}_{14,t}$$

Stop trails behind price (only moves in favorable direction).

### 3. Additional Exit Rules

- **3 consecutive adverse bars** → close position
- **Counter-trend volume spike** → reverse position (LONG ↔ SHORT)

### 4. HMM Regime Detection (Hamilton 1989)

Gaussian HMM fitted via Baum-Welch EM on 7-dimensional feature vector:

| Feature | Formula | Captures |
|---------|---------|---------|
| `ret_ann` | $252 \times \bar{r}_{20}$ | Trend direction |
| `vol_ann` | $\sqrt{252} \times \sigma_{20}$ | Market turbulence |
| `vol_zscore` | $(V_t - \mu_V) / \sigma_V$ | Institutional activity |
| `atr_ratio` | $\text{ATR}_{14} / \text{close}$ | Normalized volatility |
| `obv_zscore` | z-score of OBV | Volume momentum |
| `vwap_dev` | $(C_t - \text{VWAP}) / \text{VWAP}$ | Price vs fair value |
| `ret_zscore` | $(r_t - \bar{r}) / \sigma_r$ | Return extremity |

State decoding uses the **Viterbi algorithm** (globally optimal path).

States are labeled automatically by sorting on mean `ret_ann`:
- **Bull** → highest mean return state
- **Bear** → lowest mean return state
- **Crisis** → highest volatility state

---

## 📈 Performance Metrics

| Metric | Formula | Reference |
|--------|---------|-----------|
| Sharpe Ratio | $(\bar{r}_e / \sigma_e) \cdot \sqrt{365}$ | Lo (2002) |
| Sharpe t-stat | $SR \cdot \sqrt{T} / \sqrt{1 + SR^2/2}$ | Lo (2002) |
| Sortino Ratio | $(\bar{r} - MAR) / \sigma_d \cdot \sqrt{365}$ | Sortino & van der Meer (1991) |
| Calmar Ratio | $\text{CAGR} / \|\text{MDD}\|$ | Young (1991) |
| Omega Ratio | $\sum\max(r-L,0) / \sum\max(L-r,0)$ | Keating & Shadwick (2002) |
| Information Ratio | $(\bar{r}_p - \bar{r}_b) / \sigma_{r_p - r_b} \cdot \sqrt{365}$ | Grinold & Kahn (1994) |
| Kelly Criterion | $f^* = \bar{r} / \sigma^2_r \cdot 0.5$ | Kelly (1956) |
| CAGR | $(V_f/V_0)^{365/n_{\text{days}}} - 1$ | Standard |
| Profit Factor | $\sum\text{wins} / \|\sum\text{losses}\|$ | Industry standard |

---

## 🔬 Statistical Validation

### Lookahead Bias Check
Re-runs strategy on 30 randomly-sampled truncated data slices and verifies that signals at index $i$ are identical whether computed on the full series or on data up to $i$ only.

### Block Bootstrap Significance Test (Politis & Romano 1994)
```
H₀: Strategy Sharpe ≤ 0 (no real edge)

Method:
1. Compute real Sharpe ratio SR
2. Block-resample returns N=5000 times (block_size=21 days)
3. p-value = P(SR_bootstrap ≥ SR_real)

p-value < 0.05 → reject H₀ → strategy has real edge
```

### Permutation Test
Randomly shuffles trade order 5000 times to test whether the *sequence* of winning/losing trades has non-random structure.

---

## 🏗️ Mathematical Architecture

### Hidden Markov Model (Gaussian HMM)

**Observation model:**
$$p(\mathbf{x}_t \mid z_t = k) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

**Transition model:**
$$P(z_t = j \mid z_{t-1} = i) = A_{ij}$$

**E-step (Forward-Backward):**
$$\gamma_t(k) = P(z_t = k \mid \mathbf{X}) \propto \alpha_t(k) \cdot \beta_t(k)$$
$$\xi_t(i,j) = P(z_t=i, z_{t+1}=j \mid \mathbf{X}) \propto \alpha_t(i) \cdot A_{ij} \cdot b_j(\mathbf{x}_{t+1}) \cdot \beta_{t+1}(j)$$

**M-step (parameter updates):**
$$A_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)} \qquad \text{(Correct Baum-Welch)}$$

$$\boldsymbol{\mu}_k = \frac{\sum_t \gamma_t(k) \mathbf{x}_t}{\sum_t \gamma_t(k)}, \qquad \boldsymbol{\Sigma}_k = \frac{\sum_t \gamma_t(k) (\mathbf{x}_t - \boldsymbol{\mu}_k)(\mathbf{x}_t - \boldsymbol{\mu}_k)^\top}{\sum_t \gamma_t(k)}$$

**State selection via BIC (Schwarz 1978):**
$$\text{BIC}(k) = -2\ln\hat{L} + k_\text{params} \cdot \ln(n)$$

---

## 📚 Academic References

1. **Hamilton, J.D. (1989)** — "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle" — *Econometrica, 57(2)*
2. **Baum, L. et al. (1970)** — "A Maximization Technique in Statistical Estimation for Probabilistic Functions of Markov Chains" — *Ann. Math. Stat.*
3. **Viterbi, A. (1967)** — "Error Bounds for Convolutional Codes" — *IEEE Trans. Info. Theory*
4. **Lo, A.W. (2002)** — "The Statistics of Sharpe Ratios" — *Financial Analysts Journal, 58(4)*
5. **Sortino, F. & van der Meer, R. (1991)** — "Downside Risk" — *Journal of Portfolio Management*
6. **Kelly, J.L. (1956)** — "A New Interpretation of Information Rate" — *Bell System Technical Journal*
7. **Ang, A. & Bekaert, G. (2002)** — "International Asset Allocation with Regime Shifts" — *Review of Financial Studies*
8. **Ardia, D., Bluteau, K. & Rüede, M. (2019)** — "Regime Changes in Bitcoin GARCH Volatility" — *Finance Research Letters*
9. **Politis, D. & Romano, J. (1994)** — "The Stationary Bootstrap" — *JASA, 89(428)*
10. **Keating, C. & Shadwick, W. (2002)** — "A Universal Performance Measure" — *J. Performance Measurement*

---

## 📋 Data Format

Input CSV must contain:
```
datetime,open,high,low,close,volume
2018-01-01 05:30:00,13715.65,13818.55,12750.0,13380.0,8609.91
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `pandas-ta-classic` | ATR via technical analysis library |
| `matplotlib` | Static plots |
| `plotly` | Interactive candlestick charts |
| `streamlit` | Web dashboard |
| `scipy` | t-distribution for Sharpe significance |
| `seaborn` | Statistical plots |
| `pytest` | Unit test runner |

---

## 🔍 Signals Encoding

| Signal | Meaning |
|--------|---------|
| `0` | HOLD |
| `1` | BUY (or close short) |
| `-1` | SELL (or close long) |
| `2` | REVERSE: Short → Long |
| `-2` | REVERSE: Long → Short |

---

## 📝 License

MIT License — open for research and educational use.

---

*RegimeShift | BTC/USD Regime Trading Framework | Built for Tanmay's Quant + AI/ML Journey | July 2026*
