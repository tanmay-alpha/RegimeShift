# RegimeShift 📈

> **Institutional-grade quant research framework** for adaptive portfolio allocation using **Student-t Hidden Markov Model (HMM) regime detection** combined with **walk-forward backtesting**, **bootstrap validation**, and **regime-conditioned portfolio optimization**.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-72%2B-passing-green.svg)](tests/)

---

## 🧠 Core Thesis

**RegimeShift** detects latent market regimes (Bull / Bear / Crisis) in multi-asset price data using a **Student-t HMM** that correctly models fat tails in financial returns. The system then applies **regime-conditioned portfolio optimization** to dynamically adjust asset allocation:

| Regime | Action | Rationale |
|--------|--------|-----------|
| **Bull** | Overweight equity, underweight bonds | Trend-following in uptrend |
| **Bear** | Overweight bonds/gold, underweight equity | Defensive positioning |
| **Crisis** | 50% reduction in all positions | Volatility protection |

**Why Student-t HMM?** Financial returns exhibit fat tails (kurtosis > 3) and skewness. Gaussian HMM systematically underestimates tail events — crisis observations get too-low likelihood, causing the model to misclassify or miss crises entirely. Student-t emissions with ν=4-6 fix this by modeling heavier tails.

---

## 🏗️ Mathematical Architecture

### Student-t Hidden Markov Model

**Observation model (replaces Gaussian):**

$$p(\mathbf{x}_t \mid z_t = k) = \mathcal{T}_\nu(\mathbf{x}_t; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

$$= \frac{\Gamma((\nu+2)/2)}{\Gamma(\nu/2) \sqrt{\nu\pi|\boldsymbol{\Sigma}_k|}} \left[1 + \frac{1}{\nu}(\mathbf{x}_t - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1}(\mathbf{x}_t - \boldsymbol{\mu}_k)\right]^{-(\nu+2)/2}$$

**Log-density (numerically stable):**

$$\log p(\mathbf{x} \mid z=k) = -\frac{1}{2}\left[\log|\boldsymbol{\Sigma}_k| + d\log(\nu\pi) + \log\Gamma\left(\frac{\nu+2}{2}\right) - \log\Gamma\left(\frac{\nu}{2}\right) + \frac{\nu+d}{2}\log\left(1 + \frac{\delta_k(\mathbf{x})}{\nu}\right)\right]$$

where $\delta_k(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1}(\mathbf{x} - \boldsymbol{\mu}_k)$ is the Mahalanobis distance.

**EM Algorithm (Baum-Welch with Student-t):**

**E-step:**
$$\gamma_t(k) = P(z_t = k \mid \mathbf{X}) = \frac{\alpha_t(k) \cdot \mathcal{T}_\nu(\mathbf{x}_t \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \alpha_t(j) \cdot \mathcal{T}_\nu(\mathbf{x}_t \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**M-step:**
$$\boldsymbol{\mu}_k = \frac{\sum_t \gamma_t(k)\mathbf{x}_t}{\sum_t \gamma_t(k)}, \quad \boldsymbol{\Sigma}_k = \frac{1}{\nu+d}\frac{\sum_t \gamma_t(k)(\mathbf{x}_t - \boldsymbol{\mu}_k)(\mathbf{x}_t - \boldsymbol{\mu}_k)^\top}{\sum_t \gamma_t(k)} + \epsilon\mathbf{I}$$

**Dirichlet Prior on Transition Matrix:**

$$A_{kk} \sim \text{Dir}(\alpha_{kk}=50), \quad A_{kj} \sim \text{Dir}(\alpha_{kj}=1) \text{ for } j \neq k$$

Expected regime duration: $E[D] = 1/(1 - A_{kk}) \approx 50$ days (realistic for market regimes).

### Regime Confidence Scoring

Outputs **posterior probabilities** (not just hard labels):

$$P(z_t = k \mid \mathbf{x}_{1:T}) = \frac{\text{forward}_t(k) \cdot \text{backward}_t(k)}{p(\mathbf{X})}$$

Quality metrics:
- **Silhouette Score**: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$
  - $> 0.5$: good regime separation
  - $< 0.2$: regimes poorly separated — flag as warning

---

## 📁 Project Structure

```
RegimeShift/
├── run_backtest.py                 ← CLI entry point for backtests
├── requirements.txt
│
├── src/regime_shift/
│   ├── __init__.py                 ← Package exports
│   ├── regime_features.py          ← 54-feature engineering (returns, vol, momentum, tail, cross-asset)
│   ├── regime_detector.py          ← Student-t HMM (Baum-Welch EM + Viterbi + Dirichlet prior)
│   ├── regime_signal.py            ← RegimeSignal dataclass (label, confidence, posteriors, duration)
│   ├── data_loader.py              ← Price loading + feature computation
│   ├── optimizer.py                ← Projected gradient descent portfolio optimizer
│   ├── backtest.py                 ← Walk-forward backtest engine
│   ├── evaluate.py                 ← Performance metrics + bootstrap CIs + regime metrics
│   ├── transaction_costs.py        ← Almgren-Chriss market impact model
│   ├── benchmarks.py               ← BuyAndHold, EqualWeight, RiskParity, Momentum
│   └── visualize.py                ← 15+ production-quality plots (regimes, confidence, silhouette, etc.)
│
├── tests/
│   ├── test_regime_detector.py     ← 10 core HMM tests
│   ├── test_phase3.py              ← 21 robustness / visualization tests (68 total passing)
│   ├── test_features.py            ← Feature computation tests
│   ├── test_stats.py               ← Metric formula tests
│   ├── test_strategy.py            ← Signal encoding tests
│   ├── test_hmm.py                 ← HMM convergence tests
│   └── test_backtest.py            ← Backtest engine tests
│
└── notebooks/
    ├── generate_notebook.py        ← Jupyter notebook generator
    └── analysis.ipynb               ← Full analysis pipeline (10 cells)
```

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run full backtest pipeline
python run_backtest.py

# Run with simulated data (no CSV required)
python run_backtest.py --simulated

# Auto-select n_states via BIC
python run_backtest.py --select-nstates

# Generate Jupyter notebook
python notebooks/generate_notebook.py

# Run all tests
pytest tests/ -v

# Run Phase 3 robustness tests only
pytest tests/test_phase3.py -v

# Quick smoke test
python -c "from src.regime_shift import RegimeDetector; print('OK')"
```

---

## 📊 Feature Engineering (54 Dimensions)

### Per-Asset Features (3 assets × 17 features = 51)

| Group | Feature | Formula | Window |
|-------|---------|---------|--------|
| **Returns** | ret_1m, ret_3m, ret_6m, ret_1y | mean(daily_ret, w) × 252 | 21/63/126/252d |
| **Volatility** | vol_1m, vol_3m, vol_of_vol, vol_ratio | std, std-of-vol, vol_ratio | 21/63/21/21d |
| **Momentum** | mom_3m, mom_6m, rsi_14 | price momentum, Wilder RSI | 63/126/14d |
| **Tail Risk** | skewness, kurtosis, max_dd, var_95 | rolling skew, excess kurtosis, drawdown, VaR | 63d |

### Cross-Asset Features (9)

| Feature | Formula |
|---------|---------|
| eq_gold_corr | rolling_corr(equity, gold, 63d) |
| eq_bond_corr | rolling_corr(equity, bonds, 63d) |
| gold_bond_corr | rolling_corr(gold, bonds, 63d) |
| eq_bond_spread | rolling mean(equity_ret - bond_ret) × 252 |
| gold_eq_ratio | rolling mean(gold_ret - equity_ret) × 252 |
| momentum_spread | momentum_3m(equity) - momentum_3m(bonds) |
| vol_spread | vol_1m(equity) - vol_1m(bonds) |
| corr_regime | mean(eq_gold_corr + eq_bond_corr, 21d) |
| skew_spread | skewness(equity) - skewness(gold) |

All features are **z-scored** using rolling calibration windows to prevent look-ahead bias.

---

## 📈 Performance Metrics

| Metric | Formula | Reference |
|--------|---------|-----------|
| Sharpe Ratio | $(\bar{r}_e / \sigma_e) \cdot \sqrt{252}$ | Lo |
| Sortino Ratio | $(\bar{r} - MAR) / \sigma_d \cdot \sqrt{252}$ | Sortino & van der Meer |
| Calmar Ratio | $\text{CAGR} / \|\text{MDD}\|$ | Young |
| Omega Ratio | $\sum\max(r-L,0) / \sum\max(L-r,0)$ | Keating & Shadwick |
| Information Ratio | $(\bar{r}_p - \bar{r}_b) / \sigma_{r_p - r_b} \cdot \sqrt{252}$ | Grinold & Kahn |
| Kelly Criterion | $f^* = \bar{r} / \sigma^2_r \cdot 0.5$ | Kelly |
| CAGR | $(V_f/V_0)^{252/n} - 1$ | Standard |

---

## 🔬 Statistical Validation

### Lookahead Bias Prevention
All features use **rolling windows** with `min_periods=max(int(window × 0.7), 10)`. Standardization uses only data available at time $t$.

### Bootstrap Confidence Intervals (Politis & Romano 1994)
Block-bootstrap with 21-day blocks, 500+ iterations. 95% confidence intervals computed for all key metrics.

### Regime Quality Metrics
- **Silhouette Score**: validates regime separation quality
- **Transition Matrix**: validates regime persistence (self-transition > 0.5)
- **Regime Duration**: expected duration = 1/(1-A_kk) ≈ 50 days

---

## 📚 Academic References

1. **Hamilton, J.D.** — "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle" — *Econometrica, 57(2)*
2. **Baum, L. et al.** — "A Maximization Technique in Statistical Estimation for Probabilistic Functions of Markov Chains" — *Ann. Math. Stat.*
3. **Viterbi, A.** — "Error Bounds for Convolutional Codes" — *IEEE Trans. Info. Theory*
4. **Lo, A.W.** — "The Statistics of Sharpe Ratios" — *Financial Analysts Journal, 58(4)*
5. **Politis, D. & Romano, J.** — "The Stationary Bootstrap" — *JASA, 89(428)*
6. **Almgren, R. & Chriss, N.** — "Optimal Execution of Portfolio Transactions" — *J. Risk*
7. **Sortino, F. & van der Meer, R.** — "Downside Risk" — *Journal of Portfolio Management*
8. **Kelly, J.L.** — "A New Interpretation of Information Rate" — *Bell System Technical Journal*
9. **Ang, A. & Bekaert, G.** — "International Asset Allocation with Regime Shifts" — *Review of Financial Studies*
10. **Schwarz, G.** — "Estimating the Dimension of a Model" — *Annals of Statistics*

---

## 🔑 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Emissions | Student-t (ν=5 default) | Handles fat tails in financial returns |
| Feature standardization | Rolling z-score | Prevents look-ahead bias |
| Transition prior | Dirichlet(α_kk=50, α_kj=1) | Enforces realistic regime persistence |
| Regime decoding | Viterbi (hard labels) + forward-backward (soft posteriors) | Hard labels for trading, posteriors for confidence |
| Portfolio optimization | Projected gradient descent | Pure numpy, no scipy dependency |
| Transaction costs | Almgren-Chriss market impact | Production-grade cost model |
| Validation | Block bootstrap (21d blocks) | Preserves autocorrelation structure |

---

## ⚠️ Disclaimer

This is a **research framework** for educational purposes. Not financial advice. Past performance does not guarantee future results. Always paper-trade before live deployment.

---

## 📝 License

MIT License — open for research and educational use.

---

*RegimeShift | Multi-Asset Regime Trading Framework | Institutional-grade implementation | July 2026*
