# Legacy Student-t HMM & Advanced Research Tools

This subfolder contains the custom Student-t Hidden Markov Model engine and 54-dimensional feature extraction system built during initial research iterations.

## Preserved Modules

- **`regime_detector.py`**: Custom implementation of Student-t HMM, including EM algorithm (Baum-Welch), log-gamma distribution estimation, Dirichlet transition priors, and Viterbi state decoding.
- **`regime_features.py`**: 54-feature pipeline generating rolling returns, volatility ratios, momentum signals, tail risk, and cross-asset correlations.
- **`transaction_costs.py`**: Advanced market-impact cost model based on Almgren-Chriss market micro-structure theory.
- **`monte_carlo.py`**: Circular block bootstrap and synthetic scenario generation.
- **`stats.py`**: Matrix conditioning and statistical helper functions.
- **`visualize.py`**: 15+ diagnostic charts for regime transitions, posteriors, and feature importance.

## Isolation Statement

> [!NOTE]
> These modules are preserved for historical reference on the `legacy/pre-cleanup` and `cleanup/iitb-2026-foundation` branches under `research/`. They are NOT part of the official `src/regime_shift` package and MUST NOT be imported in production/submission code.
