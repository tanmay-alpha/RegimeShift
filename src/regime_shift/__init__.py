"""
RegimeShift — src package.

Sub-modules:
  data_loader     : BTC/multi-asset data loading + OHLCV validation
  features        : Single-asset BTC feature engineering (7 HMM features)
  regime_detector : Gaussian HMM with correct Baum-Welch EM + Viterbi
  strategy        : Regime-conditional volume-spike trading strategy
  backtest        : Walk-forward backtest with regime-conditioned optimization
  stats           : All quantitative metrics (Sharpe, Sortino, Calmar, Omega...)
  optimizer       : Mean-variance portfolio optimizer (projected gradient)
  monte_carlo     : Block bootstrap significance testing
  evaluate        : Bootstrap confidence intervals
  benchmarks      : Buy-and-hold, 60/40 benchmarks
"""
