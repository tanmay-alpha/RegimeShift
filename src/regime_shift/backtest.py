"""
Walk-forward backtest for RegimeShift strategy.

Key design:
- Expanding-window regime detection: at each rebalance date we fit
  the HMM on all features up to the END of the PREVIOUS trading day,
  then use the LAST predicted regime as today's signal.
- This avoids look-ahead bias: today's feature (which contains
  today's return) is never in the training window.
- Portfolio optimization is regime-conditioned using the
  PortfolioOptimizer from regime_shift.optimizer.
- Enhanced feature engineering: 54 standardized features including
  returns, volatility, momentum, tail risk, and cross-asset signals.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from .regime_detector import RegimeDetector
from .regime_signal import RegimeSignal
from .optimizer import PortfolioOptimizer

from backtester import BackTester, sign
import config


class WalkForwardBacktest:
    """Walk-forward backtest with HMM regime detection.

    Parameters
    ----------
    prices : DataFrame
        Multi-asset price history, one column per asset.
    tickers : list[str]
        Ordered list of asset names matching *prices* columns.
    rebalance_freq : str
        Pandas offset for rebalance cadence (e.g. "1M", "1W", "5D").
    window_size : int
        Rolling window for feature computation.
    n_regimes : int
        Number of HMM states.
    transaction_cost : float
        Proportional cost per trade (e.g. 0.0015 = 0.15%).
    """

    def __init__(
        self,
        prices,
        tickers,
        rebalance_freq="1M",
        window_size=252,
        n_regimes=3,
        transaction_cost=0.0015,
        regime_persistence=3,
    ):
        self.prices = prices
        self.tickers = tickers
        self.rebalance_freq = rebalance_freq
        self.window_size = window_size
        self.n_regimes = n_regimes
        self.transaction_cost = transaction_cost
        self.regime_persistence = regime_persistence

        self.detector = RegimeDetector(
            n_states=n_regimes,
            lookback=window_size,
            random_state=42,
        )
        self.optimizer = PortfolioOptimizer(n_assets=len(tickers))

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def compute_features(self, returns):
        """Build enhanced features for regime detection.

        Uses RegimeFeatureEngineer to compute 54 standardized features
        including returns, volatility, momentum, tail risk, and
        cross-asset signals. All features are z-scored using a rolling
        calibration window.

        Args:
            returns: DataFrame of daily returns

        Returns:
            DataFrame of standardized features
        """
        from .data_loader import compute_features
        return compute_features(returns, self.tickers, window=self.window_size)

    # ------------------------------------------------------------------
    # Main backtest loop
    # ------------------------------------------------------------------

    def run(self, returns, start_date=None, end_date=None):
        """Run the walk-forward backtest.

        Parameters
        ----------
        returns : DataFrame
            Daily returns, one column per ticker, datetime-indexed.
        start_date, end_date : str or Timestamp, optional
            Subset boundaries.

        Returns
        -------
        dict with keys: returns, weights, regimes, turnover, equity_curve
        """
        returns = returns.loc[start_date:end_date].copy()
        features = self.compute_features(returns)
        dates = returns.index

        # Rebalance schedule: first day of each month (or chosen freq)
        rebalance_dates = pd.date_range(
            start=dates[self.window_size],
            end=dates[-1],
            freq=self.rebalance_freq,
        )
        rebalance_dates = rebalance_dates[rebalance_dates.isin(dates)]

        n = len(dates)
        regimes = np.full(n, -1, dtype=int)
        weights_history = np.zeros((n, len(self.tickers)))
        turnover = np.zeros(n)
        equity = np.zeros(n)
        current_weights = np.ones(len(self.tickers)) / len(self.tickers)
        current_regime = 0  # default to Bull
        regime_counter = {}  # persistence counter per regime label
        capital = 1.0

        for day_idx in range(self.window_size, n):
            today_date = dates[day_idx]
            rebalance = today_date in rebalance_dates

            if rebalance:
                # ----------------------------------------------------------
                # LOOK-AHEAD BIAS FIX:
                # Train ONLY on features up to the END of the PREVIOUS
                # trading day (today_idx - 1). Today's feature row
                # includes today's return — it must NOT be in the fit set.
                # ----------------------------------------------------------
                train_features = features.iloc[:day_idx - 1]
                regime_series = self.detector.fit_predict(train_features)
                if not regime_series.empty:
                    pred_regime = regime_series.iloc[-1]
                else:
                    pred_regime = current_regime

                # -- Regime persistence filter --
                # Require N consecutive days in the same regime before
                # acting on it, to avoid spurious single-day flips.
                if pred_regime == current_regime:
                    regime_counter[pred_regime] = regime_counter.get(pred_regime, 0) + 1
                else:
                    regime_counter = {pred_regime: 1}
                if regime_counter.get(pred_regime, 0) >= self.regime_persistence:
                    current_regime = pred_regime
                else:
                    pred_regime = current_regime

                regimes[day_idx] = pred_regime

                # Regime-conditioned expected returns / covariance
                mu, cov = self._estimate_moments(returns, regimes, pred_regime, day_idx)

                new_weights = self.optimizer.solve(
                    mu,
                    cov,
                    lb=np.zeros(len(self.tickers)),
                    ub=np.ones(len(self.tickers)),
                    max_turnover=0.5,
                    current_weights=current_weights,
                )
                turnover[day_idx] = np.abs(new_weights - current_weights).sum()
                current_weights = new_weights
            else:
                regimes[day_idx] = current_regime

            weights_history[day_idx] = current_weights

            # Portfolio return for the day
            daily_ret = (current_weights * returns.iloc[day_idx].values).sum()
            if rebalance and turnover[day_idx] > 0:
                port_ret = daily_ret - turnover[day_idx] * self.transaction_cost
            else:
                port_ret = daily_ret
            capital *= 1.0 + port_ret
            equity[day_idx] = capital

        return {
            "returns": returns.iloc[self.window_size:].copy(),
            "weights": pd.DataFrame(
                weights_history[self.window_size:],
                index=dates[self.window_size:],
                columns=self.tickers,
            ),
            "regimes": pd.Series(
                regimes[self.window_size:],
                index=dates[self.window_size:],
                name="regime",
            ),
            "turnover": pd.Series(
                turnover[self.window_size:],
                index=dates[self.window_size:],
            ),
            "equity_curve": pd.Series(
                equity[self.window_size:],
                index=dates[self.window_size:],
                name="equity",
            ),
        }

    # ------------------------------------------------------------------
    # Moment estimation conditioned on regime
    # ------------------------------------------------------------------

    def _estimate_moments(self, returns, regimes, regime_id, today_idx):
        """Estimate regime-conditional expected returns and covariance.

        Falls back to equal-weight / equal-variance estimates if
        there are too few samples in the target regime.
        """
        regime_mask = regimes[:today_idx] == regime_id
        n_samples = regime_mask.sum()

        if n_samples < 10:
            mu = np.zeros(len(self.tickers))
            cov = np.eye(len(self.tickers)) * 0.0004
        else:
            regime_rets = returns.iloc[:today_idx][regime_mask]
            mu = regime_rets.mean().values
            cov = regime_rets.cov().values
            cov += 1e-4 * np.eye(len(self.tickers))

        return mu, cov

    # ------------------------------------------------------------------
    # Run from price data directly
    # ------------------------------------------------------------------

    def run_from_prices(self, prices, start_date=None, end_date=None):
        """Convenience wrapper: compute returns, then call run()."""
        returns = prices.pct_change().dropna()
        return self.run(returns, start_date, end_date)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def sharpe_ratio(self, equity_curve, rf_annual=0.0):
        """Annualised Sharpe ratio from an equity curve Series."""
        daily_returns = equity_curve.pct_change().dropna()
        if len(daily_returns) < 2:
            return 0.0
        rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
        excess = daily_returns - rf_daily
        std = excess.std()
        if std == 0:
            return 0.0
        return (excess.mean() / std) * np.sqrt(252)

    def regime_statistics(self, regime_series):
        """Return dict of regime quality metrics."""
        stats = {}
        for state_id in regime_series.unique():
            if state_id < 0:
                continue
            runs = regime_series.eq(state_id).astype(int)
            # Find consecutive runs
            changes = runs.diff().fillna(0).ne(0)
            run_starts = changes[changes].index
            run_lengths = []
            for i, start in enumerate(run_starts):
                end = run_starts[i + 1] if i + 1 < len(run_starts) else regime_series.index[-1]
                run_lengths.append(
                    regime_series.loc[start:end].eq(state_id).sum()
                )
            stats[f"regime_{state_id}"] = {
                "name": self.detector.get_state_name(state_id),
                "freq_pct": (regime_series == state_id).mean() * 100,
                "avg_duration_days": float(np.mean(run_lengths)) if run_lengths else 0,
                "max_duration_days": float(np.max(run_lengths)) if run_lengths else 0,
            }
        return stats

    def get_detector_metrics(self) -> dict:
        """Return diagnostic metrics from the regime detector."""
        return self.detector.get_regime_metrics()

    def plot_equity_and_regimes(self, result):
        """Plot equity curve with regime background shading."""
        from .visualize import plot_equity_and_regimes
        plot_equity_and_regimes(result)
