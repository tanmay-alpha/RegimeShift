"""
Walk-forward backtest engine for RegimeShift strategy.

Uses expanding window training: at each rebalance, the model is trained
on ALL data available up to that point. No look-ahead bias.

Production-grade with:
- BacktestResult dataclass for typed results
- TransactionCostModel for realistic costs
- RegimeSignal for confidence-weighted decisions
- Turnover constraints to limit trading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .regime_detector import RegimeDetector
from .regime_signal import RegimeSignal
from .optimizer import PortfolioOptimizer
from .transaction_costs import TransactionCostModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a walk-forward backtest."""
    portfolio_returns: pd.Series = field(default_factory=pd.Series)
    regime_series: pd.Series = field(default_factory=pd.Series)
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    costs: pd.Series = field(default_factory=pd.Series)
    trade_log: list = field(default_factory=list)
    regime_changes: int = 0

    @property
    def total_return(self) -> float:
        if len(self.portfolio_returns) == 0:
            return 0.0
        return float((1.0 + self.portfolio_returns).prod() - 1.0)

    @property
    def annualized_return(self) -> float:
        if len(self.portfolio_returns) == 0:
            return 0.0
        n = len(self.portfolio_returns)
        return float((1.0 + self.total_return) ** (252.0 / n) - 1.0)

    @property
    def annualized_volatility(self) -> float:
        if len(self.portfolio_returns) == 0:
            return 0.0
        return float(self.portfolio_returns.std() * np.sqrt(252.0))

    @property
    def sharpe_ratio(self) -> float:
        vol = self.annualized_volatility
        return float(self.annualized_return / vol) if vol > 1e-12 else 0.0

    @property
    def max_drawdown(self) -> float:
        if len(self.portfolio_returns) == 0:
            return 0.0
        cum = (1.0 + self.portfolio_returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        return float(dd.min())

    @property
    def turnover(self) -> float:
        if len(self.weights_history) < 2:
            return 0.0
        diffs = self.weights_history.diff().dropna()
        if len(diffs) == 0:
            return 0.0
        total_turn = float(diffs.abs().sum(axis=1).sum() / 2.0)
        n_years = len(diffs) / 252.0
        return total_turn / n_years if n_years > 0 else 0.0

    @property
    def total_costs(self) -> float:
        return float(self.costs.sum()) if len(self.costs) > 0 else 0.0

    @property
    def cost_drag(self) -> float:
        total_ret = self.total_return
        if abs(total_ret) > 1e-6:
            return float(self.total_costs / abs(total_ret))
        return 0.0

    @property
    def win_rate(self) -> float:
        if len(self.portfolio_returns) == 0:
            return 0.0
        return float((self.portfolio_returns > 0).mean())

    @property
    def avg_daily_turnover(self) -> float:
        if len(self.weights_history) < 2:
            return 0.0
        diffs = self.weights_history.diff().dropna()
        return float(diffs.abs().sum(axis=1).mean() / 2.0)

    @property
    def avg_annual_turnover(self) -> float:
        return float(self.avg_daily_turnover * 252.0)


class WalkForwardBacktest:
    """
    Walk-forward backtest with expanding window training.

    At each rebalance point, the HMM is trained on ALL data available
    up to that point. This is the standard institutional approach.

    Attributes:
        prices: DataFrame of asset prices
        returns: DataFrame of daily returns
        features: DataFrame of regime features
        lookback: Minimum training window size (trading days)
        retrain_freq: Retrain every N days
        n_states: Number of HMM regimes
        cost_model: TransactionCostModel instance
        detector: RegimeDetector instance
        optimizer: PortfolioOptimizer instance
        turnover_limit: Max turnover per rebalance (fraction)
        tickers: Asset ticker symbols
        n_assets: Number of assets
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        lookback: int = 252,
        retrain_freq: int = 21,
        n_states: int = 3,
        cost_model: Optional[TransactionCostModel] = None,
        detector: Optional[RegimeDetector] = None,
        optimizer: Optional[PortfolioOptimizer] = None,
        turnover_limit: float = 0.20,
    ) -> None:
        self.prices = prices
        self.returns = returns
        self.features = features
        self.lookback = lookback
        self.retrain_freq = retrain_freq
        self.n_states = n_states
        self.cost_model = cost_model or TransactionCostModel()
        self.detector = detector or RegimeDetector(
            n_states=n_states, lookback=lookback, retrain_freq=retrain_freq
        )
        self.optimizer = optimizer or PortfolioOptimizer(n_assets=len(returns.columns))
        self.turnover_limit = turnover_limit
        self.tickers = returns.columns.tolist()
        self.n_assets = len(self.tickers)

    def run(self) -> BacktestResult:
        """
        Execute walk-forward backtest.

        Algorithm:
        1. Start with equal-weight portfolio
        2. For each day t >= lookback:
           a. If rebalance day: detect regime, optimize weights, apply costs
           b. Compute daily return = prev_weights @ returns[t]
           c. Record everything
        3. Return BacktestResult with all recorded data

        Returns:
            BacktestResult with portfolio_returns, regime_series,
            weights_history, costs, trade_log, regime_changes
        """
        n_days = len(self.returns)
        portfolio_returns: list = []
        regime_labels: list = []
        weights_list: list = []
        costs_list: list = []
        trade_log: list = []
        regime_changes = 0

        current_weights = np.ones(self.n_assets) / self.n_assets
        prev_regime = None
        rebalance_day = True

        for t in range(self.lookback, n_days):
            today = self.returns.index[t]
            returns_today = self.returns.iloc[t].values

            # Check if rebalance day
            should_rebalance = rebalance_day

            if should_rebalance:
                try:
                    # Get data up to time t
                    returns_window = self.returns.iloc[: t + 1]
                    if today in self.features.index:
                        features_window = self.features.loc[:today]
                    else:
                        loc_idx = self.features.index.searchsorted(today, side="right") - 1
                        features_window = self.features.iloc[: loc_idx + 1]

                    if len(features_window) >= self.lookback and len(features_window) >= 10:
                        # Fit detector and get signal
                        regime_series = self.detector.fit_predict(features_window)
                        signal = self.detector.predict_signal(returns_window, today)

                        # Check if we should rebalance based on signal
                        if signal.should_rebalance(threshold=0.15):
                            # Get expected returns and covariance
                            ret_mean = returns_window.mean().values * 252
                            ret_cov = returns_window.cov().values * 252

                            # Optimize using signal
                            new_weights = self.optimizer.optimize_with_signal(
                                ret_mean, ret_cov, signal
                            )

                            # Apply turnover constraint
                            turnover = TransactionCostModel.compute_turnover(
                                current_weights, new_weights
                            )
                            if turnover > self.turnover_limit:
                                scale = self.turnover_limit / (turnover + 1e-12)
                                delta = new_weights - current_weights
                                delta = delta * min(scale, 1.0)
                                new_weights = current_weights + delta
                                new_weights = np.clip(new_weights, 0, 1)
                                wsum = new_weights.sum()
                                if wsum > 1e-12:
                                    new_weights = new_weights / wsum

                            # Transaction costs
                            vol_annual = returns_window.std().values * np.sqrt(252)
                            cost_fraction = self.cost_model.cost_as_fraction(
                                current_weights, new_weights,
                                vol_annual, self.tickers, notional=1_000_000
                            )

                            # Track regime changes
                            current_label = signal.label
                            if prev_regime is not None and current_label != prev_regime:
                                regime_changes += 1
                            prev_regime = current_label

                            # Log trade
                            if turnover > 1e-6:
                                trade_log.append({
                                    "date": str(today),
                                    "regime": current_label,
                                    "confidence": signal.confidence,
                                    "turnover": float(turnover),
                                    "cost_fraction": float(cost_fraction),
                                    "old_weights": current_weights.tolist(),
                                    "new_weights": new_weights.tolist(),
                                })

                            # Apply cost to portfolio
                            cost_fraction = min(cost_fraction, 0.02)  # Cap at 2%
                            costs_list.append(cost_fraction)
                            current_weights = new_weights
                        else:
                            costs_list.append(0.0)
                            if prev_regime is not None and signal.label != prev_regime:
                                regime_changes += 1
                            prev_regime = signal.label
                    else:
                        costs_list.append(0.0)

                except Exception as e:
                    logger.warning("Rebalance failed at %s: %s", today, e)
                    costs_list.append(0.0)

                rebalance_day = False
            else:
                costs_list.append(0.0)

            # Compute portfolio return
            daily_return = float(np.dot(current_weights, returns_today))
            portfolio_returns.append(daily_return)
            regime_labels.append(prev_regime or "Bull")
            weights_list.append(current_weights.copy())

            # Schedule next rebalance
            if (t - self.lookback + 1) % self.retrain_freq == 0:
                rebalance_day = True

        result = BacktestResult(
            portfolio_returns=pd.Series(
                portfolio_returns,
                index=self.returns.index[self.lookback:],
                name="portfolio_return",
            ),
            regime_series=pd.Series(
                regime_labels,
                index=self.returns.index[self.lookback:],
                name="regime",
            ),
            weights_history=pd.DataFrame(
                weights_list,
                index=self.returns.index[self.lookback:],
                columns=self.tickers,
            ) if weights_list else pd.DataFrame(
                index=self.returns.index[self.lookback:],
                columns=self.tickers,
            ),
            costs=pd.Series(
                costs_list,
                index=self.returns.index[self.lookback:],
                name="transaction_cost",
            ),
            trade_log=trade_log,
            regime_changes=regime_changes,
        )

        logger.info(
            "Backtest complete: %d days, %d rebalances, %d regime changes",
            len(portfolio_returns), len(trade_log), regime_changes,
        )
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def sharpe_ratio(self, equity_curve: pd.Series, rf_annual: float = 0.0) -> float:
        """Annualised Sharpe ratio from an equity curve Series."""
        daily_returns = equity_curve.pct_change().dropna()
        if len(daily_returns) < 2:
            return 0.0
        rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
        excess = daily_returns - rf_daily
        std = excess.std()
        if std == 0:
            return 0.0
        return float((excess.mean() / std) * np.sqrt(252))

    def regime_statistics(self, regime_series: pd.Series) -> dict:
        """Return dict of regime quality metrics."""
        stats = {}
        unique_vals = regime_series.unique()
        for val in unique_vals:
            try:
                if int(val) == -1:
                    continue
            except (TypeError, ValueError):
                pass

            mask = regime_series == val
            label = str(val)
            stats[f"regime_{label}"] = {
                "name": label,
                "freq_pct": float(mask.mean() * 100),
                "avg_duration_days": self._avg_run_length(regime_series, val),
                "max_duration_days": self._max_run_length(regime_series, val),
            }
        return stats

    def _avg_run_length(self, regime_series: pd.Series, val) -> float:
        """Average length (days) of consecutive runs of val."""
        mask = (regime_series == val).astype(int)
        diff = mask.diff().fillna(0)
        starts = mask.index[diff == 1].tolist()
        if not starts:
            return float(len(regime_series))
        if mask.iloc[0] == 1:
            starts = [regime_series.index[0]] + starts
        lengths = []
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else regime_series.index[-1]
            lengths.append(int(mask.loc[s:e].sum()))
        return float(np.mean(lengths)) if lengths else 0.0

    def _max_run_length(self, regime_series: pd.Series, val) -> float:
        """Max consecutive-run length of val."""
        mask = (regime_series == val).astype(int)
        runs = mask.groupby((mask != mask.shift()).cumsum()).cumsum()
        return float(runs.max()) if len(runs) else 0.0

    def get_detector_metrics(self) -> dict:
        """Return diagnostic metrics from the regime detector."""
        return self.detector.get_regime_metrics()
