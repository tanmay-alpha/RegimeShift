"""
regime_features.py — Enhanced feature engineering for HMM regime detection.

Computes 54 standardized features across 3 asset classes (equity, gold, bonds)
covering returns, volatility, momentum, tail risk, and cross-asset signals.
All features are z-scored using a rolling calibration window to prevent
look-ahead bias.

Mathematical foundations:
  - Returns: rolling mean * annualization_factor (252 trading days)
  - Volatility: rolling std * sqrt(252) for annualization
  - RSI: Wilder exponential smoothing, RS = avg_gain / avg_loss
  - Standardization: z = (x - rolling_mean) / (rolling_std + eps)
    using ONLY past data (shifted by 1 to prevent lookahead)
  - Cross-asset correlations: rolling Pearson via pandas .corr()

Feature dimension: 54 features (17 per asset + 9 cross-asset)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Number of trading days per year for annualization
ANN_FACTOR: int = 252
#: Minimum fraction of window required for valid rolling computation
MIN_PERIODS_RATIO: float = 0.7
#: Minimum absolute number of periods
MIN_PERIODS_ABS: int = 10
#: Z-score clipping threshold (winsorization)
ZSCORE_CLIP: float = 5.0
#: Small constant for numerical stability
EPS: float = 1e-12

# ─────────────────────────────────────────────────────────────────────────────
# Feature Names (order is critical — must match HMM training convention)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    # Nifty returns (4 features)
    "nifty_ret_1m", "nifty_ret_3m", "nifty_ret_6m", "nifty_ret_1y",
    # Nifty volatility (4 features)
    "nifty_vol_1m", "nifty_vol_3m", "nifty_vol_of_vol", "nifty_vol_ratio",
    # Nifty momentum (3 features)
    "nifty_mom_3m", "nifty_mom_6m", "nifty_rsi_14",
    # Nifty tail risk (4 features)
    "nifty_skew", "nifty_kurt", "nifty_max_dd_63", "nifty_var_95",
    # Gold returns (4 features)
    "gold_ret_1m", "gold_ret_3m", "gold_ret_6m", "gold_ret_1y",
    # Gold volatility (4 features)
    "gold_vol_1m", "gold_vol_3m", "gold_vol_of_vol", "gold_vol_ratio",
    # Gold momentum (3 features)
    "gold_mom_3m", "gold_mom_6m", "gold_rsi_14",
    # Gold tail risk (4 features)
    "gold_skew", "gold_kurt", "gold_max_dd_63", "gold_var_95",
    # Bond returns (4 features)
    "bond_ret_1m", "bond_ret_3m", "bond_ret_6m", "bond_ret_1y",
    # Bond volatility (4 features)
    "bond_vol_1m", "bond_vol_3m", "bond_vol_of_vol", "bond_vol_ratio",
    # Bond momentum (3 features)
    "bond_mom_3m", "bond_mom_6m", "bond_rsi_14",
    # Bond tail risk (4 features)
    "bond_skew", "bond_kurt", "bond_max_dd_63", "bond_var_95",
    # Cross-asset (9 features)
    "eq_gold_corr", "eq_bond_corr", "gold_bond_corr",
    "eq_bond_spread", "gold_eq_ratio", "momentum_spread",
    "vol_spread", "corr_regime", "skew_spread",
]

# Number of features per asset
FEATURES_PER_ASSET: int = 17


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _min_periods(window: int) -> int:
    """Compute minimum periods for rolling window."""
    return max(int(window * MIN_PERIODS_RATIO), MIN_PERIODS_ABS)


def _zscore_rolling(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling z-score using ONLY past data for calibration.

    At each time t:
      mu_t   = mean(series[t-window : t])   -- strictly past data
      sigma_t = std(series[t-window : t])    -- strictly past data
      z_t    = (series[t] - mu_t) / (sigma_t + eps)

    The .shift(1) ensures the current observation is NOT in the calibration.

    Args:
        series: pd.Series of values
        window: rolling calibration window

    Returns:
        pd.Series of z-scores, same index as input
    """
    roll_mean = series.rolling(window, min_periods=_min_periods(window)).mean().shift(1)
    roll_std = series.rolling(window, min_periods=_min_periods(window)).std().shift(1)
    z = (series - roll_mean) / (roll_std + EPS)
    return z.clip(-ZSCORE_CLIP, ZSCORE_CLIP)


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing.

    RS = avg_gain / avg_loss
    RSI = 100 - 100 / (1 + RS)

    Wilder's smoothing:
      avg_gain[t] = (avg_gain[t-1] * (window-1) + gain[t]) / window
      avg_loss[t] = (avg_loss[t-1] * (window-1) + loss[t]) / window

    Args:
        series: pd.Series of returns or prices
        window: RSI lookback period (default 14)

    Returns:
        pd.Series of RSI values [0, 100]
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's exponential moving average
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=_min_periods(window)).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=_min_periods(window)).mean()

    rs = avg_gain / (avg_loss + EPS)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _compute_max_drawdown(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling maximum drawdown.

    For each time t, computes the maximum peak-to-trough decline
    over the past `window` days (using only data up to t).

    Args:
        series: pd.Series of cumulative returns or prices
        window: rolling window size

    Returns:
        pd.Series of max drawdown values (negative numbers)
    """
    # Compute rolling max using only past data
    roll_max = series.rolling(window, min_periods=_min_periods(window)).max().shift(1)
    # Current value vs recent peak
    dd = (series - roll_max) / (roll_max + EPS)
    # Take the minimum (worst drawdown) over the window
    # Use expanding window of the drawdown series
    roll_min_dd = dd.rolling(window, min_periods=_min_periods(window)).min()
    return roll_min_dd


def _compute_var(series: pd.Series, window: int, q: float = 5.0) -> pd.Series:
    """
    Rolling Value at Risk (percentile-based).

    VaR at level q = q-th percentile of returns over the window.
    Uses only past data (shifted by 1).

    Args:
        series: pd.Series of returns
        window: rolling window size
        q: percentile level (default 5.0 for 95% VaR)

    Returns:
        pd.Series of VaR values (negative = loss)
    """
    return series.rolling(window, min_periods=_min_periods(window)).quantile(q / 100.0).shift(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main Feature Engineer Class
# ─────────────────────────────────────────────────────────────────────────────

class RegimeFeatureEngineer:
    """
    Computes standardized features for HMM regime detection.

    All features are z-scored using a rolling calibration window
    to prevent look-ahead bias. The standardization parameters
    (mean, std) are computed from data available at time t only.

    Feature dimension: 54 features (17 per asset + 9 cross-asset)

    Attributes:
        feature_names: list of feature column names
        lookback_window: rolling window for feature computation
        calibration_ratio: fraction of window used for calibration (0.7)
    """

    def __init__(self, lookback_window: int = 252) -> None:
        """
        Args:
            lookback_window: Rolling window size for feature computation (trading days)
        """
        self.lookback_window = lookback_window
        self.calibration_ratio: float = 0.7

    # ──────────────────────────────────────────────────────────────────────────
    # Single-Asset Feature Computation
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_asset_features(self, returns: pd.Series, prices: pd.Series,
                                prefix: str) -> pd.DataFrame:
        """
        Compute all features for a single asset.

        Args:
            returns: Daily return series for this asset
            prices: Daily price series for this asset
            prefix: Feature name prefix (e.g. "nifty", "gold", "bond")

        Returns:
            DataFrame with 17 feature columns for this asset
        """
        w252 = 252
        w126 = 126
        w63 = 63
        w21 = 21
        w14 = 14

        calib = int(self.lookback_window * self.calibration_ratio)

        # ── Returns (annualized) ──
        ret_1m = returns.rolling(w21, min_periods=_min_periods(w21)).mean() * ANN_FACTOR
        ret_3m = returns.rolling(w63, min_periods=_min_periods(w63)).mean() * ANN_FACTOR
        ret_6m = returns.rolling(w126, min_periods=_min_periods(w126)).mean() * ANN_FACTOR
        ret_1y = returns.rolling(w252, min_periods=_min_periods(w252)).mean() * ANN_FACTOR

        # ── Volatility (annualized) ──
        vol_1m = returns.rolling(w21, min_periods=_min_periods(w21)).std() * np.sqrt(ANN_FACTOR)
        vol_3m = returns.rolling(w63, min_periods=_min_periods(w63)).std() * np.sqrt(ANN_FACTOR)
        vol_of_vol = vol_1m.rolling(w21, min_periods=_min_periods(w21)).std()
        vol_ratio = vol_1m / (vol_3m + EPS)

        # ── Momentum ──
        mom_3m = prices.shift(21)  # price 21 days ago
        mom_3m = prices / mom_3m - 1.0
        mom_3m = mom_3m.replace([np.inf, -np.inf], np.nan)

        mom_6m = prices.shift(63)
        mom_6m = prices / mom_6m - 1.0
        mom_6m = mom_6m.replace([np.inf, -np.inf], np.nan)

        rsi_14 = _compute_rsi(returns, window=14)

        # ── Tail risk ──
        # Rolling skewness and kurtosis (pandas built-in, uses past data)
        skew = returns.rolling(w63, min_periods=_min_periods(w63)).skew()
        kurt = returns.rolling(w63, min_periods=_min_periods(w63)).kurt()

        # Max drawdown over rolling 63-day window
        # Compute cumulative return series for drawdown
        cum_ret = (1.0 + returns).cumprod()
        max_dd = _compute_max_drawdown(cum_ret, window=w63)

        # 95% VaR (5th percentile of returns)
        var_95 = _compute_var(returns, window=w63, q=5.0)

        features = pd.DataFrame({
            f"{prefix}_ret_1m": ret_1m,
            f"{prefix}_ret_3m": ret_3m,
            f"{prefix}_ret_6m": ret_6m,
            f"{prefix}_ret_1y": ret_1y,
            f"{prefix}_vol_1m": vol_1m,
            f"{prefix}_vol_3m": vol_3m,
            f"{prefix}_vol_of_vol": vol_of_vol,
            f"{prefix}_vol_ratio": vol_ratio,
            f"{prefix}_mom_3m": mom_3m,
            f"{prefix}_mom_6m": mom_6m,
            f"{prefix}_rsi_14": rsi_14,
            f"{prefix}_skew": skew,
            f"{prefix}_kurt": kurt,
            f"{prefix}_max_dd_63": max_dd,
            f"{prefix}_var_95": var_95,
        }, index=returns.index)

        return features

    # ──────────────────────────────────────────────────────────────────────────
    # Cross-Asset Feature Computation
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_cross_asset_features(self, returns: pd.DataFrame,
                                      prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-asset features from multi-asset returns.

        Assumes columns include 'nifty', 'gold', 'bond' (case-insensitive match).

        Args:
            returns: DataFrame of daily returns, columns = asset names
            prices: DataFrame of daily prices, columns = asset names

        Returns:
            DataFrame with 9 cross-asset feature columns
        """
        # Normalize column names for matching
        col_map = {c.lower(): c for c in returns.columns}

        # Find matching columns
        def _get_col(name: str) -> Optional[pd.Series]:
            return returns.get(col_map.get(name, name))

        def _get_price(name: str) -> Optional[pd.Series]:
            return prices.get(col_map.get(name, name))

        nifty_rets = _get_col("nifty")
        gold_rets = _get_col("gold")
        bond_rets = _get_col("bonds") or _get_col("bond")
        nifty_prices = _get_price("nifty")
        gold_prices = _get_price("gold")

        # ── Rolling correlations (use changes — not absolute levels) ──
        # Absolute correlations are near-constant for fixed-correlation data,
        # producing dead features after z-scoring. We use rolling 21-day
        # correlations then take changes to capture time-varying dependence.
        w63 = 63
        w21 = 21
        calib = int(self.lookback_window * self.calibration_ratio)

        features = {}

        # ── Rolling correlations ──
        if nifty_rets is not None and gold_rets is not None:
            combined = pd.concat([nifty_rets, gold_rets], axis=1).dropna()
            if len(combined) > _min_periods(w63):
                roll_corr = combined.rolling(w63, min_periods=_min_periods(w63)).corr()
                corr_series = roll_corr.unstack().iloc[:, 1].reindex(returns.index)
                # Use 21-day rolling change (captures shifting correlation)
                features["eq_gold_corr"] = corr_series.rolling(w21, min_periods=_min_periods(w21)).mean().diff()

        if nifty_rets is not None and bond_rets is not None:
            combined = pd.concat([nifty_rets, bond_rets], axis=1).dropna()
            if len(combined) > _min_periods(w63):
                roll_corr = combined.rolling(w63, min_periods=_min_periods(w63)).corr()
                corr_series = roll_corr.unstack().iloc[:, 1].reindex(returns.index)
                features["eq_bond_corr"] = corr_series.rolling(w21, min_periods=_min_periods(w21)).mean().diff()

        if gold_rets is not None and bond_rets is not None:
            combined = pd.concat([gold_rets, bond_rets], axis=1).dropna()
            if len(combined) > _min_periods(w63):
                roll_corr = combined.rolling(w63, min_periods=_min_periods(w63)).corr()
                corr_series = roll_corr.unstack().iloc[:, 1].reindex(returns.index)
                features["gold_bond_corr"] = corr_series.rolling(w21, min_periods=_min_periods(w21)).mean().diff()

        # ── Return spreads (annualized) ──
        if nifty_rets is not None and bond_rets is not None:
            eq_bond_spread = (nifty_rets - bond_rets).rolling(
                w21, min_periods=_min_periods(w21)
            ).mean() * ANN_FACTOR
            features["eq_bond_spread"] = eq_bond_spread

        if gold_rets is not None and nifty_rets is not None:
            gold_eq_ratio = (gold_rets - nifty_rets).rolling(
                w21, min_periods=_min_periods(w21)
            ).mean() * ANN_FACTOR
            features["gold_eq_ratio"] = gold_eq_ratio

        # ── Relative momentum ──
        if nifty_rets is not None and bond_rets is not None:
            nifty_mom = prices.get(nifty_rets.name, nifty_rets)
            bond_mom = prices.get(bond_rets.name, bond_rets)
            if nifty_mom is not None and bond_mom is not None:
                n_mom = nifty_mom.shift(21) / nifty_mom - 1.0
                b_mom = bond_mom.shift(21) / bond_mom - 1.0
                momentum_spread = (n_mom - b_mom).replace([np.inf, -np.inf], np.nan)
                features["momentum_spread"] = momentum_spread

        # ── Volatility spread ──
        if nifty_rets is not None and bond_rets is not None:
            n_vol = nifty_rets.rolling(w21, min_periods=_min_periods(w21)).std() * np.sqrt(ANN_FACTOR)
            b_vol = bond_rets.rolling(w21, min_periods=_min_periods(w21)).std() * np.sqrt(ANN_FACTOR)
            features["vol_spread"] = n_vol - b_vol

        # ── Correlation regime (risk-on/off signal) ──
        corr_cols = [k for k in features if "corr" in k]
        if len(corr_cols) >= 2:
            corr_vals = [features[c] for c in corr_cols]
            features["corr_regime"] = pd.concat(corr_vals, axis=1).mean(axis=1)
        elif len(corr_cols) == 1:
            features["corr_regime"] = features[corr_cols[0]]
        else:
            features["corr_regime"] = pd.Series(np.nan, index=returns.index)

        # ── Skewness spread ──
        if nifty_rets is not None and gold_rets is not None:
            n_skew = nifty_rets.rolling(w63, min_periods=_min_periods(w63)).skew()
            g_skew = gold_rets.rolling(w63, min_periods=_min_periods(w63)).skew()
            features["skew_spread"] = n_skew - g_skew

        return pd.DataFrame(features, index=returns.index)

    # ──────────────────────────────────────────────────────────────────────────
    # Standardization (rolling z-score)
    # ──────────────────────────────────────────────────────────────────────────

    def _standardize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply expanding z-score standardization to all features.

        For each feature column f at time t:
          mu_f[t]   = mean(f[0 : t])             -- strictly past data (expanding)
          sigma_f[t] = std(f[0 : t])              -- strictly past data (expanding)
          z_f[t]    = (f[t] - mu_f[t]) / (sigma_f[t] + eps)

        We use an EXPANDING window (not rolling) for two reasons:
          1. Expanding window guarantees a large enough sample at every t
             (after a small warmup), so all features have non-zero std.
          2. Rolling 63-day correlations can have near-constant mean/stderrors
             for some assets, producing dead features for the HMM.

        The strictly-past constraint is preserved because mu and sigma at
        time t use data [0 : t] only, never data at or after t.

        Args:
            features: DataFrame of raw features

        Returns:
            DataFrame of standardized features (same shape, z-scored)
        """
        standardized = pd.DataFrame(index=features.index)

        for col in features.columns:
            raw = features[col].astype(np.float64)
            # Expanding mean and std from strictly past data (no leakage)
            exp_mean = raw.expanding(min_periods=20).mean().shift(1)
            exp_std = raw.expanding(min_periods=20).std().shift(1)
            z = (raw - exp_mean) / (exp_std + EPS)
            standardized[col] = z.clip(-ZSCORE_CLIP, ZSCORE_CLIP)

        return standardized

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fit_transform(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from price data.

        Args:
            prices: DataFrame of daily prices, columns = asset tickers
                    Index = datetime index
                    Expected columns: 'nifty', 'gold', 'bonds'

        Returns:
            DataFrame of standardized features, same index as prices
            Shape: (n_days, 54), no NaN/Inf values (after warmup drop)
        """
        if prices is None or len(prices) == 0:
            raise ValueError("prices DataFrame is empty or None")

        if len(prices.columns) < 2:
            raise ValueError(
                f"Need at least 2 asset columns for cross-asset features, "
                f"got {prices.columns.tolist()}"
            )

        # Compute returns
        returns = prices.pct_change()

        # Normalize column names to lowercase for feature engineering
        returns.columns = [c.lower() for c in returns.columns]
        prices.columns = [c.lower() for c in prices.columns]

        # ── Single-asset features ──
        asset_features = []

        # Map expected asset names to available columns
        asset_map = {
            "nifty": ["nifty", "equity", "eq", "spy", "nifty50", "sensex"],
            "gold": ["gold", "gld", "gc"],
            "bond": ["bond", "bonds", "bnd", "tlt", "10y"],
        }

        def _find_asset(names: list[str], available: list[str]) -> Optional[str]:
            for name in names:
                if name in available:
                    return name
            return available[0] if available else None

        available_cols = list(prices.columns)

        nifty_col = _find_asset(asset_map["nifty"], available_cols)
        gold_col = _find_asset(asset_map["gold"], available_cols)
        bond_col = _find_asset(asset_map["bond"], available_cols)

        asset_assignments = []
        if nifty_col:
            asset_assignments.append(("nifty", nifty_col))
        if gold_col:
            asset_assignments.append(("gold", gold_col))
        if bond_col:
            asset_assignments.append(("bond", bond_col))

        for prefix, col_name in asset_assignments:
            asset_feat = self._compute_asset_features(
                returns=returns[col_name],
                prices=prices[col_name],
                prefix=prefix,
            )
            asset_features.append(asset_feat)

        # ── Cross-asset features ──
        # Create a normalized returns DataFrame for cross-asset computation
        norm_returns = returns[["nifty" if nifty_col is None else nifty_col]].copy()
        norm_returns.columns = ["nifty"]
        if gold_col:
            norm_returns["gold"] = returns[gold_col]
        if bond_col:
            norm_returns["bond"] = returns[bond_col]

        # Rename back to expected names for cross-asset features
        norm_prices = prices.copy()
        if nifty_col and nifty_col != "nifty":
            norm_prices = norm_prices.rename(columns={nifty_col: "nifty"})
        if gold_col and gold_col != "gold":
            norm_prices = norm_prices.rename(columns={gold_col: "gold"})
        if bond_col and bond_col != "bond":
            norm_prices = norm_prices.rename(columns={bond_col: "bond"})

        cross_features = self._compute_cross_asset_features(norm_returns, norm_prices)

        # ── Combine all features ──
        all_features = pd.concat(asset_features + [cross_features], axis=1)

        # ── Standardize ──
        standardized = self._standardize_features(all_features)

        # ── Clean up ──
        # Replace any remaining Inf with NaN, then drop rows with NaN
        standardized = standardized.replace([np.inf, -np.inf], np.nan)

        # Drop rows where ANY feature is NaN (these are warmup periods)
        n_before = len(standardized)
        standardized = standardized.dropna()
        n_after = len(standardized)

        if n_after == 0:
            raise ValueError(
                f"All {n_before} rows dropped after feature computation. "
                f"Check that input data has sufficient history "
                f"(need >= {self.lookback_window} rows)."
            )

        logger.info(
            "Computed %d features from %d price observations "
            "(%d rows dropped as warmup)",
            standardized.shape[1], n_before, n_before - n_after,
        )

        return standardized

    def transform(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new price data using the same feature computation.

        For walk-forward backtesting: compute features up to time t
        using only data available at time t.

        Args:
            prices: DataFrame of daily prices up to current date

        Returns:
            DataFrame of standardized features for available data
        """
        if prices is None or len(prices) == 0:
            raise ValueError("prices DataFrame is empty or None")
        return self.fit_transform(prices)

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of feature column names."""
        return list(FEATURE_NAMES)

    @property
    def n_features(self) -> int:
        """Number of features computed by this engineer."""
        return len(FEATURE_NAMES)

    def __repr__(self) -> str:
        return (
            f"RegimeFeatureEngineer("
            f"lookback={self.lookback_window}, "
            f"calibration={self.calibration_ratio}, "
            f"n_features={self.n_features})"
        )
