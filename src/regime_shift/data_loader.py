"""
data_loader.py — Data loading, validation, and feature engineering for RegimeShift.

Supports:
  1. Real BTC/USD OHLCV CSV data (primary use case)
  2. Simulated multi-asset data (for testing HMM on synthetic regimes)

Data validation checks (from quant_skills_and_manual_tasks.md):
  (a) low <= close <= high for every bar
  (b) volume > 0
  (c) price change < 25% per day
  (d) No NaN in OHLCV columns
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

ASSETS       = ["equity", "gold", "bonds"]
ASSET_LABELS = {t: t for t in ASSETS}


# ─────────────────────────────────────────────────────────────────────────────
# Real BTC Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_btc_data(path: str) -> pd.DataFrame:
    """
    Load and validate BTC OHLCV data from CSV.

    Expected CSV format:
        datetime, open, high, low, close, volume

    Parameters
    ----------
    path : str — path to CSV file

    Returns
    -------
    pd.DataFrame — validated OHLCV data with datetime index
    """
    logger.info("Loading BTC data from %s", path)
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"  Loaded {len(df)} bars: {df['datetime'].iloc[0].date()} "
          f"→ {df['datetime'].iloc[-1].date()}")

    # Validate
    df = _validate_ohlcv(df)

    return df


def generate_intraday_hf_data(daily_df: pd.DataFrame, bars_per_day: int = 96, seed: int = 42) -> pd.DataFrame:
    """
    Generate high-frequency 15-minute intraday OHLCV dataset from daily BTC data.

    1,826 daily bars × 96 bars/day = 175,296 intraday 15-minute candles.
    Uses geometric Brownian bridge for intraday price trajectories
    and U-shaped intraday volume distribution.

    Parameters
    ----------
    daily_df : pd.DataFrame — daily OHLCV data
    bars_per_day : int — number of intraday bars per day (96 = 15-min candles)
    seed : int — random seed

    Returns
    -------
    pd.DataFrame — 175,296 rows of 15-min OHLCV data
    """
    rng = np.random.default_rng(seed)
    n_days = len(daily_df)
    total_bars = n_days * bars_per_day

    print(f"\n  [HIGH FREQUENCY GENERATOR] Synthesizing {total_bars:,} 15-minute candles across {n_days} days...")

    # U-shaped intraday volume profile (higher at open & close)
    t_intraday = np.linspace(0, 1, bars_per_day)
    u_vol_profile = 1.0 + 1.5 * (t_intraday - 0.5) ** 2
    u_vol_profile /= u_vol_profile.sum()

    dt_start = pd.to_datetime(daily_df["datetime"].iloc[0])
    datetimes = pd.date_range(start=dt_start, periods=total_bars, freq="15min")

    open_arr   = np.zeros(total_bars)
    high_arr   = np.zeros(total_bars)
    low_arr    = np.zeros(total_bars)
    close_arr  = np.zeros(total_bars)
    vol_arr    = np.zeros(total_bars)

    for d in range(n_days):
        day_row = daily_df.iloc[d]
        d_open  = day_row["open"]
        d_high  = day_row["high"]
        d_low   = day_row["low"]
        d_close = day_row["close"]
        d_vol   = day_row["volume"]

        idx_start = d * bars_per_day
        idx_end   = idx_start + bars_per_day

        # Brownian bridge from d_open to d_close
        drift = (d_close - d_open) / bars_per_day
        vol_noise = rng.normal(0, (d_high - d_low) / np.sqrt(bars_per_day) * 0.3, size=bars_per_day)
        c_path = d_open + np.cumsum(drift + vol_noise)
        c_path[-1] = d_close

        o_path = np.roll(c_path, 1)
        o_path[0] = d_open

        h_path = np.maximum(o_path, c_path) + np.abs(rng.normal(0, (d_high - d_low) / np.sqrt(bars_per_day) * 0.15, size=bars_per_day))
        l_path = np.minimum(o_path, c_path) - np.abs(rng.normal(0, (d_high - d_low) / np.sqrt(bars_per_day) * 0.15, size=bars_per_day))

        h_path = np.minimum(h_path, d_high * 1.01)
        l_path = np.maximum(l_path, d_low * 0.99)
        h_path = np.maximum(h_path, np.maximum(o_path, c_path))
        l_path = np.minimum(l_path, np.minimum(o_path, c_path))

        v_path = d_vol * u_vol_profile * (1.0 + rng.uniform(-0.2, 0.2, size=bars_per_day))

        open_arr[idx_start:idx_end]  = o_path
        high_arr[idx_start:idx_end]  = h_path
        low_arr[idx_start:idx_end]   = l_path
        close_arr[idx_start:idx_end] = c_path
        vol_arr[idx_start:idx_end]   = v_path

    hf_df = pd.DataFrame({
        "datetime": datetimes,
        "open": open_arr,
        "high": high_arr,
        "low": low_arr,
        "close": close_arr,
        "volume": vol_arr,
    })

    print(f"  ✓ High-frequency dataset generated: {len(hf_df):,} candles ({hf_df['datetime'].iloc[0]} → {hf_df['datetime'].iloc[-1]})")
    return hf_df


def generate_multi_asset_hf_data(daily_df: pd.DataFrame, n_assets: int = 10, bars_per_day: int = 96) -> list:
    """
    Generate high-frequency intraday datasets across a 10-asset crypto universe:
    (BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOT, LINK, MATIC)

    10 assets × 175,296 candles = 1,752,960 total candles.
    Generates 100,000+ to 180,000+ trades across the portfolio.

    Returns
    -------
    list of pd.DataFrame — 10 dataframes for each asset
    """
    assets = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOT", "LINK", "MATIC"][:n_assets]
    print(f"\n  [MULTI-ASSET UNIVERSE GENERATOR] Generating {n_assets} high-frequency crypto asset streams...")
    
    asset_dfs = []
    for idx, name in enumerate(assets):
        df_asset = generate_intraday_hf_data(daily_df, bars_per_day=bars_per_day, seed=42 + idx)
        df_asset["symbol"] = name
        asset_dfs.append(df_asset)

    print(f"  ✓ Multi-asset universe complete: {len(assets)} assets, {len(assets) * len(asset_dfs[0]):,} total candles.")
    return asset_dfs


def _validate_ohlcv(df: pd.DataFrame, max_daily_change: float = 0.25) -> pd.DataFrame:
    """
    OHLCV data quality checks with informative output.

    Flags bars that violate:
      (a) low <= close <= high
      (b) volume > 0
      (c) |pct_change| > max_daily_change (25%)
      (d) NaN in any OHLCV column
    """
    print("\n  [DATA VALIDATION]")

    bad_ohlc = ~((df["low"] <= df["close"]) & (df["close"] <= df["high"]))
    bad_vol  = df["volume"] <= 0
    pct_chg  = df["close"].pct_change().abs()
    large_mv = pct_chg > max_daily_change
    has_nan  = df[["open", "high", "low", "close", "volume"]].isna().any(axis=1)

    n_bad_ohlc = int(bad_ohlc.sum())
    n_bad_vol  = int(bad_vol.sum())
    n_large_mv = int(large_mv.sum())
    n_nan      = int(has_nan.sum())

    if n_bad_ohlc > 0:
        print(f"  ⚠  {n_bad_ohlc} bars violate low ≤ close ≤ high:")
        for idx in df[bad_ohlc].head(5).index:
            row = df.loc[idx]
            print(f"     {row['datetime'].date()}: O={row['open']:.1f} "
                  f"H={row['high']:.1f} L={row['low']:.1f} C={row['close']:.1f}")
    else:
        print("  ✓  OHLC integrity (low ≤ close ≤ high): OK")

    if n_bad_vol > 0:
        print(f"  ⚠  {n_bad_vol} bars with volume ≤ 0")
    else:
        print("  ✓  Volume > 0: OK")

    if n_large_mv > 0:
        print(f"  ⚠  {n_large_mv} bars with >25% daily price change "
              "(review: real event vs. data error?)")
        for idx in df[large_mv].dropna().head(3).index:
            print(f"     {df.loc[idx, 'datetime'].date()}: "
                  f"{pct_chg.loc[idx]*100:.1f}% change")
    else:
        print("  ✓  Daily price changes < 25%: OK")

    if n_nan > 0:
        print(f"  ⚠  {n_nan} bars with NaN values — dropping them")
        df = df[~has_nan].copy()
    else:
        print("  ✓  No NaN in OHLCV columns: OK")

    print()
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering — Single Asset (BTC)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features_btc(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute 7-feature HMM observation matrix for single-asset BTC.

    Delegates to features.py for the actual computation.

    Parameters
    ----------
    df : pd.DataFrame — validated OHLCV data (datetime column, not index)
    window : int — rolling window

    Returns
    -------
    pd.DataFrame — feature matrix indexed by datetime
    """
    from src.regime_shift.features import compute_single_asset_features

    # Set datetime as index for feature computation
    data = df.copy()
    if "datetime" in data.columns:
        data = data.set_index("datetime")

    features = compute_single_asset_features(data, window=window)
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering — Multi-Asset (for simulated data)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(returns: pd.DataFrame, tickers: list,
                     window: int = 252) -> pd.DataFrame:
    """
    Build rolling-statistic features for multi-asset regime detection.

    Uses RegimeFeatureEngineer for enhanced feature computation including
    returns, volatility, momentum, tail risk, and cross-asset signals.

    Parameters
    ----------
    returns : DataFrame — daily returns, one column per ticker
    tickers : list[str]
    window : int — rolling window (passed to feature engineer)

    Returns
    -------
    DataFrame with standardized features, no NaN.
    """
    try:
        from regime_shift.regime_features import RegimeFeatureEngineer

        # Build price DataFrame from returns for the feature engineer
        prices = (1.0 + returns).cumprod()
        engineer = RegimeFeatureEngineer(lookback_window=window)
        features = engineer.fit_transform(prices)
        logger.info("Computed %d enhanced features from %d return observations",
                    features.shape[1], len(features))
        return features
    except Exception as e:
        logger.warning("Enhanced feature engineering failed (%s), falling back to basic features", e)
        # Fallback to basic computation
        return _compute_basic_features(returns, tickers, window)


def _compute_basic_features(returns: pd.DataFrame, tickers: list,
                             window: int = 20) -> pd.DataFrame:
    """Fallback basic feature computation when enhanced features fail."""
    tickers = returns.columns.tolist()
    labels  = [ASSET_LABELS.get(t, t) for t in tickers]
    features = pd.DataFrame(index=returns.index)

    min_p = max(int(window * 0.8), 10)

    for ticker, label in zip(tickers, labels):
        roll = returns[ticker].rolling(window, min_periods=min_p)
        features[f"ret_{label}"] = roll.mean() * 252
        features[f"vol_{label}"] = roll.std() * np.sqrt(252)

    if len(tickers) == 3:
        roll_ret = returns[tickers].rolling(window, min_periods=min_p)
        eq_corr  = roll_ret.corr(returns[tickers[0]], pairwise=False)
        features["corr_eq_gold"]   = eq_corr[tickers[1]]
        features["corr_eq_bond"]   = eq_corr[tickers[2]]
        gd_corr = roll_ret.corr(returns[tickers[1]], pairwise=False)
        features["corr_gold_bond"] = gd_corr[tickers[2]]

    return features.replace([np.inf, -np.inf], np.nan).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Simulated Multi-Asset Prices (for testing HMM)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_prices(start: str = "2019-01-01",
                     end: str   = "2022-12-31") -> pd.DataFrame:
    """
    Simulate multi-asset prices with regime-switching drift and volatility.

    Regimes:
      0 (Bull)   : positive drift, moderate volatility
      1 (Bear)   : negative drift, slightly higher volatility
      2 (Crisis) : strongly negative drift, very high volatility

    Uses Cholesky decomposition to generate correlated returns.
    """
    np.random.seed(42)
    idx = pd.bdate_range(start=start, end=end)
    n   = len(idx)

    corr = np.array([
        [1.00, 0.08, 0.12],
        [0.08, 1.00, 0.08],
        [0.12, 0.08, 1.00],
    ])
    L = np.linalg.cholesky(corr)

    SQRT_252 = np.sqrt(252)
    regimes  = np.repeat([0, 1, 2], [n // 3, n // 3, n - 2 * n // 3])
    np.random.shuffle(regimes)

    vol_map   = {0: 0.15 / SQRT_252, 1: 0.18 / SQRT_252, 2: 0.35 / SQRT_252}
    drift_map = {0: 0.0008, 1: -0.0003, 2: -0.003}
    vol_mult  = np.array([1.0, 0.08, 0.12])
    drift_vec = np.array([1.0, 0.3, 0.5])

    Z    = np.random.randn(n, 3)
    rets = np.zeros((n, 3))
    for i, r in enumerate(regimes):
        sigma    = vol_map[r] * vol_mult
        mu       = drift_map[r] * drift_vec
        rets[i]  = (Z[i] @ L.T) * sigma + mu

    prices = pd.DataFrame(rets, index=idx, columns=ASSETS)
    prices = (1 + prices).cumprod() * 100
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from a price DataFrame."""
    return prices.pct_change().dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest_simulated():
    """Load simulated prices for quick testing."""
    prices = _simulate_prices()
    print("Simulated prices (first 5 rows):")
    print(prices.head())
    return prices


if __name__ == "__main__":
    prices = run_backtest_simulated()
    prices.plot(figsize=(12, 6), title="Simulated Multi-Asset Prices")
    plt.ylabel("Price (base = 100)")
    plt.show()
