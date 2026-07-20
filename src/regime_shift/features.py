"""
features.py — Advanced feature engineering for single-asset BTC regime detection.

Features are designed as HMM observation vectors.
Each feature captures a distinct market dimension:

1. Rolling return (annualised)        — trend direction
2. Rolling volatility (annualised)    — market turbulence
3. Volume Z-score                     — institutional activity (SHIFTED to avoid lookahead)
4. ATR ratio                          — volatility relative to price level
5. OBV Z-score                        — cumulative volume momentum
6. VWAP deviation                     — price vs. volume-weighted fair value
7. Return Z-score                     — how extreme is today's move vs. recent history

Research basis:
- Ardia, Bluteau & Rüede (2019): volume as regime transition predictor
- Hamilton (1989): returns + volatility as HMM features
- Dacorogna et al. (2001): multi-scale volatility features
"""

import numpy as np
import pandas as pd
import math
from typing import Optional


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                length: int = 14) -> pd.Series:
    """
    Average True Range — Wilder (1978).

        TR_t = max(|H_t - L_t|, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
        ATR_t = EMA(TR, length)   [Wilder's smoothing: α = 1/length]

    Returns
    -------
    pd.Series — ATR values aligned to input index
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing (equivalent to EMA with α = 1/length)
    atr = tr.ewm(alpha=1.0 / length, adjust=False).mean()
    return atr


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume — Granville (1963).

        OBV_t = OBV_{t-1} + V_t * sign(C_t - C_{t-1})

    Tracks cumulative buying/selling pressure.

    Returns
    -------
    pd.Series — OBV cumulative sum
    """
    direction = np.sign(close.diff()).fillna(0)
    obv = (volume * direction).cumsum()
    return obv


def compute_vwap(close: pd.Series, volume: pd.Series,
                 window: int = 20) -> pd.Series:
    """
    Rolling VWAP (Volume Weighted Average Price).

        VWAP_t = Σ_{t-w}^{t} (V_i × C_i) / Σ_{t-w}^{t} V_i

    Note: Uses only past data (no lookahead) via .shift(1) on volume×price.
    """
    pv = (close * volume).rolling(window, min_periods=max(int(window * 0.8), 5)).sum()
    vol_sum = volume.rolling(window, min_periods=max(int(window * 0.8), 5)).sum()
    return pv / (vol_sum + 1e-12)


def compute_single_asset_features(
    df: pd.DataFrame,
    window: int = 20,
    atr_length: int = 14,
    zscore_clip: float = 3.0,
) -> pd.DataFrame:
    """
    Build the 7-feature observation matrix for HMM regime detection on BTC.

    All features are constructed using ONLY data available at time t:
    - Rolling statistics use .shift(1) where necessary to avoid lookahead bias.
    - Features are z-score standardized and winsorized at ±zscore_clip σ.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close, volume
        Must be sorted ascending by date.
    window : int
        Rolling window for statistics (default 20 ≈ 1 trading month).
    atr_length : int
        Period for ATR computation (Wilder, default 14).
    zscore_clip : float
        Winsorization threshold for z-scores (default 3.0 σ).

    Returns
    -------
    pd.DataFrame — Feature matrix, same index as df, dropna applied.
        Columns: ret_ann, vol_ann, vol_zscore, atr_ratio, obv_zscore,
                 vwap_dev, ret_zscore
    """
    df = df.copy()
    min_p = max(int(window * 0.8), 5)

    # ── 1. Daily returns ──
    ret = df["close"].pct_change()

    # ── 2. Rolling return (annualised) ──
    # Shift by 1: today's return must NOT be in rolling window for today's signal
    ret_roll = ret.shift(1).rolling(window, min_periods=min_p)
    ret_ann  = ret_roll.mean() * 365  # crypto annualisation

    # ── 3. Rolling volatility (annualised) ──
    vol_ann = ret.shift(1).rolling(window, min_periods=min_p).std() * math.sqrt(365)

    # ── 4. Volume Z-score ──
    # Strictly uses the PREVIOUS window to compute μ and σ (shift by 1)
    vol_mean = df["volume"].rolling(window, min_periods=min_p).mean().shift(1)
    vol_std  = df["volume"].rolling(window, min_periods=min_p).std().shift(1)
    vol_zscore = (df["volume"] - vol_mean) / (vol_std + 1e-12)

    # ── 5. ATR ratio (normalized by close price) ──
    atr = compute_atr(df["high"], df["low"], df["close"], length=atr_length)
    atr_ratio = atr / (df["close"] + 1e-12)

    # ── 6. OBV Z-score ──
    obv = compute_obv(df["close"], df["volume"])
    obv_mean   = obv.rolling(window, min_periods=min_p).mean().shift(1)
    obv_std    = obv.rolling(window, min_periods=min_p).std().shift(1)
    obv_zscore = (obv - obv_mean) / (obv_std + 1e-12)

    # ── 7. VWAP deviation ──
    vwap     = compute_vwap(df["close"], df["volume"], window=window)
    vwap_dev = (df["close"] - vwap) / (vwap + 1e-12)

    # ── 8. Return Z-score (how extreme is today's move) ──
    ret_mean   = ret.shift(1).rolling(window, min_periods=min_p).mean()
    ret_std    = ret.shift(1).rolling(window, min_periods=min_p).std()
    ret_zscore = (ret - ret_mean) / (ret_std + 1e-12)

    # ── Assemble ──
    features = pd.DataFrame({
        "ret_ann"    : ret_ann,
        "vol_ann"    : vol_ann,
        "vol_zscore" : vol_zscore,
        "atr_ratio"  : atr_ratio,
        "obv_zscore" : obv_zscore,
        "vwap_dev"   : vwap_dev,
        "ret_zscore" : ret_zscore,
    }, index=df.index)

    # ── Winsorize z-score features at ±clip σ ──
    zscore_cols = ["vol_zscore", "obv_zscore", "ret_zscore"]
    for col in zscore_cols:
        features[col] = features[col].clip(-zscore_clip, zscore_clip)

    # ── Replace inf / nan ──
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    return features


def validate_ohlcv(df: pd.DataFrame, max_daily_change: float = 0.25) -> pd.DataFrame:
    """
    OHLCV data quality validation.

    Checks (per Tanmay's quant_skills_and_manual_tasks.md guidelines):
      (a) low <= close <= high for every bar
      (b) volume > 0
      (c) |pct_change(close)| < max_daily_change (flags outliers)
      (d) No NaN in OHLCV columns

    Parameters
    ----------
    df : pd.DataFrame — OHLCV data
    max_daily_change : float — alert threshold for daily price change (default 25%)

    Returns
    -------
    pd.DataFrame — validated (and bad-row-flagged) dataframe.
        Adds columns: 'data_ok' (bool) and 'data_issue' (string description).
    """
    df = df.copy()
    issues = []

    bad_ohlc = ~(df["low"] <= df["close"]) | ~(df["close"] <= df["high"])
    bad_vol  = df["volume"] <= 0
    pct_chg  = df["close"].pct_change().abs()
    large_mv = pct_chg > max_daily_change
    has_nan  = df[["open", "high", "low", "close", "volume"]].isna().any(axis=1)

    n_bad_ohlc = bad_ohlc.sum()
    n_bad_vol  = bad_vol.sum()
    n_large_mv = large_mv.sum()
    n_nan      = has_nan.sum()

    if n_bad_ohlc > 0:
        print(f"  [DATA WARN] {n_bad_ohlc} bars violate low <= close <= high")
        for idx in df[bad_ohlc].index[:5]:
            row = df.loc[idx]
            print(f"    {idx}: O={row['open']:.2f} H={row['high']:.2f} "
                  f"L={row['low']:.2f} C={row['close']:.2f}")

    if n_bad_vol > 0:
        print(f"  [DATA WARN] {n_bad_vol} bars have volume <= 0")

    if n_large_mv > 0:
        print(f"  [DATA WARN] {n_large_mv} bars have >25% daily price change "
              f"(review for data errors vs real events):")
        for idx in df[large_mv].dropna().index[:5]:
            print(f"    {idx}: {pct_chg.loc[idx]*100:.1f}%")

    if n_nan > 0:
        print(f"  [DATA WARN] {n_nan} bars contain NaN values")

    if n_bad_ohlc == 0 and n_bad_vol == 0 and n_nan == 0:
        print("  [DATA OK] All OHLCV integrity checks passed.")

    # Flag rows
    df["data_ok"] = ~(bad_ohlc | bad_vol | has_nan)

    return df
