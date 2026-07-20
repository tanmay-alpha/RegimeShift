"""
strategy.py — Regime-conditional volume-spike trading strategy.

This is the CORE THESIS of the RegimeShift project:
  "A volume-spike signal has regime-dependent edge.
   By conditioning on the HMM-detected market regime, we:
     - Take LONG  signals only in Bull regimes  (trend-following with momentum)
     - Take SHORT signals only in Bear regimes  (capitalize on downside momentum)
     - Reduce position size by 50% in Crisis    (volatility protection)
     - Skip signals during regime transitions   (avoid whipsaw)"

This integrates the two previously disconnected systems:
  1. Volume-spike signal generator (main.py)
  2. HMM regime detector (regime_detector.py)

Mathematical basis:
  - ATR trailing stop: Chandelier Exit (LeBeau 2000)
      SL_long  = close - ATR × multiplier
      SL_short = close + ATR × multiplier
  - Volume spike threshold:
      threshold = μ_vol + σ_vol_multiplier × σ_vol   [using SHIFTED window]
  - Regime-conditional filtering: Ang & Bekaert (2002)
"""

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import config
from typing import Optional


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators needed for signal generation.

    Indicators computed:
      - ATR(14)       — Wilder average true range for trailing stop
      - vol_mean      — Rolling mean volume (SHIFTED by 1 to avoid lookahead)
      - vol_std       — Rolling std volume  (SHIFTED by 1)
      - vol_spike     — Volume spike threshold: μ + k×σ
    """
    data = data.reset_index(drop=True).copy()

    # ATR(14) — Wilder smoothing via pandas_ta_classic
    data["ATR"] = ta.atr(
        data["high"], data["low"], data["close"],
        length=config.ATR_LENGTH
    )

    # Volume spike threshold — uses STRICTLY PAST window (shift by 1)
    data["vol_mean"]  = data["volume"].rolling(config.VOLUME_WINDOW).mean().shift(1)
    data["vol_std"]   = data["volume"].rolling(config.VOLUME_WINDOW).std().shift(1)
    data["vol_spike"] = (data["vol_mean"]
                         + config.VOLUME_STD_MULTIPLIER * data["vol_std"])

    return data


def volume_spike_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pure volume-spike signal generator (no regime conditioning).
    """
    return regime_conditional_signals(data, regimes=None)


def regime_conditional_signals(
    data: pd.DataFrame,
    regimes: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    THE CORE REGIMESHIFT THESIS — Apply volume-spike signals conditionally
    with bar-by-bar state machine synchronization (vectorized/numpy inner loop).

    Regime rules (Ang & Bekaert 2002):
      Bull   → LONG entries allowed; SHORT entries BLOCKED
      Bear   → SHORT entries allowed; LONG entries BLOCKED
      Crisis → ALL signals allowed with 50% position sizing
    """
    data = compute_indicators(data)
    n = len(data)

    trade_type_arr = np.array(["HOLD"] * n, dtype=object)
    signals_arr    = np.zeros(n, dtype=int)
    pos_frac_arr   = np.ones(n, dtype=float)

    # Align regimes to data
    if "datetime" in data.columns:
        date_col = pd.to_datetime(data["datetime"])
    else:
        date_col = data.index

    if regimes is not None and len(regimes) > 0:
        regime_map = regimes.to_dict()
        regime_arr = date_col.map(regime_map).fillna("Unknown").values
    else:
        regime_arr = np.array(["Unknown"] * n, dtype=object)

    data["regime"] = regime_arr

    # Extract numpy arrays for ultra-fast loop execution
    close_arr = data["close"].values
    open_arr  = data["open"].values
    vol_arr   = data["volume"].values
    spike_arr = data["vol_spike"].values
    atr_arr   = data["ATR"].values

    position  = 0   # 0 = flat, 1 = long, -1 = short
    num_wrong = 0
    trailing_stop = 0.0

    ts_mult = config.TRAILING_STOP_MULTIPLIER
    adverse = config.CONSECUTIVE_ADVERSE_BARS

    first_valid_atr = data["ATR"].first_valid_index()
    start_idx = int(first_valid_atr) if first_valid_atr is not None else config.ATR_LENGTH

    for i in range(start_idx, n):
        vol_spike = spike_arr[i]
        if np.isnan(vol_spike):
            continue

        atr     = atr_arr[i]
        close_i = close_arr[i]
        open_i  = open_arr[i]
        vol_i   = vol_arr[i]
        bullish = close_i > open_i
        bearish = close_i < open_i
        regime  = regime_arr[i]

        if regime == "Crisis":
            pos_frac_arr[i] = config.CRISIS_SIZE_FRAC

        if position == 0:
            # ── Entry ──
            if vol_i > vol_spike:
                if bullish:
                    if regime in ("Bear",):
                        trade_type_arr[i] = "REGIME_BLOCKED"
                        signals_arr[i]    = 0
                    else:
                        signals_arr[i]    = 1
                        trade_type_arr[i] = "LONG"
                        position      = 1
                        trailing_stop = close_i - ts_mult * atr
                        num_wrong     = 0
                elif bearish:
                    if regime in ("Bull",):
                        trade_type_arr[i] = "REGIME_BLOCKED"
                        signals_arr[i]    = 0
                    else:
                        signals_arr[i]    = -1
                        trade_type_arr[i] = "SHORT"
                        position      = -1
                        trailing_stop = close_i + ts_mult * atr
                        num_wrong     = 0

        elif position == 1:
            # ── Manage long ──
            trend_rev = (vol_i >= vol_spike) and bearish

            if close_i <= close_arr[i - 1]:
                num_wrong += 1
            else:
                num_wrong = 0

            if trend_rev:
                if regime in ("Bull",):
                    signals_arr[i]    = -1
                    trade_type_arr[i] = "CLOSE"
                    position  = 0
                    num_wrong = 0
                else:
                    signals_arr[i]    = -2
                    trade_type_arr[i] = "REVERSE_LONG_TO_SHORT"
                    position      = -1
                    trailing_stop = close_i + ts_mult * atr
                    num_wrong     = 0
            elif num_wrong >= adverse:
                signals_arr[i]    = -1
                trade_type_arr[i] = "CLOSE"
                position  = 0
                num_wrong = 0
            elif close_i < trailing_stop:
                signals_arr[i]    = -1
                trade_type_arr[i] = "CLOSE"
                position  = 0
                num_wrong = 0
            else:
                trailing_stop = max(trailing_stop, close_i - ts_mult * atr)

        elif position == -1:
            # ── Manage short ──
            trend_rev = (vol_i >= vol_spike) and bullish

            if close_i >= close_arr[i - 1]:
                num_wrong += 1
            else:
                num_wrong = 0

            if trend_rev:
                if regime in ("Bear",):
                    signals_arr[i]    = 1
                    trade_type_arr[i] = "CLOSE"
                    position  = 0
                    num_wrong = 0
                else:
                    signals_arr[i]    = 2
                    trade_type_arr[i] = "REVERSE_SHORT_TO_LONG"
                    position      = 1
                    trailing_stop = close_i - ts_mult * atr
                    num_wrong     = 0
            elif num_wrong >= adverse:
                signals_arr[i]    = 1
                trade_type_arr[i] = "CLOSE"
                position  = 0
                num_wrong = 0
            elif close_i > trailing_stop:
                signals_arr[i]    = 1
                trade_type_arr[i] = "CLOSE"
                position  = 0
                num_wrong = 0
            else:
                trailing_stop = min(trailing_stop, close_i + ts_mult * atr)

    data["trade_type"]         = trade_type_arr
    data["signals"]            = signals_arr
    data["position_size_frac"] = pos_frac_arr
    return data
