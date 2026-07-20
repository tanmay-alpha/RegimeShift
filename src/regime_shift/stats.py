"""
stats.py — Comprehensive quantitative performance metrics for RegimeShift.

All formulas are research-backed:
  - Sharpe Ratio          : Lo (2002) annualised with √365 for crypto
  - Sharpe t-statistic    : Lo (2002) significance test
  - Sortino Ratio         : Sortino & van der Meer (1991)
  - Calmar Ratio          : Young (1991)
  - Omega Ratio           : Keating & Shadwick (2002)
  - Information Ratio     : Grinold & Kahn (1994)
  - Kelly Criterion       : Kelly (1956), fractional form
  - CAGR                  : Standard compound growth formula
  - Profit Factor         : Industry standard
  - Maximum Adverse/
    Favorable Excursion   : Schwager (1984)
"""

import numpy as np
import pandas as pd
import math
from typing import Optional, List


# ──────────────────────────────────────────────────────────────────────────────
# Core Return & Risk Measures
# ──────────────────────────────────────────────────────────────────────────────

def cagr(equity_curve: pd.Series, ann_factor: int = 365) -> float:
    """
    Compound Annual Growth Rate.

        CAGR = (V_final / V_initial)^(ann_factor / n_days) - 1

    Parameters
    ----------
    equity_curve : pd.Series (datetime-indexed)
        Portfolio equity values over time.
    ann_factor : int
        365 for crypto (always open), 252 for equities.

    Returns
    -------
    float — annualised growth rate (e.g. 0.25 = 25%/year)
    """
    curve = equity_curve.dropna()
    if len(curve) < 2:
        return 0.0
    n_days = (curve.index[-1] - curve.index[0]).days
    if n_days <= 0:
        return 0.0
    ratio = curve.iloc[-1] / curve.iloc[0]
    if ratio <= 0:
        return -1.0
    return float(ratio ** (ann_factor / n_days) - 1.0)


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum Drawdown (MDD) — peak-to-trough decline.

        DD_t = (C_t - max_{s<=t} C_s) / max_{s<=t} C_s
        MDD  = min_t DD_t

    Returns
    -------
    float — MDD as a positive fraction (e.g. 0.35 = 35% drawdown)
    """
    curve = equity_curve.dropna()
    if len(curve) < 2:
        return 0.0
    rolling_max = curve.cummax()
    drawdowns = (curve - rolling_max) / rolling_max
    return float(abs(drawdowns.min()))


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return the full drawdown time-series (negative values)."""
    curve = equity_curve.dropna()
    rolling_max = curve.cummax()
    return (curve - rolling_max) / rolling_max


def annualised_return(returns: pd.Series, ann_factor: int = 365) -> float:
    """Geometric annualised return from a daily returns series."""
    rets = returns.dropna()
    if len(rets) == 0:
        return 0.0
    return float((1.0 + rets).prod() ** (ann_factor / len(rets)) - 1.0)


def annualised_volatility(returns: pd.Series, ann_factor: int = 365) -> float:
    """Annualised standard deviation of returns."""
    rets = returns.dropna()
    if len(rets) < 2:
        return 0.0
    return float(rets.std() * math.sqrt(ann_factor))


# ──────────────────────────────────────────────────────────────────────────────
# Sharpe Ratio (Lo 2002)
# ──────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    ann_factor: int = 365,
) -> float:
    """
    Annualised Sharpe Ratio.

        SR = E[r - r_f] / σ(r - r_f) * √ann_factor

    Reference: Lo (2002) "The Statistics of Sharpe Ratios", FAJ.

    Parameters
    ----------
    returns : pd.Series — daily returns
    risk_free_annual : float — annual risk-free rate (e.g. 0.05 = 5%)
    ann_factor : int — 365 for crypto, 252 for equities

    Returns
    -------
    float — annualised Sharpe ratio
    """
    rets = returns.dropna()
    if len(rets) < 2:
        return 0.0
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / ann_factor) - 1.0
    excess = rets - rf_daily
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * math.sqrt(ann_factor))


def sharpe_tstat(returns: pd.Series, risk_free_annual: float = 0.0,
                  ann_factor: int = 365) -> tuple:
    """
    t-statistic for testing whether Sharpe ratio is significantly positive.

    Under IID returns, Lo (2002) derives:
        t = SR_annual / √(1 + SR_annual²/2) * √(T/ann_factor)

    where T = number of daily return observations.

    Returns
    -------
    (t_stat, p_value_one_sided) — p-value < 0.05 suggests real edge.
    """
    from scipy import stats as scipy_stats

    rets = returns.dropna()
    T = len(rets)
    if T < 30:
        return (0.0, 1.0)

    SR = sharpe_ratio(rets, risk_free_annual, ann_factor)
    # Lo (2002) Eq. (5): asymptotic std of SR estimate
    # σ(SR) ≈ √((1 + SR²/2) / T)  for IID returns
    se = math.sqrt((1.0 + SR ** 2 / 2.0) / T)
    # Rescale: SR above is annualized, se is for per-period SR
    # Convert SR back to per-period
    SR_per_period = SR / math.sqrt(ann_factor)
    se_per_period = math.sqrt((1.0 + SR_per_period ** 2 / 2.0) / T)
    t_stat = SR_per_period / se_per_period if se_per_period > 0 else 0.0
    p_value = float(scipy_stats.t.sf(t_stat, df=T - 1))  # one-sided
    return (float(t_stat), p_value)


# ──────────────────────────────────────────────────────────────────────────────
# Sortino Ratio (Sortino & van der Meer 1991)
# ──────────────────────────────────────────────────────────────────────────────

def sortino_ratio(
    returns: pd.Series,
    mar: float = 0.0,
    ann_factor: int = 365,
) -> float:
    """
    Sortino Ratio — penalizes only downside deviation.

        Sortino = (E[r] - MAR) / σ_d * √ann_factor

    where σ_d = √(E[min(r - MAR, 0)²])  is the downside deviation.

    Reference: Sortino & van der Meer (1991), Journal of Portfolio Management.

    Parameters
    ----------
    returns : pd.Series — daily returns
    mar : float — minimum acceptable return per period (default 0.0)
    ann_factor : int — annualisation factor

    Returns
    -------
    float — annualised Sortino ratio
    """
    rets = returns.dropna()
    if len(rets) < 2:
        return 0.0
    excess = rets - mar
    downside = np.minimum(excess, 0.0)
    downside_dev = math.sqrt(np.mean(downside ** 2))
    if downside_dev == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float(excess.mean() / downside_dev * math.sqrt(ann_factor))


# ──────────────────────────────────────────────────────────────────────────────
# Calmar Ratio
# ──────────────────────────────────────────────────────────────────────────────

def calmar_ratio(equity_curve: pd.Series, ann_factor: int = 365) -> float:
    """
    Calmar Ratio — CAGR divided by absolute Maximum Drawdown.

        Calmar = CAGR / |MDD|

    Reference: Young (1991), Futures Magazine.

    High Calmar (> 1) means the strategy earns more than its worst drawdown annually.
    """
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return float("inf")
    returns = equity_curve.pct_change().dropna()
    ann_ret = annualised_return(returns, ann_factor)
    return float(ann_ret / mdd)


# ──────────────────────────────────────────────────────────────────────────────
# Omega Ratio (Keating & Shadwick 2002)
# ──────────────────────────────────────────────────────────────────────────────

def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega Ratio — ratio of gains to losses relative to a threshold.

        Ω(L) = Σ max(r_t - L, 0) / Σ max(L - r_t, 0)

    Reference: Keating & Shadwick (2002), Journal of Performance Measurement.

    Ω > 1 means gains outweigh losses at threshold L.
    Ω > 2 is considered excellent for trading strategies.
    """
    rets = returns.dropna()
    gains = np.maximum(rets - threshold, 0.0).sum()
    losses = np.maximum(threshold - rets, 0.0).sum()
    if losses == 0:
        return float("inf")
    return float(gains / losses)


# ──────────────────────────────────────────────────────────────────────────────
# Information Ratio
# ──────────────────────────────────────────────────────────────────────────────

def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    ann_factor: int = 365,
) -> float:
    """
    Information Ratio — active return per unit of tracking error.

        IR = (E[r_p] - E[r_b]) / σ(r_p - r_b) * √ann_factor

    Reference: Grinold & Kahn (1994), Active Portfolio Management.

    Parameters
    ----------
    portfolio_returns : pd.Series — strategy daily returns
    benchmark_returns : pd.Series — benchmark daily returns
    ann_factor : int

    Returns
    -------
    float — annualised Information Ratio
    """
    p = portfolio_returns.dropna()
    b = benchmark_returns.dropna()
    common_idx = p.index.intersection(b.index)
    if len(common_idx) < 2:
        return 0.0
    active = p.loc[common_idx] - b.loc[common_idx]
    std = active.std()
    if std == 0:
        return 0.0
    return float(active.mean() / std * math.sqrt(ann_factor))


# ──────────────────────────────────────────────────────────────────────────────
# Kelly Criterion (Kelly 1956)
# ──────────────────────────────────────────────────────────────────────────────

def kelly_criterion(returns: pd.Series, fraction: float = 0.5) -> float:
    """
    Fractional Kelly Criterion — optimal position size fraction.

    For continuous returns (log-normal approximation):
        f* = E[r] / Var[r]

    Fractional Kelly: f = fraction * f*  (default: half-Kelly = 0.5)

    Reference: Kelly (1956), "A New Interpretation of Information Rate".

    Parameters
    ----------
    returns : pd.Series — trade or daily returns
    fraction : float — Kelly fraction (0.5 = half-Kelly, recommended)

    Returns
    -------
    float — optimal position fraction (e.g. 0.25 = bet 25% of capital)
    """
    rets = returns.dropna()
    if len(rets) < 10:
        return 0.0
    mu = rets.mean()
    var = rets.var()
    if var <= 0 or mu <= 0:
        return 0.0
    full_kelly = float(mu / var)
    # Cap at 1.0 (never bet more than 100%)
    return float(min(fraction * full_kelly, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Profit Factor
# ──────────────────────────────────────────────────────────────────────────────

def profit_factor(trade_pnls: List[float]) -> float:
    """
    Profit Factor — ratio of gross profits to gross losses.

        PF = Σ max(PnL_i, 0) / Σ max(-PnL_i, 0)

    PF > 1 means the strategy makes money overall.
    PF > 2 is considered strong.
    """
    arr = np.array(trade_pnls)
    gross_wins = arr[arr > 0].sum()
    gross_losses = abs(arr[arr < 0].sum())
    if gross_losses == 0:
        return float("inf")
    return float(gross_wins / gross_losses)


# ──────────────────────────────────────────────────────────────────────────────
# MAE / MFE (Schwager 1984)
# ──────────────────────────────────────────────────────────────────────────────

def compute_mae_mfe(
    ohlcv: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    direction: int,  # +1 for long, -1 for short
    entry_price: float,
) -> tuple:
    """
    Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).

    MAE: deepest unfavorable price move from entry (measures worst-case risk).
    MFE: highest favorable price move from entry (measures max unrealized profit).

    Reference: Schwager (1984), "A Complete Guide to the Futures Markets".

    Returns
    -------
    (mae, mfe) as positive fractions of entry price
    """
    slice_data = ohlcv.iloc[entry_idx : exit_idx + 1]
    if len(slice_data) == 0:
        return (0.0, 0.0)

    if direction == 1:  # Long
        worst = slice_data["low"].min()
        best  = slice_data["high"].max()
        mae = max((entry_price - worst) / entry_price, 0.0)
        mfe = max((best - entry_price) / entry_price, 0.0)
    else:  # Short
        worst = slice_data["high"].max()
        best  = slice_data["low"].min()
        mae = max((worst - entry_price) / entry_price, 0.0)
        mfe = max((entry_price - best) / entry_price, 0.0)

    return (float(mae), float(mfe))


# ──────────────────────────────────────────────────────────────────────────────
# Full Statistics Report
# ──────────────────────────────────────────────────────────────────────────────

def compute_full_stats(
    equity_curve: pd.Series,
    trade_pnls: List[float],
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_annual: float = 0.0,
    mar: float = 0.0,
    ann_factor: int = 365,
    kelly_fraction: float = 0.5,
) -> dict:
    """
    Compute all quantitative performance metrics in one call.

    Returns
    -------
    dict — mapping metric name → value
    """
    returns = equity_curve.pct_change().dropna()
    pnls = np.array(trade_pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    t_stat, p_val = sharpe_tstat(returns, risk_free_annual, ann_factor)

    stats = {
        # ── Return metrics ──
        "CAGR (%)":                  cagr(equity_curve, ann_factor) * 100,
        "Total Return (%)":          (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
                                     if len(equity_curve) > 1 else 0.0,
        "Annualised Vol (%)":        annualised_volatility(returns, ann_factor) * 100,

        # ── Risk-adjusted ──
        "Sharpe Ratio":              sharpe_ratio(returns, risk_free_annual, ann_factor),
        "Sharpe t-stat":             t_stat,
        "Sharpe p-value":            p_val,
        "Sortino Ratio":             sortino_ratio(returns, mar, ann_factor),
        "Calmar Ratio":              calmar_ratio(equity_curve, ann_factor),
        "Omega Ratio":               omega_ratio(returns, threshold=mar),

        # ── Drawdown ──
        "Max Drawdown (%)":          max_drawdown(equity_curve) * 100,
        "Avg Drawdown (%)":          abs(drawdown_series(equity_curve).mean()) * 100,

        # ── Trade-level ──
        "Total Trades":              len(pnls),
        "Win Rate (%)":              float(len(wins) / len(pnls) * 100) if len(pnls) > 0 else 0.0,
        "Profit Factor":             profit_factor(trade_pnls),
        "Avg Win":                   float(wins.mean()) if len(wins) > 0 else 0.0,
        "Avg Loss":                  float(losses.mean()) if len(losses) > 0 else 0.0,
        "Largest Win":               float(wins.max())   if len(wins) > 0 else 0.0,
        "Largest Loss":              float(losses.min()) if len(losses) > 0 else 0.0,
        "Expectancy":                float(pnls.mean())  if len(pnls) > 0 else 0.0,

        # ── Position sizing ──
        "Kelly Fraction":            kelly_criterion(pd.Series(pnls), kelly_fraction),
    }

    # ── Benchmark comparison ──
    if benchmark_returns is not None:
        stats["Information Ratio"] = information_ratio(
            returns, benchmark_returns, ann_factor
        )
        bm_sharpe = sharpe_ratio(benchmark_returns, risk_free_annual, ann_factor)
        stats["Benchmark Sharpe"] = bm_sharpe
        stats["Benchmark CAGR (%)"] = annualised_return(benchmark_returns, ann_factor) * 100

    return stats


def print_stats(stats: dict, title: str = "PERFORMANCE STATISTICS") -> None:
    """Pretty-print the full stats dictionary."""
    width = 50
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key:<30} : {val:>10.4f}")
        else:
            print(f"  {key:<30} : {val!s:>10}")
    print("=" * width + "\n")
