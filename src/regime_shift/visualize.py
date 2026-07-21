"""
visualize.py — Visualization utilities for RegimeShift.

All plotting functions use matplotlib only (no seaborn, plotly).
Figures are saved to disk when output_path is provided.

Functions are backward compatible with existing API while adding
richer Phase 3 visualizations.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────────────────────────────────────

REGIME_COLORS: dict[str, str] = {
    "Bull": "#90EE90",
    "Bear": "#FFE4B5",
    "Crisis": "#FFB6C1",
    "Extreme_Crisis": "#FF69B4",
}

STATE_COLORS: dict[int, str] = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e74c3c",
    3: "#9b59b6",
}

STATE_LABELS: dict[int, str] = {
    0: "Bull",
    1: "Bear",
    2: "Crisis",
    3: "Extreme_Crisis",
}

#: Fallback color map for regime labels that are string-based (not integer-based)
_STATE_COLOR_FALLBACK: dict[str, str] = {
    "Bull": "#2ecc71",
    "Bear": "#f1c40f",
    "Crisis": "#e74c3c",
    "Extreme_Crisis": "#9b59b6",
    "N/A": "#95a5a6",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: regime shading
# ─────────────────────────────────────────────────────────────────────────────

def _apply_regime_shading(ax, regime_series: pd.Series, alpha: float = 0.15) -> None:
    """Apply regime background shading to an axis."""
    prev_regime = None
    start = None
    for idx, regime in regime_series.items():
        if pd.isna(regime):
            continue
        regime_str = str(regime)
        if regime_str != prev_regime:
            if start is not None and prev_regime is not None:
                color = REGIME_COLORS.get(prev_regime, _STATE_COLOR_FALLBACK.get(prev_regime, "#f0f0f0"))
                ax.axvspan(start, idx, alpha=alpha, color=color)
            start = idx
            prev_regime = regime_str
    if start is not None and prev_regime is not None:
        color = REGIME_COLORS.get(prev_regime, "#f0f0f0")
        ax.axvspan(start, regime_series.index[-1], alpha=alpha, color=color)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Regime Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_and_regimes(
    result: dict,
    figsize: tuple[int, int] = (14, 6),
) -> tuple:
    """
    Plot equity curve with regime background shading.

    .. deprecated::
        Use ``plot_regime_timeline`` for new code. This function is
        preserved for backward compatibility.

    Args:
        result: Dict with keys ``equity_curve``, ``regimes``, ``tickers``
        figsize: Figure size (width, height)

    Returns:
        (fig, ax) tuple
    """
    equity = result["equity_curve"]
    regimes = result["regimes"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(equity.index, equity.values, label="Strategy", linewidth=1.5)
    _apply_regime_shading(ax, regimes, alpha=0.12)

    # Legend
    unique_states = sorted([s for s in regimes.unique() if not (isinstance(s, float) and np.isnan(s))])
    patches = [
        mpatches.Patch(
            facecolor=STATE_COLORS.get(int(s), "#95a5a6"),
            alpha=0.3,
            label=STATE_LABELS.get(int(s), f"State {s}"),
        )
        for s in unique_states
        if str(s).lstrip("-").isdigit()
    ]
    if patches:
        ax.legend(handles=patches, loc="upper left")

    ax.set_title("Equity Curve with Regime Shading")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_regime_confidence(
    dates: pd.DatetimeIndex,
    signals: list,
    output_path: Optional[str] = None,
) -> tuple:
    """
    Plot regime confidence over time as a stacked area chart.

    Shows how the posterior probability of each regime evolves,
    making regime transitions visually obvious.

    Args:
        dates: DatetimeIndex for x-axis
        signals: List of dicts with ``posteriors`` keys
        output_path: Optional path to save figure

    Returns:
        (fig, ax) tuple
    """
    if not signals:
        logger.warning("No signals to plot")
        fig, ax = plt.subplots()
        return fig, ax

    # Extract all regime labels
    all_labels = set()
    for sig in signals:
        all_labels.update(sig.get("posteriors", {}).keys())
    all_labels = sorted(all_labels)

    # Build matrix: (n_timesteps, n_labels)
    n = len(signals)
    matrix = np.zeros((n, len(all_labels)))
    label_idx = {label: i for i, label in enumerate(all_labels)}

    for t, sig in enumerate(signals):
        for label, prob in sig.get("posteriors", {}).items():
            if label in label_idx:
                matrix[t, label_idx[label]] = prob

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#e74c3c", "#f1c40f", "#2ecc71", "#9b59b6", "#3498db"]
    ax.stackplot(
        dates[:n],
        matrix.T,
        labels=all_labels,
        colors=colors[: len(all_labels)],
        alpha=0.8,
    )

    ax.set_title("Regime Confidence Over Time (Posterior Probabilities)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved regime confidence plot to %s", output_path)

    plt.show()
    return fig, ax


def plot_silhouette_history(
    dates: pd.DatetimeIndex,
    silhouette_scores: list[float],
    output_path: Optional[str] = None,
) -> tuple:
    """
    Plot silhouette score over time to monitor regime separation quality.

    Shaded regions:
      - Green: score > 0.5 (good separation)
      - Yellow: 0.2 < score < 0.5 (moderate)
      - Red: score < 0.2 (poor — regimes not well separated)

    Args:
        dates: DatetimeIndex for x-axis
        silhouette_scores: List of silhouette scores per time step
        output_path: Optional path to save figure

    Returns:
        (fig, ax) tuple
    """
    if not silhouette_scores:
        logger.warning("No silhouette scores to plot")
        fig, ax = plt.subplots()
        return fig, ax

    scores = np.array(silhouette_scores)
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates[: len(scores)], scores, color="#3498db", linewidth=1.5, label="Silhouette Score")
    ax.axhline(y=0.5, color="#2ecc71", linestyle="--", alpha=0.7, label="Good (0.5)")
    ax.axhline(y=0.2, color="#f39c12", linestyle="--", alpha=0.7, label="Poor (0.2)")

    # Shade regions
    ax.fill_between(dates[: len(scores)], 0.5, 1.0, alpha=0.1, color="#2ecc71", label="Good separation")
    ax.fill_between(dates[: len(scores)], 0.2, 0.5, alpha=0.1, color="#f39c12", label="Moderate separation")
    ax.fill_between(dates[: len(scores)], -1.0, 0.2, alpha=0.1, color="#e74c3c", label="Poor separation")

    ax.set_title("Silhouette Score — Regime Separation Quality Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Silhouette Score")
    ax.set_ylim(-0.5, 1.0)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved silhouette plot to %s", output_path)

    plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Rich Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_timeline(
    regime_series: pd.Series,
    prices: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    title: str = "Market Regimes Over Time",
) -> tuple:
    """
    Plot regime timeline with optional price overlay.

    Background colors: Bull=light green, Bear=light orange, Crisis=light red.
    Optionally overlay equity price on secondary axis.

    Args:
        regime_series: Series of regime labels indexed by date
        prices: Optional DataFrame of prices for overlay
        output_path: If provided, save figure to this path
        title: Plot title

    Returns:
        (fig, ax1) tuple
    """
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Regime shading
    _apply_regime_shading(ax1, regime_series, alpha=0.3)

    # Price overlay on secondary axis
    if prices is not None:
        # Try common equity ticker names
        equity_col = None
        for candidate in ["^NSEI", "NIFTY", "nifty", "SPY", "equity"]:
            if candidate in prices.columns:
                equity_col = candidate
                break
        if equity_col is None and len(prices.columns) > 0:
            equity_col = prices.columns[0]

        if equity_col is not None:
            aligned_prices = prices.loc[regime_series.index.intersection(prices.index)]
            if len(aligned_prices) > 0:
                ax2 = ax1.twinx()
                ax2.plot(
                    aligned_prices.index,
                    aligned_prices[equity_col].values,
                    color="navy",
                    linewidth=1.5,
                    alpha=0.7,
                    label=equity_col,
                )
                ax2.set_ylabel(f"{equity_col} Price", color="navy")
                ax2.tick_params(axis="y", labelcolor="navy")

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Regime")
    ax1.set_yticks([])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color, alpha=0.3, label=regime)
        for regime, color in REGIME_COLORS.items()
    ]
    ax1.legend(handles=legend_elements, loc="upper left")

    ax1.grid(True, alpha=0.2)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved regime timeline to %s", output_path)

    plt.show()
    return fig, ax1


def plot_cumulative_returns(
    strategy_returns: pd.Series,
    benchmarks: dict[str, pd.Series],
    regime_series: Optional[pd.Series] = None,
    output_path: Optional[str] = None,
    title: str = "Cumulative Returns — RegimeShift vs Benchmarks",
) -> tuple:
    """
    Plot cumulative returns of strategy vs benchmarks.

    Args:
        strategy_returns: Daily returns of RegimeShift strategy
        benchmarks: dict of name -> daily returns Series
        regime_series: Optional regime labels for background shading
        output_path: If provided, save figure
        title: Plot title

    Returns:
        (fig, ax) tuple
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Regime shading
    if regime_series is not None and len(regime_series) > 0:
        _apply_regime_shading(ax, regime_series, alpha=0.15)

    # Strategy
    cum_strat = (1 + strategy_returns).cumprod()
    ax.plot(cum_strat.index, cum_strat.values, label="RegimeShift",
            linewidth=2.5, color="#1f77b4", zorder=10)

    # Benchmarks
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (name, bench_rets) in enumerate(benchmarks.items()):
        cum_bench = (1 + bench_rets).cumprod()
        ax.plot(
            cum_bench.index,
            cum_bench.values,
            label=name,
            linewidth=1.5,
            color=colors[i % len(colors)],
            alpha=0.8,
            linestyle="--",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of INR 1")
    ax.legend(loc="upper left", ncol=min(len(benchmarks) + 1, 3))
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved cumulative returns to %s", output_path)

    plt.show()
    return fig, ax


def plot_drawdown(
    returns: pd.Series,
    output_path: Optional[str] = None,
    title: str = "Drawdown",
) -> tuple:
    """Plot underwater (drawdown) chart."""
    fig, ax = plt.subplots(figsize=(14, 4))

    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak

    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax.plot(dd.index, dd.values, color="red", linewidth=1)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig, ax


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    output_path: Optional[str] = None,
) -> tuple:
    """Plot rolling Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(14, 4))

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="#1f77b4", linewidth=1.5)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Sharpe = 1.0")
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Sharpe = 0.5")
    ax.set_title(f"Rolling {window}d Sharpe Ratio", fontsize=14)
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig, ax


def plot_feature_importance(
    feature_names: list,
    importance_scores: np.ndarray,
    output_path: Optional[str] = None,
    top_n: int = 20,
) -> tuple:
    """Plot top-N most important features by discriminative power."""
    if len(importance_scores) == 0:
        fig, ax = plt.subplots()
        return fig, ax

    # Sort by importance
    sorted_idx = np.argsort(importance_scores)[-top_n:]
    names = [feature_names[i] for i in sorted_idx]
    scores = importance_scores[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(scores)))
    ax.barh(range(len(scores)), scores, color=colors)
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Discriminative Power (|t-statistic|)")
    ax.set_title(f"Top {top_n} Features for Regime Discrimination")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig, ax


def plot_monthly_heatmap(
    returns: pd.Series,
    output_path: Optional[str] = None,
) -> tuple:
    """Plot monthly return heatmap (years x months)."""
    if len(returns) == 0:
        fig, ax = plt.subplots()
        return fig, ax

    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly.index = pd.MultiIndex.from_arrays([monthly.index.year, monthly.index.month])
    monthly = monthly.unstack()

    fig, ax = plt.subplots(figsize=(10, max(3, len(monthly) * 0.4)))

    cmap = plt.cm.RdYlGn
    vmax = max(abs(monthly.values.min()), abs(monthly.values.max()))
    im = ax.imshow(monthly.values * 100, cmap=cmap, aspect="auto",
                    vmin=-vmax * 100, vmax=vmax * 100)

    ax.set_xticks(range(len(monthly.columns)))
    ax.set_xticklabels([
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ])
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index)
    ax.set_title("Monthly Returns (%)", fontsize=14)

    for i in range(len(monthly.index)):
        for j in range(len(monthly.columns)):
            val = monthly.values[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(
                    j, i, f"{val*100:.1f}%",
                    ha="center", va="center", fontsize=7, color=text_color,
                )

    plt.colorbar(im, ax=ax, label="Return (%)")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig, ax


def plot_regime_weights(
    weights_history: pd.DataFrame,
    regime_series: pd.Series,
    output_path: Optional[str] = None,
) -> tuple:
    """Plot portfolio weights stacked by regime."""
    if len(weights_history) == 0:
        fig, axes = plt.subplots(2, 1)
        return fig, axes

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Stacked area chart
    axes[0].stackplot(
        weights_history.index,
        [weights_history[col].values for col in weights_history.columns],
        labels=weights_history.columns.tolist(),
        alpha=0.8,
        colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    )
    axes[0].set_title("Portfolio Weight Evolution", fontsize=14)
    axes[0].set_ylabel("Weight")
    axes[0].legend(loc="upper left")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Regime bar chart below
    regime_colors_list = [
        REGIME_COLORS.get(str(r), "#f0f0f0") for r in regime_series
    ]
    axes[1].bar(regime_series.index, [1] * len(regime_series),
                color=regime_colors_list, alpha=0.7, width=1)
    axes[1].set_title("Active Regime", fontsize=12)
    axes[1].set_yticks([])
    axes[1].set_ylabel("Regime")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Production Backtest Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest_results(
    result,
    benchmarks: dict,
    output_path: Optional[str] = None,
) -> tuple:
    """
    Plot cumulative returns: strategy vs all benchmarks with regime shading.

    Backward-compatible wrapper that delegates to ``plot_cumulative_returns``
    when regime_series is available.

    Args:
        result: BacktestResult with ``portfolio_returns`` and ``regime_series``
        benchmarks: dict of name -> BacktestResult
        output_path: Optional save path

    Returns:
        (fig, ax) tuple
    """
    strategy_rets = result.portfolio_returns
    bench_rets = {name: br.portfolio_returns for name, br in benchmarks.items()}
    regime_series = getattr(result, "regime_series", None)

    return plot_cumulative_returns(
        strategy_returns=strategy_rets,
        benchmarks=bench_rets,
        regime_series=regime_series,
        output_path=output_path,
        title="RegimeShift vs Benchmarks — Cumulative Returns",
    )


def plot_turnover_costs(result, output_path: Optional[str] = None) -> tuple:
    """Plot turnover and transaction costs over time."""
    if len(result.weights_history) < 2:
        fig, axes = plt.subplots(2, 1)
        return fig, axes

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    diffs = result.weights_history.diff().dropna()
    turnover = diffs.abs().sum(axis=1) / 2.0
    ax1.plot(turnover.index, turnover.values, color="#1f77b4", linewidth=1)
    ax1.axhline(y=0.20, color="red", linestyle="--", alpha=0.5, label="20% limit")
    ax1.set_ylabel("Daily Turnover")
    ax1.set_title("Portfolio Turnover")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    cum_costs = result.costs.cumsum()
    ax2.plot(cum_costs.index, cum_costs.values, color="#d62728", linewidth=1.5)
    ax2.set_ylabel("Cumulative Cost (fraction)")
    ax2.set_title("Transaction Costs")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, (ax1, ax2)


def plot_regime_performance(regime_metrics, output_path: Optional[str] = None) -> tuple:
    """Bar chart: Sharpe ratio and return by regime."""
    if len(regime_metrics) == 0:
        fig, axes = plt.subplots(1, 2)
        return fig, axes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    regimes = regime_metrics.index.tolist()
    sharpes = regime_metrics["sharpe_ratio"].values
    returns = regime_metrics["annualized_return"].values * 100  # Convert to %

    colors = ["#2ca02c" if s > 0 else "#d62728" for s in sharpes]
    ax1.bar(regimes, sharpes, color=colors, alpha=0.8)
    ax1.set_title("Sharpe Ratio by Regime")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    colors2 = ["#2ca02c" if r > 0 else "#d62728" for r in returns]
    ax2.bar(regimes, returns, color=colors2, alpha=0.8)
    ax2.set_title("Annualized Return by Regime")
    ax2.set_ylabel("Return (%)")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, (ax1, ax2)


def plot_weight_evolution(
    weights_history: pd.DataFrame,
    asset_names: Optional[list] = None,
    output_path: Optional[str] = None,
) -> tuple:
    """Stacked area chart of portfolio weights over time."""
    if len(weights_history) == 0:
        fig, ax = plt.subplots()
        return fig, ax

    if asset_names is None:
        asset_names = weights_history.columns.tolist()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        weights_history.index,
        [weights_history[col].values for col in weights_history.columns],
        labels=asset_names,
        alpha=0.8,
    )
    ax.set_title("Portfolio Weight Evolution")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", ncol=min(len(asset_names), 3))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Robustness & Analysis Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_bootstrap_distribution(
    bootstrap_results: dict,
    output_path: Optional[str] = None,
) -> tuple:
    """
    Plot bootstrap distribution of key metrics.

    Args:
        bootstrap_results: Dict from evaluate.bootstrap_metrics()
            Keys: "sharpe", "ann_return", "max_drawdown"
            Values: (median, p2_5, p97_5) tuples
        output_path: Optional save path

    Returns:
        (fig, axes) tuple
    """
    if not bootstrap_results:
        fig, ax = plt.subplots()
        return fig, ax

    n_metrics = len(bootstrap_results)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    metric_names = {"sharpe": "Sharpe Ratio", "ann_return": "Ann. Return", "max_drawdown": "Max Drawdown"}

    for ax, (key, (med, lo, hi)) in zip(axes, bootstrap_results.items()):
        label = metric_names.get(key, key)
        ax.barh([0], [hi - lo], left=[lo], height=0.4, color="#1f77b4", alpha=0.7, label="95% CI")
        ax.scatter([med], [0], color="red", s=100, zorder=5, label="Median")
        ax.axvline(x=0, color="black", linewidth=0.5) if key in ("sharpe",) else None
        ax.set_xlabel(label)
        ax.set_title(f"{label} — 95% Confidence Interval")
        ax.legend(loc="upper left")
        ax.set_yticks([])

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved bootstrap distribution to %s", output_path)

    plt.show()
    return fig, axes


def plot_feature_correlations(
    features: pd.DataFrame,
    output_path: Optional[str] = None,
    max_features: int = 30,
) -> tuple:
    """
    Plot feature correlation heatmap.

    Args:
        features: DataFrame of standardized features
        output_path: Optional save path
        max_features: Maximum features to display (take first N if more)

    Returns:
        (fig, ax) tuple
    """
    if len(features) == 0:
        fig, ax = plt.subplots()
        return fig, ax

    # Subsample if too many features
    feats = features.iloc[:, :max_features] if features.shape[1] > max_features else features
    corr = feats.corr()

    fig, ax = plt.subplots(figsize=(max(8, len(feats.columns) * 0.5), max(6, len(feats.columns) * 0.5)))
    im = ax.imshow(corr.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Feature Correlation Matrix", fontsize=14)

    plt.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature correlation heatmap to %s", output_path)

    plt.show()
    return fig, ax


def plot_regime_transitions(
    transition_matrix: np.ndarray,
    labels: Optional[list] = None,
    output_path: Optional[str] = None,
) -> tuple:
    """
    Plot regime transition matrix as a heatmap.

    Args:
        transition_matrix: (k, k) transition probability matrix
        labels: Optional list of regime labels
        output_path: Optional save path

    Returns:
        (fig, ax) tuple
    """
    k = transition_matrix.shape[0]
    if labels is None:
        labels = [STATE_LABELS.get(i, f"State {i}") for i in range(k)]

    fig, ax = plt.subplots(figsize=(max(5, k * 1.5), max(4, k * 1.2)))
    im = ax.imshow(transition_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(k))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels)
    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    ax.set_title("Regime Transition Probabilities", fontsize=14)

    # Annotate cells
    for i in range(k):
        for j in range(k):
            val = transition_matrix[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Probability")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved transition matrix to %s", output_path)

    plt.show()
    return fig, ax


def save_all_plots(
    result,
    benchmarks: dict,
    regime_metrics,
    turnover_metrics,
    features: Optional[pd.DataFrame] = None,
    output_dir: str = "results",
) -> list:
    """
    Save all standard plots to a directory.

    Args:
        result: BacktestResult
        benchmarks: dict of name -> BacktestResult
        regime_metrics: DataFrame from compute_regime_metrics
        turnover_metrics: dict from compute_turnover_metrics
        features: Optional DataFrame of features for correlation plot
        output_dir: Directory to save plots

    Returns:
        List of saved file paths
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # 1. Cumulative returns
    path = os.path.join(output_dir, "cumulative_returns.png")
    plot_cumulative_returns(
        result.portfolio_returns,
        {n: br.portfolio_returns for n, br in benchmarks.items()},
        regime_series=result.regime_series,
        output_path=path,
    )
    saved.append(path)

    # 2. Drawdown
    path = os.path.join(output_dir, "drawdown.png")
    plot_drawdown(result.portfolio_returns, output_path=path)
    saved.append(path)

    # 3. Rolling Sharpe
    path = os.path.join(output_dir, "rolling_sharpe.png")
    plot_rolling_sharpe(result.portfolio_returns, output_path=path)
    saved.append(path)

    # 4. Turnover & costs
    path = os.path.join(output_dir, "turnover_costs.png")
    plot_turnover_costs(result, output_path=path)
    saved.append(path)

    # 5. Regime performance
    path = os.path.join(output_dir, "regime_performance.png")
    plot_regime_performance(regime_metrics, output_path=path)
    saved.append(path)

    # 6. Weight evolution
    path = os.path.join(output_dir, "weight_evolution.png")
    plot_weight_evolution(result.weights_history, output_path=path)
    saved.append(path)

    # 7. Feature correlations
    if features is not None and len(features) > 0:
        path = os.path.join(output_dir, "feature_correlations.png")
        plot_feature_correlations(features, output_path=path)
        saved.append(path)

    # 8. Monthly heatmap
    path = os.path.join(output_dir, "monthly_returns.png")
    plot_monthly_heatmap(result.portfolio_returns, output_path=path)
    saved.append(path)

    logger.info("Saved %d plots to %s/", len(saved), output_dir)
    return saved
