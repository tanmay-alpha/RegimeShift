"""
visualize.py — Visualization utilities for RegimeShift.

Includes:
  - Equity curve with regime shading
  - Regime confidence plot (stacked area chart)
  - Silhouette score history
  - Backtest results vs benchmarks
  - Turnover and cost analysis
  - Regime performance comparison
  - Portfolio weight evolution
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Regime Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_and_regimes(
    result: dict,
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """
    Plot equity curve with regime background shading.

    Args:
        result: Dict from WalkForwardBacktest.run() with keys:
                equity_curve, regimes, tickers
        figsize: Figure size (width, height)
    """
    equity = result["equity_curve"]
    regimes = result["regimes"]
    tickers = result.get("tickers", [])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(equity.index, equity.values, label="Strategy", linewidth=1.5)

    colors = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c", 3: "#9b59b6"}
    labels_map = {0: "Bull", 1: "Bear", 2: "Crisis", 3: "Extreme_Crisis"}

    for state_id in regimes.unique():
        if state_id < 0:
            continue
        mask = regimes == state_id
        ax.fill_between(
            regimes.index,
            0,
            equity.values.max() * 1.1,
            where=mask,
            color=colors.get(state_id, "#95a5a6"),
            alpha=0.08,
        )

    ax.set_title("Equity Curve with Regime Shading")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")

    patches = [
        mpatches.Patch(color=c, alpha=0.3, label=labels_map.get(s, f"State {s}"))
        for s, c in colors.items()
        if s in regimes.unique()
    ]
    ax.legend(handles=patches, loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_regime_confidence(
    dates: pd.DatetimeIndex,
    signals: list,
    output_path: str | None = None,
) -> None:
    """
    Plot regime confidence over time as a stacked area chart.

    Shows how the posterior probability of each regime evolves,
    making regime transitions visually obvious.

    Args:
        dates: DatetimeIndex for x-axis
        signals: List of dicts with 'posteriors' keys (from RegimeSignal.to_dict())
        output_path: Optional path to save figure
    """
    if not signals:
        logger.warning("No signals to plot")
        return

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
    ax.stackplot(
        dates[:n],
        matrix.T,
        labels=all_labels,
        colors=["#e74c3c", "#f1c40f", "#2ecc71", "#9b59b6"][:len(all_labels)],
        alpha=0.8,
    )

    ax.set_title("Regime Confidence Over Time (Posterior Probabilities)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved regime confidence plot to %s", output_path)

    plt.show()


def plot_silhouette_history(
    dates: pd.DatetimeIndex,
    silhouette_scores: list[float],
    output_path: str | None = None,
) -> None:
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
    """
    if not silhouette_scores:
        logger.warning("No silhouette scores to plot")
        return

    scores = np.array(silhouette_scores)
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates[:len(scores)], scores, color="#3498db", linewidth=1.5, label="Silhouette Score")
    ax.axhline(y=0.5, color="#2ecc71", linestyle="--", alpha=0.7, label="Good (0.5)")
    ax.axhline(y=0.2, color="#f39c12", linestyle="--", alpha=0.7, label="Poor (0.2)")

    # Shade regions
    ax.fill_between(
        dates[:len(scores)], 0.5, 1.0,
        alpha=0.1, color="#2ecc71", label="Good separation"
    )
    ax.fill_between(
        dates[:len(scores)], 0.2, 0.5,
        alpha=0.1, color="#f39c12", label="Moderate separation"
    )
    ax.fill_between(
        dates[:len(scores)], -1.0, 0.2,
        alpha=0.1, color="#e74c3c", label="Poor separation"
    )

    ax.set_title("Silhouette Score — Regime Separation Quality Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Silhouette Score")
    ax.set_ylim(-0.5, 1.0)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved silhouette plot to %s", output_path)

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Production Backtest Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest_results(
    result,
    benchmarks,
    output_path=None,
):
    """
    Plot cumulative returns: strategy vs all benchmarks.

    Background shaded by regime (Bull=green, Bear=orange, Crisis=red).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot strategy
    cum_strat = (1 + result.portfolio_returns).cumprod()
    ax.plot(cum_strat.index, cum_strat.values, label="RegimeShift",
            linewidth=2, color="#1f77b4")

    # Plot benchmarks
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, bench) in enumerate(benchmarks.items()):
        cum_bench = (1 + bench.portfolio_returns).cumprod()
        ax.plot(cum_bench.index, cum_bench.values, label=name,
                linewidth=1.5, color=colors[i % len(colors)], alpha=0.8)

    # Regime shading
    if len(result.regime_series) > 0:
        regime_colors = {"Bull": "#90EE90", "Bear": "#FFE4B5", "Crisis": "#FFB6C1",
                         "Extreme_Crisis": "#FF69B4"}
        prev_regime = None
        start = None
        for idx, regime in result.regime_series.items():
            if regime != prev_regime:
                if start is not None and prev_regime is not None:
                    color = regime_colors.get(prev_regime, "#f0f0f0")
                    ax.axvspan(start, idx, alpha=0.15, color=color)
                start = idx
                prev_regime = regime
        if start is not None and prev_regime is not None:
            color = regime_colors.get(prev_regime, "#f0f0f0")
            ax.axvspan(start, result.regime_series.index[-1], alpha=0.15, color=color)

    ax.set_title("RegimeShift vs Benchmarks — Cumulative Returns", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of INR 1")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", output_path)
    return fig, ax


def plot_turnover_costs(result, output_path=None):
    """Plot turnover and transaction costs over time."""
    import matplotlib.pyplot as plt

    if len(result.weights_history) < 2:
        return None, None

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
    return fig, (ax1, ax2)


def plot_regime_performance(regime_metrics, output_path=None):
    """Bar chart: Sharpe ratio and return by regime."""
    import matplotlib.pyplot as plt

    if len(regime_metrics) == 0:
        return None, None

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
    return fig, (ax1, ax2)


def plot_weight_evolution(weights_history, asset_names=None, output_path=None):
    """Stacked area chart of portfolio weights over time."""
    import matplotlib.pyplot as plt

    if len(weights_history) == 0:
        return None, None

    if asset_names is None:
        asset_names = weights_history.columns.tolist()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        weights_history.index,
        [weights_history[col].values for col in weights_history.columns],
        labels=asset_names, alpha=0.8,
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
    return fig, ax
