"""
visualize.py — Visualization utilities for RegimeShift.

Includes:
  - Equity curve with regime shading
  - Regime confidence plot (stacked area chart)
  - Silhouette score history
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


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
