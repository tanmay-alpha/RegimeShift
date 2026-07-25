"""
Chart generation for RegimeShift submission.

Creates publication-quality Matplotlib figures saved to the results directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_FIGURES = [
    "regime_price_chart.png",
    "transition_matrix.png",
    "equity_curves.png",
    "drawdowns.png",
    "portfolio_weights.png",
    "regime_probabilities.png",
]


def generate_all_charts(
    result,
    benchmarks: dict,
    prices: pd.DataFrame,
    output_dir: str = "results",
) -> list:
    """
    Generate all submission charts.

    Args:
        result: BacktestResult from run_walk_forward_backtest.
        benchmarks: Dict of benchmark name → BenchmarkResult.
        prices: Raw price DataFrame (for equity price chart).
        output_dir: Directory to save charts.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    saved.append(_regime_price_chart(result, prices, output_dir))
    saved.append(_transition_matrix_chart(result, output_dir))
    saved.append(_equity_curves_chart(result, benchmarks, output_dir))
    saved.append(_drawdowns_chart(result, benchmarks, output_dir))
    saved.append(_portfolio_weights_chart(result, output_dir))
    saved.append(_regime_probabilities_chart(result, output_dir))

    logger.info("Generated %d charts in %s", len(saved), output_dir)
    return saved


def _regime_price_chart(result, prices: pd.DataFrame, output_dir: str) -> str:
    """NIFTY price with Bull/Bear/Crisis background shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot equity price
    if "equity" in prices.columns:
        eq = prices["equity"].dropna()
        ax.plot(eq.index, eq.values, label="NIFTY 50", color="#1f77b4", linewidth=1.5)
    ax.set_ylabel("Price (INR)", fontsize=11)
    ax.set_title("NIFTY 50 with Regime Background", fontsize=13)

    # Shade regimes
    regimes = result.regime_series.dropna()
    colors = {"Bull": "#90EE90", "Bear": "#FFB6C1", "Crisis": "#FF6347"}
    alpha = 0.15

    current_regime = None
    start_date = None
    for date, regime in regimes.items():
        if regime != current_regime:
            if current_regime is not None and start_date is not None:
                ax.axvspan(
                    start_date, date,
                    alpha=alpha, color=colors.get(current_regime, "gray"),
                )
            current_regime = regime
            start_date = date

    # Final regime span
    if current_regime is not None and start_date is not None:
        ax.axvspan(
            start_date, regimes.index[-1],
            alpha=alpha, color=colors.get(current_regime, "gray"),
        )

    # Legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors["Bull"], alpha=alpha, label="Bull"),
        Patch(facecolor=colors["Bear"], alpha=alpha, label="Bear"),
        Patch(facecolor=colors["Crisis"], alpha=alpha, label="Crisis"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", title="Regime")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()

    path = os.path.join(output_dir, "regime_price_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _transition_matrix_chart(result, output_dir: str) -> str:
    """Transition matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    trans_mat = result.transition_matrix
    if trans_mat is None or trans_mat.empty:
        trans_mat = pd.DataFrame(
            np.eye(3),
            index=["Bull", "Bear", "Crisis"],
            columns=["Bull", "Bear", "Crisis"],
        )

    im = ax.imshow(trans_mat.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(trans_mat.columns)))
    ax.set_yticks(range(len(trans_mat.index)))
    ax.set_xticklabels(trans_mat.columns)
    ax.set_yticklabels(trans_mat.index)
    ax.set_xlabel("To State", fontsize=11)
    ax.set_ylabel("From State", fontsize=11)
    ax.set_title("Regime Transition Matrix", fontsize=13)

    # Add text annotations
    for i in range(len(trans_mat.index)):
        for j in range(len(trans_mat.columns)):
            val = trans_mat.iloc[i, j]
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                color="white" if val > 0.5 else "black",
                fontsize=11,
            )

    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()

    path = os.path.join(output_dir, "transition_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _equity_curves_chart(result, benchmarks: dict, output_dir: str) -> str:
    """Equity curves comparing strategy and benchmarks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    # Normalize all curves to start at 1.0
    def normalize(s):
        s = s.dropna()
        if len(s) == 0:
            return s
        return s / s.iloc[0]

    ax.plot(
        result.net_equity.index, normalize(result.net_equity).values,
        label="RegimeShift Net", color="#2E86AB", linewidth=1.5,
    )
    ax.plot(
        result.gross_equity.index, normalize(result.gross_equity).values,
        label="RegimeShift Gross", color="#2E86AB",
        linewidth=1.0, linestyle="--", alpha=0.7,
    )

    for name, bench in benchmarks.items():
        if bench.net_equity is not None and len(bench.net_equity) > 0:
            ax.plot(
                bench.net_equity.index, normalize(bench.net_equity).values,
                label=f"{name} Net", linewidth=1.5,
            )
        if bench.gross_equity is not None and len(bench.gross_equity) > 0:
            ax.plot(
                bench.gross_equity.index, normalize(bench.gross_equity).values,
                label=f"{name} Gross", linewidth=1.0, linestyle="--", alpha=0.7,
            )

    ax.set_ylabel("Cumulative Return (normalised)", fontsize=11)
    ax.set_title("Portfolio Equity Curves", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "equity_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _drawdowns_chart(result, benchmarks: dict, output_dir: str) -> str:
    """Drawdown chart for net strategy and benchmarks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    def drawdown(s):
        s = s.dropna()
        if len(s) == 0:
            return pd.Series(index=s.index)
        eq = s.cumprod() if s.min() >= -0.1 else s
        cummax = eq.cummax()
        return (eq / cummax - 1) * 100

    ax.fill_between(
        result.net_equity.index, drawdown(result.net_returns).values,
        label="RegimeShift Net", color="#2E86AB", alpha=0.3,
    )
    ax.plot(
        result.net_equity.index, drawdown(result.net_returns).values,
        color="#2E86AB", linewidth=1.0,
    )

    for name, bench in benchmarks.items():
        if bench.net_returns is not None and len(bench.net_returns) > 0:
            dd = drawdown(bench.net_returns)
            ax.plot(dd.index, dd.values, label=f"{name} Net", linewidth=1.2)

    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.set_title("Portfolio Drawdowns", fontsize=13)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "drawdowns.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _portfolio_weights_chart(result, output_dir: str) -> str:
    """Stacked area chart of target weights over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))

    tw = result.target_weights
    if tw is None or tw.empty:
        tw = pd.DataFrame(
            index=result.net_returns.index,
            columns=["equity", "gold", "bond"],
        )

    colors = {"equity": "#1f77b4", "gold": "#FFD700", "bond": "#8B4513"}
    labels = {"equity": "Equity", "gold": "Gold", "bond": "Bond"}

    ax.stackplot(
        tw.index,
        [tw[c].values for c in tw.columns if c in colors],
        labels=[labels[c] for c in tw.columns if c in labels],
        colors=[colors[c] for c in tw.columns if c in colors],
        alpha=0.8,
    )

    ax.set_ylabel("Weight", fontsize=11)
    ax.set_title("Portfolio Target Weights", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    path = os.path.join(output_dir, "portfolio_weights.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _regime_probabilities_chart(result, output_dir: str) -> str:
    """Regime probability series over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))

    probs = result.regime_probabilities
    if probs is None or probs.empty:
        probs = pd.DataFrame(
            index=result.net_returns.index,
            columns=["Bull", "Bear", "Crisis"],
        )

    colors = {"Bull": "#90EE90", "Bear": "#FFB6C1", "Crisis": "#FF6347"}
    for col in probs.columns:
        if col in colors:
            ax.plot(probs.index, probs[col].values, label=col, color=colors[col],
                    linewidth=1.5)

    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title("Regime Probabilities Over Time", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "regime_probabilities.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
