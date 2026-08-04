"""Small, deterministic writers for public research artefacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd


def write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_equity(returns: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ((1 + returns).cumprod()).plot(ax=ax, linewidth=1.2)
    ax.set_title(title); ax.set_ylabel("Growth of 1 INR"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_drawdowns(returns: pd.DataFrame, path: Path, title: str) -> None:
    wealth = (1 + returns).cumprod()
    drawdown = wealth.div(wealth.cummax()).sub(1)
    fig, ax = plt.subplots(figsize=(10, 5))
    drawdown.plot(ax=ax, linewidth=1.0)
    ax.set_title(title); ax.set_ylabel("Drawdown"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
