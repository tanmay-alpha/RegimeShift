"""Causal close-only execution and explicit, assumption-led trading costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


class ExecutionModel(str, Enum):
    """When a portfolio decision can first earn a return."""

    NEXT_CLOSE = "NEXT_CLOSE"
    NEXT_OPEN = "NEXT_OPEN"


@dataclass(frozen=True)
class AssetCost:
    """One-way assumed execution costs in basis points for one asset."""

    commission_bps: float = 0.0
    half_spread_slippage_bps: float = 0.0
    taxes_and_statutory_bps: float = 0.0

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.half_spread_slippage_bps + self.taxes_and_statutory_bps


@dataclass(frozen=True)
class ExecutionCostModel:
    """Asset-level cost assumptions; market impact is deliberately excluded."""

    name: str = "base"
    assets: Mapping[str, AssetCost] = field(default_factory=dict)
    market_impact_modeled: bool = False

    def validate(self, assets: Sequence[str]) -> None:
        if self.market_impact_modeled:
            raise ValueError("Market impact requires volume/ADV data and is not supported by this dataset.")
        missing = set(assets) - set(self.assets)
        if missing:
            raise ValueError(f"Execution-cost model is missing assets: {sorted(missing)}")
        if any(cost.total_bps < 0 for cost in self.assets.values()):
            raise ValueError("Execution costs must be non-negative.")

    def cost_for_trade(self, target: np.ndarray, pre_trade: np.ndarray, assets: Sequence[str], initial: bool = False) -> float:
        self.validate(assets)
        deltas = np.abs(target) if initial else np.abs(target - pre_trade)
        return float(sum(delta * self.assets[asset].total_bps / 10_000.0 for asset, delta in zip(assets, deltas)))

    @classmethod
    def flat(cls, bps: float, assets: Sequence[str], name: str = "custom") -> "ExecutionCostModel":
        return cls(name=name, assets={asset: AssetCost(half_spread_slippage_bps=float(bps)) for asset in assets})

    @classmethod
    def scenario(cls, name: str, assets: Sequence[str]) -> "ExecutionCostModel":
        assumptions = {"optimistic": 5.0, "base": 10.0, "stressed": 20.0}
        if name not in assumptions:
            raise ValueError(f"Unknown cost scenario '{name}'. Choose one of {sorted(assumptions)}.")
        return cls.flat(assumptions[name], assets, name=name)
