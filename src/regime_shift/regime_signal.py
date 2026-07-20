"""
regime_signal.py — Rich regime detection output with confidence scoring.

Replaces simple string labels ("Bull"/"Bear"/"Crisis") with a structured
signal that includes posterior probabilities, confidence scores, and
transition flags. This enables confidence-weighted position sizing.

Mathematical basis:
  Posterior: P(z_t=k | X) = forward[t,k] * backward[t,k] / p(X)
  Confidence: max_k P(z_t=k | X)  ∈ [0, 1]
  Expected duration: E[D] = 1 / (1 - A_kk)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RegimeSignal:
    """
    Rich regime detection output with confidence scoring.

    This replaces the simple string label ("Bull"/"Bear"/"Crisis") with
    a structured signal that includes posterior probabilities, confidence
    scores, and transition flags.

    Attributes:
        label: Human-readable regime label ("Bull", "Bear", or "Crisis")
        confidence: Maximum posterior probability [0, 1].
                    0.9+ = high confidence, 0.6-0.9 = moderate, <0.6 = uncertain
        posteriors: Dict mapping each regime label to its posterior probability.
                    Must sum to 1.0.
        regime_duration: Number of consecutive days in this regime
        is_transition: True if this is the first day of a regime change
        transition_from: Previous regime label (if is_transition)
        expected_duration: Expected remaining days in this regime,
                          computed from transition matrix: 1 / (1 - A_kk)
    """

    label: str
    confidence: float
    posteriors: Dict[str, float]
    regime_duration: int = 0
    is_transition: bool = False
    transition_from: Optional[str] = None
    expected_duration: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate invariants on construction."""
        total = sum(self.posteriors.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Posteriors must sum to 1.0, got {total:.10f}. "
                f"Posteriors: {self.posteriors}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be in [0, 1], got {self.confidence}"
            )
        if self.label not in self.posteriors:
            raise ValueError(
                f"Label '{self.label}' not in posteriors: {list(self.posteriors.keys())}"
            )
        if not np.isclose(
            self.posteriors[self.label], self.confidence, atol=1e-6
        ):
            raise ValueError(
                f"Posterior for label '{self.label}' "
                f"({self.posteriors[self.label]:.10f}) != confidence "
                f"({self.confidence:.10f})"
            )

    def weight_for_regime(self, base_weights: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute confidence-weighted position sizes.

        For each regime k with posterior P(z=k|X):
          w = Σ_k P(z=k|X) * w_k*

        where w_k* is the optimal weight vector for regime k.

        This smooths positions during uncertain periods, reducing
        churn when the model is confused.

        Args:
            base_weights: Dict mapping regime label → optimal weight vector (1d ndarray)

        Returns:
            (n_assets,) confidence-blended weight vector summing to 1.0
        """
        n_assets = None
        blended = None

        for regime, prob in self.posteriors.items():
            if regime not in base_weights:
                continue
            w_regime = base_weights[regime]
            if n_assets is None:
                n_assets = len(w_regime)
                blended = np.zeros(n_assets, dtype=np.float64)
            blended += prob * w_regime

        if blended is None:
            raise ValueError("No matching regime weights found in base_weights")

        # Normalize to sum to 1
        total = blended.sum()
        if total > 1e-12:
            blended = blended / total
        else:
            # Fallback to equal weights
            blended = np.ones(n_assets, dtype=np.float64) / n_assets

        return blended

    def should_rebalance(self, threshold: float = 0.15) -> bool:
        """
        Determine if rebalancing is warranted based on regime change
        and confidence level.

        Rebalance if:
        - Regime changed (is_transition=True), OR
        - Confidence dropped below threshold (model is uncertain)

        This prevents unnecessary trading during periods of model confusion.

        Args:
            threshold: Confidence gap threshold. Default 0.15 means rebalance
                      if confidence < 0.85.

        Returns:
            True if rebalancing is recommended
        """
        confidence_threshold = 1.0 - threshold
        return self.is_transition or self.confidence < confidence_threshold

    def to_dict(self) -> Dict:
        """
        Serialize to a dictionary (for logging, JSON storage, etc.).

        Returns:
            Dict with all signal fields
        """
        return {
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "posteriors": {k: round(v, 6) for k, v in self.posteriors.items()},
            "regime_duration": self.regime_duration,
            "is_transition": self.is_transition,
            "transition_from": self.transition_from,
            "expected_duration": (
                round(self.expected_duration, 2)
                if self.expected_duration is not None
                else None
            ),
        }

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        trans_str = (
            f" from {self.transition_from}" if self.is_transition else ""
        )
        dur_str = f", duration={self.regime_duration}d"
        exp_str = (
            f", exp_dur={self.expected_duration:.1f}d"
            if self.expected_duration is not None
            else ""
        )
        return (
            f"RegimeSignal(label={self.label}, confidence={self.confidence:.3f}"
            f"{trans_str}{dur_str}{exp_str})"
        )
