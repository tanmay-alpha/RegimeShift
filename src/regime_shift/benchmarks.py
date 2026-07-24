"""
Exact benchmark weight definitions for the RegimeShift framework.

These are the official IIT Bombay benchmark definitions used to evaluate
the regime-conditioned portfolio.  They are deterministic and contain
no return calculations — only static weight specifications.

Benchmarks:
    static_60_40_weights:  60% equity, 0% gold, 40% bond
    equal_weight_weights:  1/3 each for equity, gold, bond

VIX is never included in any benchmark.
"""

from typing import Dict


def static_60_40_weights() -> Dict[str, float]:
    """
    Return the static 60/40 benchmark weights.

    The 60/40 benchmark is:
        equity = 0.60
        gold   = 0.00
        bond   = 0.40

    This is the traditional 60% equity / 40% bond portfolio.  Gold has zero
    allocation — it is NOT substituted into this benchmark.

    Returns:
        Dict with deterministic asset ordering and weights summing to 1.0.
    """
    return {
        "equity": 0.60,
        "gold": 0.00,
        "bond": 0.40,
    }


def equal_weight_weights() -> Dict[str, float]:
    """
    Return the equal-weight benchmark weights.

    The equal-weight benchmark is:
        equity = 1/3
        gold   = 1/3
        bond   = 1/3

    Returns:
        Dict with deterministic asset ordering and weights summing to 1.0.
    """
    n = 3
    return {
        "equity": 1.0 / n,
        "gold": 1.0 / n,
        "bond": 1.0 / n,
    }


def validate_benchmark_weights(weights: Dict[str, float]) -> None:
    """
    Validate that benchmark weights are well-formed.

    Checks:
        - All three assets are present
        - All weights are finite
        - Weights sum to approximately 1
        - No VIX allocation

    Raises:
        ValueError: If any validation fails.
    """
    required = {"equity", "gold", "bond"}
    present = set(weights.keys())
    missing = required - present
    if missing:
        raise ValueError(f"Benchmark weights missing assets: {sorted(missing)}")

    if "vix" in weights:
        raise ValueError("VIX must not appear in benchmark weights.")

    for asset, weight in weights.items():
        if not float("-inf") < weight < float("inf"):
            raise ValueError(f"Weight for '{asset}' is not finite: {weight}")
        if weight < 0:
            raise ValueError(f"Weight for '{asset}' is negative: {weight}")

    total = sum(weights.values())
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(
            f"Benchmark weights do not sum to 1 (sum={total})."
        )
