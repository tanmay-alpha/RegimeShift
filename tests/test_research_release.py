"""Fast integrity coverage for committed research-release artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_shift.research.bootstrap import _sample_positions
from regime_shift.research.policies import MinimumVariance, NoRegimeOptimizer, VolatilityRule


ROOT = Path(__file__).parents[1]
RESEARCH = ROOT / "results" / "research"


def test_ablation_return_index_is_common_and_complete():
    daily = pd.read_csv(RESEARCH / "ablation_daily_returns.csv", index_col="date", parse_dates=True)
    assert len(daily) == 3732
    assert not daily.isna().any().any()
    assert {"RegimeShift", "Static 60/40", "Equal Weight"}.issubset(daily.columns)


def test_bootstrap_block_sampling_is_paired_and_deterministic():
    first = _sample_positions(100, 21, np.random.default_rng(42))
    second = _sample_positions(100, 21, np.random.default_rng(42))
    assert np.array_equal(first, second)
    assert len(first) == 100 and (first >= 0).all() and (first < 100).all()


def test_policy_weight_choices_are_valid_without_return_accounting():
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(rng.normal(0, .01, size=(252, 3)), columns=["equity", "gold", "bond"])
    for policy in (NoRegimeOptimizer(), VolatilityRule(), MinimumVariance()):
        decision = policy.select(returns=returns)
        assert set(decision.weights) == {"equity", "gold", "bond"}
        assert np.isclose(sum(decision.weights.values()), 1.0)
        assert min(decision.weights.values()) >= 0.0


def test_frozen_bootstrap_and_rolling_metadata():
    bootstrap = json.loads((RESEARCH / "bootstrap_metadata.json").read_text())
    assert bootstrap == {
        "method": "paired moving-block bootstrap", "block_length_trading_days": 21,
        "samples": 2000, "seed": 42, "confidence_level": .95,
        "same_resampled_positions_for_all_strategies": True,
    }
    rolling = json.loads((RESEARCH / "rolling_origin_metadata.json").read_text())
    assert rolling["common_index"] and len(rolling["folds"]) == 5


def test_readme_and_notebook_reflect_frozen_results():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--cost-scenario base" in readme
    for row in ("| RegimeShift | 6.02% | 0.400 | 35.77% |", "| Static 60/40 | 6.60% | 0.697 | 23.39% |", "| Equal Weight | 8.75% | 0.570 | 32.96% |"):
        assert row in readme
    notebook = json.loads((ROOT / "notebooks" / "RegimeShift_Submission.ipynb").read_text())
    counts = [cell["execution_count"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert counts == list(range(1, len(counts) + 1))
