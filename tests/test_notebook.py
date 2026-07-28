"""
Tests for the submission notebook.

Covers:
- No placeholder text strings in any cell source:
  * "Implementation will be added"
  * "next development phase"
  * "foundation initialized"
- All required code cells have a non-null execution_count.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "RegimeShift_Submission.ipynb"


FORBIDDEN_PHRASES = [
    "implementation will be added",
    "next development phase",
    "foundation initialized",
    "to be implemented",
    "will be added in a later phase",
]


def _load_notebook() -> dict:
    if not NOTEBOOK_PATH.exists():
        pytest.skip(f"Notebook not found: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _all_cell_sources(nb: dict) -> list:
    out = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            out.append("".join(src))
        else:
            out.append(str(src))
        # Include output text so artifact filenames in print statements are captured
        for o in cell.get("outputs", []):
            ot = o.get("text", "")
            if isinstance(ot, list):
                out.append("".join(ot))
            elif isinstance(ot, str):
                out.append(ot)
    return out


def test_notebook_exists():
    assert NOTEBOOK_PATH.exists(), f"Missing notebook at {NOTEBOOK_PATH}"


def test_notebook_is_valid_json():
    nb = _load_notebook()
    assert "cells" in nb
    assert isinstance(nb["cells"], list)
    assert len(nb["cells"]) > 0


@pytest.mark.parametrize("phrase", FORBIDDEN_PHRASES)
def test_notebook_has_no_placeholder_text(phrase):
    nb = _load_notebook()
    for src in _all_cell_sources(nb):
        assert phrase.lower() not in src.lower(), (
            f"Notebook contains forbidden placeholder phrase: {phrase!r}"
        )


def test_notebook_has_required_sections():
    nb = _load_notebook()
    sections = _all_cell_sources(nb)
    # All 14 sections required by the brief.
    expected_titles = [
        "objective",
        "reproducibility",
        "asset universe",
        "data validation",
        "leakage-safe features",
        "walk-forward",
        "gaussian hmm",
        "bull/bear/crisis",
        "cvxpy",
        "transaction costs",
        "60/40 and equal weight",
        "performance metrics",
        "charts",
        "conclusions",
    ]
    full_text = "\n".join(sections).lower()
    missing = [t for t in expected_titles if t not in full_text]
    assert not missing, f"Missing sections: {missing}"


def test_required_code_cells_have_execution_count():
    nb = _load_notebook()
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 5, (
        f"Notebook has only {len(code_cells)} code cells — expected at least 5"
    )
    for cell in code_cells:
        ec = cell.get("execution_count")
        assert ec is not None, (
            "A code cell has null execution_count — notebook has not been "
            "executed end-to-end."
        )


def test_code_cells_have_outputs():
    """End-to-end executed notebooks produce output for every code cell.

    A code cell that executed and produced nothing would have an empty
    ``outputs`` list.  We require at least one output per cell to ensure
    that the notebook was genuinely run (not just filled in by hand).
    """
    nb = _load_notebook()
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    no_output = [
        i for i, c in enumerate(code_cells)
        if not c.get("outputs")
    ]
    assert not no_output, (
        f"Code cells at positions {no_output} have no outputs — "
        f"notebook may not have been executed end-to-end."
    )


def test_notebook_references_submission_artifacts():
    """Notebook must produce or reference all required submission artifacts.

    Checks that the notebook source code references every artifact that is
    part of the official submission (either by writing or by displaying).
    """
    nb = _load_notebook()
    full_text = "\n".join(_all_cell_sources(nb))
    required_refs = [
        "performance_summary.csv",     # summary metrics written by cell-24
        # The 6 chart PNGs written by generate_all_charts
        "regime_price_chart.png",
        "transition_matrix.png",
        "equity_curves.png",
        "drawdowns.png",
        "portfolio_weights.png",
        "regime_probabilities.png",
    ]
    missing = [r for r in required_refs if r not in full_text]
    assert not missing, (
        f"Notebook does not reference submission artifacts: {missing}"
    )


def test_notebook_references_results_charts():
    """Notebook must embed the six reference charts produced by ``generate_all_charts()``."""
    nb = _load_notebook()
    full_text = "\n".join(_all_cell_sources(nb))
    required_charts = [
        "regime_price_chart.png",
        "transition_matrix.png",
        "equity_curves.png",
        "drawdowns.png",
        "portfolio_weights.png",
        "regime_probabilities.png",
    ]
    missing = [c for c in required_charts if c not in full_text]
    assert not missing, (
        f"Notebook does not reference results charts: {missing}"
    )