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