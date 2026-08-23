"""
Tests for the submission notebook.

Covers:
- No placeholder text strings in any cell source
- Required sections present
- If executed: all code cells have execution_count and outputs
- Notebook references key submission artifacts
- No stale parameter names (check_bfill)
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
    "check_bfill",
]

# Sections required by the research brief
REQUIRED_SECTIONS = [
    "research question",
    "asset universe",
    "dataset",
    "causal next_close",
    "leakage-safe features",
    "walk-forward",
    "regime-conditioned",
    "transaction costs",
    "ablation",
    "subperiod",
    "rolling-origin",
    "bootstrap",
    "negative findings",
    "limitations",
]

# Artifacts that the executed notebook should reference
REQUIRED_ARTIFACTS = [
    "performance_summary.csv",
    "submission_market_data.csv",
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


def _is_executed(nb: dict) -> bool:
    """Return True iff the notebook has been run end-to-end (all code cells have exec count)."""
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    return all(c.get("execution_count") is not None for c in code_cells) if code_cells else False


def test_notebook_exists():
    assert NOTEBOOK_PATH.exists(), f"Missing notebook at {NOTEBOOK_PATH}"


def test_notebook_is_valid_json():
    nb = _load_notebook()
    assert "cells" in nb
    assert isinstance(nb["cells"], list)
    assert len(nb["cells"]) > 0


@pytest.mark.parametrize("phrase", FORBIDDEN_PHRASES)
def test_notebook_has_no_forbidden_phrase(phrase):
    nb = _load_notebook()
    for src in _all_cell_sources(nb):
        assert phrase.lower() not in src.lower(), (
            f"Notebook contains forbidden phrase: {phrase!r}"
        )


def test_notebook_has_required_sections():
    nb = _load_notebook()
    full_text = "\n".join(_all_cell_sources(nb)).lower()
    missing = [t for t in REQUIRED_SECTIONS if t.lower() not in full_text]
    assert not missing, f"Notebook is missing required sections: {missing}"


def test_notebook_references_submission_artifacts():
    nb = _load_notebook()
    full_text = "\n".join(_all_cell_sources(nb))
    missing = [r for r in REQUIRED_ARTIFACTS if r not in full_text]
    assert not missing, (
        f"Notebook does not reference submission artifacts: {missing}"
    )


def test_notebook_has_minimum_code_cells():
    nb = _load_notebook()
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    assert len(code_cells) >= 5, (
        f"Notebook has only {len(code_cells)} code cells — expected at least 5"
    )


def test_notebook_execution_count_consistent():
    """
    If the notebook has been executed (any cell has non-null execution_count),
    then ALL code cells must have non-null execution_count.

    If the notebook is freshly generated (all execution_count=null), this test
    passes — the notebook is awaiting genuine kernel execution.
    """
    nb = _load_notebook()
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not code_cells:
        return
    any_executed = any(c.get("execution_count") is not None for c in code_cells)
    if any_executed:
        not_executed = [i for i, c in enumerate(code_cells) if c.get("execution_count") is None]
        assert not not_executed, (
            f"Notebook is partially executed: code cells at positions {not_executed} have "
            "null execution_count. Re-run end-to-end with nbconvert."
        )


def test_notebook_outputs_consistent_with_execution():
    """
    If the notebook has been executed, all code cells must have at least one output.
    If not executed yet, this check is skipped.
    """
    nb = _load_notebook()
    if not _is_executed(nb):
        pytest.skip("Notebook has not been executed end-to-end — skipping output check.")
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    no_output = [i for i, c in enumerate(code_cells) if not c.get("outputs")]
    assert not no_output, (
        f"Code cells at positions {no_output} have no outputs — "
        "notebook may not have been executed end-to-end."
    )


def test_no_check_bfill_in_notebook_source():
    """validate_price_data no longer accepts check_bfill — must not appear in notebook."""
    nb = _load_notebook()
    full_text = "\n".join(_all_cell_sources(nb))
    assert "check_bfill" not in full_text, (
        "Notebook contains stale 'check_bfill' parameter — "
        "rebuild with scripts/build_notebook.py"
    )