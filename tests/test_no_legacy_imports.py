"""
Test that active source code contains no legacy crypto contamination or forbidden terms.
"""

from pathlib import Path
import pytest

FORBIDDEN_TERMS = [
    "Binance",
    "SatoshiFlow",
    "testnet",
    "volume spike",
    "BTC-only",
    "365-day",
    "ccxt",
]


def test_no_forbidden_terms_in_src():
    """Verify that active src/ files contain no forbidden legacy crypto terms."""
    src_dir = Path(__file__).parent.parent / "src" / "regime_shift"
    py_files = list(src_dir.glob("*.py"))

    assert len(py_files) > 0, "No source files found in src/regime_shift"

    violations = []
    for filepath in py_files:
        content = filepath.read_text(encoding="utf-8")
        for term in FORBIDDEN_TERMS:
            if term.lower() in content.lower():
                violations.append((filepath.name, term))

    assert not violations, f"Forbidden legacy crypto terms found in src/: {violations}"


def test_no_legacy_imports_in_src():
    """Verify that src/ files do not import legacy experimental or CCXT modules."""
    src_dir = Path(__file__).parent.parent / "src" / "regime_shift"
    py_files = list(src_dir.glob("*.py"))

    forbidden_imports = ["ccxt", "streamlit", "plotly", "legacy_student_t_hmm"]

    violations = []
    for filepath in py_files:
        content = filepath.read_text(encoding="utf-8")
        for f_import in forbidden_imports:
            if f"import {f_import}" in content or f"from {f_import}" in content:
                violations.append((filepath.name, f_import))

    assert not violations, f"Forbidden legacy imports found in src/: {violations}"
