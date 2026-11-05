"""Behavioral notebook artifact checks; headings and incidental source wording are not contracts."""

import json
from pathlib import Path

import nbformat

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "RegimeShift_Submission.ipynb"


def test_notebook_is_valid_and_fully_executed_without_errors():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code, "The demonstrator must contain executable research."
    for cell in code:
        assert cell.execution_count is not None, "Execute the complete generated notebook before publishing."
        assert all(output.output_type != "error" for output in cell.outputs)


def test_notebook_embedded_performance_matches_validated_submission():
    """Published numerical evidence, not narrative text, is the notebook contract."""
    import pandas as pd

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    tables = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if html:
                from io import StringIO
                try:
                    tables.extend(pd.read_html(StringIO("".join(html))))
                except ValueError:
                    pass
    summary = pd.read_csv(NOTEBOOK_PATH.parents[1] / "results/submission/performance_summary.csv")
    # The notebook displays calculated performance; identify that table by public metric names.
    performance = [table for table in tables if {"Strategy", "CAGR", "Sharpe"}.issubset(table.columns)]
    assert performance, "Notebook must display evaluated portfolio performance."
    table = performance[0].set_index("Strategy")
    for _, row in summary.iterrows():
        if row["Strategy"] in table.index:
            for metric in ("CAGR", "Sharpe"):
                assert abs(float(table.loc[row["Strategy"], metric]) - row[metric]) <= 0.0001
