"""Build the editable-source institutional RegimeShift research PDF."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "RegimeShift_Quant_Research_Report.pdf"
RESEARCH = ROOT / "results" / "research"
SUBMISSION = ROOT / "results" / "submission"


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.1 * cm, "RegimeShift quantitative research case study")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}"); canvas.restoreState()


def page(title: str, text: str, figure: Path | None = None):
    styles = getSampleStyleSheet(); story = [Paragraph(title, styles["Title"]), Spacer(1, .35 * cm)]
    for paragraph in text.split("\n\n"):
        story.extend([Paragraph(paragraph, styles["BodyText"]), Spacer(1, .18 * cm)])
    if figure and figure.exists(): story.extend([Spacer(1, .25 * cm), Image(str(figure), width=16.5 * cm, height=8.25 * cm)])
    story.append(PageBreak()); return story


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet(); story = []
    story += page("RegimeShift", "Causal market-regime allocation research\n\nQuantitative research case study | 2026\n\nThis report is an honest retrospective study, not an investment recommendation or a claim of proven alpha.")
    toc = "1 Executive summary<br/>2 Research question<br/>3 Data and tradability<br/>4 Canonical data contract<br/>5 NEXT_CLOSE execution<br/>6 Features and scaling<br/>7 Gaussian HMM<br/>8 State mapping and diagnostics<br/>9 CVXPY construction<br/>10 Cost model<br/>11 Official results<br/>12 Controlled ablations<br/>13 Chronological evaluation<br/>14 Rolling-origin evaluation<br/>15 Bootstrap intervals<br/>16 Negative findings<br/>17 Limitations<br/>18 Reproducibility"
    story += page("Table of contents", toc)
    content = [
        ("Executive summary", "Under causal NEXT_CLOSE execution and a base 10 bps one-way asset-level cost assumption, RegimeShift underperformed both static benchmarks on headline risk-adjusted performance. The project makes that negative result visible and tests whether the model complexity is justified.", SUBMISSION / "equity_curves.png"),
        ("Research question", "Can a three-state market-regime allocation improve fixed Indian multi-asset allocations after causal timing, common-horizon benchmark alignment, and stated costs? Parameters were frozen; no performance-driven tuning was performed.", None),
        ("Data and index-level tradability", "The equity leg is ^NSEI, an index-level return and signal proxy rather than a directly executable fill. GOLDBEES is the gold proxy. LIQUIDBEES is a short-duration liquid-bond/cash-equivalent proxy, not a sovereign bond or long-duration hedge.", None),
        ("Canonical data contract", "The canonical dataset contains 4,090 source rows from 2010-01-04 to 2026-07-27. Its portable, line-ending-normalized SHA-256 is bound in the official manifest. The common evaluation horizon has 3,732 observations.", None),
        ("NEXT_CLOSE execution", "Existing holdings earn the completed close-to-close return. Only afterwards may the model observe close-time data, select targets, trade, and pay costs. New targets first earn the next close-to-close return. This avoids same-bar execution.", SUBMISSION / "drawdowns.png"),
        ("Features and train-only scaling", "Feature engineering and standardization are fitted on rolling training windows available at the decision time. No full-sample scaling or backward-filled signal is used in the official experiment.", None),
        ("Three-restart Gaussian HMM", "At every rebalance, three deterministic Gaussian HMM candidates are fitted. Candidate selection is based on training likelihood, never later investment performance.", None),
        ("State mapping and diagnostics", "Restart diagnostics report 178 HMM fits and 534 converged candidates with no non-converged fallback. State labels are diagnostic interpretations, not forecasts or tradable states.", None),
        ("CVXPY portfolio construction", "The full policy uses a constrained, long-only CVXPY optimizer conditioned on the selected state. Controlled policies only select targets; a shared engine retains the same return, drift, turnover, and cost logic.", SUBMISSION / "portfolio_weights.png"),
        ("Asset-level cost model", "The official scenario is base: 10 bps one-way assumed cost for equity, gold, and defensive assets. The study does not model market impact, capacity, taxes, or realized bid-ask execution.", None),
        ("Official common-horizon results", "RegimeShift: 6.02% CAGR, 0.400 Sharpe, 35.77% maximum drawdown. Static 60/40: 6.60%, 0.697, 23.39%. Equal weight: 8.75%, 0.570, 32.96%.", None),
        ("Controlled ablations", "The HMM improves Sharpe over the no-regime optimizer (0.400 versus 0.280) and reduces drawdown. However fixed HMM allocations (0.635 Sharpe) and a transparent volatility rule (0.532 Sharpe) outperform the full optimizer on historical Sharpe.", RESEARCH / "ablation_equity_curves.png"),
        ("Chronological evaluation", "Development, validation, and 2022-2026 retrospective summaries are reported separately. The last period is not described as a pristine holdout because the complete historical sample had been inspected during earlier development.", RESEARCH / "subperiod_sharpe.png"),
        ("Rolling-origin evaluation", "Five chronological folds summarize concentration and instability. Each fold reports only dates through its end, while the underlying NEXT_CLOSE decisions use no future observations.", RESEARCH / "rolling_origin_sharpe.png"),
        ("Paired moving-block bootstrap", "Confidence intervals use 21-trading-day blocks, 2,000 samples, seed 42, and paired date-block positions. The RegimeShift minus 60/40 Sharpe interval includes zero; this report makes no statistical-significance claim.", RESEARCH / "bootstrap_difference_distributions.png"),
        ("Negative findings", "The official strategy does not beat either static benchmark on headline Sharpe. Complexity raises turnover, and simpler alternatives can compare favorably. These observations are retained rather than filtered from the release.", None),
        ("Limitations", "This is retrospective research. ^NSEI is not directly tradable; capacity, impact, taxes, borrowing, product availability, and ETF tracking differences are outside scope. Results do not establish live trading performance.", None),
        ("Reproducibility", "Run `python scripts/run_research_suite.py`, build and execute the notebook, then run the test suite and `python scripts/verify_release.py`. The official result lives in results/submission and research artefacts live in results/research.", None),
    ]
    for title, text, figure in content: story += page(title, text, figure)
    SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=1.8*cm).build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__": build()
