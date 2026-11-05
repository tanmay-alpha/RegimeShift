"""Optional Streamlit/Plotly inspector; all evaluation lives in research.workflow.

Launch with ``streamlit run research_app.py``. Merely opening the page or
changing a chart control never launches an HMM fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from regime_shift.research.spec import STRATEGIES
from regime_shift.research.workflow import (
    CAVEATS,
    DEFAULT_DATASET,
    DEFAULT_RESULTS,
    WorkflowError,
    export_experiment_zip,
    inspect_experiment,
    list_experiments,
    load_table,
    run_experiment,
    validate_config,
)


def default_config() -> dict:
    return {"schema_version": 1, "dataset_path": DEFAULT_DATASET,
            "description": "Independent research comparison",
            "strategies": ["regimeshift", "static_6040", "equal_weight"],
            "evaluation": {"start": "2012-01-01", "end": "2025-12-31"},
            "engine": {"train_window": 252, "rebalance_frequency": 21,
                       "minimum_training_observations": 126, "transaction_cost_bps": 10.0,
                       "random_seed": 42, "hmm_n_iter": 200, "hmm_n_restarts": 3}}


def render_new_experiment(st) -> None:
    st.subheader("Explicit offline experiment")
    st.caption("Use a local canonical-schema CSV. Paths are local to the process; relative paths use the working directory.")
    upload = st.file_uploader("Optional JSON configuration", type=["json"])
    try:
        baseline = validate_config(json.loads(upload.getvalue())) if upload else default_config()
    except (ValueError, UnicodeDecodeError) as exc:
        st.error(f"Invalid configuration: {exc}")
        return
    engine = baseline["engine"]
    with st.form("explicit_research_run"):
        dataset = st.text_input("Local dataset CSV", value=baseline["dataset_path"])
        description = st.text_input("Research question / description", value=baseline["description"])
        variants = st.multiselect("Model / allocation variants", options=list(STRATEGIES),
                                  default=baseline["strategies"], format_func=lambda s: STRATEGIES[s])
        st.caption("All selected outputs share the core reference execution calendar. Reference HMM and static strategies are always computed, even if hidden.")
        left, right = st.columns(2)
        start = left.text_input("Evaluation start (inclusive)", value=baseline["evaluation"]["start"])
        end = right.text_input("Evaluation end (inclusive)", value=baseline["evaluation"]["end"])
        train = left.number_input("Rolling training rows", min_value=60, max_value=2520, value=engine["train_window"], step=1)
        minimum = right.number_input("Minimum training rows", min_value=50, max_value=2520, value=engine["minimum_training_observations"], step=1)
        frequency = left.number_input("Signal interval (trading rows)", min_value=1, max_value=252, value=engine["rebalance_frequency"], step=1)
        costs = right.number_input("Assumed one-way cost (bps)", min_value=0.0, max_value=1000.0, value=float(engine["transaction_cost_bps"]))
        seed = left.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=engine["random_seed"], step=1)
        iterations = right.number_input("HMM iteration bound", min_value=1, max_value=1000, value=engine["hmm_n_iter"], step=1)
        restarts = left.number_input("HMM restarts", min_value=1, max_value=10, value=engine["hmm_n_restarts"], step=1)
        submitted = st.form_submit_button("Run and register experiment", type="primary")
    if not submitted:
        return
    config = {"schema_version": 1, "dataset_path": dataset, "description": description,
              "strategies": variants, "evaluation": {"start": start, "end": end},
              "engine": {"train_window": int(train), "minimum_training_observations": int(minimum),
                         "rebalance_frequency": int(frequency), "transaction_cost_bps": costs,
                         "random_seed": int(seed), "hmm_n_iter": int(iterations), "hmm_n_restarts": int(restarts)}}
    try:
        with st.spinner("Running one registered offline attempt…"):
            result = run_experiment(config)
        st.session_state["latest_experiment"] = str(result.output_dir)
        st.success(f"Completed: {result.output_dir}. Choose Inspect existing results to review it.")
    except WorkflowError as exc:
        st.session_state["latest_experiment"] = str(exc.output_dir)
        st.error(f"Attempt recorded at {exc.output_dir}: {exc}")


def render_results(st, px, output_dir: Path) -> None:
    manifest = inspect_experiment(output_dir)
    st.subheader(output_dir.name)
    st.caption(f"Status: {manifest.get('status', 'unknown')} · protocol {manifest.get('protocol_version')} · commit {manifest.get('git_commit')}")
    with st.expander("Reproducibility manifest and configuration"):
        st.json(manifest)
    try:
        payload = export_experiment_zip(output_dir)
        st.download_button("Download reproducibility ZIP (no raw market data)", payload,
                           file_name=f"{output_dir.name}.zip", mime="application/zip")
    except (OSError, ValueError) as exc:
        st.error(f"Export unavailable: {exc}")
    if manifest.get("status") != "completed":
        st.error(manifest.get("error") or "This attempt did not complete.")
        return
    metrics = load_table(output_dir, "metrics")
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    strategies = st.multiselect("Same-period strategy overlays", options=metrics["strategy"].tolist(),
                               default=metrics["strategy"].tolist(), format_func=lambda s: STRATEGIES[s])
    if not strategies:
        st.info("Select at least one strategy to inspect.")
        return
    tables = {s: load_table(output_dir, "accounting", s) for s in strategies}
    first_index = next(iter(tables.values())).index
    if any(not frame.index.equals(first_index) for frame in tables.values()):
        st.error("Saved strategy dates differ; refusing a filled or partial comparison.")
        return
    st.caption("Wealth is rebased to 1 immediately before the displayed evaluation window; drawdown includes that starting wealth. Holdings carry into the window from the preceding continuous run.")
    data = pd.concat([frame.assign(strategy=s).rename_axis("date").reset_index() for s, frame in tables.items()], ignore_index=True)
    wealth, drawdown, turnover, costs = st.tabs(["Wealth", "Drawdown", "Turnover", "Costs"])
    with wealth:
        st.plotly_chart(px.line(data, x="date", y="net_equity", color="strategy", labels={"net_equity": "Net wealth (initial=1)"}), use_container_width=True)
    with drawdown:
        st.plotly_chart(px.line(data, x="date", y="drawdown", color="strategy", labels={"drawdown": "Signed drawdown (fraction)"}), use_container_width=True)
    with turnover:
        st.plotly_chart(px.line(data, x="date", y="turnover", color="strategy", labels={"turnover": "One-way turnover (fraction)"}), use_container_width=True)
    with costs:
        st.plotly_chart(px.line(data, x="date", y="cumulative_cost_fraction_sum", color="strategy", labels={"cumulative_cost_fraction_sum": "Sum of paid cost fractions"}), use_container_width=True)
        st.caption("Sum of daily paid cost fractions is not compounded wealth drag. Gross/net wealth and descriptive cost drag are retained in the accounting and metrics tables.")
    st.subheader("Chronological reference probabilities and uncertainty")
    probabilities = load_table(output_dir, "regime_probabilities")
    st.plotly_chart(px.line(probabilities, y=["Bull", "Bear", "Crisis"], labels={"value": "Reference HMM probability", "index": "Date"}), use_container_width=True)
    st.plotly_chart(px.line(probabilities, y="entropy_nats", labels={"entropy_nats": "Posterior entropy (nats)"}), use_container_width=True)
    st.caption("Probabilities are the latest reference HMM signal known at each close; a new close's signal is not filled until the next close. They are context, not strategy-specific probabilities for non-HMM allocations or calibrated confidence intervals.")
    strategy = st.selectbox("Inspect allocation / rule explanation", options=strategies, format_func=lambda s: STRATEGIES[s])
    weights = load_table(output_dir, "weights", strategy)
    st.plotly_chart(px.area(weights, labels={"value": "End-of-day weight (fraction)", "index": "Date"}), use_container_width=True)
    date = st.select_slider("Selected accounting close", options=list(weights.index), value=weights.index[-1], format_func=lambda d: str(d.date()))
    st.dataframe(weights.loc[[date]], use_container_width=True)
    explanation = load_table(output_dir, "explanations", strategy).loc[date]
    st.write(explanation["rule_explanation"])
    st.caption(f"Regime of held target: {explanation['regime_for_held_target']} · signal date: {explanation['signal_date_for_held_target']}")
    st.caption("These are documented constraints and observed weights, not a causal decomposition of the optimization. Drifted holdings may leave target bounds. HMM-fixed explanations use the same-config reference HMM signal, not separately instrumented ablation diagnostics.")
    with st.expander("HMM convergence and restart diagnostics"):
        st.json(json.loads((output_dir / "diagnostics.json").read_text(encoding="utf-8")))


def main() -> None:
    # Optional dependencies are never required by the workflow or CLI.
    import plotly.express as px
    import streamlit as st

    st.set_page_config(page_title="RegimeShift Research Lab", layout="wide")
    st.title("RegimeShift · Offline Research Lab")
    st.warning("Research-grade historical simulation, not investment advice or a live-trading system. No significance or profitability claim is implied.")
    with st.expander("Scientific and data caveats"):
        for caveat in CAVEATS:
            st.write(f"• {caveat}")
    mode = st.sidebar.radio("Workflow", ["Inspect existing results", "Configure explicit experiment"])
    if mode == "Configure explicit experiment":
        render_new_experiment(st)
        return
    results_dir = Path(st.sidebar.text_input("Results folder", value=str(DEFAULT_RESULTS)))
    experiments = list_experiments(results_dir)
    if not experiments:
        st.info("No saved manifests here. Choose Configure explicit experiment, or run python -m regime_shift.cli run --config FILE. Opening this page does not run a model.")
        return
    latest = st.session_state.get("latest_experiment")
    default = next((i for i, path in enumerate(experiments) if str(path) == latest), 0)
    selected = st.sidebar.selectbox("Existing versioned experiment", options=experiments, index=default, format_func=lambda p: p.name)
    try:
        render_results(st, px, selected)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Cannot inspect saved artifacts: {exc}")


if __name__ == "__main__":
    main()
