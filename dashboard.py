"""
dashboard.py — RegimeShift Interactive Streamlit Dashboard

Panels:
  1. Header + Regime Status Badge
  2. Candlestick Chart with regime shading + trade markers
  3. Performance Metrics Grid (15+ stats)
  4. Equity Curve vs Buy-and-Hold
  5. Drawdown Chart
  6. Regime Analysis (timeline + stats table + transition matrix)
  7. Monte Carlo Distribution
  8. Walk-Forward Results

Usage:
    streamlit run dashboard.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import config
from src.regime_shift.data_loader    import load_btc_data, compute_features_btc
from src.regime_shift.regime_detector import RegimeDetector
from src.regime_shift.strategy        import compute_indicators, regime_conditional_signals
from src.regime_shift.stats           import compute_full_stats, max_drawdown, drawdown_series
from src.regime_shift.monte_carlo     import bootstrap_sharpe_test, permutation_test_pnl

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RegimeShift — BTC Algo Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0a0e1a; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f33 0%, #12172b 100%);
        border: 1px solid #2a3152;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .metric-label { color: #7986cb; font-size: 12px; font-weight: 600; letter-spacing: 0.08em; }
    .metric-value { color: #e8eaf6; font-size: 22px; font-weight: 700; }
    .metric-good  { color: #69f0ae; font-size: 22px; font-weight: 700; }
    .metric-bad   { color: #ff5252; font-size: 22px; font-weight: 700; }
    
    /* Section headers */
    .section-header {
        color: #7986cb;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        border-bottom: 1px solid #2a3152;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    
    /* Regime badge */
    .regime-bull   { background:#1b5e20; color:#69f0ae; padding:6px 18px; border-radius:20px; font-weight:700; }
    .regime-bear   { background:#b71c1c; color:#ff8a80; padding:6px 18px; border-radius:20px; font-weight:700; }
    .regime-crisis { background:#e65100; color:#ffcc02; padding:6px 18px; border-radius:20px; font-weight:700; }
    
    /* Sidebar */
    .css-1d391kg { background-color: #0d1117; }
    
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


REGIME_COLORS = {
    "Bull":   "rgba(105, 240, 174, 0.10)",
    "Bear":   "rgba(255, 82,  82,  0.10)",
    "Crisis": "rgba(255, 204,  2,  0.12)",
    "Unknown":"rgba(128, 128, 128, 0.06)",
}
REGIME_LINE_COLORS = {"Bull": "#69f0ae", "Bear": "#ff5252", "Crisis": "#ffcc02"}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Parameters
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚙️ Strategy Parameters")

atr_length      = st.sidebar.slider("ATR Length",              5, 30,  config.ATR_LENGTH)
trailing_mult   = st.sidebar.slider("Trailing Stop Multiplier",0.5, 4.0, config.TRAILING_STOP_MULTIPLIER, 0.1)
vol_window      = st.sidebar.slider("Volume Window",           5,  50,  config.VOLUME_WINDOW)
vol_std_mult    = st.sidebar.slider("Volume Spike σ Multiplier",0.5,3.0, config.VOLUME_STD_MULTIPLIER, 0.1)
adverse_bars    = st.sidebar.slider("Consecutive Adverse Bars",1,  7,   config.CONSECUTIVE_ADVERSE_BARS)
n_regimes       = st.sidebar.slider("HMM Regimes",             2,  5,   config.N_REGIMES)
use_regime      = st.sidebar.toggle("Enable Regime Filter",    True)
run_mc          = st.sidebar.toggle("Run Monte Carlo",         False)

# Override config with sidebar values
config.ATR_LENGTH              = atr_length
config.TRAILING_STOP_MULTIPLIER = trailing_mult
config.VOLUME_WINDOW           = vol_window
config.VOLUME_STD_MULTIPLIER   = vol_std_mult
config.CONSECUTIVE_ADVERSE_BARS = adverse_bars
config.N_REGIMES               = n_regimes

st.sidebar.markdown("---")
initial_capital = st.sidebar.number_input("Initial Capital ($)", 100, 100000, int(config.INITIAL_CAPITAL), step=100)
config.INITIAL_CAPITAL = float(initial_capital)


# ─────────────────────────────────────────────────────────────────────────────
# Cached Data & Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading & validating BTC data...")
def load_and_validate():
    return load_btc_data(config.DATA_PATH)


@st.cache_data(show_spinner="Computing HMM features...", hash_funcs={pd.DataFrame: lambda x: x.shape})
def get_features(data_hash: int, _data: pd.DataFrame):
    return compute_features_btc(_data, window=20)


@st.cache_data(show_spinner="Fitting HMM (Baum-Welch EM)...", hash_funcs={pd.DataFrame: lambda x: x.shape})
def detect_regimes_cached(features_hash: int, _features: pd.DataFrame, n_states: int):
    det    = RegimeDetector(n_states=n_states, n_iter=config.HMM_ITER, random_state=42)
    states = det.fit_predict(_features)
    return states, det


def run_pipeline(data: pd.DataFrame, features: pd.DataFrame,
                 regimes_series: pd.Series, detector, use_regime_filter: bool):
    """Run the full trading pipeline and return backtest results."""
    from backtester import BackTester

    # Generate signals
    regimes_arg = regimes_series if use_regime_filter else None
    result = regime_conditional_signals(data.copy(), regimes=regimes_arg)
    result.to_csv(config.OUTPUT_PATH, index=False)

    # Backtest
    bt = BackTester(
        config.SYMBOL,
        signal_data_path=config.OUTPUT_PATH,
        master_file_path=config.OUTPUT_PATH,
        compound_flag=config.COMPOUND_FLAG,
    )
    bt.get_trades(config.INITIAL_CAPITAL)

    if len(bt.trades) == 0:
        return None, result, None, None

    # Equity curve
    bt.calc_capital()
    equity_curve  = bt.data["capital"].dropna()
    daily_returns = equity_curve.pct_change().dropna()
    bm_returns    = bt.data["close"].pct_change().dropna()
    trade_pnls    = [t.pnl() for t in bt.trades]

    # Stats
    stats = compute_full_stats(
        equity_curve=equity_curve,
        trade_pnls=trade_pnls,
        benchmark_returns=bm_returns,
        risk_free_annual=config.RISK_FREE_RATE,
        ann_factor=config.ANNUALIZATION_FACTOR,
    )
    return bt, result, equity_curve, stats


# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────

data     = load_and_validate()
features = get_features(id(data), data)
regimes_series, detector = detect_regimes_cached(id(features), features, n_regimes)

# Get current regime (last known)
current_regime_id = regimes_series.iloc[-1] if len(regimes_series) > 0 else 0
current_regime    = detector.get_state_name(current_regime_id)

# Run pipeline
bt, result_data, equity_curve, stats = run_pipeline(
    data, features, regimes_series, detector, use_regime
)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1: Header
# ─────────────────────────────────────────────────────────────────────────────

col_title, col_regime, col_period = st.columns([3, 1, 2])

with col_title:
    st.markdown("# 📈 RegimeShift")
    st.markdown("*BTC/USD Volume-Spike Strategy with HMM Regime Detection*")

with col_regime:
    badge_class = f"regime-{current_regime.lower()}"
    st.markdown(f"**Current Regime**")
    st.markdown(f'<span class="{badge_class}">● {current_regime}</span>',
                unsafe_allow_html=True)

with col_period:
    d0 = pd.to_datetime(data["datetime"].iloc[0]).strftime("%b %Y")
    d1 = pd.to_datetime(data["datetime"].iloc[-1]).strftime("%b %Y")
    st.markdown(f"**Data Period**: {d0} → {d1}")
    st.markdown(f"**{len(data):,} daily bars** | **{len(bt.trades) if bt else 0} trades**")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2: Main Chart — Candlestick + Regime Shading + Trades
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="section-header">📊 Price Chart with Regime Shading & Trades</p>',
            unsafe_allow_html=True)

def build_main_chart(data: pd.DataFrame, result_data: pd.DataFrame,
                     regimes_series: pd.Series, detector, bt) -> go.Figure:
    dates_col = pd.to_datetime(data["datetime"])

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=dates_col,
        open=data["open"], high=data["high"],
        low=data["low"],   close=data["close"],
        name="BTC/USD",
        increasing_line_color="#69f0ae",
        decreasing_line_color="#ff5252",
    ))

    # Volume bar (secondary y-axis)
    fig.add_trace(go.Bar(
        x=dates_col, y=data["volume"],
        name="Volume", opacity=0.3,
        marker_color="#7986cb",
        yaxis="y2",
    ))

    # Regime shading
    if len(regimes_series) > 0:
        features_dates = regimes_series.index
        prev_regime = None
        seg_start   = None

        for date, state in regimes_series.items():
            name = detector.get_state_name(state)
            if name != prev_regime:
                if prev_regime is not None:
                    fig.add_vrect(
                        x0=seg_start, x1=date,
                        fillcolor=REGIME_COLORS.get(prev_regime, "rgba(0,0,0,0)"),
                        opacity=1.0, line_width=0,
                        annotation_text=prev_regime[0],
                        annotation_position="top left",
                    )
                prev_regime = name
                seg_start   = date

        if prev_regime and seg_start:
            fig.add_vrect(
                x0=seg_start, x1=regimes_series.index[-1],
                fillcolor=REGIME_COLORS.get(prev_regime, "rgba(0,0,0,0)"),
                opacity=1.0, line_width=0,
            )

    # Trade markers
    if bt and len(bt.trades) > 0:
        long_entries  = result_data[result_data["trade_type"] == "LONG"]
        short_entries = result_data[result_data["trade_type"] == "SHORT"]

        if "datetime" in long_entries.columns:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(long_entries["datetime"]),
                y=long_entries["close"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#69f0ae"),
                name="Long Entry",
            ))
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(short_entries["datetime"]),
                y=short_entries["close"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#ff5252"),
                name="Short Entry",
            ))

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1117",
        font=dict(color="#e8eaf6"),
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor="#1a1f33",
            showgrid=True,
        ),
        yaxis=dict(
            title="Price (USD)",
            gridcolor="#1a1f33",
            showgrid=True,
            side="right",
        ),
        yaxis2=dict(
            overlaying="y",
            side="left",
            showgrid=False,
            showticklabels=False,
        ),
        legend=dict(
            bgcolor="rgba(13,17,23,0.8)",
            bordercolor="#2a3152",
        ),
        height=520,
        margin=dict(l=20, r=60, t=30, b=20),
        hovermode="x unified",
    )
    return fig

chart = build_main_chart(data, result_data, regimes_series, detector, bt)
st.plotly_chart(chart, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3: Performance Metrics Grid
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="section-header">📋 Performance Statistics</p>', unsafe_allow_html=True)

if stats is None or bt is None:
    st.warning("⚠️ No trades generated. Adjust the volume spike parameters.")
else:
    def metric_card(label, value, fmt="{:.4f}", good_if_positive=True):
        if isinstance(value, float):
            val_str = fmt.format(value)
            if good_if_positive:
                cls = "metric-good" if value > 0 else "metric-bad"
            else:
                cls = "metric-bad" if value > 0 else "metric-good"  # e.g. drawdown
        else:
            val_str = str(value)
            cls = "metric-value"
        return f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="{cls}">{val_str}</div>
        </div>"""

    cols = st.columns(5)
    metric_data = [
        ("CAGR (%)",           stats.get("CAGR (%)", 0),           "{:.2f}%", True),
        ("Sharpe Ratio",        stats.get("Sharpe Ratio", 0),       "{:.3f}",  True),
        ("Sortino Ratio",       stats.get("Sortino Ratio", 0),      "{:.3f}",  True),
        ("Calmar Ratio",        stats.get("Calmar Ratio", 0),       "{:.3f}",  True),
        ("Omega Ratio",         stats.get("Omega Ratio", 0),        "{:.3f}",  True),
        ("Max Drawdown (%)",    stats.get("Max Drawdown (%)", 0),   "{:.2f}%", False),
        ("Win Rate (%)",        stats.get("Win Rate (%)", 0),       "{:.1f}%", True),
        ("Profit Factor",       stats.get("Profit Factor", 0),      "{:.3f}",  True),
        ("Kelly Fraction",      stats.get("Kelly Fraction", 0),     "{:.3f}",  True),
        ("Total Trades",        stats.get("Total Trades", 0),       "{:.0f}",  True),
        ("Sharpe p-value",      stats.get("Sharpe p-value", 1),     "{:.4f}",  False),
        ("Annualised Vol (%)",  stats.get("Annualised Vol (%)", 0), "{:.2f}%", False),
        ("Avg Win",             stats.get("Avg Win", 0),            "${:.2f}", True),
        ("Avg Loss",            stats.get("Avg Loss", 0),           "${:.2f}", False),
        ("Expectancy",          stats.get("Expectancy", 0),         "${:.2f}", True),
    ]

    for i, (label, value, fmt, good_pos) in enumerate(metric_data):
        with cols[i % 5]:
            if isinstance(value, float):
                if "%" in fmt:
                    val_str = f"{value:.2f}%"
                elif "$" in fmt:
                    val_str = f"${value:.2f}"
                else:
                    val_str = f"{value:.4f}"
                if good_pos:
                    cls = "metric-good" if value > 0 else "metric-bad"
                else:
                    cls = "metric-bad" if value > 0 else "metric-good"
            else:
                val_str = str(value)
                cls = "metric-value"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="{cls}">{val_str}</div></div>',
                unsafe_allow_html=True
            )

    # Significance indicator
    pval = stats.get("Sharpe p-value", 1.0)
    if pval < 0.05:
        st.success(f"✓ Sharpe ratio is **statistically significant** (p = {pval:.4f} < 0.05)")
    else:
        st.warning(f"⚠ Sharpe ratio is NOT significant (p = {pval:.4f} ≥ 0.05) — strategy may lack real edge")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 & 5: Equity Curve + Drawdown
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<p class="section-header">📈 Equity Curve vs Buy-and-Hold</p>', unsafe_allow_html=True)

if bt and equity_curve is not None:
    # Build buy-and-hold equity
    bm_prices = data["close"].values
    bm_equity  = pd.Series(
        bm_prices / bm_prices[0] * config.INITIAL_CAPITAL,
        index=pd.to_datetime(data["datetime"]),
    )
    bm_equity  = bm_equity.reindex(equity_curve.index, method="nearest")

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        mode="lines", name="RegimeShift",
        line=dict(color="#7986cb", width=2.5),
    ))
    fig_eq.add_trace(go.Scatter(
        x=bm_equity.index, y=bm_equity.values,
        mode="lines", name="Buy-and-Hold BTC",
        line=dict(color="#546e7a", width=1.5, dash="dot"),
    ))

    dd_series = drawdown_series(equity_curve) * 100
    fig_eq.add_trace(go.Scatter(
        x=dd_series.index, y=dd_series.values,
        mode="lines", name="Drawdown (%)",
        line=dict(color="#ff5252", width=1),
        fill="tozeroy", fillcolor="rgba(255,82,82,0.1)",
        yaxis="y2",
    ))

    fig_eq.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1117",
        font=dict(color="#e8eaf6"),
        yaxis=dict(title="Portfolio Value ($)", gridcolor="#1a1f33"),
        yaxis2=dict(
            title="Drawdown (%)", overlaying="y", side="right",
            showgrid=False, tickformat=".1f",
        ),
        legend=dict(bgcolor="rgba(13,17,23,0.8)"),
        height=380, margin=dict(l=20, r=60, t=30, b=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig_eq, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 6: Regime Analysis
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<p class="section-header">🎯 Regime Analysis</p>', unsafe_allow_html=True)

col_reg_stats, col_trans = st.columns([1, 1])

with col_reg_stats:
    st.markdown("**Regime Statistics**")
    regime_rows = []
    for state_id in sorted(regimes_series.unique()):
        name = detector.get_state_name(state_id)
        mask = regimes_series == state_id
        count = mask.sum()
        pct   = count / len(regimes_series) * 100

        # Run lengths
        changes    = mask.astype(int).diff().fillna(0).ne(0)
        run_starts = changes[changes].index
        run_lengths = []
        for i, start in enumerate(run_starts):
            end = run_starts[i + 1] if i + 1 < len(run_starts) else regimes_series.index[-1]
            run_lengths.append(mask.loc[start:end].sum())
        avg_dur = np.mean(run_lengths) if run_lengths else 0

        regime_rows.append({
            "Regime": name, "Days": int(count),
            "Frequency": f"{pct:.1f}%",
            "Avg Duration (days)": f"{avg_dur:.1f}",
        })

    st.table(pd.DataFrame(regime_rows).set_index("Regime"))

with col_trans:
    st.markdown("**Transition Matrix** (A[i→j])")
    A    = detector.get_transition_matrix()
    n_s  = detector.n_states
    names = [detector.get_state_name(i) for i in range(n_s)]
    df_A  = pd.DataFrame(A, index=names, columns=names).round(4)

    fig_hm = px.imshow(
        df_A.values, x=names, y=names,
        color_continuous_scale="Blues",
        aspect="auto", text_auto=".3f",
    )
    fig_hm.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1117",
        font=dict(color="#e8eaf6"),
        height=250, margin=dict(l=10, r=10, t=30, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 7: Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

if run_mc and bt and equity_curve is not None:
    st.markdown("---")
    st.markdown('<p class="section-header">🎲 Monte Carlo Significance Test</p>',
                unsafe_allow_html=True)

    with st.spinner(f"Running {config.MONTE_CARLO_RUNS:,} bootstrap iterations..."):
        daily_returns = equity_curve.pct_change().dropna()
        boot_result   = bootstrap_sharpe_test(
            daily_returns,
            n_bootstrap=config.MONTE_CARLO_RUNS,
            block_size=config.BLOCK_SIZE,
            ann_factor=config.ANNUALIZATION_FACTOR,
        )

    real_sr     = boot_result["real_sharpe"]
    boot_srs    = boot_result["bootstrap_sharpes"]
    pval_mc     = boot_result["p_value"]

    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(
        x=boot_srs, nbinsx=60,
        marker_color="#7986cb", opacity=0.75,
        name="Bootstrap Sharpe Distribution",
    ))
    fig_mc.add_vline(
        x=real_sr, line_dash="dash", line_color="#69f0ae", line_width=2,
        annotation_text=f"Real SR = {real_sr:.3f}",
        annotation_font_color="#69f0ae",
    )
    fig_mc.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1117",
        font=dict(color="#e8eaf6"),
        xaxis_title="Bootstrap Sharpe Ratio",
        yaxis_title="Frequency",
        height=320, margin=dict(l=20, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Real Sharpe Ratio", f"{real_sr:.4f}")
    col_p2.metric("Bootstrap p-value",  f"{pval_mc:.4f}",
                  delta="Significant" if pval_mc < 0.05 else "Not Significant")
    col_p3.metric("95% CI",
                  f"[{boot_result['ci_95'][0]:.3f}, {boot_result['ci_95'][1]:.3f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<center style='color:#546e7a; font-size:12px;'>"
    "RegimeShift | BTC/USD Regime Trading Framework | "
    "HMM: Hamilton (1989) | Sharpe: Lo (2002) | Bootstrap: Politis & Romano (1994)"
    "</center>",
    unsafe_allow_html=True,
)
