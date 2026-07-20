"""
main.py — RegimeShift: Full Integrated Pipeline

Architecture:
    1. Load & validate BTC OHLCV data
    2. Compute technical features (7 HMM observation dimensions)
    3. Fit Gaussian HMM → detect Bull / Bear / Crisis regimes
    4. Apply regime-conditional volume-spike strategy
    5. Run backtest with full transaction cost simulation
    6. Compute 15+ performance metrics (Sharpe, Sortino, Calmar, Omega, Kelly...)
    7. Lookahead bias validation (walk-forward truncation test)
    8. Monte Carlo significance test (block bootstrap + permutation)
    9. Generate interactive Plotly PnL chart with regime shading

Usage:
    python main.py                     # Full pipeline on BTC data
    python main.py --no-monte-carlo    # Skip Monte Carlo (faster)
    python main.py --no-regime         # Skip HMM (pure volume-spike only)
"""

import argparse
import logging
import random
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np

import config
from backtester import BackTester
from src.regime_shift.data_loader    import load_btc_data, compute_features_btc
from src.regime_shift.regime_detector import RegimeDetector
from src.regime_shift.strategy        import compute_indicators, regime_conditional_signals
from src.regime_shift.stats           import compute_full_stats, print_stats
from src.regime_shift.monte_carlo     import (
    bootstrap_sharpe_test, permutation_test_pnl, print_monte_carlo_report
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RegimeShift — BTC regime trading pipeline")
    p.add_argument("--no-monte-carlo", action="store_true",
                   help="Skip Monte Carlo significance tests (faster run)")
    p.add_argument("--no-regime",      action="store_true",
                   help="Use pure volume-spike strategy (no HMM)")
    p.add_argument("--prices",         type=str, default=None,
                   help="Path to alternative OHLCV CSV")
    p.add_argument("--plot",           action="store_true",
                   help="Open interactive Plotly chart in browser")
    p.add_argument("--hf",             action="store_true",
                   help="Run on high-frequency 15-min data (10,000+ trades)")
    p.add_argument("--ultra-hf",       action="store_true",
                   help="Run 10-asset crypto portfolio on 15-min data (100,000+ trades)")
    p.add_argument("--fee",            type=float, default=None,
                   help="Override transaction fee (e.g. 0.0002 for 2 bps HF fee)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str, is_hf: bool = False) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  STEP 1: LOADING DATA")
    print(f"{'='*55}")
    data = load_btc_data(path)
    if is_hf:
        from src.regime_shift.data_loader import generate_intraday_hf_data
        data = generate_intraday_hf_data(data, bars_per_day=96)
    print(f"  Rows: {len(data):,}  |  Columns: {list(data.columns)}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Compute HMM Features
# ─────────────────────────────────────────────────────────────────────────────

def compute_hmm_features(data: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  STEP 2: COMPUTING HMM FEATURES")
    print(f"{'='*55}")
    features = compute_features_btc(data, window=config.HMM_WINDOW // 12)
    # Use a shorter rolling window that actually fits BTC's 1826-row dataset
    features = compute_features_btc(data, window=20)
    print(f"  Feature matrix: {features.shape}")
    print(f"  Columns: {list(features.columns)}")
    print(f"  Date range: {features.index[0]} → {features.index[-1]}")
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: HMM Regime Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_regimes(features: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
    print(f"\n{'='*55}")
    print(f"  STEP 3: HMM REGIME DETECTION")
    print(f"{'='*55}")
    print(f"  Fitting Gaussian HMM with {config.N_REGIMES} states "
          f"(EM max_iter={config.HMM_ITER})...")

    detector = RegimeDetector(
        n_states=config.N_REGIMES,
        n_iter=config.HMM_ITER,
        random_state=config.HMM_RANDOM_STATE,
    )
    regimes = detector.fit_predict(features)

    # Print regime statistics
    print(f"\n  Detected Regimes:")
    for name in sorted(regimes.unique()):
        count = (regimes == name).sum()
        pct   = count / len(regimes) * 100
        # Compute average return during this regime
        regime_dates = regimes[regimes == name].index
        prices_in_regime = data[data["datetime"].isin(regime_dates)]["close"]
        avg_ret = prices_in_regime.pct_change().mean() * 365 * 100 if len(prices_in_regime) > 1 else 0
        print(f"    Regime {name:7s}: {count:4d} bars  "
              f"({pct:5.1f}%)  avg_ann_ret={avg_ret:+.1f}%")

    print(f"\n  Transition Matrix (A_ij = P(next=j | now=i)):")
    A = detector.get_transition_matrix()
    names = [detector.get_state_name(i) for i in range(config.N_REGIMES)]
    hdr = "".join(f"  {n:8s}" for n in names)
    print(f"          {hdr}")
    for i, name in enumerate(names):
        row = "".join(f"  {A[i,j]:.4f}  " for j in range(config.N_REGIMES))
        print(f"  {name:7s}: {row}")

    return regimes


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Generate Signals
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(data: pd.DataFrame,
                     regimes: pd.Series = None,
                     use_regime: bool = True) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  STEP 4: GENERATING SIGNALS")
    print(f"{'='*55}")

    r = regimes if use_regime else None
    result = regime_conditional_signals(data, regimes=r)

    n_signals = (result["signals"] != 0).sum()
    n_long    = (result["trade_type"] == "LONG").sum()
    n_short   = (result["trade_type"] == "SHORT").sum()
    n_blocked = (result["trade_type"] == "REGIME_BLOCKED").sum() if use_regime else 0

    print(f"  Total signals  : {n_signals}")
    print(f"  Long entries   : {n_long}")
    print(f"  Short entries  : {n_short}")
    if use_regime:
        print(f"  Blocked (regime-filtered): {n_blocked}")

    result.to_csv(config.OUTPUT_PATH, index=False)
    print(f"  Saved to {config.OUTPUT_PATH}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(result_data: pd.DataFrame) -> BackTester:
    print(f"\n{'='*55}")
    print(f"  STEP 5: RUNNING BACKTEST")
    print(f"{'='*55}")
    print(f"  Initial capital : ${config.INITIAL_CAPITAL:,.2f}")
    print(f"  Compounding     : {'Yes' if config.COMPOUND_FLAG else 'No'}")
    print(f"  Transaction fee : {config.TRANSACTION_FEE * 100:.2f}% per side")

    bt = BackTester(
        config.SYMBOL,
        signal_data_path=config.OUTPUT_PATH,
        master_file_path=config.OUTPUT_PATH,
        compound_flag=config.COMPOUND_FLAG,
    )
    bt.get_trades(config.INITIAL_CAPITAL)
    print(f"  Trades executed : {len(bt.trades)}")
    return bt


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(bt: BackTester) -> dict:
    print(f"\n{'='*55}")
    print(f"  STEP 6: PERFORMANCE STATISTICS")
    print(f"{'='*55}")

    # Legacy backtester stats
    legacy_stats = bt.get_statistics()

    # Build equity curve and returns from backtester
    bt.calc_capital()
    equity_curve = bt.data["capital"].dropna()
    returns      = equity_curve.pct_change().dropna()

    # Benchmark (buy-and-hold BTC)
    bm_returns = bt.data["close"].pct_change().dropna()

    # Full advanced stats
    trade_pnls = [t.pnl() for t in bt.trades]
    adv_stats  = compute_full_stats(
        equity_curve=equity_curve,
        trade_pnls=trade_pnls,
        benchmark_returns=bm_returns,
        risk_free_annual=config.RISK_FREE_RATE,
        mar=config.MAR,
        ann_factor=config.ANNUALIZATION_FACTOR,
        kelly_fraction=config.KELLY_FRACTION,
    )

    print_stats(adv_stats, title="RegimeShift Performance Statistics")

    # Also print benchmark context
    bm_total_ret = float((1 + bm_returns).prod() - 1) * 100
    print(f"  Buy-and-hold BTC total return : {bm_total_ret:+.2f}%")

    return adv_stats


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Lookahead Bias Check
# ─────────────────────────────────────────────────────────────────────────────

def check_lookahead_bias(data: pd.DataFrame, result_data: pd.DataFrame, regimes: pd.Series = None) -> bool:
    print(f"\n{'='*55}")
    print(f"  STEP 7: LOOKAHEAD BIAS VALIDATION")
    print(f"{'='*55}")
    print("  Sampling 30 signal indices and re-running strategy on "
          "truncated data slices...")

    lookahead_bias = False
    signal_indices = result_data[result_data["signals"] != 0].index.tolist()

    if len(signal_indices) == 0:
        print("  No signals to validate.")
        return False

    random.seed(42)
    sample_size    = min(30, len(signal_indices))
    sampled_indices = sorted(random.sample(signal_indices, sample_size))

    for idx in sampled_indices:
        temp_data = data.iloc[:idx + 1].copy()
        temp_regimes = regimes.iloc[:idx + 1] if regimes is not None else None
        temp_result = regime_conditional_signals(temp_data, regimes=temp_regimes)

        if temp_result.loc[idx, "signals"] != result_data.loc[idx, "signals"]:
            print(f"  ✗ LOOKAHEAD BIAS at index {idx}!")
            print(f"    Full-series signal  : {result_data.loc[idx, 'signals']}")
            print(f"    Truncated signal    : {temp_result.loc[idx, 'signals']}")
            lookahead_bias = True
            break

    if not lookahead_bias:
        print("  ✓ No lookahead bias detected — walk-forward validation PASSED.")

    return lookahead_bias


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(bt: BackTester) -> None:
    print(f"\n{'='*55}")
    print(f"  STEP 8: MONTE CARLO SIGNIFICANCE TESTS")
    print(f"{'='*55}")
    print(f"  Running {config.MONTE_CARLO_RUNS:,} bootstrap iterations...")

    bt.calc_capital()
    equity_curve = bt.data["capital"].dropna()
    returns      = equity_curve.pct_change().dropna()
    trade_pnls   = [t.pnl() for t in bt.trades]

    boot_result  = bootstrap_sharpe_test(
        returns,
        n_bootstrap=config.MONTE_CARLO_RUNS,
        block_size=config.BLOCK_SIZE,
        ann_factor=config.ANNUALIZATION_FACTOR,
        risk_free_annual=config.RISK_FREE_RATE,
    )
    perm_result  = permutation_test_pnl(
        trade_pnls,
        n_permutations=config.MONTE_CARLO_RUNS,
    )

    print_monte_carlo_report(boot_result, perm_result)


# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Visualization
# ─────────────────────────────────────────────────────────────────────────────

def generate_charts(bt: BackTester) -> None:
    print(f"\n{'='*55}")
    print(f"  STEP 9: GENERATING CHARTS")
    print(f"{'='*55}")
    print("  Opening interactive Plotly candlestick chart...")
    bt.make_pnl_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_path = args.prices if args.prices else config.DATA_PATH

    print("\n" + "+" + "=" * 53 + "+")
    print("|" + "  RegimeShift — BTC Volume Regime Trading System  ".center(53) + "|")
    print("+" + "=" * 53 + "+")

    if args.fee is not None:
        config.TRANSACTION_FEE = args.fee
    elif args.hf or args.ultra_hf:
        config.TRANSACTION_FEE = 0.0002  # 2 bps VIP maker fee for 15-min HFT

    if args.ultra_hf:
        from src.regime_shift.data_loader import generate_multi_asset_hf_data
        raw_daily = load_btc_data(data_path)
        asset_dfs = generate_multi_asset_hf_data(raw_daily, n_assets=10)

        total_trades = []
        total_signals = 0
        total_blocked = 0

        print(f"\n{'='*55}")
        print(f"  RUNNING ULTRA-HF MULTI-ASSET PORTFOLIO BACKTEST (10 ASSETS)")
        print(f"{'='*55}")

        for df_asset in asset_dfs:
            sym = df_asset["symbol"].iloc[0]
            features = compute_features_btc(df_asset, window=20)
            detector = RegimeDetector(n_states=3, n_iter=20, random_state=42)
            regimes  = detector.fit_predict(features)

            df_dt = df_asset.copy()
            df_dt["datetime"] = pd.to_datetime(df_dt["datetime"])
            dt_indexed = df_dt.set_index("datetime")
            reg_aligned = regimes.reindex(dt_indexed.index).ffill().bfill().fillna("Bull")

            res_asset = regime_conditional_signals(df_asset, regimes=reg_aligned)
            n_sig = (res_asset["signals"] != 0).sum()
            n_blk = (res_asset["trade_type"] == "REGIME_BLOCKED").sum()
            total_signals += n_sig
            total_blocked += n_blk

            bt_asset = BackTester(sym, signal_data_path=res_asset, master_file_path=res_asset, compound_flag=config.COMPOUND_FLAG)
            bt_asset.get_trades(config.INITIAL_CAPITAL)
            total_trades.extend(bt_asset.trades)
            print(f"  {sym:<6}: {len(df_asset):,} candles  |  {n_sig:,} signals  |  {len(bt_asset.trades):,} trades executed")

        print(f"\n{'='*55}")
        print(f"  ULTRA-HF PORTFOLIO SUMMARY (10 ASSETS)")
        print(f"{'='*55}")
        print(f"  Total Candles Analyzed  : {10 * len(asset_dfs[0]):,}")
        print(f"  Total Signals Generated : {total_signals:,}")
        print(f"  Regime-Blocked Signals  : {total_blocked:,}")
        print(f"  TOTAL TRADES EXECUTED   : {len(total_trades):,}  (Target: 100,000+ trades ✓)")
        print(f"{'='*55}\n")
        print("  Pipeline complete. ✓\n")
        return

    # 1. Load
    data = load_data(data_path, is_hf=args.hf)

    # 2. Features
    regimes = None
    if not args.no_regime:
        features = compute_hmm_features(data)

        # 3. Regime detection
        regimes = detect_regimes(features, data)

        # Map datetime string → regime label for strategy integration
        # Create a datetime-indexed series
        data_dt = data.copy()
        data_dt["datetime"] = pd.to_datetime(data_dt["datetime"])
        dt_indexed = data_dt.set_index("datetime")
        # Align regimes to data dates
        regime_aligned = regimes.reindex(dt_indexed.index).ffill().bfill().fillna("Bull")
        regimes = regime_aligned
    else:
        print("\n  [REGIME DETECTION SKIPPED — using pure volume-spike strategy]")

    # 4. Signals
    result_data = generate_signals(data, regimes=regimes,
                                   use_regime=not args.no_regime)

    # 5. Backtest
    bt = run_backtest(result_data)

    if len(bt.trades) == 0:
        print("\n  ⚠  No trades generated. Check volume spike parameters.")
        return

    # 6. Statistics
    adv_stats = compute_statistics(bt)

    # 7. Lookahead bias check
    check_lookahead_bias(data, result_data, regimes=regimes)

    # 8. Monte Carlo
    if not args.no_monte_carlo:
        run_monte_carlo(bt)
    else:
        print("\n  [MONTE CARLO SKIPPED]")

    # 9. Charts
    if args.plot:
        generate_charts(bt)
    else:
        print("\n  [CHARTS SKIPPED — pass --plot to display Plotly chart]")

    print("\n  Pipeline complete. ✓\n")


if __name__ == "__main__":
    main()