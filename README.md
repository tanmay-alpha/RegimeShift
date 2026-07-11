# RegimeShift 📈

> A quantitative algorithmic trading backtesting framework for BTC/USD using volume-based regime detection with ATR-based trailing stops.

---

## 🧠 Strategy Overview

**RegimeShift** is a momentum + volume regime detection strategy applied to BTC daily data. It identifies market regime shifts using:

- **Volume Spikes**: Detects abnormal volume using a rolling mean + 1.5σ threshold
- **Candle Direction**: Bullish or bearish candle at the spike determines trade direction
- **ATR-Based Trailing Stop**: Dynamic trailing stop using `close ± (ATR × 2)`
- **3-Candle Exit Rule**: Exits if price moves adversely for 3 consecutive bars
- **Trend Reversal Flip**: Automatically reverses position when opposite volume spike occurs

---

## 📁 Project Structure

```
RegimeShift/
│
├── main.py                # Strategy logic & entry point
├── backtester.py          # Full backtesting engine (trades, PnL, stats, charts)
├── btc_18_22_1d.csv       # Raw BTC/USD OHLCV data (2018–2022, daily)
├── final_data.csv         # Processed data with signals & indicators
├── requirements.txt       # Python dependencies
└── Problem_statement.pdf  # Original problem statement / spec
```

---

## ⚙️ Setup & Installation

```bash
# Install dependencies
pip install -r requirements.txt

# TA-Lib requires pre-built binaries on Windows:
# Download from: https://github.com/TA-Lib/ta-lib-python
pip install TA-Lib
```

---

## 🚀 Running the Strategy

```bash
python main.py
```

This will:
1. Load BTC daily OHLCV data
2. Compute ATR(14) indicator
3. Apply the volume-regime strategy
4. Run full backtest with $1,000 initial capital (compounding enabled)
5. Output trade-by-trade PnL and aggregate statistics
6. Check for **lookahead bias** automatically
7. Generate interactive **Plotly** candlestick charts with trade regions highlighted

---

## 📊 Signals Encoding

| Signal | Meaning                  |
|--------|--------------------------|
| `0`    | HOLD (no action)         |
| `1`    | BUY / Close Short        |
| `-1`   | SELL / Close Long        |
| `2`    | Reverse: Short → Long    |
| `-2`   | Reverse: Long → Short    |

---

## 📈 Backtester Features

The `BackTester` class (`backtester.py`) provides:

- **Trade Execution**: Supports long, short, and reversal positions
- **TP/SL Engine**: Tick-by-tick stop-loss / take-profit checking
- **Compounding**: Optional capital compounding across trades
- **Statistics**:
  - Win rate, winning/losing streaks
  - Gross/Net profit, Sharpe Ratio (annualised, daily-resampled)
  - Max & Average Drawdown (%)
  - Benchmark Return (buy-and-hold comparison)
  - Holding time analysis
- **Visualisation**: Interactive Plotly charts with trade regions shaded green (long) / red (short)

---

## 📉 Risk Management

- **Trailing Stop**: `close - ATR×2` for longs, `close + ATR×2` for shorts — updated each bar
- **3-Bar Adverse Exit**: Auto-close if price moves against the position for 3 consecutive candles
- **Trend Reversal**: Flips to opposite direction on counter-trend volume spike
- **Transaction Fees**: 0.15% per trade side applied in all PnL calculations

---

## 🔍 Lookahead Bias Check

The strategy includes a built-in **lookahead bias validator** that re-runs the strategy on rolling truncated data slices and verifies that signals are identical — ensuring no future data is used in signal generation.

---

## 📦 Dependencies

| Library      | Purpose                            |
|--------------|------------------------------------|
| `pandas`     | Data manipulation                  |
| `numpy`      | Numerical computations             |
| `TA-Lib`     | Technical indicators (ATR via talib) |
| `pandas_ta`  | Additional TA indicators           |
| `matplotlib` | Static plotting                    |
| `plotly`     | Interactive charting               |

---

## 📋 Data Format

The CSV input must contain the following columns:

```
datetime, open, high, low, close, volume
```

Example:
```csv
datetime,open,high,low,close,volume
2018-01-01 05:30:00,13715.65,13818.55,12750.0,13380.0,8609.91
```

---

## 🏗️ Architecture

```
main.py
├── process_data()    → Compute indicators (ATR-14)
├── strat()           → Generate signals (vol spike + direction logic)
└── main()            → Orchestrate: load → process → signal → backtest → analyze

backtester.py
├── BackTester        → Core engine
│   ├── get_trades()       → Execute trades from signals
│   ├── check_tp_sl()      → TP/SL evaluation per bar
│   ├── get_statistics()   → Comprehensive performance metrics
│   ├── calc_capital()     → Build capital curve
│   ├── get_sharpe_ratio() → Annualised Sharpe (daily-resampled)
│   ├── make_pnl_graph()   → Interactive Plotly chart
│   └── plot_drawdown()    → Drawdown visualisation
├── TradePair         → Individual trade record
├── Position          → Open position tracker
└── TradeType         → Enum: LONG / SHORT
```

---

## 📝 License

MIT License — open for research and educational use.
