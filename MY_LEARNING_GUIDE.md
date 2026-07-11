# 📖 RegimeShift — Simple Explanation (For Me, By Me)

> *Written for Tanmay — someone who loves the Indian stock market, trades a bit,
> and is building a career in Quant + AI/ML. No jargon. Just plain talk.*

---

## 🤔 First — What Even Is "Quant"?

You know how when you trade on **Zerodha** or **Groww**, you look at charts, maybe check
RSI or moving averages, and then *feel* like it's a good time to buy or sell?

**Quant trading** is the same thing — but instead of *you* looking at the chart and
deciding with your gut, you write a **computer program** that does it automatically,
based on math and logic.

No emotions. No "I feel like Reliance will go up today."
Just pure math + data + code.

> **Think of it like this:**
> - Normal trading = You are the trader
> - Quant trading = You write a robot trader, and the robot does the trading for you

That's it. That's quant.

---

## 📈 What Is This Project — RegimeShift?

Okay so imagine you are watching the **Nifty 50** chart all day.

Sometimes the market is **calm** — prices move slowly, nothing dramatic.
Sometimes the market goes into a **frenzy** — huge moves, everyone panicking or
euphoric.

These two situations are called **market regimes** (like "modes" the market is in).

**RegimeShift** is a program that tries to **detect when the market switches from
calm mode to frenzy mode** — and then place a trade to profit from that switch.

### How does it detect the switch?

By watching **Volume**.

Volume = how many shares/coins were traded in a time period.

> **Indian market example:**
> On a normal day, maybe 10 lakh shares of Infosys are traded.
> Suddenly one day — 50 lakh shares trade in an hour.
> *Something big is happening.* Big institutions are buying or selling.
> The price is about to move BIG.

Our program watches for exactly this — a **volume spike** — and then asks:
- Was that spike bullish (price went UP)? → **BUY**
- Was that spike bearish (price went DOWN)? → **SELL**

Right now we are testing this on **Bitcoin (BTC)** because BTC data is freely
available and easy to work with. But the same idea works on Nifty, Bank Nifty,
individual stocks — any market.

---

## 🗂️ What's In The Project Right Now?

Think of the project as having two main parts:

### Part 1: The Strategy (`main.py`)
This is the *brain* — the actual trading logic.

```
It reads BTC price data
→ Calculates ATR (a measure of how volatile the price is)
→ Watches for volume spikes
→ Decides: BUY, SELL, or DO NOTHING
→ Saves those decisions to a file
```

### Part 2: The Backtester (`backtester.py`)
This is the *testing machine*.

**Backtest** = running your strategy on *past data* to see how it would have
performed if you had actually traded it.

> **Indian analogy:**
> Imagine you invented a new rule for trading Nifty.
> Before risking real money, you go back to 2018 and pretend to trade every day
> using your rule — on paper. At the end you count: did you make money?
> That's a backtest.

Our backtester:
- Takes the BUY/SELL signals from the strategy
- Simulates actual trades (including 0.15% brokerage fee — like Zerodha's charges)
- Calculates profit/loss for each trade
- Gives a full performance report
- Makes beautiful interactive charts showing where you bought and sold

---

## 📊 What Does The Performance Report Tell Us?

When the backtest runs, it gives us numbers like:

| Metric | What It Means | Indian Analogy |
|--------|--------------|----------------|
| **Win Rate** | % of trades that made money | Out of 10 trades, how many were profitable? |
| **Net Profit** | Total money made/lost | Starting ₹1000, ending ₹1400 = ₹400 profit |
| **Max Drawdown** | Worst losing streak | "My portfolio fell 30% from peak before recovering" |
| **Sharpe Ratio** | Returns vs Risk | Higher = better returns for the risk taken |
| **Benchmark Return** | Buy-and-hold comparison | "Did my strategy beat just buying and holding BTC?" |

---

## 🎓 What Are You Actually Learning From This Project?

This is the important part — what skills this project teaches you:

### 📌 Level 1 — Python & Data (Beginner)
- Reading and cleaning financial data with **pandas**
- Math operations with **numpy**
- Understanding OHLCV data (Open, High, Low, Close, Volume)
- Writing functions and classes in Python

### 📌 Level 2 — Finance & Trading Logic (Intermediate)
- What is ATR and why traders use it for stop-losses
- What volume spikes mean (institutional activity)
- How trailing stops work (protecting profits automatically)
- What position sizing is (how much to buy)
- Long vs Short positions (profiting from UP or DOWN moves)

### 📌 Level 3 — Quantitative Methods (Advanced)
- How to write a proper backtest (most people do it wrong)
- What lookahead bias is and why it destroys backtest results
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Drawdown analysis
- Walk-forward testing (testing on data the strategy never "saw")

### 📌 Level 4 — AI/ML for Trading (Your Career Path)
- Using **Hidden Markov Models** to detect market regimes automatically
- **Clustering** different market conditions (k-means on price patterns)
- **Neural Networks** to predict signal quality
- **Reinforcement Learning** — teaching an AI to trade by itself

---

## 🛠️ Tech Stack — What You'll Learn To Use

| Tool | What It Is | Why We Use It |
|------|-----------|---------------|
| **Python** | Programming language | The main language for quant/finance globally |
| **pandas** | Data manipulation library | Like Excel but 1000x more powerful — handles price data |
| **numpy** | Math library | Fast numerical calculations |
| **TA-Lib / pandas_ta** | Technical indicators | Calculates ATR, RSI, MACD etc. automatically |
| **Plotly** | Interactive charts | Beautiful charts you can zoom into |
| **Matplotlib** | Static charts | Standard plots for analysis |
| **FastAPI** | Build APIs | Create a web server that sends trading signals |
| **Streamlit** | Web dashboards | Build a live trading dashboard with no frontend coding |
| **CCXT** | Crypto exchange library | Connect to Binance, buy/sell in real life |
| **scikit-learn** | Machine Learning | For clustering, classification of market regimes |
| **statsmodels** | Statistical models | HMM, time series analysis |
| **Docker** | Containerization | Package your app so it runs anywhere |
| **Git + GitHub** | Version control | Track changes, collaborate, share code |

---

## 🗺️ The Plan — Where We Are Going

Think of this as a journey from "hobbyist backtester" to "professional quant system."

### 🔴 Phase 1 — Fix & Learn the Foundation (NOW)
*Goal: Understand what's already built and make it work correctly*

- [ ] Fix the critical bug (wrong CSV filename — strategy currently crashes!)
- [ ] Fix the daily data timestamp bug
- [ ] Understand every line of `main.py` and `backtester.py`
- [ ] Run the strategy and see actual results
- [ ] Learn: *What does each output number mean?*

**What you'll learn:** Python basics, financial data handling, debugging

---

### 🟡 Phase 2 — Make The Strategy Smarter (NEXT 1-2 MONTHS)
*Goal: Add more indicators so the strategy is more reliable*

- [ ] Add RSI (Relative Strength Index) as a confirmation signal
- [ ] Add OBV (On-Balance Volume) to confirm volume trend
- [ ] Add VWAP (Volume Weighted Average Price)
- [ ] Make all parameters configurable (not hardcoded numbers)
- [ ] Test on Indian stocks: Nifty50, Bank Nifty, Reliance, Infosys

**What you'll learn:** Technical analysis, indicator math, parameter tuning

---

### 🟢 Phase 3 — Make It a Real Research Platform (MONTHS 2-4)
*Goal: Turn this into something that looks like what real quants build*

- [ ] Build a CLI tool (run strategy from command line with options)
- [ ] Add walk-forward optimization (test on unseen data)
- [ ] Monte Carlo simulation (stress-test the strategy)
- [ ] Build a Streamlit dashboard (visual, interactive, beautiful)
- [ ] Add Telegram alerts when a new signal is generated

**What you'll learn:** Software engineering, statistical testing, dashboards

---

### 🔵 Phase 4 — Add AI/ML (MONTHS 4-6)
*Goal: This is where quant meets your AI/ML career goal*

- [ ] Train a Hidden Markov Model to auto-detect market regimes
- [ ] Use K-Means clustering to group similar market conditions
- [ ] Build a signal quality predictor (ML model that scores each signal)
- [ ] Implement Kelly Criterion for intelligent position sizing
- [ ] Train a Reinforcement Learning agent to optimize exit timing

**What you'll learn:** Machine learning applied to finance, the core of your career

---

### 🟣 Phase 5 — Go Live (MONTHS 6+)
*Goal: Run on real money (paper trading first, then live)*

- [ ] Connect to Zerodha API (Kite Connect) for Indian stocks
- [ ] Connect to Binance via CCXT for crypto
- [ ] Paper trade for 3 months — compare real results vs backtest
- [ ] Go live with small capital
- [ ] Monitor risk in real-time

**What you'll learn:** Production systems, risk management, live trading ops

---

## 💡 Key Concepts You Must Understand Deeply

### 1. What is "Lookahead Bias"? (The Most Dangerous Bug in Trading)
Imagine you're backtesting a strategy for January 2020.
If your code accidentally uses data from March 2020 (which didn't exist in January),
your results will look amazing — but it's fake. The strategy "cheated" by
knowing the future.

Our code already checks for this — which is actually rare and impressive.

> **Indian life analogy:** It's like saying "I knew Jio would be huge" — in 2024.
> Of course you knew! You're looking back. Did you actually buy Jio stock in 2015?
> That's the real question.

---

### 2. What is Drawdown?
If your ₹1 lakh portfolio goes to ₹1.5 lakh (peak) and then falls to ₹90,000 —
that fall from peak to trough is **drawdown** = 40%.

Most traders focus only on profits. Professional quants obsess over **controlling drawdown**.
A strategy that makes 50% but has 80% drawdown is dangerous and psychologically
impossible to stick with.

---

### 3. Why Sharpe Ratio Matters More Than Raw Returns

Two strategies:
- Strategy A: Makes 30% per year, very smooth, small losses
- Strategy B: Makes 35% per year, but sometimes loses 50% in a month

Strategy A has a better **Sharpe Ratio**. Most institutions would prefer A.
Raw returns mean nothing without understanding the risk taken to get them.

---

### 4. What is ATR? (Average True Range)
ATR measures how much the price moves on an average day.

If BTC's ATR is ₹2,000 — it means on an average day BTC moves ₹2,000 up or down.

We use ATR to set our stop-loss:
- If ATR = 2,000 and we use 2×ATR → stop-loss is ₹4,000 away from entry
- This means we give the trade "room to breathe" before stopping out
- High ATR = volatile market = wider stop (so we don't get stopped out by noise)

This is called an **ATR-based trailing stop** — it moves with the price to lock in profit.

> **Relatable example:**
> Imagine you buy a house for ₹50 lakh. You set a mental "stop loss" —
> if the area deteriorates and price falls to ₹40 lakh you will sell.
> But if price goes to ₹80 lakh, you update your stop to ₹70 lakh.
> You're trailing your exit point as the price rises. That's a trailing stop.

---

## 🧠 How To Actually Learn (Stop Just Vibe Coding)

Since you want to stop "vibe coding" and actually understand your own project,
here's a practical plan:

### The "Read Before You Run" Rule
Before running any code the AI writes — read it line by line and make sure you
can explain what each line does in plain English. If you can't explain it, ask.

### The "Break It Intentionally" Rule
Once code works — break it on purpose. Change a number, delete a line, see what
error comes out. This is how you understand what each part actually does.

### The "Write It Yourself" Rule
After the AI builds something — close the AI, open a blank file, and try to
rebuild a simplified version yourself from memory. It doesn't have to be perfect.
Just the attempt wires the knowledge into your brain permanently.

### One Concept Per Day (30 min)
- Day 1: What is ATR and how is it calculated by hand?
- Day 2: What does the `Position.is_valid()` function do and why is it needed?
- Day 3: How does the Sharpe ratio formula work mathematically?
- Day 4: What is the difference between gross profit and net profit in the stats?
- Day 5: Read the volume spike logic and redraw it as a flowchart on paper
- ...and so on

---

## 🎯 Why This Project Is Perfect For Your Career

You want: **Quant + AI/ML career in Indian finance**

This project teaches you exactly that — from the ground up:

```
Stock Market Knowledge  →  you already have this ✅
        +
Python Programming      →  you're building this now
        +
Financial Math          →  this project teaches you
        +
Machine Learning        →  Phase 4 of this project
        =
Quantitative Researcher / Algo Trader / ML Engineer in Finance
```

### Companies That Hire For This In India

| Company | Type | What They Look For |
|---------|------|--------------------|
| Quadeye | HFT Firm | Python, math, backtesting |
| Tower Research | HFT Firm | C++, quant strategies |
| WorldQuant | Quant Hedge Fund | Alpha research, ML |
| Graviton | Quant Firm | Python, statistics |
| Goldman Sachs India | Investment Bank | Quant desk roles |
| Zerodha | Fintech | They have a quant team! |
| Smallcase | Fintech | Strategy research |
| Sensibull | Options Fintech | Options quant |

### Salary Range (2025, India)
- Entry level: ₹15–30 LPA
- Mid level: ₹40–80 LPA
- Senior / Researcher: ₹1 Cr+

This project — if built properly to Phase 4 — is portfolio-worthy and
interview-ready at most of these companies.

---

## 📝 One Page Summary

| Question | Simple Answer |
|----------|--------------|
| **What is this project?** | A robot trader that detects when big players enter BTC and trades with them |
| **What's the core idea?** | Big volume = institutions moving = price will move big → trade that direction |
| **What tool does it use?** | Volume spikes for entries, ATR for stop-losses |
| **What have we built?** | Signal generator + backtesting engine + performance stats + charts |
| **What's broken right now?** | Wrong CSV filename (crashes), wrong timestamp for daily bars |
| **What's the next step?** | Fix bugs → run the strategy → read the results → understand each number |
| **What will you learn?** | Python, financial math, backtesting, then ML — the full quant stack |
| **Why does this matter?** | Quant + AI/ML in Indian finance = massive career opportunity in 2025+ |

---

*Last updated: July 2026 | Project: RegimeShift | Written for: Tanmay*

