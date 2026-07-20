# config.py
# Strategy configuration parameters for RegimeShift
# All hyperparameters live here — never hardcode values in src/

# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
DATA_PATH   = "btc_18_22_1d.csv"
OUTPUT_PATH = "final_data.csv"
SYMBOL      = "BTC"

# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
ATR_LENGTH          = 14     # Wilder ATR period
VOLUME_WINDOW       = 20     # Rolling window for volume z-score (was 5 — too noisy)
VOLUME_STD_MULTIPLIER = 1.0  # Spike threshold: mean + 1.0σ (generates 100+ trades)

# ─────────────────────────────────────────────
# Volume-Spike Strategy
# ─────────────────────────────────────────────
TRAILING_STOP_MULTIPLIER  = 2.0   # ATR × 2 Chandelier Exit (LeBeau 2000)
CONSECUTIVE_ADVERSE_BARS  = 3     # Close position after N adverse bars

# ─────────────────────────────────────────────
# Hidden Markov Model (Gaussian HMM)
# Reference: Hamilton (1989), Baum et al. (1970)
# ─────────────────────────────────────────────
N_REGIMES          = 3    # Bull | Bear | Crisis
HMM_WINDOW         = 252  # Rolling window for HMM feature computation (1 trading year)
HMM_ITER           = 100  # Max EM iterations for Baum-Welch convergence
HMM_RANDOM_STATE   = 42   # Reproducibility seed
REGIME_PERSISTENCE = 3    # Consecutive days required to confirm a regime flip

# ─────────────────────────────────────────────
# Regime-Conditional Position Sizing
# ─────────────────────────────────────────────
BULL_LONG_ONLY   = True   # In Bull regime: only take LONG volume spikes
BEAR_SHORT_ONLY  = True   # In Bear regime: only take SHORT volume spikes
CRISIS_SIZE_FRAC = 0.5    # In Crisis regime: 50% of normal position size

# ─────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────
INITIAL_CAPITAL  = 1000.0  # Starting capital in USD
COMPOUND_FLAG    = 1        # 1 = compound capital across trades, 0 = fixed size
TRANSACTION_FEE  = 0.0015   # 0.15% per side (Binance taker fee approximation)

# ─────────────────────────────────────────────
# Performance Statistics
# Reference: Lo (2002), Sortino & van der Meer (1991)
# ─────────────────────────────────────────────
ANNUALIZATION_FACTOR = 365   # Crypto trades 365 days/year (not 252)
RISK_FREE_RATE       = 0.05  # Annual risk-free rate (US T-bill approx)
MAR                  = 0.0   # Minimum Acceptable Return for Sortino ratio
KELLY_FRACTION       = 0.5   # Fractional Kelly (half-Kelly for safety)

# ─────────────────────────────────────────────
# Walk-Forward Validation
# ─────────────────────────────────────────────
WALK_FORWARD_TRAIN_YEARS = 2    # 2018-2020 in-sample
WALK_FORWARD_TEST_YEARS  = 1    # 2021-2022 out-of-sample

# ─────────────────────────────────────────────
# Monte Carlo Bootstrap (Politis & Romano 1994)
# ─────────────────────────────────────────────
MONTE_CARLO_RUNS   = 5000   # Number of bootstrap iterations
BLOCK_SIZE         = 21     # Block size for block bootstrap (≈ 1 month)
