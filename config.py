# config.py
# Strategy configuration parameters for RegimeShift

DATA_PATH = "btc_18_22_1d.csv"
OUTPUT_PATH = "final_data.csv"

# Strategy Parameters
ATR_LENGTH = 14
TRAILING_STOP_MULTIPLIER = 2.0
VOLUME_WINDOW = 5
VOLUME_STD_MULTIPLIER = 1.5
CONSECUTIVE_ADVERSE_BARS = 3

# Backtester Parameters
INITIAL_CAPITAL = 1000.0
COMPOUND_FLAG = 1
SYMBOL = "BTC"
