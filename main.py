import pandas as pd
import numpy as np
import pandas_ta_classic as ta
from backtester import BackTester
import config
import random

def process_data(data):
    """
    Process the input data and return a dataframe with all the necessary indicators and data for making signals.

    Parameters:
    data (pandas.DataFrame): The input data to be processed.

    Returns:
    pandas.DataFrame: The processed dataframe with all the necessary indicators and data.
    """
    # Standardize index and avoid warning about setting on copy
    data = data.reset_index(drop=True).copy()
    
    # Generate the necessary indicators here
    data['ATR'] = ta.atr(data['high'], data['low'], data['close'], length=config.ATR_LENGTH)
    
    # Vectorized volume spike threshold calculation:
    # Shift by 1 to strictly use the 5 bars BEFORE the current bar i (avoids self-referential bias)
    data['vol_mean'] = data['volume'].rolling(config.VOLUME_WINDOW).mean().shift(1)
    data['vol_std'] = data['volume'].rolling(config.VOLUME_WINDOW).std().shift(1)
    data['vol_spike'] = data['vol_mean'] + config.VOLUME_STD_MULTIPLIER * data['vol_std']
    
    return data


def strat(data):
    """
    Create a strategy based on indicators or other factors.

    Parameters:
    - data: DataFrame
        The input data containing the necessary columns for strategy creation.

    Returns:
    - DataFrame
        The modified input data with an additional 'signals' column representing the strategy signals.
    """
    # Avoid warning on setting on copy
    data = data.copy()
    data['trade_type'] = "HOLD" 
    data['signals'] = 0
    position = 0 # Variable to keep track of the current position (0 = no position, 1 = long, -1 = short)

    # Example strategy parameters
    num_wrong = 0
    trailing_stop = 0  
    trailing_stop_multiplier = config.TRAILING_STOP_MULTIPLIER

    # Dynamically find the first valid index for ATR to avoid NaNs in trailing stop math
    first_valid_atr = data['ATR'].first_valid_index()
    start_idx = int(first_valid_atr) if first_valid_atr is not None else config.ATR_LENGTH

    for i in range(start_idx, len(data)):
        vol_spike = data.loc[i, 'vol_spike']
        
        # Skip if vol_spike is NaN
        if pd.isna(vol_spike):
            continue

        if position == 0:
            if data.loc[i, 'volume'] > vol_spike:
                if data.loc[i,'close'] > data.loc[i,'open']:
                    data.loc[i, 'signals'] = 1 # Buy signal
                    position = 1 # Update the position to keep track of the current position
                    data.loc[i, 'trade_type'] = "LONG"
                    trailing_stop = data.loc[i,'close'] - (data.iloc[i]["ATR"] * trailing_stop_multiplier) # Set the initial trailing stop
                    num_wrong = 0

                elif data.loc[i,'close'] < data.loc[i,'open']:
                    data.loc[i, 'signals'] = -1
                    position = -1
                    data.loc[i, 'trade_type'] = "SHORT"
                    trailing_stop = data.loc[i,'close'] + (data.iloc[i]["ATR"] * trailing_stop_multiplier)
                    num_wrong = 0

        elif position == 1: # We already have a long position
            # Check if the direction of the trend reversed
            trend_rev = data.loc[i, 'volume'] >= vol_spike and data.loc[i,'close'] < data.loc[i,'open']
            
            # Check if the price has gone down for 3 consecutive candles
            if data.loc[i, 'close'] <= data.loc[i - 1, 'close']:
                num_wrong += 1
            else:
                num_wrong = 0

            if trend_rev: # Trend reversal detected
                # Reverse the position
                data.loc[i, 'signals'] = -2
                position = -1
                trailing_stop = data.loc[i,'close'] + (data.iloc[i]["ATR"] * trailing_stop_multiplier)
                num_wrong = 0
                data.loc[i, 'trade_type'] = "REVERSE_LONG_TO_SHORT"
            elif num_wrong == config.CONSECUTIVE_ADVERSE_BARS: # Price has gone down for 3 consecutive candles
                # Close the position
                data.loc[i, 'signals'] = -1
                position = 0
                num_wrong = 0 
                data.loc[i, 'trade_type'] = "CLOSE"
            else: 
                # Check if the trailing stop has been hit
                if data.iloc[i]["close"] < trailing_stop:
                    data.loc[i, 'signals'] = -1
                    position = 0
                    num_wrong = 0
                    data.loc[i, 'trade_type'] = 'CLOSE'
                else: # Update the trailing stop
                    trailing_stop = max(trailing_stop, data.iloc[i]["close"] - (data.iloc[i]["ATR"] * trailing_stop_multiplier))
            
        elif position == -1: # We already have a short position
            # Check if the direction of the trend reversed
            trend_rev = data.loc[i, 'volume'] >= vol_spike and data.loc[i,'close'] > data.loc[i,'open']
            
            # Check if the price has gone up for 3 consecutive candles
            if data.loc[i, 'close'] >= data.loc[i - 1, 'close']:
                num_wrong += 1
            else:
                num_wrong = 0

            if trend_rev: # Trend reversal detected
                # Reverse the position
                data.loc[i, 'signals'] = 2
                position = 1
                trailing_stop = data.loc[i,'close'] - (data.iloc[i]["ATR"] * trailing_stop_multiplier)
                num_wrong = 0
                data.loc[i, 'trade_type'] = "REVERSE_SHORT_TO_LONG"
            elif num_wrong == config.CONSECUTIVE_ADVERSE_BARS: # Price has gone up for 3 consecutive candles
                # Close the position
                data.loc[i, 'signals'] = 1
                position = 0
                num_wrong = 0
                data.loc[i, 'trade_type'] = "CLOSE"
            else: 
                # Check if the trailing stop has been hit
                if data.iloc[i]["close"] > trailing_stop:
                    data.loc[i, 'signals'] = 1
                    position = 0
                    num_wrong = 0
                    data.loc[i, 'trade_type'] = 'CLOSE'
                else: # Update the trailing stop
                    trailing_stop = min(trailing_stop, data.iloc[i]["close"] + (data.iloc[i]["ATR"] * trailing_stop_multiplier))
    return data

def main():
    print(f"Loading data from {config.DATA_PATH}...")
    data = pd.read_csv(config.DATA_PATH)
    
    print("Processing indicators...")
    processed_data = process_data(data) # process the data
    
    print("Generating trading signals...")
    result_data = strat(processed_data) # Apply the strategy
    
    print(f"Saving final signals and data to {config.OUTPUT_PATH}...")
    result_data.to_csv(config.OUTPUT_PATH, index=False)

    print("Running backtest simulation...")
    bt = BackTester(config.SYMBOL, signal_data_path=config.OUTPUT_PATH, master_file_path=config.OUTPUT_PATH, compound_flag=config.COMPOUND_FLAG)
    bt.get_trades(config.INITIAL_CAPITAL)

    # Print results
    stats = bt.get_statistics()
    print("\n" + "="*40)
    print("           BACKTEST STATISTICS")
    print("="*40)
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"{key:<30} : {val:.4f}")
        else:
            print(f"{key:<30} : {val}")
    print("="*40)

    # Check for lookahead bias (optimized sample validation)
    print("\nChecking for lookahead bias (optimized 30-signal sample)...")
    lookahead_bias = False
    
    # Get all indices where a trade signal occurred (signals != 0)
    signal_indices = result_data[result_data['signals'] != 0].index.tolist()
    
    if len(signal_indices) > 0:
        # Set random seed for reproducibility
        random.seed(42)
        sample_size = min(30, len(signal_indices))
        sampled_indices = random.sample(signal_indices, sample_size)
        sampled_indices.sort()
        
        for idx in sampled_indices:
            # Re-run only up to the signal index (no future data)
            temp_data = data.iloc[:idx+1].copy()
            temp_data = process_data(temp_data)
            temp_data = strat(temp_data)
            
            if temp_data.loc[idx, 'signals'] != result_data.loc[idx, 'signals']:
                print(f"LOOKAHEAD BIAS DETECTED at index {idx}!")
                print(f"  Whole series signal: {result_data.loc[idx, 'signals']}")
                print(f"  Truncated series signal: {temp_data.loc[idx, 'signals']}")
                lookahead_bias = True
                break

    if not lookahead_bias:
        print("No lookahead bias detected. Walk-forward verification PASSED.")

    # Generate the PnL graph (removed make_trade_graph() which doesn't exist)
    print("Generating PnL visualization...")
    bt.make_pnl_graph()
    
if __name__ == "__main__":
    main()