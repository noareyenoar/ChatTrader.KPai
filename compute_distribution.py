import os
import pandas as pd
import numpy as np
import json

def get_accepted_symbols(manifest_path, limit=10):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    symbols = []
    for entry in manifest.get("symbols", []):
        if entry.get("status") == "PASS":
            symbols.append(entry.get("symbol"))
            if len(symbols) >= limit:
                break
    return symbols

def calculate_distribution(data_dir, symbols, horizon=5, flat_threshold=0.0003):
    global_counts = {0: 0, 1: 0, 2: 0} # 0: FLAT, 1: LONG, 2: SHORT
    symbol_stats = {}
    
    for symbol in symbols:
        # Check for both symbol.parquet and symbol.csv just in case, though manifest/listing says parquet
        path = os.path.join(data_dir, f"{symbol}.parquet")
        if not os.path.exists(path):
            continue
            
        try:
            df = pd.read_parquet(path, columns=["close", "timestamp"])
            # Target logic: percentage change over horizon
            # return = (close[t+horizon] / close[t]) - 1
            # label 1 (LONG) if return > flat_threshold
            # label 2 (SHORT) if return < -flat_threshold
            # label 0 (FLAT) otherwise
            
            future_close = df["close"].shift(-horizon)
            pct_change = (future_close / df["close"]) - 1
            
            labels = np.zeros(len(df), dtype=int)
            labels[pct_change > flat_threshold] = 1
            labels[pct_change < -flat_threshold] = 2
            
            # Remove NaNs from the end where we don't have future data
            labels = labels[:-horizon]
            
            counts = pd.Series(labels).value_counts().to_dict()
            symbol_stats[symbol] = {0: counts.get(0, 0), 1: counts.get(1, 0), 2: counts.get(2, 0)}
            
            for k in global_counts:
                global_counts[k] += symbol_stats[symbol][k]
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            
    return symbol_stats, global_counts

manifest_path = r"Dataset\binance_historical\manifest.json"
data_dir = r"Dataset\binance_historical"
symbols = get_accepted_symbols(manifest_path)

print(f"Analyzing {len(symbols)} symbols: {symbols}")

symbol_stats, global_counts = calculate_distribution(data_dir, symbols)

print("\nPer-Symbol Counts (0: FLAT, 1: LONG, 2: SHORT):")
for symbol, counts in symbol_stats.items():
    total = sum(counts.values())
    print(f"{symbol}: {counts} Total: {total}")

global_total = sum(global_counts.values())
print("\nGlobal Totals:")
for label, count in global_counts.items():
    label_name = {0: "FLAT", 1: "LONG", 2: "SHORT"}[label]
    percentage = (count / global_total * 100) if global_total > 0 else 0
    print(f"{label_name} ({label}): {count} ({percentage:.2f}%)")
