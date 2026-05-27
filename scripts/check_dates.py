import pandas as pd
from pathlib import Path

files = [
    "Dataset/binance_historical/BTCUSD.parquet",
    "Dataset/binance_historical/SOLUSDT.parquet",
    "Dataset/binance_historical/ETHUSDT.parquet",
    "Dataset/binance_historical/BNBUSDT.parquet",
]
for f in files:
    p = Path(f)
    if p.exists():
        d = pd.read_parquet(f, columns=["timestamp"])
        print(f"{p.stem}: rows={len(d)} first={d['timestamp'].min()} last={d['timestamp'].max()}")
    else:
        print(f"{p.stem}: NOT FOUND")
