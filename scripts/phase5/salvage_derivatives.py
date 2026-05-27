"""
Derivatives Salvage Script
==========================
Integrates set3 5m parquets (SOLUSDT, BNBUSDT) into the Phase 5
derivatives_microstructure layer by appending new symbol rows to each
existing monthly parquet.

What this does:
  - Adds SOLUSDT rows (from Aug 2020 listing) with real funding_rate
  - Adds BNBUSDT rows (from Jun 2020 listing) with real funding_rate
  - Does NOT modify existing BTC or ETH rows
  - All new rows: is_synthetic=True (OI data unavailable in set3)

Run once from workspace root:
    .venv\\Scripts\\python.exe salvage_derivatives.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

SET3_DIR = pathlib.Path(
    "Dataset/phase5_time_machine_dataset/another_project_raw/raw_data/raw_data_set3"
)
DERIV_DIR = pathlib.Path(
    "Dataset/phase5_time_machine_dataset/derivatives_microstructure"
)

PHASE5_START = pd.Timestamp("2013-01-01", tz="UTC")
PHASE5_END   = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")

# Symbols to add (not currently in derivatives_microstructure)
ADD_SYMBOLS = ["SOLUSDT", "BNBUSDT"]

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _to_ms(series: pd.Series) -> pd.Series:
    """Convert datetime64[ns, UTC] Series → int64 milliseconds UTC."""
    return ((series - _EPOCH) // pd.Timedelta(milliseconds=1)).astype("int64")


def load_and_resample(symbol: str) -> pd.DataFrame:
    """
    Load set3 5-minute parquet for *symbol*, resample to 1-hour bars,
    and return a DataFrame that matches the derivatives_microstructure schema.
    OI columns are zeroed (set3 open_interest is all-zero).
    Liquidation / options columns are NaN.
    """
    path = SET3_DIR / f"{symbol}.parquet"
    if not path.exists():
        print(f"  {symbol}: NOT FOUND in set3 — skipping")
        return pd.DataFrame()

    print(f"  Loading {symbol} from set3 …", flush=True)
    df = pd.read_parquet(
        path, columns=["timestamp", "close", "funding_rate", "open_interest"]
    )

    # Filter to Phase 5 window
    df = df[
        (df["timestamp"] >= PHASE5_START) & (df["timestamp"] <= PHASE5_END)
    ].copy()

    if df.empty:
        print(f"  {symbol}: no rows in 2013-2024 window")
        return pd.DataFrame()

    # Resample 5m → 1H
    df = df.set_index("timestamp")
    df_h = (
        df.resample("1h")
        .agg({"close": "last", "funding_rate": "last", "open_interest": "last"})
        .dropna(subset=["close"])
    )

    # Build the derivatives_microstructure schema
    ts_ms = _to_ms(df_h.index).values

    result = pd.DataFrame(
        {
            "timestamp_utc":          ts_ms,
            "symbol":                 symbol,
            "funding_rate":           df_h["funding_rate"].fillna(0.0).astype("float64").values,
            "oi_usd":                 0.0,
            "oi_change_usd":          0.0,
            "long_liquidations_usd":  float("nan"),
            "short_liquidations_usd": float("nan"),
            "total_liquidations_usd": float("nan"),
            "liq_long_short_ratio":   float("nan"),
            "options_max_pain_usd":   float("nan"),
            "put_call_ratio":         float("nan"),
            "basis_pct":              0.0,
            "is_synthetic":           True,
        }
    )

    fr_nonzero = int((result["funding_rate"] != 0.0).sum())
    print(
        f"  {symbol}: {len(result):,} hourly rows  "
        f"fr_nonzero={fr_nonzero:,}  "
        f"({df_h.index.min().date()} → {df_h.index.max().date()})"
    )
    return result


def main() -> None:
    # ── 1. Load & resample each new symbol ────────────────────────────────────
    print("=== Loading set3 data ===")
    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in ADD_SYMBOLS:
        df = load_and_resample(sym)
        if not df.empty:
            symbol_dfs[sym] = df

    if not symbol_dfs:
        print("No usable data found. Exiting.")
        sys.exit(0)

    # ── 2. Process each existing monthly parquet ───────────────────────────────
    deriv_files = sorted(DERIV_DIR.glob("derivatives_*.parquet"))
    print(f"\n=== Processing {len(deriv_files)} existing monthly parquets ===")

    written = skipped = rows_added = 0

    for fpath in deriv_files:
        # Parse YYYY and MM from "derivatives_YYYY_MM"
        parts = fpath.stem.split("_")   # ['derivatives', 'YYYY', 'MM']
        year, month = int(parts[1]), int(parts[2])

        existing = pd.read_parquet(fpath)
        existing_symbols = set(existing["symbol"].unique())

        # Compute month timestamp range in ms
        month_start_ms = int(
            pd.Timestamp(year=year, month=month, day=1, tz="UTC").timestamp() * 1000
        )
        if month < 12:
            month_end_ms = int(
                pd.Timestamp(year=year, month=month + 1, day=1, tz="UTC").timestamp() * 1000
            ) - 1
        else:
            month_end_ms = int(
                pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC").timestamp() * 1000
            ) - 1

        new_chunks: list[pd.DataFrame] = []
        for sym, df in symbol_dfs.items():
            if sym in existing_symbols:
                continue  # already present — do not duplicate

            chunk = df[
                (df["timestamp_utc"] >= month_start_ms)
                & (df["timestamp_utc"] <= month_end_ms)
            ].copy()
            if not chunk.empty:
                new_chunks.append(chunk)

        if not new_chunks:
            skipped += 1
            continue

        # Merge, sort, fix dtypes, write
        combined = pd.concat([existing] + new_chunks, ignore_index=True)
        combined = combined.sort_values(
            ["timestamp_utc", "symbol"]
        ).reset_index(drop=True)

        # Ensure consistent dtypes
        float_cols = [
            "funding_rate", "oi_usd", "oi_change_usd",
            "long_liquidations_usd", "short_liquidations_usd",
            "total_liquidations_usd", "liq_long_short_ratio",
            "options_max_pain_usd", "put_call_ratio", "basis_pct",
        ]
        for col in float_cols:
            combined[col] = combined[col].astype("float64")
        combined["timestamp_utc"] = combined["timestamp_utc"].astype("int64")
        combined["is_synthetic"]  = combined["is_synthetic"].astype("bool")

        combined.to_parquet(fpath, compression="zstd", compression_level=3, index=False)

        added = sum(len(c) for c in new_chunks)
        rows_added += added
        written += 1
        new_syms = sorted(set(combined["symbol"].unique()) - set(existing["symbol"].unique()))
        print(
            f"  {fpath.name}: +{added:4d} rows  "
            f"added={new_syms}  "
            f"total_symbols={sorted(combined['symbol'].unique())}"
        )

    print(f"\n=== DONE ===")
    print(f"  Files updated  : {written}")
    print(f"  Files skipped  : {skipped}  (all target symbols already present)")
    print(f"  Total new rows : {rows_added:,}")


if __name__ == "__main__":
    main()
