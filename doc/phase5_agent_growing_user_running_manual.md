# Phase 5 — Time Machine Dataset  
## Agent Growing & Running Manual

**Version:** May 2026 · **Status:** Oracle PASS (all 12 layers)  
**Dataset size:** ~650 MB  ·  **Coverage:** 2013-01-01 → 2024-12-31 (144 months)

---

## Table of Contents

1. [What Is Phase 5?](#1-what-is-phase-5)
2. [Architecture Overview](#2-architecture-overview)
3. [Dataset Directory Layout](#3-dataset-directory-layout)
4. [Layer Reference (12 Layers)](#4-layer-reference-12-layers)
5. [Environment & API Keys](#5-environment--api-keys)
6. [Running the Production Ingest](#6-running-the-production-ingest)
7. [Using the Time Machine Oracle](#7-using-the-time-machine-oracle)
8. [Data Quality Reports](#8-data-quality-reports)
9. [Adding New Data — The Salvage Pattern](#9-adding-new-data--the-salvage-pattern)
10. [Re-ingesting a Single Layer](#10-re-ingesting-a-single-layer)
11. [Anti-Leakage Contract](#11-anti-leakage-contract)
12. [Troubleshooting](#12-troubleshooting)
13. [File & Module Reference](#13-file--module-reference)

---

## 1. What Is Phase 5?

Phase 5 is the **environmental data substrate** for ChatTrader.KPai's reinforcement-learning agents. It provides a causally-clean, historically-consistent view of the world at any point in time between 2013 and 2024.

The core concept is a **Time Machine**: given any UTC timestamp `T`, the system returns the _exact_ state of all 12 data layers as they would have been known at `T` — with **zero look-ahead contamination**. No data from `T+1 ms` or later is ever returned.

### Key properties

| Property | Value |
|---|---|
| Coverage window | 2013-01-01 → 2024-12-31 |
| Temporal resolution | Hourly (Parquet) / Event-based (JSONL) |
| Storage format | Apache Parquet (ZSTD L3) + JSONL (UTF-8) |
| Timestamp type | `int64` Unix milliseconds UTC in every row |
| Anti-leakage | Enforced by Iron Curtain Rule in `TimeMachineOracle` |
| Synthetic rows | Tagged `is_synthetic=True`; never silently mixed |

---

## 2. Architecture Overview

```
ChatTrader.KPai/
├── src/phase5/environment/
│   ├── schemas.py                  ← Layer schemas, column types, LAYER_DIRS
│   ├── time_machine_oracle.py      ← THE oracle: query(), replay_window(), describe()
│   ├── ingest_time_machine.py      ← Legacy mock-generator (Black Thursday preset)
│   └── production/
│       ├── run_production_ingest.py  ← Main ingest orchestrator (12 layers)
│       ├── base_client.py            ← AsyncRateLimitedClient + TokenBucketRateLimiter
│       ├── chunked_storage.py        ← Monthly file I/O, resume logic, quality reports
│       ├── synthetic_imputer.py      ← Gap-fill when APIs return nothing
│       └── clients/
│           ├── fear_greed_client.py    ← alternative.me Crypto F&G Index
│           ├── tradfi_client.py        ← Twelve Data / yfinance / FRED
│           ├── derivatives_client.py   ← Binance Data Vision ZIPs
│           ├── onchain_client.py       ← CoinGecko + CoinMetrics community
│           ├── crypto_news_client.py   ← Polygon.io News (3 layers)
│           ├── social_client.py        ← Google Trends via pytrends
│           └── hf_enrichment_client.py ← HuggingFace social + news datasets
│
├── Dataset/phase5_time_machine_dataset/   ← The 12 layer directories
├── run_overnight_enrichment.py            ← Convenience launcher (resume mode)
└── scripts/phase5/
    ├── salvage_derivatives.py   ← One-time: added SOL/BNB to derivatives layer
    └── salvage_hf_backfill.py   ← One-time: backfilled SahandNZ + cvnberk months
```

### Data flow

```
External APIs / HuggingFace / Binance Data Vision
         │
         ▼
  Production Ingest (run_production_ingest.py)
         │
  ┌──────┴──────────────────────────────────────┐
  │ ChunkedLayerStorage (monthly file chunks)  │
  │   Parquet: ZSTD L3, timestamp_utc int64 ms │
  │   JSONL  : UTF-8, one record per line       │
  └──────┬──────────────────────────────────────┘
         │
         ▼
  TimeMachineOracle
    query(timestamp_ms)          → TimeMachineSnapshot (causally filtered)
    replay_window(start, end)    → iterator of TimeMachineSnapshots
    describe(snapshot)           → LLM-ready text summary
    full_integrity_check(ts_ms)  → dict[layer, pass/fail]
```

---

## 3. Dataset Directory Layout

```
Dataset/phase5_time_machine_dataset/
├── derivatives_microstructure/     ← Layer "derivatives"   (65 Parquet files)
│   ├── derivatives_2019_09.parquet
│   ├── derivatives_2019_10.parquet
│   └── ... (monthly through 2024_12)
├── fear_greed/                     ← 145 Parquet files
├── on_chain/                       ← 145 Parquet files
├── tradfi_macro/                   ← 145 Parquet files
├── crypto_news/                    ← 145 JSONL files
├── social_sentiment/               ← 145 JSONL files
├── crises_hacks/                   ← 145 JSONL files
├── regulatory_legal/               ← 145 JSONL files
├── kol_footprints/                 ← JSONL files
├── macro_geopolitical/             ← JSONL files
├── hf_social/                      ← 144 JSONL files
├── hf_news/                        ← 144 JSONL files
├── another_project_raw/            ← set3 raw Binance 5m parquets (source archive)
└── .env                            ← API keys (gitignored)
```

### File naming convention

| Format | Pattern | Example |
|--------|---------|---------|
| Parquet | `{layer_key}_{YYYY}_{MM:02d}.parquet` | `fear_greed_2022_03.parquet` |
| JSONL | `{layer_key}_{YYYY}_{MM:02d}.jsonl` | `hf_news_2022_11.jsonl` |

> **Note:** The derivatives layer uses the subdirectory name `derivatives_microstructure/`  
> but the layer key in code is `"derivatives"`. This mapping is defined in `LAYER_DIRS`  
> in `schemas.py`.

---

## 4. Layer Reference (12 Layers)

### Parquet layers

#### `derivatives` — Market Derivatives Microstructure
- **Directory:** `derivatives_microstructure/`
- **Coverage:** 2019-09 → 2024-12 (64 monthly files)
- **Symbols:** BTCUSDT, ETHUSDT, SOLUSDT (from 2020-08), BNBUSDT (from 2019-09)
- **Source:** Binance Data Vision public ZIPs + set3 raw data backfill
- **Rows/file:** ~1,440 (4 symbols × ~360 hourly bars)

| Column | Type | Notes |
|--------|------|-------|
| `timestamp_utc` | int64 ms | hourly |
| `symbol` | str | BTCUSDT / ETHUSDT / SOLUSDT / BNBUSDT |
| `funding_rate` | float64 | 8-hour Binance perpetual funding rate |
| `oi_usd` | float64 | Open interest USD (0.0 for backfilled rows) |
| `oi_change_usd` | float64 | Delta vs. previous period |
| `long_liquidations_usd` | float64 | |
| `short_liquidations_usd` | float64 | |
| `total_liquidations_usd` | float64 | |
| `liq_long_short_ratio` | float64 | long_liq / (long_liq + short_liq) |
| `options_max_pain_usd` | float64 | NaN when no options data |
| `put_call_ratio` | float64 | |
| `basis_pct` | float64 | (futures − spot) / spot |
| `is_synthetic` | bool | True for all current rows (OI/liq from imputer) |

#### `fear_greed` — Crypto Fear & Greed Index
- **Coverage:** 2018-02 → 2024-12 (daily)
- **Source:** alternative.me (no API key needed)
- **Key columns:** `fear_greed_index` (0–100), `fear_greed_label`, `google_trends_bitcoin`

#### `on_chain` — On-Chain Analytics
- **Coverage:** 2013-01 → 2024-12 (daily, synthetic for pre-2017)
- **Source:** CoinGecko + CoinMetrics Community API
- **Key columns:** `exchange_inflow`, `exchange_outflow`, `whale_transfer_count`, `nvt_ratio`, `sopr`

#### `tradfi_macro` — TradFi Macro Liquidity
- **Coverage:** 2013-01 → 2024-12 (daily)
- **Source:** Twelve Data, yfinance, FRED CSV
- **Key columns:** `fed_funds_rate_bps`, `dxy_close`, `sp500_close`, `vix_close`, `us_10y_yield`

---

### JSONL layers

#### `hf_social` — HuggingFace Social Tweets
- **Coverage:** 144 months, 64 non-zero (44%)
- **Sources:** `mjw/stock_market_tweets` (2015–2022), `cvnberk/bitcoin_tweets_sentiment_kaggle` (2014-09 → 2019-07)
- **Tickers:** AAPL, AMZN, GOOG, GOOGL, MSFT, TSLA, NVDA, META, COIN, MSTR, BTCUSD
- **Schema fields:** `timestamp_utc`, `text`, `ticker`, `author`, `retweet_count`, `like_count`, `comment_count`, `source`

#### `hf_news` — HuggingFace Financial News
- **Coverage:** 144 months, 21 non-zero (15%)
- **Sources:** `ashraq/financial-news-articles` (2018 CNBC), `SahandNZ/cryptonews-articles-with-price-momentum-labels` (2022-10 → 2023-03)
- **Schema fields:** `timestamp_utc`, `title`, `text` (max 1000 chars), `url`, `source`

#### `crypto_news` / `crises_hacks` / `regulatory_legal`
- **Source:** Polygon.io News API — auto-classified by keyword
- **Coverage:** Sparse — months with real API data only (3–5% of months non-zero)
- **Schema fields:** `timestamp_utc`, `headline`, `body`, `source`, `sentiment_score`, `impact_tags`

#### `social_sentiment` — Google Trends
- **Coverage:** 2013-01 → 2024-12 (weekly → daily forward-filled)
- **Source:** Google Trends via pytrends
- **Keywords:** `bitcoin`, `ethereum`, `buy crypto`, `bitcoin crash`, `sell bitcoin`

#### `kol_footprints`, `macro_geopolitical`, `crises_hacks`, `regulatory_legal`
- Sparse JSONL layers; primarily populated via Polygon.io or hand-curated events
- Always valid for resume — 0-byte files are normal and expected for data-free months

---

## 5. Environment & API Keys

All keys live in `Dataset/.env` (gitignored). Create it if it doesn't exist:

```ini
# Dataset/.env

# Twelve Data (tradfi_macro) — https://twelvedata.com
TWELVE_DATA_API_KEY=your_key_here

# Polygon.io (crypto_news, crises_hacks, regulatory_legal)
# Starter plan: free, 5 req/min, unlimited history
POLYGON_API_KEY=your_key_here

# CoinGecko Demo API (on_chain)
# Embed key in URL: https://pro-api.coingecko.com/api/v3?x_cg_demo_api_key=YOUR_KEY
COINGECKO_API_URL=https://pro-api.coingecko.com/api/v3?x_cg_demo_api_key=your_key_here

# Binance (optional — only needed for private endpoints; Data Vision is public)
BINANCE_API_KEY=
BINANCE_API_SECRET=
```

> Keys are **not required** to run the ingest. When missing, each client falls back:  
> TradFi → yfinance · On-chain → CoinMetrics · Derivatives → Binance Data Vision (no key)  
> Fear/Greed → alternative.me (no key) · HF datasets → anonymous (rate-limited but works)

---

## 6. Running the Production Ingest

### Quick start — resume enrichment (recommended)

```powershell
# From the workspace root (activates .venv automatically via PowerShell profile)
.venv\Scripts\python.exe run_overnight_enrichment.py
```

This runs the full ingest in **resume mode** — all months that already have files are skipped. Safe to re-run at any time.

### Full ingest via CLI

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe src/phase5/environment/production/run_production_ingest.py `
    --start 2013-01-01 `
    --end 2024-12-31 `
    --data-root Dataset/phase5_time_machine_dataset `
    --resume
```

### Ingest specific layers only

```powershell
# Re-ingest only hf_news and hf_social from scratch
.venv\Scripts\python.exe src/phase5/environment/production/run_production_ingest.py `
    --start 2013-01-01 --end 2024-12-31 `
    --layers hf_news hf_social `
    --no-resume
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--start` | `2013-01-01` | First month to ingest (ISO date, day ignored) |
| `--end` | `2024-12-31` | Last month to ingest |
| `--data-root` | `Dataset/phase5_time_machine_dataset` | Root of the dataset tree |
| `--layers` | all 8 layers | Space-separated list of layer keys to run |
| `--resume` / `--no-resume` | `--resume` | Skip existing monthly files |
| `--overwrite` | off | Force-overwrite all existing files |

### Layer keys for `--layers`

```
fear_greed  tradfi_macro  on_chain  derivatives
crypto_news  social_sentiment  hf_social  hf_news
```

### Expected output

```
2026-05-26 13:31:42  INFO  time_machine.ingest  === Phase 5 Production Ingest ===
2026-05-26 13:31:42  INFO  time_machine.ingest  Range  : 2013-01-01 -> 2024-12-31  (144 months)
2026-05-26 13:31:42  INFO  time_machine.ingest  Resume : True   Overwrite: False
...
2026-05-26 13:31:50  INFO  time_machine.ingest  === Ingest complete: 0 written  1440 skipped ===
2026-05-26 13:31:52  INFO  time_machine.ingest  === Oracle Anti-Leakage Check ===
2026-05-26 13:31:55  INFO  time_machine.ingest    macro_geopolitical              PASS
2026-05-26 13:31:55  INFO  time_machine.ingest    ... (12 lines)
2026-05-26 13:31:55  INFO  time_machine.ingest  All layers PASS.
```

---

## 7. Using the Time Machine Oracle

### Import and initialise

```python
from src.phase5.environment.time_machine_oracle import TimeMachineOracle

oracle = TimeMachineOracle("Dataset/phase5_time_machine_dataset")
```

### Query at a specific timestamp

```python
from datetime import datetime, timezone

# Convert any UTC datetime to milliseconds
def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000)

# Query the oracle at the moment FTX collapsed (Nov 9 2022, 08:00 UTC)
ts_ms = to_ms(datetime(2022, 11, 9, 8, 0, tzinfo=timezone.utc))

snapshot = oracle.query(
    timestamp_ms = ts_ms,
    window_ms    = 72 * 3_600_000,   # last 72 hours of data
)

print(repr(snapshot))
# TimeMachineSnapshot @ 2022-11-09 08:00:00 UTC
#   derivatives                  2884 records
#   fear_greed                     31 records
#   hf_news                     24752 records
#   ...
```

### Get an LLM-ready text summary

```python
# Inject the Time Machine context into an agent system prompt
context_text = oracle.describe(snapshot, max_events=5)
print(context_text)
```

Output is formatted for direct injection into an LLM prompt:

```
=== TIME MACHINE SNAPSHOT @ 2022-11-09 08:00:00 UTC ===
(All data reflects information available at or before 2022-11-09 08:00:00 UTC.)

── LAYER: DERIVATIVES ──
  Latest record  (2884 rows available):
    timestamp_utc      : 2022-11-09 08:00 UTC
    symbol             : SOLUSDT
    funding_rate       : -0.000375
    ...

── LAYER: HF_NEWS ──
  24752 events up to this timestamp (showing last 5):
    [2022-11-09 06:15 UTC]  FTX halts withdrawals amid liquidity crisis  (score: None)
    ...
```

### Get only the latest record per layer (point-in-time context)

```python
# Returns a flat dict: {layer_key: most_recent_record}
# Ideal for compact LLM context injection
latest = oracle.query_latest(timestamp_ms=ts_ms)

btc_funding = latest["derivatives"]["funding_rate"]     # → float
fng_score   = latest["fear_greed"]["fear_greed_index"]  # → int
```

### Query specific layers

```python
# Only load derivatives + fear_greed (faster, less memory)
snapshot = oracle.query(
    timestamp_ms = ts_ms,
    layers       = ["derivatives", "fear_greed", "hf_news"],
    window_ms    = 24 * 3_600_000,
)
```

### Time-window replay

```python
from datetime import datetime, timezone

start_ms = int(datetime(2020, 3, 11, tzinfo=timezone.utc).timestamp() * 1_000)
end_ms   = int(datetime(2020, 3, 15, tzinfo=timezone.utc).timestamp() * 1_000)

# Replay Black Thursday hour-by-hour
for frame in oracle.replay_window(
    start_ms  = start_ms,
    end_ms    = end_ms,
    step_ms   = 3_600_000,          # 1-hour steps
    layers    = ["derivatives", "fear_greed", "crises_hacks"],
    window_ms = 3_600_000,
):
    funding_rate = frame.layers["derivatives"].data
    print(frame.query_dt.strftime("%Y-%m-%d %H:%M"), "→", len(funding_rate), "rows")
```

### Pre-warm cache for replay loops

```python
# Load all layer files into memory once before a long replay
oracle.warm_cache(layers=["derivatives", "fear_greed"])

# Now replays are fast (no repeated disk I/O)
for frame in oracle.replay_window(start_ms, end_ms, step_ms=3_600_000):
    ...
```

### Named scenario: Black Thursday

```python
# Built-in convenience method: queries at BTC $3,782 bottom (2020-03-12 14:00 UTC)
snapshot = oracle.get_black_thursday_crash_moment()
```

---

## 8. Data Quality Reports

### Print the quality report

```python
from src.phase5.environment.production.chunked_storage import ChunkedLayerStorage

storage = ChunkedLayerStorage("Dataset/phase5_time_machine_dataset")
storage.print_quality_report()
```

**Current output (May 2026):**

```
Layer                         Files  Non-0  %Full       MB
------------------------------------------------------------
crises_hacks                    145      3     2%     0.01
crypto_news                     145      5     3%     0.04
derivatives                      65     65   100%     4.08
fear_greed                      145    145   100%     1.23
hf_news                         144     21    15%   171.25
hf_social                       144     64    44%   468.96
kol_footprints                    1      1   100%     0.00
macro_geopolitical                1      1   100%     0.01
on_chain                        145    145   100%     1.47
regulatory_legal                145      4     3%     0.01
social_sentiment                145    145   100%     1.35
tradfi_macro                    145    145   100%     1.69
------------------------------------------------------------
TOTAL                                                650.08
```

> **Coverage notes:**  
> - `crises_hacks`, `crypto_news`, `regulatory_legal` — sparse by design (Polygon.io free tier)  
> - `hf_news` — 15% coverage; concentrated in 2018 (ashraq) and 2022-10 → 2023-03 (SahandNZ)  
> - `hf_social` — 44% coverage; 2015–2022 (mjw equity tweets) + 2014-09 → 2014-12 (cvnberk BTC)

### Get the report as a dict

```python
report = storage.data_quality_report()
# report["derivatives"]["files_nonempty"]  → 65
# report["hf_news"]["pct_nonempty"]        → 14.6
```

### Oracle integrity check

```python
from src.phase5.environment.time_machine_oracle import TimeMachineOracle
from datetime import datetime, timezone

oracle = TimeMachineOracle("Dataset/phase5_time_machine_dataset")

# Check at end of dataset (2024-12-31 23:59 UTC)
check_ms = int(datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc).timestamp() * 1_000)
results  = oracle.full_integrity_check(check_ms)

for layer, passed in results.items():
    print(f"  {layer:<28}  {'PASS' if passed else 'FAIL'}")
```

### CLI Oracle check

```powershell
# Quick anti-leakage validation at a custom timestamp
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe -m src.phase5.environment.time_machine_oracle `
    --timestamp "2022-11-09T08:00:00Z" `
    --window-hours 72 `
    --validate
```

---

## 9. Adding New Data — The Salvage Pattern

When new source data becomes available for months that already have files, use a **salvage script** to directly write new records without touching the resume mechanism.

### Why not re-run the ingest?

The resume mechanism treats any existing file (even 0-byte) as "done". This is intentional — 0-byte means "processed, no data". The salvage pattern bypasses resume for targeted backfills.

### Template: Parquet salvage (add new symbols to existing monthly files)

```python
"""
Salvage template — add a new symbol to an existing Parquet layer.
See scripts/phase5/salvage_derivatives.py for a complete example.
"""
import pathlib
import pandas as pd

LAYER_DIR = pathlib.Path("Dataset/phase5_time_machine_dataset/derivatives_microstructure")

for parquet_path in sorted(LAYER_DIR.glob("derivatives_*.parquet")):
    df = pd.read_parquet(parquet_path)
    
    if "NEWUSDT" in df["symbol"].unique():
        continue  # already present — safe to skip
    
    # Build new rows for this month
    new_rows = build_new_symbol_rows("NEWUSDT", parquet_path)
    
    # Append and sort
    combined = pd.concat([df, new_rows], ignore_index=True)
    combined = combined.sort_values(["timestamp_utc", "symbol"]).reset_index(drop=True)
    
    # Write back with ZSTD compression
    combined.to_parquet(parquet_path, compression="zstd", compression_level=3, index=False)
    print(f"Updated {parquet_path.name}: +{len(new_rows)} rows")
```

### Template: JSONL backfill (fill 0-byte months with new data)

```python
"""
Salvage template — fill 0-byte JSONL months with real data.
See scripts/phase5/salvage_hf_backfill.py for a complete example.
"""
import json
import pathlib

LAYER_DIR = pathlib.Path("Dataset/phase5_time_machine_dataset/hf_news")

TARGET_MONTHS = [(2022, 10), (2022, 11), (2022, 12), (2023, 1)]

def _should_write(path: pathlib.Path) -> bool:
    """Only write if file doesn't exist or is 0-byte. Never overwrite real data."""
    return not path.exists() or path.stat().st_size == 0

for year, month in TARGET_MONTHS:
    out_path = LAYER_DIR / f"hf_news_{year}_{month:02d}.jsonl"
    
    if not _should_write(out_path):
        print(f"Skip {out_path.name} (already has data)")
        continue
    
    records = fetch_records_for_month(year, month)  # your data source
    
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"Wrote {len(records)} records → {out_path.name}")
```

### Critical rules for salvage scripts

1. **Never overwrite non-zero files** — always check `path.stat().st_size == 0`
2. **Keep `is_synthetic=True`** for any row where source data was unavailable
3. **Timestamp must be `int64` milliseconds UTC** — use the pattern:
   ```python
   # Correct: pandas datetime64[ns, UTC] → int64 ms
   _EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
   ts_ms = ((series - _EPOCH) // pd.Timedelta(milliseconds=1)).astype("int64")
   
   # Wrong: do NOT use astype(np.int64) // 1_000_000 with pandas 2.x
   ```
4. **Run Oracle integrity check** after any salvage to verify no look-ahead was introduced
5. **Store salvage scripts** in `scripts/phase5/` for auditability

---

## 10. Re-ingesting a Single Layer

To completely re-ingest one layer (discarding existing files):

```powershell
# Example: re-ingest fear_greed from scratch
.venv\Scripts\python.exe src/phase5/environment/production/run_production_ingest.py `
    --start 2013-01-01 --end 2024-12-31 `
    --layers fear_greed `
    --no-resume
```

> `--no-resume` + single layer = only the target layer is overwritten.  
> All other layers are untouched (they are not in `--layers`).

### Re-ingest after an API key is added

If you previously ran without an API key and want to replace synthetic data with real data:

```powershell
# Re-ingest tradfi_macro now that Twelve Data key is set
$env:TWELVE_DATA_API_KEY="your_new_key"
.venv\Scripts\python.exe src/phase5/environment/production/run_production_ingest.py `
    --start 2013-01-01 --end 2024-12-31 `
    --layers tradfi_macro `
    --no-resume
```

---

## 11. Anti-Leakage Contract

The **Iron Curtain Rule** is the foundational invariant of Phase 5:

> For every row in every layer:  `row.timestamp_utc  ≤  query_timestamp_ms`

This is enforced in `TimeMachineOracle._query_parquet_layer()` and `._query_jsonl_layer()` with explicit pandas mask / list comprehension filters. **The filter is never relaxed.**

### What counts as a violation

- A row with `timestamp_utc > query_ts_ms` appearing in any query result
- Any derived feature computed from future data (e.g. a 24h forward return)
- Any record with a timestamp that was "edited back" to appear earlier than it actually was

### Checking for violations

```python
# Validate a specific layer
oracle.validate_anti_leakage(query_ts_ms=ts_ms, layer_key="derivatives")
# Raises ValueError with exact offending rows if any violation found

# Validate all 12 layers at once
results = oracle.full_integrity_check(ts_ms)
assert all(results.values()), f"VIOLATIONS: {[k for k, v in results.items() if not v]}"
```

### Synthetic data and leakage

Synthetic rows (tagged `is_synthetic=True`) obey the same Iron Curtain Rule. The synthetic imputer assigns timestamps that are within the target month; they are never assigned future timestamps.

---

## 12. Troubleshooting

### `UnicodeEncodeError: charmap` on Windows

```powershell
# Set UTF-8 encoding before running any Phase 5 script
$env:PYTHONIOENCODING="utf-8"
```

### `HF_HUB_DISABLE_SYMLINKS_WARNING` flooding the terminal

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
```

Or suppress in code (already done in `hf_enrichment_client.py`):
```python
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
```

### HuggingFace dataset download is slow / rate-limited

Add an HF token for higher rate limits:
```powershell
$env:HF_TOKEN="hf_your_token_here"
```

### `month_exists()` returns True but file is 0-byte

This is **correct behaviour**. A 0-byte JSONL file means "we tried this month, the source returned nothing." Do not add a size threshold — that would cause infinite re-processing of valid sparse months.

To backfill a 0-byte month with new data, use the salvage pattern from [Section 9](#9-adding-new-data--the-salvage-pattern).

### All derivatives months are skipped in production ingest

This is expected after the initial ingest. All 64 monthly parquets exist, so resume skips them. If you want to add a new symbol, use the salvage script pattern rather than re-running the full ingest.

### Oracle returns "0 records" for a layer at a specific timestamp

This is not an error. It means no data exists for that layer in the requested time window. Check:

```python
months = storage.list_months_written("hf_news")
print(f"hf_news has data for {len(months)} months: {months[:5]} ... {months[-5:]}")
```

### `LINKUSDT not found` in derivatives data

LINKUSDT is defined in `DERIVATIVES_SYMBOLS` for future population via Binance Data Vision on a full re-ingest. It was not included in the set3 raw data backfill. Current monthly parquets contain only BTC, ETH, SOL, BNB.

### Verify the Oracle still passes after a salvage

```powershell
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe -m src.phase5.environment.time_machine_oracle `
    --validate `
    --timestamp "2024-12-31T23:59:00Z"
```

All 12 lines should say `PASS`.

---

## 13. File & Module Reference

### Core modules

| File | Purpose |
|------|---------|
| `src/phase5/environment/schemas.py` | Layer schemas, `LAYER_DIRS`, `PARQUET_LAYERS`, `DERIVATIVES_COLUMNS`, `DERIVATIVES_SYMBOLS` |
| `src/phase5/environment/time_machine_oracle.py` | `TimeMachineOracle` — query, replay, describe, validate |
| `src/phase5/environment/production/run_production_ingest.py` | Main async orchestrator; `IngestConfig`; `run_ingest()` |
| `src/phase5/environment/production/chunked_storage.py` | `ChunkedLayerStorage` — read/write monthly files, quality reports |
| `src/phase5/environment/production/synthetic_imputer.py` | `SyntheticImputer` — gap-fill missing months |
| `src/phase5/environment/production/base_client.py` | `AsyncRateLimitedClient` — token-bucket HTTP client |
| `src/phase5/environment/ingest_time_machine.py` | Legacy mock-generator; Black Thursday preset |

### Clients

| File | Layer(s) | Source |
|------|----------|--------|
| `clients/fear_greed_client.py` | `fear_greed` | alternative.me API |
| `clients/tradfi_client.py` | `tradfi_macro` | Twelve Data / yfinance / FRED |
| `clients/derivatives_client.py` | `derivatives` | Binance Data Vision (public ZIPs) |
| `clients/onchain_client.py` | `on_chain` | CoinGecko + CoinMetrics |
| `clients/crypto_news_client.py` | `crypto_news`, `crises_hacks`, `regulatory_legal` | Polygon.io |
| `clients/social_client.py` | `social_sentiment` | Google Trends (pytrends) |
| `clients/hf_enrichment_client.py` | `hf_social`, `hf_news` | HuggingFace Datasets |

### Launchers & utilities

| File | Purpose |
|------|---------|
| `run_overnight_enrichment.py` | One-command resume enrichment; prints quality report |
| `scripts/phase5/salvage_derivatives.py` | One-time: added SOLUSDT + BNBUSDT to all 64 monthly parquets |
| `scripts/phase5/salvage_hf_backfill.py` | One-time: backfilled SahandNZ news (2022-10 → 2023-03) + cvnberk tweets (2014-09 → 2014-12) |

### Key constants

```python
from src.phase5.environment.schemas import (
    LAYER_DIRS,           # dict: layer_key → subdirectory name
    PARQUET_LAYERS,       # frozenset of Parquet layer keys
    JSONL_LAYERS,         # frozenset of JSONL layer keys
    ALL_LAYER_KEYS,       # tuple of all 12 keys in canonical order
    DERIVATIVES_SYMBOLS,  # ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT")
    DERIVATIVES_COLUMNS,  # dict: column_name → dtype string (13 columns)
)
```

---

*Manual generated May 2026. For questions, see `doc/master_plan.md` or the session summaries in the transcript.*
