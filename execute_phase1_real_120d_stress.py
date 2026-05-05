#!/usr/bin/env python3
"""
Phase 1 (strict real data): 120-day acquisition + stress-event samples.

What this does:
- Downloads real Binance Vision daily files for:
  - aggTrades, bookTicker (spot; REQUIRED)
  - fundingRate, metrics (futures UM; OPTIONAL by symbol availability)
- Validates every downloaded zip file against Binance .CHECKSUM (SHA256).
- Builds a 120-day window for the requested symbol set.
- Detects top-N stress dates from BTCUSDT 1d klines over the last 12 months.
- Downloads those stress dates for all target symbols/data types.
- Writes parquet partitions and emits a strict integrity summary.

No synthetic data is generated.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
import zipfile
from xml.etree import ElementTree as ET

import aiohttp
import pandas as pd

BASE_URL = "https://data.binance.vision"
LIST_BASE_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

REQUIRED_SYMBOLS_DEFAULT = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BTCETH", "HYPEUSDT", "BNBUSDT"]
REQUIRED_DATA_TYPES = ["aggTrades", "bookTicker"]
OPTIONAL_DATA_TYPES = ["fundingRate", "metrics"]
ALL_DATA_TYPES = REQUIRED_DATA_TYPES + OPTIONAL_DATA_TYPES

DATA_TYPE_CONFIG = {
    "aggTrades": {
        "market": "spot",
        "path": "data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip",
        "columns": [
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
            "is_buyer_maker",
            "is_best_match",
        ],
    },
    "bookTicker": {
        "market": "spot",
        "path": "data/spot/daily/bookTicker/{symbol}/{symbol}-bookTicker-{day}.zip",
        "columns": [
            "update_id",
            "best_bid_price",
            "best_bid_qty",
            "best_ask_price",
            "best_ask_qty",
            "transaction_time",
            "event_time",
        ],
    },
    "fundingRate": {
        "market": "futures",
        "path": "data/futures/um/daily/fundingRate/{symbol}/{symbol}-fundingRate-{day}.zip",
        "columns": ["symbol", "funding_time", "funding_rate", "mark_price"],
    },
    "metrics": {
        "market": "futures",
        "path": "data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip",
        "columns": [
            "symbol",
            "sum_open_interest",
            "sum_open_interest_value",
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
            "timestamp",
        ],
    },
}

KLINES_1D_PREFIX = "data/spot/daily/klines/BTCUSDT/1d/"
KLINES_1D_PATTERN = re.compile(r"BTCUSDT-1d-(\d{4}-\d{2}-\d{2})\.zip$")


@dataclass
class FileResult:
    key: str
    symbol: str
    data_type: str
    day: date
    status: str  # downloaded | skipped | missing | failed
    rows: int = 0
    bytes_downloaded: int = 0
    checksum_ok: bool = False
    error: str = ""


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def iter_days(start: date, end: date) -> List[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def build_key(symbol: str, data_type: str, day: date) -> str:
    return DATA_TYPE_CONFIG[data_type]["path"].format(symbol=symbol, day=day.strftime("%Y-%m-%d"))


def parquet_path(output_root: Path, symbol: str, data_type: str, day: date) -> Path:
    out_dir = output_root / symbol / data_type / day.strftime("%Y-%m")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{day.strftime('%Y%m%d')}.parquet"


def extract_checksum(checksum_text: str) -> str:
    first_line = checksum_text.strip().splitlines()[0]
    return first_line.split()[0].strip()


def verify_sha256(content: bytes, expected_hex: str) -> bool:
    got = hashlib.sha256(content).hexdigest().lower()
    return got == expected_hex.lower()


def csv_rows_from_zip(zip_content: bytes) -> List[List[str]]:
    with zipfile.ZipFile(BytesIO(zip_content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError("zip contains no csv")
        with zf.open(csv_names[0], "r") as f:
            wrapper = TextIOWrapper(f, encoding="utf-8")
            return list(csv.reader(wrapper))


def rows_to_df(rows: List[List[str]], columns_hint: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns_hint)

    width = len(rows[0])
    cols = columns_hint[:width] if width <= len(columns_hint) else [f"col_{i}" for i in range(width)]
    df = pd.DataFrame(rows, columns=cols)

    for c in df.columns:
        if c in {"symbol"}:
            continue
        if c.startswith("is_"):
            df[c] = df[c].astype(str).str.lower().isin(["true", "1"])
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        non_na_ratio = float(converted.notna().mean()) if len(converted) else 0.0
        if non_na_ratio > 0.8:
            df[c] = converted

    return df


async def list_keys(
    session: aiohttp.ClientSession,
    prefix: str,
    start_after: Optional[str] = None,
    max_pages: int = 50,
) -> List[str]:
    keys: List[str] = []
    token: Optional[str] = None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    for _ in range(max_pages):
        params = [
            "list-type=2",
            f"prefix={quote(prefix, safe='/')}",
            "max-keys=1000",
        ]
        if start_after:
            params.append(f"start-after={quote(start_after, safe='/')}")
        if token:
            params.append(f"continuation-token={quote(token, safe='')}")

        url = f"{LIST_BASE_URL}?{'&'.join(params)}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=90)) as resp:
            resp.raise_for_status()
            text = await resp.text()

        root = ET.fromstring(text)
        page_keys = [
            node.text
            for node in root.findall("s3:Contents/s3:Key", ns)
            if node.text
        ]
        keys.extend(page_keys)

        is_truncated = root.find("s3:IsTruncated", ns)
        next_token = root.find("s3:NextContinuationToken", ns)
        if is_truncated is None or is_truncated.text != "true" or next_token is None or not next_token.text:
            break
        token = next_token.text

    return keys


async def fetch_text(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        if resp.status == 404:
            raise FileNotFoundError(url)
        resp.raise_for_status()
        return await resp.text()


async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> bytes:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=180)) as resp:
        if resp.status == 404:
            raise FileNotFoundError(url)
        resp.raise_for_status()
        return await resp.read()


async def fetch_text_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    attempts: int = 3,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fetch_text(session, url)
        except FileNotFoundError:
            raise
        except Exception as ex:
            last_error = ex
            if attempt < attempts:
                await asyncio.sleep(0.4 * attempt)
    assert last_error is not None
    raise last_error


async def fetch_bytes_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    attempts: int = 3,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fetch_bytes(session, url)
        except FileNotFoundError:
            raise
        except Exception as ex:
            last_error = ex
            if attempt < attempts:
                await asyncio.sleep(0.4 * attempt)
    assert last_error is not None
    raise last_error


async def download_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    output_root: Path,
    symbol: str,
    data_type: str,
    day: date,
) -> FileResult:
    key = build_key(symbol, data_type, day)
    out_file = parquet_path(output_root, symbol, data_type, day)

    if out_file.exists():
        try:
            df = pd.read_parquet(out_file)
            return FileResult(key, symbol, data_type, day, "skipped", rows=len(df), checksum_ok=True)
        except Exception:
            # Auto-repair local corruption by deleting and re-downloading.
            try:
                out_file.unlink(missing_ok=True)
            except Exception:
                pass

    url = f"{BASE_URL}/{key}"
    checksum_url = f"{url}.CHECKSUM"

    async with semaphore:
        try:
            checksum_text = await fetch_text_with_retry(session, checksum_url)
            expected = extract_checksum(checksum_text)
            content = await fetch_bytes_with_retry(session, url)
            if not verify_sha256(content, expected):
                return FileResult(key, symbol, data_type, day, "failed", error="checksum mismatch")

            rows = csv_rows_from_zip(content)
            df = rows_to_df(rows, DATA_TYPE_CONFIG[data_type]["columns"])
            df.to_parquet(out_file, index=False)
            return FileResult(
                key,
                symbol,
                data_type,
                day,
                "downloaded",
                rows=len(df),
                bytes_downloaded=len(content),
                checksum_ok=True,
            )
        except FileNotFoundError:
            return FileResult(key, symbol, data_type, day, "missing")
        except Exception as ex:
            return FileResult(key, symbol, data_type, day, "failed", error=str(ex))


def kline_df_from_zip(zip_content: bytes) -> pd.DataFrame:
    rows = csv_rows_from_zip(zip_content)
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols[: len(rows[0])])
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "open_time" in df.columns:
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    return df


async def identify_stress_dates(
    session: aiohttp.ClientSession,
    count: int = 6,
    concurrency: int = 20,
) -> List[date]:
    today = utc_today()
    start_12m = today - timedelta(days=365)
    start_after = f"{KLINES_1D_PREFIX}BTCUSDT-1d-{(start_12m - timedelta(days=1)).strftime('%Y-%m-%d')}.zip"

    keys = await list_keys(session, KLINES_1D_PREFIX, start_after=start_after, max_pages=10)
    keys = [k for k in keys if k.endswith(".zip") and KLINES_1D_PATTERN.search(k)]

    scored: List[Tuple[date, float, float, float]] = []
    sem = asyncio.Semaphore(concurrency)

    async def score_one(key: str) -> Optional[Tuple[date, float, float, float]]:
        m = KLINES_1D_PATTERN.search(key)
        if not m:
            return None
        day = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if day < start_12m or day > today:
            return None

        url = f"{BASE_URL}/{key}"
        checksum_url = f"{url}.CHECKSUM"
        async with sem:
            try:
                checksum_text = await fetch_text(session, checksum_url)
                expected = extract_checksum(checksum_text)
                content = await fetch_bytes(session, url)
                if not verify_sha256(content, expected):
                    return None
                kdf = kline_df_from_zip(content)
                if kdf.empty:
                    return None
                row = kdf.iloc[0]
                o = float(row.get("open", 0.0))
                h = float(row.get("high", 0.0))
                l = float(row.get("low", 0.0))
                c = float(row.get("close", 0.0))
                if o <= 0:
                    return None
                abs_ret = abs(c - o) / o
                intraday_range = abs(h - l) / o
                score = abs_ret + intraday_range
                return (day, score, abs_ret, intraday_range)
            except Exception:
                return None

    candidate_keys = sorted(keys)
    print(f"stress_scan_candidates={len(candidate_keys)}")
    batch_size = 100
    for i in range(0, len(candidate_keys), batch_size):
        batch = candidate_keys[i : i + batch_size]
        results = await asyncio.gather(*[score_one(k) for k in batch])
        scored.extend([r for r in results if r is not None])
        print(f"stress_scan_progress={min(i + batch_size, len(candidate_keys))}/{len(candidate_keys)}")

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:count]
    return [x[0] for x in top]


def summarize_results(results: List[FileResult]) -> Dict[str, Dict[str, Dict[str, int]]]:
    summary: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in results:
        s = summary[r.symbol][r.data_type]
        s[r.status] += 1
        s["rows"] += int(r.rows)
        s["bytes"] += int(r.bytes_downloaded)
    return summary


def ensure_required_data(results: List[FileResult], required_symbols: List[str], required_days: List[date]) -> None:
    required_set = {(sym, dt, d) for sym in required_symbols for dt in REQUIRED_DATA_TYPES for d in required_days}
    ok_set = {
        (r.symbol, r.data_type, r.day)
        for r in results
        if r.status in {"downloaded", "skipped"} and r.data_type in REQUIRED_DATA_TYPES
    }
    missing = sorted(required_set - ok_set)
    if missing:
        sample = "\n".join([f"  - {m[0]} {m[1]} {m[2]}" for m in missing[:20]])
        raise RuntimeError(
            "Required spot data incomplete; halting before Phase 2. Missing examples:\n" + sample
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="Dataset/binance_vision_real", help="Output root")
    p.add_argument("--days", type=int, default=120, help="Rolling day window")
    p.add_argument("--stress-count", type=int, default=6, help="Number of stress dates")
    p.add_argument(
        "--symbols",
        default=",".join(REQUIRED_SYMBOLS_DEFAULT),
        help="Comma-separated symbols",
    )
    p.add_argument(
        "--data-types",
        default=",".join(ALL_DATA_TYPES),
        help="Comma-separated data types to acquire",
    )
    p.add_argument("--concurrency", type=int, default=20, help="Concurrent downloads")
    return p.parse_args()


async def run_pipeline(args: argparse.Namespace) -> int:
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    selected_data_types = [d.strip() for d in args.data_types.split(",") if d.strip()]
    invalid = [d for d in selected_data_types if d not in ALL_DATA_TYPES]
    if invalid:
        raise RuntimeError(f"invalid data types: {invalid}")
    end_day = utc_today() - timedelta(days=1)
    start_day = end_day - timedelta(days=args.days - 1)
    window_days = iter_days(start_day, end_day)

    print("STRICT REAL DATA ACQUISITION")
    print(f"symbols={symbols}")
    print(f"window={start_day}..{end_day} ({len(window_days)} days)")
    print(f"data_types={selected_data_types}")
    print(f"output={output_root}")

    connector = aiohttp.TCPConnector(limit=args.concurrency, limit_per_host=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        stress_dates = await identify_stress_dates(session, count=args.stress_count, concurrency=args.concurrency)
        print(f"stress_dates={stress_dates}")

        all_days = sorted(set(window_days + stress_dates))
        print(f"total_unique_days={len(all_days)}")

        tasks = []
        for symbol in symbols:
            for data_type in selected_data_types:
                for day in all_days:
                    tasks.append(download_one(session, semaphore, output_root, symbol, data_type, day))

        results: List[FileResult] = []
        batch_size = 50
        for i in range(0, len(tasks), batch_size):
            chunk = tasks[i : i + batch_size]
            chunk_results = await asyncio.gather(*chunk)
            results.extend(chunk_results)
            downloaded = sum(1 for r in chunk_results if r.status == "downloaded")
            skipped = sum(1 for r in chunk_results if r.status == "skipped")
            missing = sum(1 for r in chunk_results if r.status == "missing")
            failed = sum(1 for r in chunk_results if r.status == "failed")
            print(
                f"progress={len(results)}/{len(tasks)} "
                f"chunk(downloaded={downloaded}, skipped={skipped}, missing={missing}, failed={failed})"
            )

    if all(d in selected_data_types for d in REQUIRED_DATA_TYPES):
        ensure_required_data(results, symbols, all_days)

    summary = summarize_results(results)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "data_types": selected_data_types,
        "window_start": start_day.isoformat(),
        "window_end": end_day.isoformat(),
        "window_days": [d.isoformat() for d in window_days],
        "stress_dates": [d.isoformat() for d in stress_dates],
        "all_days": [d.isoformat() for d in all_days],
        "summary": summary,
    }

    manifest_file = output_root / "REAL_120D_STRESS_MANIFEST.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("DOWNLOAD SUMMARY")
    total_rows = 0
    total_bytes = 0
    total_downloaded = 0
    total_skipped = 0
    total_missing = 0
    total_failed = 0

    for symbol in symbols:
        print(f"\n{symbol}:")
        for dt in selected_data_types:
            stat = summary.get(symbol, {}).get(dt, {})
            downloaded = int(stat.get("downloaded", 0))
            skipped = int(stat.get("skipped", 0))
            missing = int(stat.get("missing", 0))
            failed = int(stat.get("failed", 0))
            rows = int(stat.get("rows", 0))
            bts = int(stat.get("bytes", 0))
            print(
                f"  {dt:11s} downloaded={downloaded:4d} skipped={skipped:4d} "
                f"missing={missing:4d} failed={failed:4d} rows={rows:12d} bytes={bts:12d}"
            )
            total_rows += rows
            total_bytes += bts
            total_downloaded += downloaded
            total_skipped += skipped
            total_missing += missing
            total_failed += failed

    print("\nTOTAL")
    print(f"downloaded={total_downloaded} skipped={total_skipped} missing={total_missing} failed={total_failed}")
    print(f"rows={total_rows} bytes={total_bytes}")
    print(f"manifest={manifest_file}")

    if total_failed > 0:
        raise RuntimeError("One or more files failed to download/parse.")

    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    raise SystemExit(main())
