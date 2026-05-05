#!/usr/bin/env python3
"""
Strict real-data bootstrap for Phase 1 from data.binance.vision.

Purpose:
- Download real daily aggTrades zip files for one symbol/month.
- Validate each file with Binance-provided CHECKSUM.
- Save partitioned parquet files.
- Load month parquet into one DataFrame and report shape and memory.

This script intentionally does not generate synthetic data.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import List
from urllib.parse import quote
import zipfile
from xml.etree import ElementTree as ET

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://data.binance.vision"
LIST_BASE_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

AGGTRADES_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
]


@dataclass
class DownloadResult:
    key: str
    rows: int
    bytes_downloaded: int
    checksum_ok: bool


def month_bounds(month_yyyy_mm: str) -> tuple[date, date]:
    dt = datetime.strptime(month_yyyy_mm, "%Y-%m")
    start = date(dt.year, dt.month, 1)
    if dt.month == 12:
        end = date(dt.year + 1, 1, 1)
    else:
        end = date(dt.year, dt.month + 1, 1)
    return start, end


def parse_yyyymmdd_from_key(key: str) -> date | None:
    key_lower = key.lower()
    if not key_lower.endswith(".zip"):
        return None

    marker = "aggtrades-"
    idx = key_lower.rfind(marker)
    if idx < 0:
        return None

    date_str = key[idx + len(marker): idx + len(marker) + 10]
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


async def fetch_prefix_keys(
    session: aiohttp.ClientSession,
    prefix: str,
    start_after: str | None = None,
    max_pages: int = 20,
) -> List[str]:
    """Fetch object keys using S3 ListObjectsV2 XML API."""
    keys: List[str] = []
    continuation_token: str | None = None
    page = 0

    while True:
        page += 1
        params = [
            "list-type=2",
            f"prefix={quote(prefix, safe='/')}",
            "max-keys=1000",
        ]
        if start_after:
            params.append(f"start-after={quote(start_after, safe='/')}")
        if continuation_token:
            params.append(f"continuation-token={quote(continuation_token, safe='')}" )

        url = f"{LIST_BASE_URL}?{'&'.join(params)}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
            text = await resp.text()

        root = ET.fromstring(text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        page_keys = [
            node.text
            for node in root.findall("s3:Contents/s3:Key", ns)
            if node.text
        ]
        keys.extend(page_keys)

        is_truncated = root.find("s3:IsTruncated", ns)
        next_token = root.find("s3:NextContinuationToken", ns)

        if (
            is_truncated is None
            or is_truncated.text != "true"
            or next_token is None
            or not next_token.text
            or page >= max_pages
        ):
            break

        continuation_token = next_token.text

    return [k for k in keys if k.endswith(".zip")]


async def fetch_text(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        resp.raise_for_status()
        return await resp.text()


async def fetch_bytes(session: aiohttp.ClientSession, url: str) -> bytes:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        resp.raise_for_status()
        return await resp.read()


def extract_checksum(checksum_text: str) -> str:
    first_line = checksum_text.strip().splitlines()[0]
    first_token = first_line.split()[0]
    return first_token.strip()


def verify_sha256(content: bytes, expected_hex: str) -> bool:
    got = hashlib.sha256(content).hexdigest().lower()
    return got == expected_hex.lower()


def zip_csv_to_df(zip_content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(BytesIO(zip_content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError("zip has no csv")
        with zf.open(csv_names[0], "r") as f:
            wrapper = TextIOWrapper(f, encoding="utf-8")
            reader = csv.reader(wrapper)
            rows = list(reader)

    if not rows:
        return pd.DataFrame(columns=AGGTRADES_COLUMNS)

    width = len(rows[0])
    cols = AGGTRADES_COLUMNS[:width]
    df = pd.DataFrame(rows, columns=cols)

    numeric_cols = [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bool_cols = ["is_buyer_maker", "is_best_match"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1"])

    return df


def parquet_path(output_root: Path, symbol: str, day: date) -> Path:
    out_dir = output_root / symbol / "aggTrades" / day.strftime("%Y-%m")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{day.strftime('%Y%m%d')}.parquet"


async def download_month_real_aggtrades(symbol: str, month: str, output_root: Path) -> List[DownloadResult]:
    start, end = month_bounds(month)
    prefix = f"data/spot/daily/aggTrades/{symbol}/"
    start_after_key = (
        f"{prefix}{symbol}-aggTrades-{(start.fromordinal(start.toordinal() - 1)).strftime('%Y-%m-%d')}.zip"
    )

    connector = aiohttp.TCPConnector(limit=8, limit_per_host=8)
    async with aiohttp.ClientSession(connector=connector) as session:
        keys = await fetch_prefix_keys(session, prefix, start_after=start_after_key)
        target_keys = []
        for k in keys:
            d = parse_yyyymmdd_from_key(k)
            if d is not None and start <= d < end:
                target_keys.append(k)

        target_keys = sorted(target_keys)
        if not target_keys:
            raise RuntimeError(f"no aggTrades files found for {symbol} {month}")

        results: List[DownloadResult] = []

        for key in target_keys:
            day = parse_yyyymmdd_from_key(key)
            if day is None:
                continue

            out_file = parquet_path(output_root, symbol, day)
            if out_file.exists():
                existing_df = pd.read_parquet(out_file)
                results.append(
                    DownloadResult(
                        key=key,
                        rows=len(existing_df),
                        bytes_downloaded=0,
                        checksum_ok=True,
                    )
                )
                print(f"skip existing {out_file} rows={len(existing_df)}")
                continue

            file_url = f"{BASE_URL}/{key}"
            checksum_url = f"{file_url}.CHECKSUM"

            checksum_text = await fetch_text(session, checksum_url)
            expected = extract_checksum(checksum_text)

            content = await fetch_bytes(session, file_url)
            ok = verify_sha256(content, expected)
            if not ok:
                raise RuntimeError(f"checksum mismatch for {key}")

            df = zip_csv_to_df(content)
            df.to_parquet(out_file, index=False)

            results.append(
                DownloadResult(
                    key=key,
                    rows=len(df),
                    bytes_downloaded=len(content),
                    checksum_ok=ok,
                )
            )
            print(f"saved {out_file} rows={len(df)} bytes={len(content)}")

        return results


def load_month_dataframe(symbol: str, month: str, output_root: Path) -> pd.DataFrame:
    month_dir = output_root / symbol / "aggTrades" / month
    files = sorted(month_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no parquet files in {month_dir}")

    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def dataframe_memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True, help="example: BTCUSDT")
    p.add_argument("--month", required=True, help="YYYY-MM")
    p.add_argument("--output", default="Dataset/bn_vision_data", help="output root")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    symbol = args.symbol.upper()
    month = args.month
    output_root = Path(args.output)

    print("REAL PHASE 1 BOOTSTRAP")
    print(f"symbol={symbol} month={month} output={output_root}")

    results = asyncio.run(download_month_real_aggtrades(symbol, month, output_root))

    total_rows = sum(r.rows for r in results)
    total_bytes = sum(r.bytes_downloaded for r in results)
    all_checksums_ok = all(r.checksum_ok for r in results)

    print("DOWNLOAD SUMMARY")
    print(f"files_downloaded={len(results)}")
    print(f"rows_downloaded={total_rows}")
    print(f"bytes_downloaded={total_bytes}")
    print(f"checksums_ok={all_checksums_ok}")

    df = load_month_dataframe(symbol, month, output_root)
    mem_mb = dataframe_memory_mb(df)

    print("DATAFRAME SUMMARY")
    print(f"shape={df.shape}")
    print(f"memory_mb={mem_mb:.2f}")
    print(f"columns={list(df.columns)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
