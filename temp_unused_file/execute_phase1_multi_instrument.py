#!/usr/bin/env python3
"""
Phase 1 Expanded (strict real data): spot/futures/options multi-instrument ingestion.

What this does:
- Keeps spot data under Dataset/spot (no synthetic generation).
- Downloads futures daily files for UM and CM into Dataset/futures/.
- Downloads options BVOL index files into Dataset/options/.
- Enforces SHA256 CHECKSUM validation for every downloaded zip.
- Uses a strict 365-day window and emits machine-readable integrity summary.
- Writes a synchronized daily timestamp index for downstream joins.
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
from typing import Optional
from urllib.parse import quote
import zipfile
from xml.etree import ElementTree as ET

import aiohttp
import pandas as pd

BASE_URL = "https://data.binance.vision"
LIST_BASE_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

FUTURES_DATA_TYPES = ["aggTrades", "fundingRate", "metrics"]
TOP10_UM_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "AVAXUSDT",
]
TOP10_CM_SYMBOLS = [
    "BTCUSD_PERP",
    "ETHUSD_PERP",
    "BNBUSD_PERP",
    "SOLUSD_PERP",
    "XRPUSD_PERP",
    "DOGEUSD_PERP",
    "ADAUSD_PERP",
    "TRXUSD_PERP",
    "LINKUSD_PERP",
    "AVAXUSD_PERP",
]
OPTIONS_BVOL_SYMBOLS = ["BTCBVOLUSDT", "ETHBVOLUSDT"]

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass
class FileResult:
    key: str
    instrument: str
    data_type: str
    day: date
    status: str
    rows: int = 0
    bytes_downloaded: int = 0
    checksum_ok: bool = False
    error: str = ""


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def iter_days(start: date, end: date) -> list[date]:
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def extract_checksum(checksum_text: str) -> str:
    first_line = checksum_text.strip().splitlines()[0]
    return first_line.split()[0].strip()


def verify_sha256(content: bytes, expected_hex: str) -> bool:
    got = hashlib.sha256(content).hexdigest().lower()
    return got == expected_hex.lower()


def csv_rows_from_zip(zip_content: bytes) -> list[list[str]]:
    with zipfile.ZipFile(BytesIO(zip_content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError("zip contains no csv")
        with zf.open(csv_names[0], "r") as f:
            wrapper = TextIOWrapper(f, encoding="utf-8")
            return list(csv.reader(wrapper))


def rows_to_df(rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    # Some files include header, others do not.
    first = rows[0]
    has_header = any(not re.match(r"^-?\d+(\.\d+)?$", cell.strip()) for cell in first)
    if has_header:
        header = [h.strip() or f"col_{i}" for i, h in enumerate(first)]
        body = rows[1:]
        width = len(header)
        body = [r + [""] * (width - len(r)) if len(r) < width else r[:width] for r in body]
        df = pd.DataFrame(body, columns=header)
    else:
        width = max(len(r) for r in rows)
        cols = [f"col_{i}" for i in range(width)]
        body = [r + [""] * (width - len(r)) if len(r) < width else r[:width] for r in rows]
        df = pd.DataFrame(body, columns=cols)

    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce")
        if float(converted.notna().mean()) > 0.8:
            df[c] = converted
    return df


def extract_day_from_key(key: str) -> Optional[date]:
    match = DATE_PATTERN.search(key)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def output_path(root: Path, dataset: str, market: str, symbol: str, data_type: str, day: date) -> Path:
    if dataset == "futures":
        out_dir = root / "futures" / market / symbol / data_type / day.strftime("%Y-%m")
    elif dataset == "options":
        out_dir = root / "options" / "BVOLIndex" / symbol / day.strftime("%Y-%m")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{day.strftime('%Y%m%d')}.parquet"


async def list_keys(
    session: aiohttp.ClientSession,
    prefix: str,
    max_pages: int = 60,
) -> list[str]:
    keys: list[str] = []
    token: Optional[str] = None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    for _ in range(max_pages):
        params = [
            "list-type=2",
            f"prefix={quote(prefix, safe='/')}",
            "max-keys=1000",
        ]
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
            if node.text and (node.text.endswith(".zip") or node.text.endswith(".csv"))
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


async def fetch_text_with_retry(session: aiohttp.ClientSession, url: str, attempts: int = 3) -> str:
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


async def fetch_bytes_with_retry(session: aiohttp.ClientSession, url: str, attempts: int = 3) -> bytes:
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
    root: Path,
    dataset: str,
    market: str,
    symbol: str,
    data_type: str,
    day: date,
    key: str,
) -> FileResult:
    out_file = output_path(root, dataset, market, symbol, data_type, day)
    if out_file.exists():
        return FileResult(key, symbol, data_type, day, "skipped", rows=0, checksum_ok=True)

    url = f"{BASE_URL}/{key}"
    checksum_url = f"{url}.CHECKSUM"

    async with semaphore:
        try:
            checksum_text = await fetch_text_with_retry(session, checksum_url)
            expected = extract_checksum(checksum_text)
            content = await fetch_bytes_with_retry(session, url)
            if not verify_sha256(content, expected):
                return FileResult(key, symbol, data_type, day, "failed", error="checksum mismatch")

            if key.endswith(".zip"):
                rows = csv_rows_from_zip(content)
            else:
                text = content.decode("utf-8", errors="replace")
                rows = list(csv.reader(text.splitlines()))
            df = rows_to_df(rows)
            df.to_parquet(out_file, index=False)
            return FileResult(
                key=key,
                instrument=symbol,
                data_type=data_type,
                day=day,
                status="downloaded",
                rows=len(df),
                bytes_downloaded=len(content),
                checksum_ok=True,
            )
        except FileNotFoundError:
            return FileResult(key, symbol, data_type, day, "missing", error="404")
        except Exception as ex:
            return FileResult(key, symbol, data_type, day, "failed", error=f"{type(ex).__name__}: {ex}")


def _prefix_for_futures(market: str, data_type: str, symbol: str) -> str:
    return f"data/futures/{market}/daily/{data_type}/{symbol}/"


def _prefix_for_options(symbol: str) -> str:
    return f"data/option/daily/BVOLIndex/{symbol}/"


async def run_ingestion(root: Path, days: int, concurrency: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    (root / "spot").mkdir(parents=True, exist_ok=True)
    (root / "futures").mkdir(parents=True, exist_ok=True)
    (root / "options").mkdir(parents=True, exist_ok=True)
    (root / "multi_instrument").mkdir(parents=True, exist_ok=True)

    today = utc_today()
    start_day = today - timedelta(days=days - 1)
    required_days = set(iter_days(start_day, today))

    summary: dict = {
        "phase": "PHASE 1 - MULTI INSTRUMENT UPGRADE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "start_day": start_day.isoformat(),
        "end_day": today.isoformat(),
        "futures": {"um": {}, "cm": {}},
        "options": {},
        "counts": {"downloaded": 0, "skipped": 0, "missing": 0, "failed": 0},
    }

    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=max(32, concurrency * 2), ssl=False)

    per_group_available_days: dict[str, set[date]] = defaultdict(set)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks: list[asyncio.Task[FileResult]] = []
        list_semaphore = asyncio.Semaphore(8)

        listing_jobs: list[tuple[str, str, str, str, str]] = []
        for market, symbols in (("um", TOP10_UM_SYMBOLS), ("cm", TOP10_CM_SYMBOLS)):
            for data_type in FUTURES_DATA_TYPES:
                for symbol in symbols:
                    prefix = _prefix_for_futures(market, data_type, symbol)
                    listing_jobs.append(("futures", market, symbol, data_type, prefix))
        for symbol in OPTIONS_BVOL_SYMBOLS:
            listing_jobs.append(("options", "bvol", symbol, "BVOLIndex", _prefix_for_options(symbol)))

        async def _list_one(job: tuple[str, str, str, str, str]) -> tuple[tuple[str, str, str, str, str], list[str]]:
            dataset, market, symbol, data_type, prefix = job
            async with list_semaphore:
                keys = await list_keys(session, prefix)
            print(
                f"[multi-phase1] listed dataset={dataset} market={market} symbol={symbol} data_type={data_type} keys={len(keys)}",
                flush=True,
            )
            return job, keys

        listing_results = await asyncio.gather(*[_list_one(job) for job in listing_jobs])

        for (dataset, market, symbol, data_type, _prefix), keys in listing_results:
            by_day: dict[date, str] = {}
            for key in keys:
                d = extract_day_from_key(key)
                if d is None:
                    continue
                by_day[d] = key

            missing_days = []
            for d in sorted(required_days):
                key = by_day.get(d)
                if key is None:
                    missing_days.append(d.isoformat())
                    continue

                if dataset == "futures":
                    tasks.append(
                        asyncio.create_task(
                            download_one(
                                session,
                                semaphore,
                                root,
                                "futures",
                                market,
                                symbol,
                                data_type,
                                d,
                                key,
                            )
                        )
                    )
                else:
                    tasks.append(
                        asyncio.create_task(
                            download_one(
                                session,
                                semaphore,
                                root,
                                "options",
                                "bvol",
                                symbol,
                                "BVOLIndex",
                                d,
                                key,
                            )
                        )
                    )

            if dataset == "futures":
                target = summary["futures"][market].setdefault(symbol, {})
                target[data_type] = {
                    "required_days": len(required_days),
                    "available_source_days": len(by_day),
                    "missing_source_days": len(missing_days),
                    "missing_source_dates": missing_days[:40],
                }
            else:
                summary["options"][symbol] = {
                    "required_days": len(required_days),
                    "available_source_days": len(by_day),
                    "missing_source_days": len(missing_days),
                    "missing_source_dates": missing_days[:40],
                }

        print(f"[multi-phase1] queued_download_tasks={len(tasks)}", flush=True)

        results: list[FileResult] = []
        for i, task in enumerate(asyncio.as_completed(tasks), start=1):
            result = await task
            results.append(result)
            summary["counts"][result.status] = summary["counts"].get(result.status, 0) + 1

            if result.status in {"downloaded", "skipped"}:
                if result.instrument in OPTIONS_BVOL_SYMBOLS:
                    per_group_available_days[f"options:{result.instrument}"].add(result.day)
                else:
                    # instrument name here is symbol; market/data_type encoded in key path.
                    parts = result.key.split("/")
                    # data/futures/{market}/daily/{data_type}/{symbol}/...
                    if len(parts) >= 6 and parts[0] == "data" and parts[1] == "futures":
                        market = parts[2]
                        data_type = parts[4]
                        symbol = parts[5]
                        per_group_available_days[f"futures:{market}:{symbol}:{data_type}"].add(result.day)

            if i % 100 == 0:
                print(
                    f"[multi-phase1] progress {i}/{len(tasks)} "
                    f"downloaded={summary['counts'].get('downloaded', 0)} "
                    f"missing={summary['counts'].get('missing', 0)} "
                    f"failed={summary['counts'].get('failed', 0)}",
                    flush=True,
                )

    # Build synchronized daily index as intersection across required groups.
    required_group_keys = []
    for symbol in TOP10_UM_SYMBOLS:
        for dt in FUTURES_DATA_TYPES:
            required_group_keys.append(f"futures:um:{symbol}:{dt}")
    for symbol in TOP10_CM_SYMBOLS:
        for dt in FUTURES_DATA_TYPES:
            required_group_keys.append(f"futures:cm:{symbol}:{dt}")
    for symbol in OPTIONS_BVOL_SYMBOLS:
        required_group_keys.append(f"options:{symbol}")

    common_days = set(required_days)
    for group in required_group_keys:
        group_days = per_group_available_days.get(group, set())
        common_days &= group_days

    sync_df = pd.DataFrame({"timestamp": sorted(required_days)})
    sync_df["timestamp"] = pd.to_datetime(sync_df["timestamp"], utc=True)
    sync_df["all_instruments_available"] = sync_df["timestamp"].dt.date.isin(common_days)
    sync_path = root / "multi_instrument" / "common_timestamp_index.parquet"
    sync_df.to_parquet(sync_path, index=False)

    summary["synchronized_index"] = {
        "path": str(sync_path).replace("\\", "/"),
        "required_days": len(required_days),
        "common_days": len(common_days),
    }

    summary_path = root / "multi_instrument" / "PHASE1_MULTI_INSTRUMENT_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[multi-phase1] summary={summary_path}", flush=True)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict real-data multi-instrument ingestion")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--root", type=str, default="Dataset")
    args = parser.parse_args()

    print("=" * 100)
    print("PHASE 1 EXPANDED: MULTI-INSTRUMENT REAL DATA INGESTION")
    print("=" * 100)
    print(f"start_utc={datetime.now(timezone.utc).isoformat()}")
    print(f"days={args.days} concurrency={args.concurrency} root={args.root}")

    try:
        summary = asyncio.run(run_ingestion(Path(args.root), args.days, args.concurrency))
        print(
            "[multi-phase1] complete "
            f"downloaded={summary['counts'].get('downloaded', 0)} "
            f"skipped={summary['counts'].get('skipped', 0)} "
            f"missing={summary['counts'].get('missing', 0)} "
            f"failed={summary['counts'].get('failed', 0)}",
            flush=True,
        )
        return 0
    except Exception as ex:
        print(f"[multi-phase1] fatal={type(ex).__name__}: {ex}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
