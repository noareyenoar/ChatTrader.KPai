"""
HF Layer Backfill Script
========================
Directly writes new records from two additional HF datasets into months
that are currently 0-byte "no-data" markers.  Bypasses the resume check
so that existing 0-byte files are filled with real data.

New coverage added:
  hf_news  2022-10 → 2023-03  (144K articles from SahandNZ/cryptonews)
  hf_social 2014-09 → 2014-12 (77K BTC tweets from cvnberk)

Safe to re-run: already-filled months (size > 0) are NEVER overwritten.

Run once from workspace root:
    .venv\\Scripts\\python.exe salvage_hf_backfill.py
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
from datetime import datetime, timezone

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

DATA_ROOT     = pathlib.Path("Dataset/phase5_time_machine_dataset")
HF_NEWS_DIR   = DATA_ROOT / "hf_news"
HF_SOCIAL_DIR = DATA_ROOT / "hf_social"

# Months where new data is available and current files are 0-byte
NEWS_BACKFILL_MONTHS   = frozenset(
    {(2022, 10), (2022, 11), (2022, 12), (2023, 1), (2023, 2), (2023, 3)}
)
SOCIAL_BACKFILL_MONTHS = frozenset(
    {(2014, 9), (2014, 10), (2014, 11), (2014, 12)}
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ts_ms(year: int, month: int, day: int = 15, hour: int = 12) -> int:
    """Return UTC epoch milliseconds for the given date/time."""
    dt = datetime(year, month, day, hour, 0, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000)


def _parse_ym(s: str) -> tuple[int, int] | None:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM' → (year, month). None on failure."""
    if not s or len(s) < 7:
        return None
    try:
        return (int(s[:4]), int(s[5:7]))
    except (ValueError, IndexError):
        return None


def _safe_day(date_str: str) -> int:
    try:
        return int(date_str[8:10]) if len(date_str) >= 10 else 15
    except (ValueError, IndexError):
        return 15


def write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("  Wrote %5d records → %s  (%d bytes)", len(records), path.name, path.stat().st_size)


def _should_write(path: pathlib.Path) -> bool:
    """Return True only if the file does not exist or is 0-byte."""
    return not path.exists() or path.stat().st_size == 0


# ── news backfill ─────────────────────────────────────────────────────────────

def backfill_news() -> None:
    """
    Stream SahandNZ/cryptonews-articles-with-price-momentum-labels and write
    records to hf_news JSONL files for months 2022-10 → 2023-03.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets library not installed; cannot run news backfill")
        return

    log.info("=== NEWS BACKFILL  SahandNZ/cryptonews-articles ===")

    monthly: dict[tuple[int, int], list[dict]] = {ym: [] for ym in NEWS_BACKFILL_MONTHS}

    try:
        ds = load_dataset(
            "SahandNZ/cryptonews-articles-with-price-momentum-labels",
            split="train",
            streaming=True,
        )
    except Exception as exc:
        log.error("Cannot load SahandNZ dataset: %s", exc)
        return

    loaded = 0
    for row in ds:
        date_str = str(row.get("datetime") or row.get("date") or "")
        ym = _parse_ym(date_str)
        if ym not in NEWS_BACKFILL_MONTHS:
            continue
        day = _safe_day(date_str)
        ts  = _ts_ms(ym[0], ym[1], day)
        text = str(row.get("text") or "")[:1000]
        monthly[ym].append(
            {
                "timestamp_utc": ts,
                "title":         text[:60],   # dataset has no separate title field
                "text":          text,
                "url":           str(row.get("url") or "")[:256],
                "source":        "hf_cryptonews",
            }
        )
        loaded += 1

    log.info("Loaded %d records for %d target months", loaded, len([m for m in monthly if monthly[m]]))

    for ym in sorted(NEWS_BACKFILL_MONTHS):
        year, month = ym
        path = HF_NEWS_DIR / f"hf_news_{year}_{month:02d}.jsonl"
        if not _should_write(path):
            log.info("  SKIP %s (already contains data)", path.name)
            continue
        records = monthly.get(ym, [])
        if not records:
            log.info("  SKIP %s (no records from SahandNZ for this month)", path.name)
            continue
        write_jsonl(path, records)


# ── social backfill ───────────────────────────────────────────────────────────

def backfill_social() -> None:
    """
    Stream cvnberk/bitcoin_tweets_sentiment_kaggle and write records to
    hf_social JSONL files for months 2014-09 → 2014-12.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets library not installed; cannot run social backfill")
        return

    log.info("=== SOCIAL BACKFILL  cvnberk/bitcoin_tweets_sentiment_kaggle ===")

    monthly: dict[tuple[int, int], list[dict]] = {ym: [] for ym in SOCIAL_BACKFILL_MONTHS}

    try:
        ds = load_dataset(
            "cvnberk/bitcoin_tweets_sentiment_kaggle",
            split="train",
            streaming=True,
        )
    except Exception as exc:
        log.error("Cannot load cvnberk dataset: %s", exc)
        return

    loaded = 0
    for row in ds:
        date_str = str(row.get("Date") or "")
        ym = _parse_ym(date_str)
        if ym not in SOCIAL_BACKFILL_MONTHS:
            continue
        day = _safe_day(date_str)
        ts  = _ts_ms(ym[0], ym[1], day)
        monthly[ym].append(
            {
                "timestamp_utc": ts,
                "text":          str(row.get("text") or "")[:512],
                "ticker":        "BTCUSD",
                "author":        "",
                "retweet_count": 0,
                "like_count":    0,
                "comment_count": 0,
                "source":        "hf_btc_tweets",
            }
        )
        loaded += 1

    log.info("Loaded %d records for %d target months", loaded, len([m for m in monthly if monthly[m]]))

    for ym in sorted(SOCIAL_BACKFILL_MONTHS):
        year, month = ym
        path = HF_SOCIAL_DIR / f"hf_social_{year}_{month:02d}.jsonl"
        if not _should_write(path):
            log.info("  SKIP %s (already contains data)", path.name)
            continue
        records = monthly.get(ym, [])
        if not records:
            log.info("  SKIP %s (no records from cvnberk for this month)", path.name)
            continue
        write_jsonl(path, records)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    backfill_news()
    backfill_social()
    log.info("=== HF backfill complete ===")
