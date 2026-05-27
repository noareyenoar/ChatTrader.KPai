"""
HuggingFace Enrichment Client
=================================
Streams publicly available HF datasets and groups records into monthly
JSONL chunks for two new Time Machine layers:

  hf_social  –  Primary: mjw/stock_market_tweets (~1.7M tweets, 2015-2022)
                Secondary: cvnberk/bitcoin_tweets_sentiment_kaggle (77K BTC tweets,
                            2014-09 → 2019-07)

  hf_news    –  Primary: ashraq/financial-news-articles (~306K articles, 2018)
                Secondary: SahandNZ/cryptonews-articles-with-price-momentum-labels
                            (144K crypto articles, 2022-10 → 2023-03)

Coverage note
-------------
  No public HF crypto-specific dataset provides clean 2013-2017 coverage.
  Pre-2014 months for hf_social and pre-2018 months for hf_news will be 0-byte
  (valid "no data" markers for the resume mechanism).

Data quality note on jsonl_exists
----------------------------------
  `jsonl_exists` returns True for any file that exists, including 0-byte.
  This is CORRECT for the resume mechanism: a 0-byte file means "we tried
  this month and the API / dataset returned nothing."  Do NOT add a size
  threshold — that would cause infinite re-processing of valid sparse months.
"""
from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

log = logging.getLogger("time_machine.hf_enrichment")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── constants ──────────────────────────────────────────────────────────────────

# Tickers to keep from stock_market_tweets as crypto-correlated risk proxies
_SOCIAL_TICKERS = frozenset({"AAPL", "AMZN", "GOOG", "GOOGL", "MSFT", "TSLA",
                               "NVDA", "META", "COIN", "MSTR"})

# Regex to extract YYYY/MM/DD from a URL path
_URL_DATE_RE = re.compile(r"/(20\d{2})/(\d{2})/(\d{2})/")


def _ts_ms(year: int, month: int, day: int = 1,
           hour: int = 12, minute: int = 0) -> int:
    """Return UTC epoch milliseconds for the given date/time."""
    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000)


def _parse_ym(date_str: str) -> tuple[int, int] | None:
    """Parse YYYY-MM-DD or YYYY-MM into (year, month).  Returns None on failure."""
    if not date_str:
        return None
    try:
        if len(date_str) >= 10:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        elif len(date_str) >= 7:
            dt = datetime.strptime(date_str[:7], "%Y-%m")
        else:
            return None
        return (dt.year, dt.month)
    except ValueError:
        return None


def _url_to_date(url: str) -> tuple[int, int, int] | None:
    """Extract (year, month, day) from a URL containing /YYYY/MM/DD/."""
    m = _URL_DATE_RE.search(url or "")
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


# ── HuggingFace Social Client ──────────────────────────────────────────────────

class HFSocialClient:
    """
    Streams mjw/stock_market_tweets (primary) + cvnberk/bitcoin_tweets (secondary)
    and groups records into monthly buckets.

    Schema per JSONL record:
        timestamp_utc  : int64 ms UTC
        text           : str – tweet body
        ticker         : str – e.g. "TSLA" or "BTCUSD"
        author         : str – Twitter screen name (empty for cvnberk)
        retweet_count  : int
        like_count     : int
        comment_count  : int
        source         : "hf_stock_tweets" | "hf_btc_tweets"
    """

    DATASET_NAME         = "mjw/stock_market_tweets"
    SECONDARY_DATASET    = "cvnberk/bitcoin_tweets_sentiment_kaggle"

    def stream_by_month(
        self,
        start_ym: tuple[int, int] = (2013, 1),
        end_ym:   tuple[int, int] = (2024, 12),
    ) -> Generator[tuple[tuple[int, int], list[dict]], None, None]:
        """
        Yields ``(year, month), records`` for each calendar month in range.
        Primary source (mjw) is streamed row-by-row to avoid loading 1.7M rows.
        Secondary source (cvnberk, 77K rows) is pre-loaded in memory and merged.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            log.error("datasets library not installed: %s", exc)
            return

        # ── Pre-load secondary source: cvnberk BTC tweets (small, safe in memory) ──
        btc_tweets: dict[tuple[int, int], list[dict]] = defaultdict(list)
        try:
            ds_btc = load_dataset(self.SECONDARY_DATASET, split="train", streaming=True)
            for row in ds_btc:
                date_str = str(row.get("Date") or "")
                ym = _parse_ym(date_str)
                if ym is None or ym < start_ym or ym > end_ym:
                    continue
                day = 15
                if len(date_str) >= 10:
                    try:
                        day = int(date_str[8:10])
                    except ValueError:
                        pass
                btc_tweets[ym].append({
                    "timestamp_utc": _ts_ms(ym[0], ym[1], day),
                    "text":          str(row.get("text") or "")[:512],
                    "ticker":        "BTCUSD",
                    "author":        "",
                    "retweet_count": 0,
                    "like_count":    0,
                    "comment_count": 0,
                    "source":        "hf_btc_tweets",
                })
            log.info(
                "HFSocialClient: pre-loaded %d BTC tweet rows (%d months) from cvnberk",
                sum(len(v) for v in btc_tweets.values()), len(btc_tweets),
            )
        except Exception as exc:
            log.warning("HFSocialClient: cvnberk load failed (continuing without): %s", exc)

        # ── Stream primary source: mjw (large dataset, streamed row-by-row) ──
        log.info("HFSocialClient: loading %s (streaming) …", self.DATASET_NAME)
        try:
            ds = load_dataset(self.DATASET_NAME, split="train", streaming=True)
        except Exception as exc:
            log.error("HFSocialClient: cannot load primary dataset: %s", exc)
            ds = []

        current_month: tuple[int, int] | None = None
        bucket: list[dict] = []

        for row in ds:
            ym = _parse_ym(str(row.get("post_date", "")))
            if ym is None:
                continue
            if ym < start_ym or ym > end_ym:
                if ym > end_ym and current_month is not None:
                    # Flush last mjw month, merged with any cvnberk records
                    combined = bucket + list(btc_tweets.pop(current_month, []))
                    if combined:
                        yield current_month, combined
                    current_month = None
                    bucket = []
                    break
                continue

            ticker = str(row.get("ticker_symbol", "")).upper()
            if _SOCIAL_TICKERS and ticker not in _SOCIAL_TICKERS:
                continue

            # Flush on month boundary
            if ym != current_month:
                if current_month is not None:
                    combined = bucket + list(btc_tweets.pop(current_month, []))
                    if combined:
                        yield current_month, combined
                current_month = ym
                bucket = []

            date_str = str(row.get("post_date", ""))
            day = 15
            if len(date_str) >= 10:
                try:
                    day = int(date_str[8:10])
                except ValueError:
                    pass
            ts = _ts_ms(ym[0], ym[1], day)

            bucket.append({
                "timestamp_utc": ts,
                "text":          str(row.get("body", ""))[:512],
                "ticker":        ticker,
                "author":        str(row.get("writer", "")),
                "retweet_count": int(row.get("retweet_num") or 0),
                "like_count":    int(row.get("like_num")    or 0),
                "comment_count": int(row.get("comment_num") or 0),
                "source":        "hf_stock_tweets",
            })

        # Flush final mjw month
        if current_month is not None:
            combined = bucket + list(btc_tweets.pop(current_month, []))
            if combined:
                yield current_month, combined

        # Yield cvnberk-only months (e.g., 2014-09 to 2014-12 before mjw coverage)
        for ym in sorted(btc_tweets.keys()):
            if btc_tweets[ym]:
                yield ym, btc_tweets[ym]


# ── HuggingFace News Client ────────────────────────────────────────────────────

class HFNewsClient:
    """
    Streams two financial news sources and groups into monthly buckets:
      - ashraq/financial-news-articles  (~306K articles, 2018, CNBC focus)
      - SahandNZ/cryptonews-articles-with-price-momentum-labels
                                         (144K crypto articles, 2022-10 → 2023-03)

    Schema per JSONL record:
        timestamp_utc  : int64 ms UTC
        title          : str
        text           : str (first 1000 chars)
        url            : str
        source         : "hf_financial_news" | "hf_cryptonews"
    """

    DATASET_NAME      = "ashraq/financial-news-articles"
    SECONDARY_DATASET = "SahandNZ/cryptonews-articles-with-price-momentum-labels"

    def stream_by_month(
        self,
        start_ym: tuple[int, int] = (2013, 1),
        end_ym:   tuple[int, int] = (2024, 12),
    ) -> Generator[tuple[tuple[int, int], list[dict]], None, None]:
        """Yields ``(year, month), records`` for each month with data."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            log.error("datasets library not installed: %s", exc)
            return

        monthly: dict[tuple[int, int], list[dict]] = defaultdict(list)
        skipped_no_date = 0

        # ── Source 1: ashraq/financial-news-articles (CNBC 2018) ──────────────
        log.info("HFNewsClient: loading %s …", self.DATASET_NAME)
        try:
            ds1 = load_dataset(self.DATASET_NAME, split="train", streaming=True)
            for row in ds1:
                url = str(row.get("url", ""))
                ymd = _url_to_date(url)
                if ymd is None:
                    skipped_no_date += 1
                    continue
                year, month, day = ymd
                ym = (year, month)
                if ym < start_ym or ym > end_ym:
                    continue
                monthly[ym].append({
                    "timestamp_utc": _ts_ms(year, month, day),
                    "title":         str(row.get("title", ""))[:256],
                    "text":          str(row.get("text",  ""))[:1000],
                    "url":           url[:256],
                    "source":        "hf_financial_news",
                })
        except Exception as exc:
            log.warning("HFNewsClient: ashraq load failed (continuing): %s", exc)

        # ── Source 2: SahandNZ crypto news (2022-10 → 2023-03) ───────────────
        log.info("HFNewsClient: loading %s …", self.SECONDARY_DATASET)
        try:
            ds2 = load_dataset(self.SECONDARY_DATASET, split="train", streaming=True)
            added2 = 0
            for row in ds2:
                date_str = str(row.get("datetime") or row.get("date") or "")
                ym = _parse_ym(date_str)
                if ym is None or ym < start_ym or ym > end_ym:
                    continue
                day = 15
                if len(date_str) >= 10:
                    try:
                        day = int(date_str[8:10])
                    except ValueError:
                        pass
                text = str(row.get("text") or "")[:1000]
                monthly[ym].append({
                    "timestamp_utc": _ts_ms(ym[0], ym[1], day),
                    "title":         text[:60],
                    "text":          text,
                    "url":           str(row.get("url") or "")[:256],
                    "source":        "hf_cryptonews",
                })
                added2 += 1
            log.info("HFNewsClient: loaded %d SahandNZ records", added2)
        except Exception as exc:
            log.warning("HFNewsClient: SahandNZ load failed (continuing): %s", exc)

        log.info(
            "HFNewsClient: %d records across %d months  (%d skipped no-date)",
            sum(len(v) for v in monthly.values()), len(monthly), skipped_no_date,
        )

        for ym in sorted(monthly.keys()):
            yield ym, monthly[ym]
