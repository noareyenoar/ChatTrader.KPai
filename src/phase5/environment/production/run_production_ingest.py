"""
Production Ingestion Orchestrator
==================================
Deep History Scavenger Protocol  -  2013-01-01 to 2024-12-31

API keys loaded from Dataset/.env:
    TWELVE_DATA_API_KEY      Twelve Data  (tradfi_macro)
    POLYGON_API_KEY          Polygon.io News (crypto_news/crises/regulatory)
    BINANCE_API_KEY          Binance (rate limit enhancement)
    BINANCE_API_SECRET       Binance
    COINGECKO_API_URL        CoinGecko demo key embedded in URL

Usage
-----
    python src/phase5/environment/production/run_production_ingest.py \
        --start 2013-01-01 --end 2024-12-31 \
        --data-root Dataset/phase5_time_machine_dataset

Layers
------
    fear_greed       Parquet  - alternative.me (full history)
    tradfi_macro     Parquet  - Twelve Data + yfinance + FRED CSV
    on_chain         Parquet  - CoinGecko + CoinMetrics community
    derivatives      Parquet  - Binance Data Vision klines + fundingRate ZIPs
    crypto_news      JSONL    - Polygon.io News API (BTC+ETH)
    crises_hacks     JSONL    - Polygon.io News (crisis-classified)
    regulatory_legal JSONL    - Polygon.io News (regulatory-classified)
    social_sentiment JSONL    - Google Trends via pytrends
"""
from __future__ import annotations

import argparse
import asyncio
import calendar
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── Load .env (Dataset/.env in repo root) ─────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ENV_FILE  = _REPO_ROOT / "Dataset" / ".env"
try:
    from dotenv import load_dotenv
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
        _envloaded = True
    else:
        _envloaded = False
except ImportError:
    _envloaded = False

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("time_machine.ingest")

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Local imports ─────────────────────────────────────────────────────────────
from src.phase5.environment.production.chunked_storage  import ChunkedLayerStorage
from src.phase5.environment.production.synthetic_imputer import SyntheticImputer
from src.phase5.environment.production.clients.fear_greed_client  import FearGreedClient
from src.phase5.environment.production.clients.tradfi_client      import TradFiClient
from src.phase5.environment.production.clients.derivatives_client import DerivativesClient
from src.phase5.environment.production.clients.onchain_client     import OnChainClient
from src.phase5.environment.production.clients.crypto_news_client import CryptoNewsClient
from src.phase5.environment.production.clients.social_client      import SocialClient
from src.phase5.environment.production.clients.hf_enrichment_client import HFSocialClient, HFNewsClient

# Derivatives symbols – order determines storage sort order.
# SOLUSDT and BNBUSDT backfilled via scripts/phase5/salvage_derivatives.py.
# LINKUSDT will be populated on a full re-ingest from Binance Data Vision.
from src.phase5.environment.schemas import DERIVATIVES_SYMBOLS
_DERIVATIVES_SYMBOLS = list(DERIVATIVES_SYMBOLS)


@dataclass
class IngestConfig:
    start:     str       = "2013-01-01"
    end:       str       = "2024-12-31"
    data_root: Path      = Path("Dataset/phase5_time_machine_dataset")
    resume:    bool      = True
    layers:    list[str] = field(default_factory=lambda: [
        "fear_greed", "tradfi_macro", "on_chain",
        "derivatives", "crypto_news", "social_sentiment",
        "hf_social", "hf_news",
    ])
    overwrite: bool = False


def _months_in_range(start: str, end: str) -> list[tuple[int, int]]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    months: list[tuple[int, int]] = []
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


# ── fear_greed ────────────────────────────────────────────────────────────────

async def _ingest_fear_greed(
    cfg: IngestConfig, storage: ChunkedLayerStorage,
    imputer: SyntheticImputer, months: list[tuple[int, int]],
) -> dict[str, int]:
    written = skipped = 0
    try:
        async with FearGreedClient() as client:
            full_df = await client.fetch_all()
        if full_df.empty:
            raise RuntimeError("empty F&G response")
        for y, m in months:
            if cfg.resume and storage.month_exists("fear_greed", y, m):
                skipped += 1; continue
            month_df = client.slice_month(full_df, y, m)
            if month_df.empty:
                month_df = imputer.create_fear_greed_month(y, m)
            p = storage.save_monthly_parquet(month_df, "fear_greed", y, m)
            if p:
                written += 1; log.info("fear_greed  %d-%02d  -> %s", y, m, p.name)
    except Exception as exc:
        log.error("F&G fetch failed: %s – synthetic fallback", exc)
        for y, m in months:
            if cfg.resume and storage.month_exists("fear_greed", y, m):
                skipped += 1; continue
            df2 = imputer.create_fear_greed_month(y, m)
            df2["is_synthetic"] = True
            p = storage.save_monthly_parquet(df2, "fear_greed", y, m)
            if p: written += 1
    return {"written": written, "skipped": skipped}


# ── tradfi_macro ──────────────────────────────────────────────────────────────

async def _ingest_tradfi(
    cfg: IngestConfig, storage: ChunkedLayerStorage,
    imputer: SyntheticImputer, months: list[tuple[int, int]],
) -> dict[str, int]:
    written = skipped = 0
    loop   = asyncio.get_event_loop()
    client = TradFiClient()
    try:
        full_df = await loop.run_in_executor(None, client.fetch_daily_layer, cfg.start, cfg.end)
        for y, m in months:
            if cfg.resume and storage.month_exists("tradfi_macro", y, m):
                skipped += 1; continue
            month_df = client.slice_month(full_df, y, m) if not full_df.empty else None
            if month_df is None or month_df.empty:
                log.warning("tradfi_macro %d-%02d: no real data – synthetic", y, m)
                month_df = imputer.create_tradfi_macro_month(y, m)
                month_df["is_synthetic"] = True
            p = storage.save_monthly_parquet(month_df, "tradfi_macro", y, m)
            if p: written += 1; log.info("tradfi_macro  %d-%02d  -> %s", y, m, p.name)
    except Exception as exc:
        log.error("TradFi fetch failed: %s", exc)
        for y, m in months:
            if cfg.resume and storage.month_exists("tradfi_macro", y, m):
                skipped += 1; continue
            df2 = imputer.create_tradfi_macro_month(y, m)
            df2["is_synthetic"] = True
            p = storage.save_monthly_parquet(df2, "tradfi_macro", y, m)
            if p: written += 1
    return {"written": written, "skipped": skipped}


# ── on_chain ──────────────────────────────────────────────────────────────────

async def _ingest_on_chain(
    cfg: IngestConfig, storage: ChunkedLayerStorage,
    imputer: SyntheticImputer, months: list[tuple[int, int]],
) -> dict[str, int]:
    written = skipped = 0
    client = OnChainClient()
    for y, m in months:
        if cfg.resume and storage.month_exists("on_chain", y, m):
            skipped += 1; continue
        try:
            month_df = await client.build_layer(y, m)
        except Exception as exc:
            log.warning("on_chain %d-%02d error: %s – synthetic", y, m, exc)
            month_df = imputer.create_on_chain_month("BTC", y, m)
            month_df["is_synthetic"] = True
        if month_df.empty:
            month_df = imputer.create_on_chain_month("BTC", y, m)
            month_df["is_synthetic"] = True
        p = storage.save_monthly_parquet(month_df, "on_chain", y, m)
        if p: written += 1; log.info("on_chain  %d-%02d  -> %s", y, m, p.name)
    return {"written": written, "skipped": skipped}


# ── derivatives ───────────────────────────────────────────────────────────────

async def _ingest_derivatives(
    cfg: IngestConfig, storage: ChunkedLayerStorage,
    imputer: SyntheticImputer, months: list[tuple[int, int]],
) -> dict[str, int]:
    import pandas as pd
    written = skipped = 0
    client = DerivativesClient()
    for y, m in months:
        if (y, m) < (2019, 9):          # Binance perps didn't exist before Sep 2019
            skipped += 1; continue
        if cfg.resume and storage.month_exists("derivatives", y, m):
            skipped += 1; continue
        try:
            month_df = await client.build_all_symbols(_DERIVATIVES_SYMBOLS, y, m)
        except Exception as exc:
            log.warning("Derivatives %d-%02d error: %s", y, m, exc)
            month_df = pd.DataFrame()
        if month_df.empty:
            frames = [imputer.create_derivatives_month(s, y, m) for s in _DERIVATIVES_SYMBOLS]
            month_df = pd.concat(frames, ignore_index=True)
            month_df["is_synthetic"] = True
        p = storage.save_monthly_parquet(month_df, "derivatives", y, m)
        if p: written += 1; log.info("derivatives  %d-%02d  -> %s", y, m, p.name)
    return {"written": written, "skipped": skipped}


# ── crypto_news / crises / regulatory ────────────────────────────────────────

async def _ingest_crypto_news(
    cfg: IngestConfig, storage: ChunkedLayerStorage,
    months: list[tuple[int, int]],
) -> dict[str, int]:
    written = skipped = 0
    layer_keys = ["crypto_news", "crises_hacks", "regulatory_legal"]
    client = CryptoNewsClient()
    for y, m in months:
        all_skip = all(cfg.resume and storage.month_exists(lk, y, m) for lk in layer_keys)
        if all_skip:
            skipped += len(layer_keys); continue
        try:
            results = await client.fetch_month(y, m)
        except Exception as exc:
            log.warning("CryptoNews %d-%02d failed: %s", y, m, exc)
            results = {lk: [] for lk in layer_keys}
        for lk in layer_keys:
            if cfg.resume and storage.month_exists(lk, y, m):
                skipped += 1; continue
            events = results.get(lk, [])
            p = storage.save_monthly_jsonl(events, lk, y, m)
            if p:
                written += 1
                log.info("%s  %d-%02d  -> %s (%d events)", lk, y, m, p.name, len(events))
    return {"written": written, "skipped": skipped}


# ── social_sentiment ──────────────────────────────────────────────────────────

async def _ingest_social(
    cfg: IngestConfig, storage: ChunkedLayerStorage,
    months: list[tuple[int, int]],
) -> dict[str, int]:
    written = skipped = 0
    client = SocialClient()
    loop   = asyncio.get_event_loop()

    # Google Trends: fetch entire history once (pytrends is synchronous)
    weekly_df = await loop.run_in_executor(None, client.fetch_all_time)
    if weekly_df is None:
        log.warning("Google Trends unavailable – social_sentiment will be empty")
        # Still create empty files so Oracle does not fail
        for y, m in months:
            if cfg.resume and storage.month_exists("social_sentiment", y, m):
                skipped += 1; continue
            storage.save_monthly_jsonl([], "social_sentiment", y, m)
            written += 1
        return {"written": written, "skipped": skipped}

    for y, m in months:
        if cfg.resume and storage.month_exists("social_sentiment", y, m):
            skipped += 1; continue
        try:
            records = client.slice_month(weekly_df, y, m)
        except Exception as exc:
            log.warning("Social slice %d-%02d failed: %s", y, m, exc)
            records = []
        p = storage.save_monthly_jsonl(records, "social_sentiment", y, m)
        if p:
            written += 1
            log.info("social_sentiment  %d-%02d  -> %s (%d records)", y, m, p.name, len(records))
    return {"written": written, "skipped": skipped}


# ── hf_social ─────────────────────────────────────────────────────────────────

async def _ingest_hf_social(
    cfg: IngestConfig, storage: ChunkedLayerStorage,
) -> dict[str, int]:
    """Stream mjw/stock_market_tweets → hf_social monthly JSONL."""
    written = skipped = 0
    loop   = asyncio.get_event_loop()
    client = HFSocialClient()
    start_ym = tuple(int(x) for x in cfg.start[:7].split("-"))
    end_ym   = tuple(int(x) for x in cfg.end[:7].split("-"))

    def _stream():
        return list(client.stream_by_month(
            start_ym=start_ym, end_ym=end_ym  # type: ignore[arg-type]
        ))

    try:
        monthly_data = await loop.run_in_executor(None, _stream)
    except Exception as exc:
        log.error("hf_social stream failed: %s", exc)
        return {"written": 0, "skipped": 0}

    for (y, m), records in monthly_data:
        if cfg.resume and storage.month_exists("hf_social", y, m):
            skipped += 1; continue
        p = storage.save_monthly_jsonl(records, "hf_social", y, m)
        if p:
            written += 1
            log.info("hf_social  %d-%02d  -> %s (%d records)", y, m, p.name, len(records))

    # Write 0-byte files for months in range that had no data (needed for resume)
    produced_months = {ym for (ym, _) in monthly_data}
    all_months = _months_in_range(cfg.start, cfg.end)
    for y, m in all_months:
        if not storage.month_exists("hf_social", y, m) and (y, m) not in produced_months:
            storage.save_monthly_jsonl([], "hf_social", y, m)
            written += 1

    return {"written": written, "skipped": skipped}


# ── hf_news ───────────────────────────────────────────────────────────────────

async def _ingest_hf_news(
    cfg: IngestConfig, storage: ChunkedLayerStorage,
) -> dict[str, int]:
    """Stream ashraq/financial-news-articles → hf_news monthly JSONL."""
    written = skipped = 0
    loop   = asyncio.get_event_loop()
    client = HFNewsClient()
    start_ym = tuple(int(x) for x in cfg.start[:7].split("-"))
    end_ym   = tuple(int(x) for x in cfg.end[:7].split("-"))

    def _stream():
        return list(client.stream_by_month(
            start_ym=start_ym, end_ym=end_ym  # type: ignore[arg-type]
        ))

    try:
        monthly_data = await loop.run_in_executor(None, _stream)
    except Exception as exc:
        log.error("hf_news stream failed: %s", exc)
        return {"written": 0, "skipped": 0}

    for (y, m), records in monthly_data:
        if cfg.resume and storage.month_exists("hf_news", y, m):
            skipped += 1; continue
        p = storage.save_monthly_jsonl(records, "hf_news", y, m)
        if p:
            written += 1
            log.info("hf_news  %d-%02d  -> %s (%d records)", y, m, p.name, len(records))

    # Write 0-byte files for months with no data
    produced_months = {ym for (ym, _) in monthly_data}
    all_months = _months_in_range(cfg.start, cfg.end)
    for y, m in all_months:
        if not storage.month_exists("hf_news", y, m) and (y, m) not in produced_months:
            storage.save_monthly_jsonl([], "hf_news", y, m)
            written += 1

    return {"written": written, "skipped": skipped}


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_ingest(cfg: IngestConfig) -> None:
    data_root = cfg.data_root if cfg.data_root.is_absolute() else _REPO_ROOT / cfg.data_root
    storage   = ChunkedLayerStorage(data_root, overwrite=cfg.overwrite)
    imputer   = SyntheticImputer(seed=42)
    months    = _months_in_range(cfg.start, cfg.end)

    log.info("=== Phase 5 Production Ingest  [Deep History Scavenger Protocol] ===")
    log.info(".env loaded: %s  (%s)", _envloaded, _ENV_FILE)
    log.info("Range  : %s -> %s  (%d months)", cfg.start, cfg.end, len(months))
    log.info("Layers : %s", ", ".join(cfg.layers))
    log.info("Resume : %s   Overwrite: %s", cfg.resume, cfg.overwrite)
    log.info("Data   : %s", data_root)
    log.info("TWELVE_DATA_KEY  : %s", "SET" if os.getenv("TWELVE_DATA_API_KEY") else "missing")
    log.info("POLYGON_KEY      : %s", "SET" if os.getenv("POLYGON_API_KEY")      else "missing")
    log.info("COINGECKO_URL    : %s", "SET" if os.getenv("COINGECKO_API_URL")    else "missing")

    stats: dict[str, dict] = {}

    if "fear_greed" in cfg.layers:
        log.info("── fear_greed ──")
        stats["fear_greed"] = await _ingest_fear_greed(cfg, storage, imputer, months)

    if "tradfi_macro" in cfg.layers:
        log.info("── tradfi_macro ──")
        stats["tradfi_macro"] = await _ingest_tradfi(cfg, storage, imputer, months)

    if "on_chain" in cfg.layers:
        log.info("── on_chain ──")
        stats["on_chain"] = await _ingest_on_chain(cfg, storage, imputer, months)

    if "derivatives" in cfg.layers:
        log.info("── derivatives ──")
        stats["derivatives"] = await _ingest_derivatives(cfg, storage, imputer, months)

    if "crypto_news" in cfg.layers:
        log.info("── crypto_news + crises + regulatory ──")
        stats["crypto_news"] = await _ingest_crypto_news(cfg, storage, months)

    if "social_sentiment" in cfg.layers:
        log.info("── social_sentiment ──")
        stats["social_sentiment"] = await _ingest_social(cfg, storage, months)

    if "hf_social" in cfg.layers:
        log.info("── hf_social (HuggingFace stock tweets) ──")
        stats["hf_social"] = await _ingest_hf_social(cfg, storage)

    if "hf_news" in cfg.layers:
        log.info("── hf_news (HuggingFace financial news) ──")
        stats["hf_news"] = await _ingest_hf_news(cfg, storage)

    total_w = sum(v["written"] for v in stats.values())
    total_s = sum(v["skipped"] for v in stats.values())
    log.info("=== Ingest complete: %d written  %d skipped ===", total_w, total_s)
    for lk, s in stats.items():
        log.info("  %-25s  written=%d  skipped=%d", lk, s["written"], s["skipped"])

    # Oracle check
    log.info("=== Oracle Anti-Leakage Check ===")
    try:
        from src.phase5.environment.time_machine_oracle import TimeMachineOracle
        oracle   = TimeMachineOracle(data_root)
        end_dt   = datetime.strptime(cfg.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        check_ms = int(end_dt.timestamp() * 1_000)
        results  = oracle.full_integrity_check(check_ms)
        any_fail = False
        for lk, passed in results.items():
            log.info("  %-30s  %s", lk, "PASS" if passed else "FAIL")
            if not passed:
                any_fail = True
        if any_fail:
            log.error("INTEGRITY CHECK FAILED")
        else:
            log.info("All layers PASS.")
    except Exception as exc:
        log.error("Oracle check error: %s", exc)


def _parse_args() -> IngestConfig:
    p = argparse.ArgumentParser(description="Phase 5 Time Machine – production ingest")
    p.add_argument("--start",     default="2013-01-01")
    p.add_argument("--end",       default="2024-12-31")
    p.add_argument("--data-root", default="Dataset/phase5_time_machine_dataset")
    p.add_argument("--layers",    nargs="*",
                   default=["fear_greed","tradfi_macro","on_chain",
                            "derivatives","crypto_news","social_sentiment",
                            "hf_social","hf_news"])
    p.add_argument("--resume",    action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--overwrite", action="store_true", default=False)
    args = p.parse_args()
    return IngestConfig(
        start=args.start, end=args.end,
        data_root=Path(args.data_root),
        resume=args.resume, layers=args.layers, overwrite=args.overwrite,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    asyncio.run(run_ingest(cfg))
