"""
Crypto News client  -  Layers 4, 7 & 8 (crypto_news, regulatory_legal, crises_hacks)

Source: Polygon.io News API  https://api.polygon.io/v2/reference/news
Key:    POLYGON_API_KEY env var   (Starter plan free: 5 req/min, unlimited history)

Coverage: X:BTCUSD + X:ETHUSD news. Paginated via cursor. Rate: 12 s/request.
Classification:
  - Polygon provides insights[].sentiment: positive / negative / neutral
  - Crisis keywords  -> crises_hacks layer
  - Regulatory terms -> regulatory_legal layer
  - Otherwise        -> crypto_news layer
"""
from __future__ import annotations
import asyncio, logging, os, time
from datetime import datetime, timezone
from typing import Any
import aiohttp
from ..base_client import AsyncRateLimitedClient

log = logging.getLogger("time_machine.crypto_news")

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")
_BASE = "https://api.polygon.io/v2/reference/news"
_TICKERS = ["BTC", "ETH"]

_CRISIS_KEYWORDS = {
    "hack","hacked","hacking","exploit","exploited","breach","stolen","theft","flash loan",
    "rug pull","exit scam","ponzi","fraud","bankrupt","insolvency","collapse","crashed",
    "luna","terra","ftx","celsius","three arrows","3ac","bitfinex hack","mt gox","bitconnect",
    "security incident","vulnerability","attack","drained","frozen","investigation",
    "black thursday","liquidation cascade",
}
_REGULATORY_KEYWORDS = {
    "sec","cftc","fca","bafin","mas","fsb","fatf","bis","oecd","g20",
    "regulation","regulatory","compliance","enforcement","lawsuit","subpoena",
    "ban","illegal","legal","legislation","law","policy","sanction","sanctioned",
    "kyc","aml","tax","irs","treasury","cbdc","licensing","registered","approved",
    "court","ruling","settlement","consent order","fine","penalty",
    "gensler","warren","mnuchin","yellen","powell",
}


def _classify(title: str, description: str, keywords: list[str]) -> str:
    text = (title + " " + description + " " + " ".join(keywords or [])).lower()
    words = set(text.replace(",", " ").replace(".", " ").split())
    if words & _CRISIS_KEYWORDS:
        return "crises_hacks"
    if words & _REGULATORY_KEYWORDS:
        return "regulatory_legal"
    return "crypto_news"


def _sentiment_score(insights: list[dict]) -> float:
    """Map Polygon sentiment to [-1, 1]."""
    if not insights:
        return 0.0
    mapping = {"positive": 1.0, "bullish": 1.0, "negative": -1.0, "bearish": -1.0, "neutral": 0.0}
    scores = [mapping.get(i.get("sentiment", "neutral"), 0.0) for i in insights]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


class CryptoNewsClient:
    """Fetches and classifies crypto news from Polygon.io News API."""

    def __init__(self) -> None:
        self._http = AsyncRateLimitedClient(rps=0.08, max_retries=4, backoff_base=2.0, timeout_s=60)
        # 0.08 rps = ~12 s per request  (Polygon free: 5 req/min)

    async def fetch_month(
        self, year: int, month: int
    ) -> dict[str, list[dict]]:
        """Return {'crypto_news': [...], 'crises_hacks': [...], 'regulatory_legal': [...]}."""
        if not POLYGON_KEY:
            log.warning("POLYGON_API_KEY not set - skipping news fetch")
            return {"crypto_news": [], "crises_hacks": [], "regulatory_legal": []}

        import calendar
        days = calendar.monthrange(year, month)[1]
        start_str = f"{year}-{month:02d}-01T00:00:00Z"
        end_str   = f"{year}-{month:02d}-{days:02d}T23:59:59Z"

        all_articles: list[dict] = []
        async with self._http:
            for ticker in _TICKERS:
                articles = await self._paginate_ticker(ticker, start_str, end_str)
                all_articles.extend(articles)
                log.info("Polygon news %s %d-%02d: %d articles", ticker, year, month, len(articles))

        # De-duplicate by article_url + classify
        seen_urls: set[str] = set()
        result: dict[str, list[dict]] = {"crypto_news": [], "crises_hacks": [], "regulatory_legal": []}

        for art in all_articles:
            url = art.get("article_url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            pub_str = art.get("published_utc", "")
            try:
                ts_ms = int(datetime.fromisoformat(pub_str.replace("Z", "+00:00")).timestamp() * 1_000)
            except Exception:
                continue

            title       = art.get("title", "")
            description = art.get("description", "") or ""
            keywords    = art.get("keywords", []) or []
            insights    = art.get("insights",  []) or []

            layer = _classify(title, description, keywords)
            record = {
                "timestamp_utc":    ts_ms,
                "source":           art.get("publisher", {}).get("name", "unknown"),
                "headline":         title,
                "body_snippet":     description[:500],
                "url":              url,
                "sentiment_label":  (insights[0].get("sentiment", "neutral") if insights else "neutral"),
                "sentiment_score":  _sentiment_score(insights),
                "keywords":         ",".join(keywords[:10]),
                "layer":            layer,
                "is_synthetic":     False,
            }
            result[layer].append(record)

        log.info("Polygon %d-%02d classified: news=%d crises=%d reg=%d",
                 year, month, len(result["crypto_news"]),
                 len(result["crises_hacks"]), len(result["regulatory_legal"]))
        return result

    async def _paginate_ticker(
        self, ticker: str, start: str, end: str
    ) -> list[dict]:
        """Paginate through all Polygon news for one ticker in a date range."""
        articles: list[dict] = []
        params: dict[str, Any] = {
            "ticker":               ticker,
            "limit":                1000,
            "published_utc.gte":    start,
            "published_utc.lte":    end,
            "order":                "asc",
            "sort":                 "published_utc",
            "apiKey":               POLYGON_KEY,
        }
        page = 0
        while True:
            data = await self._http.get(_BASE, params=params)
            if not data:
                break
            batch = data.get("results", [])
            articles.extend(batch)
            page += 1
            next_url = data.get("next_url")
            if not next_url or not batch:
                break
            # next_url already has cursor; just append apiKey
            params = {"cursor": _extract_cursor(next_url), "apiKey": POLYGON_KEY}
            if page % 10 == 0:
                log.info("Polygon %s: fetched %d articles so far (page %d)", ticker, len(articles), page)
        return articles


def _extract_cursor(next_url: str) -> str:
    """Extract cursor parameter from Polygon next_url."""
    import urllib.parse
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(next_url).query)
    return qs.get("cursor", [""])[0]
