"""
Async HTTP base client with token-bucket rate limiting,
exponential back-off, per-host proxy support, and structured logging.

All production API clients in this package inherit from or instantiate
``AsyncRateLimitedClient``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import aiohttp

log = logging.getLogger("time_machine.http")


# ─────────────────────────────────────────────────────────────────────────────
# Token-bucket rate limiter
# ─────────────────────────────────────────────────────────────────────────────

class TokenBucketRateLimiter:
    """
    Thread-safe async token bucket.

    Parameters
    ----------
    rate : float
        Maximum sustained requests-per-second.
    burst : float
        Maximum burst capacity (tokens).  Defaults to ``rate``.
    """

    def __init__(self, rate: float, burst: float | None = None) -> None:
        self._rate   = rate
        self._burst  = burst if burst is not None else rate
        self._tokens = self._burst
        self._last   = time.monotonic()
        self._lock   = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last
            self._last   = now
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)

            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Main async client
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_HEADERS = {
    "User-Agent": (
        "ChatTrader.KPai/2.0 TimeMachineProductionIngestor "
        "(+https://github.com/noareyenoar/ChatTrader.KPai)"
    ),
    "Accept": "application/json, text/plain, */*",
}


class AsyncRateLimitedClient:
    """
    Async HTTP client with:
      * Token-bucket rate limiting
      * Exponential back-off with full jitter
      * Respect for ``Retry-After`` headers (429 / 503)
      * Optional proxy support
      * Clean ``async with`` context-manager lifecycle

    Usage
    -----
        async with AsyncRateLimitedClient(rps=2.0) as client:
            data = await client.get("https://api.example.com/data")
            raw  = await client.get("https://example.com/file.zip", as_bytes=True)
    """

    def __init__(
        self,
        rps: float = 2.0,
        burst: float | None = None,
        max_retries: int = 6,
        backoff_base: float = 1.5,
        timeout_s: int = 60,
        proxy: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._rps          = rps
        self._max_retries  = max_retries
        self._backoff_base = backoff_base
        self._timeout_s    = timeout_s
        self._proxy        = proxy
        self._headers      = {**_DEFAULT_HEADERS, **(extra_headers or {})}
        self._limiter      = TokenBucketRateLimiter(rps, burst)
        self._session: aiohttp.ClientSession | None = None

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "AsyncRateLimitedClient":
        self._session = aiohttp.ClientSession(
            headers=self._headers,
            timeout=aiohttp.ClientTimeout(total=self._timeout_s),
            connector=aiohttp.TCPConnector(ssl=True, limit=20),
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ── Core HTTP methods ─────────────────────────────────────────────────────

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        as_bytes: bool = False,
    ) -> Any:
        """
        HTTP GET with rate limiting and automatic retry.

        Returns parsed JSON dict/list by default; raw bytes when
        ``as_bytes=True`` (e.g. for ZIP file downloads).
        """
        assert self._session is not None, "Use as async context manager."

        await self._limiter.acquire()

        for attempt in range(self._max_retries):
            try:
                async with self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    proxy=self._proxy,
                ) as resp:

                    # ── Rate-limit response handling ───────────────────────
                    if resp.status in (429, 503):
                        retry_after = float(
                            resp.headers.get("Retry-After", 2 ** attempt * self._backoff_base)
                        )
                        log.warning(
                            "HTTP %d on %s (attempt %d/%d) – waiting %.1fs",
                            resp.status, url, attempt + 1, self._max_retries, retry_after,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    resp.raise_for_status()

                    if as_bytes:
                        return await resp.read()
                    return await resp.json(content_type=None)

            except aiohttp.ClientResponseError as exc:
                if exc.status in (400, 401, 403, 404):
                    log.error("Non-retryable HTTP %d for %s: %s", exc.status, url, exc.message)
                    raise
                self._log_retry(url, attempt, exc)

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                self._log_retry(url, attempt, exc)

            # Exponential backoff with ±25 % jitter
            import random
            base_wait = self._backoff_base ** attempt
            wait      = base_wait * random.uniform(0.75, 1.25)
            await asyncio.sleep(wait)

        raise RuntimeError(
            f"All {self._max_retries} attempts exhausted for GET {url}"
        )

    def _log_retry(self, url: str, attempt: int, exc: Exception) -> None:
        if attempt < self._max_retries - 1:
            log.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt + 1, self._max_retries, url, exc,
            )
        else:
            log.error("All retries exhausted for %s: %s", url, exc)
