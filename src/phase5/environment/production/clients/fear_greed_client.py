"""
Fear & Greed Index client – alternative.me Crypto Fear & Greed API.

No API key required.  Returns the **complete** historical daily series
in a single call (limit=0 fetches all available rows back to 2018-02-01).

Endpoint: https://api.alternative.me/fng/?limit=0&format=json
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from src.phase5.environment.production.base_client import AsyncRateLimitedClient

log = logging.getLogger("time_machine.fear_greed")

_FNG_URL = "https://api.alternative.me/fng/"

_LABEL_MAP = {
    "Extreme Fear":  "extreme_fear",
    "Fear":          "fear",
    "Neutral":       "neutral",
    "Greed":         "greed",
    "Extreme Greed": "extreme_greed",
}


class FearGreedClient:
    """
    Fetches the complete Crypto Fear & Greed Index history.

    Usage
    -----
        async with FearGreedClient() as client:
            df = await client.fetch_all()
    """

    def __init__(self) -> None:
        self._http = AsyncRateLimitedClient(rps=1.0, max_retries=4)

    async def __aenter__(self) -> "FearGreedClient":
        await self._http.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._http.__aexit__(*args)

    async def fetch_all(self) -> pd.DataFrame:
        """
        Return all historical daily rows as a DataFrame.

        Columns match Layer 10 schema plus ``is_synthetic`` (always False
        for real data).  ``timestamp_utc`` is int64 ms epoch UTC aligned
        to the start of each UTC day.
        """
        log.info("Fetching Fear & Greed full history from alternative.me …")
        payload = await self._http.get(
            _FNG_URL,
            params={"limit": "0", "format": "json"},
        )

        data = payload.get("data", [])
        if not data:
            log.error("Empty response from alternative.me FNG API")
            return pd.DataFrame()

        rows = []
        for entry in data:
            # API timestamp is a Unix epoch **seconds** string
            ts_s   = int(entry["timestamp"])
            ts_ms  = ts_s * 1_000
            dt     = datetime.fromtimestamp(ts_s, tz=timezone.utc)
            # Normalise to midnight UTC
            midnight = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
            ts_ms    = int(midnight.timestamp() * 1_000)

            fg_val   = int(entry["value"])
            raw_label = entry.get("value_classification", "")
            label    = _LABEL_MAP.get(raw_label, raw_label.lower().replace(" ", "_"))

            rows.append({
                "timestamp_utc":               ts_ms,
                "date_str":                    midnight.strftime("%Y-%m-%d"),
                "fear_greed_index":            fg_val,
                "fear_greed_label":            label,
                # Google Trends columns are not available from this API
                # – left as NaN for downstream imputation / real Trends fetch
                "google_trends_bitcoin":       float("nan"),
                "google_trends_crypto":        float("nan"),
                "google_trends_buy_crypto":    float("nan"),
                "google_trends_ethereum":      float("nan"),
                "google_trends_bitcoin_crash": float("nan"),
                "google_trends_sell_bitcoin":  float("nan"),
                "retail_search_composite":     float("nan"),
                "is_synthetic":                False,
            })

        df = (
            pd.DataFrame(rows)
            .sort_values("timestamp_utc")
            .drop_duplicates("timestamp_utc")
            .reset_index(drop=True)
        )
        log.info("Fear & Greed: %d daily rows fetched (oldest: %s)", len(df), df["date_str"].min())
        return df

    def slice_month(self, df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
        """Filter a full-history DataFrame down to a single year/month."""
        from datetime import timezone
        import calendar
        days = calendar.monthrange(year, month)[1]
        start_ms = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1_000)
        end_ms   = int(datetime(year, month, days, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1_000)
        mask = (df["timestamp_utc"] >= start_ms) & (df["timestamp_utc"] <= end_ms)
        return df[mask].copy()
