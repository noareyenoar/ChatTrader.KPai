"""
Social Sentiment client  -  Layer 3 (Social Sentiment Pulse)

Source: Google Trends via pytrends (free, no API key).
  Fetches weekly relative search interest (0-100) for bitcoin/ethereum terms.
  Maps weekly data to daily rows via forward-fill.
  Stores records compatible with the social_sentiment JSONL layer.
"""
from __future__ import annotations
import logging, time
from datetime import datetime, timezone
import numpy as np

log = logging.getLogger("time_machine.social")

_KEYWORDS = ["bitcoin", "ethereum", "buy crypto", "bitcoin crash", "sell bitcoin"]
# Normalize raw Google Trends score (0-100) to [-1, 1]
_NORM = lambda x: round((x / 50.0) - 1.0, 4) if not np.isnan(x) else 0.0


class SocialClient:
    """Fetch Google Trends data and emit social_sentiment JSONL records."""

    def fetch_all_time(self) -> "pd.DataFrame | None":
        """
        Download all-time weekly Google Trends for crypto keywords.
        Returns DataFrame indexed by UTC-aware DatetimeIndex, columns = keywords.
        Returns None on failure.
        """
        try:
            from pytrends.request import TrendReq
        except ImportError:
            log.warning("pytrends not installed – social layer will be empty")
            return None

        # urllib3 v2 renamed method_whitelist -> allowed_methods; pytrends 4.x still uses the old name
        try:
            import urllib3.util.retry as _retry_mod
            _orig_retry_init = _retry_mod.Retry.__init__
            def _compat_retry_init(self, *args, **kwargs):
                if "method_whitelist" in kwargs:
                    kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
                _orig_retry_init(self, *args, **kwargs)
            _retry_mod.Retry.__init__ = _compat_retry_init
        except Exception:
            pass  # best-effort patch

        try:
            pt = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=3, backoff_factor=2)
            pt.build_payload(_KEYWORDS, timeframe="all", geo="", gprop="")
            time.sleep(3)   # avoid Google blocking
            df = pt.interest_over_time()
            if df is None or df.empty:
                log.warning("pytrends returned empty DataFrame")
                return None
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            df.index = df.index.tz_localize("UTC") if df.index.tzinfo is None else df.index.tz_convert("UTC")
            log.info("Google Trends: %d weekly rows, keywords=%s", len(df), list(df.columns))
            return df
        except Exception as exc:
            log.warning("pytrends fetch failed: %s", exc)
            return None

    def slice_month(
        self, weekly_df: "pd.DataFrame", year: int, month: int
    ) -> list[dict]:
        """
        Convert weekly Trends data for a calendar month to daily JSONL records.
        Each record has timestamp_utc, avg_sentiment_score, mention_volume, etc.
        """
        import pandas as pd, calendar as cal
        days = cal.monthrange(year, month)[1]
        s    = pd.Timestamp(year=year, month=month, day=1,    tz="UTC")
        e    = pd.Timestamp(year=year, month=month, day=days, tz="UTC")

        # Forward-fill weekly to daily
        daily_idx = pd.date_range(s, e, freq="1D", tz="UTC")
        month_df  = weekly_df.reindex(weekly_df.index.union(daily_idx)).ffill().reindex(daily_idx)

        if month_df.empty or month_df.isnull().all().all():
            return []

        records: list[dict] = []
        for ts, row in month_df.iterrows():
            ts_ms   = int(ts.timestamp() * 1_000)
            btc_raw = float(row.get("bitcoin",  np.nan))
            eth_raw = float(row.get("ethereum", np.nan))

            # Composite sentiment: blend bitcoin + crash/sell signals
            crash_raw = float(row.get("bitcoin crash", np.nan))
            buy_raw   = float(row.get("buy crypto",    np.nan))
            sell_raw  = float(row.get("sell bitcoin",  np.nan))

            pos_signal = np.nanmean([btc_raw, eth_raw, buy_raw]) if not all(np.isnan([btc_raw, eth_raw, buy_raw])) else np.nan
            neg_signal = np.nanmean([crash_raw, sell_raw]) if not all(np.isnan([crash_raw, sell_raw])) else np.nan

            if not np.isnan(pos_signal) and not np.isnan(neg_signal):
                composite = (pos_signal - neg_signal) / 100.0
                composite = max(-1.0, min(1.0, composite))
            elif not np.isnan(pos_signal):
                composite = _NORM(pos_signal)
            else:
                composite = 0.0

            total_vol = sum(float(row.get(k, 0) or 0) for k in _KEYWORDS)
            keywords  = sorted(
                [(k, float(row.get(k, 0) or 0)) for k in _KEYWORDS],
                key=lambda x: x[1], reverse=True,
            )
            top_kw = [k for k, v in keywords[:3] if v > 0]

            bull_pct = max(0.0, min(1.0, (_NORM(btc_raw) + 1.0) / 2.0)) if not np.isnan(btc_raw) else 0.5
            bear_pct = max(0.0, min(1.0, (_NORM(neg_signal) + 1.0) / 2.0)) if not np.isnan(neg_signal) else 0.5
            neut_pct = round(max(0.0, 1.0 - bull_pct - bear_pct), 4)

            records.append({
                "timestamp_utc":       ts_ms,
                "platform":            "google_trends",
                "avg_sentiment_score": round(composite, 4),
                "mention_volume":      int(total_vol),
                "bullish_pct":         round(bull_pct, 4),
                "bearish_pct":         round(bear_pct, 4),
                "neutral_pct":         round(neut_pct, 4),
                "top_keywords":        top_kw,
                "is_synthetic":        False,
            })
        return records
