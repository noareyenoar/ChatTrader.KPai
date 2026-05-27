"""
Binance Derivatives client  -  Layer 6 (Market Derivatives Microstructure)

Data sources (Binance Data Vision  -  public, no API key needed)
  fundingRate ZIPs : https://data.binance.vision/data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{Y}-{M}.zip
  klines 1h ZIPs   : https://data.binance.vision/data/futures/um/monthly/klines/{SYM}/1h/{SYM}-1h-{Y}-{M}.zip
                     Available from 2019-09 (Binance perp launch).
                     ~126 ZIPs x 3MB each = ~378 MB for BTCUSDT+ETHUSDT 2019-2024.
"""
from __future__ import annotations
import io, logging, zipfile
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd
from ..base_client import AsyncRateLimitedClient

log = logging.getLogger("time_machine.derivatives")

_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
_BDV = "https://data.binance.vision/data/futures/um/monthly"
_FUNDING_COLS  = ["symbol","timestamp_utc","funding_rate","mark_price","is_synthetic"]
_KLINE_COLS_IN = ["open_time","open","high","low","close","volume","close_time",
                  "quote_volume","count","taker_buy_volume","taker_buy_quote_volume","ignore"]
_SCHEMA_COLS = ["timestamp_utc","resolution","symbol",
                "funding_rate","open_interest","long_short_ratio",
                "liquidations_usd","oi_change_pct",
                "open","high","low","close","volume","quote_volume","num_trades",
                "is_synthetic"]


class DerivativesClient:
    def __init__(self) -> None:
        self._http = AsyncRateLimitedClient(rps=3.0, max_retries=4, backoff_base=1.5, timeout_s=120)

    async def build_month(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        async with self._http:
            funding = await self._fetch_funding(symbol, year, month)
            klines  = await self._fetch_klines(symbol, year, month)

        if funding.empty and klines.empty:
            log.info("No data for %s %d-%02d", symbol, year, month)
            return pd.DataFrame()   # no synthetic fallback for missing months

        if funding.empty and not klines.empty:
            merged = klines.copy()
            merged["funding_rate"] = np.nan
        elif klines.empty and not funding.empty:
            merged = funding.copy()
            for c in ["open","high","low","close","volume","quote_volume","num_trades"]:
                merged[c] = np.nan
        else:
            # Merge on hourly timestamp  (funding repeats at 8-hour intervals - ffill)
            klines = klines.set_index("timestamp_utc")
            funding_hr = (
                funding.set_index("timestamp_utc")["funding_rate"]
                .reindex(klines.index).ffill()
            )
            klines["funding_rate"] = funding_hr.values
            merged = klines.reset_index()

        merged["symbol"]           = symbol
        merged["resolution"]       = "1h"
        merged["open_interest"]    = np.nan
        merged["long_short_ratio"] = np.nan
        merged["liquidations_usd"] = np.nan
        merged["oi_change_pct"]    = np.nan
        merged["mark_price"]       = np.nan
        merged["is_synthetic"]     = False

        for col in _SCHEMA_COLS:
            if col not in merged.columns:
                merged[col] = np.nan
        return merged[_SCHEMA_COLS].sort_values("timestamp_utc").reset_index(drop=True)

    async def _fetch_funding(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        url = f"{_BDV}/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
        raw = await self._http.get(url, as_bytes=True)
        if not raw:
            log.debug("fundingRate 404/empty: %s %d-%02d", symbol, year, month)
            return pd.DataFrame()
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                name = zf.namelist()[0]
                df = pd.read_csv(zf.open(name), header=0,
                                 names=["calc_time","funding_interval_hours","last_funding_rate"])
            df["timestamp_utc"] = pd.to_numeric(df["calc_time"], errors="coerce").astype("int64")
            df["funding_rate"]  = pd.to_numeric(df["last_funding_rate"], errors="coerce")
            df = df[["timestamp_utc","funding_rate"]].dropna()
            log.info("fundingRate %s %d-%02d: %d rows", symbol, year, month, len(df))
            return df
        except Exception as exc:
            log.warning("fundingRate parse error %s %d-%02d: %s", symbol, year, month, exc)
            return pd.DataFrame()

    async def _fetch_klines(self, symbol: str, year: int, month: int) -> pd.DataFrame:
        # Binance perpetual futures launched 2019-09
        if (year, month) < (2019, 9):
            return pd.DataFrame()
        url = f"{_BDV}/klines/{symbol}/1h/{symbol}-1h-{year}-{month:02d}.zip"
        raw = await self._http.get(url, as_bytes=True)
        if not raw:
            log.debug("klines 404/empty: %s %d-%02d", symbol, year, month)
            return pd.DataFrame()
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                name = zf.namelist()[0]
                df = pd.read_csv(zf.open(name), header=None, names=_KLINE_COLS_IN)
            # 2024+ Binance kline files include a header row — drop any non-numeric open_time rows
            df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()].copy()
            df["timestamp_utc"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["quote_volume"] = pd.to_numeric(df["quote_volume"], errors="coerce")
            df["num_trades"]   = pd.to_numeric(df["count"],        errors="coerce").replace([np.inf, -np.inf], np.nan)
            df = df[["timestamp_utc","open","high","low","close","volume","quote_volume","num_trades"]].dropna(subset=["timestamp_utc"])
            log.info("klines %s %d-%02d: %d rows", symbol, year, month, len(df))
            return df
        except Exception as exc:
            log.warning("klines parse error %s %d-%02d: %s", symbol, year, month, exc)
            return pd.DataFrame()

    async def build_all_symbols(self, symbols: list[str], year: int, month: int) -> pd.DataFrame:
        # Run sequentially: AsyncRateLimitedClient is NOT re-entrant (shares self._session)
        frames = []
        for sym in symbols:
            f = await self.build_month(sym, year, month)
            frames.append(f)
        valid = [f for f in frames if not f.empty]
        return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()
