"""
On-Chain client  -  Layer 9 (On-Chain & Network Health)

Priority
  1. CoinGecko Demo API  : historical price, market_cap, volume for BTC + ETH
     Key: parse x_cg_demo_api_key from COINGECKO_API_URL env var (30 req/min)
  2. CoinMetrics Community API : on-chain metrics, free, no key, goes to genesis
     https://community-api.coinmetrics.io/v4/timeseries/asset-metrics
     metrics: AdrActCnt,TxCnt,HashRate,DiffMean,NVTAdj
  3. blockchain.com charts  : hash rate, difficulty fallback
"""
from __future__ import annotations
import asyncio, logging, os, urllib.parse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from ..base_client import AsyncRateLimitedClient

log = logging.getLogger("time_machine.onchain")

def _parse_cg_key() -> str:
    url = os.getenv("COINGECKO_API_URL", "")
    if not url:
        return ""
    params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    return params.get("x_cg_demo_api_key", [""])[0]

CG_KEY   = _parse_cg_key()
# Demo key requires pro-api.coingecko.com; without key fall back to public endpoint
_CG_BASE = "https://pro-api.coingecko.com/api/v3" if CG_KEY else "https://api.coingecko.com/api/v3"
_CM_BASE = "https://community-api.coinmetrics.io/v4"
_BC_BASE = "https://api.blockchain.info"
_CG_COINS = {"BTC": "bitcoin", "ETH": "ethereum"}
_CM_ASSETS = {"BTC": "btc", "ETH": "eth"}
_CM_METRICS = "AdrActCnt,TxCnt,HashRate,DiffMean,NVTAdj"
_SCHEMA_COLS = [
    "timestamp_utc","resolution",
    "btc_price_usd","btc_market_cap","btc_volume_24h",
    "eth_price_usd","eth_market_cap","eth_volume_24h",
    "btc_hash_rate","btc_difficulty","btc_block_count",
    "btc_active_addresses","btc_tx_count","btc_nvt",
    "eth_active_addresses","eth_tx_count",
    "is_synthetic"
]


def _fetch_yf_crypto_prices(coin_id: str, ts_from: int, ts_to: int) -> pd.DataFrame:
    """Fetch daily price/volume from yfinance. Used as CoinGecko fallback."""
    import yfinance as yf
    ticker_map = {"bitcoin": "BTC-USD", "ethereum": "ETH-USD"}
    yf_ticker = ticker_map.get(coin_id, f"{coin_id.upper()}-USD")
    prefix = "btc" if coin_id == "bitcoin" else "eth"
    try:
        start = pd.Timestamp(ts_from, unit="s", tz="UTC").strftime("%Y-%m-%d")
        end   = (pd.Timestamp(ts_to,   unit="s", tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw   = yf.download(yf_ticker, start=start, end=end, interval="1d",
                            progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        close  = raw["Close"].iloc[:,  0] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
        volume = raw["Volume"].iloc[:, 0] if isinstance(raw.columns, pd.MultiIndex) else raw["Volume"]
        if close.index.tzinfo is None:
            close.index  = close.index.tz_localize("UTC")
            volume.index = volume.index.tz_localize("UTC")
        close.index  = close.index.normalize()
        volume.index = volume.index.normalize()
        ts_idx = ((close.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")
        df = pd.DataFrame({
            "timestamp_utc":       ts_idx,
            f"{prefix}_price_usd": close.values.astype(float),
            f"{prefix}_market_cap": np.nan,
            f"{prefix}_volume_24h": volume.values.astype(float),
        })
        log.info("yfinance %s (%s): %d rows", coin_id, yf_ticker, len(df))
        return df.dropna(subset=[f"{prefix}_price_usd"])
    except Exception as exc:
        log.warning("yfinance crypto %s failed: %s", coin_id, exc)
        return pd.DataFrame()


class OnChainClient:
    def __init__(self) -> None:
        self._http_cg = AsyncRateLimitedClient(rps=0.45, max_retries=4, backoff_base=2.0, timeout_s=60)
        self._http_cm = AsyncRateLimitedClient(rps=2.0,  max_retries=4, backoff_base=1.5, timeout_s=60)
        self._http_bc = AsyncRateLimitedClient(rps=0.5,  max_retries=3, backoff_base=2.0, timeout_s=60)

    async def build_layer(self, year: int, month: int) -> pd.DataFrame:
        import calendar
        days   = calendar.monthrange(year, month)[1]
        ts_s   = int(datetime(year, month,    1, tzinfo=timezone.utc).timestamp())
        ts_e   = int(datetime(year, month, days, 23, 59, 59, tzinfo=timezone.utc).timestamp())
        ts_s_ms = ts_s * 1_000
        ts_e_ms = ts_e * 1_000

        spine = pd.DataFrame({"timestamp_utc": ((pd.date_range(
            start=pd.Timestamp(year=year, month=month, day=1,  tz="UTC"),
            end=  pd.Timestamp(year=year, month=month, day=days, tz="UTC"),
            freq="1D", tz="UTC",
        ) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype(np.int64)})
        spine["resolution"] = "daily"

        # CoinGecko Demo key returns 401; use yfinance for BTC/ETH price/volume instead
        loop = asyncio.get_event_loop()
        try:
            cg_btc = await loop.run_in_executor(None, _fetch_yf_crypto_prices, "bitcoin",  ts_s, ts_e)
            cg_eth = await loop.run_in_executor(None, _fetch_yf_crypto_prices, "ethereum", ts_s, ts_e)
        except Exception as exc:
            log.warning("yfinance crypto prices %d-%02d failed: %s", year, month, exc)
            cg_btc = cg_eth = pd.DataFrame()
        try:
            async with self._http_cm:
                cm_btc = await self._fetch_coinmetrics("btc", year, month)
                cm_eth = await self._fetch_coinmetrics("eth", year, month)
        except Exception as exc:
            log.warning("CoinMetrics %d-%02d failed: %s", year, month, exc)
            cm_btc = cm_eth = pd.DataFrame()

        def _merge_daily(spine: pd.DataFrame, other: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
            if other.empty:
                return spine
            other = other.copy()
            # Compute day-aligned ms timestamp, replace original sub-daily timestamp to avoid duplicates
            day_ts = ((pd.to_datetime(other["timestamp_utc"], unit="ms", utc=True)
                       .dt.normalize() - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")
            other = other.drop(columns=["timestamp_utc"])
            other.insert(0, "timestamp_utc", day_ts.values)
            other_grp = other.groupby("timestamp_utc").last().reset_index()
            return spine.merge(other_grp, on="timestamp_utc", how="left")

        spine = _merge_daily(spine, cg_btc)
        spine = _merge_daily(spine, cg_eth)
        spine = _merge_daily(spine, cm_btc)
        spine = _merge_daily(spine, cm_eth)

        # Forward-fill sparse on-chain metrics over gaps
        for col in ["btc_hash_rate","btc_difficulty","btc_active_addresses","btc_tx_count","btc_nvt",
                    "eth_active_addresses","eth_tx_count"]:
            if col in spine.columns:
                spine[col] = spine[col].ffill()

        spine["btc_block_count"] = np.nan

        for col in _SCHEMA_COLS:
            if col not in spine.columns:
                spine[col] = float("nan") if col != "resolution" else "daily"
        # is_synthetic: True if no real on-chain metrics available (handled after schema fill)
        spine["is_synthetic"]    = spine[["btc_hash_rate","btc_active_addresses"]].isnull().all(axis=1)
        return spine[_SCHEMA_COLS].sort_values("timestamp_utc").reset_index(drop=True)

    # ── CoinGecko ─────────────────────────────────────────────────────────────
    async def _fetch_cg_history(self, coin_id: str, ts_from: int, ts_to: int) -> pd.DataFrame:
        """Daily price / market_cap / volume from CoinGecko market_chart/range."""
        params  = {"vs_currency":"usd","from":str(ts_from),"to":str(ts_to)}
        headers = {"x-cg-demo-api-key": CG_KEY} if CG_KEY else {}
        try:
            data = await self._http_cg.get(f"{_CG_BASE}/coins/{coin_id}/market_chart/range",
                                            params=params, headers=headers)
        except Exception as exc:
            log.warning("CoinGecko HTTP error %s: %s", coin_id, exc)
            return pd.DataFrame()
        if not data:
            return pd.DataFrame()
        try:
            prices  = pd.DataFrame(data.get("prices",       []), columns=["timestamp_utc","price"])
            mcaps   = pd.DataFrame(data.get("market_caps",  []), columns=["timestamp_utc","market_cap"])
            volumes = pd.DataFrame(data.get("total_volumes",[]), columns=["timestamp_utc","volume_24h"])
            df = prices.merge(mcaps, on="timestamp_utc", how="outer").merge(volumes, on="timestamp_utc", how="outer")
            df["timestamp_utc"] = df["timestamp_utc"].astype("int64")
            prefix = "btc" if coin_id == "bitcoin" else "eth"
            df = df.rename(columns={
                "price":     f"{prefix}_price_usd",
                "market_cap":f"{prefix}_market_cap",
                "volume_24h":f"{prefix}_volume_24h",
            })
            log.info("CoinGecko %s: %d rows", coin_id, len(df))
            return df
        except Exception as exc:
            log.warning("CoinGecko parse error %s: %s", coin_id, exc)
            return pd.DataFrame()

    # ── CoinMetrics Community ─────────────────────────────────────────────────
    async def _fetch_coinmetrics(self, asset: str, year: int, month: int) -> pd.DataFrame:
        import calendar
        days = calendar.monthrange(year, month)[1]
        start_str = f"{year}-{month:02d}-01"
        end_str   = f"{year}-{month:02d}-{days:02d}"
        params = {
            "assets":    asset,
            "metrics":   _CM_METRICS,
            "frequency": "1d",
            "start_time": start_str,
            "end_time":   end_str,
            "page_size": 10000,
        }
        data = await self._http_cm.get(f"{_CM_BASE}/timeseries/asset-metrics", params=params)
        if not data or "data" not in data:
            return pd.DataFrame()
        try:
            rows = data["data"]
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["timestamp_utc"] = ((pd.to_datetime(df["time"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")
            prefix = "btc" if asset == "btc" else "eth"
            col_map = {
                "AdrActCnt": f"{prefix}_active_addresses",
                "TxCnt":     f"{prefix}_tx_count",
            }
            if asset == "btc":
                col_map.update({"HashRate":"btc_hash_rate","DiffMean":"btc_difficulty","NVTAdj":"btc_nvt"})
            df = df.rename(columns=col_map)
            for c in col_map.values():
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            keep = ["timestamp_utc"] + [c for c in col_map.values() if c in df.columns]
            df = df[keep]
            log.info("CoinMetrics %s %d-%02d: %d rows", asset, year, month, len(df))
            return df
        except Exception as exc:
            log.warning("CoinMetrics parse %s: %s", asset, exc)
            return pd.DataFrame()
