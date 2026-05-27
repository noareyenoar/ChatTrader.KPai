"""
TradFi Macro client  -  Layer 5 (TradFi Macro Liquidity)

Priority: Twelve Data API -> yfinance fallback -> synthetic
FRED:     direct CSV download (no API key needed)
Pandas 3.x: .ffill() everywhere instead of .fillna(method=...)
"""
from __future__ import annotations
import io, logging, os, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

log = logging.getLogger("time_machine.tradfi")
TWELVE_DATA_KEY: str = os.getenv("TWELVE_DATA_API_KEY", "")
_TWELVE_BASE = "https://api.twelvedata.com"
_TWELVE_TICKERS: dict[str, str] = {"sp500":"SPX","ndx":"NDX","dxy":"DXY","vix":"VIX","us10y":"TNX"}
_YF_TICKERS:     dict[str, str] = {"sp500":"^GSPC","ndx":"^NDX","dxy":"DX-Y.NYB","vix":"^VIX","us10y":"^TNX"}
_FRED_SERIES:    dict[str, str] = {"fed_funds":"FEDFUNDS","cpi":"CPIAUCSL","ppi":"PPIACO"}
_FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_SCHEMA_COLS = ["timestamp_utc","resolution","fed_funds_rate_bps","fed_rate_change_bps",
    "is_fomc_day","is_cpi_release_day","cpi_yoy_pct","ppi_yoy_pct",
    "dxy_close","dxy_change_pct","sp500_close","sp500_change_pct",
    "vix_close","nasdaq_close","nasdaq_change_pct","us_10y_yield","is_synthetic"]
_SES = requests.Session()
_SES.headers.update({"User-Agent": "TimeMachine/1.0"})

def _fetch_twelve_series(symbol: str, start: str, end: str) -> pd.Series:
    if not TWELVE_DATA_KEY:
        return pd.Series(dtype=float)
    params = {"symbol":symbol,"interval":"1day","outputsize":5000,"start_date":start,
              "end_date":end,"timezone":"UTC","format":"JSON","apikey":TWELVE_DATA_KEY}
    try:
        resp = _SES.get(f"{_TWELVE_BASE}/time_series", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("TwelveData %s failed: %s", symbol, exc)
        return pd.Series(dtype=float)
    if data.get("status") == "error":
        log.warning("TwelveData error %s: %s", symbol, data.get("message",""))
        return pd.Series(dtype=float)
    records: dict[int, float] = {}
    for row in data.get("values", []):
        try:
            dt = pd.Timestamp(row["datetime"], tz="UTC")
            records[int(dt.timestamp() * 1_000)] = float(row["close"])
        except (KeyError, ValueError):
            continue
    s = pd.Series(records).sort_index()
    log.info("TwelveData %s: %d rows", symbol, len(s))
    return s

def _fetch_yf_series(ticker: str, start: str, end: str) -> pd.Series:
    try:
        import yfinance as yf
        raw = yf.download(tickers=ticker, start=start, end=end, interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return pd.Series(dtype=float)
        close = raw["Close"].iloc[:, 0] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
        if close.index.tzinfo is None:
            close.index = close.index.tz_localize("UTC")
        else:
            close.index = close.index.tz_convert("UTC")
        close.index = close.index.normalize()
        ts_idx = ((close.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")
        return pd.Series(close.values, index=ts_idx, dtype=float).dropna()
    except Exception as exc:
        log.warning("yfinance %s failed: %s", ticker, exc)
        return pd.Series(dtype=float)

def _fetch_fred_csv(series_id: str) -> pd.Series:
    try:
        resp = _SES.get(f"{_FRED_CSV_BASE}?id={series_id}", timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
        s = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
        log.info("FRED %s: %d obs", series_id, len(s))
        return s
    except Exception as exc:
        log.warning("FRED CSV %s failed: %s", series_id, exc)
        return pd.Series(dtype=float)

class TradFiClient:
    def fetch_daily_layer(self, start: str, end: str) -> pd.DataFrame:
        equity = self._fetch_equities(start, end)
        macro  = self._fetch_fred_macro()
        return self._assemble(equity, macro)

    def slice_month(self, df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
        import calendar
        days = calendar.monthrange(year, month)[1]
        s = int(datetime(year, month,    1, tzinfo=timezone.utc).timestamp() * 1_000)
        e = int(datetime(year, month, days, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1_000)
        return df[(df["timestamp_utc"] >= s) & (df["timestamp_utc"] <= e)].copy()

    def _fetch_equities(self, start: str, end: str) -> pd.DataFrame:
        """
        Twelve Data free tier does NOT include equity indices (SPX/NDX/VIX).
        Use yfinance directly as the primary free source.
        """
        series: dict[str, pd.Series] = {}
        for name, yf_ticker in _YF_TICKERS.items():
            s = _fetch_yf_series(yf_ticker, start, end)
            if not s.empty:
                series[name] = s
            else:
                log.warning("No equity data for %s (%s)", name, yf_ticker)
        if not series:
            return pd.DataFrame()
        df = pd.DataFrame(series).sort_index()
        # Drop duplicate index entries (yfinance can occasionally return dupes)
        df = df[~df.index.duplicated(keep="first")]
        df["sp500_change_pct"]  = df["sp500"].pct_change() * 100 if "sp500" in df.columns else np.nan
        df["dxy_change_pct"]    = df["dxy"].pct_change()   * 100 if "dxy"   in df.columns else np.nan
        df["nasdaq_change_pct"] = df["ndx"].pct_change()   * 100 if "ndx"   in df.columns else np.nan
        # Daily spine + ffill weekends
        if len(df) > 0:
            start_dt = pd.Timestamp(df.index[0],  unit="ms", tz="UTC").normalize()
            end_dt   = pd.Timestamp(df.index[-1], unit="ms", tz="UTC").normalize()
            spine_idx = pd.date_range(start_dt, end_dt, freq="1D", tz="UTC")
            ts_spine  = ((spine_idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")
            df = df.reindex(ts_spine).ffill(limit=3)
        df["timestamp_utc"] = df.index.astype("int64")
        return df.reset_index(drop=True)

    def _fetch_fred_macro(self) -> dict[str, pd.Series]:
        result: dict[str, pd.Series] = {}
        for name, sid in _FRED_SERIES.items():
            s = _fetch_fred_csv(sid)
            if not s.empty:
                result[name] = s
            time.sleep(0.5)
        return result

    def _assemble(self, equity: pd.DataFrame, macro: dict[str, pd.Series]) -> pd.DataFrame:
        if equity.empty:
            return pd.DataFrame()
        df = equity.copy()
        dt_idx = pd.to_datetime(df["timestamp_utc"], unit="ms", utc=True).dt.normalize()

        def _align(series: pd.Series) -> pd.Series:
            series.index = series.index.normalize()
            return series.reindex(dt_idx).ffill()

        fed_s = macro.get("fed_funds")
        if fed_s is not None and not fed_s.empty:
            df["fed_funds_rate_bps"] = (_align(fed_s * 100).round().astype(float).values)
        else:
            df["fed_funds_rate_bps"] = np.nan

        df["fed_rate_change_bps"] = pd.Series(df["fed_funds_rate_bps"]).diff().fillna(0).astype(int).values
        df["is_fomc_day"]         = df["fed_rate_change_bps"] != 0
        df["is_cpi_release_day"]  = False

        for macro_col, raw_key in [("cpi_yoy_pct", "cpi"), ("ppi_yoy_pct", "ppi")]:
            s = macro.get(raw_key)
            if s is not None and len(s) > 12:
                yoy = ((s / s.shift(12)) - 1) * 100
                df[macro_col] = _align(yoy).values
            else:
                df[macro_col] = np.nan

        df = df.rename(columns={"sp500":"sp500_close","dxy":"dxy_close","vix":"vix_close",
                                 "ndx":"nasdaq_close","us10y":"us_10y_yield"})
        df["resolution"]   = "daily"
        df["is_synthetic"] = False
        for col in _SCHEMA_COLS:
            if col not in df.columns:
                df[col] = np.nan if col != "resolution" else "daily"
        return df[_SCHEMA_COLS].sort_values("timestamp_utc").reset_index(drop=True)
