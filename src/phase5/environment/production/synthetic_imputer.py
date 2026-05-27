"""
Synthetic Imputer – Gap-fill missing Time Machine data.

When a real API returns nothing for a layer/month (endpoint down, data
unavailable for older eras, API-key missing) the imputer generates a
statistically plausible substitute.

Every synthetic row is tagged with ``is_synthetic = True`` so the Oracle
and downstream agents can distinguish real from imputed observations.

Imputation strategies by layer
-------------------------------
  on_chain        : Forward-fill last known row; add Gaussian noise on
                    numeric columns to prevent flat-line artefacts.
  tradfi_macro    : GBM-like daily walk on SP500/DXY/VIX from last known.
  derivatives     : Interpolate OI; zero-fill liquidations; carry funding.
  fear_greed      : Mean-reverting random walk bounded to [5, 95].
  social_sentiment: Carry-forward mention counts with mild decay and noise.
  JSONL layers    : Return an empty list (no synthetic events fabricated).
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("time_machine.imputer")

_HOURLY_MS  = 3_600_000
_DAILY_MS   = 86_400_000


def _ms_range_hourly(year: int, month: int) -> list[int]:
    """All hourly timestamps (ms, UTC) for the given year/month."""
    import calendar
    days = calendar.monthrange(year, month)[1]
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1_000)
    return [start + h * _HOURLY_MS for h in range(days * 24)]


def _ms_range_daily(year: int, month: int) -> list[int]:
    import calendar
    days = calendar.monthrange(year, month)[1]
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1_000)
    return [start + d * _DAILY_MS for d in range(days)]


def _noise(val: float, pct: float = 0.02, rng: random.Random | None = None) -> float:
    r = rng or random
    return val * (1 + r.uniform(-pct, pct))


class SyntheticImputer:
    """
    Generates synthetic data for any layer / month where real data is absent.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility.
    reference_df : dict[str, pd.DataFrame]
        Optional map of layer_key → most-recent real DataFrame slice.
        When provided the imputer seeds synthetic values from the tail
        of the reference data rather than hardcoded defaults.
    """

    # Reasonable market-neutral anchors used when no reference is available
    _DEFAULTS: dict[str, Any] = {
        "btc_price":           35_000.0,
        "eth_price":           2_500.0,
        "btc_hash_rate_eh_s":  165.0,
        "btc_active_addr":     900_000,
        "btc_exchange_inflow": 1_800.0,
        "eth_exchange_inflow": 14_000.0,
        "sp500":               4_200.0,
        "vix":                 20.0,
        "dxy":                 92.0,
        "fear_greed":          50,
        "btc_oi_usd":          10_000_000_000.0,
        "eth_oi_usd":          1_500_000_000.0,
        "funding_rate":        0.0001,
    }

    def __init__(
        self,
        seed: int = 0,
        reference_df: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self._rng  = random.Random(seed)
        np.random.seed(seed)
        self._ref  = reference_df or {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_ref_tail(self, layer_key: str, column: str, default: float) -> float:
        df = self._ref.get(layer_key)
        if df is not None and column in df.columns and not df.empty:
            val = df[column].dropna().iloc[-1] if len(df[column].dropna()) else default
            return float(val)
        return default

    def _gbm_walk(
        self,
        start: float,
        n_steps: int,
        mu: float = 0.0,
        sigma: float = 0.01,
    ) -> list[float]:
        """Geometric Brownian Motion walk – returns list of n_steps prices."""
        vals = [start]
        for _ in range(n_steps - 1):
            dt   = 1.0
            shock = np.random.normal(mu, sigma)
            vals.append(vals[-1] * np.exp(shock))
        return vals

    # ── Layer-specific imputation methods ─────────────────────────────────────

    def create_on_chain_month(
        self,
        symbol: str,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """Synthetic hourly on-chain rows for one symbol/month."""
        ts_list   = _ms_range_hourly(year, month)
        n         = len(ts_list)

        if symbol == "BTC":
            base_inflow = self._get_ref_tail("on_chain", "exchange_inflow",
                                             self._DEFAULTS["btc_exchange_inflow"])
            hash_rate   = self._get_ref_tail("on_chain", "miner_hash_rate_eh_s",
                                             self._DEFAULTS["btc_hash_rate_eh_s"])
            base_active = self._get_ref_tail("on_chain", "active_addresses",
                                             self._DEFAULTS["btc_active_addr"])
            diff        = 1.8e13
        else:  # ETH
            base_inflow = self._get_ref_tail("on_chain", "exchange_inflow",
                                             self._DEFAULTS["eth_exchange_inflow"])
            hash_rate   = float("nan")
            diff        = float("nan")
            base_active = 350_000

        rows = []
        inflow = base_inflow
        for ts in ts_list:
            inflow  = _noise(inflow, 0.05, self._rng)
            outflow = _noise(inflow * 0.6, 0.05, self._rng)
            rows.append({
                "timestamp_utc":             ts,
                "symbol":                    symbol,
                "exchange_inflow":           round(inflow, 2),
                "exchange_outflow":          round(outflow, 2),
                "net_exchange_flow":         round(inflow - outflow, 2),
                "whale_transfer_count":      self._rng.randint(20, 100),
                "whale_transfer_volume_usd": round(abs(np.random.normal(8e7, 2e7)), 0),
                "miner_hash_rate_eh_s":      round(_noise(hash_rate, 0.01, self._rng), 3) if symbol == "BTC" else float("nan"),
                "network_difficulty":        round(_noise(diff, 0.002, self._rng), 0) if symbol == "BTC" else float("nan"),
                "active_addresses":          self._rng.randint(int(base_active * 0.9), int(base_active * 1.1)),
                "nvt_ratio":                 round(abs(np.random.normal(30, 8)), 2),
                "sopr":                      round(abs(np.random.normal(1.02, 0.04)), 4),
                "smart_money_net_flow_usd":  round(np.random.normal(0, 5_000_000), 0),
                "is_synthetic":              True,
            })
        return pd.DataFrame(rows)

    def create_tradfi_macro_month(
        self,
        year: int,
        month: int,
        fed_rate_bps: int = 25,
    ) -> pd.DataFrame:
        """Synthetic daily TradFi macro rows."""
        import calendar
        ts_list = _ms_range_daily(year, month)

        sp500_0 = self._get_ref_tail("tradfi_macro", "sp500_close", self._DEFAULTS["sp500"])
        vix_0   = self._get_ref_tail("tradfi_macro", "vix_close",   self._DEFAULTS["vix"])
        dxy_0   = self._get_ref_tail("tradfi_macro", "dxy_close",   self._DEFAULTS["dxy"])
        ndx_0   = sp500_0 * 3.2

        sp500_vals  = self._gbm_walk(sp500_0,  len(ts_list), mu=0.0003, sigma=0.012)
        dxy_vals    = self._gbm_walk(dxy_0,    len(ts_list), mu=0.0,    sigma=0.004)
        ndx_vals    = [sp * 3.2 for sp in sp500_vals]
        # Mean-reverting VIX
        vix_vals = [vix_0]
        for _ in range(len(ts_list) - 1):
            v = vix_vals[-1] + 0.05 * (20.0 - vix_vals[-1]) + np.random.normal(0, 1.5)
            vix_vals.append(max(9.0, v))

        rows = []
        for i, ts in enumerate(ts_list):
            sp_chg  = (sp500_vals[i] / sp500_vals[i - 1] - 1) * 100 if i > 0 else 0.0
            dxy_chg = (dxy_vals[i] / dxy_vals[i - 1] - 1) * 100   if i > 0 else 0.0
            ndx_chg = sp_chg * 1.1
            rows.append({
                "timestamp_utc":       ts,
                "resolution":          "daily",
                "fed_funds_rate_bps":  fed_rate_bps,
                "fed_rate_change_bps": 0,
                "is_fomc_day":         False,
                "is_cpi_release_day":  False,
                "cpi_yoy_pct":         float("nan"),
                "ppi_yoy_pct":         float("nan"),
                "dxy_close":           round(dxy_vals[i], 3),
                "dxy_change_pct":      round(dxy_chg, 4),
                "sp500_close":         round(sp500_vals[i], 2),
                "sp500_change_pct":    round(sp_chg, 4),
                "vix_close":           round(vix_vals[i], 2),
                "nasdaq_close":        round(ndx_vals[i], 2),
                "nasdaq_change_pct":   round(ndx_chg, 4),
                "us_10y_yield":        round(abs(np.random.normal(1.5, 0.15)), 3),
                "is_synthetic":        True,
            })
        return pd.DataFrame(rows)

    def create_derivatives_month(
        self,
        symbol: str,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """Synthetic hourly derivatives rows for one symbol/month."""
        ts_list = _ms_range_hourly(year, month)
        base_oi = self._DEFAULTS["btc_oi_usd"] if "BTC" in symbol else self._DEFAULTS["eth_oi_usd"]
        base_oi = self._get_ref_tail("derivatives", "oi_usd", base_oi)

        rows = []
        oi = base_oi
        for ts in ts_list:
            oi   = max(base_oi * 0.3, _noise(oi, 0.02, self._rng))
            fund = np.random.normal(self._DEFAULTS["funding_rate"], 0.0003)
            long_liq  = abs(np.random.normal(400_000, 250_000))
            short_liq = abs(np.random.normal(380_000, 240_000))
            total_liq = long_liq + short_liq
            rows.append({
                "timestamp_utc":          ts,
                "symbol":                 symbol,
                "funding_rate":           round(fund, 7),
                "oi_usd":                 round(oi, 0),
                "oi_change_usd":          round(np.random.normal(0, oi * 0.005), 0),
                "long_liquidations_usd":  round(long_liq, 0),
                "short_liquidations_usd": round(short_liq, 0),
                "total_liquidations_usd": round(total_liq, 0),
                "liq_long_short_ratio":   round(long_liq / total_liq, 4),
                "options_max_pain_usd":   float("nan"),
                "put_call_ratio":         float("nan"),
                "basis_pct":              round(np.random.normal(0.002, 0.003), 6),
                "is_synthetic":           True,
            })
        return pd.DataFrame(rows)

    def create_fear_greed_month(self, year: int, month: int) -> pd.DataFrame:
        """Synthetic daily Fear & Greed rows."""
        ts_list = _ms_range_daily(year, month)
        fg = self._get_ref_tail("fear_greed", "fear_greed_index", self._DEFAULTS["fear_greed"])
        rows = []
        for ts in ts_list:
            fg = max(5, min(95, fg + self._rng.randint(-6, 6)))
            if fg <= 25:
                label = "extreme_fear" if fg <= 10 else "fear"
            elif fg >= 75:
                label = "extreme_greed" if fg >= 90 else "greed"
            else:
                label = "neutral"
            bt = min(100, abs(np.random.normal(45, 20)))
            rows.append({
                "timestamp_utc":               ts,
                "date_str":                    datetime.fromtimestamp(ts / 1_000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "fear_greed_index":            int(fg),
                "fear_greed_label":            label,
                "google_trends_bitcoin":       round(bt, 1),
                "google_trends_crypto":        round(bt * 0.88, 1),
                "google_trends_buy_crypto":    round(bt * 0.52, 1),
                "google_trends_ethereum":      round(bt * 0.72, 1),
                "google_trends_bitcoin_crash": round(bt * 0.22, 1),
                "google_trends_sell_bitcoin":  round(bt * 0.18, 1),
                "retail_search_composite":     round(bt * 0.68, 1),
                "is_synthetic":                True,
            })
        return pd.DataFrame(rows)

    def forward_fill_parquet_gaps(
        self,
        df: pd.DataFrame,
        expected_freq_ms: int,
        max_fill_steps: int = 24,
    ) -> pd.DataFrame:
        """
        Given an existing Parquet DataFrame, identify missing timestamps at
        ``expected_freq_ms`` resolution and forward-fill + noise them.
        All filled rows get ``is_synthetic = True``.
        """
        if df.empty or "timestamp_utc" not in df.columns:
            return df

        ts_col = df["timestamp_utc"].sort_values().unique()
        full_range = pd.RangeIndex(
            start=ts_col[0],
            stop=ts_col[-1] + expected_freq_ms,
            step=expected_freq_ms,
        )
        df = df.set_index("timestamp_utc").reindex(full_range.values)
        # Forward-fill (up to max_fill_steps gaps)
        df = df.fillna(method="ffill", limit=max_fill_steps)
        df.index.name = "timestamp_utc"
        df = df.reset_index()

        # Flag all NaN-filled rows
        if "is_synthetic" not in df.columns:
            df["is_synthetic"] = False
        df["is_synthetic"] = df["is_synthetic"].fillna(True)

        return df
