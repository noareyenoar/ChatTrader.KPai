"""
data_feeder.py
Reads Binance aggTrade Parquet files, resamples to OHLCV bars, and computes
technical features needed by the debate engine.

Supported timeframes: 1m, 3m, 5m, 15m, 1h, 4h
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TIMEFRAME_MAP = {
    "1m": "1min", "3m": "3min", "5m": "5min",
    "15m": "15min", "1h": "1h", "4h": "4h",
}


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _zscore(series: pd.Series, window: int = 64) -> pd.Series:
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    return (series - roll_mean) / roll_std.replace(0, np.nan)


class DataFeeder:
    """
    Converts raw aggTrade Parquet data into a stream of market_context dicts
    ready for the DebateEngine.

    Usage:
        feeder = DataFeeder("Dataset/.../20251010.parquet", timeframe="5m")
        for bar_idx, context in feeder.iterate():
            result = engine.run_debate(symbol, timeframe, context)
    """

    def __init__(
        self,
        parquet_path: str,
        timeframe: str = "5m",
        symbol: str = "BTCUSDT",
        warmup_bars: int = 64,
        max_bars: Optional[int] = None,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.timeframe = timeframe
        self.symbol = symbol
        self.warmup_bars = warmup_bars
        self.max_bars = max_bars

        if timeframe not in _TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from: {list(_TIMEFRAME_MAP)}")

        logger.info("DataFeeder: loading %s ...", self.parquet_path.name)
        self._bars = self._load_and_compute()
        logger.info(
            "DataFeeder: %d bars computed for %s @ %s (warmup=%d, usable=%d)",
            len(self._bars), symbol, timeframe, warmup_bars,
            max(0, len(self._bars) - warmup_bars),
        )

    # ─────────────────────────────────────────────
    # Loading & feature computation
    # ─────────────────────────────────────────────
    def _load_and_compute(self) -> pd.DataFrame:
        raw = pd.read_parquet(self.parquet_path)

        # Parse timestamps (microseconds)
        raw["dt"] = pd.to_datetime(raw["transact_time"], unit="us", utc=True)
        raw = raw.set_index("dt").sort_index()

        freq = _TIMEFRAME_MAP[self.timeframe]

        # OHLCV
        ohlcv = raw["price"].resample(freq).ohlc()
        ohlcv.columns = ["open", "high", "low", "close"]
        ohlcv["volume"] = raw["quantity"].resample(freq).sum()

        # Order Flow Imbalance (OFI) = (buy_vol - sell_vol) / total_vol
        buy_vol = raw.loc[~raw["is_buyer_maker"], "quantity"].resample(freq).sum()
        sell_vol = raw.loc[raw["is_buyer_maker"], "quantity"].resample(freq).sum()
        total_vol = buy_vol + sell_vol
        ohlcv["ofi"] = (buy_vol - sell_vol) / total_vol.replace(0, np.nan)

        ohlcv = ohlcv.dropna(subset=["close"])

        c = ohlcv["close"]
        h = ohlcv["high"]
        lo = ohlcv["low"]

        # EMA spread (EMA9 vs EMA21)
        ema9 = _ema(c, 9)
        ema21 = _ema(c, 21)
        ohlcv["ema_spread"] = (ema9 - ema21) / c

        # Price slope (linear regression slope over 20 bars, normalized)
        ohlcv["price_slope_20"] = (
            c.rolling(20).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x[-1],
                raw=True,
            )
        )

        # Z-score of close over 64 bars
        ohlcv["zscore_close_64"] = _zscore(c, 64)

        # ATR (14)
        ohlcv["atr_14"] = _atr(h, lo, c, 14) / c
        ohlcv["atr_mean"] = ohlcv["atr_14"].rolling(50).mean()

        # Bollinger band width (20, 2σ)
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        ohlcv["bb_width"] = (2 * bb_std * 2) / bb_mid
        ohlcv["bb_distance"] = (c - bb_mid) / (bb_std.replace(0, np.nan))

        # RSI
        ohlcv["rsi_14"] = _rsi(c, 14)

        # Spread proxy (high-low as fraction of close)
        ohlcv["spread_proxy"] = (h - lo) / c

        # Z-score of spread_proxy (for stat arb)
        ohlcv["spread_z_64"] = _zscore(ohlcv["spread_proxy"], 64)

        # Vol regime code: 0=low, 1=normal, 2=high
        atr_z = _zscore(ohlcv["atr_14"], 50)
        ohlcv["vol_regime_code"] = pd.cut(
            atr_z.fillna(0),
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=[0, 1, 2],
        ).astype(float)

        # Fractal diff proxy (close - close.shift(5) normalized)
        ohlcv["fracdiff_close_d04"] = c.diff(5) / c.shift(5)

        # Pattern score (momentum: 3-bar return vs 20-bar return)
        ohlcv["pattern_score"] = (c.pct_change(3) - c.pct_change(20)).fillna(0)

        # Pair correlation (BTC vs its own lagged self as proxy — placeholder)
        ohlcv["pair_correlation"] = c.rolling(30).corr(c.shift(1)).fillna(0.5)

        # Inventory level proxy (cumulative OFI normalized to [-1, 1])
        cum_ofi = ohlcv["ofi"].fillna(0).cumsum()
        max_abs = cum_ofi.abs().rolling(50).max().replace(0, 1)
        ohlcv["inventory_level"] = (cum_ofi / max_abs).clip(-1, 1)

        # Forward close for PnL calculation (used by backtest)
        ohlcv["close_next"] = ohlcv["close"].shift(-1)

        return ohlcv

    # ─────────────────────────────────────────────
    # Iterator
    # ─────────────────────────────────────────────
    def iterate(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        Yields (bar_index, market_context) for each usable bar (after warmup).
        market_context contains all features needed by the DebateEngine.
        """
        bars = self._bars
        n = len(bars)
        limit = (self.warmup_bars + self.max_bars) if self.max_bars else n

        for i in range(self.warmup_bars, min(n, limit)):
            row = bars.iloc[i]
            if row.isnull().any():
                continue  # Skip bars with NaN features (early warmup)

            features = {
                "price_slope_20":       float(row["price_slope_20"]),
                "zscore_close_64":      float(row["zscore_close_64"]),
                "atr_14":               float(row["atr_14"]),
                "atr_mean":             float(row["atr_mean"] or row["atr_14"]),
                "ema_spread":           float(row["ema_spread"]),
                "bb_distance":          float(row["bb_distance"]),
                "bb_width":             float(row["bb_width"]),
                "rsi_14":               float(row["rsi_14"]),
                "ofi_proxy":            float(row["ofi"] if not np.isnan(row["ofi"]) else 0.0),
                "spread_proxy":         float(row["spread_proxy"]),
                "vol_regime_code":      float(row["vol_regime_code"] if not np.isnan(row["vol_regime_code"]) else 1.0),
                "spread_z_64":          float(row["spread_z_64"]),
                "fracdiff_close_d04":   float(row["fracdiff_close_d04"]),
                "pair_correlation":     float(row["pair_correlation"]),
                "inventory_level":      float(row["inventory_level"]),
                "pattern_score":        float(row["pattern_score"]),
            }

            price_summary = (
                f"BTCUSDT {self.timeframe} | "
                f"O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f} | "
                f"Vol={row['volume']:.3f} OFI={features['ofi_proxy']:+.3f} | "
                f"EMA_spread={features['ema_spread']:+.5f} slope={features['price_slope_20']:+.6f} | "
                f"zscore={features['zscore_close_64']:+.2f} RSI={features['rsi_14']:.1f} | "
                f"ATR={features['atr_14']:.5f} BB_width={features['bb_width']:.4f}"
            )

            context: Dict[str, Any] = {
                "features": features,
                "price_summary": price_summary,
                "bar_index": i,
                "bar_time": str(bars.index[i]),
                "close_price": float(row["close"]),
                "close_next": float(row["close_next"]) if not np.isnan(row["close_next"]) else None,
            }

            yield i, context

    def get_bar(self, index: int) -> Optional[pd.Series]:
        """Return raw bar by index (for post-trade PnL calculation)."""
        if 0 <= index < len(self._bars):
            return self._bars.iloc[index]
        return None

    @property
    def total_bars(self) -> int:
        return len(self._bars)

    @property
    def usable_bars(self) -> int:
        return max(0, len(self._bars) - self.warmup_bars)
