"""Discretionary dataset builder — Phase 4.

Generates rasterized candlestick chart images from OHLCV data.
No external plotting library needed — encodes OHLCV as 4-channel
normalized tensors in [0,1] range.

Chart encoding (4 channels):
  Channel 0: Normalized open  (relative to bar range)
  Channel 1: Normalized high
  Channel 2: Normalized low
  Channel 3: Normalized close

Each image covers `img_bars` consecutive bars mapped onto a
(H=32, W=32) spatial grid — time on X-axis, price on Y-axis.

Tabular features (for Multimodal model): 5 momentum indicators.

Input shapes:
  image:   [Batch, 4, 32, 32]
  tabular: [Batch, 5]
Target:    3-class (0=down, 1=flat, 2=up)
Split:     Iron Wall 70/15/15
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import bisect

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_pipeline.config import PipelineConfig
from data_pipeline.features import FeatureFactory
from data_pipeline.quality_gate import DataQualityGate
from data_pipeline.splitter import IronWallSplitter


DISC_TAB_FEATURES = [
    "log_return",
    "zscore_close_64",
    "ema_spread",
    "atr_14",
    "price_slope_20",
]

IMG_H = 32
IMG_W = 32   # also = img_bars (1 column per bar)


def _rasterize_window(ohlcv_window: np.ndarray) -> np.ndarray:
    """Convert a (W, 4) OHLCV window into a (4, H, W) chart image tensor.

    ohlcv_window: shape (W, 4) — columns are [open, high, low, close]
    Returns: float32 array of shape (4, H, W) with values in [0, 1]
    """
    W, _ = ohlcv_window.shape
    H = IMG_H
    out = np.zeros((4, H, W), dtype=np.float32)

    # Normalize each bar independently into [0, 1] over bar's H-L range
    for col_idx in range(4):
        col = ohlcv_window[:, col_idx]  # (W,)
        lo = ohlcv_window[:, 2]         # low
        hi = ohlcv_window[:, 1]         # high
        rng = hi - lo + 1e-8
        normalized = (col - lo) / rng   # in [0,1] relative to each bar's range

        # Map to pixel rows: 0.0 → bottom row (H-1), 1.0 → top row (0)
        pixel_rows = np.clip(((1.0 - normalized) * (H - 1)).astype(int), 0, H - 1)
        for w in range(W):
            out[col_idx, pixel_rows[w], w] = 1.0

    return out


def _load_symbol(path: Path) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    df = pd.read_parquet(path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


@dataclass
class DiscDatasets:
    train: Dataset
    val: Dataset
    test: Dataset
    tab_input_dim: int


class RollingDiscDataset(Dataset):
    """Lazy discretionary dataset.

    Stores per-symbol OHLCV/tabular/target arrays and rasterizes chart images
    on demand in __getitem__, avoiding huge eager [N, 4, 32, 32] allocations.
    """

    def __init__(
        self,
        ohlcv_list: list[np.ndarray],
        tab_list: list[np.ndarray],
        target_list: list[np.ndarray],
        img_bars: int,
    ):
        self.ohlcv_list = [np.ascontiguousarray(x, dtype=np.float32) for x in ohlcv_list]
        self.tab_list = [np.ascontiguousarray(x, dtype=np.float32) for x in tab_list]
        self.target_list = [np.ascontiguousarray(x, dtype=np.int64) for x in target_list]
        self.img_bars = int(img_bars)

        self.lengths = [max(0, len(x) - self.img_bars) for x in self.ohlcv_list]
        self.cum = np.cumsum(self.lengths).tolist()

    def __len__(self) -> int:
        return int(self.cum[-1]) if self.cum else 0

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        s = bisect.bisect_right(self.cum, idx)
        prev = 0 if s == 0 else self.cum[s - 1]
        local_i = idx - prev
        end_i = local_i + self.img_bars

        ohlcv = self.ohlcv_list[s]
        tabs = self.tab_list[s]
        targets = self.target_list[s]

        window = ohlcv[end_i - self.img_bars : end_i]
        img = _rasterize_window(window)
        tab = tabs[end_i]
        y = targets[end_i]
        return torch.from_numpy(img), torch.from_numpy(tab), torch.tensor(y, dtype=torch.long)


def _log(message: str) -> None:
    print(message, flush=True)


def build_disc_datasets(config: dict[str, Any]) -> DiscDatasets:
    pipe_cfg = PipelineConfig(
        dataset_dir=Path(config["dataset_dir"]),
        manifest_path=Path(config["manifest_path"]),
        min_history_bars=int(config["min_history_bars"]),
        purge_gap_bars=int(config["purge_gap_bars"]),
    )
    accepted = [r for r in DataQualityGate(pipe_cfg).evaluate() if r.decision == "ACCEPT"]
    symbols = [r.symbol for r in accepted[:int(config["max_symbols"])]]
    if not symbols:
        raise RuntimeError("No accepted symbols for discretionary training")

    _log(f"[disc-data] accepted_symbols={len(accepted)} selected_symbols={len(symbols)} horizon={config['horizon']}")

    splitter = IronWallSplitter(purge_gap_bars=int(config["purge_gap_bars"]))
    img_bars = IMG_W
    horizon = int(config["horizon"])
    flat_threshold = float(config.get("flat_threshold", 0.003))
    cap_rows = int(config.get("max_rows_per_symbol", 12000))

    tr_ohlcv, tr_tabs, tr_y = [], [], []
    va_ohlcv, va_tabs, va_y = [], [], []
    te_ohlcv, te_tabs, te_y = [], [], []

    for symbol_idx, symbol in enumerate(symbols, start=1):
        _log(f"[disc-data] loading {symbol_idx}/{len(symbols)} symbol={symbol}")
        raw = _load_symbol(pipe_cfg.dataset_dir / f"{symbol}.parquet")
        trend_feat = FeatureFactory.build_trend_features(raw)

        ohlcv_cols = ["open", "high", "low", "close"]
        trend_feat = trend_feat[["timestamp", *ohlcv_cols, *DISC_TAB_FEATURES]].dropna()

        fwd_ret = trend_feat["close"].shift(-horizon) / trend_feat["close"] - 1.0
        trend_feat["target"] = 1
        trend_feat.loc[fwd_ret > flat_threshold, "target"] = 2
        trend_feat.loc[fwd_ret < -flat_threshold, "target"] = 0
        trend_feat = trend_feat.dropna().reset_index(drop=True)

        if cap_rows > 0:
            trend_feat = trend_feat.iloc[-cap_rows:].copy()

        split = splitter.split(trend_feat, time_col="timestamp")
        scaler = FeatureFactory.fit_scaler_train_only(split.train, DISC_TAB_FEATURES)

        for (df_, ohlcv_l, tab_l, y_l) in [
            (split.train, tr_ohlcv, tr_tabs, tr_y),
            (split.val, va_ohlcv, va_tabs, va_y),
            (split.test, te_ohlcv, te_tabs, te_y),
        ]:
            df_s = FeatureFactory.transform_with_scaler(df_.reset_index(drop=True), scaler)
            ohlcv = df_s[ohlcv_cols].to_numpy(np.float32)
            tabs = df_s[DISC_TAB_FEATURES].to_numpy(np.float32)
            targets = df_s["target"].to_numpy(np.int64)

            n = len(ohlcv)
            if n < img_bars:
                continue
            ohlcv_l.append(ohlcv)
            tab_l.append(tabs)
            y_l.append(targets)
        _log(f"[disc-data] ready symbol={symbol} rows={len(trend_feat)}")

    def _ds(ohlcv_l, tab_l, y_l):
        return RollingDiscDataset(ohlcv_l, tab_l, y_l, img_bars=img_bars)

    datasets = DiscDatasets(
        train=_ds(tr_ohlcv, tr_tabs, tr_y),
        val=_ds(va_ohlcv, va_tabs, va_y),
        test=_ds(te_ohlcv, te_tabs, te_y),
        tab_input_dim=len(DISC_TAB_FEATURES),
    )
    _log(f"[disc-data] datasets complete train={len(datasets.train)} val={len(datasets.val)} test={len(datasets.test)}")
    return datasets
