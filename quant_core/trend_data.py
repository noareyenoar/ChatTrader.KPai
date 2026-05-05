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


FEATURE_COLUMNS = [
    "log_return",
    "zscore_close_64",
    "ema_spread",
    "atr_14",
    "price_slope_20",
]


@dataclass
class TrendDatasets:
    train: Dataset
    val: Dataset
    test: Dataset
    input_dim: int


class RollingWindowDataset(Dataset):
    """Lazy sequence dataset over multiple symbol arrays.

    Stores per-symbol feature/target arrays and creates sequence windows on
    demand in __getitem__, avoiding NxTxF eager expansion in memory.

    When ``returns_list`` is provided the dataset returns a 3-tuple
    (x, y, actual_return) so training loops can use execution-grade PnL.
    """

    def __init__(
        self,
        features_list: list[np.ndarray],
        target_list: list[np.ndarray],
        seq_len: int,
        returns_list: list[np.ndarray] | None = None,
    ):
        # Force writable copies; some upstream arrays are contiguous but read-only views.
        self.features_list = [torch.from_numpy(np.array(x, dtype=np.float32, copy=True)) for x in features_list]
        self.target_list = [torch.from_numpy(np.array(y, dtype=np.float32, copy=True)) for y in target_list]
        self.returns_list = (
            [torch.from_numpy(np.array(r, dtype=np.float32, copy=True)) for r in returns_list]
            if returns_list is not None else None
        )
        self.seq_len = seq_len

        self.lengths = [max(0, len(x) - seq_len + 1) for x in self.features_list]
        self.cum = np.cumsum(self.lengths).tolist()

    def __len__(self) -> int:
        return int(self.cum[-1]) if self.cum else 0

    def __getitem__(self, idx: int) -> tuple:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        s = bisect.bisect_right(self.cum, idx)
        prev = 0 if s == 0 else self.cum[s - 1]
        local_i = idx - prev

        x_arr = self.features_list[s]
        y_arr = self.target_list[s]

        x = x_arr[local_i : local_i + self.seq_len]
        y = y_arr[local_i + self.seq_len - 1]
        if self.returns_list is not None:
            r = self.returns_list[s][local_i + self.seq_len - 1]
            return x, y, r
        return x, y


def _log(message: str) -> None:
    print(message, flush=True)


def _load_symbol(path: Path) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    frame = pd.read_parquet(path, columns=cols)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame


def _window_count(features_list: list[np.ndarray], seq_len: int) -> int:
    return int(sum(max(0, len(x) - seq_len + 1) for x in features_list))


def build_trend_datasets(config: dict[str, Any]) -> TrendDatasets:
    pipe_cfg = PipelineConfig(
        dataset_dir=Path(config["dataset_dir"]),
        manifest_path=Path(config["manifest_path"]),
        min_history_bars=int(config["min_history_bars"]),
        purge_gap_bars=int(config["purge_gap_bars"]),
    )

    gate = DataQualityGate(pipe_cfg)
    accepted = [r for r in gate.evaluate() if r.decision == "ACCEPT"]
    max_symbols = int(config["max_symbols"])
    symbols = [r.symbol for r in accepted[:max_symbols]]
    if not symbols:
        raise RuntimeError("No accepted symbols available for trend training")

    _log(
        f"[trend-data] accepted_symbols={len(accepted)} selected_symbols={len(symbols)} "
        f"seq_len={config['seq_len']} horizon={config['horizon']}"
    )

    splitter = IronWallSplitter(purge_gap_bars=int(config["purge_gap_bars"]))

    seq_len = int(config["seq_len"])
    horizon = int(config["horizon"])
    cap_rows = int(config.get("max_rows_per_symbol", 0))

    x_train_list: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    r_train_list: list[np.ndarray] = []
    x_val_list: list[np.ndarray] = []
    y_val_list: list[np.ndarray] = []
    r_val_list: list[np.ndarray] = []
    x_test_list: list[np.ndarray] = []
    y_test_list: list[np.ndarray] = []
    r_test_list: list[np.ndarray] = []

    for symbol_idx, symbol in enumerate(symbols, start=1):
        path = pipe_cfg.dataset_dir / f"{symbol}.parquet"
        _log(f"[trend-data] loading {symbol_idx}/{len(symbols)} symbol={symbol}")
        raw = _load_symbol(path)
        feat = FeatureFactory.build_trend_features(raw)

        fwd_return = (feat["close"].shift(-horizon) / feat["close"]) - 1.0
        # Binary classification label: 1 = up, 0 = down
        feat["target_label"] = (fwd_return > 0).astype(np.float32)
        # Keep actual return for execution-grade PnL evaluation
        feat["target_return"] = fwd_return
        feat = feat[["timestamp", *FEATURE_COLUMNS, "target_label", "target_return"]].dropna().reset_index(drop=True)
        if cap_rows > 0:
            feat = feat.iloc[-cap_rows:].copy()

        split = splitter.split(feat, time_col="timestamp")

        scaler = FeatureFactory.fit_scaler_train_only(split.train, FEATURE_COLUMNS)
        train_scaled = FeatureFactory.transform_with_scaler(split.train, scaler)
        val_scaled = FeatureFactory.transform_with_scaler(split.val, scaler)
        test_scaled = FeatureFactory.transform_with_scaler(split.test, scaler)

        x_tr = train_scaled[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        # Binary classification label (not scaled; scaler only touches FEATURE_COLUMNS)
        y_tr = split.train["target_label"].to_numpy(dtype=np.float32)
        r_tr = split.train["target_return"].to_numpy(dtype=np.float32)
        x_va = val_scaled[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y_va = split.val["target_label"].to_numpy(dtype=np.float32)
        r_va = split.val["target_return"].to_numpy(dtype=np.float32)
        x_te = test_scaled[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y_te = split.test["target_label"].to_numpy(dtype=np.float32)
        r_te = split.test["target_return"].to_numpy(dtype=np.float32)

        if len(x_tr) >= seq_len:
            x_train_list.append(x_tr)
            y_train_list.append(y_tr)
            r_train_list.append(r_tr)
        if len(x_va) >= seq_len:
            x_val_list.append(x_va)
            y_val_list.append(y_va)
            r_val_list.append(r_va)
        if len(x_te) >= seq_len:
            x_test_list.append(x_te)
            y_test_list.append(y_te)
            r_test_list.append(r_te)

        _log(
            f"[trend-data] ready symbol={symbol} rows={len(feat)} "
            f"train={max(0, len(x_tr) - seq_len + 1)} val={max(0, len(x_va) - seq_len + 1)} "
            f"test={max(0, len(x_te) - seq_len + 1)}"
        )

    if _window_count(x_train_list, seq_len) == 0 or _window_count(x_val_list, seq_len) == 0 or _window_count(x_test_list, seq_len) == 0:
        raise RuntimeError("Split produced empty train/val/test sequences")

    train_ds = RollingWindowDataset(x_train_list, y_train_list, seq_len, returns_list=r_train_list)
    val_ds = RollingWindowDataset(x_val_list, y_val_list, seq_len, returns_list=r_val_list)
    test_ds = RollingWindowDataset(x_test_list, y_test_list, seq_len, returns_list=r_test_list)
    _log(
        f"[trend-data] datasets complete train_windows={len(train_ds)} "
        f"val_windows={len(val_ds)} test_windows={len(test_ds)}"
    )
    return TrendDatasets(train=train_ds, val=val_ds, test=test_ds, input_dim=len(FEATURE_COLUMNS))
