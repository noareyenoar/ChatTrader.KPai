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
from data_pipeline.splitter import IronWallSplitter, PurgedWalkForwardSplitter


FEATURE_COLUMNS = [
    "log_return",
    "zscore_close_64",
    "ema_spread",
    "atr_14",
    "price_slope_20",
]

# V2.0 beta-neutralization columns (appended when BTC reference is available)
BTC_RESIDUAL_COLUMNS = [
    "btc_residual_return",
    "btc_relative_vol",
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
        future_list: list[np.ndarray] | None = None,
        stride: int = 1,
    ):
        # Force writable copies; some upstream arrays are contiguous but read-only views.
        self.features_list = [torch.from_numpy(np.array(x, dtype=np.float32, copy=True)) for x in features_list]
        self.target_list = [torch.from_numpy(np.array(y, dtype=np.float32, copy=True)) for y in target_list]
        self.returns_list = (
            [torch.from_numpy(np.array(r, dtype=np.float32, copy=True)) for r in returns_list]
            if returns_list is not None else None
        )
        # V2.0 LUPI: privileged future structural signals (train split only).
        # ██████  IRON WALL  ██████  Must be None for val/test datasets.
        self.future_list = (
            [torch.from_numpy(np.array(f, dtype=np.float32, copy=True)) for f in future_list]
            if future_list is not None else None
        )
        self.seq_len = seq_len
        self.stride = max(1, int(stride))

        self.lengths = [max(0, (len(x) - seq_len) // self.stride + 1) for x in self.features_list]
        self.cum = np.cumsum(self.lengths).tolist()

    def __len__(self) -> int:
        return int(self.cum[-1]) if self.cum else 0

    def __getitem__(self, idx: int) -> tuple:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        s = bisect.bisect_right(self.cum, idx)
        prev = 0 if s == 0 else self.cum[s - 1]
        local_i = idx - prev
        start = local_i * self.stride

        x_arr = self.features_list[s]
        y_arr = self.target_list[s]

        x = x_arr[start : start + self.seq_len]
        y = y_arr[start + self.seq_len - 1]
        if self.returns_list is not None:
            r = self.returns_list[s][start + self.seq_len - 1]
            if self.future_list is not None:
                # 4-tuple: (x, y, actual_return, future_priv)  ← LUPI training batch
                f = self.future_list[s][start + self.seq_len - 1]
                return x, y, r, f
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


def _load_btc_reference(dataset_dir: Path) -> pd.DataFrame | None:
    """Load BTC reference data for V2.0 beta neutralization.

    Tries BTCUSDT.parquet first, then BTCUSD.parquet as fallback.
    Returns None when no BTC reference is found (beta neutralization skipped).
    """
    for name in ("BTCUSDT.parquet", "BTCUSD.parquet"):
        p = dataset_dir / name
        if p.exists():
            try:
                df = pd.read_parquet(p, columns=["timestamp", "close"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
                _log(f"[trend-data] BTC reference loaded: {name}  rows={len(df)}")
                return df
            except Exception as e:
                _log(f"[trend-data] WARNING: could not load BTC reference {name}: {e}")
    _log("[trend-data] WARNING: no BTC reference found — beta neutralization DISABLED")
    return None


def _window_count(features_list: list[np.ndarray], seq_len: int, stride: int = 1) -> int:
    stride = max(1, int(stride))
    return int(sum(max(0, (len(x) - seq_len) // stride + 1) for x in features_list))


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
    stride = max(1, int(config.get("stride", 1)))
    use_lupi: bool = bool(config.get("use_lupi", False))
    walk_forward_enabled: bool = bool(config.get("walk_forward_enabled", False))
    walk_forward_folds: int = int(config.get("walk_forward_folds", 5))
    walk_forward_min_train_bars: int = int(config.get("walk_forward_min_train_bars", 500))

    # V2.0: Load BTC reference for beta neutralization
    btc_frame: pd.DataFrame | None = None
    if bool(config.get("use_btc_beta", True)):
        btc_frame = _load_btc_reference(pipe_cfg.dataset_dir)
    btc_beta_window: int = int(config.get("btc_beta_window", 1440))

    x_train_list: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    r_train_list: list[np.ndarray] = []
    f_train_list: list[np.ndarray] = []   # V2.0 LUPI privileged future signals
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
        # V2.0: BTC-beta-neutralized features when BTC reference is available
        feat = FeatureFactory.build_trend_features(raw, btc_frame=btc_frame, btc_beta_window=btc_beta_window)

        # Resolve active feature columns (base + optional BTC residuals)
        active_cols = list(FEATURE_COLUMNS)
        if btc_frame is not None:
            active_cols += [c for c in BTC_RESIDUAL_COLUMNS if c in feat.columns]

        fwd_return = (feat["close"].shift(-horizon) / feat["close"]) - 1.0
        # Binary classification label: 1 = up, 0 = down
        feat["target_label"] = (fwd_return > 0).astype(np.float32)
        # Keep actual return for execution-grade PnL evaluation
        feat["target_return"] = fwd_return
        keep_cols = ["timestamp", *active_cols, "target_label", "target_return"]
        feat = feat[[c for c in keep_cols if c in feat.columns]].dropna().reset_index(drop=True)
        if cap_rows > 0:
            feat = feat.iloc[-cap_rows:].copy()

        sym_x_tr: list[np.ndarray] = []
        sym_y_tr: list[np.ndarray] = []
        sym_r_tr: list[np.ndarray] = []
        sym_f_tr: list[np.ndarray] = []
        sym_x_va: list[np.ndarray] = []
        sym_y_va: list[np.ndarray] = []
        sym_r_va: list[np.ndarray] = []
        sym_x_te: list[np.ndarray] = []
        sym_y_te: list[np.ndarray] = []
        sym_r_te: list[np.ndarray] = []

        if walk_forward_enabled:
            wf = PurgedWalkForwardSplitter(
                n_folds=walk_forward_folds,
                purge_gap_bars=int(config["purge_gap_bars"]),
                min_train_bars=walk_forward_min_train_bars,
            )
            folds = wf.split(feat, time_col="timestamp")
            if len(folds) < 2:
                raise RuntimeError("walk_forward_enabled requires at least 2 valid folds")

            for fold_idx, fold in enumerate(folds):
                scaler = FeatureFactory.fit_scaler_train_only(fold.train, active_cols)
                train_scaled = FeatureFactory.transform_with_scaler(fold.train, scaler)
                test_scaled = FeatureFactory.transform_with_scaler(fold.test, scaler)

                x_fold_tr = train_scaled[active_cols].to_numpy(dtype=np.float32)
                y_fold_tr = fold.train["target_label"].to_numpy(dtype=np.float32)
                r_fold_tr = fold.train["target_return"].to_numpy(dtype=np.float32)
                x_fold_te = test_scaled[active_cols].to_numpy(dtype=np.float32)
                y_fold_te = fold.test["target_label"].to_numpy(dtype=np.float32)
                r_fold_te = fold.test["target_return"].to_numpy(dtype=np.float32)

                if len(x_fold_tr) >= seq_len:
                    sym_x_tr.append(x_fold_tr)
                    sym_y_tr.append(y_fold_tr)
                    sym_r_tr.append(r_fold_tr)
                    if use_lupi:
                        raw_log_ret = r_fold_tr.astype(np.float64)
                        future_avg = np.array(
                            [raw_log_ret[i + 1: i + 1 + horizon].mean() if i + 1 + horizon <= len(raw_log_ret)
                             else 0.0 for i in range(len(raw_log_ret))],
                            dtype=np.float32,
                        )
                        future_vol = np.array(
                            [raw_log_ret[i + 1: i + 1 + horizon].std() if i + 1 + horizon <= len(raw_log_ret)
                             else 0.0 for i in range(len(raw_log_ret))],
                            dtype=np.float32,
                        )
                        sym_f_tr.append(np.stack([future_avg, future_vol], axis=1))

                # Fold tests become walk-forward validation windows, except the
                # final fold which is reserved as strict OOS test.
                if fold_idx < len(folds) - 1:
                    if len(x_fold_te) >= seq_len:
                        sym_x_va.append(x_fold_te)
                        sym_y_va.append(y_fold_te)
                        sym_r_va.append(r_fold_te)
                else:
                    if len(x_fold_te) >= seq_len:
                        sym_x_te.append(x_fold_te)
                        sym_y_te.append(y_fold_te)
                        sym_r_te.append(r_fold_te)
        else:
            split = splitter.split(feat, time_col="timestamp")

            scaler = FeatureFactory.fit_scaler_train_only(split.train, active_cols)
            train_scaled = FeatureFactory.transform_with_scaler(split.train, scaler)
            val_scaled = FeatureFactory.transform_with_scaler(split.val, scaler)
            test_scaled = FeatureFactory.transform_with_scaler(split.test, scaler)

            sym_x_tr.append(train_scaled[active_cols].to_numpy(dtype=np.float32))
            sym_y_tr.append(split.train["target_label"].to_numpy(dtype=np.float32))
            sym_r_tr.append(split.train["target_return"].to_numpy(dtype=np.float32))
            sym_x_va.append(val_scaled[active_cols].to_numpy(dtype=np.float32))
            sym_y_va.append(split.val["target_label"].to_numpy(dtype=np.float32))
            sym_r_va.append(split.val["target_return"].to_numpy(dtype=np.float32))
            sym_x_te.append(test_scaled[active_cols].to_numpy(dtype=np.float32))
            sym_y_te.append(split.test["target_label"].to_numpy(dtype=np.float32))
            sym_r_te.append(split.test["target_return"].to_numpy(dtype=np.float32))

        for _x, _y, _r in zip(sym_x_tr, sym_y_tr, sym_r_tr):
            if len(_x) >= seq_len:
                x_train_list.append(_x)
                y_train_list.append(_y)
                r_train_list.append(_r)
        if use_lupi and sym_f_tr:
            for _f in sym_f_tr:
                if len(_f) >= seq_len:
                    f_train_list.append(_f)
        for _x, _y, _r in zip(sym_x_va, sym_y_va, sym_r_va):
            if len(_x) >= seq_len:
                x_val_list.append(_x)
                y_val_list.append(_y)
                r_val_list.append(_r)
        for _x, _y, _r in zip(sym_x_te, sym_y_te, sym_r_te):
            if len(_x) >= seq_len:
                x_test_list.append(_x)
                y_test_list.append(_y)
                r_test_list.append(_r)

        _log(
            f"[trend-data] ready symbol={symbol} rows={len(feat)} "
            f"input_dim={len(active_cols)} btc_beta={'yes' if btc_frame is not None else 'no'} "
            f"train_win={sum(max(0, (len(v) - seq_len) // stride + 1) for v in sym_x_tr)} "
            f"val_win={sum(max(0, (len(v) - seq_len) // stride + 1) for v in sym_x_va)} "
            f"test_win={sum(max(0, (len(v) - seq_len) // stride + 1) for v in sym_x_te)} "
            f"walk_forward={'yes' if walk_forward_enabled else 'no'}"
        )
        final_input_dim = len(active_cols)  # capture after first symbol resolves

    if _window_count(x_train_list, seq_len, stride) == 0 or _window_count(x_val_list, seq_len, stride) == 0 or _window_count(x_test_list, seq_len, stride) == 0:
        raise RuntimeError("Split produced empty train/val/test sequences")

    # Derive input_dim from the first processed symbol's active columns
    final_input_dim = x_train_list[0].shape[1] if x_train_list else len(FEATURE_COLUMNS)

    train_ds = RollingWindowDataset(
        x_train_list, y_train_list, seq_len,
        returns_list=r_train_list,
        future_list=f_train_list if (use_lupi and f_train_list) else None,
        stride=stride,
    )
    val_ds = RollingWindowDataset(x_val_list, y_val_list, seq_len, returns_list=r_val_list, stride=stride)
    test_ds = RollingWindowDataset(x_test_list, y_test_list, seq_len, returns_list=r_test_list, stride=stride)
    _log(
        f"[trend-data] datasets complete train_windows={len(train_ds)} "
        f"val_windows={len(val_ds)} test_windows={len(test_ds)} "
        f"input_dim={final_input_dim} stride={stride} btc_neutralized={'yes' if btc_frame is not None else 'no'}"
    )
    return TrendDatasets(train=train_ds, val=val_ds, test=test_ds, input_dim=final_input_dim)
