"""Scalper dataset builder — Phase 4.

Simulates order-book microstructure features from OHLCV data since
raw Level-2 LOB data is not available.  All derived features are causal
(no lookahead) and computed from public OHLCV fields.

Input shape:  [Batch, Seq_Len=32, 5]
Target:       3-class label (0=down, 1=flat, 2=up) based on next-bar return
Features:     ofi_proxy, microprice_dev, spread_pct, log_return, vol_imbalance
Split:        Iron Wall 70/15/15 with purge_gap_bars
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import bisect

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_pipeline.config import PipelineConfig
from data_pipeline.quality_gate import DataQualityGate
from data_pipeline.splitter import IronWallSplitter
from data_pipeline.features import FeatureFactory


SCALPER_FEATURES = [
    "ofi_proxy",       # Order-flow imbalance proxy: (close-open)/(high-low+1e-8)
    "microprice_dev",  # Mid-price deviation: (close - vwap) / atr
    "spread_pct",      # Estimated bid-ask spread as fraction of price
    "log_return",      # Bar log-return
    "vol_imbalance",   # Volume imbalance: (vol_up - vol_dn) / (vol_up + vol_dn + 1e-8)
    "fracdiff_close_d04",
    "fracdiff_volume_d04",
    "buy_sell_pressure",
    "price_velocity_5",
    "price_velocity_10",
    "price_velocity_15",
    "volatility_z_32",
    "vol_regime_code",
]


def _build_scalper_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().sort_values("timestamp").reset_index(drop=True)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)
    qvol = df["quote_volume"].astype(float)
    taker_buy_base = df["taker_buy_base"].astype(float)
    taker_buy_quote = df["taker_buy_quote"].astype(float)

    atr = FeatureFactory.atr(df, window=14)
    vwap = qvol / (vol.replace(0.0, np.nan) + 1e-8)

    df["ofi_proxy"] = (close - open_) / (high - low + 1e-8)
    df["microprice_dev"] = (close - vwap) / (atr + 1e-8)
    df["spread_pct"] = (high - low) / (close + 1e-8)
    df["log_return"] = FeatureFactory.log_return(close, use_torch_cuda=False)

    vol_up = vol.where(close >= open_, 0.0).fillna(0.0)
    vol_dn = vol.where(close < open_, 0.0).fillna(0.0)
    df["vol_imbalance"] = (vol_up - vol_dn) / (vol_up + vol_dn + 1e-8)

    # v26-4 feature upgrades
    df["fracdiff_close_d04"] = FeatureFactory.fractional_diff(close, d=0.4)
    df["fracdiff_volume_d04"] = FeatureFactory.fractional_diff(vol, d=0.4)
    df["buy_sell_pressure"] = taker_buy_base / (taker_buy_quote + 1e-8)
    df["price_velocity_5"] = (close - close.shift(5)) / 5.0
    df["price_velocity_10"] = (close - close.shift(10)) / 10.0
    df["price_velocity_15"] = (close - close.shift(15)) / 15.0
    realized_vol = df["log_return"].rolling(window=32, min_periods=32).std()
    df["volatility_z_32"] = FeatureFactory.rolling_zscore(realized_vol, window=32)
    vol_q1 = realized_vol.quantile(0.33)
    vol_q2 = realized_vol.quantile(0.66)
    df["vol_regime_code"] = np.where(realized_vol <= vol_q1, 0.0, np.where(realized_vol <= vol_q2, 1.0, 2.0))
    return df


def _make_seq(features: np.ndarray, target: np.ndarray, seq_len: int):
    n, f = features.shape
    if n < seq_len:
        raise ValueError("Too few rows for sequence")
    m = n - seq_len + 1
    s0, s1 = features.strides
    x = np.lib.stride_tricks.as_strided(
        features, shape=(m, seq_len, f), strides=(s0, s0, s1), writeable=False
    ).copy()
    y = target[seq_len - 1:]
    return x, y


def _load_symbol(path: Path) -> pd.DataFrame:
    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    df = pd.read_parquet(path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


@dataclass
class ScalperDatasets:
    train: Dataset
    val: Dataset
    test: Dataset
    input_dim: int
    seq_len: int
    scaler: Any = field(default=None, repr=False)  # fitted StandardScaler for eval reuse


class RollingClassWindowDataset(Dataset):
    """Lazy sequence dataset for classification targets.

    Stores per-symbol feature/target arrays and creates windows on demand,
    avoiding giant eager [N, T, F] concatenations.

    When ``returns_list`` is provided the dataset returns a 3-tuple
    (x, y, actual_return) for execution-grade PnL tracking.
    """

    def __init__(
        self,
        features_list: list[np.ndarray],
        target_list: list[np.ndarray],
        seq_len: int,
        returns_list: list[np.ndarray] | None = None,
    ):
        self.features_list = [torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)) for x in features_list]
        self.target_list = [torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)) for y in target_list]
        self.returns_list = (
            [torch.from_numpy(np.ascontiguousarray(r, dtype=np.float32)) for r in returns_list]
            if returns_list is not None else None
        )
        self.seq_len = int(seq_len)
        self.lengths = [max(0, len(x) - self.seq_len + 1) for x in self.features_list]
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


def _window_count(features_list: list[np.ndarray], seq_len: int) -> int:
    return int(sum(max(0, len(x) - seq_len + 1) for x in features_list))


def _log(message: str) -> None:
    print(message, flush=True)


def build_scalper_datasets(config: dict[str, Any]) -> ScalperDatasets:
    pipe_cfg = PipelineConfig(
        dataset_dir=Path(config["dataset_dir"]),
        manifest_path=Path(config["manifest_path"]),
        min_history_bars=int(config["min_history_bars"]),
        purge_gap_bars=int(config["purge_gap_bars"]),
    )
    accepted = [r for r in DataQualityGate(pipe_cfg).evaluate() if r.decision == "ACCEPT"]
    symbols = [r.symbol for r in accepted[:int(config["max_symbols"])]]
    if not symbols:
        raise RuntimeError("No accepted symbols for scalper training")

    _log(f"[scalper-data] accepted_symbols={len(accepted)} selected_symbols={len(symbols)} seq_len={config['seq_len']} horizon={config['horizon']}")

    splitter = IronWallSplitter(purge_gap_bars=int(config["purge_gap_bars"]))
    seq_len = int(config["seq_len"])
    horizon = int(config["horizon"])
    cap_rows = int(config.get("max_rows_per_symbol", 0))
    flat_threshold = float(config.get("flat_threshold", 0.0003))

    x_tr, y_tr, r_tr, x_va, y_va, r_va, x_te, y_te, r_te = [], [], [], [], [], [], [], [], []

    for symbol_idx, symbol in enumerate(symbols, start=1):
        _log(f"[scalper-data] loading {symbol_idx}/{len(symbols)} symbol={symbol}")
        raw = _load_symbol(pipe_cfg.dataset_dir / f"{symbol}.parquet")
        feat = _build_scalper_features(raw)

        # 3-class target: down=0, flat=1, up=2
        fwd_ret = feat["close"].shift(-horizon) / feat["close"] - 1.0
        feat["target"] = 1  # flat
        feat.loc[fwd_ret > flat_threshold, "target"] = 2
        feat.loc[fwd_ret < -flat_threshold, "target"] = 0
        # Actual signed forward return for execution-grade PnL metric
        feat["_actual_return"] = fwd_ret

        feat = feat[["timestamp", *SCALPER_FEATURES, "target", "_actual_return"]].dropna().reset_index(drop=True)
        if cap_rows > 0:
            feat = feat.iloc[-cap_rows:].copy()

        split = splitter.split(feat, time_col="timestamp")
        scaler = FeatureFactory.fit_scaler_train_only(split.train, SCALPER_FEATURES)
        tr = FeatureFactory.transform_with_scaler(split.train, scaler)
        va = FeatureFactory.transform_with_scaler(split.val, scaler)
        te = FeatureFactory.transform_with_scaler(split.test, scaler)

        for (df_, xl, yl, rl) in [(tr, x_tr, y_tr, r_tr), (va, x_va, y_va, r_va), (te, x_te, y_te, r_te)]:
            fx = df_[SCALPER_FEATURES].to_numpy(np.float32)
            fy = df_["target"].to_numpy(np.int64)
            fr = df_["_actual_return"].to_numpy(np.float32)
            if len(fx) >= seq_len:
                xl.append(fx)
                yl.append(fy)
                rl.append(fr)
        _log(f"[scalper-data] ready symbol={symbol} train={len(tr)} val={len(va)} test={len(te)}")

    if _window_count(x_tr, seq_len) == 0 or _window_count(x_va, seq_len) == 0 or _window_count(x_te, seq_len) == 0:
        raise RuntimeError("Split produced empty train/val/test sequences for scalper")

    # Save scaler from the last symbol for checkpoint reuse during evaluation.
    datasets = ScalperDatasets(
        train=RollingClassWindowDataset(x_tr, y_tr, seq_len, returns_list=r_tr),
        val=RollingClassWindowDataset(x_va, y_va, seq_len, returns_list=r_va),
        test=RollingClassWindowDataset(x_te, y_te, seq_len, returns_list=r_te),
        input_dim=len(SCALPER_FEATURES),
        seq_len=seq_len,
        scaler=scaler,
    )
    _log(f"[scalper-data] datasets complete train={len(datasets.train)} val={len(datasets.val)} test={len(datasets.test)}")
    return datasets


def save_scalper_scaler(scaler: Any, ckpt_dir: Path) -> None:
    """Persist the fitted StandardScaler alongside the model checkpoint."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = ckpt_dir / "feature_scaler.pkl"
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[scalper-data] scaler saved -> {scaler_path}", flush=True)


def load_scalper_scaler(ckpt_dir: Path) -> Any | None:
    """Load the fitted StandardScaler from checkpoint directory. Returns None if not found."""
    scaler_path = ckpt_dir / "feature_scaler.pkl"
    if not scaler_path.exists():
        return None
    with open(scaler_path, "rb") as fh:
        return pickle.load(fh)
