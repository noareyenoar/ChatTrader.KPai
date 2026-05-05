"""Mean Reversion dataset builder — Phase 4.

Input shape:  [Batch, 5]  (tabular, no time dimension)
Target:       Binary — 1 if price reverses upward in next `horizon` bars.
Features:     vwap_dev, bb_distance, zscore_close_20, rsi_14, rsi_div_5
Split:        Iron Wall 70/15/15 with purge_gap_bars
Scaler:       Fitted on train only; applied to val/test
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from data_pipeline.config import PipelineConfig
from data_pipeline.features import FeatureFactory
from data_pipeline.quality_gate import DataQualityGate
from data_pipeline.splitter import IronWallSplitter


MR_FEATURE_COLUMNS = [
    "vwap_dev",
    "bb_distance",
    "zscore_close_20",
    "rsi_14",
    "rsi_div_5",
]


@dataclass
class MRDatasets:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    input_dim: int


def _log(message: str) -> None:
    print(message, flush=True)


def _load_symbol(path: Path) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    frame = pd.read_parquet(path, columns=cols)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def build_mr_datasets(config: dict[str, Any]) -> MRDatasets:
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
        raise RuntimeError("No accepted symbols available for mean reversion training")

    _log(f"[mr-data] accepted_symbols={len(accepted)} selected_symbols={len(symbols)} horizon={horizon if 'horizon' in locals() else config['horizon']}")

    splitter = IronWallSplitter(purge_gap_bars=int(config["purge_gap_bars"]))
    horizon = int(config["horizon"])
    cap_rows = int(config.get("max_rows_per_symbol", 0))

    x_tr, y_tr, r_tr, x_va, y_va, r_va, x_te, y_te, r_te = [], [], [], [], [], [], [], [], []

    for symbol_idx, symbol in enumerate(symbols, start=1):
        path = pipe_cfg.dataset_dir / f"{symbol}.parquet"
        _log(f"[mr-data] loading {symbol_idx}/{len(symbols)} symbol={symbol}")
        raw = _load_symbol(path)
        feat = FeatureFactory.build_mean_reversion_features(raw)

        # Target: 1 if price moves up over next `horizon` bars, else 0
        feat["target"] = (feat["close"].shift(-horizon) > feat["close"]).astype(np.float32)
        # Actual signed forward return for execution-grade PnL metric
        feat["_actual_return"] = feat["close"].shift(-horizon) / feat["close"] - 1.0
        feat = feat[["timestamp", *MR_FEATURE_COLUMNS, "target", "_actual_return"]].dropna().reset_index(drop=True)
        if cap_rows > 0:
            feat = feat.iloc[-cap_rows:].copy()

        split = splitter.split(feat, time_col="timestamp")
        scaler = FeatureFactory.fit_scaler_train_only(split.train, MR_FEATURE_COLUMNS)
        tr = FeatureFactory.transform_with_scaler(split.train, scaler)
        va = FeatureFactory.transform_with_scaler(split.val, scaler)
        te = FeatureFactory.transform_with_scaler(split.test, scaler)

        x_tr.append(tr[MR_FEATURE_COLUMNS].to_numpy(np.float32))
        y_tr.append(tr["target"].to_numpy(np.float32))
        r_tr.append(tr["_actual_return"].to_numpy(np.float32))
        x_va.append(va[MR_FEATURE_COLUMNS].to_numpy(np.float32))
        y_va.append(va["target"].to_numpy(np.float32))
        r_va.append(va["_actual_return"].to_numpy(np.float32))
        x_te.append(te[MR_FEATURE_COLUMNS].to_numpy(np.float32))
        y_te.append(te["target"].to_numpy(np.float32))
        r_te.append(te["_actual_return"].to_numpy(np.float32))
        _log(f"[mr-data] ready symbol={symbol} train={len(tr)} val={len(va)} test={len(te)}")
    def _ds(xs, ys, rs):
        x = torch.tensor(np.concatenate(xs, axis=0))
        y = torch.tensor(np.concatenate(ys, axis=0))
        r = torch.tensor(np.concatenate(rs, axis=0))
        return TensorDataset(x, y, r)

    datasets = MRDatasets(
        train=_ds(x_tr, y_tr, r_tr),
        val=_ds(x_va, y_va, r_va),
        test=_ds(x_te, y_te, r_te),
        input_dim=len(MR_FEATURE_COLUMNS),
    )
    _log(f"[mr-data] datasets complete train={len(datasets.train)} val={len(datasets.val)} test={len(datasets.test)}")
    return datasets
