"""Statistical Arbitrage dataset builder — Phase 4.

Builds multi-asset spread sequences from accepted symbols.
Pairs are constructed from fractionally-differenced close prices.

Input shape:  [Batch, Seq_Len, Num_Assets]
Target:       Next-bar mean spread Z-score (regression)
Split:        Iron Wall 70/15/15 with purge_gap_bars
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


@dataclass
class StatArbDatasets:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    num_assets: int
    seq_len: int


def _log(message: str) -> None:
    print(message, flush=True)


def _load_symbol(path: Path) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    df = pd.read_parquet(path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _make_seq(features: np.ndarray, target: np.ndarray, seq_len: int):
    n, f = features.shape
    if n < seq_len + 1:
        raise ValueError("Too few rows")
    m = n - seq_len
    s0, s1 = features.strides
    x = np.lib.stride_tricks.as_strided(
        features, shape=(m, seq_len, f), strides=(s0, s0, s1), writeable=False
    ).copy()
    y = target[seq_len:]
    return x, y


def _verify_fracdiff_column(sym: str, frame: pd.DataFrame) -> None:
    if "fracdiff_close_d04" not in frame.columns:
        raise RuntimeError(f"[stat-arb-data] missing fracdiff_close_d04 for {sym}")
    series = frame["fracdiff_close_d04"].dropna()
    if len(series) < 500:
        raise RuntimeError(f"[stat-arb-data] insufficient fracdiff rows for {sym}: {len(series)}")
    lag1 = float(series.autocorr(lag=1)) if len(series) > 2 else 1.0
    std = float(series.std())
    if not np.isfinite(std) or std < 1e-8:
        raise RuntimeError(f"[stat-arb-data] fracdiff verification failed for {sym}: invalid_std={std}")
    _log(
        f"[stat-arb-data] fracdiff_verify symbol={sym} rows={len(series)} "
        f"lag1_autocorr={lag1:.4f} std={std:.6f}"
    )
    if np.isfinite(lag1) and abs(lag1) > 0.995:
        _log(
            f"[stat-arb-data] warning symbol={sym} high_lag1_autocorr={lag1:.4f}; "
            "continuing because fracdiff column and variance checks passed"
        )


def build_stat_arb_datasets(config: dict[str, Any]) -> StatArbDatasets:
    pipe_cfg = PipelineConfig(
        dataset_dir=Path(config["dataset_dir"]),
        manifest_path=Path(config["manifest_path"]),
        min_history_bars=int(config["min_history_bars"]),
        purge_gap_bars=int(config["purge_gap_bars"]),
    )
    accepted = [r for r in DataQualityGate(pipe_cfg).evaluate() if r.decision == "ACCEPT"]
    max_assets = int(config["max_assets"])
    symbols = [r.symbol for r in accepted[:max_assets]]
    if len(symbols) < 2:
        raise RuntimeError("Need at least 2 accepted symbols for stat arb")

    _log(f"[stat-arb-data] accepted_symbols={len(accepted)} selected_assets={len(symbols)} seq_len={config['seq_len']} horizon={config['horizon']}")

    cap_rows = int(config.get("max_rows_per_symbol", 0))
    seq_len = int(config["seq_len"])
    horizon = int(config["horizon"])

    # Load all symbols, align on common timestamps
    frames: dict[str, pd.DataFrame] = {}
    for sym_idx, sym in enumerate(symbols, start=1):
        _log(f"[stat-arb-data] loading {sym_idx}/{len(symbols)} symbol={sym}")
        raw = _load_symbol(pipe_cfg.dataset_dir / f"{sym}.parquet")
        feat = FeatureFactory.build_stat_arb_features(raw)
        feat = feat[["timestamp", "fracdiff_close_d04", "spread_z_64"]].dropna()
        _verify_fracdiff_column(sym, feat)
        # Do NOT cap per-symbol before alignment — different listing dates
        # would produce disjoint timestamp ranges.
        frames[sym] = feat.set_index("timestamp")
        _log(f"[stat-arb-data] ready symbol={sym} rows={len(feat)}")

    # Align to common index
    aligned = pd.concat({s: frames[s]["fracdiff_close_d04"] for s in symbols}, axis=1).dropna()
    aligned.columns = symbols
    aligned = aligned.reset_index()
    _log(f"[stat-arb-data] aligned_rows={len(aligned)} assets={len(symbols)}")

    # Cap the aligned (post-intersection) rows — take LAST N to keep recency
    if cap_rows > 0 and len(aligned) > cap_rows:
        aligned = aligned.iloc[-cap_rows:].copy()

    # Iron Wall split based on timestamp column
    splitter = IronWallSplitter(purge_gap_bars=int(config["purge_gap_bars"]))
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True)
    split = splitter.split(aligned, time_col="timestamp")

    def _process_split(df: pd.DataFrame):
        arr = df[symbols].to_numpy(np.float32)
        # Fit scaler per-asset on this slice (we refit for each split segment
        # but only using the train slice's stats — passed in from outside)
        return arr

    tr_arr = _process_split(split.train)
    va_arr = _process_split(split.val)
    te_arr = _process_split(split.test)

    # Normalize: fit scaler on train only
    mean = tr_arr.mean(axis=0, keepdims=True)
    std = tr_arr.std(axis=0, keepdims=True) + 1e-8
    tr_arr = (tr_arr - mean) / std
    va_arr = (va_arr - mean) / std
    te_arr = (te_arr - mean) / std

    # Target: mean spread Z-score at next `horizon` steps
    def _make_target(arr: np.ndarray) -> np.ndarray:
        z = (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-8)
        return z.mean(axis=1)  # (N,) scalar mean Z across assets

    for arr in [tr_arr, va_arr, te_arr]:
        pass  # arrays already normalized

    def _build(arr):
        tgt = _make_target(arr)
        x, y = _make_seq(arr, tgt, seq_len)
        return TensorDataset(torch.tensor(x), torch.tensor(y.astype(np.float32)))

    datasets = StatArbDatasets(
        train=_build(tr_arr),
        val=_build(va_arr),
        test=_build(te_arr),
        num_assets=len(symbols),
        seq_len=seq_len,
    )
    _log(f"[stat-arb-data] datasets complete train={len(datasets.train)} val={len(datasets.val)} test={len(datasets.test)}")
    return datasets
