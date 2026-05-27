"""Data loading and preparation for TG-MNN model.

Integrates with ChatTrader.KPai's data pipeline:
- Loads preprocessed parquet files
- Applies wave feature extraction
- Creates train/val/test splits with strict chronological ordering
- Fits scalers on training set only
- Creates lazy-loading datasets for memory efficiency
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import bisect
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_pipeline.config import PipelineConfig
from data_pipeline.features import FeatureFactory, ScalerStats
from data_pipeline.quality_gate import DataQualityGate
from data_pipeline.splitter import IronWallSplitter
from .wave_extractor import WaveFeatureBuilder, ZigZagExtractor


@dataclass
class TGMNNDatasets:
    """Container for train/val/test datasets with metadata."""
    train: Dataset
    val: Dataset
    test: Dataset
    input_dim: int
    scaler_stats: ScalerStats


class WaveDataset(Dataset):
    """Lazy multi-symbol dataset for wave-based training.

    Stores per-symbol feature/target arrays and creates sequence windows on
    demand, avoiding memory explosion for large datasets.
    """

    def __init__(
        self,
        features_list: list[np.ndarray],
        state_list: list[np.ndarray],
        magnitude_list: list[np.ndarray],
        duration_list: list[np.ndarray],
        seq_len: int,
        returns_list: Optional[list[np.ndarray]] = None,
    ):
        """
        Args:
            features_list: List of [T, F] arrays per symbol
            state_list: List of [T] state labels per symbol
            magnitude_list: List of [T] magnitude targets per symbol
            duration_list: List of [T] duration targets per symbol
            seq_len: Sequence length for rolling windows
            returns_list: Optional list of [T] actual returns per symbol
        """
        # Convert to writable float32 tensors
        self.features_list = [
            torch.from_numpy(np.array(x, dtype=np.float32, copy=True))
            for x in features_list
        ]
        self.state_list = [
            torch.from_numpy(np.array(x, dtype=np.int64, copy=True))
            for x in state_list
        ]
        self.magnitude_list = [
            torch.from_numpy(np.array(x, dtype=np.float32, copy=True))
            for x in magnitude_list
        ]
        self.duration_list = [
            torch.from_numpy(np.array(x, dtype=np.float32, copy=True))
            for x in duration_list
        ]
        self.returns_list = (
            [torch.from_numpy(np.array(r, dtype=np.float32, copy=True)) for r in returns_list]
            if returns_list is not None else None
        )
        self.seq_len = seq_len

        # Precompute cumulative indices for fast binary search
        self.lengths = [max(0, len(x) - seq_len + 1) for x in self.features_list]
        self.cum = np.cumsum(self.lengths).tolist()

    def __len__(self) -> int:
        return int(self.cum[-1]) if self.cum else 0

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sequence window and its targets.

        Returns:
            If returns_list provided: (x, state, magnitude, duration, actual_return)
            Otherwise: (x, state, magnitude, duration)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Find which symbol this index belongs to
        s = bisect.bisect_right(self.cum, idx)
        prev = 0 if s == 0 else self.cum[s - 1]
        local_i = idx - prev

        # Extract sequence window
        x_arr = self.features_list[s]
        state_arr = self.state_list[s]
        mag_arr = self.magnitude_list[s]
        dur_arr = self.duration_list[s]

        x = x_arr[local_i : local_i + self.seq_len]
        state = state_arr[local_i + self.seq_len - 1]
        magnitude = mag_arr[local_i + self.seq_len - 1]
        duration = dur_arr[local_i + self.seq_len - 1]

        if self.returns_list is not None:
            r = self.returns_list[s][local_i + self.seq_len - 1]
            return x, state, magnitude, duration, r

        return x, state, magnitude, duration


def _log(message: str) -> None:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = message.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe, flush=True)


def _load_symbol(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load symbol data from parquet, optionally capping at max_rows most-recent bars."""
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    frame = pd.read_parquet(path, columns=cols)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[-max_rows:].reset_index(drop=True)
    return frame


def prepare_tg_mnn_datasets(
    data_dir: Path,
    seq_len: int = 50,
    symbols: Optional[list[str]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_gap: int = 20,
    scaler: Optional[ScalerStats] = None,
    max_rows_per_symbol: Optional[int] = None,
) -> TGMNNDatasets:
    """
    Prepare train/val/test datasets for TG-MNN with wave labeling.

    Args:
        data_dir: Directory containing parquet files
        seq_len: Sequence length for rolling windows
        symbols: List of symbols to load; if None, auto-discover
        train_ratio, val_ratio: Split proportions
        purge_gap: Number of bars between splits
        scaler: Pre-fitted scaler; if None, fit on training data

    Returns:
        TGMNNDatasets with train/val/test loaders and metadata
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Auto-discover symbols
    if symbols is None:
        parquet_files = sorted(data_dir.glob("*.parquet"))
        symbols = [f.stem for f in parquet_files]
        if not symbols:
            raise ValueError(f"No parquet files found in {data_dir}")

    _log(f"[TG-MNN] Loading {len(symbols)} symbols from {data_dir}")

    # Load and process each symbol
    splitter = IronWallSplitter(train_ratio, val_ratio, test_ratio=0.15, purge_gap_bars=purge_gap)
    feature_builder = WaveFeatureBuilder()

    train_features = []
    train_states = []
    train_magnitudes = []
    train_durations = []

    val_features = []
    val_states = []
    val_magnitudes = []
    val_durations = []

    test_features = []
    test_states = []
    test_magnitudes = []
    test_durations = []

    for symbol in symbols:
        parquet_path = data_dir / f"{symbol}.parquet"
        if not parquet_path.exists():
            _log(f"[TG-MNN] Warning: {parquet_path} not found; skipping")
            continue

        try:
            df_raw = _load_symbol(parquet_path, max_rows=max_rows_per_symbol)
            _log(f"[TG-MNN] {symbol}: Loaded {len(df_raw)} bars")

            # Build wave features
            df_feat = feature_builder.build_wave_features(df_raw)

            # Normalize regression targets to stable ranges before splitting
            # magnitude: raw price diff → fractional move [0, 1]; avoids huge loss values
            df_feat['target_magnitude'] = (
                df_feat['target_magnitude'] / (df_feat['close'] + 1e-8)
            ).clip(0.0, 1.0)
            # duration: bars → fraction of 200-bar max [0, 1]
            df_feat['target_duration'] = (
                df_feat['target_duration'] / 200.0
            ).clip(0.0, 1.0)

            # Drop rows with NaN features (initial rolling-window warm-up bars)
            df_feat = df_feat.dropna(
                subset=WaveFeatureBuilder.FEATURE_COLUMNS
            ).reset_index(drop=True)

            # Split chronologically with purge gap
            split = splitter.split(df_feat, time_col="timestamp")

            # Extract features and targets
            feature_cols = WaveFeatureBuilder.FEATURE_COLUMNS
            x_train = split.train[feature_cols].values.astype(np.float32)
            y_state_train = split.train['target_state'].values.astype(np.int64)
            y_mag_train = split.train['target_magnitude'].values.astype(np.float32)
            y_dur_train = split.train['target_duration'].values.astype(np.float32)

            x_val = split.val[feature_cols].values.astype(np.float32)
            y_state_val = split.val['target_state'].values.astype(np.int64)
            y_mag_val = split.val['target_magnitude'].values.astype(np.float32)
            y_dur_val = split.val['target_duration'].values.astype(np.float32)

            x_test = split.test[feature_cols].values.astype(np.float32)
            y_state_test = split.test['target_state'].values.astype(np.int64)
            y_mag_test = split.test['target_magnitude'].values.astype(np.float32)
            y_dur_test = split.test['target_duration'].values.astype(np.float32)

            # Fit scaler on training data if not provided
            if scaler is None and len(train_features) == 0:
                scaler = FeatureFactory.fit_scaler_train_only(split.train, feature_cols)
                _log(f"[TG-MNN] Fitted scaler on training data")

            # Apply scaler
            x_train = (x_train - scaler.mean) / scaler.std
            x_val = (x_val - scaler.mean) / scaler.std
            x_test = (x_test - scaler.mean) / scaler.std

            # Append to lists
            train_features.append(x_train)
            train_states.append(y_state_train)
            train_magnitudes.append(y_mag_train)
            train_durations.append(y_dur_train)

            val_features.append(x_val)
            val_states.append(y_state_val)
            val_magnitudes.append(y_mag_val)
            val_durations.append(y_dur_val)

            test_features.append(x_test)
            test_states.append(y_state_test)
            test_magnitudes.append(y_mag_test)
            test_durations.append(y_dur_test)

            _log(f"[TG-MNN] {symbol}: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

        except Exception as e:
            _log(f"[TG-MNN] Error processing {symbol}: {e}")
            continue

    if not train_features:
        raise ValueError("No training data available")

    _log(f"[TG-MNN] Total train windows: {sum(max(0, len(x) - seq_len + 1) for x in train_features)}")
    _log(f"[TG-MNN] Total val windows: {sum(max(0, len(x) - seq_len + 1) for x in val_features)}")
    _log(f"[TG-MNN] Total test windows: {sum(max(0, len(x) - seq_len + 1) for x in test_features)}")

    # Create datasets
    input_dim = train_features[0].shape[1]

    train_ds = WaveDataset(train_features, train_states, train_magnitudes, train_durations, seq_len)
    val_ds = WaveDataset(val_features, val_states, val_magnitudes, val_durations, seq_len)
    test_ds = WaveDataset(test_features, test_states, test_magnitudes, test_durations, seq_len)

    return TGMNNDatasets(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        input_dim=input_dim,
        scaler_stats=scaler,
    )
