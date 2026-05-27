"""APV-PLN dataset builder — Phase 4.

Dual-stream price + volume rolling-window dataset for the Adaptive
Price-Volume Probabilistic Learner Network.

Enforces the Iron-Wall chronological split (70/15/15, no random shuffle)
via IronWallSplitter and isolates Oracle future-path data so it is only
accessible during training (LUPI principle).

Batch layout
------------
train    →  4-tuple  (x_price, x_volume, y_bin, x_oracle)
val/test →  3-tuple  (x_price, x_volume, y_bin)

Tensor shapes
-------------
x_price   : [seq_len, PRICE_DIM=5]
x_volume  : [seq_len, VOLUME_DIM=5]
y_bin     : scalar int64  —  bin index in [0, num_bins-1]
x_oracle  : [horizon, ORACLE_DIM=2]  — future (log_return, log_volume)
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_pipeline.config import PipelineConfig
from data_pipeline.features import FeatureFactory
from data_pipeline.quality_gate import DataQualityGate
from data_pipeline.splitter import IronWallSplitter


# ─────────────────────────────────────────────────────────────────────────────
# Feature column definitions
# ─────────────────────────────────────────────────────────────────────────────

PRICE_FEATURES: list[str] = [
    "log_return",        # log(close_t / close_{t-1})
    "zscore_close_64",   # rolling z-score of close, window=64
    "ema_spread",        # ema(12) - ema(26)
    "atr_14",            # average true range, window=14
    "price_slope_20",    # (close - close.shift(20)) / 20
]

VOLUME_FEATURES: list[str] = [
    "log_volume",        # log1p(volume)
    "volume_zscore_64",  # rolling z-score of log_volume, window=64
    "taker_buy_ratio",   # taker_buy_base / volume
    "vwap_deviation",    # (close - vwap) / (atr_14 + eps)
    "vol_imbalance",     # (vol_up - vol_dn) / (vol_up + vol_dn + eps)
]

# Oracle features are the same two columns evaluated on FUTURE bars.
# The same column names exist in the DataFrame; we slice them at future positions.
ORACLE_FEATURES: list[str] = ["log_return", "log_volume"]

PRICE_DIM: int = len(PRICE_FEATURES)
VOLUME_DIM: int = len(VOLUME_FEATURES)
ORACLE_DIM: int = len(ORACLE_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_apvpln_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build all PRICE + VOLUME features from raw OHLCV data.

    All computations are strictly causal (feature at t uses only t-1 or earlier).
    """
    df = raw.copy().sort_values("timestamp").reset_index(drop=True)
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    vol = df["volume"].astype(float)
    qvol = df["quote_volume"].astype(float)
    taker_buy_base = df["taker_buy_base"].astype(float)

    # ── Price features ────────────────────────────────────────────────────────
    df["log_return"] = FeatureFactory.log_return(close, use_torch_cuda=False)
    df["zscore_close_64"] = FeatureFactory.rolling_zscore(close, window=64)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df["ema_spread"] = ema_fast - ema_slow
    df["atr_14"] = FeatureFactory.atr(df, window=14)
    df["price_slope_20"] = (close - close.shift(20)) / 20.0

    # ── Volume features ───────────────────────────────────────────────────────
    df["log_volume"] = np.log1p(vol)
    df["volume_zscore_64"] = FeatureFactory.rolling_zscore(df["log_volume"], window=64)
    df["taker_buy_ratio"] = taker_buy_base / (vol + 1e-8)

    atr = df["atr_14"]
    vwap = qvol / (vol.replace(0.0, np.nan) + 1e-8)
    df["vwap_deviation"] = (close - vwap) / (atr + 1e-8)

    vol_up = vol.where(close >= open_, 0.0).fillna(0.0)
    vol_dn = vol.where(close < open_, 0.0).fillna(0.0)
    df["vol_imbalance"] = (vol_up - vol_dn) / (vol_up + vol_dn + 1e-8)

    return df


def _load_symbol(path: Path) -> pd.DataFrame:
    cols = [
        "timestamp", "open", "high", "low", "close",
        "volume", "quote_volume", "taker_buy_base", "taker_buy_quote",
    ]
    df = pd.read_parquet(path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Bin utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_bin_bounds(returns: np.ndarray, clip_pct: float = 0.5) -> tuple[float, float]:
    """Return (bin_min, bin_max) from training set returns using percentile clipping."""
    valid = returns[np.isfinite(returns)]
    lo = float(np.percentile(valid, clip_pct))
    hi = float(np.percentile(valid, 100.0 - clip_pct))
    return lo, hi


def return_to_bin(r: float, bin_min: float, bin_max: float, num_bins: int) -> int:
    """Quantise a scalar return to a bin index in [0, num_bins-1]."""
    if bin_max <= bin_min:
        return num_bins // 2
    idx = int((r - bin_min) / (bin_max - bin_min) * num_bins)
    return max(0, min(num_bins - 1, idx))


def make_bin_centers(bin_min: float, bin_max: float, num_bins: int) -> np.ndarray:
    """Return centre value of each bin as a 1-D float32 array."""
    edges = np.linspace(bin_min, bin_max, num_bins + 1, dtype=np.float64)
    return ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# APVPLNDataset
# ─────────────────────────────────────────────────────────────────────────────

class APVPLNDataset(Dataset):
    """Rolling-window dataset with optional Oracle future path.

    Parameters
    ----------
    price_list, volume_list : list of (N_i, dim) float32 arrays
        One array per symbol; rows are time-sorted.
    oracle_list : list of (N_i, ORACLE_DIM) float32 arrays | None
        Oracle features for each bar (log_return, log_volume after scaling).
        Must be provided when include_oracle=True.
    return_list : list of (N_i,) float32 arrays
        H-bar forward returns for each bar; NaN at last `horizon` rows.
    seq_len : int       Student look-back length.
    horizon : int       Prediction horizon (= oracle length).
    num_bins : int      Number of probability bins.
    bin_min, bin_max :  Bin boundary scalars (computed from train split).
    include_oracle : bool
        True  → train mode, returns 4-tuple.
        False → val/test mode, returns 3-tuple.
    """

    def __init__(
        self,
        price_list: list[np.ndarray],
        volume_list: list[np.ndarray],
        return_list: list[np.ndarray],
        oracle_list: list[np.ndarray] | None,
        seq_len: int,
        horizon: int,
        num_bins: int,
        bin_min: float,
        bin_max: float,
        include_oracle: bool = False,
    ) -> None:
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_bins = num_bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.include_oracle = include_oracle

        self.price_list = [
            torch.from_numpy(np.array(x, dtype=np.float32, copy=True)) for x in price_list
        ]
        self.volume_list = [
            torch.from_numpy(np.array(x, dtype=np.float32, copy=True)) for x in volume_list
        ]
        self.return_list = [
            torch.from_numpy(np.array(r, dtype=np.float32, copy=True)) for r in return_list
        ]
        self.oracle_list = (
            [torch.from_numpy(np.array(o, dtype=np.float32, copy=True)) for o in oracle_list]
            if oracle_list is not None
            else None
        )

        # Valid windows require: student window [i, i+seq_len) + oracle [i+seq_len, i+seq_len+horizon)
        # ⟹ last valid i = N - seq_len - horizon   (ensures target is not NaN)
        self.lengths = [
            max(0, len(x) - seq_len - horizon + 1) for x in self.price_list
        ]
        self.cum = np.cumsum(self.lengths).tolist()

    def __len__(self) -> int:
        return int(self.cum[-1]) if self.cum else 0

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        s = bisect.bisect_right(self.cum, idx)
        prev = 0 if s == 0 else self.cum[s - 1]
        local_i = idx - prev

        x_price = self.price_list[s][local_i: local_i + self.seq_len]
        x_volume = self.volume_list[s][local_i: local_i + self.seq_len]

        # Target: H-bar forward return at last bar of student window, quantised to bin
        raw_return = self.return_list[s][local_i + self.seq_len - 1].item()
        y_bin = torch.tensor(
            return_to_bin(raw_return, self.bin_min, self.bin_max, self.num_bins),
            dtype=torch.long,
        )

        if self.include_oracle:
            assert self.oracle_list is not None, "oracle_list must be provided when include_oracle=True"
            # Oracle sees future bars: positions [t+1, t+horizon] where t = local_i + seq_len - 1
            x_oracle = self.oracle_list[s][
                local_i + self.seq_len: local_i + self.seq_len + self.horizon
            ]
            return x_price, x_volume, y_bin, x_oracle

        return x_price, x_volume, y_bin


# ─────────────────────────────────────────────────────────────────────────────
# Dataset container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class APVPLNDatasets:
    train: APVPLNDataset    # 4-tuple (includes oracle)
    val: APVPLNDataset      # 3-tuple (student only)
    test: APVPLNDataset     # 3-tuple (student only)
    price_dim: int
    volume_dim: int
    oracle_dim: int
    num_bins: int
    bin_min: float
    bin_max: float
    bin_centers: np.ndarray  # shape [num_bins]


# ─────────────────────────────────────────────────────────────────────────────
# Build function
# ─────────────────────────────────────────────────────────────────────────────

def _log(message: str) -> None:
    print(message, flush=True)


def build_apvpln_datasets(config: dict[str, Any]) -> APVPLNDatasets:
    """Load symbols, build features, split chronologically, return datasets.

    Iron-Wall guarantees
    --------------------
    1. Chronological split 70/15/15 via IronWallSplitter (no random shuffle).
    2. Scaler is fit ONLY on train split; applied to val/test.
    3. bin_min / bin_max computed from train forward-returns only.
    4. Oracle arrays built from train features; not exposed in val/test datasets.
    """
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
        raise RuntimeError("No accepted symbols available for APV-PLN training")

    seq_len = int(config["seq_len"])
    horizon = int(config["horizon"])
    num_bins = int(config.get("num_bins", 51))
    cap_rows = int(config.get("max_rows_per_symbol", 0))
    splitter = IronWallSplitter(purge_gap_bars=int(config["purge_gap_bars"]))

    all_feature_cols = PRICE_FEATURES + VOLUME_FEATURES

    _log(
        f"[apv-data] symbols={len(symbols)} seq_len={seq_len} horizon={horizon} "
        f"num_bins={num_bins} cap_rows={cap_rows}"
    )

    # ── Accumulate per-symbol arrays ──────────────────────────────────────────
    train_price: list[np.ndarray] = []
    train_volume: list[np.ndarray] = []
    train_return: list[np.ndarray] = []   # includes trailing NaN → length formula guards it
    train_oracle: list[np.ndarray] = []

    val_price: list[np.ndarray] = []
    val_volume: list[np.ndarray] = []
    val_return: list[np.ndarray] = []

    test_price: list[np.ndarray] = []
    test_volume: list[np.ndarray] = []
    test_return: list[np.ndarray] = []

    all_train_returns: list[np.ndarray] = []  # for bin bound computation

    for sym_idx, symbol in enumerate(symbols, start=1):
        path = pipe_cfg.dataset_dir / f"{symbol}.parquet"
        _log(f"[apv-data] loading {sym_idx}/{len(symbols)} symbol={symbol}")
        raw = _load_symbol(path)

        feat = _build_apvpln_features(raw)

        # Drop rows where any feature is NaN (leading rows from rolling windows).
        feat = feat.dropna(subset=all_feature_cols).reset_index(drop=True)

        if cap_rows > 0:
            feat = feat.iloc[-cap_rows:].copy()

        if len(feat) < seq_len + horizon + 10:
            _log(f"[apv-data] skip symbol={symbol} rows={len(feat)} (too few after dropna)")
            continue

        # Forward return — NaN at last `horizon` positions (intentionally kept).
        # APVPLNDataset length formula ensures we never sample those NaN rows as targets.
        fwd_return = (feat["close"].shift(-horizon) / feat["close"]) - 1.0
        feat["fwd_return"] = fwd_return.astype(np.float32)

        # Chronological split (on feature-complete rows only)
        split = splitter.split(feat, time_col="timestamp")

        # Scaler fit on TRAIN features only
        scaler = FeatureFactory.fit_scaler_train_only(split.train, all_feature_cols)
        tr_scaled = FeatureFactory.transform_with_scaler(split.train, scaler)
        va_scaled = FeatureFactory.transform_with_scaler(split.val, scaler)
        te_scaled = FeatureFactory.transform_with_scaler(split.test, scaler)

        # Extract arrays — PRICE and VOLUME features (scaled)
        x_tr_p = tr_scaled[PRICE_FEATURES].to_numpy(dtype=np.float32)
        x_tr_v = tr_scaled[VOLUME_FEATURES].to_numpy(dtype=np.float32)
        r_tr = split.train["fwd_return"].to_numpy(dtype=np.float32)
        # Oracle: future (log_return, log_volume) per bar — also scaled
        o_tr = tr_scaled[ORACLE_FEATURES].to_numpy(dtype=np.float32)

        x_va_p = va_scaled[PRICE_FEATURES].to_numpy(dtype=np.float32)
        x_va_v = va_scaled[VOLUME_FEATURES].to_numpy(dtype=np.float32)
        r_va = split.val["fwd_return"].to_numpy(dtype=np.float32)

        x_te_p = te_scaled[PRICE_FEATURES].to_numpy(dtype=np.float32)
        x_te_v = te_scaled[VOLUME_FEATURES].to_numpy(dtype=np.float32)
        r_te = split.test["fwd_return"].to_numpy(dtype=np.float32)

        # Collect valid train returns for bin bound computation
        # (exclude trailing NaN positions)
        n_tr = len(x_tr_p)
        valid_tr_returns = r_tr[: max(0, n_tr - horizon)]
        valid_tr_returns = valid_tr_returns[np.isfinite(valid_tr_returns)]
        if len(valid_tr_returns) > 0:
            all_train_returns.append(valid_tr_returns)

        # Minimum viable windows check
        n_tr_windows = max(0, n_tr - seq_len - horizon + 1)
        n_va_windows = max(0, len(x_va_p) - seq_len - horizon + 1)
        n_te_windows = max(0, len(x_te_p) - seq_len - horizon + 1)

        if n_tr_windows < 1 or n_va_windows < 1 or n_te_windows < 1:
            _log(
                f"[apv-data] skip symbol={symbol} "
                f"tr_win={n_tr_windows} va_win={n_va_windows} te_win={n_te_windows}"
            )
            continue

        train_price.append(x_tr_p)
        train_volume.append(x_tr_v)
        train_return.append(r_tr)
        train_oracle.append(o_tr)

        val_price.append(x_va_p)
        val_volume.append(x_va_v)
        val_return.append(r_va)

        test_price.append(x_te_p)
        test_volume.append(x_te_v)
        test_return.append(r_te)

        _log(
            f"[apv-data] ready symbol={symbol} tr_win={n_tr_windows} "
            f"va_win={n_va_windows} te_win={n_te_windows}"
        )

    if not train_price:
        raise RuntimeError("APV-PLN: no symbols produced valid windows")

    # ── Compute bin bounds from train returns only (Iron-Wall) ────────────────
    all_tr = np.concatenate(all_train_returns)
    bin_min, bin_max = compute_bin_bounds(all_tr, clip_pct=0.5)
    bin_centers = make_bin_centers(bin_min, bin_max, num_bins)

    _log(
        f"[apv-data] bin_bounds: bin_min={bin_min:.6f} bin_max={bin_max:.6f} "
        f"num_bins={num_bins} train_return_samples={len(all_tr)}"
    )

    # ── Build datasets ────────────────────────────────────────────────────────
    common = dict(
        seq_len=seq_len, horizon=horizon,
        num_bins=num_bins, bin_min=bin_min, bin_max=bin_max,
    )
    train_ds = APVPLNDataset(
        train_price, train_volume, train_return, train_oracle,
        include_oracle=True, **common,
    )
    val_ds = APVPLNDataset(
        val_price, val_volume, val_return, oracle_list=None,
        include_oracle=False, **common,
    )
    test_ds = APVPLNDataset(
        test_price, test_volume, test_return, oracle_list=None,
        include_oracle=False, **common,
    )

    _log(
        f"[apv-data] datasets complete "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )

    return APVPLNDatasets(
        train=train_ds, val=val_ds, test=test_ds,
        price_dim=PRICE_DIM, volume_dim=VOLUME_DIM, oracle_dim=ORACLE_DIM,
        num_bins=num_bins, bin_min=bin_min, bin_max=bin_max,
        bin_centers=bin_centers,
    )
