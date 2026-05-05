#!/usr/bin/env python3
"""
evaluate_all_checkpoints.py
============================
True OOS hardware evaluation of all Phase-4 model checkpoints.

- Traverses models/checkpoints/ for all .pt files.
- Loads each architecture with saved weights via torch.load.
- Runs a real forward pass on the OOS test split of binance_historical data.
- Computes: Directional Accuracy, Sharpe Ratio, Profit Factor, Max Drawdown.
- Rebuilds model_registry.json from computed numbers only (no hallucinated N/A).
- Assigns PASSED / RESUME_TRAINING_REQUIRED per strict KPI gate.

Hardware target: AMD Radeon RX 6750 via torch_directml backend.

Verified checkpoint dims (from torch.load introspection):
  Trend LSTM:          cells.0.W_i.weight [128, 5]  → input_dim=5
  Scalper CNN:         stem.weight [64, 13, 1]       → input_dim=13
  StatArb Autoencoder: encoder.cells.0.W_r [32, 2]  → num_assets=2
  StatArb LSTM:        cells.0.W_i [64, 2]           → num_assets=2
  StatArb GAT:         node_proj [32, 1]             → num_assets=2 (projects 1 per asset)
"""
from __future__ import annotations

import importlib
import json
import sys
import time
import gc
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from quant_core.validation_policy import PRODUCTION_GATES, passes_production_gates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─── GPU backend ─────────────────────────────────────────────────────────────
try:
    import torch_directml
    DEVICE = torch_directml.device()
    DEVICE_NAME = "directml"
    print("[eval] GPU backend: torch_directml  (AMD Radeon RX 6750)", flush=True)
except Exception as _e:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "cpu_fallback"
    print(f"[eval] WARNING: DirectML unavailable ({_e}), falling back to CPU", flush=True)

# ─── Constants ───────────────────────────────────────────────────────────────
REGISTRY_PATH = ROOT / "model_registry.json"
CHECKPOINT_ROOT = ROOT / "models" / "checkpoints"
DATASET_DIR = ROOT / "Dataset" / "binance_historical"
BATCH_SIZE = 1024
MAX_EVAL_SAMPLES = 20_000   # cap per archetype: statistically valid & fast
INFER_DEVICE = DEVICE
PERIODS_PER_YEAR = 252

# All parquet columns available (verified on disk)
PARQUET_COLS = ["timestamp", "open", "high", "low", "close", "volume",
                "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]

# Exact feature columns as defined in training data modules
TREND_FEAT_COLS = ["log_return", "zscore_close_64", "ema_spread", "atr_14", "price_slope_20"]  # 5
MR_FEAT_COLS    = ["vwap_dev", "bb_distance", "zscore_close_20", "rsi_14", "rsi_div_5"]         # 5
# Scalper uses 13 features — imported from quant_core.scalper_data.SCALPER_FEATURES


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═════════════════════════════════════════════════════════════════════════════

def _banner(msg: str) -> None:
    sep = "=" * 80
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def _log(msg: str) -> None:
    # Windows cp1252 consoles can fail on symbols like "→" / "✓".
    # Emit a safe representation rather than crashing evaluation.
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_msg = msg.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe_msg, flush=True)


def _release_memory(*objects: Any) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def directional_accuracy(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    return float(np.mean(pred_labels == true_labels))


def sharpe_ratio(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns))
    if std < 1e-10:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(PERIODS_PER_YEAR))


def profit_factor(returns: np.ndarray) -> float:
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    if losses < 1e-10:
        return float("nan")
    return gains / losses


def max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    eq = np.cumprod(1.0 + np.clip(returns, -0.99, 1.0))
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / (peak + 1e-10)
    return float(np.max(dd))


ROUND_TRIP_COST = 0.001  # 0.1% per completed round-trip (maker+taker, crypto futures)


def compute_all_metrics(
    logits: np.ndarray,
    true_labels: np.ndarray,
    output_type: str,
    actual_returns: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all four KPI metrics from raw logits/predictions.

    When ``actual_returns`` is provided the PnL uses the real forward return
    magnitude (minus round-trip cost) instead of a fixed 0.1% scale.  This
    gives a much more realistic Profit Factor for models with directional edge.
    """
    # ── Directional accuracy ──────────────────────────────────────────────
    if output_type == "binary":
        pred_dir = (logits.ravel() > 0).astype(int)
        true_dir = true_labels.astype(int)
        direction = pred_dir  # 1=up, 0=down
        true_direction = true_dir
        ret_sign = np.where(pred_dir == 1, 1.0, -1.0) * np.where(true_dir == 1, 1.0, -1.0)
    elif output_type == "multiclass3":
        pred_class = np.argmax(logits, axis=1)
        true_class = true_labels.astype(int)
        direction = pred_class
        true_direction = true_class
        # return: +1 if correct (not flat), -1 if wrong, 0 if flat
        ret_sign = np.where(pred_class == true_class, 1.0, -1.0)
        ret_sign = np.where(pred_class == 1, 0.0, ret_sign)  # flat predictions = 0
    elif output_type == "regression":
        # StatArb: predict sign of next-step spread
        pred_dir = (logits.ravel() > 0).astype(int)
        true_dir = (true_labels > 0).astype(int)
        direction = pred_dir
        true_direction = true_dir
        ret_sign = np.where(pred_dir == true_dir, 1.0, -1.0)
    elif output_type in ("rl_continuous", "rl_discrete"):
        # RL: action maps to direction
        if output_type == "rl_discrete":
            pred_action = np.argmax(logits, axis=1)  # 0=tight, 1=medium, 2=wide
            direction = (pred_action >= 1).astype(int)  # treat medium/wide as "active"
        else:
            direction = (logits[:, 0] > 0.5).astype(int)  # bid offset > 0.5 = buy lean
        true_direction = true_labels.astype(int)
        ret_sign = np.where(direction == true_direction, 1.0, -1.0)
    else:
        direction = (logits.ravel() > 0).astype(int)
        true_direction = true_labels.astype(int)
        ret_sign = np.where(direction == true_direction, 1.0, -1.0)

    dir_acc = directional_accuracy(direction, true_direction)

    # ── PnL using actual forward returns when available ───────────────────
    if actual_returns is not None:
        abs_ret = np.abs(actual_returns).astype(np.float64)
        if output_type == "multiclass3":
            # flat predictions = no trade, no transaction cost
            trade_mask = (ret_sign != 0).astype(np.float64)
            returns = ret_sign * abs_ret - trade_mask * ROUND_TRIP_COST
        else:
            returns = ret_sign * abs_ret - ROUND_TRIP_COST
    else:
        returns = ret_sign * 0.001  # fallback: fixed 0.1% scale

    return {
        "directional_accuracy": round(dir_acc, 6),
        "sharpe": round(sharpe_ratio(returns), 6),
        "profit_factor": round(profit_factor(returns), 6),
        "max_drawdown": round(max_drawdown(returns), 6),
    }


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═════════════════════════════════════════════════════════════════════════════

def _load_parquets(max_files: int = 40) -> list[pd.DataFrame]:
    """Load a representative sample of binance_historical parquets (all columns)."""
    files = sorted(DATASET_DIR.glob("*.parquet"))[:max_files]
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_parquet(f)  # load all columns — quote_volume needed for MR/Scalper
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            if len(df) > 2000:
                frames.append(df)
        except Exception as e:
            _log(f"[eval] skip {f.name}: {e}")
    return frames


def _test_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Return last 15% as OOS test set (Iron Wall protocol)."""
    n = len(df)
    return df.iloc[int(n * 0.85):].reset_index(drop=True)


def _stride_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Vectorised sliding window using stride tricks. arr: (N, F) → (N-seq_len, seq_len, F)"""
    N, F = arr.shape
    if N <= seq_len:
        return np.empty((0, seq_len, F), dtype=arr.dtype)
    shape = (N - seq_len, seq_len, F)
    strides = (arr.strides[0], arr.strides[0], arr.strides[1])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides).copy()


def load_trend_data(frames: list[pd.DataFrame], seq_len: int = 96, horizon: int = 20) -> TensorDataset | None:
    """Build sequential OOS test data for Trend models. Input: [B, seq_len, 5]
    Returns 3-tensor dataset: (X, y_direction, actual_forward_return)."""
    all_X, all_y, all_ret = [], [], []
    for df in frames:
        try:
            from data_pipeline.features import FeatureFactory
            feat = FeatureFactory.build_trend_features(df)
            available = [c for c in TREND_FEAT_COLS if c in feat.columns]
            if len(available) < 3:
                continue
            close = feat["close"].to_numpy(np.float32)
            # Binary direction label: 1 if close[t+horizon] > close[t]
            future_close = np.roll(close, -horizon)
            target = (future_close > close).astype(np.float32)
            target[-horizon:] = 0  # mask invalid tail
            # Actual forward return (signed fraction)
            fwd_ret = (future_close - close) / (np.abs(close) + 1e-8)
            fwd_ret[-horizon:] = 0.0
            feat_arr = feat[available].to_numpy(np.float32)
            n = len(feat_arr)
            test_start = int(n * 0.85)
            # Use stride tricks on the test portion
            feat_test = feat_arr[test_start:]
            tgt_test  = target[test_start:]
            ret_test  = fwd_ret[test_start:]
            seqs = _stride_sequences(feat_test, seq_len)   # (M, seq_len, F)
            # label: direction at seq_end (already shifted above)
            labels = tgt_test[seq_len - 1: seq_len - 1 + len(seqs)]
            rets   = ret_test[seq_len - 1: seq_len - 1 + len(seqs)]
            # Drop last `horizon` samples (target is meaningless there)
            valid = len(seqs) - horizon
            if valid <= 0:
                continue
            all_X.append(seqs[:valid])
            all_y.append(labels[:valid])
            all_ret.append(rets[:valid])
        except Exception as e:
            _log(f"[eval] trend data build warning: {e}")
    if not all_X:
        return None
    X_t = torch.tensor(np.concatenate(all_X), dtype=torch.float32)
    y_t = torch.tensor(np.concatenate(all_y), dtype=torch.float32)
    r_t = torch.tensor(np.concatenate(all_ret), dtype=torch.float32)
    return TensorDataset(X_t, y_t, r_t)


def load_mr_data(frames: list[pd.DataFrame], horizon: int = 20) -> TensorDataset | None:
    """Build tabular OOS test data for Mean Reversion models. Input: [B, 5]
    Returns 3-tensor dataset: (X, y_direction, actual_forward_return)."""
    all_X, all_y, all_ret = [], [], []
    for df in frames:
        try:
            from data_pipeline.features import FeatureFactory
            feat = FeatureFactory.build_mean_reversion_features(df)
            available = [c for c in MR_FEAT_COLS if c in feat.columns]
            if len(available) < 3:
                continue
            close = feat["close"].to_numpy(np.float32)
            future_close = np.roll(close, -horizon)
            target = (future_close > close).astype(np.float32)
            target[-horizon:] = np.nan
            fwd_ret = (future_close - close) / (np.abs(close) + 1e-8)
            fwd_ret[-horizon:] = np.nan
            feat = feat.assign(_target=target, _fwd_ret=fwd_ret).dropna(subset=["_target"])
            n = len(feat)
            test = feat.iloc[int(n * 0.85):]
            all_X.append(test[available].to_numpy(np.float32))
            all_y.append(test["_target"].to_numpy(np.float32))
            all_ret.append(test["_fwd_ret"].to_numpy(np.float32))
        except Exception as e:
            _log(f"[eval] mr data build warning: {e}")
    if not all_X:
        return None
    X_t = torch.tensor(np.concatenate(all_X), dtype=torch.float32)
    y_t = torch.tensor(np.concatenate(all_y), dtype=torch.float32)
    r_t = torch.tensor(np.concatenate(all_ret), dtype=torch.float32)
    return TensorDataset(X_t, y_t, r_t)


def load_scalper_data(frames: list[pd.DataFrame], seq_len: int = 32, horizon: int = 5) -> TensorDataset | None:
    """Build sequential OOS test data for Scalper models (13 features). Input: [B, seq_len, 13]

    Applies the saved StandardScaler from each model checkpoint so feature ranges match
    what the model was trained on. Without this normalisation, accuracy falls to ~10%.
    """
    FLAT_THR = 0.0003
    try:
        from quant_core.scalper_data import _build_scalper_features, SCALPER_FEATURES, load_scalper_scaler
        scalper_feat_fn = _build_scalper_features
        feat_cols = SCALPER_FEATURES
    except Exception as e:
        _log(f"[eval] scalper feature import failed: {e} — skipping scalper datasets")
        return None

    # Try to load a saved scaler from any scalper checkpoint directory
    _scaler = None
    for _ckpt_name in ["CNN_Scalper_v1", "LinearAttn_Scalper_v1", "GRU_Scalper_v1"]:
        _candidate = CHECKPOINT_ROOT / "scalper" / _ckpt_name
        _scaler = load_scalper_scaler(_candidate)
        if _scaler is not None:
            _log(f"[eval] scalper scaler loaded from {_candidate}")
            break
    if _scaler is None:
        _log("[eval] WARNING: no feature_scaler.pkl found for scalper — using fallback "
             "pre-test normalization per symbol.")

    all_X, all_y, all_ret = [], [], []
    for df in frames:
        try:
            feat = scalper_feat_fn(df)
            available = [c for c in feat_cols if c in feat.columns]
            if len(available) < 5:
                continue
            close = df["close"].to_numpy(np.float32)
            future_ret = np.roll(close, -horizon) / (close + 1e-10) - 1.0
            target = np.where(future_ret > FLAT_THR, 2, np.where(future_ret < -FLAT_THR, 0, 1)).astype(np.int64)
            target[-horizon:] = 1  # mark tail as flat (will be excluded)
            feat_arr = feat[available].to_numpy(np.float32)
            # Align lengths
            min_n = min(len(feat_arr), len(target))
            feat_arr = feat_arr[:min_n]
            target = target[:min_n]
            test_start = int(min_n * 0.85)

            # Apply scaler from checkpoint when available; otherwise fit a causal
            # fallback normalizer on pre-test rows only to avoid leakage.
            if _scaler is not None:
                try:
                    if hasattr(_scaler, "transform"):
                        feat_arr = _scaler.transform(feat_arr)
                    elif hasattr(_scaler, "mean") and hasattr(_scaler, "std"):
                        mean = np.asarray(_scaler.mean, dtype=np.float32)
                        std = np.asarray(_scaler.std, dtype=np.float32)
                        std = np.where(std < 1e-12, 1.0, std)
                        feat_arr = (feat_arr - mean) / std
                    else:
                        raise TypeError(f"unsupported scaler type: {type(_scaler).__name__}")
                except Exception as _e:
                    _log(f"[eval] scaler apply failed ({_e}) — falling back to pre-test normalization")
                    train_slice = feat_arr[:max(test_start, 1)]
                    mu = np.nanmean(train_slice, axis=0)
                    sd = np.nanstd(train_slice, axis=0)
                    sd = np.where(sd < 1e-12, 1.0, sd)
                    feat_arr = (feat_arr - mu) / sd
            else:
                train_slice = feat_arr[:max(test_start, 1)]
                mu = np.nanmean(train_slice, axis=0)
                sd = np.nanstd(train_slice, axis=0)
                sd = np.where(sd < 1e-12, 1.0, sd)
                feat_arr = (feat_arr - mu) / sd

            feat_test = feat_arr[test_start:]
            tgt_test  = target[test_start:]
            ret_test  = future_ret[:min_n][test_start:]
            seqs = _stride_sequences(feat_test, seq_len)
            labels = tgt_test[seq_len - 1: seq_len - 1 + len(seqs)]
            rets   = ret_test[seq_len - 1: seq_len - 1 + len(seqs)]
            valid = len(seqs) - horizon
            if valid <= 0:
                continue
            all_X.append(seqs[:valid])
            all_y.append(labels[:valid])
            all_ret.append(rets[:valid])
        except Exception as e:
            _log(f"[eval] scalper data build warning: {e}")
    if not all_X:
        return None
    X_t = torch.tensor(np.concatenate(all_X), dtype=torch.float32)
    y_t = torch.tensor(np.concatenate(all_y), dtype=torch.int64)
    r_t = torch.tensor(np.concatenate(all_ret), dtype=torch.float32)
    return TensorDataset(X_t, y_t, r_t)


def load_stat_arb_data(frames: list[pd.DataFrame], seq_len: int = 64, horizon: int = 10,
                       num_assets: int = 10) -> TensorDataset | None:
    """Build multi-asset sequences for StatArb models. Input: [B, seq_len, num_assets]
    num_assets is selected dynamically from checkpoint dims to avoid shape mismatches."""
    selected = frames[:num_assets]
    if len(selected) < 2:
        return None
    try:
        min_len = min(len(df) for df in selected)
        if min_len < seq_len + horizon + 10:
            return None
        returns = np.stack(
            [df["close"].iloc[:min_len].pct_change().fillna(0).to_numpy(np.float32)
             for df in selected],
            axis=1,
        )  # (min_len, num_assets)
        train_n = int(min_len * 0.85)
        mu = returns[:train_n].mean(0)
        std = returns[:train_n].std(0) + 1e-8
        z_scores = (returns - mu) / std   # (min_len, num_assets)
        # Test portion only
        test_z = z_scores[train_n:]
        seqs = _stride_sequences(test_z, seq_len)      # (M, seq_len, num_assets)
        # Target: mean of next `horizon` bars of first asset
        tgt_raw = test_z[:, 0]
        labels = np.array([tgt_raw[i + seq_len: i + seq_len + horizon].mean()
                           for i in range(len(seqs))], dtype=np.float32)
        valid = len(seqs) - horizon
        if valid <= 0:
            return None
        X_t = torch.tensor(seqs[:valid], dtype=torch.float32)
        y_t = torch.tensor(labels[:valid], dtype=torch.float32)
        # Use label z-score magnitude as actual return proxy for StatArb
        r_t = torch.tensor(labels[:valid], dtype=torch.float32)
        return TensorDataset(X_t, y_t, r_t)
    except Exception as e:
        _log(f"[eval] stat_arb data build warning: {e}")
        return None


def load_disc_data(frames: list[pd.DataFrame], seq_len: int = 32, horizon: int = 5) -> TensorDataset | None:
    """Rasterize OHLCV windows into 4×32×32 chart images for Disc models.

    Uses _rasterize_window() from quant_core.discretionary_data to guarantee
    IDENTICAL encoding to what models were trained on:
      Channel 0: normalised open  (relative to bar's H-L range)
      Channel 1: normalised high
      Channel 2: normalised low
      Channel 3: normalised close
    This replaces the previous vectorised renderer that used a completely different
    encoding (close price line / volume bars / log-return sparkline).
    """
    FLAT_THR = 0.001
    try:
        from quant_core.discretionary_data import _rasterize_window
    except Exception as e:
        _log(f"[eval] disc renderer import failed: {e} — skipping disc datasets")
        return None

    all_imgs, all_y, all_ret = [], [], []

    for df in frames[:3]:  # limit to 3 symbols for speed
        try:
            n = len(df)
            test_start = int(n * 0.85)
            close = df["close"].to_numpy(np.float32)
            open_ = df["open"].to_numpy(np.float32)
            high  = df["high"].to_numpy(np.float32)
            low   = df["low"].to_numpy(np.float32)

            # 3-class target (same convention as training: 0=down, 1=flat, 2=up)
            future_ret = np.roll(close, -horizon) / (close + 1e-10) - 1.0
            targets = np.where(future_ret > FLAT_THR, 2,
                      np.where(future_ret < -FLAT_THR, 0, 1)).astype(np.int64)

            ohlcv = np.stack([open_, high, low, close], axis=1)  # (N, 4)

            # Test portion windows, capped at 5000
            i0 = test_start
            i1 = n - seq_len - horizon + 1
            if i1 <= i0:
                continue
            i0_eff = max(i0, i1 - 5_000)

            imgs_list, tgt_list, ret_list = [], [], []
            for i in range(i0_eff, i1):
                window = ohlcv[i: i + seq_len]          # (seq_len, 4)
                img = _rasterize_window(window)          # (4, H, W)
                imgs_list.append(img)
                tgt_list.append(targets[i + seq_len - 1])
                ret_list.append(float(future_ret[i + seq_len - 1]))

            if not imgs_list:
                continue
            all_imgs.append(np.stack(imgs_list))        # (M, 4, H, W)
            all_y.append(np.array(tgt_list, dtype=np.int64))
            all_ret.append(np.array(ret_list, dtype=np.float32))
        except Exception as e:
            _log(f"[eval] disc data build warning: {e}")

    if not all_imgs:
        return None
    X_t = torch.tensor(np.concatenate(all_imgs), dtype=torch.float32)
    y_t = torch.tensor(np.concatenate(all_y), dtype=torch.int64)
    r_t = torch.tensor(np.concatenate(all_ret), dtype=torch.float32)
    return TensorDataset(X_t, y_t, r_t)


def load_mm_data(frames: list[pd.DataFrame], n_steps: int = 3000, state_dim: int = 10) -> TensorDataset | None:
    """Build market-maker state vectors from last n_steps test bars.

    The evaluator can slice to model-required dimensions (7/8/10) while keeping
    a consistent base state construction.
    """
    all_X, all_y, all_ret = [], [], []
    for df in frames[:3]:
        try:
            n = len(df)
            test_start = int(n * 0.85)
            test = df.iloc[test_start:].reset_index(drop=True)
            close = test["close"].to_numpy(np.float32)[-n_steps:]
            volume = test["volume"].to_numpy(np.float32)[-n_steps:]
            nb = len(close)
            if nb < 50:
                continue
            returns = np.concatenate([[0.0], np.diff(np.log(close + 1e-10))]).astype(np.float32)
            vol = pd.Series(returns).rolling(20, min_periods=1).std().fillna(0.01).to_numpy(np.float32)
            vwap_num = pd.Series(close * volume).rolling(20, min_periods=1).sum().to_numpy(np.float32)
            vwap_den = pd.Series(volume).rolling(20, min_periods=1).sum().to_numpy(np.float32)
            vwap = vwap_num / (vwap_den + 1e-10)
            spread = vol * 0.5
            vwap_dev = (close - vwap) / (vwap + 1e-10)
            vol_mean = float(volume.mean()) + 1e-10
            base_states = np.column_stack([
                np.zeros(nb, dtype=np.float32),                                # inventory_norm
                returns,                                                         # mid_change
                spread,                                                          # spread
                vol,                                                             # volatility
                (volume / vol_mean - 1.0).astype(np.float32),                   # ofi_proxy
                np.linspace(0.0, 1.0, nb, dtype=np.float32),                    # time progress
                np.cumsum(returns).astype(np.float32),                           # pnl_norm proxy
                np.zeros(nb, dtype=np.float32),                                  # inv_skew
                np.zeros(nb, dtype=np.float32),                                  # funding_rate
                np.zeros(nb, dtype=np.float32),                                  # oi_norm
            ])
            states = base_states[:, : int(max(1, min(state_dim, base_states.shape[1])))]
            y = (returns > 0).astype(np.float32)
            all_X.append(states)
            all_y.append(y)
            all_ret.append(returns)  # actual log-price returns as PnL proxy
        except Exception as e:
            _log(f"[eval] mm data build warning: {e}")
    if not all_X:
        return None
    X_t = torch.tensor(np.concatenate(all_X), dtype=torch.float32)
    y_t = torch.tensor(np.concatenate(all_y), dtype=torch.float32)
    r_t = torch.tensor(np.concatenate(all_ret), dtype=torch.float32)
    return TensorDataset(X_t, y_t, r_t)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY: architecture / checkpoint mapping
# ═════════════════════════════════════════════════════════════════════════════

# Each entry: checkpoint_path (relative to CHECKPOINT_ROOT), module, cls, init_kwargs, data_type, output_type
MODEL_MANIFEST: dict[str, dict[str, Any]] = {
    "LSTM_Trend_v1": {
        "ckpt": "trend/LSTM_Trend_v1/model_best.pt",
        "module": "quant_core.trend_models", "cls": "TrendLSTMModel",
        "kwargs": {"input_dim": 5, "hidden_size": 128, "num_layers": 3, "dropout": 0.0},
        "data": "trend", "out": "binary", "archetype": "trend_follower",
    },
    "Transformer_Trend_v1": {
        "ckpt": "trend/Transformer_Trend_v1/model_best.pt",
        "module": "quant_core.trend_models", "cls": "TrendTransformerModel",
        "kwargs": {"input_dim": 5, "seq_len": 64, "d_model": 128, "nhead": 4, "num_layers": 2, "dropout": 0.0},
        "data": "trend", "out": "binary", "archetype": "trend_follower",
    },
    "TCN_Trend_v1": {
        "ckpt": "trend_verify/TCN_Trend_v1/model_best.pt",
        "module": "quant_core.trend_models", "cls": "TrendTCNModel",
        "kwargs": {"input_dim": 5, "channels": 128, "dropout": 0.0},
        "data": "trend", "out": "binary", "archetype": "trend_follower",
    },
    "MLP_MR_v1": {
        "ckpt": "mean_reversion/MLP_MR_v1/model_best.pt",
        "module": "quant_core.mean_reversion_models", "cls": "MeanReversionMLP",
        "kwargs": {"input_dim": 5, "hidden_size": 256, "num_layers": 4, "dropout": 0.0},
        "data": "mr", "out": "binary", "archetype": "mean_reversion",
    },
    "ResNet_MR_v1": {
        "ckpt": "mean_reversion/ResNet_MR_v1/model_best.pt",
        "module": "quant_core.mean_reversion_models", "cls": "MeanReversionResNet",
        "kwargs": {"input_dim": 5, "hidden_size": 256, "depth": 6, "dropout": 0.0},
        "data": "mr", "out": "binary", "archetype": "mean_reversion",
    },
    "GRN_MR_v1": {
        "ckpt": "mean_reversion/GRN_MR_v1/model_best.pt",
        "module": "quant_core.mean_reversion_models", "cls": "MeanReversionGRN",
        "kwargs": {"input_dim": 5, "hidden_size": 128, "depth": 4, "dropout": 0.0},
        "data": "mr", "out": "binary", "archetype": "mean_reversion",
    },
    "CNN_Scalper_v1": {
        "ckpt": "scalper/CNN_Scalper_v1/model_best.pt",
        "module": "quant_core.scalper_models", "cls": "ScalperCNN",
        "kwargs": {"input_dim": 13, "channels": 64, "dropout": 0.0},
        "data": "scalper", "out": "multiclass3", "archetype": "scalping_microstructure",
    },
    "LinearAttn_Scalper_v1": {
        "ckpt": "scalper/LinearAttn_Scalper_v1/model_best.pt",
        "module": "quant_core.scalper_models", "cls": "ScalperLinearAttn",
        "kwargs": {"input_dim": 13, "d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.0},
        "data": "scalper", "out": "multiclass3", "archetype": "scalping_microstructure",
    },
    "GRU_Scalper_v1": {
        "ckpt": "scalper/GRU_Scalper_v1/model_best.pt",
        "module": "quant_core.scalper_models", "cls": "ScalperGRU",
        "kwargs": {"input_dim": 13, "hidden_size": 64, "num_layers": 2, "dropout": 0.0},
        "data": "scalper", "out": "multiclass3", "archetype": "scalping_microstructure",
    },
    "Autoencoder_StatArb_v1": {
        "ckpt": "stat_arb/Autoencoder_StatArb_v1/model_best.pt",
        "module": "quant_core.stat_arb_models", "cls": "StatArbAutoencoder",
        "kwargs": {"num_assets": 2, "latent_dim": 32, "seq_len": 64, "dropout": 0.0},
        "data": "stat_arb", "out": "regression", "archetype": "statistical_arbitrage",
    },
    "GAT_StatArb_v1": {
        "ckpt": "stat_arb/GAT_StatArb_v1/model_best.pt",
        "module": "quant_core.stat_arb_models", "cls": "StatArbGAT",
        "kwargs": {"num_assets": 2, "d_model": 32, "num_layers": 2, "dropout": 0.0},
        "data": "stat_arb", "out": "regression", "archetype": "statistical_arbitrage",
    },
    "LSTM_StatArb_v1": {
        "ckpt": "stat_arb/LSTM_StatArb_v1/model_best.pt",
        "module": "quant_core.stat_arb_models", "cls": "StatArbLSTM",
        "kwargs": {"num_assets": 2, "hidden_size": 64, "num_layers": 2, "dropout": 0.0},
        "data": "stat_arb", "out": "regression", "archetype": "statistical_arbitrage",
    },
    "ViT_Disc_v1": {
        "ckpt": "discretionary/ViT_Disc_v1/model_best.pt",
        "module": "quant_core.discretionary_models", "cls": "DiscretionaryViT",
        "kwargs": {"img_size": 32, "patch_size": 4, "in_chans": 4, "embed_dim": 64,
                   "num_layers": 4, "nhead": 4, "dropout": 0.0},
        "data": "disc", "out": "multiclass3", "archetype": "discretionary_multimodal",
    },
    "Multimodal_Disc_v1": {
        "ckpt": "discretionary/Multimodal_Disc_v1/model_best.pt",
        "module": "quant_core.discretionary_models", "cls": "DiscretionaryMultimodal",
        "kwargs": {"tab_input_dim": 5, "img_embed": 64, "tab_embed": 64, "dropout": 0.0},
        "data": "disc_multimodal", "out": "multiclass3", "archetype": "discretionary_multimodal",
    },
    "CNNChart_Disc_v1": {
        "ckpt": "discretionary/CNNChart_Disc_v1/model_best.pt",
        "module": "quant_core.discretionary_models", "cls": "DiscretionaryCNNChart",
        "kwargs": {"in_chans": 4, "channels": 32, "dropout": 0.0},
        "data": "disc", "out": "multiclass3", "archetype": "discretionary_multimodal",
    },
    "PPO_MM_v1": {
        "ckpt": "market_maker/PPO_MM_v1/PPO_MM_v1_best.pt",
        "module": "quant_core.market_maker_models", "cls": "PPOActorCritic",
        "kwargs": {"state_dim": 7, "action_dim": 2, "hidden": 256, "dropout": 0.0},
        "data": "mm", "out": "rl_continuous", "archetype": "market_making_rl",
    },
    "SAC_MM_v1": {
        "ckpt": "market_maker/SAC_MM_v1/SAC_MM_v1_best.pt",
        "module": "quant_core.market_maker_models", "cls": "SACActorNetwork",
        "kwargs": {"state_dim": 7, "action_dim": 2, "hidden": 256, "dropout": 0.0},
        "data": "mm", "out": "rl_continuous", "archetype": "market_making_rl",
    },
    "DQN_MM_v1": {
        "ckpt": "market_maker/DQN_MM_v1/DQN_MM_v1_best.pt",
        "module": "quant_core.market_maker_models", "cls": "DQNNetwork",
        "kwargs": {"state_dim": 8, "num_actions": 3, "hidden": 128, "dropout": 0.0},
        "data": "mm", "out": "rl_discrete", "archetype": "market_making_rl",
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_model(manifest_entry: dict[str, Any]) -> torch.nn.Module | None:
    """Import model class, instantiate with kwargs, load .pt weights."""
    ckpt_path = CHECKPOINT_ROOT / manifest_entry["ckpt"]
    if not ckpt_path.exists():
        _log(f"[eval] MISSING checkpoint: {ckpt_path}")
        return None

    try:
        mod = importlib.import_module(manifest_entry["module"])
        cls = getattr(mod, manifest_entry["cls"])
    except Exception as e:
        _log(f"[eval] import error {manifest_entry['module']}.{manifest_entry['cls']}: {e}")
        return None

    try:
        state = torch.load(str(ckpt_path), map_location="cpu")
        # Handle wrapped state dicts
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
        # Infer runtime dimensions from checkpoint weights to avoid architecture mismatch.
        kwargs = dict(manifest_entry["kwargs"])
        cls_name = manifest_entry["cls"]
        if cls_name == "StatArbAutoencoder":
            k = state.get("encoder.cells.0.W_r.weight")
            if k is not None:
                kwargs["num_assets"] = int(k.shape[1])
        elif cls_name == "StatArbLSTM":
            k = state.get("cells.0.W_i.weight")
            if k is not None:
                kwargs["num_assets"] = int(k.shape[1])
        elif cls_name == "PPOActorCritic":
            k = state.get("trunk.0.weight")
            if k is not None:
                kwargs["state_dim"] = int(k.shape[1])
                kwargs["hidden"] = int(k.shape[0])
            k2 = state.get("actor_mean.weight")
            if k2 is not None:
                kwargs["action_dim"] = int(k2.shape[0])
        elif cls_name == "SACActorNetwork":
            k = state.get("net.0.weight")
            if k is not None:
                kwargs["state_dim"] = int(k.shape[1])
                kwargs["hidden"] = int(k.shape[0])
            k2 = state.get("net.6.weight")
            if k2 is not None:
                kwargs["action_dim"] = int(k2.shape[0] // 2)
        elif cls_name == "DQNNetwork":
            k = state.get("trunk.0.weight")
            if k is not None:
                kwargs["state_dim"] = int(k.shape[1])
                kwargs["hidden"] = int(k.shape[0])
            k2 = state.get("advantage_head.weight")
            if k2 is not None:
                kwargs["num_actions"] = int(k2.shape[0])
        elif cls_name == "TrendTransformerModel":
            pos = state.get("positional")
            if pos is not None and len(pos.shape) == 3:
                kwargs["seq_len"] = int(pos.shape[1])

        model = cls(**kwargs)
        model.load_state_dict(state, strict=False)
        model._eval_inferred_seq_len = int(kwargs.get("seq_len", 0))
        _log(f"[eval] loaded weights: {ckpt_path.relative_to(ROOT)}")
    except Exception as e:
        _log(f"[eval] weight load error for {manifest_entry['cls']}: {e}")
        return None

    model.to(INFER_DEVICE)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataset: TensorDataset,
    output_type: str,
    is_multimodal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run batched forward passes and return (logits, labels, actual_returns).

    actual_returns is None when the dataset does not carry a third tensor.
    """
    # Cap samples for speed — take the LAST MAX_EVAL_SAMPLES (most-recent OOS data)
    n_ds = len(dataset)
    if n_ds > MAX_EVAL_SAMPLES:
        indices = torch.arange(n_ds - MAX_EVAL_SAMPLES, n_ds)
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_actual_rets: list[np.ndarray] = []
    has_returns = False

    for batch in loader:
        # Detect 3-tensor batches (X, y, actual_return)
        if len(batch) == 3:
            x_or_img, labels, actual_ret = batch
            all_actual_rets.append(actual_ret.numpy())
            has_returns = True
        else:
            x_or_img, labels = batch

        amp_ctx = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if torch.cuda.is_available() and getattr(INFER_DEVICE, "type", "") == "cuda"
            else nullcontext()
        )
        if is_multimodal:
            imgs = x_or_img.to(INFER_DEVICE)
            tab = torch.zeros(imgs.size(0), 5, device=INFER_DEVICE)
            with torch.no_grad(), amp_ctx:
                out = model(imgs, tab)
        else:
            x = x_or_img.to(INFER_DEVICE)
            with torch.no_grad(), amp_ctx:
                out = model(x)

        if isinstance(out, tuple):
            out = out[0]  # PPO returns (mean, log_std, value)

        out_cpu = out.detach().cpu()

        if output_type == "binary":
            all_logits.append(out_cpu.numpy().ravel())
        else:
            all_logits.append(out_cpu.numpy())

        all_labels.append(labels.numpy().ravel())
        _release_memory(out_cpu, out)

    actual_returns_arr = np.concatenate(all_actual_rets) if has_returns else None

    if output_type == "binary":
        return np.concatenate(all_logits), np.concatenate(all_labels), actual_returns_arr
    else:
        return np.concatenate(all_logits, axis=0), np.concatenate(all_labels), actual_returns_arr


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    _banner("EVALUATE ALL CHECKPOINTS — Post-Outage Recovery Evaluation")
    _log(f"[eval] timestamp: {datetime.now(timezone.utc).isoformat()}")
    _log(f"[eval] device: {DEVICE_NAME}")
    _log(f"[eval] checkpoint root: {CHECKPOINT_ROOT}")
    _log(f"[eval] data root: {DATASET_DIR}")

    # ── Load all market data once ─────────────────────────────────────────
    _banner("Loading OOS test data from binance_historical")
    frames = _load_parquets(max_files=40)
    _log(f"[eval] loaded {len(frames)} symbol DataFrames for evaluation")
    if not frames:
        _log("[eval] FATAL: no parquet files found in Dataset/binance_historical/")
        return 1

    # ── Pre-build datasets by type ────────────────────────────────────────
    _log("[eval] building OOS datasets per archetype type...")

    _log("[eval] → building trend dataset...")
    datasets: dict[str, TensorDataset | None] = {}
    datasets["trend"]    = load_trend_data(frames, seq_len=96, horizon=20)
    _log(f"[eval]   trend:   {len(datasets['trend']) if datasets['trend'] else 'FAIL'} samples")

    _log("[eval] → building mean_reversion dataset...")
    datasets["mr"]       = load_mr_data(frames, horizon=20)
    _log(f"[eval]   mr:      {len(datasets['mr']) if datasets['mr'] else 'FAIL'} samples")

    _log("[eval] → building scalper dataset (fractional diff — may be slow)...")
    datasets["scalper"]  = load_scalper_data(frames, seq_len=32, horizon=5)
    _log(f"[eval]   scalper: {len(datasets['scalper']) if datasets['scalper'] else 'FAIL'} samples")

    _log("[eval] → building stat_arb dataset...")
    datasets["stat_arb"] = load_stat_arb_data(frames, seq_len=64, horizon=10, num_assets=min(10, len(frames)))
    _log(f"[eval]   stat_arb:{len(datasets['stat_arb']) if datasets['stat_arb'] else 'FAIL'} samples")

    _log("[eval] → building disc (chart image) dataset...")
    datasets["disc"]     = load_disc_data(frames, seq_len=32, horizon=5)
    _log(f"[eval]   disc:    {len(datasets['disc']) if datasets['disc'] else 'FAIL'} samples")

    _log("[eval] → building market_maker dataset...")
    datasets["mm"]       = load_mm_data(frames, n_steps=3000, state_dim=10)
    _log(f"[eval]   mm:      {len(datasets['mm']) if datasets['mm'] else 'FAIL'} samples")

    # Transformer was trained with seq_len=64 — build a separate slice
    _log("[eval] \u2192 building trend64 dataset (seq_len=64 for Transformer)...")
    datasets["trend64"]  = load_trend_data(frames, seq_len=64, horizon=20)
    _log(f"[eval]   trend64: {len(datasets['trend64']) if datasets['trend64'] else 'FAIL'} samples")

    # disc_multimodal reuses disc dataset
    datasets["disc_multimodal"] = datasets["disc"]

    # ── Evaluate each model ───────────────────────────────────────────────
    results: dict[str, dict[str, Any]] = {}

    for model_name, manifest in MODEL_MANIFEST.items():
        _banner(f"Evaluating: {model_name}")
        _log(f"[eval] arch={manifest['cls']}  archetype={manifest['archetype']}")
        _log(f"[eval] checkpoint={manifest['ckpt']}")

        result: dict[str, Any] = {
            "architecture_name": model_name,
            "archetype": manifest["archetype"],
            "weights_path": f"models/checkpoints/{manifest['ckpt']}",
            "eval_timestamp": datetime.now(timezone.utc).isoformat(),
            "device": DEVICE_NAME,
        }

        ckpt_path = CHECKPOINT_ROOT / manifest["ckpt"]
        if not ckpt_path.exists():
            _log(f"[eval] ✗ checkpoint not found — RESUME_TRAINING_REQUIRED")
            result["validation"] = {
                "status": "RESUME_TRAINING_REQUIRED",
                "reason": "checkpoint_missing",
                "directional_accuracy": None,
                "sharpe": None,
                "profit_factor": None,
                "max_drawdown": None,
            }
            results[model_name] = result
            continue

        # ── load weights ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        model = load_model(manifest)
        if model is None:
            result["validation"] = {
                "status": "RESUME_TRAINING_REQUIRED",
                "reason": "architecture_load_failed",
                "directional_accuracy": None,
                "sharpe": None,
                "profit_factor": None,
                "max_drawdown": None,
            }
            results[model_name] = result
            continue

        # ── get dataset ───────────────────────────────────────────────────
        # Build per-model dataset variants where feature dimensions must match checkpoint.
        if manifest["data"] == "stat_arb":
            target_assets = int(manifest["kwargs"].get("num_assets", 2))
            try:
                raw_state = torch.load(str(CHECKPOINT_ROOT / manifest["ckpt"]), map_location="cpu")
                if isinstance(raw_state, dict) and "model_state_dict" in raw_state:
                    raw_state = raw_state["model_state_dict"]
                if isinstance(raw_state, dict) and "state_dict" in raw_state:
                    raw_state = raw_state["state_dict"]
                if manifest["cls"] == "StatArbAutoencoder" and isinstance(raw_state, dict):
                    w = raw_state.get("encoder.cells.0.W_r.weight")
                    if w is not None:
                        target_assets = int(w.shape[1])
                if manifest["cls"] == "StatArbLSTM" and isinstance(raw_state, dict):
                    w = raw_state.get("cells.0.W_i.weight")
                    if w is not None:
                        target_assets = int(w.shape[1])
            except Exception:
                pass
            key = f"stat_arb_{target_assets}"
            if key not in datasets:
                datasets[key] = load_stat_arb_data(frames, seq_len=64, horizon=10, num_assets=target_assets)
            ds = datasets.get(key)
        elif manifest["data"] == "mm":
            target_state = int(manifest["kwargs"].get("state_dim", 7))
            try:
                raw_state = torch.load(str(CHECKPOINT_ROOT / manifest["ckpt"]), map_location="cpu")
                if isinstance(raw_state, dict) and "model_state_dict" in raw_state:
                    raw_state = raw_state["model_state_dict"]
                if isinstance(raw_state, dict) and "state_dict" in raw_state:
                    raw_state = raw_state["state_dict"]
                if isinstance(raw_state, dict):
                    w = raw_state.get("trunk.0.weight")
                    if w is None:
                        w = raw_state.get("net.0.weight")
                    if w is not None:
                        target_state = int(w.shape[1])
            except Exception:
                pass
            key = f"mm_{target_state}"
            if key not in datasets:
                datasets[key] = load_mm_data(frames, n_steps=3000, state_dim=target_state)
            ds = datasets.get(key)
        elif manifest["cls"] == "TrendTransformerModel":
            target_seq = int(getattr(model, "_eval_inferred_seq_len", 64) or 64)
            key = f"trend_seq_{target_seq}"
            if key not in datasets:
                datasets[key] = load_trend_data(frames, seq_len=target_seq, horizon=20)
            ds = datasets.get(key)
        else:
            ds = datasets.get(manifest["data"])
        if ds is None or len(ds) == 0:
            _log(f"[eval] ✗ test dataset unavailable for data type: {manifest['data']}")
            result["validation"] = {
                "status": "RESUME_TRAINING_REQUIRED",
                "reason": "test_data_unavailable",
                "directional_accuracy": None,
                "sharpe": None,
                "profit_factor": None,
                "max_drawdown": None,
            }
            results[model_name] = result
            continue

        # ── forward pass ──────────────────────────────────────────────────
        try:
            is_multi = (manifest["data"] == "disc_multimodal")
            logits, labels, actual_rets = run_inference(model, ds, manifest["out"], is_multimodal=is_multi)
            elapsed = time.perf_counter() - t0
            _log(f"[eval] forward pass: {len(labels):,} samples  |  {elapsed:.2f}s  |  {len(labels)/elapsed:.0f} samples/s")
        except Exception as e:
            _log(f"[eval] ✗ inference error: {e}")
            result["validation"] = {
                "status": "RESUME_TRAINING_REQUIRED",
                "reason": f"inference_error: {e}",
                "directional_accuracy": None,
                "sharpe": None,
                "profit_factor": None,
                "max_drawdown": None,
            }
            results[model_name] = result
            # free GPU memory
            del model
            continue

        # ── compute metrics ───────────────────────────────────────────────
        metrics = compute_all_metrics(logits, labels, manifest["out"], actual_returns=actual_rets)

        dir_acc = metrics["directional_accuracy"]
        sharpe  = metrics["sharpe"]
        pf      = metrics["profit_factor"]
        mdd     = metrics["max_drawdown"]

        strict_pass = passes_production_gates(metrics, require_directional_accuracy=True)
        status = "PASSED" if strict_pass else "RESUME_TRAINING_REQUIRED"

        gate_str = "✓ PASSED" if strict_pass else "✗ FAILED"
        _log(f"[eval] Directional Accuracy : {dir_acc:.4f}  (gate > {PRODUCTION_GATES.directional_accuracy_min})")
        _log(f"[eval] Sharpe Ratio         : {sharpe:.4f}  (gate > {PRODUCTION_GATES.sharpe_min})")
        _log(f"[eval] Profit Factor        : {pf:.4f}  (gate > {PRODUCTION_GATES.profit_factor_min})")
        _log(f"[eval] Max Drawdown         : {mdd:.4f}  (gate < {PRODUCTION_GATES.max_drawdown_max})")
        _log(f"[eval] KPI Gate             : {gate_str}")

        result["validation"] = {
            "status": status,
            "directional_accuracy": dir_acc,
            "sharpe": sharpe,
            "profit_factor": pf,
            "max_drawdown": mdd,
            "n_samples": int(len(labels)),
            "device": DEVICE_NAME,
            "strict_sharpe_gate": PRODUCTION_GATES.sharpe_min,
            "strict_acc_gate": PRODUCTION_GATES.directional_accuracy_min,
            "strict_profit_factor_gate": PRODUCTION_GATES.profit_factor_min,
            "strict_max_drawdown_gate": PRODUCTION_GATES.max_drawdown_max,
        }
        results[model_name] = result

        # free GPU memory
        _release_memory(model)

    # ═════════════════════════════════════════════════════════════════════
    # REBUILD model_registry.json — hard numbers only, no malformed entries
    # ═════════════════════════════════════════════════════════════════════
    _banner("Rebuilding model_registry.json")
    registry_entries = list(results.values())
    REGISTRY_PATH.write_text(
        json.dumps(registry_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _log(f"[eval] registry written: {REGISTRY_PATH}  ({len(registry_entries)} entries)")

    # ═════════════════════════════════════════════════════════════════════
    # TERMINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    _banner("EVALUATION COMPLETE — TRUE HARDWARE METRICS SUMMARY")

    passed   = [n for n, r in results.items() if r["validation"].get("status") == "PASSED"]
    failed   = [n for n, r in results.items() if r["validation"].get("status") == "RESUME_TRAINING_REQUIRED"]
    no_ckpt  = [n for n, r in results.items() if r["validation"].get("reason") == "checkpoint_missing"]

    print(f"\n{'Model':<35} {'Acc':>8} {'Sharpe':>10} {'PF':>8} {'MDD':>9}  Status")
    print("-" * 90)
    for name, r in sorted(results.items(), key=lambda x: x[1]["validation"].get("sharpe") or -999, reverse=True):
        v = r["validation"]
        acc   = f"{v.get('directional_accuracy', 'N/A'):>8.4f}" if v.get("directional_accuracy") is not None else f"{'N/A':>8}"
        sh    = f"{v.get('sharpe', 'N/A'):>10.4f}"              if v.get("sharpe") is not None else f"{'N/A':>10}"
        pf_v  = f"{v.get('profit_factor', 'N/A'):>8.4f}"       if v.get("profit_factor") is not None else f"{'N/A':>8}"
        mdd_v = f"{v.get('max_drawdown', 'N/A'):>9.4f}"         if v.get("max_drawdown") is not None else f"{'N/A':>9}"
        stat  = v.get("status", "UNKNOWN")
        print(f"{name:<35} {acc} {sh} {pf_v} {mdd_v}  {stat}")

    print(f"\nTotal models evaluated : {len(results)}")
    print(f"PASSED (strict gates)  : {len(passed)}")
    print(f"RESUME_TRAINING_REQUIRED: {len(failed)}")
    print(f"Missing checkpoints    : {len(no_ckpt)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
