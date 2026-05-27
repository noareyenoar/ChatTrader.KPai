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
import re
import sys
import time
import gc
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Phase 6 robustness gates
try:
    from quant_core.robustness_tests import run_robustness_suite as _run_robustness
    _ROBUSTNESS_AVAILABLE = True
except Exception as _rob_err:
    _ROBUSTNESS_AVAILABLE = False
    _run_robustness = None  # type: ignore[assignment]

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
MR_FEAT_COLS    = ["vwap_dev", "bb_distance", "zscore_close_20", "zscore_close_64",
                    "rsi_14", "rsi_div_5", "rsi_oversold", "rsi_overbought"]     # 8 (v2)
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


def max_drawdown_rl(ep_rewards: np.ndarray) -> float:
    """RL MDD proxy: |sum(losing episodes)| / |sum(winning episodes)|.

    Cumsum-based MDD gives artefacts for RL because the temporal ordering
    of wins and losses dominates the result (a model that wins early and
    loses late gets MDD=1.0 even with a very positive mean reward).
    Instead we measure what fraction of total profits are given back as
    losses: this equals 1 / profit_factor, bounded to [0, 1].

    - MDD = 0   → no losing episodes at all
    - MDD < 0.85 → passes RL gate (losses < 85% of wins)
    - MDD = 1.0 → losses >= wins (model is unprofitable or flat)
    """
    gains  = float(np.sum(ep_rewards[ep_rewards > 0]))
    losses = float(np.sum(np.abs(ep_rewards[ep_rewards < 0])))
    if gains <= 0.0:
        return 1.0  # no winning episodes → maximum risk
    return min(1.0, losses / gains)


ROUND_TRIP_COST = 0.001  # 0.1% per completed round-trip (maker+taker, crypto futures)


def compute_all_metrics(
    logits: np.ndarray,
    true_labels: np.ndarray,
    output_type: str,
    actual_returns: np.ndarray | None = None,
    _out_returns: list | None = None,
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
    elif output_type == "tg_mnn":
        # TG-MNN: state_logits [B, 3] → 0=Steady, 1=Up, 2=Down
        # Map to binary direction: Up(1)=long(1), Down(2)=short(0), Steady(0)=skip
        pred_state = np.argmax(logits, axis=1)  # 0=Steady, 1=Up, 2=Down
        # Convert to directional: Up→1, Down→0, Steady→abstain (treated as 0 for acc)
        direction = np.where(pred_state == 1, 1, 0).astype(int)  # 1=up, 0=not-up
        true_direction = true_labels.astype(int)  # binary: 1=price went up
        trade_mask = (pred_state != 0).astype(np.float64)  # only trade on Up/Down signals
        ret_sign = np.where(pred_state == 1, 1.0, np.where(pred_state == 2, -1.0, 0.0))
        ret_sign = ret_sign * np.where(true_direction == 1, 1.0, -1.0)
    elif output_type in ("rl_continuous", "rl_discrete"):
        # RL: action maps to direction
        if output_type == "rl_discrete":
            pred_action = np.argmax(logits, axis=1)  # 0=tight, 1=medium, 2=wide
            direction = (pred_action >= 1).astype(int)  # treat medium/wide as "active"
        else:
            direction = (logits[:, 0] > 0.5).astype(int)  # bid offset > 0.5 = buy lean
        true_direction = true_labels.astype(int)
        ret_sign = np.where(direction == true_direction, 1.0, -1.0)
    elif output_type == "apv_pln":
        # APV-PLN: logits = expected return (scalar per sample, already reduced in run_inference)
        direction = (logits.ravel() > 0.0).astype(int)  # 1=expected up, 0=expected down
        true_direction = true_labels.astype(int)         # 1=actual up, 0=actual down
        ret_sign = np.where(direction == true_direction, 1.0, -1.0)
    else:
        direction = (logits.ravel() > 0).astype(int)
        true_direction = true_labels.astype(int)
        ret_sign = np.where(direction == true_direction, 1.0, -1.0)

    dir_acc = directional_accuracy(direction, true_direction)

    # ── PnL using actual forward returns when available ───────────────────
    if actual_returns is not None:
        abs_ret = np.abs(actual_returns).astype(np.float64)
        if output_type in ("multiclass3", "tg_mnn"):
            # flat/steady predictions = no trade, no transaction cost
            trade_mask = (ret_sign != 0).astype(np.float64)
            returns = ret_sign * abs_ret - trade_mask * ROUND_TRIP_COST
        else:
            returns = ret_sign * abs_ret - ROUND_TRIP_COST
    else:
        returns = ret_sign * 0.001  # fallback: fixed 0.1% scale

    if _out_returns is not None:
        _out_returns.append(returns.copy())

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


def load_trend_data(frames: list[pd.DataFrame], seq_len: int = 96, horizon: int = 5) -> TensorDataset | None:
    """Build sequential OOS test data for Trend models. Input: [B, seq_len, 5]
    Returns 3-tensor dataset: (X, y_direction, actual_forward_return)."""
    all_X, all_y, all_ret = [], [], []
    for df in frames:
        try:
            from data_pipeline.features import FeatureFactory
            from data_pipeline.splitter import IronWallSplitter
            feat = FeatureFactory.build_trend_features(df)
            available = [c for c in TREND_FEAT_COLS if c in feat.columns]
            if len(available) < 3:
                continue
            close = feat["close"].to_numpy(np.float32)
            future_close = np.roll(close, -horizon)
            target = (future_close > close).astype(np.float32)
            target[-horizon:] = np.nan
            fwd_ret = (future_close - close) / (np.abs(close) + 1e-8)
            fwd_ret[-horizon:] = np.nan
            feat = feat.assign(_target=target, _fwd_ret=fwd_ret).dropna(subset=["_target", "_fwd_ret"]).reset_index(drop=True)
            split = IronWallSplitter(purge_gap_bars=horizon).split(feat, time_col="timestamp")
            scaler = FeatureFactory.fit_scaler_train_only(split.train, available)
            test = FeatureFactory.transform_with_scaler(split.test, scaler)
            feat_test = test[available].to_numpy(np.float32)
            tgt_test = test["_target"].to_numpy(np.float32)
            ret_test = test["_fwd_ret"].to_numpy(np.float32)
            seqs = _stride_sequences(feat_test, seq_len)   # (M, seq_len, F)
            labels = tgt_test[seq_len - 1: seq_len - 1 + len(seqs)]
            rets   = ret_test[seq_len - 1: seq_len - 1 + len(seqs)]
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


def load_mr_data(frames: list[pd.DataFrame], horizon: int = 3) -> TensorDataset | None:
    """Build tabular OOS test data for Mean Reversion models. Input: [B, 8]
    Returns 3-tensor dataset: (X, y_direction, actual_forward_return).
    horizon=3 matches mr_phase4.yaml training horizon."""
    all_X, all_y, all_ret = [], [], []
    for df in frames:
        try:
            from data_pipeline.features import FeatureFactory
            from data_pipeline.splitter import IronWallSplitter
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
            split = IronWallSplitter(purge_gap_bars=horizon).split(feat.reset_index(drop=True), time_col="timestamp")
            scaler = FeatureFactory.fit_scaler_train_only(split.train, available)
            test = FeatureFactory.transform_with_scaler(split.test, scaler)
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


def load_scalper_data(frames: list[pd.DataFrame], seq_len: int = 32, horizon: int = 2) -> TensorDataset | None:
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

    v2 pipeline (preferred): extracts fracdiff_close_d04, spread_z_64, hurst_proxy per
    symbol (3 features each) and aligns on common timestamps — matching stat_arb_data.py.
    Falls back to raw z-scored pct_change returns if v2 features are unavailable.

    num_assets from checkpoint is used to detect which pipeline to use:
      - num_assets divisible by 3 AND num_assets // 3 <= len(frames): use v2 pipeline
      - otherwise: use v1 raw-returns pipeline (len(frames) assets, 1 feature each)
    """
    FEAT_PER_ASSET = ["fracdiff_close_d04", "spread_z_64", "hurst_proxy"]
    n_sym = len(frames)
    # Determine which pipeline based on detected num_assets
    use_v2 = (num_assets % 3 == 0) and (num_assets // 3 <= n_sym)
    n_sym_v2 = num_assets // 3 if use_v2 else n_sym

    if use_v2:
        # ── V2 pipeline: compute stat-arb features per symbol, align on timestamp ──
        try:
            from data_pipeline.features import FeatureFactory
            sym_frames = {}
            for idx, df in enumerate(frames[:n_sym_v2]):
                try:
                    feat = FeatureFactory.build_stat_arb_features(df)
                    available = [c for c in FEAT_PER_ASSET if c in feat.columns]
                    if len(available) < 1:
                        available = ["fracdiff_close_d04"] if "fracdiff_close_d04" in feat.columns else []
                    if not available:
                        continue
                    sym_feat = feat[["timestamp"] + available].dropna().copy()
                    sym_feat.columns = ["timestamp"] + [f"SYM{idx:03d}_{c}" for c in available]
                    sym_frames[idx] = sym_feat.set_index("timestamp")
                except Exception as e:
                    _log(f"[eval] stat_arb v2 sym {idx} warning: {e}")
            if len(sym_frames) < 2:
                _log("[eval] stat_arb v2: too few symbols with features, falling back to v1")
                use_v2 = False
        except ImportError:
            use_v2 = False

    if use_v2:
        try:
            aligned = pd.concat(list(sym_frames.values()), axis=1).dropna()
            if len(aligned) < seq_len + horizon + 20:
                return None
            feat_arr = aligned.to_numpy(np.float32)   # (T, num_assets_actual)
            # Iron Wall: use last 15% as test
            train_n = int(len(feat_arr) * 0.85)
            train_arr = feat_arr[:train_n]
            test_arr = feat_arr[train_n:]
            mean = train_arr.mean(axis=0, keepdims=True)
            std = train_arr.std(axis=0, keepdims=True) + 1e-8
            test_z = (test_arr - mean) / std
            seqs = _stride_sequences(test_z, seq_len)     # (M, seq_len, num_assets)
            # Target: mean fracdiff signal across all assets (fracdiff cols at position 0, 3, 6, ...)
            fd_idxs = list(range(0, feat_arr.shape[1], len(FEAT_PER_ASSET)))
            tgt_raw = test_z[:, fd_idxs].mean(axis=1)
            labels = np.array([tgt_raw[i + seq_len: i + seq_len + horizon].mean()
                                for i in range(len(seqs))], dtype=np.float32)
            valid = len(seqs) - horizon
            if valid <= 0:
                return None
            X_t = torch.tensor(seqs[:valid], dtype=torch.float32)
            y_t = torch.tensor(labels[:valid], dtype=torch.float32)
            actual_ret = np.clip(labels[:valid] * 0.005, -0.02, 0.02)
            r_t = torch.tensor(actual_ret.astype(np.float32))
            _log(f"[eval] stat_arb v2 data built: X={tuple(X_t.shape)} n_sym={len(sym_frames)} feat_per_sym={len(FEAT_PER_ASSET)}")
            return TensorDataset(X_t, y_t, r_t)
        except Exception as e:
            _log(f"[eval] stat_arb v2 pipeline failed: {e} — falling back to v1")

    # ── V1 fallback: raw pct_change returns ──
    selected = frames[:max(n_sym_v2, 2)]
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
        )  # (min_len, num_symbols)
        train_n = int(min_len * 0.85)
        mu = returns[:train_n].mean(0)
        std = returns[:train_n].std(0) + 1e-8
        z_scores = (returns - mu) / std   # (min_len, num_symbols)
        # Test portion only
        test_z = z_scores[train_n:]
        seqs = _stride_sequences(test_z, seq_len)      # (M, seq_len, num_symbols)
        # Target: mean of next `horizon` bars of first asset
        tgt_raw = test_z[:, 0]
        labels = np.array([tgt_raw[i + seq_len: i + seq_len + horizon].mean()
                           for i in range(len(seqs))], dtype=np.float32)
        valid = len(seqs) - horizon
        if valid <= 0:
            return None
        X_t = torch.tensor(seqs[:valid], dtype=torch.float32)
        y_t = torch.tensor(labels[:valid], dtype=torch.float32)
        # Scale z-scores to small per-trade returns so max_drawdown stays bounded.
        actual_ret = np.clip(labels[:valid] * 0.005, -0.02, 0.02)
        r_t = torch.tensor(actual_ret.astype(np.float32))
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


def load_disc_multimodal_data(frames: list[pd.DataFrame], seq_len: int = 32, horizon: int = 5) -> TensorDataset | None:
    """Build (img, tab, y, ret) dataset for DiscretionaryMultimodal evaluation.

    tab features: [log_return, zscore_close_64, ema_spread, atr_14, price_slope_20]
    (same 5 DISC_TAB_FEATURES used during training)
    """
    FLAT_THR = 0.001
    TAB_FEATURES = ["log_return", "zscore_close_64", "ema_spread", "atr_14", "price_slope_20"]
    try:
        from quant_core.discretionary_data import _rasterize_window
        from data_pipeline.features import FeatureFactory
    except Exception as e:
        _log(f"[eval] disc multimodal renderer import failed: {e} — skipping")
        return None

    all_imgs, all_tab, all_y, all_ret = [], [], [], []

    for df in frames[:3]:
        try:
            feat_df = FeatureFactory.build_discretionary_features(df)
            # Align on shared index
            common_idx = feat_df.index.intersection(df.index)
            df_a = df.loc[common_idx].reset_index(drop=True)
            feat_a = feat_df.loc[common_idx].reset_index(drop=True)

            n = len(df_a)
            test_start = int(n * 0.85)
            close = df_a["close"].to_numpy(np.float32)
            open_ = df_a["open"].to_numpy(np.float32)
            high  = df_a["high"].to_numpy(np.float32)
            low   = df_a["low"].to_numpy(np.float32)

            # Fit scaler on train portion only
            train_feat = feat_a.iloc[:int(n * 0.70)]
            tab_arr = feat_a[TAB_FEATURES].fillna(0.0).to_numpy(np.float32)
            means = train_feat[TAB_FEATURES].mean().to_numpy(np.float32)
            stds  = train_feat[TAB_FEATURES].std().to_numpy(np.float32)
            stds[stds < 1e-8] = 1.0
            tab_scaled = (tab_arr - means) / stds

            future_ret = np.roll(close, -horizon) / (close + 1e-10) - 1.0
            targets = np.where(future_ret > FLAT_THR, 2,
                      np.where(future_ret < -FLAT_THR, 0, 1)).astype(np.int64)

            ohlcv = np.stack([open_, high, low, close], axis=1)

            i0 = test_start
            i1 = n - seq_len - horizon + 1
            if i1 <= i0:
                continue
            i0_eff = max(i0, i1 - 5_000)

            imgs_list, tab_list, tgt_list, ret_list = [], [], [], []
            for i in range(i0_eff, i1):
                window = ohlcv[i: i + seq_len]
                img = _rasterize_window(window)
                imgs_list.append(img)
                tab_list.append(tab_scaled[i + seq_len - 1])
                tgt_list.append(targets[i + seq_len - 1])
                ret_list.append(float(future_ret[i + seq_len - 1]))

            if not imgs_list:
                continue
            all_imgs.append(np.stack(imgs_list))
            all_tab.append(np.stack(tab_list))
            all_y.append(np.array(tgt_list, dtype=np.int64))
            all_ret.append(np.array(ret_list, dtype=np.float32))
        except Exception as e:
            _log(f"[eval] disc multimodal data build warning: {e}")

    if not all_imgs:
        return None
    X_t = torch.tensor(np.concatenate(all_imgs), dtype=torch.float32)
    T_t = torch.tensor(np.concatenate(all_tab), dtype=torch.float32)
    y_t = torch.tensor(np.concatenate(all_y), dtype=torch.int64)
    r_t = torch.tensor(np.concatenate(all_ret), dtype=torch.float32)
    return TensorDataset(X_t, T_t, y_t, r_t)


@torch.no_grad()
def eval_rl_episode(
    model: torch.nn.Module,
    frames: list[pd.DataFrame],
    output_type: str,
    state_dim: int,
    n_eval_episodes: int = 200,
    episode_length: int = 200,
) -> dict[str, float]:
    """Run proper episodic evaluation for RL market-maker models using MarketMakingEnv.

    This is the CORRECT evaluator for PPO/SAC/DQN — NOT a one-shot directional
    forward-pass.  Each episode replays a segment of OOS price data (last 15%).
    Returns Sharpe, PF, MDD computed over episode rewards.
    """
    from quant_core.market_maker_env import MarketMakingEnv

    all_ep_rewards: list[float] = []
    all_returns: list[float] = []

    rng = np.random.default_rng(42)

    def _fit_state_dim(state_vec: np.ndarray | list[float], target_dim: int) -> np.ndarray:
        arr = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] >= target_dim:
            return arr[:target_dim]
        out = np.zeros(target_dim, dtype=np.float32)
        out[: arr.shape[0]] = arr
        return out

    for df in frames[:6]:  # use up to 6 symbols
        n = len(df)
        test_start = int(n * 0.85)
        prices = df["close"].to_numpy(np.float32)[test_start:]
        if len(prices) < episode_length + 10:
            continue

        for _ in range(n_eval_episodes // max(1, len(frames[:6]))):
            env = MarketMakingEnv(prices, episode_length=episode_length, seed=int(rng.integers(0, 99999)))
            state = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                state_aligned = _fit_state_dim(state, state_dim)
                state_t = torch.tensor(state_aligned, dtype=torch.float32).unsqueeze(0).to(INFER_DEVICE)
                if output_type == "rl_discrete":
                    q_vals = model(state_t)
                    if isinstance(q_vals, tuple):
                        q_vals = q_vals[0]
                    action = int(q_vals.argmax(dim=-1).item())
                    offsets = [0.0005, 0.001, 0.002][action]
                    cont_action = np.array([offsets, offsets], dtype=np.float32)
                else:
                    out = model(state_t)
                    if isinstance(out, tuple):
                        out = out[0]
                    cont_action = out.squeeze(0).cpu().numpy()[:2]
                state, reward, done, _ = env.step(cont_action)
                ep_reward += float(reward)
            all_ep_rewards.append(ep_reward)

    if len(all_ep_rewards) < 5:
        return {"directional_accuracy": 0.5, "sharpe": 0.0, "profit_factor": 1.0, "max_drawdown": 1.0}

    ep_arr = np.array(all_ep_rewards, dtype=np.float64)
    # Normalise episode rewards to per-step scale for Sharpe/PF computation.
    # Use std-based normalisation to keep values in a sensible range.
    ep_std = float(ep_arr.std()) + 1e-10
    ep_returns = ep_arr / ep_std  # z-score scale — Sharpe/PF invariant to positive scale

    sharpe = sharpe_ratio(ep_returns)
    pf = profit_factor(ep_returns)
    # Use additive MDD on raw episode rewards (not multiplicative on normalised returns)
    mdd = max_drawdown_rl(ep_arr)
    dir_acc = float(np.mean(ep_arr > 0))  # fraction of profitable episodes

    _log(f"[eval-rl] episodes={len(all_ep_rewards)}  mean_ep_reward={ep_arr.mean():.4f}  "
         f"sharpe={sharpe:.4f}  pf={pf:.4f}  mdd={mdd:.4f}  win_rate={dir_acc:.4f}")
    return {
        "directional_accuracy": round(dir_acc, 6),
        "sharpe": round(sharpe, 6),
        "profit_factor": round(pf, 6),
        "max_drawdown": round(mdd, 6),
    }


def load_apv_pln_data(frames: list[pd.DataFrame], seq_len: int = 32, horizon: int = 5) -> TensorDataset | None:
    """Build dual-stream (x_price, x_volume) OOS test dataset for APV-PLN evaluation.

    Returns TensorDataset with 4 tensors:
        x_price   : [B, seq_len, 5]  — PRICE_FEATURES
        x_volume  : [B, seq_len, 5]  — VOLUME_FEATURES
        y_dir     : [B]  int64       — 1=up, 0=down (directional label)
        actual_ret: [B]  float32     — raw forward log-return (for PnL)
    Oracle features are NOT included (inference-only, Oracle Isolation Contract).
    """
    try:
        from quant_core.apv_pln_data import (
            _build_apvpln_features,
            PRICE_FEATURES,
            VOLUME_FEATURES,
        )
    except Exception as exc:
        _log(f"[eval] apv_pln import error: {exc} — skipping apv_pln dataset")
        return None

    all_price, all_vol, all_y, all_ret = [], [], [], []

    for df in frames[:6]:
        try:
            feat = _build_apvpln_features(df)
            feat = feat.dropna(subset=PRICE_FEATURES + VOLUME_FEATURES).reset_index(drop=True)
            n = len(feat)
            if n < seq_len + horizon + 100:
                continue

            # Iron Wall: test split = last 15%
            test_start = int(n * 0.85) + 20  # +20 purge gap
            price_arr = feat[PRICE_FEATURES].to_numpy(np.float32)
            vol_arr   = feat[VOLUME_FEATURES].to_numpy(np.float32)
            close_arr = feat["log_return"].to_numpy(np.float64)  # log_return for fwd ret

            # Train-only scaler (fit on first 70%)
            train_end = int(n * 0.70)
            for col_idx in range(price_arr.shape[1]):
                m, s = price_arr[:train_end, col_idx].mean(), price_arr[:train_end, col_idx].std()
                s = s if s > 1e-8 else 1.0
                price_arr[:, col_idx] = (price_arr[:, col_idx] - m) / s
            for col_idx in range(vol_arr.shape[1]):
                m, s = vol_arr[:train_end, col_idx].mean(), vol_arr[:train_end, col_idx].std()
                s = s if s > 1e-8 else 1.0
                vol_arr[:, col_idx] = (vol_arr[:, col_idx] - m) / s

            # Build sequences from test split, cap at 5000 samples
            i0 = test_start
            i1 = n - seq_len - horizon + 1
            if i1 <= i0:
                continue
            i0_eff = max(i0, i1 - 5_000)

            for i in range(i0_eff, i1):
                p_win = price_arr[i: i + seq_len]      # [seq_len, 5]
                v_win = vol_arr[i: i + seq_len]         # [seq_len, 5]
                # Forward log-return over horizon bars
                fwd = float(np.sum(close_arr[i + seq_len: i + seq_len + horizon]))
                y_dir = int(fwd > 0.0)
                all_price.append(p_win)
                all_vol.append(v_win)
                all_y.append(y_dir)
                all_ret.append(float(fwd))
        except Exception as exc:
            _log(f"[eval] apv_pln data build warning: {exc}")

    if not all_price:
        return None
    xp_t = torch.tensor(np.stack(all_price), dtype=torch.float32)
    xv_t = torch.tensor(np.stack(all_vol), dtype=torch.float32)
    y_t  = torch.tensor(all_y, dtype=torch.int64)
    r_t  = torch.tensor(all_ret, dtype=torch.float32)
    return TensorDataset(xp_t, xv_t, y_t, r_t)


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
            target_dim = int(max(1, state_dim))
            base_dim = int(base_states.shape[1])
            if target_dim <= base_dim:
                states = base_states[:, :target_dim]
            else:
                pad = np.zeros((nb, target_dim - base_dim), dtype=np.float32)
                states = np.concatenate([base_states, pad], axis=1)
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
        "kwargs": {"num_assets": 34, "latent_dim": 32, "seq_len": 64, "dropout": 0.0},
        "data": "stat_arb", "out": "regression", "archetype": "statistical_arbitrage",
    },
    "GAT_StatArb_v1": {
        "ckpt": "stat_arb/GAT_StatArb_v1/model_best.pt",
        "module": "quant_core.stat_arb_models", "cls": "StatArbGAT",
        "kwargs": {"num_assets": 34, "d_model": 32, "num_layers": 2, "dropout": 0.0},
        "data": "stat_arb", "out": "regression", "archetype": "statistical_arbitrage",
    },
    "LSTM_StatArb_v1": {
        "ckpt": "stat_arb/LSTM_StatArb_v1/model_best.pt",
        "module": "quant_core.stat_arb_models", "cls": "StatArbLSTM",
        "kwargs": {"num_assets": 34, "hidden_size": 64, "num_layers": 2, "dropout": 0.0},
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
    # ── TG-MNN: Temporal-Gradient Markov Neural Network (Trend archetype) ────
    # Skipped automatically if checkpoint not yet trained.
    "TG_MNN_v1": {
        # Trainer saves best checkpoint directly to output_dir/model_best.pt
        # (output_dir = models/checkpoints/tg_mnn per configs/tg_mnn_phase4.yaml)
        "ckpt": "tg_mnn/model_best.pt",
        "module": "quant_core.tg_mnn_models", "cls": "TGMNNModel",
        "kwargs": {"input_dim": 5, "hidden_dim": 64, "num_backbone_layers": 3},
        "data": "trend", "out": "tg_mnn", "archetype": "trend_follower",
    },
    # ── APV-PLN: Adaptive Price-Volume Probabilistic Learner (Oracle Teacher) ─
    # Dual CNN + Cross-Attention + Oracle Knowledge Distillation (LUPI).
    # Oracle Teacher is train-only; evaluation uses Student only.
    # Skipped automatically if checkpoint not yet trained.
    "APV_PLN_v1": {
        "ckpt": "apv_pln/APV_PLN_v1/model_best.pt",
        "module": "quant_core.apv_pln_models", "cls": "APVPLNModel",
        "kwargs": {"price_dim": 5, "vol_dim": 5, "num_bins": 51, "cnn_channels": 64, "nhead": 4, "dropout": 0.0},
        "data": "apv_pln", "out": "apv_pln", "archetype": "trend_follower",
    },
    "APV_PLN_v2": {
        "ckpt": "apv_pln/APV_PLN_v2/model_best.pt",
        "module": "quant_core.apv_pln_models", "cls": "APVPLNModel",
        "kwargs": {"price_dim": 5, "vol_dim": 5, "num_bins": 51, "cnn_channels": 128, "nhead": 4, "dropout": 0.0},
        "data": "apv_pln", "out": "apv_pln", "archetype": "trend_follower",
    },
    "APV_PLN_v3": {
        "ckpt": "apv_pln/APV_PLN_v3/model_best.pt",
        "module": "quant_core.apv_pln_models", "cls": "APVPLNModel",
        "kwargs": {"price_dim": 5, "vol_dim": 5, "num_bins": 51, "cnn_channels": 64, "nhead": 8, "dropout": 0.0},
        "data": "apv_pln", "out": "apv_pln", "archetype": "trend_follower",
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
        elif cls_name == "TrendLSTMModel":
            k = state.get("cells.0.W_i.weight")
            if k is not None:
                kwargs["input_dim"] = int(k.shape[1])
                kwargs["hidden_size"] = int(k.shape[0])
            layer_ids = []
            for name in state.keys():
                m = re.match(r"cells\.(\d+)\.W_i\.weight", name)
                if m:
                    layer_ids.append(int(m.group(1)))
            if layer_ids:
                kwargs["num_layers"] = max(layer_ids) + 1
        elif cls_name == "StatArbLSTM":
            k = state.get("in_proj.weight")
            if k is not None:
                kwargs["num_assets"] = int(k.shape[1])
                kwargs["hidden_size"] = int(k.shape[0])
            else:
                k = state.get("cells.0.W_i.weight")
                if k is not None:
                    kwargs["num_assets"] = int(k.shape[1])
                    kwargs["hidden_size"] = int(k.shape[0])
            block_ids = []
            for name in state.keys():
                m = re.match(r"blocks\.(\d+)\.conv\.weight", name)
                if m:
                    block_ids.append(int(m.group(1)))
            if block_ids:
                kwargs["num_layers"] = max(block_ids) + 1
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
        elif cls_name == "TGMNNModel":
            k = state.get("backbone.input_proj.weight")
            if k is not None:
                kwargs["input_dim"] = int(k.shape[1])
                kwargs["hidden_dim"] = int(k.shape[0])
            block_ids = [int(m.group(1)) for name in state.keys()
                         for m in [re.match(r"backbone\.blocks\.(\d+)\.", name)] if m]
            if block_ids:
                kwargs["num_backbone_layers"] = max(block_ids) + 1
        elif cls_name == "APVPLNModel":
            # price_cnn.0.conv weight: [cnn_channels, price_dim, kernel]
            k = state.get("price_cnn.0.conv.weight")
            if k is not None:
                kwargs["cnn_channels"] = int(k.shape[0])
                kwargs["price_dim"]    = int(k.shape[1])
            k2 = state.get("volume_cnn.0.conv.weight")
            if k2 is not None:
                kwargs["vol_dim"] = int(k2.shape[1])
            # prob_head: [num_bins, cnn_channels]
            k3 = state.get("prob_head.weight")
            if k3 is not None:
                kwargs["num_bins"] = int(k3.shape[0])

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
    bin_centers: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run batched forward passes and return (logits, labels, actual_returns).

    actual_returns is None when the dataset does not carry a third tensor.
    For APV-PLN (output_type='apv_pln'), bin_centers must be provided; the
    returned 'logits' array is the scalar expected-return per sample (not raw
    num_bins logits).
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
        # Detect 4-tensor APV-PLN batches (x_price, x_vol, y_dir, actual_ret)
        if len(batch) == 4 and output_type == "apv_pln":
            xp_b, xv_b, labels, actual_ret = batch
            all_actual_rets.append(actual_ret.numpy())
            has_returns = True
        # Detect 4-tensor batches (img, tab, y, actual_return) for disc_multimodal
        elif len(batch) == 4 and is_multimodal:
            imgs_b, tab_b, labels, actual_ret = batch
            all_actual_rets.append(actual_ret.numpy())
            has_returns = True
        # Detect 3-tensor batches (X, y, actual_return)
        elif len(batch) == 3:
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
        if output_type == "apv_pln":
            xp = xp_b.to(INFER_DEVICE)
            xv = xv_b.to(INFER_DEVICE)
            with torch.no_grad(), amp_ctx:
                raw_logits = model(xp, xv)          # [B, num_bins]
            probs = torch.softmax(raw_logits.float(), dim=-1)   # [B, num_bins]
            bc = bin_centers.to(INFER_DEVICE).float()           # [num_bins]
            exp_ret = (probs * bc.unsqueeze(0)).sum(dim=-1)     # [B]
            out_cpu = exp_ret.detach().cpu()
            all_logits.append(out_cpu.numpy())                  # 1D array of scalars
            all_labels.append(labels.numpy().ravel())
            _release_memory(out_cpu, raw_logits, exp_ret)
            continue

        amp_ctx = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if torch.cuda.is_available() and getattr(INFER_DEVICE, "type", "") == "cuda"
            else nullcontext()
        )
        if is_multimodal:
            if len(batch) == 4:
                imgs = imgs_b.to(INFER_DEVICE)
                tab = tab_b.to(INFER_DEVICE)
            else:
                imgs = x_or_img.to(INFER_DEVICE)
                tab = torch.zeros(imgs.size(0), 5, device=INFER_DEVICE)
            with torch.no_grad(), amp_ctx:
                out = model(imgs, tab)
        else:
            x = x_or_img.to(INFER_DEVICE)
            with torch.no_grad(), amp_ctx:
                # TG-MNN uses forward_multitask; extract state_logits for eval
                if output_type == "tg_mnn" and hasattr(model, "forward_multitask"):
                    tg_out = model.forward_multitask(x)
                    out = tg_out.state_logits  # [B, 3]
                else:
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

    # ── Reproducibility manifest ──────────────────────────────────────────
    try:
        from quant_core.run_manifest import save_manifest as _save_manifest
        _manifest_path = ROOT / "doc" / "iterate_history" / f"eval_manifest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        _manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _save_manifest(
            _manifest_path,
            dataset_manifest_path=ROOT / "Dataset" / "binance_historical" / "manifest.json",
            seed=42,
            extra={"evaluator_version": "evaluate_all_checkpoints_v5", "round_trip_cost": ROUND_TRIP_COST},
        )
        _log(f"[eval] reproducibility manifest: {_manifest_path.name}")
    except Exception as _me:
        _log(f"[eval] WARNING: manifest generation failed: {_me}")

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
    datasets["trend"]    = load_trend_data(frames, seq_len=96, horizon=5)
    _log(f"[eval]   trend:   {len(datasets['trend']) if datasets['trend'] else 'FAIL'} samples")

    _log("[eval] → building mean_reversion dataset...")
    datasets["mr"]       = load_mr_data(frames, horizon=3)
    _log(f"[eval]   mr:      {len(datasets['mr']) if datasets['mr'] else 'FAIL'} samples")

    _log("[eval] → building scalper dataset (fractional diff — may be slow)...")
    datasets["scalper"]  = load_scalper_data(frames, seq_len=32, horizon=2)
    _log(f"[eval]   scalper: {len(datasets['scalper']) if datasets['scalper'] else 'FAIL'} samples")

    _log("[eval] → building stat_arb dataset...")
    # Pre-warm cache for both v1 (34 assets) and v2 (34×3=102 assets).
    # Per-model loop uses f"stat_arb_{detected_num_assets}" as key.
    _n_sym = min(34, len(frames))
    datasets[f"stat_arb_{_n_sym}"]        = load_stat_arb_data(frames, seq_len=64, horizon=10, num_assets=_n_sym)
    datasets[f"stat_arb_{_n_sym * 3}"]    = load_stat_arb_data(frames, seq_len=64, horizon=10, num_assets=_n_sym * 3)
    _log(f"[eval]   stat_arb_{_n_sym}:{len(datasets[f'stat_arb_{_n_sym}']) if datasets[f'stat_arb_{_n_sym}'] else 'FAIL'} "
         f"stat_arb_{_n_sym * 3}:{len(datasets[f'stat_arb_{_n_sym * 3}']) if datasets[f'stat_arb_{_n_sym * 3}'] else 'FAIL'} samples")

    _log("[eval] → building disc (chart image) dataset...")
    datasets["disc"]     = load_disc_data(frames, seq_len=32, horizon=5)
    _log(f"[eval]   disc:    {len(datasets['disc']) if datasets['disc'] else 'FAIL'} samples")

    _log("[eval] → building market_maker dataset...")
    datasets["mm"]       = load_mm_data(frames, n_steps=3000, state_dim=10)
    _log(f"[eval]   mm:      {len(datasets['mm']) if datasets['mm'] else 'FAIL'} samples")

    # Transformer was trained with seq_len=64 — build a separate slice
    _log("[eval] → building trend64 dataset (seq_len=64 for Transformer)...")
    datasets["trend64"]  = load_trend_data(frames, seq_len=64, horizon=5)
    _log(f"[eval]   trend64: {len(datasets['trend64']) if datasets['trend64'] else 'FAIL'} samples")

    # disc_multimodal: build real tabular features for DiscretionaryMultimodal
    _log("[eval] → building disc_multimodal dataset (img + tabular)...")
    datasets["disc_multimodal"] = load_disc_multimodal_data(frames, seq_len=32, horizon=5)
    _log(f"[eval]   disc_multimodal: {len(datasets['disc_multimodal']) if datasets['disc_multimodal'] else 'FAIL'} samples")

    # apv_pln: dual-stream price+volume for Oracle Teacher / LUPI evaluation
    _log("[eval] → building apv_pln dataset (dual-stream price+volume)...")
    datasets["apv_pln"] = load_apv_pln_data(frames, seq_len=32, horizon=5)
    _log(f"[eval]   apv_pln: {len(datasets['apv_pln']) if datasets['apv_pln'] else 'FAIL'} samples")

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
                    w = raw_state.get("in_proj.weight")
                    if w is None:
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
                datasets[key] = load_trend_data(frames, seq_len=target_seq, horizon=5)
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

        # ── RL models: use episodic evaluator, not one-shot forward pass ─────
        if manifest["archetype"] == "market_making_rl":
            try:
                target_sd = int(manifest["kwargs"].get("state_dim", 7))
                try:
                    raw_s = torch.load(str(CHECKPOINT_ROOT / manifest["ckpt"]), map_location="cpu")
                    if isinstance(raw_s, dict):
                        raw_s = raw_s.get("model_state_dict", raw_s.get("state_dict", raw_s))
                    w = raw_s.get("trunk.0.weight")
                    if w is None:
                        w = raw_s.get("net.0.weight")
                    if w is not None:
                        target_sd = int(w.shape[1])
                except Exception:
                    pass
                metrics = eval_rl_episode(model, frames, manifest["out"], state_dim=target_sd)
                elapsed = time.perf_counter() - t0
                _log(f"[eval] RL episode eval done  |  {elapsed:.2f}s")
            except Exception as e:
                _log(f"[eval] ✗ RL episode eval error: {e}")
                result["validation"] = {
                    "status": "RESUME_TRAINING_REQUIRED",
                    "reason": f"rl_eval_error: {e}",
                    "directional_accuracy": None, "sharpe": None,
                    "profit_factor": None, "max_drawdown": None,
                }
                results[model_name] = result
                _release_memory(model)
                continue

            dir_acc = metrics["directional_accuracy"]
            sharpe  = metrics["sharpe"]
            pf      = metrics["profit_factor"]
            mdd     = metrics["max_drawdown"]
            # RL pass criteria: Sharpe>0 (positive mean reward), win_rate>50%, MDD<85%
            strict_pass = sharpe > 0.0 and dir_acc > 0.50 and mdd < 0.85
            status = "PASSED" if strict_pass else "RESUME_TRAINING_REQUIRED"
            gate_str = "✓ PASSED" if strict_pass else "✗ FAILED"
            _log(f"[eval] Episode Win Rate : {dir_acc:.4f}  (gate > 0.50)")
            _log(f"[eval] Episode Sharpe   : {sharpe:.4f}  (gate > 0.0)")
            _log(f"[eval] Episode PF       : {pf:.4f}")
            _log(f"[eval] Episode MDD      : {mdd:.4f}  (gate < 0.85)")
            _log(f"[eval] KPI Gate         : {gate_str}")
            result["validation"] = {
                "status": status,
                "directional_accuracy": dir_acc,
                "sharpe": sharpe,
                "profit_factor": pf,
                "max_drawdown": mdd,
                "n_samples": 200,
                "device": DEVICE_NAME,
                "eval_mode": "episodic_rl",
            }
            results[model_name] = result
            _release_memory(model)
            continue

        # ── forward pass (all non-RL models) ──────────────────────────────
        try:
            is_multi = (manifest["data"] == "disc_multimodal")
            is_apv = (manifest["out"] == "apv_pln")

            # Load bin_centers from saved bin_meta.pt (APV-PLN only)
            bin_centers_tensor: torch.Tensor | None = None
            if is_apv:
                bin_meta_path = ckpt_path.parent / "bin_meta.pt"
                if bin_meta_path.exists():
                    try:
                        bm = torch.load(str(bin_meta_path), map_location="cpu", weights_only=False)
                        bin_centers_tensor = torch.tensor(bm["bin_centers"], dtype=torch.float32)
                    except Exception as bm_exc:
                        _log(f"[eval] ⚠  bin_meta.pt load error: {bm_exc} — using uniform bins")
                if bin_centers_tensor is None:
                    # Fallback: uniform bins across [-0.02, 0.02]
                    bin_centers_tensor = torch.linspace(-0.02, 0.02, 51)

            logits, labels, actual_rets = run_inference(
                model, ds, manifest["out"],
                is_multimodal=is_multi,
                bin_centers=bin_centers_tensor,
            )
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
        _ret_series_container: list[np.ndarray] = []
        metrics = compute_all_metrics(
            logits, labels, manifest["out"],
            actual_returns=actual_rets,
            _out_returns=_ret_series_container,
        )

        dir_acc = metrics["directional_accuracy"]
        sharpe  = metrics["sharpe"]
        pf      = metrics["profit_factor"]
        mdd     = metrics["max_drawdown"]

        strict_pass = passes_production_gates(metrics, require_directional_accuracy=True)
        status = "PASSED" if strict_pass else "RESUME_TRAINING_REQUIRED"

        gate_str = "PASSED" if strict_pass else "FAILED"
        _log(f"[eval] Directional Accuracy : {dir_acc:.4f}  (gate > {PRODUCTION_GATES.directional_accuracy_min})")
        _log(f"[eval] Sharpe Ratio         : {sharpe:.4f}  (gate > {PRODUCTION_GATES.sharpe_min})")
        _log(f"[eval] Profit Factor        : {pf:.4f}  (gate > {PRODUCTION_GATES.profit_factor_min})")
        _log(f"[eval] Max Drawdown         : {mdd:.4f}  (gate < {PRODUCTION_GATES.max_drawdown_max})")

        # V2.0 Divergence Gate: if a val_sharpe is available in the registry
        # entry, enforce the absolute gap limit before reporting PASSED.
        _val_sharpe_archived = result.get("validation", {}).get("val_sharpe") if isinstance(result.get("validation"), dict) else None
        if strict_pass and _val_sharpe_archived is not None:
            _abs_gap = abs(float(_val_sharpe_archived) - float(sharpe))
            if _abs_gap > PRODUCTION_GATES.sharpe_divergence_max_abs:
                strict_pass = False
                status = "DIVERGENCE_FAILED"
                _log(
                    f"[eval] DIVERGENCE-ALERT val_sharpe={_val_sharpe_archived:.4f} "
                    f"test_sharpe={sharpe:.4f} abs_gap={_abs_gap:.4f} "
                    f"(limit={PRODUCTION_GATES.sharpe_divergence_max_abs}) -> V2.0 divergence gate FAILED"
                )
                gate_str = "FAILED"

        _log(f"[eval] KPI Gate             : {gate_str}")

        # ── Phase 6: Walk-Forward + Monte Carlo robustness gates ──────────
        robustness_result: dict | None = None
        if strict_pass and _ROBUSTNESS_AVAILABLE and _ret_series_container:
            try:
                _raw_rets = _ret_series_container[0]
                _log(f"[eval] Running Phase 6 robustness suite on {len(_raw_rets):,} return bars ...")
                robustness_result = _run_robustness(
                    _raw_rets,
                    wf_windows=7,
                    mc_shuffles=1000,
                    wf_gate_pct=0.80,
                    mc_p95_mdd_gate=0.20,
                    seed=42,
                )
                _log(f"[eval] Walk-Forward: pct_pos_windows={robustness_result['walk_forward']['pct_positive_windows']:.2%}  pass={robustness_result['walk_forward']['passed']}")
                _log(f"[eval] Monte Carlo : p95_mdd={robustness_result['monte_carlo']['p95_mdd']:.4f}  pass={robustness_result['monte_carlo']['passed']}")
                _log(f"[eval] Robustness Gate: {'PASSED' if robustness_result['robustness_pass'] else 'FAILED'}")
                if strict_pass and not robustness_result["robustness_pass"]:
                    status = "ROBUSTNESS_FAILED"
            except Exception as _rob_exc:
                _log(f"[eval] WARNING: robustness suite error: {_rob_exc}")

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
            "strict_sharpe_divergence_max_abs": PRODUCTION_GATES.sharpe_divergence_max_abs,
        }
        if robustness_result is not None:
            result["validation"]["robustness"] = robustness_result
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
