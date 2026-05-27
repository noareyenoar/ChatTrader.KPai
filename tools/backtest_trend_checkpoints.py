#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_pipeline.config import PipelineConfig
from data_pipeline.features import FeatureFactory
from data_pipeline.quality_gate import DataQualityGate
from data_pipeline.splitter import IronWallSplitter
from quant_core.trend_models import TrendLSTMModel, TrendTCNModel, TrendTransformerModel
from quant_core.trend_training import resolve_device

FEATURE_COLUMNS = [
    "log_return",
    "zscore_close_64",
    "ema_spread",
    "atr_14",
    "price_slope_20",
]
ROUND_TRIP_COST = 0.0004


def _annualization_factor() -> float:
    return math.sqrt(252 * 24 * 12)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_symbol(path: Path) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
    frame = pd.read_parquet(path, columns=cols)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _stride_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    if len(arr) <= seq_len:
        return np.empty((0, seq_len, arr.shape[1]), dtype=arr.dtype)
    shape = (len(arr) - seq_len + 1, seq_len, arr.shape[1])
    strides = (arr.strides[0], arr.strides[0], arr.strides[1])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides).copy()


def _build_model(model_key: str, input_dim: int, seq_len: int, cfg: dict[str, Any]) -> torch.nn.Module:
    if model_key == "lstm":
        return TrendLSTMModel(input_dim=input_dim, **cfg)
    if model_key == "transformer":
        return TrendTransformerModel(input_dim=input_dim, seq_len=seq_len, **cfg)
    if model_key == "tcn":
        return TrendTCNModel(input_dim=input_dim, **cfg)
    raise ValueError(f"Unknown model key: {model_key}")


def _model_key_from_dirname(dirname: str) -> str:
    lname = dirname.lower()
    if "lstm" in lname:
        return "lstm"
    if "transformer" in lname:
        return "transformer"
    if "tcn" in lname:
        return "tcn"
    raise ValueError(f"Cannot infer model key from {dirname}")


def _max_drawdown_duration(equity: np.ndarray) -> int:
    peak = np.maximum.accumulate(equity)
    underwater = equity < peak
    max_run = 0
    current = 0
    for flag in underwater:
        if flag:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return int(max_run)


def _profit_factor(pnl: np.ndarray) -> float:
    gains = float(np.sum(pnl[pnl > 0]))
    losses = float(abs(np.sum(pnl[pnl < 0])))
    if losses == 0.0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _trade_pnls(signal: np.ndarray, pnl_steps: np.ndarray) -> list[float]:
    if len(signal) == 0:
        return []
    out: list[float] = []
    running = float(pnl_steps[0])
    for idx in range(1, len(signal)):
        if signal[idx] != signal[idx - 1]:
            out.append(running)
            running = float(pnl_steps[idx])
        else:
            running += float(pnl_steps[idx])
    out.append(running)
    return out


def _asset_report(model: torch.nn.Module, device: torch.device, raw: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any] | None:
    seq_len = int(cfg["data"]["seq_len"])
    horizon = int(cfg["data"]["horizon"])
    feat = FeatureFactory.build_trend_features(raw)
    fwd_return = (feat["close"].shift(-horizon) / feat["close"]) - 1.0
    feat["target_label"] = (fwd_return > 0).astype(np.float32)
    feat["target_return"] = fwd_return
    feat = feat[["timestamp", *FEATURE_COLUMNS, "target_label", "target_return"]].dropna().reset_index(drop=True)
    cap_rows = int(cfg["data"].get("max_rows_per_symbol", 0))
    if cap_rows > 0:
        feat = feat.iloc[-cap_rows:].copy()

    split = IronWallSplitter(purge_gap_bars=int(cfg["data"]["purge_gap_bars"])).split(feat, time_col="timestamp")
    scaler = FeatureFactory.fit_scaler_train_only(split.train, FEATURE_COLUMNS)
    test = FeatureFactory.transform_with_scaler(split.test, scaler)

    x_test = test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test["target_label"].to_numpy(dtype=np.float32)
    r_test = test["target_return"].to_numpy(dtype=np.float32)
    ts_test = test["timestamp"].astype(str).to_numpy()

    seqs = _stride_sequences(x_test, seq_len)
    if len(seqs) <= horizon:
        return None

    labels = y_test[seq_len - 1: seq_len - 1 + len(seqs)]
    returns = r_test[seq_len - 1: seq_len - 1 + len(seqs)]
    times = ts_test[seq_len - 1: seq_len - 1 + len(seqs)]

    valid = len(seqs) - horizon
    seqs = seqs[:valid]
    labels = labels[:valid]
    returns = returns[:valid]
    times = times[:valid]

    batch_size = 1024
    pred_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            batch = torch.from_numpy(seqs[start:start + batch_size]).to(device)
            logits = model(batch).squeeze(-1).detach().cpu().numpy()
            pred_chunks.append(logits)
    logits = np.concatenate(pred_chunks)
    signal = np.where(logits > 0, 1.0, -1.0)
    trade_occurs = np.ones_like(signal)
    trade_occurs[1:] = (signal[1:] != signal[:-1]).astype(np.float32)
    pnl_steps = signal * returns - ROUND_TRIP_COST * trade_occurs
    equity_curve = 1.0 + np.cumsum(pnl_steps)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / np.maximum(peak, 1e-8)
    trade_pnls = _trade_pnls(signal, pnl_steps)
    win_rate = float(np.mean(np.asarray(trade_pnls) > 0)) if trade_pnls else 0.0
    sharpe = float((np.mean(pnl_steps) / (np.std(pnl_steps) + 1e-8)) * _annualization_factor()) if len(pnl_steps) > 1 else 0.0
    directional_acc = float(np.mean((logits > 0).astype(np.float32) == labels)) if len(labels) else 0.0

    return {
        "bars": int(len(pnl_steps)),
        "directional_accuracy": directional_acc,
        "sharpe": sharpe,
        "profit_factor": _profit_factor(pnl_steps),
        "max_drawdown": float(np.max(drawdown)) if len(drawdown) else 0.0,
        "max_drawdown_duration_bars": _max_drawdown_duration(equity_curve),
        "win_rate": win_rate,
        "trade_count": len(trade_pnls),
        "equity_curve": [float(x) for x in equity_curve.tolist()],
        "timestamps": times.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-asset backtest completed trend checkpoints")
    parser.add_argument("--config", default="configs/trend_phase4_fast_fail.yaml")
    parser.add_argument("--models", nargs="*", default=["TCN_Trend_v1", "Transformer_Trend_v1", "LSTM_Trend_v1"])
    parser.add_argument("--output-dir", default="doc")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    data_cfg = cfg["data"]
    model_cfgs = cfg["models"]

    pipe_cfg = PipelineConfig(
        dataset_dir=Path(data_cfg["dataset_dir"]),
        manifest_path=Path(data_cfg["manifest_path"]),
        min_history_bars=int(data_cfg["min_history_bars"]),
        purge_gap_bars=int(data_cfg["purge_gap_bars"]),
    )
    accepted = [r for r in DataQualityGate(pipe_cfg).evaluate() if r.decision == "ACCEPT"]
    symbols = [r.symbol for r in accepted[: int(data_cfg["max_symbols"])]]
    device, backend = resolve_device("auto")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_index: dict[str, Any] = {}
    for model_dir_name in args.models:
        ckpt_path = Path("models/checkpoints/trend") / model_dir_name / "model_best.pt"
        if not ckpt_path.exists():
            continue
        model_key = _model_key_from_dirname(model_dir_name)
        model = _build_model(model_key, input_dim=len(FEATURE_COLUMNS), seq_len=int(data_cfg["seq_len"]), cfg=model_cfgs[model_key])
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.to(device)

        per_asset: dict[str, Any] = {}
        pnl_all: list[np.ndarray] = []
        for symbol in symbols:
            raw = _load_symbol(Path(data_cfg["dataset_dir"]) / f"{symbol}.parquet")
            report = _asset_report(model, device, raw, cfg)
            if report is None:
                continue
            per_asset[symbol] = report
            pnl_all.append(np.diff(np.array(report["equity_curve"], dtype=np.float64), prepend=1.0))

        if not per_asset:
            continue

        pnl = np.concatenate(pnl_all) if pnl_all else np.array([], dtype=np.float64)
        agg = {
            "model": model_dir_name,
            "backend": backend,
            "asset_count": len(per_asset),
            "aggregate_sharpe": float((np.mean(pnl) / (np.std(pnl) + 1e-8)) * _annualization_factor()) if len(pnl) > 1 else 0.0,
            "aggregate_profit_factor": _profit_factor(pnl),
            "aggregate_max_drawdown": float(np.max([(v["max_drawdown"]) for v in per_asset.values()])),
            "aggregate_win_rate": float(np.mean([v["win_rate"] for v in per_asset.values()])),
            "per_asset": per_asset,
        }
        out_path = out_dir / f"trend_backtest_{model_dir_name}.json"
        out_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        aggregate_index[model_dir_name] = {
            "report_path": str(out_path).replace("\\", "/"),
            "aggregate_sharpe": agg["aggregate_sharpe"],
            "aggregate_profit_factor": agg["aggregate_profit_factor"],
            "aggregate_max_drawdown": agg["aggregate_max_drawdown"],
            "aggregate_win_rate": agg["aggregate_win_rate"],
            "asset_count": agg["asset_count"],
        }
        model.cpu()

    index_path = out_dir / "trend_backtest_index.json"
    index_path.write_text(json.dumps(aggregate_index, indent=2), encoding="utf-8")
    print(json.dumps(aggregate_index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
