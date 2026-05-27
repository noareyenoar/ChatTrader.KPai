from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.config import PipelineConfig
from data_pipeline.features import FeatureFactory
from data_pipeline.quality_gate import DataQualityGate
from quant_core.trend_data import RollingWindowDataset, build_trend_datasets
from quant_core.trend_models import TrendTCNModel
from quant_core.trend_training import evaluate_model, resolve_device, set_global_seed


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_trend_feature_static_audit() -> dict[str, Any]:
    src = inspect.getsource(FeatureFactory.build_trend_features)
    has_negative_shift = "shift(-" in src
    has_centered_rolling = "center=True" in src.replace(" ", "")
    return {
        "function": "FeatureFactory.build_trend_features",
        "has_negative_shift": has_negative_shift,
        "has_centered_rolling": has_centered_rolling,
        "is_causal_static_pass": not (has_negative_shift or has_centered_rolling),
    }


def _target_alignment_audit(data_cfg: dict[str, Any], sample_checks: int = 8) -> dict[str, Any]:
    pipe_cfg = PipelineConfig(
        dataset_dir=Path(data_cfg["dataset_dir"]),
        manifest_path=Path(data_cfg["manifest_path"]),
        min_history_bars=int(data_cfg["min_history_bars"]),
        purge_gap_bars=int(data_cfg["purge_gap_bars"]),
    )
    gate = DataQualityGate(pipe_cfg)
    accepted = [r for r in gate.evaluate() if r.decision == "ACCEPT"]
    if not accepted:
        raise RuntimeError("No accepted symbols available for alignment audit")

    symbol = accepted[0].symbol
    frame = pd.read_parquet(
        pipe_cfg.dataset_dir / f"{symbol}.parquet",
        columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    feat = FeatureFactory.build_trend_features(frame)
    horizon = int(data_cfg["horizon"])
    fwd_return = (feat["close"].shift(-horizon) / feat["close"]) - 1.0

    feat["target_label"] = (fwd_return > 0).astype(np.float32)
    feat["target_return"] = fwd_return
    feat = feat[
        [
            "timestamp",
            "close",
            "log_return",
            "zscore_close_64",
            "ema_spread",
            "atr_14",
            "price_slope_20",
            "target_label",
            "target_return",
        ]
    ].dropna().reset_index(drop=True)

    independent = (feat["close"].shift(-horizon) / feat["close"]) - 1.0
    diff = (independent - feat["target_return"]).abs().dropna()
    max_abs_target_return_diff = float(diff.max()) if len(diff) else 0.0

    sample_rows = np.linspace(0, max(0, len(feat) - horizon - 1), num=min(sample_checks, max(1, len(feat) - horizon)), dtype=int)
    checks = []
    for idx in sample_rows.tolist():
        t_end = feat.loc[idx, "timestamp"]
        t_label = feat.loc[idx + horizon, "timestamp"]
        checks.append(
            {
                "row_idx": int(idx),
                "feature_time": str(t_end),
                "label_horizon_time": str(t_label),
                "strictly_future": bool(t_label > t_end),
            }
        )

    strict_future_ok = all(item["strictly_future"] for item in checks)
    return {
        "symbol": symbol,
        "rows_after_dropna": int(len(feat)),
        "horizon": horizon,
        "max_abs_target_return_diff": max_abs_target_return_diff,
        "strict_future_checks": checks,
        "strict_future_all_pass": strict_future_ok,
        "alignment_pass": strict_future_ok and max_abs_target_return_diff < 1e-12,
    }


def _shuffle_labels(train_ds: RollingWindowDataset, seed: int) -> RollingWindowDataset:
    rng = np.random.default_rng(seed)
    shuffled_targets: list[np.ndarray] = []
    features_np: list[np.ndarray] = []
    returns_np: list[np.ndarray] = []

    for feat_t, target_t, ret_t in zip(train_ds.features_list, train_ds.target_list, train_ds.returns_list or []):
        y = target_t.detach().cpu().numpy().copy()
        rng.shuffle(y)
        shuffled_targets.append(y.astype(np.float32, copy=False))
        features_np.append(feat_t.detach().cpu().numpy().astype(np.float32, copy=False))
        returns_np.append(ret_t.detach().cpu().numpy().astype(np.float32, copy=False))

    if train_ds.returns_list is None:
        raise RuntimeError("Expected returns_list for trend forensic test")

    return RollingWindowDataset(
        features_list=features_np,
        target_list=shuffled_targets,
        seq_len=train_ds.seq_len,
        returns_list=returns_np,
    )


def _annualization_factor() -> float:
    return float(np.sqrt(252 * 24 * 12))


def _naive_signal_sharpe(returns: np.ndarray, signal: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    prev = np.concatenate([signal[:1], signal[:-1]])
    flips = (signal != prev).astype(np.float32)
    pnl = signal * returns - 0.0004 * flips
    std = float(np.std(pnl)) + 1e-8
    return float(np.mean(pnl) / std * _annualization_factor())


def _dataset_returns(ds: RollingWindowDataset) -> np.ndarray:
    if ds.returns_list is None:
        return np.array([], dtype=np.float32)
    return np.concatenate([r.detach().cpu().numpy().astype(np.float32) for r in ds.returns_list])


def _baseline_sharpes(val_ds: RollingWindowDataset, test_ds: RollingWindowDataset, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    val_ret = _dataset_returns(val_ds)
    test_ret = _dataset_returns(test_ds)

    val_long = np.ones_like(val_ret, dtype=np.float32)
    test_long = np.ones_like(test_ret, dtype=np.float32)
    val_short = -np.ones_like(val_ret, dtype=np.float32)
    test_short = -np.ones_like(test_ret, dtype=np.float32)
    val_rand = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=len(val_ret))
    test_rand = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=len(test_ret))

    return {
        "val_always_long_sharpe": _naive_signal_sharpe(val_ret, val_long),
        "test_always_long_sharpe": _naive_signal_sharpe(test_ret, test_long),
        "val_always_short_sharpe": _naive_signal_sharpe(val_ret, val_short),
        "test_always_short_sharpe": _naive_signal_sharpe(test_ret, test_short),
        "val_random_signal_sharpe": _naive_signal_sharpe(val_ret, val_rand),
        "test_random_signal_sharpe": _naive_signal_sharpe(test_ret, test_rand),
    }


def _run_shuffled_label_probe(config: dict[str, Any], epochs: int, max_train_batches: int, seed: int) -> dict[str, Any]:
    data_cfg = deepcopy(config["data"])
    train_cfg = deepcopy(config["training"])

    # Fast forensic probe: keep Iron Wall logic, limit scope for rapid leakage falsification.
    data_cfg["max_symbols"] = int(min(3, int(data_cfg.get("max_symbols", 3))))
    data_cfg["max_rows_per_symbol"] = int(data_cfg.get("max_rows_per_symbol", 15000) or 15000)

    set_global_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    datasets = build_trend_datasets(data_cfg)
    shuffled_train = _shuffle_labels(datasets.train, seed=seed)
    baseline = _baseline_sharpes(datasets.val, datasets.test, seed=seed)

    device, backend = resolve_device(str(train_cfg.get("preferred_backend", "auto")))
    model = TrendTCNModel(
        input_dim=datasets.input_dim,
        channels=64,
        dropout=0.2,
    ).to(device)

    train_loader = DataLoader(shuffled_train, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
    val_loader = DataLoader(datasets.val, batch_size=512, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(datasets.test, batch_size=512, shuffle=False, num_workers=0, pin_memory=False)

    criterion = nn.BCEWithLogitsLoss()
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epoch_metrics: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            out = model(x).squeeze(-1)
            loss = criterion(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item())
            seen += 1
            if batch_idx >= max_train_batches:
                break

        val_m = evaluate_model(model, val_loader, device)
        epoch_metrics.append(
            {
                "epoch": float(epoch),
                "train_loss": float(running / max(seen, 1)),
                "val_loss": float(val_m["loss"]),
                "val_acc": float(val_m["directional_acc"]),
                "val_sharpe": float(val_m["sharpe"]),
            }
        )

    final_val = evaluate_model(model, val_loader, device)
    final_test = evaluate_model(model, test_loader, device)

    suspicious_leakage = bool(
        final_test["sharpe"] > 1.2
        and final_test["sharpe"] > (baseline["test_always_long_sharpe"] + 1.0)
    )
    return {
        "backend": backend,
        "device": str(device),
        "epochs": epochs,
        "max_train_batches": max_train_batches,
        "train_windows": len(shuffled_train),
        "val_windows": len(datasets.val),
        "test_windows": len(datasets.test),
        "epoch_metrics": epoch_metrics,
        "naive_baseline_sharpes": baseline,
        "final_val": {
            "loss": float(final_val["loss"]),
            "acc": float(final_val["directional_acc"]),
            "sharpe": float(final_val["sharpe"]),
            "profit_factor": float(final_val["profit_factor"]),
            "max_drawdown": float(final_val["max_drawdown"]),
        },
        "final_test": {
            "loss": float(final_test["loss"]),
            "acc": float(final_test["directional_acc"]),
            "sharpe": float(final_test["sharpe"]),
            "profit_factor": float(final_test["profit_factor"]),
            "max_drawdown": float(final_test["max_drawdown"]),
        },
        "suspicious_leakage_signal": suspicious_leakage,
        "suspicious_rule": "Flag only if shuffled-label TEST sharpe > 1.2 and > always-long TEST sharpe + 1.0",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend emergency forensic audit")
    parser.add_argument("--config", required=True, help="Path to trend config yaml")
    parser.add_argument("--epochs", type=int, default=3, help="Probe epochs on shuffled labels")
    parser.add_argument("--max-train-batches", type=int, default=250, help="Batch cap per epoch for fast probe")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        default="doc/trend_forensic_audit_report.json",
        help="Report output path",
    )
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))

    feature_audit = _build_trend_feature_static_audit()
    alignment_audit = _target_alignment_audit(cfg["data"])
    shuffled_probe = _run_shuffled_label_probe(
        config=cfg,
        epochs=int(args.epochs),
        max_train_batches=int(args.max_train_batches),
        seed=int(args.seed),
    )

    decision = {
        "pivot_to_mean_reversion": bool(
            (not shuffled_probe["suspicious_leakage_signal"])
            and feature_audit["is_causal_static_pass"]
            and alignment_audit["alignment_pass"]
        ),
        "reason": (
            "No structural leakage signal in shuffled-label test and causality/alignment checks passed"
            if (
                (not shuffled_probe["suspicious_leakage_signal"])
                and feature_audit["is_causal_static_pass"]
                and alignment_audit["alignment_pass"]
            )
            else "Leakage risk remains or causality/alignment check failed"
        ),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config)),
        "feature_factory_static_audit": feature_audit,
        "target_alignment_audit": alignment_audit,
        "shuffled_label_probe": shuffled_probe,
        "decision": decision,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()