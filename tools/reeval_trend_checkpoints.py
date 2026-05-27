#!/usr/bin/env python3
"""Re-evaluate existing Trend checkpoints with the fixed Sharpe metric (no per-window tx cost).

Usage:
    python tools/reeval_trend_checkpoints.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_core.trend_data import build_trend_datasets
from quant_core.trend_training import evaluate_model, _log, _model_artifact_name
from quant_core.trend_models import TrendLSTMModel, TrendTCNModel, TrendTransformerModel
from quant_core.shared_training import append_registry

CONFIG_PATH = ROOT / "configs" / "trend_phase4_v2.yaml"
CHECKPOINT_ROOT = ROOT / "models" / "checkpoints" / "trend"
REGISTRY_PATH = ROOT / "model_registry.json"

MODEL_CONSTRUCTORS = {
    "tcn": lambda input_dim, seq_len, cfg: TrendTCNModel(input_dim=input_dim, **cfg),
    "transformer": lambda input_dim, seq_len, cfg: TrendTransformerModel(input_dim=input_dim, seq_len=seq_len, **cfg),
    "lstm": lambda input_dim, seq_len, cfg: TrendLSTMModel(input_dim=input_dim, **cfg),
}


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg: dict = cfg["data"]
    train_cfg: dict = cfg["training"]
    models_cfg: dict = cfg["models"]

    # Merge data + training keys for build_trend_datasets
    combined = {**data_cfg, **train_cfg}

    _log(f"[reeval] [{_ts()}] Loading datasets from config={CONFIG_PATH}")
    datasets = build_trend_datasets(combined)
    device = torch.device("cpu")

    from torch.utils.data import DataLoader
    val_loader = DataLoader(datasets.val, batch_size=1024, shuffle=False)
    test_loader = DataLoader(datasets.test, batch_size=1024, shuffle=False)

    input_dim = datasets.input_dim
    seq_len = int(combined["seq_len"])
    divergence_limit = float(train_cfg.get("sharpe_divergence_max_abs", 3.0))

    results = []
    for arch in ("tcn", "transformer", "lstm"):
        artifact_name = _model_artifact_name(arch)
        ckpt_path = CHECKPOINT_ROOT / artifact_name / "model_best.pt"

        if not ckpt_path.exists():
            _log(f"[reeval] [{_ts()}] {arch}: SKIP — checkpoint not found at {ckpt_path}")
            continue

        _log(f"[reeval] [{_ts()}] {arch}: loading checkpoint from {ckpt_path}")
        model = MODEL_CONSTRUCTORS[arch](input_dim, seq_len, models_cfg[arch]).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        val_m = evaluate_model(model, val_loader, device)
        test_m = evaluate_model(model, test_loader, device)

        abs_gap = abs(val_m["sharpe"] - test_m["sharpe"])
        divergence_alert = abs_gap > divergence_limit

        is_valid = (
            val_m["directional_acc"] > 0.52
            and test_m["directional_acc"] > 0.52
            and val_m["sharpe"] > 1.0
            and test_m["sharpe"] > 1.0
            and test_m["profit_factor"] > 1.3
            and test_m["max_drawdown"] < 0.20
            and not divergence_alert
        )

        _log(
            f"[reeval] [{_ts()}] {arch}: "
            f"val_acc={val_m['directional_acc']:.4f} test_acc={test_m['directional_acc']:.4f} "
            f"val_sharpe={val_m['sharpe']:.4f} test_sharpe={test_m['sharpe']:.4f} "
            f"abs_gap={abs_gap:.4f} (limit={divergence_limit}) "
            f"pf={test_m['profit_factor']:.4f} mdd={test_m['max_drawdown']:.4f} "
            f"val_flip={val_m['flip_rate']:.3f} test_flip={test_m['flip_rate']:.3f} "
            f"is_valid={is_valid}"
        )

        if divergence_alert:
            _log(f"[reeval] [{_ts()}] {arch}: DIVERGENCE_ALERT gap={abs_gap:.4f}")
        if is_valid:
            _log(f"[reeval] [{_ts()}] {arch}: *** PASSED V2.0 GATES ***")
        else:
            _log(f"[reeval] [{_ts()}] {arch}: FAILED — gates not met")

        results.append({
            "artifact_name": artifact_name,
            "architecture_name": artifact_name,
            "archetype": "trend_follower",
            "weights_path": str(ckpt_path).replace("\\", "/"),
            "is_valid": is_valid,
            "val_sharpe": round(val_m["sharpe"], 6),
            "test_sharpe": round(test_m["sharpe"], 6),
            "val_directional_acc": round(val_m["directional_acc"], 6),
            "test_directional_acc": round(test_m["directional_acc"], 6),
            "test_profit_factor": round(test_m["profit_factor"], 6),
            "test_max_drawdown": round(test_m["max_drawdown"], 6),
            "val_flip_rate": round(val_m["flip_rate"], 4),
            "test_flip_rate": round(test_m["flip_rate"], 4),
            "divergence_alert": divergence_alert,
            "abs_sharpe_gap": round(abs_gap, 4),
            "eval_timestamp": _ts(),
        })

    # Write results to a JSON report
    report_path = ROOT / "doc" / "reeval_trend_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _log(f"[reeval] [{_ts()}] Report written to {report_path}")

    # Update model_registry.json for is_valid entries
    registry_path = ROOT / "model_registry.json"
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    for res in results:
        # Find and update existing entry or append new one
        found = False
        for entry in registry.get("models", []):
            if entry.get("artifact_name") == res["artifact_name"] or entry.get("architecture_name") == res["artifact_name"]:
                entry["is_valid"] = res["is_valid"]
                entry["val_sharpe"] = res["val_sharpe"]
                entry["test_sharpe"] = res["test_sharpe"]
                entry["val_directional_acc"] = res["val_directional_acc"]
                entry["test_directional_acc"] = res["test_directional_acc"]
                entry["test_profit_factor"] = res["test_profit_factor"]
                entry["test_max_drawdown"] = res["test_max_drawdown"]
                entry["val_flip_rate"] = res["val_flip_rate"]
                entry["test_flip_rate"] = res["test_flip_rate"]
                entry["divergence_alert"] = res["divergence_alert"]
                found = True
                break
        if not found:
            registry.setdefault("models", []).append(res)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    _log(f"[reeval] [{_ts()}] Registry updated at {registry_path}")

    # Summary
    passed = [r for r in results if r["is_valid"]]
    failed = [r for r in results if not r["is_valid"]]
    _log(f"\n[reeval] ════ SUMMARY ════")
    _log(f"[reeval] PASSED ({len(passed)}): {[r['artifact_name'] for r in passed]}")
    _log(f"[reeval] FAILED ({len(failed)}): {[r['artifact_name'] for r in failed]}")
    _log(f"[reeval] Phase 4 Trend gate status: {len(passed)}/3 models valid")


if __name__ == "__main__":
    main()
