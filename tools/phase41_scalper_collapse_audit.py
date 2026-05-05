from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant_core.scalper_data import build_scalper_datasets
from quant_core.scalper_training import _artifact_name, _build_model
from quant_core.shared_training import resolve_device, load_best_checkpoint

CFG_PATH = ROOT / "configs" / "scalper_phase4.yaml"
OUT_PATH = ROOT / "doc" / "training_more_27-4" / "phase41_scalper_audit.json"
MAX_VAL_SAMPLES = 200_000
MAX_SYMBOLS_AUDIT = 8
MAX_ROWS_PER_SYMBOL_AUDIT = 120_000


def _class_ratio(labels: np.ndarray) -> list[float]:
    counts = np.bincount(labels.astype(np.int64), minlength=3).astype(np.float64)
    total = max(float(counts.sum()), 1.0)
    return (counts / total).round(6).tolist()


@torch.no_grad()
def _pred_distribution(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[float], float]:
    model.eval()
    preds_all = []
    labels_all = []
    for x, y in loader:
        logits = model(x.to(device))
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
        preds_all.append(preds)
        labels_all.append(y.numpy().astype(np.int64))
    preds_cat = np.concatenate(preds_all, axis=0)
    y_cat = np.concatenate(labels_all, axis=0)
    pred_ratio = _class_ratio(preds_cat)
    acc = float((preds_cat == y_cat).mean())
    return pred_ratio, acc


def main() -> int:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    data_cfg = dict(cfg["data"])
    data_cfg["max_symbols"] = min(int(data_cfg.get("max_symbols", MAX_SYMBOLS_AUDIT)), MAX_SYMBOLS_AUDIT)
    data_cfg["max_rows_per_symbol"] = int(data_cfg.get("max_rows_per_symbol", MAX_ROWS_PER_SYMBOL_AUDIT))
    train_cfg = cfg["training"]

    datasets = build_scalper_datasets(data_cfg)
    train_labels = datasets.train.tensors[1].detach().cpu().numpy().astype(np.int64)
    val_labels = datasets.val.tensors[1].detach().cpu().numpy().astype(np.int64)
    test_labels = datasets.test.tensors[1].detach().cpu().numpy().astype(np.int64)

    device, backend = resolve_device(str(train_cfg.get("preferred_backend", "auto")))
    batch_size = int(train_cfg.get("batch_size", 1024))
    val_ds = datasets.val
    if len(val_ds) > MAX_VAL_SAMPLES:
        idx = np.linspace(0, len(val_ds) - 1, MAX_VAL_SAMPLES, dtype=np.int64)
        val_ds = Subset(val_ds, idx.tolist())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    report: dict[str, object] = {
        "backend": backend,
        "audit_caps": {
            "max_symbols": int(data_cfg["max_symbols"]),
            "max_rows_per_symbol": int(data_cfg["max_rows_per_symbol"]),
            "max_val_samples": int(MAX_VAL_SAMPLES),
        },
        "dataset": {
            "train_size": len(datasets.train),
            "val_size": len(datasets.val),
            "test_size": len(datasets.test),
            "train_class_ratio": _class_ratio(train_labels),
            "val_class_ratio": _class_ratio(val_labels),
            "test_class_ratio": _class_ratio(test_labels),
        },
        "models": {},
    }

    for model_name, model_cfg in cfg["models"].items():
        model_key = _artifact_name(model_name)
        ckpt_dir = Path(train_cfg["checkpoint_root"]) / model_key
        model = _build_model(model_name, input_dim=datasets.input_dim, seq_len=datasets.seq_len, cfg=model_cfg).to(device)
        if ckpt_dir.exists():
            try:
                load_best_checkpoint(model, ckpt_dir)
            except Exception as exc:
                report["models"][model_key] = {
                    "checkpoint_loaded": False,
                    "error": str(exc),
                }
                continue
        pred_ratio, acc = _pred_distribution(model, val_loader, device)
        flat_ratio = float(pred_ratio[1])
        report["models"][model_key] = {
            "checkpoint_loaded": True,
            "val_pred_class_ratio": pred_ratio,
            "val_accuracy": round(acc, 6),
            "flat_collapse_flag": flat_ratio >= 0.9,
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[phase41] scalper audit written: {OUT_PATH}")
    for k, v in report["models"].items():
        if isinstance(v, dict):
            print(f"{k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
