from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from .shared_training import aggressive_cleanup
from .scalper_data import build_scalper_datasets, save_scalper_scaler
from .scalper_training import train_scalper_model, write_scalper_registry


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Scalper models (Phase 4)")
    parser.add_argument("--config", type=str, default="configs/scalper_phase4.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    print(f"[scalper-run] config={args.config} backend={cfg['training'].get('preferred_backend', 'auto')} max_epochs={cfg['training']['max_epochs']}", flush=True)
    datasets = build_scalper_datasets(cfg["data"])
    print(f"[scalper-run] datasets ready train={len(datasets.train)} val={len(datasets.val)} test={len(datasets.test)} input_dim={datasets.input_dim} seq_len={datasets.seq_len}", flush=True)

    # Persist the fitted scaler into each model's checkpoint directory so the
    # evaluator can apply the same normalisation during OOS inference.
    checkpoint_root = Path(cfg["training"]["checkpoint_root"])
    started = time.time()
    results = []
    for model_name, model_cfg in cfg["models"].items():
        print(f"[scalper-run] launching model={model_name}", flush=True)
        result = train_scalper_model(
            name=model_name,
            model_cfg=model_cfg,
            train_ds=datasets.train,
            val_ds=datasets.val,
            test_ds=datasets.test,
            common_cfg=cfg["training"],
            input_dim=datasets.input_dim,
            seq_len=datasets.seq_len,
        )
        print(
            f"trained={result.model_name} backend={result.backend} "
            f"val_acc={result.val_accuracy:.4f} test_acc={result.test_accuracy:.4f} "
            f"latency={result.inference_ms:.2f}ms is_valid={result.is_valid}"
        )
        results.append(result)
        aggressive_cleanup(result)
        # Save the fitted scaler alongside this model's checkpoint
        if datasets.scaler is not None:
            from .scalper_training import _artifact_name
            model_ckpt_dir = checkpoint_root / _artifact_name(model_name)
            save_scalper_scaler(datasets.scaler, model_ckpt_dir)

    registry_path = Path(cfg.get("registry_path", "model_registry.json"))
    write_scalper_registry(results, registry_path)
    print(f"registry={registry_path} elapsed_s={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
