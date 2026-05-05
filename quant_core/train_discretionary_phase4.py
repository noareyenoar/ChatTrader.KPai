from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from .shared_training import aggressive_cleanup
from .discretionary_data import build_disc_datasets
from .discretionary_training import train_disc_model, write_disc_registry


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Discretionary models (Phase 4)")
    parser.add_argument("--config", type=str, default="configs/discretionary_phase4.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    print(f"[disc-run] config={args.config} backend={cfg['training'].get('preferred_backend', 'auto')} max_epochs={cfg['training']['max_epochs']}", flush=True)
    datasets = build_disc_datasets(cfg["data"])
    print(f"[disc-run] datasets ready train={len(datasets.train)} val={len(datasets.val)} test={len(datasets.test)} tab_input_dim={datasets.tab_input_dim}", flush=True)

    started = time.time()
    results = []
    for model_name, model_cfg in cfg["models"].items():
        print(f"[disc-run] launching model={model_name}", flush=True)
        result = train_disc_model(
            name=model_name,
            model_cfg=model_cfg,
            train_ds=datasets.train,
            val_ds=datasets.val,
            test_ds=datasets.test,
            common_cfg=cfg["training"],
            tab_input_dim=datasets.tab_input_dim,
        )
        print(
            f"trained={result.model_name} backend={result.backend} "
            f"val_acc={result.val_accuracy:.4f} test_f1={result.test_f1:.4f} "
            f"is_valid={result.is_valid}"
        )
        results.append(result)
        aggressive_cleanup(result)

    registry_path = Path(cfg.get("registry_path", "model_registry.json"))
    write_disc_registry(results, registry_path)
    print(f"registry={registry_path} elapsed_s={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
