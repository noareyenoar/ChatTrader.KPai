from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from .shared_training import aggressive_cleanup
from .trend_data import build_trend_datasets
from .trend_training import train_one_model, write_registry


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Trend Follower models (Phase 4)")
    parser.add_argument("--config", type=str, default="configs/trend_phase4.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    models_cfg = cfg["models"]

    print(
        f"[trend-run] config={args.config} backend={train_cfg.get('preferred_backend', 'auto')} "
        f"seq_len={train_cfg['seq_len']} max_epochs={train_cfg['max_epochs']}",
        flush=True,
    )
    datasets = build_trend_datasets(data_cfg)
    print(
        f"[trend-run] datasets ready train_windows={len(datasets.train)} "
        f"val_windows={len(datasets.val)} test_windows={len(datasets.test)} input_dim={datasets.input_dim}",
        flush=True,
    )

    started = time.time()
    results = []
    for model_name, model_cfg in models_cfg.items():
        print(f"[trend-run] launching model={model_name}", flush=True)
        result = train_one_model(
            name=model_name,
            model_cfg=model_cfg,
            train_ds=datasets.train,
            val_ds=datasets.val,
            test_ds=datasets.test,
            common_cfg=train_cfg,
            input_dim=datasets.input_dim,
        )
        print(
            f"trained={result.model_name} backend={result.backend} cuda_used={result.cuda_used} "
            f"val_acc={result.val_directional_acc:.4f} test_acc={result.test_directional_acc:.4f} "
            f"is_valid={result.is_valid}"
        )
        results.append(result)
        aggressive_cleanup(result)

    registry_path = Path(cfg.get("registry_path", "model_registry.json"))
    write_registry(results, registry_path)
    print(f"registry={registry_path} elapsed_s={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
