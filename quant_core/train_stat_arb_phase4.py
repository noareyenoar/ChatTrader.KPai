from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from .shared_training import aggressive_cleanup
from .stat_arb_data import build_stat_arb_datasets
from .stat_arb_training import train_stat_arb_model, write_stat_arb_registry


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Statistical Arbitrage models (Phase 4)")
    parser.add_argument("--config", type=str, default="configs/stat_arb_phase4.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    print(f"[stat-arb-run] config={args.config} backend={cfg['training'].get('preferred_backend', 'auto')} max_epochs={cfg['training']['max_epochs']}", flush=True)
    datasets = build_stat_arb_datasets(cfg["data"])
    print(f"[stat-arb-run] datasets ready train={len(datasets.train)} val={len(datasets.val)} test={len(datasets.test)} assets={datasets.num_assets} seq_len={datasets.seq_len}", flush=True)

    started = time.time()
    results = []
    for model_name, model_cfg in cfg["models"].items():
        print(f"[stat-arb-run] launching model={model_name}", flush=True)
        result = train_stat_arb_model(
            name=model_name,
            model_cfg=model_cfg,
            train_ds=datasets.train,
            val_ds=datasets.val,
            test_ds=datasets.test,
            common_cfg=cfg["training"],
            num_assets=datasets.num_assets,
            seq_len=datasets.seq_len,
        )
        print(
            f"trained={result.model_name} backend={result.backend} "
            f"val_mae={result.val_mae:.4f} test_mae={result.test_mae:.4f} "
            f"tracking_err={result.test_tracking_error:.4f} is_valid={result.is_valid}"
        )
        results.append(result)
        aggressive_cleanup(result)

    registry_path = Path(cfg.get("registry_path", "model_registry.json"))
    write_stat_arb_registry(results, registry_path)
    print(f"registry={registry_path} elapsed_s={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
