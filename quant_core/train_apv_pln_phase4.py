"""APV-PLN Phase-4 training launcher.

Usage:
    python -m quant_core.train_apv_pln_phase4 --config configs/apv_pln_phase4.yaml
    python -m quant_core.train_apv_pln_phase4 --config configs/apv_pln_phase4_smoke.yaml
"""
from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import yaml

from .apv_pln_data import build_apvpln_datasets
from .apv_pln_training import train_apv_pln, write_registry
from .shared_training import aggressive_cleanup


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train APV-PLN models (Phase 4)")
    parser.add_argument(
        "--config", type=str, default="configs/apv_pln_phase4.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    models_cfg: dict = deepcopy(cfg["models"])

    print(
        f"[apv-run] config={args.config} "
        f"backend={train_cfg.get('preferred_backend', 'auto')} "
        f"seq_len={train_cfg['seq_len']} horizon={train_cfg['horizon']} "
        f"max_epochs={train_cfg['max_epochs']}",
        flush=True,
    )

    # ── Build datasets once for all model variants ────────────────────────────
    datasets = build_apvpln_datasets(data_cfg)
    print(
        f"[apv-run] datasets ready "
        f"train={len(datasets.train)} val={len(datasets.val)} test={len(datasets.test)} "
        f"num_bins={datasets.num_bins} "
        f"bin_min={datasets.bin_min:.6f} bin_max={datasets.bin_max:.6f}",
        flush=True,
    )

    # ── Train each model variant ──────────────────────────────────────────────
    started = time.time()
    results = []

    for model_name, model_cfg in models_cfg.items():
        print(f"[apv-run] launching model={model_name}", flush=True)
        result = train_apv_pln(
            name=model_name,
            model_cfg=model_cfg,
            datasets=datasets,
            common_cfg=train_cfg,
        )
        print(
            f"[apv-run] done model={result.model_name} "
            f"backend={result.backend} "
            f"val_sharpe={result.val_sharpe:.4f} test_sharpe={result.test_sharpe:.4f} "
            f"test_acc={result.test_directional_acc:.4f} "
            f"is_valid={result.is_valid} "
            f"divergence_alert={result.divergence_alert}",
            flush=True,
        )
        results.append(result)
        aggressive_cleanup(result)

    # ── Write registry ────────────────────────────────────────────────────────
    registry_path = Path(cfg.get("registry_path", "model_registry.json"))
    write_registry(results, registry_path)

    elapsed = time.time() - started
    valid_count = sum(1 for r in results if r.is_valid)
    print(
        f"[apv-run] COMPLETE "
        f"models_trained={len(results)} valid={valid_count} "
        f"registry={registry_path} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
