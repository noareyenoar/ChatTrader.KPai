from __future__ import annotations

import argparse
import time
from pathlib import Path
from copy import deepcopy

import yaml

from .shared_training import aggressive_cleanup
from .trend_data import build_trend_datasets
from .trend_training import train_one_model, write_registry


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _auto_regularize_remaining_queue(
    *,
    completed_model_name: str,
    result,
    train_cfg: dict,
    remaining_models_cfg: dict,
) -> None:
    if not bool(train_cfg.get("auto_regularize_on_divergence", True)):
        return
    if not (getattr(result, "divergence_alert", False) or getattr(result, "overfit_alert", False)):
        return

    dropout_step = float(train_cfg.get("auto_dropout_step", 0.10))
    max_dropout = float(train_cfg.get("auto_max_dropout", 0.50))
    weight_decay_mult = float(train_cfg.get("auto_weight_decay_multiplier", 2.0))
    simplify_layers = bool(train_cfg.get("auto_simplify_architecture", True))

    old_weight_decay = float(train_cfg["weight_decay"])
    train_cfg["weight_decay"] = float(old_weight_decay * weight_decay_mult)

    for next_name, next_cfg in remaining_models_cfg.items():
        if "dropout" in next_cfg:
            next_cfg["dropout"] = min(max_dropout, float(next_cfg["dropout"]) + dropout_step)
        if simplify_layers and isinstance(next_cfg.get("num_layers"), int) and int(next_cfg["num_layers"]) > 2:
            next_cfg["num_layers"] = int(next_cfg["num_layers"]) - 1
        if simplify_layers and isinstance(next_cfg.get("channels"), int) and int(next_cfg["channels"]) >= 128:
            next_cfg["channels"] = int(int(next_cfg["channels"]) * 0.75)
        if simplify_layers and isinstance(next_cfg.get("hidden_size"), int) and int(next_cfg["hidden_size"]) >= 256:
            next_cfg["hidden_size"] = int(int(next_cfg["hidden_size"]) * 0.75)

    print(
        f"[trend-run] auto-regularization applied after model={completed_model_name} "
        f"divergence_alert={getattr(result, 'divergence_alert', False)} "
        f"overfit_alert={getattr(result, 'overfit_alert', False)} "
        f"weight_decay={old_weight_decay}->{train_cfg['weight_decay']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Trend Follower models (Phase 4)")
    parser.add_argument("--config", type=str, default="configs/trend_phase4.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    models_cfg = deepcopy(cfg["models"])

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
    model_names = list(models_cfg.keys())
    for model_index, model_name in enumerate(model_names):
        model_cfg = models_cfg[model_name]
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
            f"divergence_alert={result.divergence_alert} overfit_alert={result.overfit_alert} "
            f"is_valid={result.is_valid}"
        )
        results.append(result)

        remaining_model_names = model_names[model_index + 1 :]
        remaining_models_cfg = {name: models_cfg[name] for name in remaining_model_names}
        _auto_regularize_remaining_queue(
            completed_model_name=model_name,
            result=result,
            train_cfg=train_cfg,
            remaining_models_cfg=remaining_models_cfg,
        )

        aggressive_cleanup(result)

    registry_path = Path(cfg.get("registry_path", "model_registry.json"))
    write_registry(results, registry_path)
    print(f"registry={registry_path} elapsed_s={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
