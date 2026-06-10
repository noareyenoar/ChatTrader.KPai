"""Discretionary training loop — Phase 4.

Loss:     CrossEntropyLoss (3-class)
Metrics:  Accuracy, F1-Score (macro), Sharpe, MaxDD, ProfitFactor
Special:  Multimodal model receives both image and tabular tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.gpu_utils import cleanup_cuda
from .interfaces import TrendModelInterface
from .discretionary_models import (
    DiscretionaryCNNChart,
    DiscretionaryMultimodal,
    DiscretionaryViT,
)
from .shared_training import (
    append_working_log,
    append_registry,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe,
    load_best_checkpoint,
    make_optimizer,
    resolve_device,
    save_checkpoint,
    set_global_seed,
)


@dataclass
class DiscResult:
    model_name: str
    checkpoint_dir: str
    val_accuracy: float
    test_accuracy: float
    val_f1: float
    test_f1: float
    val_sharpe: float
    test_sharpe: float
    test_profit_factor: float
    test_max_drawdown: float
    is_valid: bool
    backend: str
    cuda_used: bool


def _build_model(name: str, tab_input_dim: int, cfg: dict[str, Any]) -> nn.Module:
    if name == "vit":
        return DiscretionaryViT(**cfg)
    if name == "multimodal":
        return DiscretionaryMultimodal(tab_input_dim=tab_input_dim, **cfg)
    if name == "cnn_chart":
        return DiscretionaryCNNChart(**cfg)
    raise ValueError(f"Unknown discretionary model: {name}")


def _artifact_name(name: str) -> str:
    return {
        "vit": "ViT_Disc_v1",
        "multimodal": "Multimodal_Disc_v1",
        "cnn_chart": "CNNChart_Disc_v1",
    }[name]


def _log(message: str) -> None:
    print(message, flush=True)


def _is_multimodal(name: str) -> bool:
    return name == "multimodal"


def _f1_macro(preds: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> float:
    f1s = []
    for c in range(num_classes):
        tp = float(np.sum((preds == c) & (labels == c)))
        fp = float(np.sum((preds == c) & (labels != c)))
        fn = float(np.sum((preds != c) & (labels == c)))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return float(np.mean(f1s))


@torch.no_grad()
def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, multimodal: bool
) -> dict[str, float]:
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        img, tab, y = batch
        img = img.to(device)
        if multimodal:
            logits = model(img, tab.to(device))
        else:
            logits = model(img)
        all_logits.append(logits.detach().cpu().float())
        all_y.append(y.cpu())

    logits_cat = torch.cat(all_logits)
    y_cat = torch.cat(all_y).numpy()
    preds = logits_cat.argmax(dim=1).numpy()

    label_map = np.array([-1.0, 0.0, 1.0])
    pnl = label_map[preds] * label_map[y_cat]
    loss_val = float(nn.CrossEntropyLoss()(logits_cat, torch.tensor(y_cat, dtype=torch.long)).item())

    return {
        "loss": loss_val,
        "accuracy": float(np.mean(preds == y_cat)),
        "f1": _f1_macro(preds, y_cat),
        "sharpe": compute_sharpe(pnl),
        "max_drawdown": compute_max_drawdown(pnl),
        "profit_factor": compute_profit_factor(pnl),
    }


def sanity_check(
    model: nn.Module, batch: int, device: torch.device, multimodal: bool, tab_dim: int
) -> bool:
    try:
        img = torch.randn(batch, 4, 32, 32, device=device)
        y = torch.zeros(batch, dtype=torch.long, device=device)
        if multimodal:
            tab = torch.randn(batch, tab_dim, device=device)
            out = model(img, tab)
        else:
            out = model(img)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        return True
    except Exception as exc:
        print(f"  [sanity] FAILED: {exc}")
        return False


def train_disc_model(
    name: str,
    model_cfg: dict[str, Any],
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    test_ds: TensorDataset,
    common_cfg: dict[str, Any],
    tab_input_dim: int,
) -> DiscResult:
    device, backend = resolve_device(str(common_cfg.get("preferred_backend", "auto")))
    set_global_seed(int(common_cfg["seed"]))

    if backend == "cuda":
        num_workers = int(common_cfg["num_workers"])
    elif backend == "directml":
        num_workers = int(common_cfg.get("num_workers_directml", 0))
    else:
        num_workers = int(common_cfg.get("num_workers_non_cuda", 0))
    pin_memory = backend == "cuda"
    batch_size = int(common_cfg["batch_size"])
    multimodal = _is_multimodal(name)

    def _loader(ds, shuffle):
        kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return DataLoader(ds, **kwargs)

    train_loader = _loader(train_ds, True)
    val_loader = _loader(val_ds, False)
    test_loader = _loader(test_ds, False)

    _log(
        f"[disc:{name}] start backend={backend} device={device} train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds)} batch_size={batch_size} num_workers={num_workers}"
    )

    model = _build_model(name, tab_input_dim=tab_input_dim, cfg=model_cfg).to(device)

    if not sanity_check(model, min(4, batch_size), device, multimodal, tab_input_dim):
        raise RuntimeError(f"Sanity check failed for discretionary model: {name}")
    _log(f"[disc:{name}] sanity-check passed")

    optimizer = make_optimizer(model, backend, float(common_cfg["lr"]), float(common_cfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(common_cfg["max_epochs"])))
    criterion = nn.CrossEntropyLoss()

    model_key = _artifact_name(name)
    ckpt_dir = Path(common_cfg["checkpoint_root"]) / model_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(common_cfg["tensorboard_root"]) / model_key))

    best_val = float("inf")
    patience = int(common_cfg["patience"])
    wait = 0
    total_batches = max(1, len(train_loader))
    heartbeat_every = min(100, total_batches)

    for epoch in range(1, int(common_cfg["max_epochs"]) + 1):
        model.train()
        losses = []
        train_correct = 0.0
        train_seen = 0
        epoch_started = time.time()
        _log(f"[disc:{name}] epoch {epoch}/{int(common_cfg['max_epochs'])} started")
        for batch_idx, batch in enumerate(train_loader, start=1):
            img, tab, y = batch
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if multimodal:
                logits = model(img, tab.to(device, non_blocking=True))
            else:
                logits = model(img)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            preds = logits.detach().argmax(dim=1)
            train_correct += float((preds == y).sum().item())
            train_seen += int(y.numel())
            cleanup_cuda(img, y)
            if batch_idx == 1 or batch_idx % heartbeat_every == 0 or batch_idx == total_batches:
                _log(f"[disc:{name}] epoch {epoch} batch {batch_idx}/{total_batches} loss={loss.item():.6f}")

        scheduler.step()
        vm = _evaluate(model, val_loader, device, multimodal)
        train_loss = float(np.mean(losses))
        train_acc = float(train_correct / max(1, train_seen))
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", vm["loss"], epoch)
        writer.add_scalar("val/accuracy", vm["accuracy"], epoch)
        writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), epoch)

        epoch_elapsed_s = float(time.time() - epoch_started)
        samples_per_s = float(len(train_ds) / max(epoch_elapsed_s, 1e-6))
        _log(
            f"[disc:{name}] epoch {epoch} done train_loss={train_loss:.6f} train_acc={train_acc:.4f} val_loss={vm['loss']:.6f} "
            f"val_acc={vm['accuracy']:.4f} val_f1={vm['f1']:.4f} elapsed_s={epoch_elapsed_s:.1f}"
        )
        append_working_log(
            model_key,
            "EPOCH",
            {
                "backend": backend,
                "elapsed_s": epoch_elapsed_s,
                "samples_per_s": samples_per_s,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": vm["loss"],
                "val_acc": vm["accuracy"],
                "val_f1": vm["f1"],
                "test_status": "pending",
            },
            epoch=epoch,
            total_epochs=int(common_cfg["max_epochs"]),
        )

        if vm["loss"] < best_val:
            best_val = vm["loss"]
            wait = 0
            save_checkpoint(model, optimizer, ckpt_dir)
            _log(f"[disc:{name}] checkpoint saved val_loss={vm['loss']:.6f}")
        else:
            wait += 1
            if wait >= patience:
                _log(f"[disc:{name}] early-stop patience={patience}")
                break

    writer.close()
    load_best_checkpoint(model, ckpt_dir)
    model = model.to(device)
    vm = _evaluate(model, val_loader, device, multimodal)
    tm = _evaluate(model, test_loader, device, multimodal)
    _log(f"[disc:{name}] final val_acc={vm['accuracy']:.4f} test_acc={tm['accuracy']:.4f} test_f1={tm['f1']:.4f} test_sharpe={tm['sharpe']:.4f}")
    append_working_log(
        model_key,
        "FINAL",
        {
            "val_acc": vm["accuracy"],
            "val_f1": vm["f1"],
            "val_sharpe": vm["sharpe"],
            "test_acc": tm["accuracy"],
            "test_f1": tm["f1"],
            "test_sharpe": tm["sharpe"],
            "test_profit_factor": tm["profit_factor"],
            "test_max_drawdown": tm["max_drawdown"],
        },
    )

    is_valid = (
        tm["accuracy"] > 0.50
        and tm["f1"] > 0.45
        and tm["sharpe"] > 1.2
        and tm["profit_factor"] > 1.5
        and tm["max_drawdown"] < 0.20
    )

    result = DiscResult(
        model_name=model_key,
        checkpoint_dir=str(ckpt_dir).replace("\\", "/"),
        val_accuracy=vm["accuracy"],
        test_accuracy=tm["accuracy"],
        val_f1=vm["f1"],
        test_f1=tm["f1"],
        val_sharpe=vm["sharpe"],
        test_sharpe=tm["sharpe"],
        test_profit_factor=tm["profit_factor"],
        test_max_drawdown=tm["max_drawdown"],
        is_valid=is_valid,
        backend=backend,
        cuda_used=backend in ("cuda", "directml"),
    )

    model.cpu()
    cleanup_cuda(model, optimizer)
    return result


def write_disc_registry(results: list[DiscResult], registry_path: Path) -> None:
    entries = [
        {
            "architecture_name": r.model_name,
            "archetype": "discretionary_multimodal",
            "weights_path": f"{r.checkpoint_dir}/model_best.pt",
            "optimizer_state_path": f"{r.checkpoint_dir}/optimizer_state.pt",
            "design_premise": "Visual chart pattern recognition via rasterized candlestick image analysis.",
            "standard_interface": {"outputs": ["class_logits_3"]},
            "validation": {
                "val_accuracy": r.val_accuracy,
                "test_accuracy": r.test_accuracy,
                "val_f1": r.val_f1,
                "test_f1": r.test_f1,
                "val_sharpe": r.val_sharpe,
                "test_sharpe": r.test_sharpe,
                "test_profit_factor": r.test_profit_factor,
                "test_max_drawdown": r.test_max_drawdown,
                "backend": r.backend,
                "cuda_used": r.cuda_used,
                "is_valid": r.is_valid,
                "oos_consistency_passed": r.is_valid,  # mirrors is_valid; set False when gates fail
            },
        }
        for r in results
    ]
    append_registry(entries, registry_path)
