"""Scalper training loop — Phase 4.

Loss:       CrossEntropyLoss (3-class)
Metrics:    Accuracy, F1 (macro), Inference latency (ms), Sharpe, MaxDD
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.gpu_utils import cleanup_cuda
from .interfaces import TrendModelInterface
from .sequence_augmentation import augment_time_series_batch
from .scalper_models import ScalperCNN, ScalperGRU, ScalperLinearAttn
from .shared_training import (
    annualization_factor,
    append_working_log,
    append_registry,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe,
    load_best_checkpoint,
    load_epoch_checkpoint,
    make_optimizer,
    resolve_device,
    save_checkpoint,
    save_epoch_checkpoint,
    set_global_seed,
)


@dataclass
class ScalperResult:
    model_name: str
    checkpoint_dir: str
    val_accuracy: float
    test_accuracy: float
    val_f1: float
    test_f1: float
    inference_ms: float
    val_sharpe: float
    test_sharpe: float
    test_profit_factor: float
    test_max_drawdown: float
    is_valid: bool
    backend: str
    cuda_used: bool


def _build_model(name: str, input_dim: int, seq_len: int, cfg: dict[str, Any]) -> TrendModelInterface:
    if name == "cnn":
        return ScalperCNN(input_dim=input_dim, **cfg)
    if name == "linear_attn":
        return ScalperLinearAttn(input_dim=input_dim, **cfg)
    if name == "gru":
        return ScalperGRU(input_dim=input_dim, **cfg)
    raise ValueError(f"Unknown scalper model: {name}")


def _artifact_name(name: str) -> str:
    return {"cnn": "CNN_Scalper_v1", "linear_attn": "LinearAttn_Scalper_v1", "gru": "GRU_Scalper_v1"}[name]


def _log(message: str) -> None:
    print(message, flush=True)


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
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_logits, all_y, all_ret = [], [], []
    has_returns = False
    for batch in loader:
        if len(batch) == 3:
            x, y, actual_ret = batch
            all_ret.append(actual_ret.cpu().float())
            has_returns = True
        else:
            x, y = batch
        logits = model(x.to(device))
        all_logits.append(logits.detach().cpu().float())
        all_y.append(y.cpu())
    logits_cat = torch.cat(all_logits)
    y_cat = torch.cat(all_y).numpy()
    preds = logits_cat.argmax(dim=1).numpy()

    # Execution-grade PnL using actual returns when available
    ROUND_TRIP_COST = 0.0004
    label_map = np.array([-1.0, 0.0, 1.0])  # 0=down, 1=flat, 2=up
    if has_returns:
        ret_cat = torch.cat(all_ret).numpy()
        pred_dir = label_map[preds]          # -1, 0, or +1
        trade_mask = (pred_dir != 0).astype(float)
        # correct direction: earn |return|; wrong: lose |return|; flat pred: no PnL
        correct = (preds == y_cat).astype(float)
        wrong   = ((preds != y_cat) & (preds != 1)).astype(float)
        pnl = (correct - wrong) * np.abs(ret_cat) - trade_mask * ROUND_TRIP_COST
    else:
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


def _measure_latency(model: nn.Module, input_dim: int, seq_len: int, device: torch.device) -> float:
    """Measure single-sample inference latency in milliseconds."""
    model.eval()
    x = torch.randn(1, seq_len, input_dim, device=device)
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(x)
    # Measure
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            model(x)
    return (time.perf_counter() - t0) / 100 * 1000


def sanity_check(model: nn.Module, batch: int, seq_len: int, input_dim: int, device: torch.device) -> bool:
    try:
        x = torch.randn(batch, seq_len, input_dim, device=device)
        y = torch.zeros(batch, dtype=torch.long, device=device)
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        return True
    except Exception as exc:
        print(f"  [sanity] FAILED: {exc}")
        return False


def train_scalper_model(
    name: str,
    model_cfg: dict[str, Any],
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    test_ds: TensorDataset,
    common_cfg: dict[str, Any],
    input_dim: int,
    seq_len: int,
) -> ScalperResult:
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
        f"[scalper:{name}] start backend={backend} device={device} train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds)} batch_size={batch_size} num_workers={num_workers}"
    )

    model = _build_model(name, input_dim=input_dim, seq_len=seq_len, cfg=model_cfg).to(device)

    if not sanity_check(model, min(32, batch_size), seq_len, input_dim, device):
        raise RuntimeError(f"Sanity check failed for scalper model: {name}")
    _log(f"[scalper:{name}] sanity-check passed")

    optimizer = make_optimizer(model, backend, float(common_cfg["lr"]), float(common_cfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(common_cfg["max_epochs"])))

    # Class weighting prevents collapse to dominant regime labels.
    # Support both lazy RollingClassWindowDataset and old TensorDataset APIs.
    if hasattr(train_ds, 'target_list'):
        # Lazy dataset: concatenate per-symbol targets
        train_labels = torch.cat(train_ds.target_list).cpu().numpy().astype(np.int64)
    else:
        # Old TensorDataset API
        train_labels = train_ds.tensors[1].detach().cpu().numpy().astype(np.int64)
    class_counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    inv_freq = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = inv_freq / np.mean(inv_freq)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    _log(f"[scalper:{name}] class_counts={class_counts.tolist()} class_weights={class_weights.tolist()}")

    cyclic_scheduler = None
    if bool(common_cfg.get("use_cyclic_lr", True)):
        base_lr = float(common_cfg.get("cyclic_base_lr", float(common_cfg["lr"]) * 0.2))
        max_lr = float(common_cfg.get("cyclic_max_lr", float(common_cfg["lr"]) * 1.5))
        step_up = max(1, len(train_loader) * int(common_cfg.get("cyclic_step_up_epochs", 2)))
        cyclic_scheduler = CyclicLR(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_up,
            cycle_momentum=False,
        )

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
        _log(f"[scalper:{name}] epoch {epoch}/{int(common_cfg['max_epochs'])} started")
        for batch_idx, batch in enumerate(train_loader, start=1):
            # Support 3-tensor batches (x, y, actual_return) from updated scalper_data
            x, y = batch[0], batch[1]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if name == "gru":
                x = augment_time_series_batch(
                    x,
                    enabled=bool(common_cfg.get("use_sequence_augmentation", True)),
                    mask_prob=float(common_cfg.get("augmentation_mask_prob", 0.03)),
                    max_warp=float(common_cfg.get("augmentation_time_warp", 0.08)),
                )
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if cyclic_scheduler is not None:
                cyclic_scheduler.step()
            losses.append(float(loss.item()))
            preds = logits.detach().argmax(dim=1)
            train_correct += float((preds == y).sum().item())
            train_seen += int(y.numel())
            cleanup_cuda(x, y)
            if batch_idx == 1 or batch_idx % heartbeat_every == 0 or batch_idx == total_batches:
                _log(f"[scalper:{name}] epoch {epoch} batch {batch_idx}/{total_batches} loss={loss.item():.6f}")

        if cyclic_scheduler is None:
            scheduler.step()
        vm = _evaluate(model, val_loader, device)
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
            f"[scalper:{name}] epoch {epoch} done train_loss={train_loss:.6f} train_acc={train_acc:.4f} val_loss={vm['loss']:.6f} "
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
            _log(f"[scalper:{name}] checkpoint saved val_loss={vm['loss']:.6f}")
        else:
            wait += 1
            if wait >= patience:
                _log(f"[scalper:{name}] early-stop patience={patience}")
                break

        # Per-epoch checkpoint — enables resume from exact epoch on power failure
        save_epoch_checkpoint(model, optimizer, ckpt_dir, epoch, vm["loss"])

    writer.close()
    load_best_checkpoint(model, ckpt_dir)
    model = model.to(device)

    vm = _evaluate(model, val_loader, device)
    tm = _evaluate(model, test_loader, device)
    latency_ms = _measure_latency(model, input_dim, seq_len, device)
    _log(f"[scalper:{name}] final test_acc={tm['accuracy']:.4f} test_f1={tm['f1']:.4f} latency_ms={latency_ms:.2f} test_sharpe={tm['sharpe']:.4f}")
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
            "latency_ms": latency_ms,
        },
    )

    is_valid = (
        tm["accuracy"] > 0.52
        and tm["sharpe"] > 1.0
    )

    result = ScalperResult(
        model_name=model_key,
        checkpoint_dir=str(ckpt_dir).replace("\\", "/"),
        val_accuracy=vm["accuracy"],
        test_accuracy=tm["accuracy"],
        val_f1=vm["f1"],
        test_f1=tm["f1"],
        inference_ms=latency_ms,
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


def write_scalper_registry(results: list[ScalperResult], registry_path: Path) -> None:
    entries = [
        {
            "architecture_name": r.model_name,
            "archetype": "scalping_microstructure",
            "weights_path": f"{r.checkpoint_dir}/model_best.pt",
            "optimizer_state_path": f"{r.checkpoint_dir}/optimizer_state.pt",
            "design_premise": "Short-horizon order-flow microstructure modeling for intraday scalping.",
            "standard_interface": {"outputs": ["class_logits_3"]},
            "validation": {
                "val_accuracy": r.val_accuracy,
                "test_accuracy": r.test_accuracy,
                "val_f1": r.val_f1,
                "test_f1": r.test_f1,
                "inference_ms": r.inference_ms,
                "val_sharpe": r.val_sharpe,
                "test_sharpe": r.test_sharpe,
                "test_profit_factor": r.test_profit_factor,
                "test_max_drawdown": r.test_max_drawdown,
                "backend": r.backend,
                "cuda_used": r.cuda_used,
                "is_valid": r.is_valid,
            },
        }
        for r in results
    ]
    append_registry(entries, registry_path)
