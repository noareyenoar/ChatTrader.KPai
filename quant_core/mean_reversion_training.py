"""Mean Reversion training loop — Phase 4.

Loss:       BCEWithLogitsLoss
Metrics:    Accuracy, Precision@Reversal (precision when |zscore| > 1.5),
            Sharpe, MaxDD, ProfitFactor
Optimizer:  SGD(DirectML) / AdamW(CPU/CUDA)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.gpu_utils import cleanup_cuda
from .interfaces import TrendModelInterface
from .mean_reversion_models import MeanReversionGRN, MeanReversionMLP, MeanReversionResNet
from .shared_training import (
    annualization_factor,
    append_working_log,
    append_registry,
    compute_directional_acc,
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
class MRTrainingResult:
    model_name: str
    checkpoint_dir: str
    val_accuracy: float
    test_accuracy: float
    val_precision_reversal: float
    test_precision_reversal: float
    val_sharpe: float
    test_sharpe: float
    test_profit_factor: float
    test_max_drawdown: float
    is_valid: bool
    backend: str
    cuda_used: bool


def _build_model(name: str, input_dim: int, cfg: dict[str, Any]) -> TrendModelInterface:
    if name == "mlp":
        return MeanReversionMLP(input_dim=input_dim, **cfg)
    if name == "resnet":
        return MeanReversionResNet(input_dim=input_dim, **cfg)
    if name == "grn":
        return MeanReversionGRN(input_dim=input_dim, **cfg)
    raise ValueError(f"Unknown mean reversion model: {name}")


def _artifact_name(name: str) -> str:
    return {"mlp": "MLP_MR_v1", "resnet": "ResNet_MR_v1", "grn": "GRN_MR_v1"}[name]


def _log(message: str) -> None:
    print(message, flush=True)


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
        logits = model(x.to(device)).squeeze(-1)
        all_logits.append(logits.detach().cpu().float())
        all_y.append(y.cpu().float())
    logits_cat = torch.cat(all_logits).numpy()
    y_cat = torch.cat(all_y).numpy()

    probs = 1.0 / (1.0 + np.exp(-logits_cat))
    preds = (probs >= 0.5).astype(float)
    acc = float(np.mean(preds == y_cat))

    # Execution-grade PnL: use actual forward return when available
    ROUND_TRIP_COST = 0.0004
    pred_signed = preds * 2.0 - 1.0          # {-1, +1}
    if has_returns:
        ret_cat = torch.cat(all_ret).numpy()
        pnl = pred_signed * ret_cat - ROUND_TRIP_COST
    else:
        y_signed = y_cat * 2.0 - 1.0
        pnl = pred_signed * y_signed

    # precision@reversal: subset where |raw logit| > 0.4 (high-conviction calls)
    extreme_mask = np.abs(logits_cat) > 0.4
    if extreme_mask.sum() > 0:
        prec_rev = float(np.mean(preds[extreme_mask] == y_cat[extreme_mask]))
    else:
        prec_rev = float(acc)

    # 2-class CrossEntropyLoss is DirectML-safe (verified in Trend training)
    logits_t = torch.tensor(logits_cat)  # (N,) 1D
    logits_2d = torch.stack([-logits_t, logits_t], dim=1)  # (N,2)
    loss_val = float(nn.CrossEntropyLoss()(logits_2d, torch.tensor(y_cat, dtype=torch.long)).item())
    return {
        "loss": loss_val,
        "accuracy": acc,
        "precision_reversal": prec_rev,
        "sharpe": compute_sharpe(pnl),
        "max_drawdown": compute_max_drawdown(pnl),
        "profit_factor": compute_profit_factor(pnl),
    }


def sanity_check(model: nn.Module, batch: int, input_dim: int, device: torch.device) -> bool:
    try:
        x = torch.randn(batch, input_dim, device=device)
        out = model(x)  # (B, 1)
        logits_2d = torch.cat([-out, out], dim=-1)  # (B, 2)
        loss = nn.CrossEntropyLoss()(logits_2d, torch.zeros(out.size(0), dtype=torch.long, device=out.device))
        loss.backward()
        return True
    except Exception as exc:
        print(f"  [sanity] FAILED: {exc}")
        return False


def train_mr_model(
    name: str,
    model_cfg: dict[str, Any],
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    test_ds: TensorDataset,
    common_cfg: dict[str, Any],
    input_dim: int,
) -> MRTrainingResult:
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
        f"[mr:{name}] start backend={backend} device={device} train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds)} batch_size={batch_size} num_workers={num_workers}"
    )

    model = _build_model(name, input_dim=input_dim, cfg=model_cfg).to(device)

    if not sanity_check(model, batch=min(32, batch_size), input_dim=input_dim, device=device):
        raise RuntimeError(f"Sanity check failed for MR model: {name}")
    _log(f"[mr:{name}] sanity-check passed")

    optimizer = make_optimizer(model, backend, float(common_cfg["lr"]), float(common_cfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(common_cfg["max_epochs"])))
    # Convert binary to 2-class CE (DirectML-safe: cat([-logit, logit]) → CE == BCE)
    criterion = lambda logits, y: nn.CrossEntropyLoss()(
        torch.cat([-logits, logits], dim=-1), y.long()
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
        _log(f"[mr:{name}] epoch {epoch}/{int(common_cfg['max_epochs'])} started")
        for batch_idx, batch in enumerate(train_loader, start=1):
            # Support 3-tensor batches (x, y, actual_return) from updated mr_data
            x, y = batch[0], batch[1]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            preds = (torch.sigmoid(logits.squeeze(-1).detach()) >= 0.5).float()
            train_correct += float((preds == y.float()).sum().item())
            train_seen += int(y.numel())
            cleanup_cuda(x, y)
            if batch_idx == 1 or batch_idx % heartbeat_every == 0 or batch_idx == total_batches:
                _log(f"[mr:{name}] epoch {epoch} batch {batch_idx}/{total_batches} loss={loss.item():.6f}")

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
            f"[mr:{name}] epoch {epoch} done train_loss={train_loss:.6f} train_acc={train_acc:.4f} val_loss={vm['loss']:.6f} "
            f"val_acc={vm['accuracy']:.4f} val_sharpe={vm['sharpe']:.4f} elapsed_s={epoch_elapsed_s:.1f}"
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
                "val_sharpe": vm["sharpe"],
                "test_status": "pending",
            },
            epoch=epoch,
            total_epochs=int(common_cfg["max_epochs"]),
        )

        if vm["loss"] < best_val:
            best_val = vm["loss"]
            wait = 0
            save_checkpoint(model, optimizer, ckpt_dir)
            _log(f"[mr:{name}] checkpoint saved val_loss={vm['loss']:.6f}")
        else:
            wait += 1
            if wait >= patience:
                _log(f"[mr:{name}] early-stop patience={patience}")
                break

    writer.close()
    load_best_checkpoint(model, ckpt_dir)
    model = model.to(device)
    vm = _evaluate(model, val_loader, device)
    tm = _evaluate(model, test_loader, device)
    _log(f"[mr:{name}] final val_acc={vm['accuracy']:.4f} test_acc={tm['accuracy']:.4f} test_sharpe={tm['sharpe']:.4f}")
    append_working_log(
        model_key,
        "FINAL",
        {
            "val_acc": vm["accuracy"],
            "val_precision_reversal": vm["precision_reversal"],
            "val_sharpe": vm["sharpe"],
            "test_acc": tm["accuracy"],
            "test_precision_reversal": tm["precision_reversal"],
            "test_sharpe": tm["sharpe"],
            "test_profit_factor": tm["profit_factor"],
            "test_max_drawdown": tm["max_drawdown"],
        },
    )

    is_valid = (
        vm["accuracy"] > 0.55
        and tm["accuracy"] > 0.55
        and vm["precision_reversal"] > 0.60
        and tm["sharpe"] > 1.2
        and tm["profit_factor"] > 1.5
        and tm["max_drawdown"] < 0.20
    )

    result = MRTrainingResult(
        model_name=model_key,
        checkpoint_dir=str(ckpt_dir).replace("\\", "/"),
        val_accuracy=vm["accuracy"],
        test_accuracy=tm["accuracy"],
        val_precision_reversal=vm["precision_reversal"],
        test_precision_reversal=tm["precision_reversal"],
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


def write_mr_registry(results: list[MRTrainingResult], registry_path: Path) -> None:
    entries = [
        {
            "architecture_name": r.model_name,
            "archetype": "mean_reversion",
            "weights_path": f"{r.checkpoint_dir}/model_best.pt",
            "optimizer_state_path": f"{r.checkpoint_dir}/optimizer_state.pt",
            "design_premise": "Tabular classification of price overextension and mean-reverting regimes.",
            "standard_interface": {"outputs": ["logit", "probability"]},
            "validation": {
                "val_accuracy": r.val_accuracy,
                "test_accuracy": r.test_accuracy,
                "val_precision_reversal": r.val_precision_reversal,
                "test_precision_reversal": r.test_precision_reversal,
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
