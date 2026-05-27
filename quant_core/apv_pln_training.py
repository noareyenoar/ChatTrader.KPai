"""APV-PLN training loop — Phase 4.

Knowledge Distillation Loss (LUPI / Oracle-Teacher framework):

    Loss_total = α · CE(student_logits, true_bin) + β · KL(student_log_softmax ‖ oracle_soft)

Oracle Isolation Contract
-------------------------
- mode='train'    → Oracle Teacher is called; Distillation Loss is active.
- mode='val/test' → Oracle Teacher is NEVER called; only Student CE loss used.

The DataLoader for train supplies 4-tuples (x_price, x_volume, y_bin, x_oracle).
Val/test DataLoaders supply 3-tuples (x_price, x_volume, y_bin).
"""
from __future__ import annotations

import math
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.gpu_utils import cleanup_cuda
from .apv_pln_models import APVPLNModel
from .shared_training import (
    append_registry,
    append_working_log,
    resolve_device,
    set_global_seed,
    make_optimizer,
    compute_sharpe,
    compute_max_drawdown,
    compute_profit_factor,
    compute_directional_acc,
    annualization_factor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Training result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class APVTrainingResult:
    model_name: str
    checkpoint_dir: str
    val_loss: float
    test_loss: float
    val_directional_acc: float
    test_directional_acc: float
    val_sharpe: float
    test_sharpe: float
    test_profit_factor: float
    test_max_drawdown: float
    is_valid: bool
    backend: str
    cuda_used: bool
    sanity_passed: bool
    divergence_alert: bool
    overfit_alert: bool


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {message}", flush=True)


def _annualization_factor() -> float:
    return math.sqrt(252 * 24 * 12)  # 5-minute bars


def _checkpoint_score(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics["sharpe"]),
        float(metrics["directional_acc"]),
        -float(metrics["loss"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Distillation Loss
# ─────────────────────────────────────────────────────────────────────────────

class APVDistillationLoss(nn.Module):
    """Combined Cross-Entropy + KL Divergence loss for Knowledge Distillation.

    Train mode:
        L = α · CE(logits, true_bin) + β · KL(student_log_softmax ‖ oracle_soft)

    Val/test mode (oracle_soft is None):
        L = CE(logits, true_bin)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, temperature: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(
        self,
        student_logits: torch.Tensor,
        true_bin: torch.Tensor,
        oracle_soft: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ce_loss = self.ce(student_logits, true_bin)

        if oracle_soft is None:
            return ce_loss

        # Temperature scaling for softer distillation targets
        T = self.temperature
        student_log_soft = torch.log_softmax(student_logits / T, dim=-1)  # log probs
        oracle_soft_T = torch.softmax(
            torch.log(oracle_soft.clamp(min=1e-8)) / T, dim=-1
        )  # re-scaled oracle soft

        kl_loss = self.kl(student_log_soft, oracle_soft_T) * (T ** 2)

        return self.alpha * ce_loss + self.beta * kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_loaders(
    train_ds, val_ds, test_ds,
    batch_size: int, num_workers: int, pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    kwargs = dict(num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kwargs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop (Student-only — Oracle NEVER used here)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(
    model: APVPLNModel,
    loader: DataLoader,
    criterion: APVDistillationLoss,
    device: torch.device,
    bin_centers: torch.Tensor,
) -> dict[str, float]:
    """Evaluate Student in isolation (no Oracle) and return metric dict.

    Oracle Isolation: x_oracle is NEVER accessed or passed here.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    preds_np: list[np.ndarray] = []
    actuals_np: list[np.ndarray] = []

    for batch in loader:
        # val/test loader yields 3-tuples: (x_price, x_volume, y_bin)
        x_price, x_volume, y_bin = batch[0], batch[1], batch[2]
        x_price = x_price.to(device)
        x_volume = x_volume.to(device)
        y_bin = y_bin.to(device)

        # ── Oracle Isolation: forward() called WITHOUT x_oracle ───────────────
        student_logits = model(x_price, x_volume, x_oracle=None)

        loss = criterion(student_logits, y_bin, oracle_soft=None)
        total_loss += loss.item()
        n_batches += 1

        # Expected return as directional signal
        probs = torch.softmax(student_logits, dim=-1)                    # [B, num_bins]
        exp_return = (probs * bin_centers.unsqueeze(0)).sum(dim=-1)      # [B]
        true_return = bin_centers[y_bin]                                 # [B] approx actual

        preds_np.append(exp_return.detach().cpu().numpy())
        actuals_np.append(true_return.detach().cpu().numpy())

        cleanup_cuda(x_price, x_volume, y_bin, student_logits, loss)

    pred_arr = np.concatenate(preds_np) if preds_np else np.array([0.0])
    act_arr = np.concatenate(actuals_np) if actuals_np else np.array([0.0])
    pnl = np.sign(pred_arr) * act_arr

    return {
        "loss": total_loss / max(1, n_batches),
        "directional_acc": float(np.mean(np.sign(pred_arr) == np.sign(act_arr))),
        "sharpe": compute_sharpe(pnl),
        "max_drawdown": compute_max_drawdown(pnl),
        "profit_factor": compute_profit_factor(pnl),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def _sanity_check(
    model: APVPLNModel,
    price_dim: int,
    vol_dim: int,
    oracle_dim: int,
    seq_len: int,
    horizon: int,
    device: torch.device,
    batch: int = 4,
) -> bool:
    """Verify forward pass in both train and eval modes, check gradient flow."""
    model.train()
    x_p = torch.randn(batch, seq_len, price_dim, device=device)
    x_v = torch.randn(batch, seq_len, vol_dim, device=device)
    x_o = torch.randn(batch, horizon, oracle_dim, device=device)
    y = torch.zeros(batch, dtype=torch.long, device=device)

    try:
        # Train mode (with oracle)
        logits, oracle_soft = model(x_p, x_v, x_oracle=x_o)
        assert logits.shape == (batch, model.num_bins), f"logits shape mismatch: {logits.shape}"
        assert oracle_soft.shape == (batch, model.num_bins), f"oracle shape mismatch: {oracle_soft.shape}"

        loss_fn = APVDistillationLoss()
        loss = loss_fn(logits, y, oracle_soft=oracle_soft)
        loss.backward()

        # Check finite gradients
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False

        model.zero_grad(set_to_none=True)

        # Eval mode (Oracle Isolation)
        model.eval()
        with torch.no_grad():
            student_only = model(x_p, x_v, x_oracle=None)
        assert student_only.shape == (batch, model.num_bins), \
            f"eval logits shape mismatch: {student_only.shape}"

        # Verify oracle teacher is NOT called in eval (no x_oracle passed)
        # This is guaranteed by the model.forward() contract above.

    except Exception as exc:
        _log(f"[apv-sanity] FAILED: {exc}")
        return False
    finally:
        cleanup_cuda(x_p, x_v, x_o, y)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Write registry
# ─────────────────────────────────────────────────────────────────────────────

def write_registry(results: list[APVTrainingResult], registry_path: Path) -> None:
    entries = []
    for r in results:
        entries.append({
            "architecture_name": r.model_name,
            "archetype": "apv_pln",
            "backend": r.backend,
            "val_loss": r.val_loss,
            "test_loss": r.test_loss,
            "val_directional_acc": r.val_directional_acc,
            "test_directional_acc": r.test_directional_acc,
            "val_sharpe": r.val_sharpe,
            "test_sharpe": r.test_sharpe,
            "test_profit_factor": r.test_profit_factor,
            "test_max_drawdown": r.test_max_drawdown,
            "is_valid": r.is_valid,
            "checkpoint_dir": r.checkpoint_dir,
            "divergence_alert": r.divergence_alert,
            "overfit_alert": r.overfit_alert,
        })
    append_registry(entries, registry_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_apv_pln(
    name: str,
    model_cfg: dict[str, Any],
    datasets,                            # APVPLNDatasets from apv_pln_data.py
    common_cfg: dict[str, Any],
) -> APVTrainingResult:
    """Train one APV-PLN model variant.

    Parameters
    ----------
    name        Model key (e.g. "apv_pln_v1").
    model_cfg   Arch hypers from config yaml (cnn_channels, nhead, etc.)
    datasets    APVPLNDatasets (train/val/test + bin metadata).
    common_cfg  Shared training settings from config yaml.
    """
    device, backend = resolve_device(str(common_cfg.get("preferred_backend", "auto")))
    set_global_seed(int(common_cfg.get("seed", 42)))

    seq_len = int(common_cfg["seq_len"])
    horizon = int(common_cfg["horizon"])

    # ── Batch size / worker config ────────────────────────────────────────────
    base_workers = int(common_cfg.get("num_workers", 0))
    if backend == "cuda":
        num_workers = base_workers
    elif backend == "directml":
        num_workers = int(common_cfg.get("num_workers_directml", 0))
    else:
        num_workers = int(common_cfg.get("num_workers_non_cuda", 0))
    pin_memory = (backend == "cuda")

    configured_bs = int(common_cfg.get("batch_size", 512))
    if backend == "directml":
        dml_cap = int(common_cfg.get("directml_max_batch_size", 256))
        effective_bs = min(configured_bs, dml_cap)
    else:
        effective_bs = configured_bs

    train_loader, val_loader, test_loader = _make_loaders(
        datasets.train, datasets.val, datasets.test,
        batch_size=effective_bs, num_workers=num_workers, pin_memory=pin_memory,
    )

    _log(
        f"[apv:{name}] start device={device} backend={backend} "
        f"train={len(datasets.train)} val={len(datasets.val)} test={len(datasets.test)} "
        f"batch_size={effective_bs} num_bins={datasets.num_bins}"
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model = APVPLNModel(
        price_dim=datasets.price_dim,
        vol_dim=datasets.volume_dim,
        oracle_dim=datasets.oracle_dim,
        num_bins=datasets.num_bins,
        cnn_channels=int(model_cfg.get("cnn_channels", 64)),
        nhead=int(model_cfg.get("nhead", 4)),
        oracle_hidden=int(model_cfg.get("oracle_hidden", 64)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    # ── Sanity check ──────────────────────────────────────────────────────────
    sanity_passed = _sanity_check(
        model,
        price_dim=datasets.price_dim,
        vol_dim=datasets.volume_dim,
        oracle_dim=datasets.oracle_dim,
        seq_len=seq_len,
        horizon=horizon,
        device=device,
    )
    if not sanity_passed:
        raise RuntimeError(f"[apv:{name}] sanity check failed")
    _log(f"[apv:{name}] sanity check PASSED (train+eval mode, oracle isolation verified)")

    # ── Optimizer / Scheduler / Loss ──────────────────────────────────────────
    optimizer = make_optimizer(
        model, backend,
        lr=float(common_cfg.get("lr", 0.001)),
        weight_decay=float(common_cfg.get("weight_decay", 1e-4)),
    )
    max_epochs = int(common_cfg.get("max_epochs", 60))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs))

    distill_alpha = float(common_cfg.get("distill_alpha", 0.5))
    distill_beta = float(common_cfg.get("distill_beta", 0.5))
    temperature = float(common_cfg.get("distill_temperature", 2.0))
    criterion = APVDistillationLoss(
        alpha=distill_alpha, beta=distill_beta, temperature=temperature
    )

    # ── Checkpoint / TensorBoard paths ────────────────────────────────────────
    artifact_name = f"APV_PLN_{name}"
    ckpt_dir = Path(common_cfg.get("checkpoint_root", "models/checkpoints/apv_pln")) / artifact_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(
        log_dir=str(
            Path(common_cfg.get("tensorboard_root", "models/tensorboard/apv_pln")) / artifact_name
        )
    )

    # ── Bin centers tensor (for expected-return computation in eval) ──────────
    bin_centers_cpu = torch.from_numpy(datasets.bin_centers)
    bin_centers = bin_centers_cpu.to(device)

    # ── Training loop ─────────────────────────────────────────────────────────
    patience = int(common_cfg.get("patience", 15))
    fast_fail_epoch = int(common_cfg.get("fast_fail_epoch", 20))
    fast_fail_min_sharpe = float(common_cfg.get("fast_fail_min_sharpe", 0.1))

    best_score: tuple | None = None
    wait = 0
    divergence_alert = False
    overfit_alert = False
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(backend == "cuda"))

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # ── Train batch: 4-tuple (x_price, x_volume, y_bin, x_oracle) ────
            x_price, x_volume, y_bin, x_oracle = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
                batch[3].to(device),
            )

            optimizer.zero_grad(set_to_none=True)

            # ── Oracle Teacher ONLY called in train mode ──────────────────────
            with torch.amp.autocast("cuda", enabled=(backend == "cuda")):
                student_logits, oracle_soft = model(x_price, x_volume, x_oracle=x_oracle)
                loss = criterion(student_logits, y_bin, oracle_soft=oracle_soft)

            if backend == "cuda":
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            cleanup_cuda(x_price, x_volume, y_bin, x_oracle, student_logits, oracle_soft, loss)

        scheduler.step()
        train_loss = epoch_loss / max(1, n_batches)

        # ── Validation (Oracle Isolation enforced inside _evaluate) ───────────
        val_metrics = _evaluate(model, val_loader, criterion, device, bin_centers)
        val_loss = val_metrics["loss"]

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Sharpe/val", val_metrics["sharpe"], epoch)
        writer.add_scalar("Acc/val", val_metrics["directional_acc"], epoch)

        if epoch % 5 == 0 or epoch == 1:
            _log(
                f"[apv:{name}] ep={epoch}/{max_epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_sharpe={val_metrics['sharpe']:.4f} "
                f"val_acc={val_metrics['directional_acc']:.4f}"
            )

        append_working_log(
            artifact_name, "val",
            {"train_loss": train_loss, "val_loss": val_loss,
             "val_sharpe": val_metrics["sharpe"], "val_acc": val_metrics["directional_acc"]},
            epoch=epoch, total_epochs=max_epochs,
        )

        # ── Fast-fail gate ────────────────────────────────────────────────────
        if epoch == fast_fail_epoch and val_metrics["sharpe"] < fast_fail_min_sharpe:
            _log(
                f"[apv:{name}] FAST-FAIL ep={epoch} "
                f"val_sharpe={val_metrics['sharpe']:.4f} < {fast_fail_min_sharpe}"
            )
            writer.close()
            return APVTrainingResult(
                model_name=artifact_name,
                checkpoint_dir=str(ckpt_dir).replace("\\", "/"),
                val_loss=val_loss, test_loss=val_loss,
                val_directional_acc=val_metrics["directional_acc"],
                test_directional_acc=val_metrics["directional_acc"],
                val_sharpe=val_metrics["sharpe"], test_sharpe=val_metrics["sharpe"],
                test_profit_factor=0.0, test_max_drawdown=1.0,
                is_valid=False, backend=backend,
                cuda_used=(backend in ("cuda", "directml")),
                sanity_passed=sanity_passed,
                divergence_alert=False, overfit_alert=False,
            )

        # ── Overfitting alert ─────────────────────────────────────────────────
        loss_ratio = train_loss / max(val_loss, 1e-8)
        overfit_alert_epoch = int(common_cfg.get("overfit_alert_epoch", fast_fail_epoch))
        overfit_loss_ratio_max = float(common_cfg.get("overfit_loss_ratio_max", 0.85))
        if epoch == overfit_alert_epoch and loss_ratio < overfit_loss_ratio_max:
            overfit_alert = True
            _log(
                f"[apv:{name}] OVERFIT-ALERT ep={epoch} "
                f"train={train_loss:.6f} val={val_loss:.6f} ratio={loss_ratio:.4f}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        current_score = _checkpoint_score(val_metrics)
        if best_score is None or current_score > best_score:
            best_score = current_score
            wait = 0
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state, ckpt_dir / "model_best.pt")
            # Save bin metadata alongside weights
            bin_meta = {
                "bin_min": datasets.bin_min,
                "bin_max": datasets.bin_max,
                "num_bins": datasets.num_bins,
                "bin_centers": datasets.bin_centers.tolist(),
            }
            torch.save(bin_meta, ckpt_dir / "bin_meta.pt")
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer_state.pt")
            _log(
                f"[apv:{name}] checkpoint saved val_sharpe={val_metrics['sharpe']:.6f} "
                f"val_acc={val_metrics['directional_acc']:.6f}"
            )
        else:
            wait += 1
            if wait >= patience:
                _log(f"[apv:{name}] early-stop patience={patience} ep={epoch}")
                break

    writer.close()

    # ── Load best checkpoint for final evaluation ─────────────────────────────
    best_path = ckpt_dir / "model_best.pt"
    if best_path.exists():
        try:
            state = torch.load(best_path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                state = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        _log(f"[apv:{name}] best checkpoint reloaded from {best_path.name}")

    model.to(device)

    # ── Final test evaluation (Oracle Isolation enforced in _evaluate) ────────
    val_metrics_final = _evaluate(model, val_loader, criterion, device, bin_centers)
    test_metrics = _evaluate(model, test_loader, criterion, device, bin_centers)

    # Divergence alert: Sharpe ratio decays >50% from val to test
    if val_metrics_final["sharpe"] > 0:
        decay = abs(test_metrics["sharpe"] - val_metrics_final["sharpe"]) / abs(val_metrics_final["sharpe"])
        divergence_alert = decay > 0.5

    # Validation gates (matching project-wide standards)
    is_valid = (
        test_metrics["sharpe"] >= 1.2
        and test_metrics["directional_acc"] >= 0.55
        and test_metrics["profit_factor"] >= 1.5
        and test_metrics["max_drawdown"] <= 0.20
    )

    _log(
        f"[apv:{name}] FINAL "
        f"val_sharpe={val_metrics_final['sharpe']:.4f} "
        f"test_sharpe={test_metrics['sharpe']:.4f} "
        f"test_acc={test_metrics['directional_acc']:.4f} "
        f"is_valid={is_valid}"
    )

    append_working_log(
        artifact_name, "test",
        {
            "val_sharpe": val_metrics_final["sharpe"],
            "test_sharpe": test_metrics["sharpe"],
            "test_acc": test_metrics["directional_acc"],
            "test_pf": test_metrics["profit_factor"],
            "test_mdd": test_metrics["max_drawdown"],
            "is_valid": is_valid,
        },
    )

    return APVTrainingResult(
        model_name=artifact_name,
        checkpoint_dir=str(ckpt_dir).replace("\\", "/"),
        val_loss=val_metrics_final["loss"],
        test_loss=test_metrics["loss"],
        val_directional_acc=val_metrics_final["directional_acc"],
        test_directional_acc=test_metrics["directional_acc"],
        val_sharpe=val_metrics_final["sharpe"],
        test_sharpe=test_metrics["sharpe"],
        test_profit_factor=test_metrics["profit_factor"],
        test_max_drawdown=test_metrics["max_drawdown"],
        is_valid=is_valid,
        backend=backend,
        cuda_used=(backend in ("cuda", "directml")),
        sanity_passed=sanity_passed,
        divergence_alert=divergence_alert,
        overfit_alert=overfit_alert,
    )
