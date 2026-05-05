"""Statistical Arbitrage training loop — Phase 4.

Loss:       MSELoss (spread Z-score regression)
Metrics:    MAE, Reconstruction Error (autoencoder), Tracking Error,
            Sharpe (sign prediction), MaxDD, ProfitFactor
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.gpu_utils import cleanup_cuda
from .interfaces import TrendModelInterface
from .sequence_augmentation import augment_time_series_batch
from .stat_arb_models import StatArbAutoencoder, StatArbGAT, StatArbLSTM
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
class StatArbResult:
    model_name: str
    checkpoint_dir: str
    val_mae: float
    test_mae: float
    val_tracking_error: float
    test_tracking_error: float
    val_sharpe: float
    test_sharpe: float
    test_profit_factor: float
    test_max_drawdown: float
    is_valid: bool
    backend: str
    cuda_used: bool


def _build_model(name: str, num_assets: int, seq_len: int, cfg: dict[str, Any]) -> TrendModelInterface:
    if name == "autoencoder":
        return StatArbAutoencoder(num_assets=num_assets, seq_len=seq_len, **cfg)
    if name == "gat":
        return StatArbGAT(num_assets=num_assets, **cfg)
    if name == "lstm":
        return StatArbLSTM(num_assets=num_assets, **cfg)
    raise ValueError(f"Unknown stat arb model: {name}")


def _artifact_name(name: str) -> str:
    return {
        "autoencoder": "Autoencoder_StatArb_v1",
        "gat": "GAT_StatArb_v1",
        "lstm": "LSTM_StatArb_v1",
    }[name]


def _log(message: str) -> None:
    print(message, flush=True)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_pred, all_y = [], []
    recon_errors = []

    for batch in loader:
        # Support 3-tensor batches (x, y, actual_return) — stat_arb labels ARE the return proxy
        x, y = batch[0], batch[1]
        x_dev = x.to(device)
        pred = model(x_dev).squeeze(-1).detach().cpu().float()
        all_pred.append(pred)
        all_y.append(y.float())

        # Reconstruction error for autoencoder models
        if hasattr(model, "reconstruct"):
            recon = model.reconstruct(x_dev).detach().cpu().float()
            recon_errors.append(float(nn.MSELoss()(recon, x).item()))

    pred_cat = torch.cat(all_pred).numpy()
    y_cat = torch.cat(all_y).numpy()

    mae = float(np.mean(np.abs(pred_cat - y_cat)))
    tracking_error = float(np.std(pred_cat - y_cat))
    # Use actual z-score return as PnL: sign(pred) * y - cost
    ROUND_TRIP_COST = 0.0004
    pnl = np.sign(pred_cat) * y_cat - ROUND_TRIP_COST
    loss_val = float(nn.MSELoss()(torch.tensor(pred_cat), torch.tensor(y_cat)).item())

    return {
        "loss": loss_val,
        "mae": mae,
        "tracking_error": tracking_error,
        "recon_error": float(np.mean(recon_errors)) if recon_errors else 0.0,
        "sharpe": compute_sharpe(pnl),
        "max_drawdown": compute_max_drawdown(pnl),
        "profit_factor": compute_profit_factor(pnl),
    }


def sanity_check(model: nn.Module, batch: int, num_assets: int, seq_len: int, device: torch.device) -> bool:
    try:
        x = torch.randn(batch, seq_len, num_assets, device=device)
        pred = model(x).squeeze(-1)
        y = torch.zeros(batch, device=device)
        loss = nn.MSELoss()(pred, y)
        # Add reconstruction loss if autoencoder
        if hasattr(model, "reconstruct"):
            recon = model.reconstruct(x)
            loss = loss + nn.MSELoss()(recon, x)
        loss.backward()
        return True
    except Exception as exc:
        print(f"  [sanity] FAILED: {exc}")
        return False


def _move_model_to_device(model: nn.Module, device: torch.device, backend: str, model_name: str) -> nn.Module:
    retries = 3 if backend == "directml" else 1
    for attempt in range(1, retries + 1):
        try:
            if backend == "directml":
                # Probe first so model transfer doesn't become the first failing op.
                _probe = torch.zeros(1).to(device)
                del _probe
            return model.to(device)
        except RuntimeError as exc:
            msg = str(exc)
            if backend != "directml" or "GPU will not respond" not in msg or attempt >= retries:
                raise
            _log(
                f"[stat-arb:{model_name}] directml transfer retry {attempt}/{retries} after runtime error: {msg}"
            )
            cleanup_cuda()
            gc.collect()
    return model


def train_stat_arb_model(
    name: str,
    model_cfg: dict[str, Any],
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    test_ds: TensorDataset,
    common_cfg: dict[str, Any],
    num_assets: int,
    seq_len: int,
) -> StatArbResult:
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
        f"[stat-arb:{name}] start backend={backend} device={device} train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds)} batch_size={batch_size} num_workers={num_workers}"
    )

    model = _build_model(name, num_assets=num_assets, seq_len=seq_len, cfg=model_cfg)
    model = _move_model_to_device(model, device, backend, name)

    if not sanity_check(model, min(32, batch_size), num_assets, seq_len, device):
        raise RuntimeError(f"Sanity check failed for stat arb model: {name}")
    _log(f"[stat-arb:{name}] sanity-check passed")

    optimizer = make_optimizer(model, backend, float(common_cfg["lr"]), float(common_cfg["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(common_cfg["max_epochs"])))
    mse = nn.MSELoss()
    recon_weight = float(common_cfg.get("recon_weight", 0.5))

    model_key = _artifact_name(name)
    ckpt_dir = Path(common_cfg["checkpoint_root"]) / model_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(common_cfg["tensorboard_root"]) / model_key))

    best_val = float("inf")
    patience = int(common_cfg["patience"])
    wait = 0
    last_val_sharpe: float | None = None
    frozen_sharpe_epochs = 0
    total_batches = max(1, len(train_loader))
    heartbeat_every = min(100, total_batches)

    for epoch in range(1, int(common_cfg["max_epochs"]) + 1):
        model.train()
        losses = []
        epoch_started = time.time()
        _log(f"[stat-arb:{name}] epoch {epoch}/{int(common_cfg['max_epochs'])} started")
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = augment_time_series_batch(
                x,
                enabled=bool(common_cfg.get("use_sequence_augmentation", True)),
                mask_prob=float(common_cfg.get("augmentation_mask_prob", 0.03)),
                max_warp=float(common_cfg.get("augmentation_time_warp", 0.08)),
            )
            optimizer.zero_grad(set_to_none=True)
            pred = model(x).squeeze(-1)
            loss = mse(pred, y)
            if hasattr(model, "reconstruct"):
                recon = model.reconstruct(x)
                loss = loss + recon_weight * mse(recon, x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            cleanup_cuda(x, y)
            if batch_idx == 1 or batch_idx % heartbeat_every == 0 or batch_idx == total_batches:
                _log(f"[stat-arb:{name}] epoch {epoch} batch {batch_idx}/{total_batches} loss={loss.item():.6f}")

        scheduler.step()
        vm = _evaluate(model, val_loader, device)
        writer.add_scalar("train/loss", float(np.mean(losses)), epoch)
        writer.add_scalar("val/loss", vm["loss"], epoch)
        writer.add_scalar("val/mae", vm["mae"], epoch)
        writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), epoch)
        _log(
            f"[stat-arb:{name}] epoch {epoch} done train_loss={float(np.mean(losses)):.6f} val_loss={vm['loss']:.6f} "
            f"val_mae={vm['mae']:.4f} val_sharpe={vm['sharpe']:.4f} elapsed_s={time.time() - epoch_started:.1f}"
        )
        if last_val_sharpe is not None and abs(vm["sharpe"] - last_val_sharpe) < 1e-6:
            frozen_sharpe_epochs += 1
        else:
            frozen_sharpe_epochs = 0
        last_val_sharpe = vm["sharpe"]
        if frozen_sharpe_epochs >= 2:
            _log(
                f"[stat-arb:{name}] warning sharpe_frozen_epochs={frozen_sharpe_epochs + 1} "
                f"last_val_sharpe={vm['sharpe']:.6f}"
            )
        append_working_log(
            model_key,
            "EPOCH",
            {
                "train_loss": float(np.mean(losses)),
                "val_loss": vm["loss"],
                "val_mae": vm["mae"],
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
            _log(f"[stat-arb:{name}] checkpoint saved val_loss={vm['loss']:.6f}")
        else:
            wait += 1
            if wait >= patience:
                _log(f"[stat-arb:{name}] early-stop patience={patience}")
                break

    writer.close()
    load_best_checkpoint(model, ckpt_dir)
    model = _move_model_to_device(model, device, backend, name)
    vm = _evaluate(model, val_loader, device)
    tm = _evaluate(model, test_loader, device)
    _log(f"[stat-arb:{name}] final val_mae={vm['mae']:.4f} test_mae={tm['mae']:.4f} test_sharpe={tm['sharpe']:.4f}")
    append_working_log(
        model_key,
        "FINAL",
        {
            "val_mae": vm["mae"],
            "val_sharpe": vm["sharpe"],
            "test_mae": tm["mae"],
            "test_sharpe": tm["sharpe"],
            "test_profit_factor": tm["profit_factor"],
            "test_max_drawdown": tm["max_drawdown"],
        },
    )

    is_valid = (
        tm["mae"] < 0.5
        and tm["tracking_error"] < 0.5
        and (tm["recon_error"] < 0.05 if hasattr(model, "reconstruct") else True)
        and tm["sharpe"] > 1.2
        and tm["profit_factor"] > 1.5
        and tm["max_drawdown"] < 0.20
    )

    result = StatArbResult(
        model_name=model_key,
        checkpoint_dir=str(ckpt_dir).replace("\\", "/"),
        val_mae=vm["mae"],
        test_mae=tm["mae"],
        val_tracking_error=vm["tracking_error"],
        test_tracking_error=tm["tracking_error"],
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


def write_stat_arb_registry(results: list[StatArbResult], registry_path: Path) -> None:
    entries = [
        {
            "architecture_name": r.model_name,
            "archetype": "statistical_arbitrage",
            "weights_path": f"{r.checkpoint_dir}/model_best.pt",
            "optimizer_state_path": f"{r.checkpoint_dir}/optimizer_state.pt",
            "design_premise": "Multi-asset spread convergence via learned latent mispricing representations.",
            "standard_interface": {"outputs": ["spread_prediction"]},
            "validation": {
                "val_mae": r.val_mae,
                "test_mae": r.test_mae,
                "val_tracking_error": r.val_tracking_error,
                "test_tracking_error": r.test_tracking_error,
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
