from __future__ import annotations

import math
import pickle
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.gpu_utils import cleanup_cuda
from .interfaces import TrendModelInterface
from .sequence_augmentation import augment_time_series_batch
from .shared_training import append_registry, append_working_log
from .trend_models import TrendLSTMModel, TrendTCNModel, TrendTransformerModel


@dataclass
class TrainingResult:
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


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred_backend: str = "auto") -> tuple[torch.device, str]:
    pref = preferred_backend.lower().strip()

    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"

    if pref in ("auto", "directml"):
        try:
            import torch_directml
            dml_device = torch_directml.device()
            try:
                _probe = torch.zeros(1).to(dml_device)
                del _probe
                return dml_device, "directml"
            except Exception as probe_exc:
                if pref == "directml":
                    raise RuntimeError(f"DirectML probe failed in forced mode: {probe_exc}") from probe_exc
                _log(f"[trend] directml probe failed in auto mode reason={probe_exc}")
        except Exception as exc:
            if pref == "directml":
                raise RuntimeError(
                    "preferred_backend=directml but torch_directml unavailable or unhealthy"
                ) from exc
            _log(f"[trend] directml unavailable in auto mode reason={exc}")

    return torch.device("cpu"), "cpu"


def _build_model(name: str, input_dim: int, seq_len: int, cfg: dict[str, Any]) -> TrendModelInterface:
    if name == "lstm":
        return TrendLSTMModel(input_dim=input_dim, **cfg)
    if name == "transformer":
        return TrendTransformerModel(input_dim=input_dim, seq_len=seq_len, **cfg)
    if name == "tcn":
        return TrendTCNModel(input_dim=input_dim, **cfg)
    raise ValueError(f"Unknown model: {name}")


def _model_artifact_name(name: str) -> str:
    mapping = {
        "lstm": "LSTM_Trend_v1",
        "transformer": "Transformer_Trend_v1",
        "tcn": "TCN_Trend_v1",
    }
    return mapping[name]


def _annualization_factor() -> float:
    return math.sqrt(252 * 24 * 12)


def _log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {message}", flush=True)


def _compute_pnl(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sign(pred) * y


def _compute_max_drawdown(pnl: torch.Tensor) -> float:
    eq = torch.cumsum(pnl, dim=0)
    peak = torch.cummax(eq, dim=0).values
    dd = peak - eq
    denom = torch.clamp(peak.abs(), min=1e-8)
    return float((dd / denom).max().item())


def _compute_profit_factor(pnl: torch.Tensor) -> float:
    gains = pnl[pnl > 0].sum().item()
    losses = pnl[pnl < 0].sum().item()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / abs(losses))


def evaluate_model(model: TrendModelInterface, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    pred_buf = []
    tgt_buf = []
    ret_buf = []

    with torch.no_grad():
        for batch in loader:
            # Support 3-tensor batches (x, y, actual_return) from updated trend_data
            if len(batch) == 3:
                x, y, actual_ret = batch
                ret_buf.append(actual_ret.detach().cpu())
            else:
                x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x).squeeze(-1)
            loss = criterion(out, y)
            losses.append(loss.item())
            pred_buf.append(out.detach().cpu())
            tgt_buf.append(y.detach().cpu())

    pred = torch.cat(pred_buf)   # logits
    tgt = torch.cat(tgt_buf)     # binary {0, 1}

    # Directional accuracy: logit > 0 means predict "up" (class 1)
    pred_cls = (pred > 0).float()
    directional_acc = float((pred_cls == tgt).float().mean().item())

    # Execution-grade PnL: signal = +1 for up, -1 for down
    signal = pred_cls * 2.0 - 1.0  # {-1, +1}
    ROUND_TRIP_COST = 0.0004
    if ret_buf:
        actual_returns = torch.cat(ret_buf)
        pnl = signal * actual_returns - ROUND_TRIP_COST
    else:
        # Fallback: use label direction
        y_signed = tgt * 2.0 - 1.0
        pnl = signal * y_signed

    pnl_np = pnl.numpy()
    pnl_std = float(np.std(pnl_np)) + 1e-8
    sharpe = float(np.mean(pnl_np) / pnl_std * _annualization_factor())

    return {
        "loss": float(np.mean(losses)),
        "directional_acc": directional_acc,
        "sharpe": sharpe,
        "profit_factor": _compute_profit_factor(pnl),
        "max_drawdown": _compute_max_drawdown(pnl),
    }


def sanity_check(model: TrendModelInterface, batch: int, seq_len: int, input_dim: int, device: torch.device) -> bool:
    model.train()
    x = torch.randn(batch, seq_len, input_dim, device=device)
    y = torch.randint(0, 2, (batch,), device=device).float()  # binary {0, 1}
    criterion = nn.BCEWithLogitsLoss()
    out = model(x).squeeze(-1)
    loss = criterion(out, y)
    loss.backward()

    finite = True
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            finite = False
            break

    model.zero_grad(set_to_none=True)
    cleanup_cuda(x, y, out, loss)
    return finite


def _create_loaders(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    test_ds: TensorDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def train_one_model(
    name: str,
    model_cfg: dict[str, Any],
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    test_ds: TensorDataset,
    common_cfg: dict[str, Any],
    input_dim: int,
) -> TrainingResult:
    device, backend = resolve_device(str(common_cfg.get("preferred_backend", "auto")))
    set_global_seed(int(common_cfg["seed"]))

    seq_len = int(common_cfg["seq_len"])
    base_workers = int(common_cfg["num_workers"])
    if backend == "cuda":
        num_workers = base_workers
    elif backend == "directml":
        num_workers = int(common_cfg.get("num_workers_directml", 0))
    else:
        num_workers = int(common_cfg.get("num_workers_non_cuda", 0))
    pin_memory = backend == "cuda"
    configured_batch_size = int(common_cfg["batch_size"])
    effective_batch_size = configured_batch_size
    if backend == "directml":
        # DirectML VRAM is typically tighter; cap batch to avoid mid-epoch OOM.
        dml_cap = int(common_cfg.get("directml_max_batch_size", 256))
        if configured_batch_size > dml_cap:
            effective_batch_size = dml_cap
            _log(
                f"[trend:{name}] directml batch cap applied configured_batch_size={configured_batch_size} "
                f"effective_batch_size={effective_batch_size}"
            )

    train_loader, val_loader, test_loader = _create_loaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        batch_size=effective_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    _log(
        f"[trend:{name}] start backend={backend} device={device} "
        f"train_windows={len(train_ds)} val_windows={len(val_ds)} test_windows={len(test_ds)} "
        f"batch_size={effective_batch_size} num_workers={num_workers}"
    )

    model = _build_model(name, input_dim=input_dim, seq_len=seq_len, cfg=model_cfg).to(device)
    if bool(common_cfg.get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    sanity_passed = sanity_check(
        model,
        batch=min(32, effective_batch_size),
        seq_len=seq_len,
        input_dim=input_dim,
        device=device,
    )
    if not sanity_passed:
        raise RuntimeError(f"Sanity check failed for {name}")

    _log(f"[trend:{name}] sanity-check passed")

    if backend == "directml":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(common_cfg["lr"]),
            momentum=0.9,
            nesterov=True,
            weight_decay=float(common_cfg["weight_decay"]),
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=float(common_cfg["lr"]),
            weight_decay=float(common_cfg["weight_decay"]),
            foreach=False,
        )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(common_cfg["max_epochs"])))
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=backend == "cuda")

    model_key = _model_artifact_name(name)
    ckpt_dir = Path(common_cfg["checkpoint_root"]) / model_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(Path(common_cfg["tensorboard_root"]) / model_key))

    best_val = float("inf")
    patience = int(common_cfg["patience"])
    wait = 0
    total_batches = max(1, len(train_loader))
    heartbeat_every = min(100, total_batches)
    heartbeat_seconds = int(common_cfg.get("heartbeat_seconds", 60))

    # Optional resume controls for outage recovery.
    resume_from_checkpoint = bool(common_cfg.get("resume_from_checkpoint", False))
    resume_epoch_map = common_cfg.get("resume_start_epoch_by_model", {})
    if isinstance(resume_epoch_map, dict):
        start_epoch = int(resume_epoch_map.get(name, common_cfg.get("resume_start_epoch", 1)))
    else:
        start_epoch = int(common_cfg.get("resume_start_epoch", 1))
    start_epoch = max(1, start_epoch)

    if resume_from_checkpoint and start_epoch > 1:
        model_path = ckpt_dir / "model_best.pt"
        optim_path = ckpt_dir / "optimizer_state.pt"
        if model_path.exists():
            try:
                state = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
            except pickle.UnpicklingError:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"You are using `torch\.load` with `weights_only=False`.*",
                        category=FutureWarning,
                    )
                    state = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
            model.load_state_dict(state)
            _log(f"[trend:{name}] resume loaded model={model_path.name} start_epoch={start_epoch}")
        else:
            _log(f"[trend:{name}] resume requested but missing checkpoint model={model_path}")
            start_epoch = 1

        if start_epoch > 1 and optim_path.exists():
            try:
                optimizer.load_state_dict(torch.load(optim_path, map_location=torch.device("cpu"), weights_only=False))
                _log(f"[trend:{name}] resume loaded optimizer={optim_path.name}")
            except Exception as exc:
                _log(f"[trend:{name}] optimizer resume skipped reason={exc}")

        if start_epoch > 1:
            for _ in range(start_epoch - 1):
                scheduler.step()

    for epoch in range(start_epoch, int(common_cfg["max_epochs"]) + 1):
        model.train()
        epoch_losses = []
        train_correct = 0.0
        train_seen = 0
        epoch_started = time.time()
        _log(f"[trend:{name}] epoch {epoch}/{int(common_cfg['max_epochs'])} started")

        last_heartbeat_ts = time.time()
        for batch_idx, batch in enumerate(train_loader, start=1):
            # Support 3-tensor batches (x, y, actual_return) from updated trend_data
            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = augment_time_series_batch(
                x,
                enabled=bool(common_cfg.get("use_sequence_augmentation", True)),
                mask_prob=float(common_cfg.get("augmentation_mask_prob", 0.02)),
                max_warp=float(common_cfg.get("augmentation_time_warp", 0.05)),
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=backend == "cuda"):
                pred = model(x).squeeze(-1)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(float(loss.item()))
            # Binary accuracy: logit > 0 means predict "up" (class 1)
            train_correct += float(((pred.detach() > 0).float() == y).float().sum().item())
            train_seen += int(y.numel())

            now = time.time()
            should_emit = (
                batch_idx == 1
                or batch_idx % heartbeat_every == 0
                or batch_idx == total_batches
                or (now - last_heartbeat_ts) >= heartbeat_seconds
            )
            if should_emit:
                elapsed_epoch = max(1e-6, now - epoch_started)
                batches_done = float(batch_idx)
                batch_s = elapsed_epoch / batches_done
                eta_s = batch_s * max(0.0, float(total_batches - batch_idx))
                _log(
                    f"[trend:{name}] epoch {epoch} batch {batch_idx}/{total_batches} "
                    f"loss={loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.6g} "
                    f"batch_s={batch_s:.3f} eta_s={eta_s:.0f}"
                )
                last_heartbeat_ts = now

        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device)

        train_loss = float(np.mean(epoch_losses))
        train_acc = float(train_correct / max(1, train_seen))
        val_loss = float(val_metrics["loss"])
        lr = float(optimizer.param_groups[0]["lr"])

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", float(val_metrics["directional_acc"]), epoch)
        writer.add_scalar("train/lr", lr, epoch)

        epoch_elapsed_s = float(time.time() - epoch_started)
        samples_per_s = float(len(train_ds) / max(epoch_elapsed_s, 1e-6))

        _log(
            f"[trend:{name}] epoch {epoch} done train_loss={train_loss:.6f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.6f} val_acc={val_metrics['directional_acc']:.4f} "
            f"val_sharpe={val_metrics['sharpe']:.4f} elapsed_s={epoch_elapsed_s:.1f}"
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
                "val_loss": val_loss,
                "val_acc": float(val_metrics["directional_acc"]),
                "val_sharpe": float(val_metrics["sharpe"]),
                "test_status": "pending",
            },
            epoch=epoch,
            total_epochs=int(common_cfg["max_epochs"]),
        )

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state, ckpt_dir / "model_best.pt")
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer_state.pt")
            _log(f"[trend:{name}] checkpoint saved val_loss={val_loss:.6f}")
        else:
            wait += 1
            if wait >= patience:
                _log(f"[trend:{name}] early-stop patience={patience}")
                break

        cleanup_cuda(pred, loss)

    writer.close()

    load_location = torch.device("cpu")
    try:
        state = torch.load(ckpt_dir / "model_best.pt", map_location=load_location, weights_only=True)
    except pickle.UnpicklingError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch\.load` with `weights_only=False`.*",
                category=FutureWarning,
            )
            state = torch.load(ckpt_dir / "model_best.pt", map_location=load_location, weights_only=False)
    model.load_state_dict(state)
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
    _log(
        f"[trend:{name}] final val_loss={val_metrics['loss']:.6f} test_loss={test_metrics['loss']:.6f} "
        f"val_acc={val_metrics['directional_acc']:.4f} test_acc={test_metrics['directional_acc']:.4f} "
        f"test_sharpe={test_metrics['sharpe']:.4f}"
    )
    append_working_log(
        model_key,
        "FINAL",
        {
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["directional_acc"]),
            "val_sharpe": float(val_metrics["sharpe"]),
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["directional_acc"]),
            "test_sharpe": float(test_metrics["sharpe"]),
            "test_profit_factor": float(test_metrics["profit_factor"]),
            "test_max_drawdown": float(test_metrics["max_drawdown"]),
        },
    )

    overfit_flag = val_metrics["loss"] > 2.0 * max(best_val, 1e-6)
    decay_flag = test_metrics["sharpe"] < 0.5 * val_metrics["sharpe"]

    is_valid = (
        val_metrics["directional_acc"] > 0.55
        and test_metrics["directional_acc"] > 0.55
        and val_metrics["sharpe"] > 1.2
        and test_metrics["sharpe"] > 1.2
        and test_metrics["profit_factor"] > 1.5
        and test_metrics["max_drawdown"] < 0.2
        and not overfit_flag
        and not decay_flag
    )

    result = TrainingResult(
        model_name=model_key,
        checkpoint_dir=str(ckpt_dir).replace("\\", "/"),
        val_loss=float(val_metrics["loss"]),
        test_loss=float(test_metrics["loss"]),
        val_directional_acc=float(val_metrics["directional_acc"]),
        test_directional_acc=float(test_metrics["directional_acc"]),
        val_sharpe=float(val_metrics["sharpe"]),
        test_sharpe=float(test_metrics["sharpe"]),
        test_profit_factor=float(test_metrics["profit_factor"]),
        test_max_drawdown=float(test_metrics["max_drawdown"]),
        is_valid=bool(is_valid),
        backend=backend,
        cuda_used=bool(backend in ("cuda", "directml")),
        sanity_passed=bool(sanity_passed),
    )

    model.cpu()
    cleanup_cuda(model, optimizer, scheduler, scaler)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def write_registry(results: list[TrainingResult], out_path: Path) -> None:
    payload = []
    for r in results:
        payload.append(
            {
                "architecture_name": r.model_name,
                "archetype": "trend_follower",
                "weights_path": f"{r.checkpoint_dir}/model_best.pt",
                "optimizer_state_path": f"{r.checkpoint_dir}/optimizer_state.pt",
                "design_premise": "Trend regime modeling from sequential momentum features.",
                "standard_interface": {"outputs": ["prediction", "confidence"]},
                "validation": {
                    "val_loss": r.val_loss,
                    "test_loss": r.test_loss,
                    "val_directional_accuracy": r.val_directional_acc,
                    "test_directional_accuracy": r.test_directional_acc,
                    "val_sharpe": r.val_sharpe,
                    "test_sharpe": r.test_sharpe,
                    "test_profit_factor": r.test_profit_factor,
                    "test_max_drawdown": r.test_max_drawdown,
                    "backend": r.backend,
                    "cuda_used": r.cuda_used,
                    "sanity_passed": r.sanity_passed,
                    "is_valid": r.is_valid,
                },
            }
        )

    append_registry(payload, out_path)
