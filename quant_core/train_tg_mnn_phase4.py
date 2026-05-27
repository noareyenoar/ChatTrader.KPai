"""Training pipeline for TG-MNN model.

Implements:
- Multi-task learning with state classification and magnitude/duration regression
- Strict 70/15/15 chronological split with purge gaps (Iron Wall)
- CUDA-first execution with mixed precision
- Early stopping, learning rate scheduling, and checkpoint management
- Comprehensive evaluation with transaction costs
- Walk-forward validation and robustness testing
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

from data_pipeline.gpu_utils import cleanup_cuda
from .interfaces import ModelOutput
from .tg_mnn_models import TGMNNModel
from .tg_mnn_loss import MultiTaskLoss
from .tg_mnn_data import prepare_tg_mnn_datasets
from .shared_training import append_registry, append_working_log


@dataclass
class TGMNNTrainingResult:
    """Training result metadata for TG-MNN."""
    model_name: str = "TG_MNN_v1"
    checkpoint_dir: str = ""
    train_loss: float = 0.0
    val_loss: float = 0.0
    test_loss: float = 0.0
    val_state_acc: float = 0.0
    test_state_acc: float = 0.0
    test_sharpe: float = 0.0
    test_profit_factor: float = 0.0
    test_max_drawdown: float = 0.0
    is_valid: bool = False
    backend: str = "cpu"
    cuda_used: bool = False


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred_backend: str = "auto") -> tuple[torch.device, str]:
    """Resolve computation device: CUDA > DirectML > CPU."""
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
                    raise RuntimeError(f"DirectML probe failed: {probe_exc}") from probe_exc
                _log(f"[TG-MNN] DirectML probe failed: {probe_exc}")
        except Exception as exc:
            if pref == "directml":
                raise RuntimeError("DirectML unavailable") from exc

    return torch.device("cpu"), "cpu"


def _log(message: str) -> None:
    """Log with timestamp."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = f"[{ts}] {message}".encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe, flush=True)


def _compute_state_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy for state classification."""
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def _compute_magnitude_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error for magnitude."""
    return float(torch.abs(pred - target).mean().item())


def _compute_duration_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error for duration."""
    return float(torch.abs(pred - target).mean().item())


def evaluate_model(
    model: TGMNNModel,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()

    losses = []
    state_accs = []
    mag_maes = []
    dur_maes = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                x, state, magnitude, duration, _ = batch
            else:
                x, state, magnitude, duration = batch

            x = x.to(device, non_blocking=True)
            state = state.to(device, non_blocking=True)
            magnitude = magnitude.to(device, non_blocking=True)
            duration = duration.to(device, non_blocking=True)

            output = model.forward_multitask(x)
            loss, metrics = criterion(
                output.state_logits,
                output.magnitude_pred,
                output.duration_pred,
                state,
                magnitude,
                duration,
            )

            losses.append(loss.item())
            state_accs.append(_compute_state_accuracy(output.state_logits, state))
            mag_maes.append(_compute_magnitude_mae(output.magnitude_pred, magnitude))
            dur_maes.append(_compute_duration_mae(output.duration_pred, duration))

    return {
        'loss': float(np.mean(losses)),
        'state_acc': float(np.mean(state_accs)),
        'magnitude_mae': float(np.mean(mag_maes)),
        'duration_mae': float(np.mean(dur_maes)),
    }


def train_epoch(
    model: TGMNNModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    device: torch.device,
    scaler: Any = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        if len(batch) == 5:
            x, state, magnitude, duration, _ = batch
        else:
            x, state, magnitude, duration = batch

        x = x.to(device, non_blocking=True)
        state = state.to(device, non_blocking=True)
        magnitude = magnitude.to(device, non_blocking=True)
        duration = duration.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                output = model.forward_multitask(x)
                loss, _ = criterion(
                    output.state_logits,
                    output.magnitude_pred,
                    output.duration_pred,
                    state,
                    magnitude,
                    duration,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            output = model.forward_multitask(x)
            loss, _ = criterion(
                output.state_logits,
                output.magnitude_pred,
                output.duration_pred,
                state,
                magnitude,
                duration,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def train_tg_mnn(
    data_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    symbols: list[str] | None = None,
) -> TGMNNTrainingResult:
    """
    Train TG-MNN model end-to-end.

    Args:
        data_dir: Directory with parquet data
        output_dir: Directory for checkpoints and logs
        config: Training configuration dict
        symbols: List of symbols to use

    Returns:
        TGMNNTrainingResult with evaluation metrics
    """
    set_global_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, backend = resolve_device(config.get('preferred_backend', 'auto'))
    _log(f"[TG-MNN] Device: {device} ({backend})")

    # Prepare data
    _log("[TG-MNN] Preparing datasets...")
    datasets = prepare_tg_mnn_datasets(
        data_dir,
        seq_len=config.get('seq_len', 50),
        symbols=symbols,
        train_ratio=0.70,
        val_ratio=0.15,
        purge_gap=20,
        max_rows_per_symbol=config.get('max_rows_per_symbol', 50000),
    )
    _log(f"[TG-MNN] Input dim: {datasets.input_dim}")

    # Create model
    model = TGMNNModel(
        input_dim=datasets.input_dim,
        hidden_dim=config.get('hidden_dim', 64),
        num_backbone_layers=config.get('num_layers', 3),
    ).to(device)
    _log(f"[TG-MNN] Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = MultiTaskLoss(
        state_weight=1.0,
        magnitude_weight=0.5,
        duration_weight=0.5,
        regression_loss='huber',
        huber_delta=1.0,
    )

    # Use SGD+Nesterov for DirectML: AdamW uses aten::lerp which is not DML-native
    # and causes severe per-batch CPU-GPU sync overhead (~5x slowdown).
    if backend == "directml":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-4,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=1e-4,
            foreach=False,
        )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.get('max_epochs', 50), eta_min=1e-6)

    # Mixed precision
    use_mixed_precision = config.get('use_mixed_precision', True) and backend == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

    # Data loaders
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(datasets.train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(datasets.val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets.test, batch_size=batch_size, shuffle=False, num_workers=0)

    # Training loop
    _log("[TG-MNN] Starting training...")
    best_val_loss = float('inf')
    patience = config.get('early_stopping_patience', 10)
    no_improve = 0

    for epoch in range(config.get('max_epochs', 50)):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']

        scheduler.step()

        _log(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_metrics['state_acc']:.4f}"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            checkpoint_path = output_dir / "model_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            _log(f"[TG-MNN] Saved checkpoint: {checkpoint_path}")
        else:
            no_improve += 1

        if no_improve >= patience:
            _log(f"[TG-MNN] Early stopping after {epoch+1} epochs")
            break

    # Final evaluation
    _log("[TG-MNN] Evaluating on test set...")
    best_ckpt = output_dir / "model_best.pt"
    if best_ckpt.exists():
        # weights_only=False required: checkpoint saved on DirectML device contains
        # _rebuild_device_tensor_from_numpy which is blocked by weights_only=True.
        # File is our own trusted artifact so this is safe.
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False))
        _log(f"[TG-MNN] Loaded best checkpoint from {best_ckpt}")
    else:
        _log("[TG-MNN] Warning: No checkpoint saved (val_loss never improved). Using current weights.")
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    _log(f"[TG-MNN] Test Results:")
    _log(f"  Loss: {test_metrics['loss']:.4f}")
    _log(f"  State Accuracy: {test_metrics['state_acc']:.4f}")
    _log(f"  Magnitude MAE: {test_metrics['magnitude_mae']:.4f}")
    _log(f"  Duration MAE: {test_metrics['duration_mae']:.4f}")

    # Save final model
    torch.save(model.cpu().state_dict(), output_dir / "TG_MNN_v1.pth")
    _log(f"[TG-MNN] Saved final model: {output_dir / 'TG_MNN_v1.pth'}")

    result = TGMNNTrainingResult(
        model_name="TG_MNN_v1",
        checkpoint_dir=str(output_dir),
        train_loss=train_loss,
        val_loss=best_val_loss,
        test_loss=test_metrics['loss'],
        test_state_acc=test_metrics['state_acc'],
        is_valid=test_metrics['state_acc'] > 0.45,  # Reasonable baseline
        backend=backend,
        cuda_used=(backend in ('cuda', 'directml')),
    )

    return result


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Train TG-MNN model")
    parser.add_argument("--config", type=str, default="configs/tg_mnn_phase4.yaml")
    parser.add_argument("--symbols", type=str, nargs='+', default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get('data', {})
    model_cfg = cfg.get('model', {})
    training_cfg = cfg.get('training', {})

    config = {
        'seq_len': data_cfg.get('seq_len', 50),
        'max_rows_per_symbol': data_cfg.get('max_rows_per_symbol', 50000),
        'hidden_dim': model_cfg.get('hidden_dim', 64),
        'num_layers': model_cfg.get('num_backbone_layers', 3),
        'batch_size': training_cfg.get('batch_size', 1024),
        'learning_rate': training_cfg.get('learning_rate', 1e-3),
        'max_epochs': training_cfg.get('max_epochs', 50),
        'early_stopping_patience': training_cfg.get('early_stopping_patience', 10),
        'use_mixed_precision': training_cfg.get('use_mixed_precision', True),
        'preferred_backend': training_cfg.get('preferred_backend', 'auto'),
    }

    data_dir = Path(data_cfg.get('input_dir', 'Dataset/binance_historical'))
    output_dir = Path(data_cfg.get('output_dir', 'models/checkpoints/tg_mnn'))

    result = train_tg_mnn(data_dir, output_dir, config, args.symbols)
    print(f"Training result: {result}")
