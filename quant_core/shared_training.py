"""Shared training utilities for all Phase-4 archetypes.

Every archetype training module imports from here to guarantee:
- Consistent device resolution (CUDA → DirectML → CPU)
- Global reproducibility seed
- Registry append (never overwrites entries from other archetypes)
- Common performance metrics
"""
from __future__ import annotations

import json
import gc
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


WORKING_LOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "doc"
    / "training_more_27-4"
    / "27-04-2026_plan_REVISED_workingLog.md"
)


# ---------------------------------------------------------------------------
# Device / seed
# ---------------------------------------------------------------------------

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
                print(f"[device] directml probe failed in auto mode: {probe_exc}", flush=True)
        except Exception as exc:
            if pref == "directml":
                raise RuntimeError(
                    "preferred_backend=directml but torch_directml is unavailable or unhealthy"
                ) from exc
    return torch.device("cpu"), "cpu"


def make_optimizer(model: torch.nn.Module, backend: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Return SGD for DirectML (avoids lerp.Scalar_out fallback), AdamW otherwise."""
    if backend == "directml":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        foreach=False,
    )


def aggressive_cleanup(*objects: Any) -> None:
    """Best-effort cleanup between model transitions to reduce allocator stalls."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def annualization_factor() -> float:
    return math.sqrt(252 * 24 * 12)  # 5-minute bars


def compute_sharpe(pnl: np.ndarray) -> float:
    if len(pnl) < 2:
        return 0.0
    mu = float(np.mean(pnl))
    sigma = float(np.std(pnl, ddof=1)) + 1e-8
    return float(mu / sigma * annualization_factor())


def compute_max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    eq = 1.0 + np.cumsum(np.asarray(pnl, dtype=float))
    eq = np.maximum(eq, 1e-8)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / (peak + 1e-8)
    return float(np.max(dd))


def compute_profit_factor(pnl: np.ndarray) -> float:
    gross_win = float(np.sum(pnl[pnl > 0]) + 1e-8)
    gross_loss = float(np.abs(np.sum(pnl[pnl < 0])) + 1e-8)
    return gross_win / gross_loss


def compute_directional_acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sign(pred) == np.sign(y)))


# ---------------------------------------------------------------------------
# Registry (append-safe — reads existing entries first)
# ---------------------------------------------------------------------------

def append_registry(entries: list[dict[str, Any]], registry_path: Path) -> None:
    """Merge new entries into existing registry, keyed by architecture_name."""
    existing: list[dict] = []
    if registry_path.exists():
        try:
            existing = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    by_name: dict[str, dict] = {e["architecture_name"]: e for e in existing}
    for entry in entries:
        by_name[entry["architecture_name"]] = entry

    registry_path.write_text(json.dumps(list(by_name.values()), indent=2), encoding="utf-8")


def append_working_log(
    model_name: str,
    stage: str,
    metrics: dict[str, Any],
    *,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> None:
    """Append one markdown line to AFK working log with UTC+7 timestamp.

    This function is intentionally best-effort: logging must never interrupt training.
    """
    try:
        ts = datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S UTC+7")
        WORKING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not WORKING_LOG_PATH.exists():
            WORKING_LOG_PATH.write_text(
                "# Phase 4 Working Log (AFK Monitor)\n"
                "\n"
                "Auto-generated training/validation/testing updates from model loops.\n"
                "\n",
                encoding="utf-8",
            )

        epoch_text = ""
        if epoch is not None and total_epochs is not None:
            epoch_text = f" epoch={epoch}/{total_epochs}"
        elif epoch is not None:
            epoch_text = f" epoch={epoch}"

        parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.6f}")
            else:
                parts.append(f"{k}={v}")
        metric_text = " ".join(parts)

        with WORKING_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"- [{ts}] model={model_name} stage={stage}{epoch_text} {metric_text}\n")
    except Exception:
        # Never break training if file logging fails.
        return


# ---------------------------------------------------------------------------
# Checkpoint helpers (DirectML-safe)
# ---------------------------------------------------------------------------

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, ckpt_dir: Path) -> None:
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, ckpt_dir / "model_best.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer_state.pt")


def save_epoch_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ckpt_dir: Path,
    epoch: int,
    val_loss: float,
    keep_last_n: int = 3,
) -> None:
    """Save a per-epoch checkpoint and update last_completed_checkpoint.json.

    Keeps only the most recent `keep_last_n` epoch files to prevent disk exhaustion.
    This enables instant resume from the exact epoch on power failure.
    """
    import json as _json
    from datetime import datetime, timezone

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = ckpt_dir / f"epoch_{epoch}.pt"
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {"epoch": epoch, "model_state_dict": cpu_state, "optimizer_state_dict": optimizer.state_dict()},
        epoch_path,
    )

    # Write resume metadata
    meta = {
        "epoch": epoch,
        "val_loss": round(float(val_loss), 6),
        "checkpoint_file": epoch_path.name,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    (ckpt_dir / "last_completed_checkpoint.json").write_text(
        _json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Prune old epoch files — keep only last `keep_last_n`
    epoch_files = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in epoch_files[:-keep_last_n]:
        try:
            old.unlink()
        except OSError:
            pass


def load_epoch_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, ckpt_dir: Path) -> int:
    """Load the last completed epoch checkpoint for training resume.

    Returns the epoch number that was loaded, or 0 if no checkpoint found.
    """
    import json as _json

    meta_path = ckpt_dir / "last_completed_checkpoint.json"
    if not meta_path.exists():
        return 0
    try:
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        ckpt_path = ckpt_dir / meta["checkpoint_file"]
        if not ckpt_path.exists():
            return 0
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return int(meta["epoch"])
    except Exception:
        return 0


def load_best_checkpoint(model: torch.nn.Module, ckpt_dir: Path) -> None:
    import pickle
    import warnings
    cpu = torch.device("cpu")
    try:
        state = torch.load(ckpt_dir / "model_best.pt", map_location=cpu, weights_only=True)
    except pickle.UnpicklingError:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*weights_only.*", category=FutureWarning)
            state = torch.load(ckpt_dir / "model_best.pt", map_location=cpu, weights_only=False)
    model.load_state_dict(state)
