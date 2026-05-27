"""Reproducibility manifest generator — Phase 6 gap closure.

Emits a JSON artifact per run that captures:
  - git commit hash (HEAD)
  - config file digest (SHA-256)
  - dataset manifest digest (SHA-256)
  - seed lineage
  - Python / PyTorch / torch_directml versions
  - hardware backend
  - run timestamp

Usage:
    from quant_core.run_manifest import generate_manifest
    manifest = generate_manifest(config_path, dataset_manifest_path, seed)
    manifest_path.write_text(json.dumps(manifest, indent=2))
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_file(path: str | Path) -> str:
    """Return hex SHA-256 digest of file contents."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "FILE_NOT_FOUND"


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    """Return the current HEAD commit hash (short) or 'UNKNOWN'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def _git_dirty() -> bool:
    """Return True if working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "NOT_INSTALLED"


def _directml_version() -> str:
    try:
        import torch_directml
        return getattr(torch_directml, "__version__", "INSTALLED_NO_VERSION")
    except ImportError:
        return "NOT_INSTALLED"


def _backend_name() -> str:
    """Detect active compute backend."""
    try:
        import torch_directml
        return "directml"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return "cpu"


def generate_manifest(
    config_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a reproducibility manifest dict.

    Args:
        config_path:            Path to the YAML config used for this run.
        dataset_manifest_path:  Path to the dataset manifest JSON.
        seed:                   RNG seed used for training/evaluation.
        extra:                  Any additional key-value metadata to embed.

    Returns:
        Dict suitable for JSON serialisation.
    """
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git_commit(),
            "dirty_tree": _git_dirty(),
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": _torch_version(),
            "torch_directml_version": _directml_version(),
            "backend": _backend_name(),
            "cpu_count": os.cpu_count(),
        },
        "seed": seed,
    }

    if config_path is not None:
        p = Path(config_path)
        manifest["config"] = {
            "path": str(p),
            "sha256": _sha256_file(p),
            "exists": p.exists(),
        }
        # Also embed config contents as inline string for full reproducibility
        try:
            manifest["config"]["contents"] = p.read_text(encoding="utf-8")
        except OSError:
            manifest["config"]["contents"] = None

    if dataset_manifest_path is not None:
        p = Path(dataset_manifest_path)
        manifest["dataset_manifest"] = {
            "path": str(p),
            "sha256": _sha256_file(p),
            "exists": p.exists(),
        }

    if extra:
        manifest["extra"] = extra

    return manifest


def save_manifest(
    output_path: str | Path,
    config_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and save manifest to ``output_path``.

    Returns the manifest dict.
    """
    m = generate_manifest(config_path, dataset_manifest_path, seed, extra)
    Path(output_path).write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return m
