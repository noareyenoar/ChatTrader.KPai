#!/usr/bin/env python3
"""
execute_all_phases.py  —  Resilient Queue-Based Model Training Orchestrator
============================================================================

Re-engineered for Phase-4 resilience:
  - Queue-Based Execution: 6 batches, each a list of (archetype, config) pairs
  - Fault Tolerance: try/except per model; failures logged to error_log.json;
    execution always continues to the next model
  - Aggressive Checkpointing: per-epoch saves inside each training loop;
    last_completed_checkpoint.json tracks exact resume point per model dir
  - DirectML Forced: all configs override preferred_backend to "directml"
  - Idempotent: already-PASSED models in registry are skipped unless --force

Usage:
    python execute_all_phases.py [--force] [--batch BATCH_NUM]

    --force       Re-train all models even if already PASSED in registry
    --batch N     Run only batch N (1-6). Default: all batches sequentially.

KPI Gate:  Sharpe > 1.0 AND Directional Accuracy > 0.52

Original Phase 1-4 pipeline scripts remain callable directly:
  execute_phase1_vision_scraper.py
  execute_phase2_feature_engineering.py
  execute_phase3_rl_training.py
  execute_phase4_feature_pruning.py
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_core.validation_policy import PRODUCTION_GATES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
REGISTRY_PATH = ROOT / "model_registry.json"
ERROR_LOG_PATH = ROOT / "error_log.json"
SCALPER_SIGNAL_FIX_FLAG = ROOT / "SCALPER_SIGNAL_INVERSION_FIXED.flag"
COOLDOWN_SECONDS = 30

# ─── KPI Gate ────────────────────────────────────────────────────────────────
SHARPE_GATE = PRODUCTION_GATES.sharpe_min
ACC_GATE = PRODUCTION_GATES.directional_accuracy_min

# ─── Batch Queue ─────────────────────────────────────────────────────────────
# Format: (batch_id, batch_label, script_path, config_path)
# Batches run sequentially; BATCH 2 (Scalper) and BATCH 6 (Disc) deploy after
# Phase A code fixes (scaler persistence + renderer sync).
TRAINING_QUEUE: list[tuple[int, str, str, str]] = [
    (1, "StatArb  [GAT + Autoencoder + LSTM]",
     "quant_core/train_stat_arb_phase4.py", "configs/stat_arb_phase4.yaml"),
    (2, "Scalper  [CNN + LinearAttn + GRU]",
     "quant_core/train_scalper_phase4.py", "configs/scalper_phase4.yaml"),
    (3, "MarketMaker  [PPO from scratch + SAC + DQN]",
     "quant_core/train_mm_phase4.py", "configs/mm_phase4.yaml"),
    (4, "Trend  [Transformer + LSTM]",
     "quant_core/train_trend_phase4.py", "configs/trend_phase4.yaml"),
    (5, "MeanReversion  [ResNet + GRN]",
     "quant_core/train_mr_phase4.py", "configs/mr_phase4.yaml"),
    (6, "Discretionary  [ViT + Multimodal + CNNChart]",
     "quant_core/train_discretionary_phase4.py", "configs/discretionary_phase4.yaml"),
]


# ─── Registry helpers ────────────────────────────────────────────────────────

def _load_registry() -> dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        entries = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return {e["architecture_name"]: e for e in entries}
    except Exception:
        return {}


def _load_error_log() -> list[dict]:
    if not ERROR_LOG_PATH.exists():
        return []
    try:
        return json.loads(ERROR_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _append_error(batch_id: int, batch_label: str, error: str) -> None:
    errors = _load_error_log()
    errors.append({
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "batch_id": batch_id,
        "batch_label": batch_label,
        "error": error,
    })
    ERROR_LOG_PATH.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    logger.error(f"Error appended → {ERROR_LOG_PATH}")


def _require_phase41_gate() -> bool:
    """Phase 4.1 hard lock: block brute-force sweeps until scalper inversion audit is closed."""
    if SCALPER_SIGNAL_FIX_FLAG.exists():
        logger.info(f"Phase 4.1 gate OPEN via {SCALPER_SIGNAL_FIX_FLAG.name}")
        return True
    logger.error(
        "Phase 4.1 gate CLOSED: scalper inversion fix flag missing. "
        "Create SCALPER_SIGNAL_INVERSION_FIXED.flag only after root-cause + code fix is verified."
    )
    return False


# ─── Config patching ─────────────────────────────────────────────────────────

def _patch_config_directml(config_path: str) -> dict[str, Any]:
    """Read YAML config and override preferred_backend to directml."""
    import yaml
    full = ROOT / config_path
    with full.open("r", encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)
    training = cfg.setdefault("training", {})
    training["preferred_backend"] = "directml"
    # Keep CPU pressure low on DirectML runs; TensorDataset loaders do not need many workers.
    training.setdefault("num_workers_directml", 0)

    # Throughput profile for RX 6750: keep preprocessing bounded so GPU is fed continuously.
    data = cfg.setdefault("data", {})
    lower_cfg = config_path.lower()
    if "scalper" in lower_cfg:
        data.setdefault("max_symbols", 16)
        data.setdefault("max_rows_per_symbol", 200000)
        training["batch_size"] = min(int(training.get("batch_size", 1024)), 768)
    elif "trend" in lower_cfg:
        data.setdefault("max_symbols", 20)
        data.setdefault("max_rows_per_symbol", 250000)
        training["batch_size"] = min(int(training.get("batch_size", 1024)), 768)
    elif "stat_arb" in lower_cfg:
        data.setdefault("max_assets", 16)
        data.setdefault("max_rows_per_symbol", 200000)
        training["batch_size"] = min(int(training.get("batch_size", 512)), 384)
    elif "mm" in lower_cfg:
        data.setdefault("max_symbols", 12)
        data.setdefault("max_rows", 4000000)
        training.setdefault("num_envs", 4)
    return cfg


def _write_temp_config(original: str, cfg: dict[str, Any]) -> Path:
    import yaml
    temp = ROOT / f"_temp_cfg_{Path(original).stem}.yaml"
    with temp.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    return temp


# ─── Skip logic ──────────────────────────────────────────────────────────────

# Short model name → registry architecture_name
_NAME_MAP = {
    "GAT": "GAT_StatArb_v1",
    "Autoencoder": "Autoencoder_StatArb_v1",
    "LSTM": "LSTM_StatArb_v1",
    "CNN": "CNN_Scalper_v1",
    "LinearAttn": "LinearAttn_Scalper_v1",
    "GRU": "GRU_Scalper_v1",
    "PPO": "PPO_MM_v1",
    "SAC": "SAC_MM_v1",
    "DQN": "DQN_MM_v1",
    "Transformer": "Transformer_Trend_v1",
    "ResNet": "ResNet_MR_v1",
    "GRN": "GRN_MR_v1",
    "ViT": "ViT_Disc_v1",
    "Multimodal": "Multimodal_Disc_v1",
    "CNNChart": "CNNChart_Disc_v1",
}


def _all_passed(batch_label: str, registry: dict[str, dict]) -> bool:
    """Return True only if EVERY model mentioned in the batch label is PASSED."""
    for short, full in _NAME_MAP.items():
        if short in batch_label:
            entry = registry.get(full, {})
            if entry.get("validation", {}).get("status") != "PASSED":
                return False
    return True


# ─── Batch runner ────────────────────────────────────────────────────────────

def run_batch(batch_id: int, batch_label: str, script: str, config: str, force: bool) -> bool:
    """Execute one training batch. Returns True on success, False on failure.

    Wrapped in try/except — a failure never hangs the outer queue loop.
    """
    sep = "=" * 90
    print(f"\n{sep}", flush=True)
    print(f"  BATCH {batch_id}: {batch_label}", flush=True)
    print(f"  Script : {script}", flush=True)
    print(f"  Config : {config}", flush=True)
    print(f"  Started: {datetime.now(tz=timezone.utc).isoformat()}", flush=True)
    print(sep, flush=True)

    if not force:
        registry = _load_registry()
        if _all_passed(batch_label, registry):
            logger.info(f"Batch {batch_id}: all models already PASSED — skipping (use --force to override)")
            return True

    temp_cfg_path: Path | None = None
    try:
        patched = _patch_config_directml(config)
        temp_cfg_path = _write_temp_config(config, patched)
        module_name = script.replace("/", ".").replace("\\", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        cmd = [PYTHON, "-m", module_name, "--config", str(temp_cfg_path)]
        logger.info(f"CMD: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=ROOT, check=False)
        if proc.returncode == 0:
            logger.info(f"Batch {batch_id} completed (exit 0)")
            return True
        msg = f"Batch {batch_id} exited with code {proc.returncode}"
        logger.warning(msg)
        _append_error(batch_id, batch_label, msg)
        return False
    except FileNotFoundError as exc:
        msg = f"Batch {batch_id} script not found: {exc}"
        logger.error(msg)
        _append_error(batch_id, batch_label, msg)
        return False
    except Exception as exc:
        msg = f"Batch {batch_id} unexpected error: {type(exc).__name__}: {exc}"
        logger.error(msg)
        _append_error(batch_id, batch_label, msg)
        return False
    finally:
        if temp_cfg_path is not None:
            try:
                temp_cfg_path.unlink(missing_ok=True)
            except OSError:
                pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resilient queue-based model training orchestrator"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-train all models even if already PASSED in registry",
    )
    parser.add_argument(
        "--batch", type=int, default=None, metavar="N",
        help="Run only batch N (1-6). Default: all batches.",
    )
    args = parser.parse_args()

    print("=" * 90)
    print("  ChatTrader.KPai — RESILIENT MODEL TRAINING ORCHESTRATOR")
    print("  Hardware  : AMD Radeon RX 6750 (DirectML forced)")
    print(f"  KPI Gate  : Sharpe > {SHARPE_GATE}  AND  OOS Acc > {ACC_GATE}")
    print(f"  Started   : {datetime.now(tz=timezone.utc).isoformat()}")
    print(f"  Force mode: {args.force}")
    print("=" * 90)

    if not _require_phase41_gate():
        print("\n  Training blocked by Phase 4.1 safety gate.")
        print(f"  Required flag: {SCALPER_SIGNAL_FIX_FLAG}")
        return 2

    queue: deque[tuple[int, str, str, str]] = deque(TRAINING_QUEUE)
    if args.batch is not None:
        queue = deque(b for b in TRAINING_QUEUE if b[0] == args.batch)
        if not queue:
            logger.error(f"Batch {args.batch} not found. Valid IDs: 1–6.")
            return 1

    results: list[dict] = []

    while queue:
        batch_id, batch_label, script, config = queue.popleft()
        try:
            success = run_batch(batch_id, batch_label, script, config, force=args.force)
        except Exception as exc:
            # Ultimate safety net
            msg = f"FATAL in batch {batch_id}: {type(exc).__name__}: {exc}"
            logger.error(msg)
            _append_error(batch_id, batch_label, msg)
            success = False

        icon = "✓" if success else "✗"
        logger.info(f"{icon} Batch {batch_id} ({batch_label}) → {'PASS' if success else 'FAIL'}")
        results.append({"batch_id": batch_id, "batch_label": batch_label,
                        "success": success, "ts": datetime.now(tz=timezone.utc).isoformat()})

        # Hardware preservation protocol between batch runs.
        if queue and COOLDOWN_SECONDS > 0:
            logger.info(f"Cooldown {COOLDOWN_SECONDS}s before next batch")
            time.sleep(COOLDOWN_SECONDS)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed

    print("\n" + "=" * 90)
    print("  TRAINING RUN COMPLETE")
    print("=" * 90)
    print(f"\n  Batches : {len(results)}  |  Passed : {passed}  |  Failed : {failed}")
    for r in results:
        icon = "✓" if r["success"] else "✗"
        print(f"    {icon}  Batch {r['batch_id']}: {r['batch_label']}")
    if failed:
        print(f"\n  Failures logged → {ERROR_LOG_PATH}")
        print("  Re-run failed batches with:  python execute_all_phases.py --batch N")
    print(f"\n  Next: validate all models → {PYTHON} evaluate_all_checkpoints.py")
    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
