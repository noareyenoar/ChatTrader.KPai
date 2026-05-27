#!/usr/bin/env python
"""Sequential training queue runner for all Phase 4 retrain steps.

Each step waits for the previous to complete before starting.
Run from repo root with the .venv activated:

    python tools/run_training_queue.py [--start-step N] [--dry-run]

Steps (1-indexed):
  1  -- Transformer + TCN  (trend_phase4_remaining.yaml)
  2  -- LSTM v2            (trend_phase4_lstm_v2.yaml)
  3  -- MR (GRN/ResNet/MLP)(mr_phase4.yaml)
  4  -- Scalper (3 models) (scalper_phase4.yaml)
  5  -- StatArb (3 models) (stat_arb_phase4.yaml)
  6  -- MM SAC+DQN         (mm_phase4_sac_dqn.yaml)
  7  -- Discretionary      (discretionary_phase4.yaml)
  8  -- TG-MNN             (tg_mnn_phase4.yaml)
  9  -- APV-PLN v1/v2/v3   (apv_pln_phase4.yaml)  [oracle teacher, LUPI]
  10 -- evaluator_run5     (evaluate_all_checkpoints.py)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

VENV_PYTHON = Path(__file__).parent.parent / ".venv" / "Scripts" / "python.exe"
LOG_DIR = Path("doc/iterate_history")

QUEUE: list[dict] = [
    {
        "step": 1,
        "label": "Transformer+TCN (Trend)",
        "module": "quant_core.train_trend_phase4",
        "config": "configs/trend_phase4_remaining.yaml",
        "log": "trend_retrain_remaining.log",
    },
    {
        "step": 2,
        "label": "LSTM v2 (Trend)",
        "module": "quant_core.train_trend_phase4",
        "config": "configs/trend_phase4_lstm_v2.yaml",
        "log": "trend_lstm_v2.log",
    },
    {
        "step": 3,
        "label": "MR (GRN/ResNet/MLP)",
        "module": "quant_core.train_mr_phase4",
        "config": "configs/mr_phase4.yaml",
        "log": "mr_retrain_run1.log",
    },
    {
        "step": 4,
        "label": "Scalper (GRU/CNN/LinearAttn)",
        "module": "quant_core.train_scalper_phase4",
        "config": "configs/scalper_phase4.yaml",
        "log": "scalper_retrain_run1.log",
    },
    {
        "step": 5,
        "label": "StatArb (LSTM/GAT/Autoencoder)",
        "module": "quant_core.train_stat_arb_phase4",
        "config": "configs/stat_arb_phase4.yaml",
        "log": "stat_arb_retrain_run1.log",
    },
    {
        "step": 6,
        "label": "MM SAC+DQN (PPO already passed gates -- skip to protect PASSED model)",
        "module": "quant_core.train_mm_phase4",
        "config": "configs/mm_phase4_sac_dqn.yaml",
        "log": "mm_retrain_run1.log",
    },
    {
        "step": 7,
        "label": "Discretionary (CNNChart/Multimodal/ViT)",
        "module": "quant_core.train_discretionary_phase4",
        "config": "configs/discretionary_phase4.yaml",
        "log": "discretionary_retrain_run1.log",
    },
    {
        "step": 8,
        "label": "TG-MNN",
        "module": "quant_core.train_tg_mnn_phase4",
        "config": "configs/tg_mnn_phase4.yaml",
        "log": "tg_mnn_retrain_run1.log",
    },
    {
        "step": 9,
        "label": "APV-PLN v1/v2/v3 (oracle teacher / LUPI)",
        "module": "quant_core.train_apv_pln_phase4",
        "config": "configs/apv_pln_phase4.yaml",
        "log": "apv_pln_train_run1.log",
    },
    {
        "step": 10,
        "label": "evaluator_run5 (all models OOS evaluation)",
        "script": "evaluate_all_checkpoints.py",
        "log": "evaluator_run5.log",
    },
]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _run_step(step: dict, dry_run: bool) -> int:
    label = step["label"]
    log_path = LOG_DIR / step["log"]
    print(f"\n{'='*70}")
    print(f"[{_ts()}] STARTING STEP {step['step']}: {label}")
    print(f"  log -> {log_path}")
    print(f"{'='*70}")

    if step.get("already_running"):
        print(f"  [SKIP] step {step['step']} already running -- waiting for log file to show completion...")
        # Detect completion by watching the log for "Training complete" or process end
        return 0

    if "script" in step:
        cmd = [str(VENV_PYTHON), step["script"]]
    else:
        cmd = [str(VENV_PYTHON), "-m", step["module"], "--config", step["config"]]

    if dry_run:
        print(f"  [DRY-RUN] would execute: {' '.join(cmd)}")
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_f:
        log_f.write(f"# Training queue step {step['step']}: {label}\n")
        log_f.write(f"# Started: {_ts()}\n")
        log_f.write(f"# Command: {' '.join(cmd)}\n\n")
        log_f.flush()

        # Stream output line-by-line to BOTH terminal and log file so the
        # user can see live progress and confirm the process is not hung.
        proc = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in proc.stdout:
            line_stripped = line.rstrip("\n")
            print(line_stripped, flush=True)   # live terminal output
            log_f.write(line)                  # persist to log file
            log_f.flush()

        proc.wait()
        log_f.write(f"\n# Exit code: {proc.returncode}\n")
        log_f.write(f"# Finished: {_ts()}\n")

    status = "SUCCESS" if proc.returncode == 0 else f"FAILED(rc={proc.returncode})"
    print(f"[{_ts()}] STEP {step['step']} DONE -- {status}", flush=True)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential Phase 4 training queue runner")
    parser.add_argument("--start-step", type=int, default=1, help="Start from this step (default: 1 = full retrain from beginning)")
    parser.add_argument("--end-step", type=int, default=10, help="Stop after this step (inclusive)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--list", action="store_true", help="List all steps and exit")
    args = parser.parse_args()

    if args.list:
        print("Phase 4 training queue:")
        for step in QUEUE:
            print(f"  Step {step['step']:2d}: {step['label']}")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    queue_log = LOG_DIR / "training_queue_run.log"
    with open(queue_log, "a", encoding="utf-8") as ql:
        ql.write(f"\n# Queue started: {_ts()} steps={args.start_step}-{args.end_step}\n")

    failed_steps = []
    for step in QUEUE:
        sn = step["step"]
        if sn < args.start_step or sn > args.end_step:
            continue

        rc = _run_step(step, args.dry_run)
        with open(queue_log, "a", encoding="utf-8") as ql:
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            ql.write(f"  step {sn} {step['label']}: {status} at {_ts()}\n")

        if rc != 0:
            failed_steps.append(sn)
            print(f"\n[ERROR] Step {sn} failed with rc={rc}. Stopping queue.")
            print(f"  Fix the issue and restart with: python {__file__} --start-step {sn}")
            sys.exit(rc)

    print(f"\n{'='*70}")
    print(f"[{_ts()}] QUEUE COMPLETE -- steps {args.start_step}-{args.end_step}")
    if failed_steps:
        print(f"  FAILED: {failed_steps}")
    else:
        print("  ALL STEPS SUCCEEDED")
    print("="*70)


if __name__ == "__main__":
    main()
