#!/usr/bin/env python3
"""
Phase 4 Safe Sequential Orchestrator
=====================================
Runs all remaining training jobs ONE AT A TIME (no parallelism)
with conservative batch sizes to prevent PC hang.

Usage:
    python scripts/phase4_safe_sequential.py

Crash recovery awareness (as of 2026-06-09 22:08 hang):
    DONE:  CNN_Scalper_v1, Autoencoder_StatArb_v1, APV_PLN_v1, TG_MNN_v1
    PARTIAL: LinearAttn_Scalper_v1 (epoch 64/120 — restart from scratch with safe batch)
    TODO:  GRU_Scalper_v1, GAT/LSTM_StatArb, APV_PLN_v2/v3,
           Trend×3, Discretionary×3, PPO_MM_v1
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

LOG_PATH = ROOT / "doc" / "iterate_history" / f"phase4_safe_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

STEPS = [
    # ── Scalper: linear_attn + gru (safe batch) ───────────────────────────
    {
        "name": "scalper-safe-no-cnn",
        "cmd": [PY, "-m", "quant_core.train_scalper_phase4",
                "--config", "configs/scalper_phase4_safe_no_cnn.yaml"],
        "cooldown_s": 30,
    },
    # ── Trend: TCN only first (fast ~0.6s/ep DML) ─────────────────────────
    {
        "name": "trend-tcn-only",
        "cmd": [PY, "-m", "quant_core.train_trend_phase4",
                "--config", "configs/trend_phase4_tcn_only.yaml"],
        "cooldown_s": 30,
    },
    # ── TCN OOS eval — decide whether to run full trend ───────────────────
    {
        "name": "eval-tcn",
        "cmd": [PY, "evaluate_all_checkpoints.py", "--model", "TCN_Trend_v1"],
        "cooldown_s": 10,
    },
    # ── Full trend (LSTM + Transformer) — runs only if TCN passes eval ────
    # Controlled in run_step: if TCN_Trend_v1 PASSED_PHASE4 → skip this step
    {
        "name": "trend-lstm-transformer",
        "cmd": [PY, "-m", "quant_core.train_trend_phase4",
                "--config", "configs/trend_phase4.yaml"],
        "skip_if_tcn_passed": True,
        "cooldown_s": 60,
    },
    # ── Discretionary: all 3 (label threshold changed — must retrain) ─────
    {
        "name": "discretionary-full",
        "cmd": [PY, "-m", "quant_core.train_discretionary_phase4",
                "--config", "configs/discretionary_phase4.yaml"],
        "cooldown_s": 60,
    },
    # ── StatArb: GAT + LSTM (Autoencoder already done) ────────────────────
    {
        "name": "stat-arb-gat-lstm",
        "cmd": [PY, "-m", "quant_core.train_stat_arb_phase4",
                "--config", "configs/stat_arb_phase4_gat_lstm_cpu.yaml"],
        "cooldown_s": 30,
    },
    # ── APV-PLN: v2 + v3 (v1 already done) ───────────────────────────────
    {
        "name": "apv-pln-v2v3",
        "cmd": [PY, "-m", "quant_core.train_apv_pln_phase4",
                "--config", "configs/apv_pln_phase4_v2v3_cpu.yaml"],
        "cooldown_s": 30,
    },
    # ── PPO only (safe budget on CPU) ─────────────────────────────────────
    {
        "name": "ppo-safe",
        "cmd": [PY, "-m", "quant_core.train_mm_phase4",
                "--config", "configs/mm_phase4_ppo_safe.yaml",
                "--model", "ppo"],
        "cooldown_s": 30,
    },
    # ── Final OOS evaluation of all models ────────────────────────────────
    {
        "name": "eval-all",
        "cmd": [PY, "evaluate_all_checkpoints.py", "--all"],
        "cooldown_s": 0,
    },
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def registry_tcn_status() -> str | None:
    reg = ROOT / "model_registry.json"
    if not reg.exists():
        return None
    try:
        data = json.loads(reg.read_text(encoding="utf-8"))
        for entry in data:
            if entry.get("architecture_name") == "TCN_Trend_v1":
                return entry.get("validation", {}).get("status")
    except Exception:
        pass
    return None


def run_step(step: dict, skip_tcn_check: bool = False) -> bool:
    """Run one step. Returns True on success."""
    name = step["name"]

    # Skip full trend if TCN already passed
    if step.get("skip_if_tcn_passed"):
        status = registry_tcn_status()
        if status == "PASSED_PHASE4":
            log(f"SKIP {name}: TCN_Trend_v1 already PASSED_PHASE4 — LSTM/Transformer not needed")
            return True
        log(f"TCN status={status} — proceeding with full trend LSTM+Transformer")

    log(f"START {name}")
    t0 = time.time()
    try:
        result = subprocess.run(
            step["cmd"],
            cwd=str(ROOT),
            check=False,
        )
        elapsed = round(time.time() - t0, 1)
        if result.returncode != 0:
            log(f"FAIL  {name}  exit={result.returncode}  elapsed_s={elapsed}")
            return False
        log(f"DONE  {name}  exit=0  elapsed_s={elapsed}")
        return True
    except Exception as exc:
        log(f"ERROR {name}  exc={exc}")
        return False
    finally:
        cd = step.get("cooldown_s", 10)
        if cd > 0:
            log(f"Cooldown {cd}s after {name}...")
            time.sleep(cd)


def main() -> int:
    log("=" * 70)
    log("Phase 4 Safe Sequential Orchestrator — starting")
    log(f"Python: {PY}")
    log(f"Root:   {ROOT}")
    log(f"Log:    {LOG_PATH}")
    log("=" * 70)

    failures: list[str] = []
    for step in STEPS:
        ok = run_step(step)
        if not ok:
            failures.append(step["name"])

    log("=" * 70)
    log(f"DONE — {len(STEPS) - len(failures)}/{len(STEPS)} steps succeeded")

    if failures:
        log(f"FAILED steps: {', '.join(failures)}")
        # Write problem report
        report = ROOT / "doc" / "iterate_history" / f"phase4_problems_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            f"timestamp={datetime.now().isoformat()}\n"
            f"failed_steps={', '.join(failures)}\n"
            f"log={LOG_PATH}\n"
            f"action_required=review log and restart failed steps\n",
            encoding="utf-8",
        )
        log(f"Problem report: {report}")
        return 1

    # Write completion note
    summary = ROOT / "doc" / "iterate_history" / f"phase4_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        reg = ROOT / "model_registry.json"
        registry_data = json.loads(reg.read_text(encoding="utf-8")) if reg.exists() else []
        passed = [e["architecture_name"] for e in registry_data
                  if e.get("validation", {}).get("status") == "PASSED_PHASE4"]
        failed_models = [e["architecture_name"] for e in registry_data
                         if e.get("validation", {}).get("status") not in ("PASSED_PHASE4",)]
        summary.write_text(
            f"timestamp={datetime.now().isoformat()}\n"
            f"total_models={len(registry_data)}\n"
            f"passed_count={len(passed)}\n"
            f"passed={', '.join(passed)}\n"
            f"failed_count={len(failed_models)}\n"
            f"failed_models={', '.join(failed_models)}\n",
            encoding="utf-8",
        )
        log(f"Summary: {summary}")
    except Exception as exc:
        log(f"WARNING: could not write summary: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
