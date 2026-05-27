"""
run_48h_autonomous.py
─────────────────────
48-hour autonomous Phase-4 sweep runner.

• Runs every Phase-4 archetype in sequence (trend → MR → scalper → stat_arb
  → discretionary → market_maker → finalize).
• Writes a persistent Glass-Tank log to doc/glass_tank_48h.log.
• Writes a machine-readable heartbeat JSON to doc/glass_tank_heartbeat.json.
• Emits a heartbeat every HEARTBEAT_SEC (default 300 s = 5 min) with:
    - Timestamp, current step, elapsed time, last log tail.
• Records PASS/FAIL for every model checkpoint that appears in the training
  output and stores them in doc/glass_tank_results.json.
• On fatal failure (step exits non-zero twice) → records it, continues to
  the next archetype rather than aborting the whole sweep.

Usage
-----
python tools/run_48h_autonomous.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

LOG_DIR = ROOT / "doc"
LOG_DIR.mkdir(parents=True, exist_ok=True)
GLASS_LOG = LOG_DIR / "glass_tank_48h.log"
HEARTBEAT_JSON = LOG_DIR / "glass_tank_heartbeat.json"
RESULTS_JSON = LOG_DIR / "glass_tank_results.json"

HEARTBEAT_SEC = 300   # emit heartbeat every 5 min
MAX_RETRIES = 2

# ─── Phase-4 sweep steps ─────────────────────────────────────────────────────
STEPS: list[tuple[str, list[str]]] = [
    (
        "trend",
        [
            str(PYTHON),
            "-m",
            "quant_core.train_trend_phase4",
            "--config",
            "configs/trend_phase4_v2.yaml",
        ],
    ),
    (
        "mean_reversion",
        [
            str(PYTHON),
            "-m",
            "quant_core.train_mr_phase4",
            "--config",
            "configs/mr_phase4.yaml",
        ],
    ),
    (
        "scalper",
        [
            str(PYTHON),
            "-m",
            "quant_core.train_scalper_phase4",
            "--config",
            "configs/scalper_phase4.yaml",
        ],
    ),
    (
        "stat_arb",
        [
            str(PYTHON),
            "-m",
            "quant_core.train_stat_arb_phase4",
            "--config",
            "configs/stat_arb_phase4.yaml",
        ],
    ),
    (
        "discretionary",
        [
            str(PYTHON),
            "-m",
            "quant_core.train_discretionary_phase4",
            "--config",
            "configs/discretionary_phase4.yaml",
        ],
    ),
    (
        "market_maker",
        [
            str(PYTHON),
            "-m",
            "quant_core.train_mm_phase4",
            "--config",
            "configs/mm_phase4.yaml",
        ],
    ),
    (
        "finalize",
        [str(PYTHON), "tools/finalize_phase4_results.py"],
    ),
]

# ─── Globals for heartbeat thread ─────────────────────────────────────────────
_current_step: str = "init"
_step_started: float = time.time()
_sweep_started: float = time.time()
_last_lines: list[str] = []
_results: list[dict] = []
_hb_lock = threading.Lock()
_stop_hb = threading.Event()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _emit(msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    try:
        with GLASS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _update_heartbeat(extra: dict | None = None) -> None:
    with _hb_lock:
        payload = {
            "ts": _ts(),
            "current_step": _current_step,
            "step_elapsed_s": round(time.time() - _step_started, 1),
            "sweep_elapsed_s": round(time.time() - _sweep_started, 1),
            "last_lines": list(_last_lines[-10:]),
            "results": list(_results),
        }
        if extra:
            payload.update(extra)
        try:
            HEARTBEAT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def _record_result(archetype: str, model: str, metrics: dict, is_valid: bool) -> None:
    with _hb_lock:
        _results.append({
            "ts": _ts(),
            "archetype": archetype,
            "model": model,
            "metrics": metrics,
            "is_valid": is_valid,
        })
    try:
        RESULTS_JSON.write_text(
            json.dumps(_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


# ─── Line parser: extract model checkpoint events from training logs ──────────

# Matches log lines like:
#   [2026-05-26 10:00:00] [trend:tcn] is_valid=True  sharpe_val=1.23 ...
_VALID_RE = re.compile(
    r"\[trend:(?P<model>\w+)\].*is_valid=(?P<valid>True|False)"
    r".*val_sharpe=(?P<vs>[-\d.]+)"
    r".*test_sharpe=(?P<ts>[-\d.]+)"
    r".*pf=(?P<pf>[-\d.]+)"
    r".*mdd=(?P<mdd>[-\d.]+)",
    re.IGNORECASE,
)

# Also accept [mr:xxx], [scalper:xxx], etc.
_VALID_RE_ARCH = re.compile(
    r"\[(?P<arch>\w+):(?P<model>\w+)\].*is_valid=(?P<valid>True|False)"
    r".*(?:val_sharpe|vs)=(?P<vs>[-\d.]+)"
    r".*(?:test_sharpe|ts)=(?P<ts>[-\d.]+)",
    re.IGNORECASE,
)

# Gate summary events
_GATE_RE = re.compile(
    r"\[(?P<arch>\w+):(?P<model>\w+)\]\s+(?:PASS|FAIL|is_valid=(?:True|False))",
    re.IGNORECASE,
)

# Early stopping / fast-fail events
_EARLY_RE = re.compile(
    r"(early.stop|fast.fail|checkpoint.saved|epoch=?\d+.*val_sharpe=[-\d.]+)",
    re.IGNORECASE,
)


def _parse_and_record(line: str, archetype: str) -> None:
    m = _VALID_RE.search(line) or _VALID_RE_ARCH.search(line)
    if m:
        gd = m.groupdict()
        arch = gd.get("arch", archetype)
        model = gd.get("model", "unknown")
        is_valid = gd.get("valid", "False").lower() == "true"
        metrics = {
            "val_sharpe": float(gd.get("vs") or 0),
            "test_sharpe": float(gd.get("ts") or 0),
            "pf": float(gd.get("pf") or 0),
            "mdd": float(gd.get("mdd") or 0),
        }
        _record_result(arch, model, metrics, is_valid)
        flag = "✓ PASS" if is_valid else "✗ FAIL"
        _emit(
            f"[glass-tank] {flag} arch={arch} model={model} "
            f"val_sharpe={metrics['val_sharpe']:.4f} test_sharpe={metrics['test_sharpe']:.4f} "
            f"pf={metrics['pf']:.4f} mdd={metrics['mdd']:.4f}"
        )
        return

    if _EARLY_RE.search(line):
        _emit(f"[glass-tank] EVENT arch={archetype} | {line.strip()}")


# ─── Heartbeat thread ─────────────────────────────────────────────────────────

def _heartbeat_thread() -> None:
    while not _stop_hb.is_set():
        _update_heartbeat()
        _emit(
            f"[heartbeat] step={_current_step} "
            f"step_elapsed={_fmt_dur(time.time() - _step_started)} "
            f"sweep_elapsed={_fmt_dur(time.time() - _sweep_started)} "
            f"valid_models={sum(1 for r in _results if r['is_valid'])}"
        )
        _stop_hb.wait(HEARTBEAT_SEC)


def _fmt_dur(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


# ─── Step runner ──────────────────────────────────────────────────────────────

def _run_step(step_name: str, command: list[str]) -> int:
    global _current_step, _step_started, _last_lines

    _current_step = step_name
    _step_started = time.time()
    _last_lines.clear()

    _emit(f"[phase4-sweep] ▶ START step={step_name}")
    _emit(f"[phase4-sweep] command={' '.join(command)}")

    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )

    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\r\n")
        if not line:
            continue

        # Split on embedded timestamps (chunked output from subprocess)
        parts = re.split(r"(?=\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])", line)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            print(f"[{_ts()}] {part}", flush=True)
            try:
                with GLASS_LOG.open("a", encoding="utf-8") as fh:
                    fh.write(f"[{_ts()}] {part}\n")
            except Exception:
                pass
            with _hb_lock:
                _last_lines.append(part)
                if len(_last_lines) > 200:
                    _last_lines.pop(0)
            _parse_and_record(part, step_name)

    rc = proc.wait()
    elapsed = time.time() - _step_started
    _emit(
        f"[phase4-sweep] {'✓' if rc == 0 else '✗'} DONE step={step_name} "
        f"exit_code={rc} elapsed={_fmt_dur(elapsed)}"
    )
    return rc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    global _sweep_started
    _sweep_started = time.time()

    _emit("=" * 72)
    _emit(f"[run_48h_autonomous] START  {_ts()}")
    _emit(f"[run_48h_autonomous] Steps: {[s for s,_ in STEPS]}")
    _emit(f"[run_48h_autonomous] Heartbeat every {HEARTBEAT_SEC}s")
    _emit("=" * 72)

    hb_thread = threading.Thread(target=_heartbeat_thread, daemon=True)
    hb_thread.start()

    failed_steps: list[str] = []

    for step_name, command in STEPS:
        rc = -1
        for attempt in range(1, MAX_RETRIES + 1):
            rc = _run_step(step_name, command)
            if rc == 0:
                break
            _emit(f"[phase4-sweep] step={step_name} attempt={attempt}/{MAX_RETRIES} FAILED (exit={rc})")
            if attempt < MAX_RETRIES:
                _emit(f"[phase4-sweep] retrying {step_name} in 15s...")
                time.sleep(15)

        if rc != 0:
            _emit(f"[phase4-sweep] ⚠ GIVING UP step={step_name} — continuing to next archetype")
            failed_steps.append(step_name)
        else:
            _emit(f"[phase4-sweep] ✓ COMPLETED step={step_name}")

    # ── Summary ────────────────────────────────────────────────────────────────
    _stop_hb.set()

    total = time.time() - _sweep_started
    valid_models = [r for r in _results if r["is_valid"]]
    _emit("=" * 72)
    _emit(f"[run_48h_autonomous] SWEEP COMPLETE  elapsed={_fmt_dur(total)}")
    _emit(f"[run_48h_autonomous] Valid models: {len(valid_models)}/{len(_results)}")
    for r in valid_models:
        _emit(
            f"  ✓  arch={r['archetype']} model={r['model']}  "
            f"vs={r['metrics'].get('val_sharpe', 0):.4f}  ts={r['metrics'].get('test_sharpe', 0):.4f}"
        )
    if failed_steps:
        _emit(f"[run_48h_autonomous] Failed steps: {failed_steps}")
    _emit("=" * 72)

    _update_heartbeat({"sweep_complete": True, "failed_steps": failed_steps})

    return 0 if not failed_steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
