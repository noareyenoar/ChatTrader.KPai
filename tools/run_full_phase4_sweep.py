from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

STEPS: list[tuple[str, list[str]]] = [
    (
        "trend",
        [
            str(PYTHON),
            "-m",
            "quant_core.train_trend_phase4",
            "--config",
            "configs/trend_phase4.yaml",
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


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _emit(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _emit_chunked_output(raw_line: str) -> None:
    line = raw_line.rstrip("\r\n")
    if not line:
        return
    # Child output occasionally arrives as multiple timestamped records in one chunk.
    # Split on embedded timestamp starts so each record gets its own emitted line.
    parts = re.split(r"(?=\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\])", line)
    for part in parts:
        part = part.strip()
        if part:
            _emit(part)


def _run_step(name: str, command: list[str]) -> int:
    backend_override = os.environ.get("PHASE4_BACKEND_OVERRIDE", "").strip().lower()
    if backend_override:
        command = _apply_backend_override(command, backend_override)
    started = time.time()
    _emit(f"[phase4-sweep] start step={name}")
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
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            _emit_chunked_output(line)
        return_code = proc.wait()
    except KeyboardInterrupt:
        _emit(f"[phase4-sweep] interrupted step={name}; terminating child process")
        proc.terminate()
        try:
            return_code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            return_code = proc.wait()
        return 130

    elapsed = time.time() - started
    _emit(f"[phase4-sweep] done step={name} exit_code={return_code} elapsed_s={elapsed:.1f}")
    return return_code


def _apply_backend_override(command: list[str], backend: str) -> list[str]:
    if "--config" not in command:
        return command
    idx = command.index("--config")
    if idx + 1 >= len(command):
        return command

    cfg_arg = command[idx + 1]
    cfg_path = (ROOT / cfg_arg).resolve() if not Path(cfg_arg).is_absolute() else Path(cfg_arg)
    if not cfg_path.exists():
        return command

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "training" not in cfg:
        return command
    cfg["training"]["preferred_backend"] = backend

    out_path = ROOT / f"_temp_cfg_{cfg_path.stem}_{backend}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    updated = command.copy()
    updated[idx + 1] = str(out_path)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full Phase-4 training sweep sequentially")
    parser.add_argument(
        "--start-from",
        type=str,
        default="trend",
        choices=[name for name, _ in STEPS],
        help="Resume from a specific step",
    )
    args = parser.parse_args()

    start_index = next(i for i, (name, _) in enumerate(STEPS) if name == args.start_from)
    for name, command in STEPS[start_index:]:
        max_retries = 2
        for attempt in range(1, max_retries + 1):
            code = _run_step(name, command)
            if code == 0:
                break
            _emit(f"[phase4-sweep] step={name} attempt={attempt}/{max_retries} failed exit_code={code}")
            if attempt < max_retries:
                _emit(f"[phase4-sweep] retrying step={name} in 10s")
                time.sleep(10)
        if code != 0:
            _emit(f"[phase4-sweep] stopping on failed step={name} after {max_retries} attempts")
            return code

    _emit("[phase4-sweep] all steps completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())