from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
REGISTRY = ROOT / "model_registry.json"
LOG_PATH = ROOT / "supervision_log.json"

MODEL_TO_BATCH = {
    "GAT_StatArb_v1": 1,
    "Autoencoder_StatArb_v1": 1,
    "LSTM_StatArb_v1": 1,
    "CNN_Scalper_v1": 2,
    "LinearAttn_Scalper_v1": 2,
    "GRU_Scalper_v1": 2,
    "PPO_MM_v1": 3,
    "SAC_MM_v1": 3,
    "DQN_MM_v1": 3,
    "Transformer_Trend_v1": 4,
    "LSTM_Trend_v1": 4,
    "TCN_Trend_v1": 4,
    "ResNet_MR_v1": 5,
    "GRN_MR_v1": 5,
    "MLP_MR_v1": 5,
    "ViT_Disc_v1": 6,
    "Multimodal_Disc_v1": 6,
    "CNNChart_Disc_v1": 6,
}


def _run(cmd: list[str]) -> int:
    print(f"[supervisor] CMD: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    return int(proc.returncode)


def _evaluate() -> int:
    return _run([PY, "evaluate_all_checkpoints.py"])


def _load_registry() -> list[dict]:
    if not REGISTRY.exists():
        return []
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _status(entries: list[dict]) -> tuple[list[str], list[str]]:
    passed = []
    failed = []
    for e in entries:
        name = str(e.get("architecture_name", ""))
        status = str(e.get("validation", {}).get("status", ""))
        if status == "PASSED":
            passed.append(name)
        else:
            failed.append(name)
    return passed, failed


def _append_log(row: dict) -> None:
    rows: list[dict] = []
    if LOG_PATH.exists():
        try:
            rows = json.loads(LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    rows.append(row)
    LOG_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-supervise training until all models pass")
    parser.add_argument("--max-rounds", type=int, default=6)
    args = parser.parse_args()

    print("[supervisor] start", flush=True)

    # Always evaluate first so decisions are based on real OOS results.
    _evaluate()

    for round_idx in range(1, int(args.max_rounds) + 1):
        entries = _load_registry()
        passed, failed = _status(entries)
        print(
            f"[supervisor] round={round_idx} total={len(entries)} passed={len(passed)} failed={len(failed)}",
            flush=True,
        )

        _append_log(
            {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "round": round_idx,
                "total": len(entries),
                "passed": len(passed),
                "failed": len(failed),
                "failed_models": failed,
            }
        )

        if entries and not failed:
            print("[supervisor] all models PASSED", flush=True)
            return 0

        failed_batches = sorted({MODEL_TO_BATCH.get(m) for m in failed if MODEL_TO_BATCH.get(m) is not None})
        if not failed_batches:
            print("[supervisor] no mappable failed batches; stopping", flush=True)
            return 2

        for b in failed_batches:
            rc = _run([PY, "execute_all_phases.py", "--batch", str(b), "--force"])
            if rc != 0:
                print(f"[supervisor] batch {b} exited rc={rc}; continuing", flush=True)

        _evaluate()

    print("[supervisor] max rounds reached without all-pass", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
