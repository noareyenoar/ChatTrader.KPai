#!/usr/bin/env python3
"""
Ultimate system upgrade orchestrator.

Workflow:
1. Run expanded Phase 1 for Futures + Options (strict 365-day by default)
2. Recursive trigger: rerun Phase 2 and Phase 3 when new files are ingested
3. Run full Phase 4 18-model sweep
4. Enforce strict KPI gates and auto-retry with architecture/hyperparameter tuning
5. Emit strict pass report and full run user guide
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent

STRICT_SHARPE_GATE = 1.0
STRICT_ACC_GATE = 0.52
MAX_RETRY_ROUNDS = 3

EXPECTED_MODELS = [
    "LSTM_Trend_v1",
    "Transformer_Trend_v1",
    "TCN_Trend_v1",
    "MLP_MR_v1",
    "ResNet_MR_v1",
    "GRN_MR_v1",
    "CNN_Scalper_v1",
    "LinearAttn_Scalper_v1",
    "GRU_Scalper_v1",
    "Autoencoder_StatArb_v1",
    "GAT_StatArb_v1",
    "LSTM_StatArb_v1",
    "ViT_Disc_v1",
    "Multimodal_Disc_v1",
    "CNNChart_Disc_v1",
    "PPO_MM_v1",
    "SAC_MM_v1",
    "DQN_MM_v1",
]

ARCHETYPE_TO_CONFIG = {
    "trend_follower": "configs/trend_phase4.yaml",
    "mean_reversion": "configs/mr_phase4.yaml",
    "scalping_microstructure": "configs/scalper_phase4.yaml",
    "statistical_arbitrage": "configs/stat_arb_phase4.yaml",
    "discretionary_multimodal": "configs/discretionary_phase4.yaml",
    "market_making_rl": "configs/mm_phase4.yaml",
}

ARCHETYPE_TO_TRAIN_CMD = {
    "trend_follower": [sys.executable, "-m", "quant_core.train_trend_phase4", "--config", "configs/trend_phase4.yaml"],
    "mean_reversion": [sys.executable, "-m", "quant_core.train_mr_phase4", "--config", "configs/mr_phase4.yaml"],
    "scalping_microstructure": [sys.executable, "-m", "quant_core.train_scalper_phase4", "--config", "configs/scalper_phase4.yaml"],
    "statistical_arbitrage": [sys.executable, "-m", "quant_core.train_stat_arb_phase4", "--config", "configs/stat_arb_phase4.yaml"],
    "discretionary_multimodal": [sys.executable, "-m", "quant_core.train_discretionary_phase4", "--config", "configs/discretionary_phase4.yaml"],
    "market_making_rl": [sys.executable, "-m", "quant_core.train_mm_phase4", "--config", "configs/mm_phase4.yaml"],
}


def run_cmd(cmd: list[str], step: str) -> None:
    print(f"[upgrade] start step={step}")
    print(f"[upgrade] cmd={' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"{step} failed with exit code {proc.returncode}")
    print(f"[upgrade] done step={step}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_yaml(path: Path, obj: dict) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _pick_float(d: dict[str, Any], keys: list[str]) -> float | None:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except Exception:
                return None
    return None


def check_strict_pass(registry_path: Path) -> tuple[bool, list[dict[str, Any]]]:
    entries = load_json(registry_path)
    by_name = {e.get("architecture_name", ""): e for e in entries if isinstance(e, dict)}

    failed: list[dict[str, Any]] = []
    for name in EXPECTED_MODELS:
        e = by_name.get(name)
        if e is None:
            failed.append({"model": name, "reason": "missing_registry_entry", "archetype": "unknown"})
            continue

        v = e.get("validation", {}) if isinstance(e.get("validation", {}), dict) else {}
        sharpe = _pick_float(v, ["test_sharpe", "sharpe_rewards"])
        acc = _pick_float(v, ["test_directional_accuracy", "test_accuracy", "val_directional_accuracy", "val_accuracy"])
        status = str(v.get("status", ""))
        archetype = str(e.get("archetype", "unknown"))

        if sharpe is None or acc is None:
            failed.append(
                {
                    "model": name,
                    "reason": "missing_metric",
                    "archetype": archetype,
                    "sharpe": sharpe,
                    "accuracy": acc,
                    "status": status,
                }
            )
            continue

        if not (sharpe > STRICT_SHARPE_GATE and acc > STRICT_ACC_GATE and status.upper() == "PASSED"):
            failed.append(
                {
                    "model": name,
                    "reason": "strict_gate_failed",
                    "archetype": archetype,
                    "sharpe": sharpe,
                    "accuracy": acc,
                    "status": status,
                }
            )

    return len(failed) == 0, failed


def tune_config_for_retry(config_path: Path, attempt: int) -> None:
    cfg = load_yaml(config_path)

    train = cfg.setdefault("training", {})
    train["preferred_backend"] = "directml"  # AMD RX 6750 path
    train["max_epochs"] = int(train.get("max_epochs", 8)) + (4 * attempt)
    train["patience"] = int(train.get("patience", 6)) + 2
    train["lr"] = max(float(train.get("lr", 1e-3)) * 0.8, 1e-5)
    train["batch_size"] = max(int(int(train.get("batch_size", 256)) * 0.8), 128)

    models = cfg.get("models", {})
    for _, mcfg in models.items():
        if not isinstance(mcfg, dict):
            continue

        if "num_layers" in mcfg:
            mcfg["num_layers"] = min(int(mcfg.get("num_layers", 2)) + 1, 8)
        if "hidden_size" in mcfg:
            mcfg["hidden_size"] = min(int(mcfg.get("hidden_size", 128)) + 32, 512)
        if "channels" in mcfg:
            mcfg["channels"] = min(int(mcfg.get("channels", 64)) + 32, 512)
        if "dropout" in mcfg:
            mcfg["dropout"] = max(float(mcfg.get("dropout", 0.1)) - 0.02, 0.05)

        if "d_model" in mcfg:
            d_model = int(mcfg.get("d_model", 128)) + 32
            nhead = int(mcfg.get("nhead", 4))
            if nhead <= 0:
                nhead = 4
            if d_model % nhead != 0:
                d_model += (nhead - (d_model % nhead))
            mcfg["d_model"] = min(d_model, 512)

    save_yaml(config_path, cfg)


def write_strict_report(path: Path, summary: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Strict PASS Report - Multi-Instrument Upgrade")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Strict Gates: Sharpe > {STRICT_SHARPE_GATE}, Directional Acc > {STRICT_ACC_GATE}")
    lines.append("")

    lines.append("## Phase 1 Expanded")
    lines.append("")
    lines.append(f"- Summary File: {summary.get('summary_path', 'n/a')}")
    lines.append(f"- Downloaded: {summary.get('downloaded', 0)}")
    lines.append(f"- Skipped: {summary.get('skipped', 0)}")
    lines.append(f"- Missing: {summary.get('missing', 0)}")
    lines.append(f"- Failed: {summary.get('failed', 0)}")
    lines.append("")

    lines.append("## Recursive Training")
    lines.append("")
    lines.append(f"- Phase 2 rerun: {summary.get('phase2_rerun', False)}")
    lines.append(f"- Phase 3 rerun: {summary.get('phase3_rerun', False)}")
    lines.append(f"- Phase 4 sweep reruns: {summary.get('phase4_attempts', 0)}")
    lines.append("")

    lines.append("## Final Status")
    lines.append("")
    if not failures:
        lines.append("- STRICT PASS: ALL 18 MODELS GREEN")
    else:
        lines.append("- STRICT PASS: FAILED")
        lines.append("")
        lines.append("| Model | Archetype | Reason | Sharpe | Accuracy | Status |")
        lines.append("|---|---|---|---:|---:|---|")
        for f in failures:
            lines.append(
                f"| {f.get('model', 'n/a')} | {f.get('archetype', 'n/a')} | {f.get('reason', 'n/a')} | "
                f"{f.get('sharpe', 'n/a')} | {f.get('accuracy', 'n/a')} | {f.get('status', 'n/a')} |"
            )

    error_msg = summary.get("error")
    if error_msg:
        lines.append("")
        lines.append("## Execution Error")
        lines.append("")
        lines.append(f"- {error_msg}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_user_guide(path: Path) -> None:
    guide = """# Full Run User Guide - Multi-Instrument Upgrade

## 1) Environment

1. Activate venv.
2. Ensure GPU backend packages are installed (DirectML for AMD):
   - torch
   - torch-directml
   - pandas, pyarrow, aiohttp, pyyaml

## 2) Run Full Upgrade

```powershell
d:/kp_ai_agent/ChatTrader.KPai/.venv/Scripts/python.exe execute_multi_instrument_upgrade.py --days 365 --concurrency 16
```

## 3) What It Executes

1. Expanded Phase 1 downloader:
   - Futures UM + CM: aggTrades, fundingRate, metrics
   - Options BVOL: BTCBVOLUSDT, ETHBVOLUSDT
   - Output roots:
     - Dataset/spot
     - Dataset/futures
     - Dataset/options
2. Recursive trigger:
   - If new files ingested, reruns:
     - execute_phase2_feature_engineering.py
     - execute_phase3_rl_training.py
3. Full 18-model sweep:
   - tools/run_full_phase4_sweep.py
4. Strict KPI enforcement:
   - Sharpe > 1.0
   - Directional Accuracy > 52%
   - Auto-retry with architecture/hyperparameter tuning per failing archetype

## 4) Key Artifacts

- Phase 1 summary:
  - Dataset/multi_instrument/PHASE1_MULTI_INSTRUMENT_SUMMARY.json
- Common sync index:
  - Dataset/multi_instrument/common_timestamp_index.parquet
- Multi-instrument Phase 2 feature table:
  - Dataset/processed/multi_instrument/BTCUSDT_multi_instrument_state.parquet
- Registry and summary:
  - model_registry.json
  - model_performance_summary.md
- Strict pass report:
  - MULTI_INSTRUMENT_STRICT_PASS_REPORT.md

## 5) Resume / Recovery

- Re-run full command. Existing parquet files are checksum-validated and skipped when valid.
- If strict pass fails, rerun with same command; tuner continues from latest configs.

## 6) GPU Notes (AMD RX 6750)

- All Phase 4 configs are forced to `preferred_backend: directml` during retry tuning.
- This keeps AMD GPU as main accelerator where supported.
"""
    path.write_text(guide, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full multi-instrument system upgrade")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    summary: dict[str, Any] = {
        "phase2_rerun": False,
        "phase3_rerun": False,
        "phase4_attempts": 0,
    }

    strict_ok = False
    failures: list[dict[str, Any]] = []
    try:
        run_cmd(
            [
                sys.executable,
                "execute_phase1_multi_instrument.py",
                "--days",
                str(args.days),
                "--concurrency",
                str(args.concurrency),
            ],
            "phase1_multi_instrument",
        )

        p1_summary_path = ROOT / "Dataset" / "multi_instrument" / "PHASE1_MULTI_INSTRUMENT_SUMMARY.json"
        p1 = load_json(p1_summary_path)
        summary["summary_path"] = str(p1_summary_path).replace("\\", "/")
        summary["downloaded"] = int(p1.get("counts", {}).get("downloaded", 0))
        summary["skipped"] = int(p1.get("counts", {}).get("skipped", 0))
        summary["missing"] = int(p1.get("counts", {}).get("missing", 0))
        summary["failed"] = int(p1.get("counts", {}).get("failed", 0))

        if summary["downloaded"] > 0:
            run_cmd([sys.executable, "execute_phase2_feature_engineering.py"], "phase2_rerun")
            run_cmd([sys.executable, "execute_phase3_rl_training.py"], "phase3_rerun")
            summary["phase2_rerun"] = True
            summary["phase3_rerun"] = True

        run_cmd([sys.executable, "tools/run_full_phase4_sweep.py"], "phase4_full_sweep")
        run_cmd([sys.executable, "tools/finalize_phase4_results.py"], "finalize_initial")
        summary["phase4_attempts"] += 1

        registry_path = ROOT / "model_registry.json"
        strict_ok, failures = check_strict_pass(registry_path)

        retry_round = 0
        while (not strict_ok) and retry_round < MAX_RETRY_ROUNDS:
            retry_round += 1
            print(f"[upgrade] strict gate failures={len(failures)} retry_round={retry_round}")

            failing_archetypes = sorted({str(f.get("archetype", "unknown")) for f in failures})
            for archetype in failing_archetypes:
                cfg_rel = ARCHETYPE_TO_CONFIG.get(archetype)
                train_cmd = ARCHETYPE_TO_TRAIN_CMD.get(archetype)
                if cfg_rel is None or train_cmd is None:
                    continue
                tune_config_for_retry(ROOT / cfg_rel, retry_round)
                run_cmd(train_cmd, f"retrain_{archetype}_r{retry_round}")

            run_cmd([sys.executable, "tools/finalize_phase4_results.py"], f"finalize_r{retry_round}")
            summary["phase4_attempts"] += 1
            strict_ok, failures = check_strict_pass(registry_path)
    except KeyboardInterrupt:
        summary["error"] = "Interrupted by user (KeyboardInterrupt)"
    except Exception as exc:
        summary["error"] = str(exc)

    report_path = ROOT / "MULTI_INSTRUMENT_STRICT_PASS_REPORT.md"
    write_strict_report(report_path, summary, failures)

    guide_path = ROOT / "RUN_USER_GUIDE_MULTI_INSTRUMENT.md"
    write_user_guide(guide_path)

    print(f"[upgrade] strict_report={report_path}")
    print(f"[upgrade] user_guide={guide_path}")

    if summary.get("error"):
        print(f"[upgrade] execution failed: {summary['error']}")
        if "KeyboardInterrupt" in str(summary["error"]):
            return 130
        return 1

    if not strict_ok:
        print("[upgrade] strict pass not achieved within retry budget")
        return 2

    print("[upgrade] STRICT PASS achieved for all 18 models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
