#!/usr/bin/env python3
"""
glass_tank_monitor.py  — Training Glass Tank for ChatTrader.KPai V2.0
======================================================================
Reads the live training log, parses key events and emits a structured
heartbeat every HEARTBEAT_INTERVAL_SECONDS (default 7200 = 2 hours).

Usage:
    python tools/glass_tank_monitor.py --log doc/training_v2_glass_tank.log
                                        [--heartbeat 7200]
                                        [--training-log doc/training_v2.log]

Events tracked (regex-matched from training stdout):
    EPOCH_DONE     — per-epoch train/val metrics
    CKPT_SAVED     — new best checkpoint written
    FAST_FAIL      — fast-fail abort
    DIVERGENCE     — divergence alert
    OVERFIT        — overfit alert
    FINAL          — final val/test metrics after full training
    MODEL_DONE     — model archetype complete
    LUPI           — LUPI activation detected
    ERROR          — any Python traceback

Heartbeat report written to:
    doc/training_v2_glass_tank.log   (append-safe, human-readable)
    doc/training_v2_heartbeat.json   (machine-readable, last heartbeat only)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ─── Regex patterns ────────────────────────────────────────────────────────
# Matches: [trend:tcn] epoch 46 done train_loss=0.449 train_acc=0.630 val_loss=0.773 val_acc=0.552 val_sharpe=1.374 elapsed_s=478.2
RE_EPOCH = re.compile(
    r"\[trend:(?P<model>\w+)\] epoch (?P<epoch>\d+) done "
    r"train_loss=(?P<train_loss>[\d.]+) train_acc=(?P<train_acc>[\d.]+) "
    r"val_loss=(?P<val_loss>[\d.]+) val_acc=(?P<val_acc>[\d.]+) "
    r"val_sharpe=(?P<val_sharpe>[-\d.]+)"
)
# Matches: [trend:tcn] epoch 47/120 started
RE_EPOCH_START = re.compile(
    r"\[trend:(?P<model>\w+)\] epoch (?P<epoch>\d+)/(?P<max_epoch>\d+) started"
)
RE_CKPT = re.compile(
    r"\[trend:(?P<model>\w+)\] checkpoint saved val_sharpe=(?P<val_sharpe>[-\d.]+) "
    r"val_acc=(?P<val_acc>[\d.]+) val_loss=(?P<val_loss>[\d.]+)"
)
RE_FAST_FAIL = re.compile(
    r"\[trend:(?P<model>\w+)\] FAST-FAIL triggered epoch=(?P<epoch>\d+) "
    r"val_sharpe=(?P<val_sharpe>[-\d.]+)"
)
RE_DIVERGENCE = re.compile(
    r"\[trend:(?P<model>\w+)\] DIVERGENCE-ALERT val_sharpe=(?P<val_sharpe>[-\d.]+) "
    r"test_sharpe=(?P<test_sharpe>[-\d.]+) abs_gap=(?P<abs_gap>[\d.]+)"
)
RE_OVERFIT = re.compile(
    r"\[trend:(?P<model>\w+)\] OVERFIT-ALERT epoch=(?P<epoch>\d+) "
    r"val_sharpe=(?P<val_sharpe>[-\d.]+)"
)
RE_FINAL = re.compile(
    r"\[trend:(?P<model>\w+)\] final val_loss=(?P<val_loss>[\d.]+) test_loss=(?P<test_loss>[\d.]+) "
    r"val_acc=(?P<val_acc>[\d.]+) test_acc=(?P<test_acc>[\d.]+) "
    r"test_sharpe=(?P<test_sharpe>[-\d.]+)"
)
RE_TRAINED = re.compile(
    r"trained=(?P<model>\w+) .*?val_acc=(?P<val_acc>[\d.]+) test_acc=(?P<test_acc>[\d.]+) "
    r".*?is_valid=(?P<is_valid>True|False)"
)
RE_EARLY_STOP = re.compile(r"\[trend:(?P<model>\w+)\] early-stop patience=(?P<patience>\d+)")
RE_SANITY = re.compile(r"\[trend:(?P<model>\w+)\] sanity-check passed")
RE_LUPI = re.compile(r"\[trend:(?P<model>\w+)\] LUPI enabled.*kl_weight=(?P<kl_weight>[\d.]+)")
RE_DATASETS = re.compile(
    r"\[trend-run\] datasets ready train_windows=(?P<train>\d+) "
    r"val_windows=(?P<val>\d+) test_windows=(?P<test>\d+) input_dim=(?P<dim>\d+)"
)
RE_TRACEBACK = re.compile(r"Traceback \(most recent call last\)")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_sharpe(v) -> str:
    try:
        return f"{float(v):+.4f}"
    except Exception:
        return str(v)


class GlassTank:
    def __init__(self, training_log: Path, out_log: Path, heartbeat_json: Path) -> None:
        self.training_log = training_log
        self.out_log = out_log
        self.heartbeat_json = heartbeat_json

        # Per-model state
        self.model_state: dict[str, dict] = defaultdict(lambda: {
            "epochs_done": 0,
            "max_epochs": 0,
            "best_val_sharpe": float("-inf"),
            "best_val_acc": 0.0,
            "latest_train_loss": None,
            "latest_val_loss": None,
            "final_test_sharpe": None,
            "final_test_acc": None,
            "is_valid": None,
            "events": [],
        })
        self.dataset_windows: dict = {}
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        self.lines_read = 0
        self.error_count = 0
        self.heartbeat_count = 0
        self._log_file_handle = None

    def _emit(self, text: str) -> None:
        """Write line to glass-tank log AND stdout."""
        line = f"{text}\n"
        print(text, flush=True)
        if self._log_file_handle:
            self._log_file_handle.write(line)
            self._log_file_handle.flush()

    def _parse_line(self, line: str) -> None:
        """Parse a single training log line and update state."""
        m = RE_DATASETS.search(line)
        if m:
            self.dataset_windows = {
                "train": int(m.group("train")),
                "val": int(m.group("val")),
                "test": int(m.group("test")),
                "input_dim": int(m.group("dim")),
            }
            self._emit(f"  [DATASETS] {_ts()} — train={m.group('train')} val={m.group('val')} test={m.group('test')} dim={m.group('dim')}")
            return

        m = RE_SANITY.search(line)
        if m:
            model = m.group("model")
            self.model_state[model]["events"].append(("SANITY_PASS", _ts()))
            self._emit(f"  [SANITY  ] {_ts()} model={model} — sanity-check PASSED, training starting")
            return

        m = RE_LUPI.search(line)
        if m:
            model = m.group("model")
            self.model_state[model]["events"].append(("LUPI_ACTIVE", _ts()))
            self._emit(f"  [LUPI    ] {_ts()} model={model} kl_weight={m.group('kl_weight')} — Oracle Teacher ACTIVE")
            return

        m = RE_EPOCH_START.search(line)
        if m:
            model = m.group("model")
            self.model_state[model]["max_epochs"] = int(m.group("max_epoch"))
            return

        m = RE_EPOCH.search(line)
        if m:
            model = m.group("model")
            epoch = int(m.group("epoch"))
            val_sharpe = float(m.group("val_sharpe"))
            val_acc = float(m.group("val_acc"))
            s = self.model_state[model]
            s["epochs_done"] = epoch
            s["latest_train_loss"] = float(m.group("train_loss"))
            s["latest_val_loss"] = float(m.group("val_loss"))
            max_ep = s["max_epochs"] or "?"
            pct = f"{100*epoch//max_ep}%" if isinstance(max_ep, int) and max_ep else "?%"
            is_best = val_sharpe > s["best_val_sharpe"]
            if is_best:
                s["best_val_sharpe"] = val_sharpe
                s["best_val_acc"] = val_acc
            # Emit progress line for every epoch (compact, always visible in log)
            flag = " [NEW BEST]" if is_best else ""
            self._emit(
                f"  [EPOCH   ] {_ts()} model={model} epoch={epoch}/{max_ep} ({pct})"
                f"  train_loss={m.group('train_loss')}  val_sharpe={_fmt_sharpe(val_sharpe)}"
                f"  val_acc={val_acc:.4f}{flag}"
            )
            return

        m = RE_CKPT.search(line)
        if m:
            model = m.group("model")
            val_sharpe = float(m.group("val_sharpe"))
            s = self.model_state[model]
            s["best_val_sharpe"] = val_sharpe
            s["best_val_acc"] = float(m.group("val_acc"))
            s["events"].append(("CKPT_SAVED", _ts(), val_sharpe))
            self._emit(
                f"  [CKPT    ] {_ts()} model={model} "
                f"NEW_BEST val_sharpe={_fmt_sharpe(val_sharpe)} "
                f"val_acc={m.group('val_acc')}"
            )
            return

        m = RE_FAST_FAIL.search(line)
        if m:
            model = m.group("model")
            self.model_state[model]["events"].append(("FAST_FAIL", _ts()))
            self._emit(
                f"  [FAST_FAIL] {_ts()} model={model} epoch={m.group('epoch')} "
                f"val_sharpe={_fmt_sharpe(m.group('val_sharpe'))} — ABORTED"
            )
            return

        m = RE_DIVERGENCE.search(line)
        if m:
            model = m.group("model")
            self.model_state[model]["events"].append(("DIVERGENCE_ALERT", _ts(), float(m.group("abs_gap"))))
            self._emit(
                f"  [DIVERGE ] {_ts()} model={model} "
                f"val_sharpe={_fmt_sharpe(m.group('val_sharpe'))} "
                f"test_sharpe={_fmt_sharpe(m.group('test_sharpe'))} "
                f"abs_gap={m.group('abs_gap')} — V2.0 DIVERGENCE GATE FAILED"
            )
            return

        m = RE_OVERFIT.search(line)
        if m:
            model = m.group("model")
            self.model_state[model]["events"].append(("OVERFIT_ALERT", _ts()))
            self._emit(
                f"  [OVERFIT ] {_ts()} model={model} epoch={m.group('epoch')} "
                f"val_sharpe={_fmt_sharpe(m.group('val_sharpe'))} — overfit pattern detected"
            )
            return

        m = RE_EARLY_STOP.search(line)
        if m:
            model = m.group("model")
            self.model_state[model]["events"].append(("EARLY_STOP", _ts()))
            self._emit(f"  [ESTOP   ] {_ts()} model={model} — early-stop triggered (patience={m.group('patience')})")
            return

        m = RE_FINAL.search(line)
        if m:
            model = m.group("model")
            s = self.model_state[model]
            s["final_test_sharpe"] = float(m.group("test_sharpe"))
            s["final_test_acc"] = float(m.group("test_acc"))
            s["events"].append(("FINAL_EVAL", _ts()))
            self._emit(
                f"  [FINAL   ] {_ts()} model={model} "
                f"val_acc={m.group('val_acc')} test_acc={m.group('test_acc')} "
                f"test_sharpe={_fmt_sharpe(m.group('test_sharpe'))}"
            )
            return

        m = RE_TRAINED.search(line)
        if m:
            model = m.group("model")
            is_valid = m.group("is_valid") == "True"
            self.model_state[model]["is_valid"] = is_valid
            verdict = "✅ PASSED V2.0 GATES" if is_valid else "❌ FAILED — RESUME REQUIRED"
            self._emit(
                f"  [VERDICT ] {_ts()} model={model} is_valid={is_valid} — {verdict} "
                f"val_acc={m.group('val_acc')} test_acc={m.group('test_acc')}"
            )
            return

        if RE_TRACEBACK.search(line):
            self.error_count += 1
            self._emit(f"  [ERROR   ] {_ts()} — Python traceback detected! (error #{self.error_count})")
            return

    def _heartbeat(self) -> None:
        self.heartbeat_count += 1
        elapsed_h = (time.time() - self.start_time) / 3600.0
        sep = "=" * 80

        self._emit(f"\n{sep}")
        self._emit(f"  💓 HEARTBEAT #{self.heartbeat_count}  |  {_ts()}  |  Elapsed: {elapsed_h:.2f}h")
        self._emit(sep)

        # ── Dataset summary ───────────────────────────────────────────────
        if self.dataset_windows:
            dw = self.dataset_windows
            self._emit(
                f"  DATASET  train_windows={dw.get('train','?')}  "
                f"val_windows={dw.get('val','?')}  "
                f"test_windows={dw.get('test','?')}  "
                f"input_dim={dw.get('input_dim','?')}"
            )

        # ── Per-model progress ────────────────────────────────────────────
        self._emit(f"\n  {'MODEL':<25} {'EPOCH':>8} {'B_VSHARPE':>10} {'B_VACC':>7} {'T_SHARPE':>9} {'T_ACC':>7} {'VALID':>7}")
        self._emit(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*7} {'-'*9} {'-'*7} {'-'*7}")
        for model, s in self.model_state.items():
            ep_str = f"{s['epochs_done']}/{s['max_epochs']}" if s["max_epochs"] else f"{s['epochs_done']}/?"
            pct = f"({100*s['epochs_done']//s['max_epochs']}%)" if s["max_epochs"] else ""
            t_sharpe = _fmt_sharpe(s["final_test_sharpe"]) if s["final_test_sharpe"] is not None else "pending"
            t_acc = f"{s['final_test_acc']:.4f}" if s["final_test_acc"] is not None else "pending"
            valid_str = ("PASS" if s["is_valid"] else "FAIL") if s["is_valid"] is not None else "running"
            self._emit(
                f"  {model:<25} {ep_str:>8}{pct:>5} "
                f"{_fmt_sharpe(s['best_val_sharpe']):>10} "
                f"{s['best_val_acc']:>7.4f} "
                f"{t_sharpe:>9} "
                f"{t_acc:>7} "
                f"{valid_str:>7}"
            )

        # ── Event log ─────────────────────────────────────────────────────
        self._emit(f"\n  NOTABLE EVENTS THIS WINDOW:")
        event_count = 0
        for model, s in self.model_state.items():
            for ev in s["events"]:
                self._emit(f"    {ev[1] if len(ev)>1 else '?'}  [{model}]  {ev[0]}  {ev[2] if len(ev)>2 else ''}")
                event_count += 1
        if event_count == 0:
            self._emit("    (none recorded yet)")

        # ── Gap Analysis vs V2.0 target ───────────────────────────────────
        self._emit(f"\n  GAP ANALYSIS vs V2.0 TARGET (Sharpe>1.0, PF>1.3, MDD<20%, |ΔSharpe|≤2.0):")
        passed = [m for m, s in self.model_state.items() if s.get("is_valid") is True]
        failed = [m for m, s in self.model_state.items() if s.get("is_valid") is False]
        running = [m for m, s in self.model_state.items() if s.get("is_valid") is None]
        self._emit(f"    PASSED  ({len(passed)}): {passed or 'none yet'}")
        self._emit(f"    FAILED  ({len(failed)}): {failed or 'none'}")
        self._emit(f"    RUNNING ({len(running)}): {running or 'none'}")

        if self.error_count > 0:
            self._emit(f"    ⚠️  ERROR COUNT: {self.error_count} — CHECK TRAINING LOG FOR TRACEBACKS")

        # ── Next action recommendation ────────────────────────────────────
        self._emit(f"\n  RECOMMENDED NEXT ACTION:")
        if running:
            self._emit(f"    Training in progress for: {running}. Continue monitoring.")
        elif failed and not passed:
            self._emit(
                "    ALL models failed V2.0 gates. Consider:\n"
                "      1. Increase max_rows_per_symbol (more data diversity)\n"
                "      2. Reduce seq_len (reduce overfitting surface)\n"
                "      3. Check BTC beta neutralization is active (btc_frame loaded)\n"
                "      4. Enable use_lupi: true and rerun"
            )
        elif passed:
            self._emit(
                f"    {len(passed)} model(s) PASSED. Run evaluate_all_checkpoints.py for OOS walk-forward validation."
            )
        else:
            self._emit("    Waiting for training to complete or produce first metrics.")

        self._emit(f"{sep}\n")

        # ── Save JSON snapshot ─────────────────────────────────────────────
        snapshot = {
            "heartbeat": self.heartbeat_count,
            "timestamp": _ts(),
            "elapsed_hours": round(elapsed_h, 3),
            "dataset_windows": self.dataset_windows,
            "models": {
                m: {k: v for k, v in s.items() if k != "events"}
                for m, s in self.model_state.items()
            },
            "passed": passed,
            "failed": failed,
            "running": running,
            "error_count": self.error_count,
        }
        self.heartbeat_json.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    def run(self, heartbeat_interval: int = 1800) -> None:
        self.out_log.parent.mkdir(parents=True, exist_ok=True)

        self._log_file_handle = self.out_log.open("a", encoding="utf-8", buffering=1)
        self._emit(f"\n{'='*80}")
        self._emit(f"  GLASS TANK MONITOR STARTED  |  {_ts()}")
        self._emit(f"  Training log : {self.training_log}")
        self._emit(f"  Output log   : {self.out_log}")
        self._emit(f"  Heartbeat    : every {heartbeat_interval//60} minutes")
        self._emit(f"{'='*80}\n")

        # Wait for training log to appear
        wait_s = 0
        while not self.training_log.exists():
            if wait_s % 30 == 0:
                print(f"[monitor] waiting for training log: {self.training_log}", flush=True)
            time.sleep(5)
            wait_s += 5

        with self.training_log.open("r", encoding="utf-8", errors="replace") as f:
            while True:
                line = f.readline()
                if line:
                    self.lines_read += 1
                    self._parse_line(line.rstrip())
                else:
                    # No new data — check if heartbeat due
                    now = time.time()
                    if now - self.last_heartbeat >= heartbeat_interval:
                        self._heartbeat()
                        self.last_heartbeat = now
                    time.sleep(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Glass Tank Training Monitor V2.0")
    ap.add_argument("--log", default="doc/training_v2_glass_tank.log", help="Output glass-tank log path")
    ap.add_argument("--heartbeat", type=int, default=7200, help="Heartbeat interval seconds (default 7200=2h)")
    ap.add_argument("--training-log", default="doc/training_v2.log", help="Path to live training stdout log")
    ap.add_argument("--heartbeat-json", default="doc/training_v2_heartbeat.json", help="JSON snapshot output")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    monitor = GlassTank(
        training_log=root / args.training_log,
        out_log=root / args.log,
        heartbeat_json=root / args.heartbeat_json,
    )
    monitor.run(heartbeat_interval=args.heartbeat)


if __name__ == "__main__":
    main()
