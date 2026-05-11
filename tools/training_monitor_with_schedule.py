#!/usr/bin/env python3
"""
Enhanced training monitor with schedule tracking and ETA prediction.

Shows:
1. Live training progress per model
2. Schedule timeline with predicted vs actual completion
3. ETA drift analysis (is training faster/slower than predicted?)
4. Backend speed metrics
5. Per-archetype completion status
"""

import argparse
import os
import re
import statistics
import time
import yaml
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple

LINE_RE = re.compile(r"^\- \[(?P<ts>[^\]]+)\] model=(?P<model>\S+) stage=(?P<stage>\S+)(?P<rest>.*)$")
KV_RE = re.compile(r"(?P<k>[A-Za-z0-9_]+)=(?P<v>[^\s]+)")


def infer_archetype(model: str) -> str:
    if "Trend" in model:
        return "trend"
    if "_MR_" in model:
        return "mean_reversion"
    if "Scalper" in model:
        return "scalper"
    if "StatArb" in model:
        return "stat_arb"
    if "_Disc_" in model:
        return "discretionary"
    if "_MM_" in model:
        return "market_maker"
    return "unknown"


def _to_num(value: str):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_working_log(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LINE_RE.match(raw.strip())
        if not m:
            continue
        entry = {
            "ts": m.group("ts"),
            "model": m.group("model"),
            "stage": m.group("stage"),
        }
        rest = m.group("rest")
        for kv in KV_RE.finditer(rest):
            entry[kv.group("k")] = _to_num(kv.group("v"))

        model = entry["model"]
        latest[model] = entry

    return latest


def load_schedule(path: Path) -> Optional[Dict]:
    """Load schedule YAML file."""
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_progress(entry: dict) -> str:
    epoch = entry.get("epoch")
    if isinstance(epoch, str) and "/" in epoch:
        return epoch
    if isinstance(epoch, (int, float)):
        total = entry.get("total_epochs")
        if isinstance(total, (int, float)):
            pct = int((float(epoch) / float(total)) * 100)
            return f"{int(epoch)}/{int(total)} ({pct}%)"
        return str(int(epoch))
    return "-"


def format_eta_drift(actual_seconds: float, predicted_seconds: float) -> Tuple[str, str]:
    """
    Compare actual elapsed vs predicted total time.
    
    Returns:
        (drift_str: "50% faster" or "30% slower", status: "✓" or "⚠" or "✗")
    """
    if predicted_seconds == 0:
        return "-", "-"
    
    drift = (actual_seconds / predicted_seconds - 1.0) * 100
    
    if drift < -20:
        return f"{-drift:.0f}% FASTER ✓", "AHEAD"
    elif drift > 20:
        return f"{drift:.0f}% SLOWER ✗", "BEHIND"
    else:
        return f"{drift:+.0f}% (on-track)", "ON_TIME"


def render(
    latest: dict[str, dict],
    schedule: Optional[Dict] = None,
    model_registry: Optional[Dict] = None
) -> str:
    """
    Render comprehensive training dashboard with schedule tracking.
    """
    
    lines = []
    lines.append("")
    lines.append("╔" + "═"*138 + "╗")
    lines.append("║ TRAINING PROGRESS MONITOR WITH SCHEDULE TRACKING                                                                  " + " "*28 + "║")
    lines.append("╚" + "═"*138 + "╝")
    
    # Current time
    now = datetime.utcnow()
    lines.append(f"Current Time: {now.isoformat()}")
    lines.append("")
    
    # === LIVE PROGRESS SECTION ===
    lines.append("┌─ LIVE TRAINING PROGRESS " + "─"*114 + "┐")
    lines.append(f"│ {'Archetype':<14} {'Model':<26} {'Stage':<10} {'Progress':<16} {'Backend':<10} {'Speed':<20} {'Elapsed':<10} │")
    lines.append("├" + "─"*138 + "┤")
    
    # Group by archetype
    by_archetype: Dict[str, List] = defaultdict(list)
    for model, entry in sorted(latest.items()):
        arch = infer_archetype(model)
        by_archetype[arch].append((model, entry))
    
    for arch in sorted(by_archetype.keys()):
        for model, entry in by_archetype[arch]:
            backend = str(entry.get("backend", "n/a"))
            stage = str(entry.get("stage", "-"))
            elapsed_s = entry.get("elapsed_s", "-")
            elapsed_label = f"{float(elapsed_s):.0f}s" if isinstance(elapsed_s, (int, float)) else "-"
            
            progress = format_progress(entry)
            
            speed = None
            speed_label = "-"
            if isinstance(entry.get("samples_per_s"), (int, float)):
                speed = float(entry["samples_per_s"])
                speed_label = f"{speed:,.1f} smp/s"
            elif isinstance(entry.get("episodes_per_min"), (int, float)):
                speed = float(entry["episodes_per_min"])
                speed_label = f"{speed:,.1f} ep/m"
            
            model_short = model if len(model) <= 26 else model[:23]+"..."
            lines.append(
                f"│ {arch:<14} {model_short:<26} {stage:<10} {progress:<16} "
                f"{backend:<10} {speed_label:<20} {elapsed_label:<10} │"
            )
    
    lines.append("└" + "─"*138 + "┘")
    lines.append("")
    
    # === SCHEDULE SECTION ===
    if schedule:
        lines.append("┌─ SCHEDULE & ETA TRACKING " + "─"*111 + "┐")
        lines.append(f"│ {'Archetype':<14} {'Status':<12} {'Est. Total':<12} {'Actual':<12} {'Drift':<16} {'Start':<10} {'ETA End':<10} │")
        lines.append("├" + "─"*138 + "┤")
        
        for arch in sorted(schedule.get('archetypes', {}).keys()):
            arch_info = schedule['archetypes'][arch]
            
            # Get actual elapsed time from latest models in this archetype
            actual_elapsed = 0.0
            active_count = 0
            for model, entry in latest.items():
                if infer_archetype(model) == arch:
                    if isinstance(entry.get("elapsed_s"), (int, float)):
                        actual_elapsed = max(actual_elapsed, float(entry["elapsed_s"]))
                    active_count += 1
            
            predicted_total = arch_info['eta_seconds']
            drift_str, drift_status = format_eta_drift(actual_elapsed, predicted_total)
            
            # Parse times
            try:
                start_iso = arch_info['start_time']
                end_iso = arch_info['end_time']
                start_time = start_iso[11:16]  # HH:MM
                end_time = end_iso[11:16]
            except Exception:
                start_time = "-"
                end_time = "-"
            
            actual_label = f"{actual_elapsed/60:.1f}m" if actual_elapsed > 0 else "-"
            pred_label = f"{predicted_total/60:.1f}m"
            
            status_str = "RUNNING" if active_count > 0 else "PENDING"
            
            lines.append(
                f"│ {arch:<14} {status_str:<12} {pred_label:<12} {actual_label:<12} "
                f"{drift_str:<16} {start_time:<10} {end_time:<10} │"
            )
        
        lines.append("└" + "─"*138 + "┘")
        lines.append("")
        
        # === TIMELINE SUMMARY ===
        meta = schedule.get('metadata', {})
        summary = schedule.get('summary', {})
        
        lines.append("┌─ COMPLETION TIMELINE " + "─"*115 + "┐")
        
        start_time_str = meta.get('start_time', 'N/A')[11:19]
        all_pass_time = summary.get('estimated_all_pass_time', 'N/A')
        all_pass_str = all_pass_time[11:19] if isinstance(all_pass_time, str) else 'N/A'
        
        first_pass_time = summary.get('estimated_first_pass_time', 'N/A')
        first_pass_str = first_pass_time[11:19] if isinstance(first_pass_time, str) else 'N/A'
        
        total_sec = summary.get('total_seconds_sequential', 0)
        total_hours = total_sec / 3600
        
        lines.append(f"│ Start Time (UTC):           {start_time_str}")
        lines.append(f"│ Estimated First Pass:       {first_pass_str}   (one model passing all gates)")
        lines.append(f"│ Estimated All-Pass:         {all_pass_str}   (all models passing all gates)")
        lines.append(f"│ Total Sequential Time:      {total_hours:.1f} hours")
        lines.append(f"│ Total Models:               {summary.get('total_models', 0)}")
        lines.append(f"│ Execution Mode:             {'PARALLEL' if meta.get('parallel_execution') else 'SEQUENTIAL'}")
        lines.append("└" + "─"*138 + "┘")
        lines.append("")
    
    # === BACKEND SPEED RECOMMENDATIONS ===
    lines.append("┌─ BACKEND SPEED ANALYSIS " + "─"*113 + "┐")
    
    backend_speed: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for model, e in latest.items():
        arch = infer_archetype(model)
        backend = str(e.get("backend", "n/a"))
        
        if isinstance(e.get("samples_per_s"), (int, float)):
            speed = float(e["samples_per_s"])
            backend_speed[(arch, backend)].append(speed)
        elif isinstance(e.get("episodes_per_min"), (int, float)):
            speed = float(e["episodes_per_min"])
            backend_speed[(arch, backend)].append(speed)
    
    archetype_to_best: Dict[str, Tuple[str, float, int]] = {}
    for (arch, backend), values in backend_speed.items():
        if not values:
            continue
        avg = float(statistics.mean(values))
        n = len(values)
        cur = archetype_to_best.get(arch)
        if cur is None or avg > cur[1]:
            archetype_to_best[arch] = (backend, avg, n)
    
    if not archetype_to_best:
        lines.append("│ No backend metrics yet. Waiting for first epoch/episode measurements...")
    else:
        lines.append(f"│ {'Archetype':<18} {'Recommended Backend':<20} {'Avg Speed':<20} {'Samples':<10} │")
        lines.append("├" + "─"*138 + "┤")
        for arch in sorted(archetype_to_best):
            backend, avg, n = archetype_to_best[arch]
            unit = "samp/s" if arch != "market_maker" else "ep/min"
            lines.append(f"│ {arch:<18} {backend:<20} {avg:>10,.2f} {unit:<8} n={n:<3}")
    
    lines.append("└" + "─"*138 + "┘")
    
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enhanced training monitor with schedule tracking"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(Path("doc") / "training_more_27-4" / "27-04-2026_plan_REVISED_workingLog.md"),
        help="Path to working log markdown file",
    )
    parser.add_argument(
        "--schedule-path",
        type=str,
        default="doc/training_schedule.yaml",
        help="Path to schedule YAML file",
    )
    parser.add_argument(
        "--refresh-sec",
        type=float,
        default=10.0,
        help="Refresh interval in seconds",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render one snapshot and exit",
    )
    
    args = parser.parse_args()
    
    log_path = Path(args.log_path)
    schedule_path = Path(args.schedule_path)
    
    while True:
        latest = parse_working_log(log_path)
        schedule = load_schedule(schedule_path)
        output = render(latest, schedule)
        
        if not args.once:
            os.system("cls" if os.name == "nt" else "clear")
        
        print(output, flush=True)
        
        if args.once:
            return 0
        
        time.sleep(max(0.5, float(args.refresh_sec)))


if __name__ == "__main__":
    raise SystemExit(main())
