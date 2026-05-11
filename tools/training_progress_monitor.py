#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

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


def format_progress(entry: dict) -> str:
    epoch = entry.get("epoch")
    if isinstance(epoch, str) and "/" in epoch:
        return epoch
    if isinstance(epoch, (int, float)):
        total = entry.get("total_epochs")
        if isinstance(total, (int, float)):
            return f"{int(epoch)}/{int(total)}"
        return str(int(epoch))
    return "-"


def render(latest: dict[str, dict]) -> str:
    rows = []
    backend_speed: dict[tuple[str, str], list[float]] = defaultdict(list)

    for model, e in sorted(latest.items()):
        archetype = infer_archetype(model)
        backend = str(e.get("backend", "n/a"))
        elapsed_s = e.get("elapsed_s", "-")

        speed = None
        speed_label = "-"
        if isinstance(e.get("samples_per_s"), (int, float)):
            speed = float(e["samples_per_s"])
            speed_label = f"{speed:,.1f} samp/s"
        elif isinstance(e.get("episodes_per_min"), (int, float)):
            speed = float(e["episodes_per_min"])
            speed_label = f"{speed:,.2f} ep/min"

        if speed is not None and backend != "n/a":
            backend_speed[(archetype, backend)].append(speed)

        elapsed_label = f"{float(elapsed_s):.1f}s" if isinstance(elapsed_s, (int, float)) else "-"
        rows.append(
            (
                archetype,
                model,
                str(e.get("stage", "-")),
                format_progress(e),
                backend,
                elapsed_label,
                speed_label,
                e.get("ts", "-"),
            )
        )

    lines = []
    lines.append("Training Progress Monitor")
    lines.append("=" * 110)
    lines.append("archetype       model                     stage      progress   backend    elapsed   speed            last_update")
    lines.append("-" * 110)
    for r in rows[-30:]:
        lines.append(f"{r[0]:<14} {r[1]:<25} {r[2]:<10} {r[3]:<10} {r[4]:<10} {r[5]:<9} {r[6]:<15} {r[7]}")

    lines.append("")
    lines.append("Backend Speed Recommendation")
    lines.append("-" * 110)

    archetype_to_best: dict[str, tuple[str, float, int]] = {}
    for (arch, backend), values in backend_speed.items():
        if not values:
            continue
        avg = float(statistics.mean(values))
        n = len(values)
        cur = archetype_to_best.get(arch)
        if cur is None or avg > cur[1]:
            archetype_to_best[arch] = (backend, avg, n)

    if not archetype_to_best:
        lines.append("No backend speed metrics yet. Waiting for first epoch/episode metrics...")
    else:
        for arch in sorted(archetype_to_best):
            backend, avg, n = archetype_to_best[arch]
            unit = "samp/s" if arch != "market_maker" else "ep/min"
            lines.append(f"{arch:<14} -> best={backend:<10} avg={avg:,.2f} {unit} (n={n})")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live training progress + backend speed monitor")
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(Path("doc") / "training_more_27-4" / "27-04-2026_plan_REVISED_workingLog.md"),
        help="Path to append_working_log markdown file",
    )
    parser.add_argument("--refresh-sec", type=float, default=10.0, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Render one snapshot and exit")
    args = parser.parse_args()

    log_path = Path(args.log_path)

    while True:
        latest = parse_working_log(log_path)
        output = render(latest)
        if not args.once:
            os.system("cls" if os.name == "nt" else "clear")
        print(output, flush=True)
        if args.once:
            return 0
        time.sleep(max(0.5, float(args.refresh_sec)))


if __name__ == "__main__":
    raise SystemExit(main())
