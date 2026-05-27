#!/usr/bin/env python3
"""Gate watcher for GAT_StatArb_v1 Profit Factor.

Usage:
  python tools/watch_gat_pf_gate.py --registry model_registry.json --day-limit 3 --started-at 2026-05-15T00:00:00

Exit codes:
  0 -> PF gate reached or deadline not reached yet
  2 -> Deadline reached and PF gate still not met (pivot required)
  1 -> Operational error (missing file / malformed data)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch GAT PF gate and trigger pivot decision at deadline")
    p.add_argument("--registry", type=str, default="model_registry.json")
    p.add_argument("--model-name", type=str, default="GAT_StatArb_v1")
    p.add_argument("--pf-gate", type=float, default=1.5)
    p.add_argument("--day-limit", type=float, default=3.0)
    p.add_argument(
        "--started-at",
        type=str,
        required=True,
        help="UTC ISO timestamp for retraining start, e.g. 2026-05-15T00:00:00",
    )
    return p.parse_args()


def _to_utc_naive(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"[watch-gat] registry missing: {registry_path}")
        return 1

    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[watch-gat] failed to parse registry: {exc}")
        return 1

    row = next((r for r in data if r.get("architecture_name") == args.model_name), None)
    if row is None:
        print(f"[watch-gat] model not found: {args.model_name}")
        return 1

    val = row.get("validation", {})
    pf = val.get("profit_factor")
    ts = row.get("eval_timestamp")

    started_at = _to_utc_naive(args.started_at)
    now_utc = datetime.now(timezone.utc)
    elapsed_days = (now_utc - started_at).total_seconds() / 86400.0

    if pf is None:
        print(
            f"[watch-gat] {args.model_name} has no PF metric yet. "
            f"elapsed_days={elapsed_days:.2f}/{args.day_limit:.2f}"
        )
        if elapsed_days >= args.day_limit:
            print("[watch-gat] PIVOT_REQUIRED: Day limit reached without PF metric.")
            return 2
        return 0

    print(
        f"[watch-gat] model={args.model_name} eval_timestamp={ts} "
        f"pf={float(pf):.6f} gate={args.pf_gate:.6f} "
        f"elapsed_days={elapsed_days:.2f}/{args.day_limit:.2f}"
    )

    if float(pf) >= args.pf_gate:
        print("[watch-gat] PASS: PF gate achieved.")
        return 0

    if elapsed_days >= args.day_limit:
        print("[watch-gat] PIVOT_REQUIRED: PF gate missed by Day 3.")
        return 2

    print("[watch-gat] HOLD: Continue current retraining approach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
