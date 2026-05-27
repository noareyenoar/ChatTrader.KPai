#!/usr/bin/env python3
"""
Phase 5 – Time Machine Oracle
==============================
The anti-leakage historical-state controller.

Given any UTC timestamp T the oracle returns the *exact and complete*
state of all 10 environmental layers as it would have been known at T –
with zero look-ahead contamination.

Anti-Leakage Contract (The Iron Curtain Rule)
---------------------------------------------
    For every row in every data source the following invariant holds:

        row.timestamp_utc  <=  query_timestamp_ms

    No data point from T+1 millisecond can appear in any query result.
    This is enforced at the pandas filter level for Parquet layers and
    at the list comprehension level for JSONL layers.

Usage
-----
    from src.phase5.environment.time_machine_oracle import TimeMachineOracle

    oracle = TimeMachineOracle("Dataset/phase5_time_machine_dataset")

    # Query state at the exact moment Bitcoin hit $3,782
    snapshot = oracle.query(timestamp_ms=1583971200000 + 14 * 3600 * 1000)

    # Pretty-print a summary for an LLM prompt
    print(oracle.describe(snapshot))

    # Replay the full Black Thursday window hour-by-hour
    for frame in oracle.replay_window(
        start_ms = 1583884800000,  # 2020-03-11 00:00 UTC
        end_ms   = 1584230400000,  # 2020-03-15 00:00 UTC
        step_ms  = 3_600_000,      # 1-hour steps
        layers   = ["derivatives", "fear_greed", "crises_hacks"],
    ):
        process(frame)
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator, Iterator

import pandas as pd

from src.phase5.environment.schemas import (
    ALL_LAYER_KEYS,
    JSONL_LAYERS,
    LAYER_DIRS,
    PARQUET_LAYERS,
)

log = logging.getLogger("time_machine.oracle")

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerSnapshot:
    """Causal snapshot of a single layer at time T."""
    layer_key:     str
    query_ts_ms:   int                        # The T that was queried
    window_ms:     int | None                 # Lookback window (None = all history)
    record_count:  int                        # Rows / events in this snapshot
    data:          pd.DataFrame | list[dict]  # The actual payload
    oldest_ts_ms:  int | None = None          # Oldest timestamp in the payload
    newest_ts_ms:  int | None = None          # Newest timestamp in the payload

    def is_empty(self) -> bool:
        if isinstance(self.data, pd.DataFrame):
            return self.data.empty
        return len(self.data) == 0


@dataclass
class TimeMachineSnapshot:
    """Complete 10-layer causal state for a single timestamp T."""
    query_ts_ms: int
    query_dt:    datetime
    layers:      dict[str, LayerSnapshot] = field(default_factory=dict)

    def __repr__(self) -> str:
        ts_str = self.query_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        parts  = [f"TimeMachineSnapshot @ {ts_str}"]
        for k, snap in self.layers.items():
            parts.append(f"  {k:<28} {snap.record_count:>5} records")
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ms_to_dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1_000.0, tz=timezone.utc)


def _dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000)


# ─────────────────────────────────────────────────────────────────────────────
# TimeMachineOracle
# ─────────────────────────────────────────────────────────────────────────────

class TimeMachineOracle:
    """
    Feeds the exact, causally-clean state of the 10 Time Machine layers to
    trading agents for any requested historical timestamp.

    Parameters
    ----------
    data_root : str | Path
        Path to ``Dataset/phase5_time_machine_dataset``.
    cache : bool
        If True (default) the oracle loads each layer file once and caches
        it in memory.  Set to False for very large datasets where you want
        per-query on-demand loading.
    """

    def __init__(
        self,
        data_root: str | pathlib.Path,
        cache: bool = True,
    ) -> None:
        self._root  = pathlib.Path(data_root).resolve()
        self._cache_enabled = cache
        self._parquet_cache: dict[str, pd.DataFrame] = {}
        self._jsonl_cache:   dict[str, list[dict]]   = {}
        log.info("TimeMachineOracle initialised at %s", self._root)

    # ── Directory helpers ─────────────────────────────────────────────────────

    def _layer_dir(self, layer_key: str) -> pathlib.Path:
        return self._root / LAYER_DIRS[layer_key]

    def _parquet_files(self, layer_key: str) -> list[pathlib.Path]:
        return sorted(self._layer_dir(layer_key).glob("*.parquet"))

    def _jsonl_files(self, layer_key: str) -> list[pathlib.Path]:
        return sorted(self._layer_dir(layer_key).glob("*.jsonl"))

    # ── Data loading (with optional cache) ───────────────────────────────────

    def _load_parquet(self, layer_key: str) -> pd.DataFrame:
        """Load (or retrieve from cache) all Parquet files for a layer."""
        if self._cache_enabled and layer_key in self._parquet_cache:
            return self._parquet_cache[layer_key]

        files = self._parquet_files(layer_key)
        if not files:
            log.warning("No Parquet files found for layer '%s'", layer_key)
            return pd.DataFrame(columns=["timestamp_utc"])

        frames = [pd.read_parquet(f, engine="pyarrow") for f in files]
        df = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc")

        # Enforce int64 on the index column for deterministic comparison
        df["timestamp_utc"] = df["timestamp_utc"].astype("int64")

        if self._cache_enabled:
            self._parquet_cache[layer_key] = df

        return df

    def _load_jsonl(self, layer_key: str) -> list[dict]:
        """Load (or retrieve from cache) all JSONL files for a layer."""
        if self._cache_enabled and layer_key in self._jsonl_cache:
            return self._jsonl_cache[layer_key]

        files  = self._jsonl_files(layer_key)
        events: list[dict] = []
        for f in files:
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))

        if not events:
            log.warning("No JSONL records found for layer '%s'", layer_key)

        # Sort by timestamp ascending
        events.sort(key=lambda e: e.get("timestamp_utc", 0))

        if self._cache_enabled:
            self._jsonl_cache[layer_key] = events

        return events

    # ── Core query  (THE IRON CURTAIN RULE enforced here) ────────────────────

    def _query_parquet_layer(
        self,
        layer_key: str,
        query_ts_ms: int,
        window_ms: int | None,
    ) -> LayerSnapshot:
        """
        Return all Parquet rows where:
            timestamp_utc <= query_ts_ms          ← ANTI-LEAKAGE GATE
            timestamp_utc >= query_ts_ms - window ← optional lookback window
        """
        df = self._load_parquet(layer_key)

        # ── IRON CURTAIN: strictly causal filter ──────────────────────────────
        mask = df["timestamp_utc"] <= query_ts_ms
        if window_ms is not None:
            lower_bound = query_ts_ms - window_ms
            mask = mask & (df["timestamp_utc"] >= lower_bound)

        result = df[mask].copy()

        oldest = int(result["timestamp_utc"].min()) if not result.empty else None
        newest = int(result["timestamp_utc"].max()) if not result.empty else None

        return LayerSnapshot(
            layer_key=layer_key,
            query_ts_ms=query_ts_ms,
            window_ms=window_ms,
            record_count=len(result),
            data=result,
            oldest_ts_ms=oldest,
            newest_ts_ms=newest,
        )

    def _query_jsonl_layer(
        self,
        layer_key: str,
        query_ts_ms: int,
        window_ms: int | None,
    ) -> LayerSnapshot:
        """
        Return all JSONL events where:
            timestamp_utc <= query_ts_ms          ← ANTI-LEAKAGE GATE
            timestamp_utc >= query_ts_ms - window ← optional lookback window
        """
        events = self._load_jsonl(layer_key)

        lower_bound = (query_ts_ms - window_ms) if window_ms is not None else None

        # ── IRON CURTAIN ──────────────────────────────────────────────────────
        filtered = [
            e for e in events
            if e.get("timestamp_utc", 0) <= query_ts_ms
            and (lower_bound is None or e.get("timestamp_utc", 0) >= lower_bound)
        ]

        ts_vals  = [e["timestamp_utc"] for e in filtered if "timestamp_utc" in e]
        oldest   = min(ts_vals) if ts_vals else None
        newest   = max(ts_vals) if ts_vals else None

        return LayerSnapshot(
            layer_key=layer_key,
            query_ts_ms=query_ts_ms,
            window_ms=window_ms,
            record_count=len(filtered),
            data=filtered,
            oldest_ts_ms=oldest,
            newest_ts_ms=newest,
        )

    # ── Public query interface ────────────────────────────────────────────────

    def query(
        self,
        timestamp_ms: int,
        layers: list[str] | None = None,
        window_ms: int | None = None,
    ) -> TimeMachineSnapshot:
        """
        Return the complete causal state of the Time Machine at ``timestamp_ms``.

        Parameters
        ----------
        timestamp_ms : int
            Unix epoch in milliseconds (UTC).  The oracle enforces a strict
            ``<= timestamp_ms`` filter on all returned data.
        layers : list[str] | None
            Subset of layer keys to query.  ``None`` queries all 10 layers.
        window_ms : int | None
            Optional lookback window in milliseconds.  If supplied, only
            returns rows within ``[timestamp_ms - window_ms, timestamp_ms]``.
            Example: ``window_ms = 86_400_000`` gives the last 24 hours.

        Returns
        -------
        TimeMachineSnapshot
        """
        if layers is None:
            layers = list(ALL_LAYER_KEYS)

        unknown = set(layers) - set(ALL_LAYER_KEYS)
        if unknown:
            raise ValueError(f"Unknown layer keys: {unknown}.  Valid keys: {list(ALL_LAYER_KEYS)}")

        snapshot = TimeMachineSnapshot(
            query_ts_ms=timestamp_ms,
            query_dt=_ms_to_dt(timestamp_ms),
        )

        for key in layers:
            if key in PARQUET_LAYERS:
                snap = self._query_parquet_layer(key, timestamp_ms, window_ms)
            else:
                snap = self._query_jsonl_layer(key, timestamp_ms, window_ms)
            snapshot.layers[key] = snap

        return snapshot

    def query_latest(
        self,
        timestamp_ms: int,
        layers: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Convenience method: return only the **single most-recent** record for
        each requested layer (e.g. the last row in a Parquet timeseries).
        Useful for feeding point-in-time context to an LLM without overwhelming
        it with full history.
        """
        full = self.query(timestamp_ms, layers=layers)
        result: dict[str, Any] = {}

        for key, snap in full.layers.items():
            if snap.is_empty():
                result[key] = None
            elif isinstance(snap.data, pd.DataFrame):
                latest_row = (
                    snap.data
                    .sort_values("timestamp_utc")
                    .iloc[-1]
                    .to_dict()
                )
                result[key] = latest_row
            else:
                result[key] = snap.data[-1] if snap.data else None

        return result

    # ── Window replay iterator ────────────────────────────────────────────────

    def replay_window(
        self,
        start_ms: int,
        end_ms: int,
        step_ms: int = 3_600_000,
        layers: list[str] | None = None,
        window_ms: int | None = None,
    ) -> Generator[TimeMachineSnapshot, None, None]:
        """
        Iterate through time in fixed steps, yielding a TimeMachineSnapshot
        at each step.  The anti-leakage guarantee is maintained at every step.

        Parameters
        ----------
        start_ms, end_ms : int
            Unix epoch milliseconds defining the replay window.
        step_ms : int
            Step size in milliseconds (default: 3,600,000 = 1 hour).
        layers : list[str] | None
            Layer subset to query at each step.
        window_ms : int | None
            Lookback window passed to each underlying query.

        Yields
        ------
        TimeMachineSnapshot
        """
        t = start_ms
        while t <= end_ms:
            yield self.query(t, layers=layers, window_ms=window_ms)
            t += step_ms

    # ── Cache management ──────────────────────────────────────────────────────

    def warm_cache(self, layers: list[str] | None = None) -> None:
        """
        Pre-load all layer data into memory.  Call once before a replay loop
        to avoid repeated disk I/O.
        """
        keys = layers if layers is not None else list(ALL_LAYER_KEYS)
        for key in keys:
            if key in PARQUET_LAYERS:
                self._load_parquet(key)
            else:
                self._load_jsonl(key)
        log.info("Cache warmed for layers: %s", keys)

    def invalidate_cache(self, layer: str | None = None) -> None:
        """Clear cached data for a specific layer or all layers."""
        if layer is None:
            self._parquet_cache.clear()
            self._jsonl_cache.clear()
        else:
            self._parquet_cache.pop(layer, None)
            self._jsonl_cache.pop(layer, None)

    # ── Human-readable summary ────────────────────────────────────────────────

    def describe(
        self,
        snapshot: TimeMachineSnapshot,
        max_events: int = 5,
    ) -> str:
        """
        Produce a concise, LLM-ready text summary of a TimeMachineSnapshot.
        Suitable for injecting directly into an agent system prompt or RAG context.

        Parameters
        ----------
        snapshot : TimeMachineSnapshot
        max_events : int
            Maximum number of JSONL event records to include per layer.
        """
        lines: list[str] = []
        dt_str = snapshot.query_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        lines.append(f"=== TIME MACHINE SNAPSHOT @ {dt_str} ===")
        lines.append(f"(All data reflects information available at or before {dt_str}.)")
        lines.append("")

        for key, snap in snapshot.layers.items():
            lines.append(f"── LAYER: {key.upper()} ──")
            if snap.is_empty():
                lines.append("  [no data available up to this timestamp]")
                lines.append("")
                continue

            if isinstance(snap.data, pd.DataFrame):
                # For Parquet layers: show the most recent row as key-value pairs
                latest = snap.data.sort_values("timestamp_utc").iloc[-1].to_dict()
                lines.append(f"  Latest record  ({snap.record_count} rows available):")
                for col, val in latest.items():
                    if col == "timestamp_utc":
                        val = _ms_to_dt(int(val)).strftime("%Y-%m-%d %H:%M UTC")
                    if isinstance(val, float) and (val != val):  # NaN check
                        continue
                    lines.append(f"    {col:<32}: {val}")
            else:
                # For JSONL layers: show the most recent N events
                recent = snap.data[-max_events:] if len(snap.data) > max_events else snap.data
                lines.append(f"  {snap.record_count} events up to this timestamp (showing last {len(recent)}):")
                for ev in recent:
                    ts_str = _ms_to_dt(ev.get("timestamp_utc", 0)).strftime("%Y-%m-%d %H:%M UTC")
                    title  = ev.get("title") or ev.get("headline") or ev.get("content", "")[:80]
                    score  = ev.get("impact_score") or ev.get("sentiment_score") or ev.get("sentiment")
                    lines.append(f"    [{ts_str}]  {title}  (score: {score})")

            lines.append("")

        return "\n".join(lines)

    # ── Named scenario convenience ────────────────────────────────────────────

    def get_black_thursday_crash_moment(self) -> TimeMachineSnapshot:
        """
        Return the snapshot at the exact Bitcoin bottom of Black Thursday:
        2020-03-12 14:00 UTC (BTC ~$3,782, BitMEX just resumed).
        All 10 layers are included.
        """
        # 2020-03-12 14:00 UTC
        crash_ms = int(
            datetime(2020, 3, 12, 14, 0, 0, tzinfo=timezone.utc).timestamp() * 1_000
        )
        return self.query(
            timestamp_ms=crash_ms,
            window_ms=3 * 24 * 3_600_000,  # 3-day lookback
        )

    # ── Integrity validation ──────────────────────────────────────────────────

    def validate_anti_leakage(self, query_ts_ms: int, layer_key: str) -> bool:
        """
        Assert that every record in a layer snapshot strictly obeys the
        causal ordering constraint.  Returns True if no violation found.

        Raises ValueError on the first violation found.
        """
        if layer_key in PARQUET_LAYERS:
            snap = self._query_parquet_layer(layer_key, query_ts_ms, window_ms=None)
            if isinstance(snap.data, pd.DataFrame) and not snap.data.empty:
                violations = snap.data[snap.data["timestamp_utc"] > query_ts_ms]
                if not violations.empty:
                    raise ValueError(
                        f"ANTI-LEAKAGE VIOLATION in layer '{layer_key}': "
                        f"{len(violations)} rows have timestamp_utc > query_ts_ms "
                        f"({query_ts_ms}).  Offending timestamps: "
                        f"{violations['timestamp_utc'].tolist()[:5]}"
                    )
        else:
            snap = self._query_jsonl_layer(layer_key, query_ts_ms, window_ms=None)
            violations = [
                e for e in snap.data
                if e.get("timestamp_utc", 0) > query_ts_ms
            ]
            if violations:
                raise ValueError(
                    f"ANTI-LEAKAGE VIOLATION in layer '{layer_key}': "
                    f"{len(violations)} events have timestamp_utc > query_ts_ms."
                )

        return True

    def full_integrity_check(self, query_ts_ms: int) -> dict[str, bool]:
        """Run validate_anti_leakage for all 10 layers.  Returns pass/fail per layer."""
        results: dict[str, bool] = {}
        for key in ALL_LAYER_KEYS:
            try:
                results[key] = self.validate_anti_leakage(query_ts_ms, key)
            except ValueError as exc:
                log.error("Integrity check FAILED for layer '%s': %s", key, exc)
                results[key] = False
        return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI  –  demo / smoke test
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Time Machine Oracle – demo query / integrity check"
    )
    p.add_argument("--data-root", default="Dataset/phase5_time_machine_dataset",
                   help="Root directory of the Time Machine dataset")
    p.add_argument("--timestamp", default="2020-03-12T14:00:00Z",
                   help="ISO-8601 UTC timestamp to query  (default: BTC crash bottom)")
    p.add_argument("--layers", nargs="*",
                   help="Layer keys to query (default: all 10)")
    p.add_argument("--window-hours", type=int, default=72,
                   help="Lookback window in hours (default: 72)")
    p.add_argument("--validate", action="store_true",
                   help="Run full anti-leakage integrity check")
    p.add_argument("--replay", action="store_true",
                   help="Replay the full Black Thursday window and print step count")
    args = p.parse_args()

    oracle = TimeMachineOracle(args.data_root)

    # Parse timestamp
    ts_dt  = datetime.strptime(args.timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    ts_ms  = int(ts_dt.timestamp() * 1_000)
    win_ms = args.window_hours * 3_600_000

    print(f"\nQuerying Time Machine at: {args.timestamp}")
    print(f"Lookback window: {args.window_hours} hours\n")

    # Warm cache
    oracle.warm_cache(args.layers)

    # Query
    snapshot = oracle.query(ts_ms, layers=args.layers, window_ms=win_ms)
    print(repr(snapshot))
    print()
    print(oracle.describe(snapshot, max_events=3))

    # Integrity check
    if args.validate:
        print("\n── Anti-Leakage Integrity Check ──")
        results = oracle.full_integrity_check(ts_ms)
        for layer, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {layer:<28}  {status}")

    # Replay
    if args.replay:
        start_ms = int(datetime(2020, 3, 11, tzinfo=timezone.utc).timestamp() * 1_000)
        end_ms   = int(datetime(2020, 3, 15, 23, 59, tzinfo=timezone.utc).timestamp() * 1_000)
        steps    = 0
        for frame in oracle.replay_window(start_ms, end_ms, step_ms=3_600_000,
                                          layers=["derivatives", "fear_greed", "crises_hacks"],
                                          window_ms=3_600_000):
            steps += 1
        print(f"\nReplay complete: {steps} hourly steps traversed over Black Thursday window.")


if __name__ == "__main__":
    main()
