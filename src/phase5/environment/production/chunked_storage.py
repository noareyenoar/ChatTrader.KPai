"""
Monthly-chunked, ZSTD-compressed storage layer for the Time Machine Dataset.

Naming convention
-----------------
  Parquet  :  <layer_key>_<YYYY>_<MM:02d>.parquet
  JSONL    :  <layer_key>_<YYYY>_<MM:02d>.jsonl

Both formats use the same directory mapping as the original
``LAYER_DIRS`` schema (e.g. ``on_chain/``, ``derivatives_microstructure/``).

The ``TimeMachineOracle`` auto-discovers all ``*.parquet`` / ``*.jsonl``
files under each layer directory, so the monthly naming is transparent
to consumers.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import pandas as pd

from src.phase5.environment.schemas import ALL_LAYER_KEYS, LAYER_DIRS, PARQUET_LAYERS

log = logging.getLogger("time_machine.storage")

# ZSTD is faster than snappy and achieves better compression on time-series.
# Requires pyarrow >= 1.0 (we have 24.0).
_PARQUET_COMPRESSION = "zstd"
_PARQUET_COMPRESSION_LEVEL = 3   # 1–22; 3 is a good speed/ratio trade-off


def _layer_subdir(data_root: pathlib.Path, layer_key: str) -> pathlib.Path:
    subdir = data_root / LAYER_DIRS[layer_key]
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def _parquet_name(layer_key: str, year: int, month: int) -> str:
    return f"{layer_key}_{year}_{month:02d}.parquet"


def _jsonl_name(layer_key: str, year: int, month: int) -> str:
    return f"{layer_key}_{year}_{month:02d}.jsonl"


class ChunkedLayerStorage:
    """
    Handles monthly-chunked writes and existence checks for all 10 layers.

    Parameters
    ----------
    data_root : str | Path
        Root of the Time Machine dataset tree.
    overwrite : bool
        If False (default) ``save_*`` methods are no-ops when the target
        file already exists (resume-friendly).  Pass ``overwrite=True``
        to force-refresh.
    """

    def __init__(
        self,
        data_root: str | pathlib.Path,
        overwrite: bool = False,
    ) -> None:
        self.root      = pathlib.Path(data_root).resolve()
        self.overwrite = overwrite
        self.root.mkdir(parents=True, exist_ok=True)
        # Ensure all 10 subdirs exist upfront
        for subdir in LAYER_DIRS.values():
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

    # ── Existence checks  (resume logic) ─────────────────────────────────────

    def parquet_exists(self, layer_key: str, year: int, month: int) -> bool:
        path = _layer_subdir(self.root, layer_key) / _parquet_name(layer_key, year, month)
        return path.exists() and path.stat().st_size > 0

    def jsonl_exists(self, layer_key: str, year: int, month: int) -> bool:
        # A 0-byte file is still valid — it means "processed, no data for this month".
        path = _layer_subdir(self.root, layer_key) / _jsonl_name(layer_key, year, month)
        return path.exists()

    def month_exists(self, layer_key: str, year: int, month: int) -> bool:
        """True if this layer already has a file for year/month."""
        if layer_key in PARQUET_LAYERS:
            return self.parquet_exists(layer_key, year, month)
        return self.jsonl_exists(layer_key, year, month)

    # ── Parquet write ─────────────────────────────────────────────────────────

    def save_monthly_parquet(
        self,
        df: pd.DataFrame,
        layer_key: str,
        year: int,
        month: int,
    ) -> pathlib.Path | None:
        """
        Write *df* to ``<layer_dir>/<layer_key>_YYYY_MM.parquet`` with ZSTD
        compression.  Returns the path written, or None if skipped.

        The DataFrame **must** contain a ``timestamp_utc`` column (int64, ms).
        """
        out = _layer_subdir(self.root, layer_key) / _parquet_name(layer_key, year, month)

        if out.exists() and not self.overwrite:
            log.debug("Skip (exists): %s", out)
            return None

        if df.empty:
            log.warning("Empty DataFrame for %s %d-%02d – skipping write.", layer_key, year, month)
            return None

        # Ensure timestamp column is int64
        if "timestamp_utc" in df.columns:
            df = df.copy()
            df["timestamp_utc"] = df["timestamp_utc"].astype("int64")

        df.to_parquet(
            out,
            engine="pyarrow",
            compression=_PARQUET_COMPRESSION,
            compression_level=_PARQUET_COMPRESSION_LEVEL,
            index=False,
        )
        size_kb = out.stat().st_size / 1024
        log.info(
            "Parquet [%s] %d-%02d → %s  (%.1f KB, %d rows)",
            layer_key, year, month, out.name, size_kb, len(df),
        )
        return out

    # ── JSONL write ───────────────────────────────────────────────────────────

    def save_monthly_jsonl(
        self,
        records: list[dict[str, Any]],
        layer_key: str,
        year: int,
        month: int,
        *,
        append: bool = False,
    ) -> pathlib.Path | None:
        """
        Write records to ``<layer_dir>/<layer_key>_YYYY_MM.jsonl``.
        Returns the path, or None if skipped.
        """
        out = _layer_subdir(self.root, layer_key) / _jsonl_name(layer_key, year, month)

        if out.exists() and not self.overwrite and not append:
            log.debug("Skip (exists): %s", out)
            return None

        mode = "a" if append else "w"
        with out.open(mode, encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        size_kb = out.stat().st_size / 1024
        log.info(
            "JSONL [%s] %d-%02d → %s  (%.1f KB, %d records)",
            layer_key, year, month, out.name, size_kb, len(records),
        )
        return out

    # ── Bulk helpers ──────────────────────────────────────────────────────────

    def list_months_written(self, layer_key: str) -> list[tuple[int, int]]:
        """Return sorted list of (year, month) tuples that have been written."""
        layer_dir = _layer_subdir(self.root, layer_key)
        pattern   = "*.parquet" if layer_key in PARQUET_LAYERS else "*.jsonl"
        months: list[tuple[int, int]] = []
        for f in sorted(layer_dir.glob(pattern)):
            # Expect names like  layer_key_2021_01.parquet
            parts = f.stem.rsplit("_", 2)
            if len(parts) >= 3:
                try:
                    months.append((int(parts[-2]), int(parts[-1])))
                except ValueError:
                    pass
        return sorted(months)

    def total_size_mb(self, layer_key: str) -> float:
        layer_dir = _layer_subdir(self.root, layer_key)
        total = sum(f.stat().st_size for f in layer_dir.iterdir() if f.is_file())
        return total / (1024 ** 2)

    # ── Data quality report  (SEPARATE from resume logic) ─────────────────────

    def data_quality_report(self) -> dict:
        """
        Returns a dict with per-layer quality stats.

        This is SEPARATE from jsonl_exists / month_exists, which only check
        file existence (correct for the resume mechanism).  This report is for
        human inspection of data coverage and density.

        Fields per layer:
            files_total    : int  – all files (including 0-byte)
            files_nonempty : int  – files with st_size > 0
            size_bytes     : int  – total bytes
            pct_nonempty   : float
        """
        report = {}
        for key in ALL_LAYER_KEYS:
            layer_dir = self.root / LAYER_DIRS[key]
            if not layer_dir.exists():
                report[key] = {"files_total": 0, "files_nonempty": 0, "size_bytes": 0, "pct_nonempty": 0.0}
                continue
            all_files    = [f for f in layer_dir.iterdir() if f.is_file()]
            nonempty     = [f for f in all_files if f.stat().st_size > 0]
            size_bytes   = sum(f.stat().st_size for f in all_files)
            pct          = 100.0 * len(nonempty) / len(all_files) if all_files else 0.0
            report[key]  = {
                "files_total":    len(all_files),
                "files_nonempty": len(nonempty),
                "size_bytes":     size_bytes,
                "pct_nonempty":   round(pct, 1),
            }
        return report

    def print_quality_report(self) -> None:
        """Print a human-readable quality report to stdout."""
        report = self.data_quality_report()
        total_bytes = sum(v["size_bytes"] for v in report.values())
        print(f"\n{'Layer':<28} {'Files':>6} {'Non-0':>6} {'%Full':>6} {'MB':>8}")
        print("-" * 60)
        for key, stats in sorted(report.items()):
            mb = stats["size_bytes"] / (1024**2)
            print(f"{key:<28} {stats['files_total']:>6} {stats['files_nonempty']:>6} "
                  f"{stats['pct_nonempty']:>5.0f}% {mb:>7.2f}")
        print("-" * 60)
        print(f"{'TOTAL':<28} {'':>6} {'':>6} {'':>6} {total_bytes/(1024**2):>7.2f}")
