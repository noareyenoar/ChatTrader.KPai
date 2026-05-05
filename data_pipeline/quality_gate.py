from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import PipelineConfig


@dataclass(frozen=True)
class SymbolQualityRecord:
    symbol: str
    status: str
    rows: int
    expected_rows: int
    missing_ratio: float
    decision: str
    reason: str
    parquet_path: str


class DataQualityGate:
    """Filter symbols with strict anti-leakage and quality constraints."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._manifest = self._load_manifest(config.manifest_path)

    @staticmethod
    def _load_manifest(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _manifest_status_by_symbol(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for item in self._manifest.get("symbols", []):
            symbol = str(item.get("symbol", "")).strip()
            status = str(item.get("status", "UNKNOWN")).upper()
            if symbol:
                mapping[symbol] = status
        return mapping

    def _parquet_files(self) -> Iterable[Path]:
        return sorted(self.config.dataset_dir.glob("*.parquet"))

    def _expected_rows(self, ts: pd.Series) -> int:
        ts = pd.to_datetime(ts, utc=True, errors="coerce").dropna()
        if ts.empty:
            return 0
        delta = ts.max() - ts.min()
        step = pd.Timedelta(minutes=self.config.timeframe_minutes)
        return int(delta // step) + 1

    def evaluate(self) -> list[SymbolQualityRecord]:
        status_map = self._manifest_status_by_symbol()
        records: list[SymbolQualityRecord] = []

        for pq in self._parquet_files():
            symbol = pq.stem
            manifest_status = status_map.get(symbol, "UNKNOWN")

            try:
                frame = pd.read_parquet(pq, columns=["timestamp"])
                rows = int(len(frame))
                expected_rows = self._expected_rows(frame["timestamp"])
                missing_ratio = 0.0
                if expected_rows > 0:
                    missing_ratio = max(0.0, 1.0 - (rows / expected_rows))

                decision = "ACCEPT"
                reason = "PASS"

                if manifest_status == "FAIL":
                    decision = "REJECT"
                    reason = "MANIFEST_FAIL"
                elif missing_ratio > self.config.max_missing_ratio:
                    decision = "REJECT"
                    reason = "DATA_QUALITY_FAILURE_MISSING_BAR_RATIO"
                elif rows < self.config.min_history_bars:
                    decision = "REJECT"
                    reason = "INSUFFICIENT_HISTORY"

                records.append(
                    SymbolQualityRecord(
                        symbol=symbol,
                        status=manifest_status,
                        rows=rows,
                        expected_rows=expected_rows,
                        missing_ratio=missing_ratio,
                        decision=decision,
                        reason=reason,
                        parquet_path=str(pq).replace('\\\\', '/'),
                    )
                )
            except Exception as exc:
                records.append(
                    SymbolQualityRecord(
                        symbol=symbol,
                        status=manifest_status,
                        rows=0,
                        expected_rows=0,
                        missing_ratio=1.0,
                        decision="REJECT",
                        reason=f"READ_ERROR:{exc}",
                        parquet_path=str(pq).replace('\\\\', '/'),
                    )
                )

        return sorted(records, key=lambda r: (r.decision, r.symbol))

    @staticmethod
    def accepted_symbols(records: list[SymbolQualityRecord]) -> list[str]:
        return [r.symbol for r in records if r.decision == "ACCEPT"]
