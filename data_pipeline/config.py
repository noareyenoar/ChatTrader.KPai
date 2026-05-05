from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    dataset_dir: Path = Path("Dataset/binance_historical")
    manifest_path: Path = Path("Dataset/binance_historical/manifest.json")
    report_path: Path = Path("data_integrity_report.md")
    artifacts_dir: Path = Path("data_pipeline/reports")
    timeframe_minutes: int = 5
    max_missing_ratio: float = 0.05
    min_history_bars: int = 50_000
    purge_gap_bars: int = 20
    split_train_ratio: float = 0.70
    split_val_ratio: float = 0.15
    split_test_ratio: float = 0.15
    sample_symbols_for_feature_stats: int = 8
