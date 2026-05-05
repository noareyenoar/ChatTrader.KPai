from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end_idx: int
    val_start_idx: int
    val_end_idx: int
    test_start_idx: int


class IronWallSplitter:
    """Strict chronological split with purge gaps and no random shuffling."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_gap_bars: int = 20,
    ) -> None:
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1.0")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.purge_gap_bars = purge_gap_bars

    def split(self, frame: pd.DataFrame, time_col: str = "timestamp") -> SplitResult:
        if time_col not in frame.columns:
            raise ValueError(f"Missing time column: {time_col}")

        ordered = frame.sort_values(time_col).reset_index(drop=True)
        n = len(ordered)
        if n < 10:
            raise ValueError("Not enough rows for chronological split")

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        base_val_start = n_train
        base_val_end = base_val_start + n_val

        train_end = max(0, n_train)
        val_start = min(n, base_val_start + self.purge_gap_bars)
        val_end = min(n, base_val_end)
        test_start = min(n, val_end + self.purge_gap_bars)

        train = ordered.iloc[:train_end].copy()
        val = ordered.iloc[val_start:val_end].copy()
        test = ordered.iloc[test_start:].copy()

        if train.empty or val.empty or test.empty:
            raise ValueError(
                "Split generated empty partition; reduce purge gap or enforce larger history"
            )

        # Enforce strict temporal ordering to prevent any leakage across partitions.
        if train[time_col].max() >= val[time_col].min():
            raise ValueError("Leakage risk detected: train overlaps validation timestamps")
        if val[time_col].max() >= test[time_col].min():
            raise ValueError("Leakage risk detected: validation overlaps test timestamps")

        return SplitResult(
            train=train,
            val=val,
            test=test,
            train_end_idx=train_end - 1,
            val_start_idx=val_start,
            val_end_idx=val_end - 1,
            test_start_idx=test_start,
        )
