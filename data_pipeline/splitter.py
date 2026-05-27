from __future__ import annotations

from dataclasses import dataclass
from typing import List

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


# ─────────────────────────────────────────────────────────────────────────────
# V2.0: PURGED ROLLING WALK-FORWARD SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WalkForwardFold:
    """A single (train, test) fold from a purged walk-forward split."""

    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int


class PurgedWalkForwardSplitter:
    """Rolling walk-forward splitter with a strict purge gap between windows.

    Implements Section 2.3 of pytorch_model_training_ruleV2.md:
    - ``n_folds`` rolling (train, test) windows are produced.
    - A ``purge_gap_bars`` equal to the prediction horizon separates the last
      training bar from the first test bar, eliminating any lookahead bleed.
    - Each fold's training window expands (walk-forward) using all history up
      to ``train_end``.

    Example with n=1000, n_folds=5, purge_gap_bars=20, fold_size=167:
      fold 0: train=[:167],  test=[187:354]
      fold 1: train=[:334],  test=[354:521]
      fold 2: train=[:501],  test=[521:688]
      fold 3: train=[:668],  test=[688:855]
      fold 4: train=[:835],  test=[855:1000]
    """

    def __init__(
        self,
        n_folds: int = 5,
        purge_gap_bars: int = 20,
        min_train_bars: int = 500,
    ) -> None:
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")
        if purge_gap_bars < 0:
            raise ValueError("purge_gap_bars must be >= 0")
        self.n_folds = n_folds
        self.purge_gap_bars = purge_gap_bars
        self.min_train_bars = min_train_bars

    def split(self, frame: pd.DataFrame, time_col: str = "timestamp") -> List[WalkForwardFold]:
        if time_col not in frame.columns:
            raise ValueError(f"Missing time column: {time_col}")

        ordered = frame.sort_values(time_col).reset_index(drop=True)
        n = len(ordered)
        if n < self.min_train_bars + self.purge_gap_bars + 2:
            raise ValueError(
                f"Insufficient rows ({n}) for walk-forward split with "
                f"min_train_bars={self.min_train_bars}, purge_gap_bars={self.purge_gap_bars}"
            )

        # Divide timeline into (n_folds + 1) roughly equal segments.
        # First segment is the minimum training base; remaining segments are
        # alternately extended as train and used as test windows.
        usable = n
        fold_size = max(1, usable // (self.n_folds + 1))

        folds: List[WalkForwardFold] = []
        for fold_id in range(self.n_folds):
            train_end = min(n, (fold_id + 1) * fold_size)
            test_start = min(n, train_end + self.purge_gap_bars)
            test_end = min(n, test_start + fold_size)

            if train_end < self.min_train_bars:
                continue
            if test_start >= n or test_end <= test_start:
                break

            train = ordered.iloc[:train_end].copy()
            test = ordered.iloc[test_start:test_end].copy()

            if train.empty or test.empty:
                continue

            # Enforce strict temporal ordering — no leakage across purge gap.
            if train[time_col].max() >= test[time_col].min():
                raise ValueError(
                    f"[WalkForward fold={fold_id}] Leakage: train overlaps test "
                    f"after purge gap of {self.purge_gap_bars} bars"
                )

            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train=train,
                    test=test,
                    train_end_idx=train_end - 1,
                    test_start_idx=test_start,
                    test_end_idx=test_end - 1,
                )
            )

        if not folds:
            raise ValueError(
                "PurgedWalkForwardSplitter produced no valid folds. "
                "Reduce n_folds, min_train_bars, or purge_gap_bars."
            )

        return folds
