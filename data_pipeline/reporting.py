from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def feature_stats_table(frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in feature_columns if c in frame.columns]
    numeric = frame.loc[:, cols].select_dtypes(include=["number"])
    stats = pd.DataFrame(
        {
            "mean": numeric.mean(skipna=True),
            "std": numeric.std(skipna=True),
            "skewness": numeric.skew(skipna=True),
            "non_null": numeric.notna().sum(),
        }
    )
    stats.index.name = "feature"
    return stats.reset_index()


def save_histograms(frame: pd.DataFrame, feature_columns: Iterable[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for col in feature_columns:
        if col not in frame.columns:
            continue
        series = frame[col].dropna()
        if series.empty:
            continue
        path = out_dir / f"{col}_hist.png"
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        ax.hist(series, bins=50)
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    return paths
