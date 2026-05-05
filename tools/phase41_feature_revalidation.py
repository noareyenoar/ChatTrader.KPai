from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_pipeline.features import FeatureFactory
from quant_core.scalper_data import SCALPER_FEATURES, _build_scalper_features
from quant_core.discretionary_data import DISC_TAB_FEATURES

DATASET_DIR = ROOT / "Dataset" / "binance_historical"
OUT_PATH = ROOT / "doc" / "training_more_27-4" / "phase41_feature_revalidation.json"
MAX_SYMBOLS = 8
MAX_ROWS_PER_SYMBOL = 100_000


def _load_symbol(path: Path) -> pd.DataFrame:
    cols = [
        "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote",
    ]
    df = pd.read_parquet(path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _bottom_k(features: list[str], importances: dict[str, float], ratio: float = 0.30) -> list[str]:
    k = max(1, math.ceil(len(features) * ratio))
    ordered = sorted(features, key=lambda f: float(importances.get(f, 0.0)))
    return ordered[:k]


def _corr_report(df: pd.DataFrame, cols: list[str]) -> dict[str, object]:
    x = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"rows": 0, "max_abs_corr": None, "high_corr_pairs": []}
    corr = x.corr().abs()
    pairs: list[tuple[str, str, float]] = []
    for i, c1 in enumerate(cols):
        for j in range(i + 1, len(cols)):
            c2 = cols[j]
            v = float(corr.loc[c1, c2])
            if v >= 0.90:
                pairs.append((c1, c2, round(v, 6)))
    max_corr = float(corr.where(~np.eye(len(cols), dtype=bool)).max().max())
    return {
        "rows": int(len(x)),
        "max_abs_corr": round(max_corr, 6) if not np.isnan(max_corr) else None,
        "high_corr_pairs": pairs,
    }


def _rf_importance(df: pd.DataFrame, cols: list[str], target: str) -> dict[str, float]:
    x = df[cols].replace([np.inf, -np.inf], np.nan)
    y = df[target]
    m = pd.concat([x, y], axis=1).dropna()
    if m.empty:
        return {c: 0.0 for c in cols}
    x_np = m[cols].to_numpy(dtype=np.float32)
    y_np = m[target].to_numpy()

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if y_np.dtype.kind in ("i", "u", "b"):
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(x_np, y_np)
        imp = rf.feature_importances_.astype(float)
        return {c: float(v) for c, v in zip(cols, imp)}
    except Exception:
        # Fallback: absolute correlation with target when sklearn is unavailable.
        out: dict[str, float] = {}
        y_ser = pd.Series(y_np)
        for c in cols:
            out[c] = float(abs(pd.Series(x_np[:, cols.index(c)]).corr(y_ser)) or 0.0)
        return out


def main() -> int:
    files = sorted(DATASET_DIR.glob("*.parquet"))[:MAX_SYMBOLS]
    if not files:
        raise RuntimeError("No parquet files found")

    scalper_frames = []
    disc_frames = []
    for f in files:
        raw = _load_symbol(f)
        if len(raw) > MAX_ROWS_PER_SYMBOL:
            raw = raw.iloc[:MAX_ROWS_PER_SYMBOL].copy()

        # Scalper features + 3-class target
        sf = _build_scalper_features(raw)
        future_ret = sf["close"].shift(-5) / sf["close"] - 1.0
        sf["target_cls"] = np.where(future_ret > 0.0003, 2, np.where(future_ret < -0.0003, 0, 1)).astype(np.int64)
        sf = sf[[*SCALPER_FEATURES, "target_cls"]].replace([np.inf, -np.inf], np.nan).dropna()
        scalper_frames.append(sf)

        # Discretionary tab features + 3-class target
        df = FeatureFactory.build_trend_features(raw)
        future_ret2 = df["close"].shift(-5) / df["close"] - 1.0
        df["target_cls"] = np.where(future_ret2 > 0.003, 2, np.where(future_ret2 < -0.003, 0, 1)).astype(np.int64)
        df = df[[*DISC_TAB_FEATURES, "target_cls"]].replace([np.inf, -np.inf], np.nan).dropna()
        disc_frames.append(df)

    scalper_df = pd.concat(scalper_frames, axis=0, ignore_index=True)
    disc_df = pd.concat(disc_frames, axis=0, ignore_index=True)

    scalper_corr = _corr_report(scalper_df, list(SCALPER_FEATURES))
    disc_corr = _corr_report(disc_df, list(DISC_TAB_FEATURES))

    scalper_imp = _rf_importance(scalper_df, list(SCALPER_FEATURES), "target_cls")
    disc_imp = _rf_importance(disc_df, list(DISC_TAB_FEATURES), "target_cls")

    report = {
        "symbols_used": [f.stem for f in files],
        "limits": {
            "max_symbols": MAX_SYMBOLS,
            "max_rows_per_symbol": MAX_ROWS_PER_SYMBOL,
        },
        "scalper": {
            "rows": int(len(scalper_df)),
            "correlation": scalper_corr,
            "importance": scalper_imp,
            "prune_bottom_30pct": _bottom_k(list(SCALPER_FEATURES), scalper_imp, 0.30),
        },
        "discretionary_tab": {
            "rows": int(len(disc_df)),
            "correlation": disc_corr,
            "importance": disc_imp,
            "prune_bottom_30pct": _bottom_k(list(DISC_TAB_FEATURES), disc_imp, 0.30),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[phase41] feature revalidation written: {OUT_PATH}")
    print("[phase41] scalper prune_bottom_30pct:", report["scalper"]["prune_bottom_30pct"])
    print("[phase41] discretionary prune_bottom_30pct:", report["discretionary_tab"]["prune_bottom_30pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
