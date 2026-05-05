from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.features import FeatureFactory
from data_pipeline.splitter import IronWallSplitter


def _sample_frame(rows: int = 1_000) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    close = np.linspace(100.0, 200.0, rows)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "quote_volume": close * 1000.0,
            "trades": np.full(rows, 10),
            "taker_buy_base": np.full(rows, 400.0),
            "taker_buy_quote": close * 400.0,
            "open_interest": np.zeros(rows),
            "funding_rate": np.zeros(rows),
        }
    )


def test_iron_wall_split_has_temporal_separation() -> None:
    frame = _sample_frame(2_000)
    splitter = IronWallSplitter(purge_gap_bars=20)
    split = splitter.split(frame)

    assert split.train["timestamp"].max() < split.val["timestamp"].min()
    assert split.val["timestamp"].max() < split.test["timestamp"].min()


def test_iron_wall_split_raises_on_excessive_gap() -> None:
    frame = _sample_frame(50)
    splitter = IronWallSplitter(purge_gap_bars=100)

    try:
        splitter.split(frame)
    except ValueError as exc:
        assert "empty partition" in str(exc).lower()
        return

    raise AssertionError("Expected ValueError for excessive purge gap")


def test_scaler_is_train_only_behavior() -> None:
    frame = _sample_frame(2_000)
    feat = FeatureFactory.build_trend_features(frame)
    splitter = IronWallSplitter(purge_gap_bars=20)
    split = splitter.split(feat)

    cols = ["log_return", "zscore_close_64", "ema_spread", "atr_14", "price_slope_20"]
    scaler = FeatureFactory.fit_scaler_train_only(split.train, cols)

    train_scaled = FeatureFactory.transform_with_scaler(split.train, scaler)
    val_scaled = FeatureFactory.transform_with_scaler(split.val, scaler)

    train_means = np.nanmean(train_scaled.loc[:, cols].to_numpy(dtype=float), axis=0)
    val_means = np.nanmean(val_scaled.loc[:, cols].to_numpy(dtype=float), axis=0)

    assert np.all(np.isfinite(train_means))
    assert np.all(np.isfinite(val_means))
    assert np.all(np.abs(train_means) < 1e-2)
    assert np.any(np.abs(val_means) > 1e-6)
