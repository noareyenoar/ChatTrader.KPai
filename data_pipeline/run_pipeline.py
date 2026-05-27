from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.config import PipelineConfig
from data_pipeline.features import FeatureFactory
from data_pipeline.gpu_utils import cleanup_cuda
from data_pipeline.quality_gate import DataQualityGate
from data_pipeline.reporting import feature_stats_table, save_histograms
from data_pipeline.splitter import IronWallSplitter


def _load_symbol_frame(dataset_dir: Path, symbol: str) -> pd.DataFrame:
    path = dataset_dir / f"{symbol}.parquet"
    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "open_interest",
        "funding_rate",
    ]
    df = pd.read_parquet(path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _render_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    return df.to_markdown(index=False) + "\n"


def main() -> int:
    config = PipelineConfig()
    gate = DataQualityGate(config)
    records = gate.evaluate()

    accepted = [r for r in records if r.decision == "ACCEPT"]
    rejected = [r for r in records if r.decision == "REJECT"]

    splitter = IronWallSplitter(
        train_ratio=config.split_train_ratio,
        val_ratio=config.split_val_ratio,
        test_ratio=config.split_test_ratio,
        purge_gap_bars=config.purge_gap_bars,
    )

    feature_frames: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []

    selected_symbols = [r.symbol for r in accepted[: config.sample_symbols_for_feature_stats]]
    for symbol in selected_symbols:
        frame = _load_symbol_frame(config.dataset_dir, symbol)

        # Vectorized feature generation for multiple archetype projections.
        trend = FeatureFactory.build_trend_features(frame)
        mean_rev = FeatureFactory.build_mean_reversion_features(frame)
        stat_arb = FeatureFactory.build_stat_arb_features(frame)
        scalper = FeatureFactory.build_scalper_features(frame)
        disc = FeatureFactory.build_discretionary_features(frame)

        feature_df = trend[
            [
                "timestamp",
                "close",
                "log_return",
                "zscore_close_64",
                "ema_spread",
                "atr_14",
                "price_slope_20",
            ]
        ].copy()
        feature_df = feature_df.merge(
            mean_rev[
                [
                    "timestamp",
                    "vwap_dev",
                    "bb_distance",
                    "zscore_close_20",
                    "rsi_14",
                    "rsi_div_5",
                ]
            ],
            on="timestamp",
            how="left",
        ).merge(
            stat_arb[["timestamp", "fracdiff_close_d04", "spread_z_64"]],
            on="timestamp",
            how="left",
        ).merge(
            scalper[["timestamp", "ofi_proxy", "microprice_dev", "vol_imbalance",
                      "volatility_z_32"]].rename(columns=lambda c: c if c == "timestamp" else f"sc_{c}"),
            on="timestamp",
            how="left",
        ).merge(
            disc[["timestamp", "ema_spread", "atr_14", "price_slope_20"]].rename(
                columns={"ema_spread": "disc_ema_spread", "atr_14": "disc_atr_14",
                          "price_slope_20": "disc_price_slope_20"}
            ),
            on="timestamp",
            how="left",
        )

        split = splitter.split(feature_df, time_col="timestamp")

        # Hard-stop if any partition ordering violation appears.
        if split.train["timestamp"].max() >= split.val["timestamp"].min():
            raise RuntimeError(f"Iron Wall violation for {symbol}: train/val overlap")
        if split.val["timestamp"].max() >= split.test["timestamp"].min():
            raise RuntimeError(f"Iron Wall violation for {symbol}: val/test overlap")

        scaled_cols = [
            "log_return",
            "zscore_close_64",
            "ema_spread",
            "atr_14",
            "price_slope_20",
            "vwap_dev",
            "bb_distance",
            "zscore_close_20",
            "rsi_14",
            "rsi_div_5",
            "fracdiff_close_d04",
            "spread_z_64",
            "sc_ofi_proxy",
            "sc_microprice_dev",
            "sc_vol_imbalance",
            "sc_volatility_z_32",
            "disc_ema_spread",
            "disc_atr_14",
            "disc_price_slope_20",
        ]

        scaler = FeatureFactory.fit_scaler_train_only(split.train, scaled_cols)
        train_scaled = FeatureFactory.transform_with_scaler(split.train, scaler)
        val_scaled = FeatureFactory.transform_with_scaler(split.val, scaler)
        test_scaled = FeatureFactory.transform_with_scaler(split.test, scaler)

        train_scaled["symbol"] = symbol
        feature_frames.append(train_scaled)

        split_rows.append(
            {
                "symbol": symbol,
                "train_rows": len(split.train),
                "val_rows": len(split.val),
                "test_rows": len(split.test),
                "purge_gap_bars": config.purge_gap_bars,
                "train_end": split.train["timestamp"].max(),
                "val_start": split.val["timestamp"].min(),
                "val_end": split.val["timestamp"].max(),
                "test_start": split.test["timestamp"].min(),
            }
        )

        cleanup_cuda(frame, trend, mean_rev, stat_arb, scalper, disc, feature_df, train_scaled, val_scaled, test_scaled)

    all_features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()

    feature_columns = [
        "log_return",
        "zscore_close_64",
        "ema_spread",
        "atr_14",
        "price_slope_20",
        "vwap_dev",
        "bb_distance",
        "zscore_close_20",
        "rsi_14",
        "rsi_div_5",
        "fracdiff_close_d04",
        "spread_z_64",
        "sc_ofi_proxy",
        "sc_microprice_dev",
        "sc_vol_imbalance",
        "sc_volatility_z_32",
        "disc_ema_spread",
        "disc_atr_14",
        "disc_price_slope_20",
    ]

    stats = feature_stats_table(all_features, feature_columns) if not all_features.empty else pd.DataFrame()
    hist_paths = save_histograms(all_features, feature_columns, config.artifacts_dir) if not all_features.empty else []

    rejected_df = pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "manifest_status": r.status,
                "rows": r.rows,
                "expected_rows": r.expected_rows,
                "missing_ratio": round(r.missing_ratio, 6),
                "decision": r.decision,
                "reason": r.reason,
            }
            for r in rejected
        ]
    )

    accepted_df = pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "rows": r.rows,
                "expected_rows": r.expected_rows,
                "missing_ratio": round(r.missing_ratio, 6),
                "decision": r.decision,
            }
            for r in accepted
        ]
    )

    split_df = pd.DataFrame(split_rows)

    lines: list[str] = []
    lines.append("# Data Integrity Report (Phase 3 Feature Factory)")
    lines.append("")
    lines.append("Date: 2026-04-25")
    lines.append("")
    lines.append("## 1) Quality Gate Summary")
    lines.append("")
    lines.append(f"- Total symbols evaluated: {len(records)}")
    lines.append(f"- Accepted symbols: {len(accepted)}")
    lines.append(f"- Rejected symbols: {len(rejected)}")
    lines.append(f"- Missing bar threshold: {config.max_missing_ratio:.2%}")
    lines.append(f"- Minimum history bars: {config.min_history_bars}")
    lines.append("")

    lines.append("### Accepted Symbols")
    lines.append("")
    lines.append(_render_markdown_table(accepted_df.head(20)))

    lines.append("### Rejected Symbols")
    lines.append("")
    lines.append(_render_markdown_table(rejected_df.head(40)))

    lines.append("## 2) Iron Wall Split Validation")
    lines.append("")
    lines.append("Rules enforced:")
    lines.append("- Chronological ordering only (no shuffle)")
    lines.append("- 70/15/15 split")
    lines.append(f"- Purge gap = {config.purge_gap_bars} bars between train/val and val/test")
    lines.append("")
    lines.append(_render_markdown_table(split_df))

    lines.append("## 3) Feature Registry")
    lines.append("")
    lines.append("Generated vectorized features:")
    for col in feature_columns:
        lines.append(f"- {col}")
    lines.append("")

    lines.append("### Distribution Statistics (Train Partition, Scaler Fit on Train Only)")
    lines.append("")
    lines.append(_render_markdown_table(stats))

    lines.append("## 4) Distribution Visualizations")
    lines.append("")
    if hist_paths:
        for path in hist_paths:
            rel = path.as_posix()
            lines.append(f"- ![]({rel})")
    else:
        lines.append("- No plots generated (no accepted feature rows).")

    lines.append("")
    lines.append("## 5) Leakage Controls")
    lines.append("")
    lines.append("- Scaler fit operation executed only on train partition.")
    lines.append("- Validation and test partitions transformed using train-fitted scaler stats.")
    lines.append("- Purge gap applied to prevent horizon bleeding across partitions.")

    config.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote report: {config.report_path}")
    print(f"Generated plots: {len(hist_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
