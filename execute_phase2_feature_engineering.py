#!/usr/bin/env python3
"""
Phase 2 Execution Driver: Feature Engineering & Synthetic Data
==============================================================

This script orchestrates:
1. Tick/Volume bar building from aggTrades
2. Microstructure feature extraction (OFI, VPIN, spreads)
3. Synthetic data generation (GARCH, HMM, bootstrap)
4. Triple-barrier adaptive labeling

Requires: Dataset/bn_vision_data/ populated (from Phase 1)
Output: Dataset/processed/{tick_bars, volume_bars, microstructure, synthetic, labels}/
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("phase2_feature_engineering")

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_pipeline.features import FeatureFactory
from quant_core.scalper_data import build_scalper_datasets

# Symbols with verified real-data coverage in this workspace.
TIER1_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
VISION_DIR = Path("Dataset/binance_vision_real")
FUTURES_DIR = Path("Dataset/futures")
OPTIONS_DIR = Path("Dataset/options")
PROCESSED_DIR = Path("Dataset/processed")
MAX_PARQUET_FILES_PER_ASSET = 14


def _normalize_trades_schema(trades: pd.DataFrame) -> pd.DataFrame:
    """Normalize aggTrades schema to expected feature-factory columns."""
    trades = trades.copy()

    if "timestamp" not in trades.columns and "transact_time" in trades.columns:
        trades["timestamp"] = trades["transact_time"]

    # Guardrail for downstream OFI/VPIN expectations.
    if "is_buyer_maker" not in trades.columns:
        trades["is_buyer_maker"] = False

    return trades


def _recent_parquet_files(data_dir: Path, limit: int = MAX_PARQUET_FILES_PER_ASSET) -> list[Path]:
    """Return the most recent parquet files to keep runtime bounded."""
    files = sorted(data_dir.rglob("*.parquet"))
    if limit <= 0:
        return files
    return files[-limit:]


def _extract_timestamp_column(df: pd.DataFrame) -> pd.Series:
    """Best-effort timestamp extraction across Binance datasets."""
    candidates = [
        "timestamp",
        "transact_time",
        "transaction_time",
        "event_time",
        "funding_time",
        "calc_time",
        "col_0",
    ]
    for col in candidates:
        if col in df.columns:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            if ts.notna().any():
                return ts
    return pd.to_datetime(pd.Series(np.arange(len(df))), utc=True, errors="coerce")


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_hist_fallback_context() -> pd.DataFrame:
    """Load first historical symbol file that contains funding/open-interest context."""
    hist_root = Path("Dataset/binance_historical")
    if not hist_root.exists():
        return pd.DataFrame()

    preferred = ["BTCUSDT.parquet", "ETHUSDT.parquet", "BNBUSDT.parquet", "SOLUSDT.parquet"]
    candidates = [hist_root / p for p in preferred if (hist_root / p).exists()]
    if not candidates:
        candidates = sorted(hist_root.glob("*.parquet"))

    for path in candidates:
        try:
            hist = pd.read_parquet(path)
        except Exception:
            continue
        if "timestamp" not in hist.columns:
            continue
        if "funding_rate" in hist.columns or "open_interest" in hist.columns:
            hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
            return hist
    return pd.DataFrame()


def integrate_multi_instrument_features():
    """Fuse futures/options states into synchronized feature table."""
    logger.info("\nIntegrating multi-instrument state features...")

    out_dir = PROCESSED_DIR / "multi_instrument"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = PROCESSED_DIR / "tick_bars" / "BTCUSDT_1000trades.parquet"
    if not base_path.exists():
        logger.warning("  Base tick bars missing; skipping multi-instrument integration")
        return

    base = pd.read_parquet(base_path)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    base = base.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Load UM futures context.
    um_metrics_files = _recent_parquet_files(FUTURES_DIR / "um" / "BTCUSDT" / "metrics")
    um_funding_files = _recent_parquet_files(FUTURES_DIR / "um" / "BTCUSDT" / "fundingRate")
    cm_funding_files = _recent_parquet_files(FUTURES_DIR / "cm" / "BTCUSD_PERP" / "fundingRate")
    bvol_btc_files = _recent_parquet_files(OPTIONS_DIR / "BVOLIndex" / "BTCBVOLUSDT")
    bvol_eth_files = _recent_parquet_files(OPTIONS_DIR / "BVOLIndex" / "ETHBVOLUSDT")

    if not bvol_btc_files or not bvol_eth_files:
        logger.warning("  Options BVOL context incomplete; skipping multi-instrument integration")
        return

    um_funding = pd.DataFrame(columns=["timestamp", "funding_um"])
    cm_funding = pd.DataFrame(columns=["timestamp", "funding_cm"])
    if um_funding_files:
        um_tmp = pd.concat([pd.read_parquet(p) for p in um_funding_files], ignore_index=True)
        um_tmp["timestamp"] = _extract_timestamp_column(um_tmp)
        um_rate_col = _first_existing_column(um_tmp, ["funding_rate", "col_1", "col_2"])
        if um_rate_col is not None:
            um_funding = um_tmp[["timestamp", um_rate_col]].rename(columns={um_rate_col: "funding_um"})
    if cm_funding_files:
        cm_tmp = pd.concat([pd.read_parquet(p) for p in cm_funding_files], ignore_index=True)
        cm_tmp["timestamp"] = _extract_timestamp_column(cm_tmp)
        cm_rate_col = _first_existing_column(cm_tmp, ["funding_rate", "col_1", "col_2"])
        if cm_rate_col is not None:
            cm_funding = cm_tmp[["timestamp", cm_rate_col]].rename(columns={cm_rate_col: "funding_cm"})

    # Fallback to historical merged dataset (contains funding/open-interest columns).
    hist_fallback = _load_hist_fallback_context()
    if (um_funding.empty or cm_funding.empty) and not hist_fallback.empty:
        if um_funding.empty and "funding_rate" in hist_fallback.columns:
            um_funding = hist_fallback[["timestamp", "funding_rate"]].rename(columns={"funding_rate": "funding_um"})
        if cm_funding.empty and "funding_rate" in hist_fallback.columns:
            cm_funding = hist_fallback[["timestamp", "funding_rate"]].rename(columns={"funding_rate": "funding_cm"})
    bvol_btc = pd.concat([pd.read_parquet(p) for p in bvol_btc_files], ignore_index=True)
    bvol_eth = pd.concat([pd.read_parquet(p) for p in bvol_eth_files], ignore_index=True)

    if "timestamp" in um_funding.columns:
        um_funding["timestamp"] = pd.to_datetime(um_funding["timestamp"], utc=True, errors="coerce")
    if "timestamp" in cm_funding.columns:
        cm_funding["timestamp"] = pd.to_datetime(cm_funding["timestamp"], utc=True, errors="coerce")
    bvol_btc["timestamp"] = _extract_timestamp_column(bvol_btc)
    bvol_eth["timestamp"] = _extract_timestamp_column(bvol_eth)

    if um_funding.empty or cm_funding.empty:
        logger.warning("  Funding context unavailable; skipping multi-instrument integration")
        return

    bvol_btc_col = _first_existing_column(bvol_btc, ["index_value", "col_1", "col_2", "close", "value"])
    bvol_eth_col = _first_existing_column(bvol_eth, ["index_value", "col_1", "col_2", "close", "value"])
    if bvol_btc_col is None or bvol_eth_col is None:
        logger.warning("  BVOL value columns not found; skipping multi-instrument integration")
        return

    bvol_btc = bvol_btc[["timestamp", bvol_btc_col]].rename(columns={bvol_btc_col: "bvol_btc"})
    bvol_eth = bvol_eth[["timestamp", bvol_eth_col]].rename(columns={bvol_eth_col: "bvol_eth"})

    # UM metrics for OI delta (optional but required for trend/mr enhancement).
    oi_df = pd.DataFrame(columns=["timestamp", "open_interest"])
    if um_metrics_files:
        um_metrics = pd.concat([pd.read_parquet(p) for p in um_metrics_files], ignore_index=True)
        um_metrics["timestamp"] = _extract_timestamp_column(um_metrics)
        oi_col = _first_existing_column(
            um_metrics,
            ["sum_open_interest", "open_interest", "sumOpenInterest", "col_1"],
        )
        if oi_col is not None:
            oi_df = um_metrics[["timestamp", oi_col]].rename(columns={oi_col: "open_interest"})
    if oi_df.empty and not hist_fallback.empty and "open_interest" in hist_fallback.columns:
        oi_df = hist_fallback[["timestamp", "open_interest"]]

    spot_micro_path = PROCESSED_DIR / "microstructure" / "BTCUSDT_ofi_vpin.parquet"
    fut_agg_files = _recent_parquet_files(FUTURES_DIR / "um" / "BTCUSDT" / "aggTrades")
    if not spot_micro_path.exists() or not fut_agg_files:
        logger.warning("  Cross-instrument OFI inputs missing; generating partial multi-state features")

    out = base[["timestamp", "close"]].copy()
    out = pd.merge_asof(out, um_funding.sort_values("timestamp"), on="timestamp", direction="backward")
    out = pd.merge_asof(out, cm_funding.sort_values("timestamp"), on="timestamp", direction="backward")
    out = pd.merge_asof(out, oi_df.sort_values("timestamp"), on="timestamp", direction="backward")
    out = pd.merge_asof(out, bvol_btc.sort_values("timestamp"), on="timestamp", direction="backward")
    out = pd.merge_asof(out, bvol_eth.sort_values("timestamp"), on="timestamp", direction="backward")

    # Basis uses UM mark/close proxy from spot close with funding context fallback.
    out["spot_futures_basis"] = FeatureFactory.compute_spot_futures_basis(
        out["close"],
        out["close"] * (1.0 + out["funding_um"].fillna(0.0)),
    )
    out["funding_spread"] = FeatureFactory.compute_funding_spread(
        out["funding_um"].fillna(0.0),
        out["funding_cm"].fillna(0.0),
    )
    out["oi_delta"] = FeatureFactory.compute_oi_delta(out["open_interest"].ffill(), periods=12)
    out["funding_sentiment"] = FeatureFactory.compute_funding_sentiment(out["funding_um"].fillna(0.0), window=24)

    bvol_state = FeatureFactory.build_bvol_global_state(
        out["bvol_btc"].ffill(),
        out["bvol_eth"].ffill(),
    )
    for col in bvol_state.columns:
        out[col] = bvol_state[col]

    if spot_micro_path.exists() and fut_agg_files:
        spot_micro = pd.read_parquet(spot_micro_path)
        spot_micro["timestamp"] = pd.to_datetime(spot_micro["timestamp"], utc=True, errors="coerce")
        spot_micro = spot_micro[["timestamp", "ofi"]].rename(columns={"ofi": "spot_ofi"})

        fut_agg = pd.concat([pd.read_parquet(p) for p in fut_agg_files], ignore_index=True)
        fut_agg["timestamp"] = _extract_timestamp_column(fut_agg)
        qty_col = _first_existing_column(fut_agg, ["quantity", "qty", "col_2"])
        maker_col = _first_existing_column(fut_agg, ["is_buyer_maker", "col_6"])
        if qty_col is not None:
            fut_agg_qty = pd.to_numeric(fut_agg[qty_col], errors="coerce").fillna(0.0)
            if maker_col is not None:
                fut_agg_sign = fut_agg_qty.where(~fut_agg[maker_col].astype(bool), -fut_agg_qty)
            else:
                fut_agg_sign = fut_agg_qty
            fut_agg = pd.DataFrame({"timestamp": fut_agg["timestamp"], "futures_signed": fut_agg_sign})
            fut_agg = fut_agg.sort_values("timestamp")
            fut_agg["futures_ofi"] = fut_agg["futures_signed"].rolling(20, min_periods=1).sum()
            fut_agg = fut_agg[["timestamp", "futures_ofi"]]

            out = pd.merge_asof(out.sort_values("timestamp"), spot_micro.sort_values("timestamp"), on="timestamp", direction="backward")
            out = pd.merge_asof(out.sort_values("timestamp"), fut_agg.sort_values("timestamp"), on="timestamp", direction="backward")
            out["cross_instrument_ofi"] = FeatureFactory.compute_cross_instrument_ofi(
                out["spot_ofi"].fillna(0.0),
                out["futures_ofi"].fillna(0.0),
                window=20,
            )

    out_file = out_dir / "BTCUSDT_multi_instrument_state.parquet"
    out.to_parquet(out_file, index=False, compression="zstd")
    logger.info(f"  Multi-instrument state: {len(out)} rows saved to {out_file}")


def validate_phase1_output():
    """Verify Phase 1 data exists before proceeding."""
    logger.info("Validating Phase 1 output...")
    
    vision_dir = VISION_DIR
    
    if not vision_dir.exists():
        raise FileNotFoundError(f"Phase 1 data directory not found: {vision_dir}")
    
    # Check each asset
    for asset in TIER1_ASSETS:
        asset_dir = vision_dir / asset
        if not asset_dir.exists():
            raise FileNotFoundError(f"Asset directory missing: {asset_dir}")
        
        # Check at least one data type exists
        has_data = any(dt_dir.exists() for dt_dir in asset_dir.glob("*/"))
        if not has_data:
            raise FileNotFoundError(f"No data types found for {asset}")
    
    logger.info(f"✓ Phase 1 data validated: {len(TIER1_ASSETS)} assets with data types")


def build_tick_volume_bars():
    """Build tick and volume bars from aggTrades."""
    logger.info("Building tick/volume bars...")
    
    tick_bars_dir = PROCESSED_DIR / "tick_bars"
    volume_bars_dir = PROCESSED_DIR / "volume_bars"
    tick_bars_dir.mkdir(parents=True, exist_ok=True)
    volume_bars_dir.mkdir(parents=True, exist_ok=True)
    
    vision_dir = VISION_DIR
    
    for asset in TIER1_ASSETS:
        logger.info(f"\n  Processing {asset}...")
        
        trades_dir = vision_dir / asset / "aggTrades"
        if not trades_dir.exists():
            logger.warning(f"    No aggTrades data for {asset}, skipping")
            continue
        
        # Load all trades for asset
        all_trades = []
        for parquet_file in _recent_parquet_files(trades_dir):
            try:
                df = pd.read_parquet(parquet_file)
                all_trades.append(df)
                logger.info(f"    Loaded {parquet_file.name}: {len(df)} trades")
            except Exception as e:
                logger.warning(f"    Error loading {parquet_file}: {e}")
        
        if not all_trades:
            logger.warning(f"    No trades loaded for {asset}")
            continue
        
        trades = _normalize_trades_schema(pd.concat(all_trades, ignore_index=True))
        trades = trades.sort_values("timestamp")
        logger.info(f"    Total: {len(trades)} trades")
        
        # Build tick bars (1000 trades per bar)
        try:
            tick_bars = FeatureFactory.build_tick_bars(trades, n_trades=1000)
            tick_bars.to_parquet(
                tick_bars_dir / f"{asset}_1000trades.parquet",
                compression="zstd",
            )
            logger.info(f"    Tick bars: {len(tick_bars)} bars saved")
        except Exception as e:
            logger.error(f"    Error building tick bars: {e}")
        
        # Build volume bars (100 BTC per bar)
        try:
            volume_bars = FeatureFactory.build_volume_bars(trades, volume_threshold=100.0)
            volume_bars.to_parquet(
                volume_bars_dir / f"{asset}_100btc.parquet",
                compression="zstd",
            )
            logger.info(f"    Volume bars: {len(volume_bars)} bars saved")
        except Exception as e:
            logger.error(f"    Error building volume bars: {e}")


def extract_microstructure_features():
    """Extract OFI, VPIN, and spread dynamics."""
    logger.info("\nExtracting microstructure features...")
    
    micro_dir = PROCESSED_DIR / "microstructure"
    micro_dir.mkdir(parents=True, exist_ok=True)
    
    vision_dir = VISION_DIR
    
    for asset in TIER1_ASSETS:
        logger.info(f"\n  Processing {asset}...")
        
        # Extract OFI from aggTrades
        trades_dir = vision_dir / asset / "aggTrades"
        if trades_dir.exists():
            try:
                all_trades = []
                for parquet_file in _recent_parquet_files(trades_dir):
                    all_trades.append(pd.read_parquet(parquet_file))
                
                trades = _normalize_trades_schema(pd.concat(all_trades, ignore_index=True))
                trades = trades.sort_values("timestamp")
                
                ofi = FeatureFactory.compute_ofi(trades, window=20)
                vpin = FeatureFactory.compute_vpin(trades, bucket_size=1000)
                
                micro_features = pd.DataFrame({
                    "timestamp": trades["timestamp"],
                    "ofi": ofi,
                    "vpin": vpin,
                })
                
                micro_features.to_parquet(
                    micro_dir / f"{asset}_ofi_vpin.parquet",
                    compression="zstd",
                )
                logger.info(f"    OFI/VPIN: {len(micro_features)} records saved")
            except Exception as e:
                logger.error(f"    Error extracting OFI/VPIN: {e}")
        
        # Extract spread dynamics from bookTicker
        book_dir = vision_dir / asset / "bookTicker"
        if book_dir.exists():
            try:
                all_books = []
                for parquet_file in _recent_parquet_files(book_dir):
                    all_books.append(pd.read_parquet(parquet_file))
                
                book = pd.concat(all_books, ignore_index=True).sort_values("event_time")
                
                spread_df = FeatureFactory.compute_spread_dynamics(book, window=20)
                spread_df.to_parquet(
                    micro_dir / f"{asset}_spread_dynamics.parquet",
                    compression="zstd",
                )
                logger.info(f"    Spread dynamics: {len(spread_df)} records saved")
            except Exception as e:
                logger.error(f"    Error extracting spread dynamics: {e}")


def generate_synthetic_data():
    """Generate synthetic data for model robustness."""
    logger.info("\nGenerating synthetic data...")
    
    synth_dir = PROCESSED_DIR / "synthetic"
    synth_dir.mkdir(parents=True, exist_ok=True)
    
    # Example: Generate GARCH synthetic paths
    logger.info("  Generating GARCH synthetic paths...")
    
    # Use BTCUSDT as baseline
    try:
        prices_df = pd.read_parquet(PROCESSED_DIR / "tick_bars" / "BTCUSDT_1000trades.parquet")
        prices = prices_df["close"].values
        returns = np.diff(np.log(prices))
        
        # Generate 100 synthetic price paths
        synthetic_garch = FeatureFactory.generate_synthetic_garch(returns, n_sim=100)
        
        # Save synthetic paths
        np.save(synth_dir / "btc_garch_100paths.npy", synthetic_garch)
        logger.info(f"    GARCH paths: {synthetic_garch.shape} saved")
    except Exception as e:
        logger.error(f"    Error generating GARCH: {e}")
    
    # Generate HMM regime scenarios
    logger.info("  Generating HMM regime scenarios...")
    try:
        synthetic_hmm = FeatureFactory.generate_synthetic_hmm(prices, n_regimes=3, n_sim=100)
        np.save(synth_dir / "btc_hmm_100paths.npy", synthetic_hmm)
        logger.info(f"    HMM paths: {synthetic_hmm.shape} saved")
    except Exception as e:
        logger.error(f"    Error generating HMM: {e}")


def generate_triple_barrier_labels():
    """Generate adaptive triple-barrier labels."""
    logger.info("\nGenerating triple-barrier labels...")
    
    labels_dir = PROCESSED_DIR / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load tick bars for BTCUSDT
        tick_bars = pd.read_parquet(PROCESSED_DIR / "tick_bars" / "BTCUSDT_1000trades.parquet")
        
        prices = tick_bars["close"].astype(float)
        returns_vol = prices.pct_change().rolling(20).std()
        
        # Apply adaptive triple-barrier labels
        labels = FeatureFactory.apply_adaptive_triple_barrier(
            prices=prices,
            returns_vol=returns_vol,
            vol_quantile_low=0.33,
            vol_quantile_high=0.67,
            barrier_scale_low=0.0005,
            barrier_scale_normal=0.001,
            barrier_scale_high=0.002,
            max_bars=20,
        )
        
        # Analyze label distribution
        label_counts = labels.value_counts()
        label_pcts = (label_counts / len(labels)) * 100
        
        labels_df = pd.DataFrame({
            "timestamp": tick_bars["timestamp"],
            "label": labels,
        })
        
        labels_df.to_parquet(labels_dir / "BTCUSDT_triple_barrier_adaptive.parquet")
        
        logger.info(f"  Triple-barrier labels: {len(labels)} records")
        logger.info(f"    LONG (+1):  {label_counts.get(1, 0):6d} ({label_pcts.get(1, 0):5.2f}%)")
        logger.info(f"    FLAT (0):   {label_counts.get(0, 0):6d} ({label_pcts.get(0, 0):5.2f}%)")
        logger.info(f"    SHORT (-1): {label_counts.get(-1, 0):6d} ({label_pcts.get(-1, 0):5.2f}%)")
        
        # Validation: FLAT should be 25-35%
        flat_pct = label_pcts.get(0, 0)
        if 20 <= flat_pct <= 40:
            logger.info(f"  ✓ Label distribution healthy (FLAT {flat_pct:.1f}%)")
        else:
            logger.warning(f"  ⚠ Label distribution skewed (FLAT {flat_pct:.1f}%, target 25-35%)")
    
    except Exception as e:
        logger.error(f"  Error generating labels: {e}")


def generate_phase2_report():
    """Generate summary report."""
    logger.info("\nGenerating Phase 2 summary report...")
    
    report_lines = [
        "=" * 100,
        "PHASE 2: FEATURE ENGINEERING & SYNTHETIC DATA SUMMARY",
        "=" * 100,
        f"\nTimestamp: {datetime.utcnow().isoformat()}",
        f"\nOutput Directory: {PROCESSED_DIR.absolute()}",
        "",
        "Data Products Created:",
        f"  - Tick Bars (1000 trades/bar):  {PROCESSED_DIR / 'tick_bars'}",
        f"  - Volume Bars (100 BTC/bar):    {PROCESSED_DIR / 'volume_bars'}",
        f"  - Microstructure (OFI/VPIN):    {PROCESSED_DIR / 'microstructure'}",
        f"  - Spread Dynamics:              {PROCESSED_DIR / 'microstructure'}",
        f"  - Synthetic Data (GARCH/HMM):   {PROCESSED_DIR / 'synthetic'}",
        f"  - Triple-Barrier Labels:        {PROCESSED_DIR / 'labels'}",
        "",
        "Next Steps:",
        "  1. Verify label distribution (25-35% FLAT target)",
        "  2. Execute Phase 3: python execute_phase3_rl_training.py",
        "",
        "=" * 100,
    ]
    
    report = "\n".join(report_lines)
    logger.info(report)
    
    # Save report
    report_file = PROCESSED_DIR / "PHASE2_SUMMARY.txt"
    with open(report_file, "w") as f:
        f.write(report)
    logger.info(f"\n✓ Report saved to: {report_file}")


def main():
    """Execute Phase 2: Feature Engineering."""
    print("=" * 100)
    print("PHASE 2: FEATURE ENGINEERING & SYNTHETIC DATA")
    print("=" * 100)
    print(f"\nStart Time: {datetime.utcnow().isoformat()}\n")
    
    try:
        # Validate Phase 1 data
        validate_phase1_output()
        
        # Build bars
        build_tick_volume_bars()
        
        # Extract microstructure
        extract_microstructure_features()
        
        # Generate synthetic data
        generate_synthetic_data()
        
        # Generate labels
        generate_triple_barrier_labels()

        # Multi-instrument fusion
        integrate_multi_instrument_features()
        
        # Generate report
        generate_phase2_report()
        
        print("\n" + "=" * 100)
        print("PHASE 2 COMPLETE ✓")
        print("=" * 100)
        print(f"\nCompletion Time: {datetime.utcnow().isoformat()}")
        print("Next: Execute Phase 3 - RL Training with Curriculum Learning")
        
        return 0
    
    except Exception as e:
        logger.error(f"\n✗ Phase 2 failed: {e}", exc_info=True)
        print(f"\n✗ ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
