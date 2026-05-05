#!/usr/bin/env python3
"""
Mock Data Generator for Phase 1-4 Pipeline Testing
===================================================

This script generates synthetic test data that mimics Phase 1 output,
allowing you to test Phase 2, 3, and 4 without network connectivity.

Execution: python generate_mock_phase1_data.py

Output: Dataset/bn_vision_data/{asset}/{data_type}/*.parquet
        (Enough data to proceed through Phase 2-4)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mock_data_generator")

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BTCETH", "HYPEUSDT"]
DATA_TYPES = ["aggTrades", "bookTicker", "fundingRate", "metrics"]
OUTPUT_DIR = Path("Dataset/bn_vision_data")


def generate_aggtrades(asset, num_records=10000):
    """Generate synthetic aggTrades (tick) data."""
    logger.info(f"  Generating {num_records:,} aggTrades records for {asset}...")
    
    timestamps = pd.date_range(
        start="2024-01-01", 
        periods=num_records, 
        freq="1S"
    )
    
    # Synthetic price data with realistic characteristics
    price_change = np.random.normal(0, 0.001, num_records)
    base_price = 40000 if asset == "BTCUSDT" else 2000 if asset == "ETHUSDT" else 150
    prices = base_price * np.exp(np.cumsum(price_change))
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "quantity": np.random.exponential(0.5, num_records),
        "side": np.random.choice(["BUY", "SELL"], num_records),
        "trade_id": range(num_records),
    })
    
    return df


def generate_bookticker(asset, num_records=5000):
    """Generate synthetic bookTicker (L2 snapshot) data."""
    logger.info(f"  Generating {num_records:,} bookTicker records for {asset}...")
    
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=num_records,
        freq="2S"
    )
    
    base_price = 40000 if asset == "BTCUSDT" else 2000
    price_change = np.random.normal(0, 0.001, num_records)
    prices = base_price * np.exp(np.cumsum(price_change))
    
    spread = np.random.exponential(0.001, num_records)
    
    df = pd.DataFrame({
        "event_time": timestamps,
        "bid_price": prices - spread / 2,
        "bid_qty": np.random.exponential(1.0, num_records),
        "ask_price": prices + spread / 2,
        "ask_qty": np.random.exponential(1.0, num_records),
    })
    
    return df


def generate_funding_rate(asset, num_records=1000):
    """Generate synthetic fundingRate data."""
    logger.info(f"  Generating {num_records:,} fundingRate records for {asset}...")
    
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=num_records,
        freq="8H"
    )
    
    df = pd.DataFrame({
        "event_time": timestamps,
        "funding_rate": np.random.normal(0.0001, 0.0002, num_records),
        "mark_price": 40000 + np.random.normal(0, 500, num_records),
    })
    
    return df


def generate_metrics(asset, num_records=1000):
    """Generate synthetic metrics (open interest, etc.) data."""
    logger.info(f"  Generating {num_records:,} metrics records for {asset}...")
    
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=num_records,
        freq="1H"
    )
    
    df = pd.DataFrame({
        "event_time": timestamps,
        "open_interest": np.random.lognormal(15, 1, num_records),
        "long_short_ratio": np.random.beta(2, 2, num_records),
        "volume_24h": np.random.lognormal(20, 2, num_records),
    })
    
    return df


def save_parquet(df, asset, data_type):
    """Save DataFrame to partitioned parquet files."""
    output_dir = OUTPUT_DIR / asset / data_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as single monthly partition (for simplicity)
    output_file = output_dir / "2024-01" / "20240115.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_file, compression="zstd", index=False)
    logger.info(f"    Saved: {output_file} ({len(df)} rows)")


def generate_all_mock_data():
    """Generate mock data for all assets and data types."""
    print("=" * 80)
    print("MOCK DATA GENERATOR FOR PHASE 1-4 PIPELINE TESTING")
    print("=" * 80)
    print(f"\nOutput Directory: {OUTPUT_DIR.absolute()}\n")
    
    logger.info(f"Generating mock data for {len(ASSETS)} assets × {len(DATA_TYPES)} data types")
    
    for asset in ASSETS:
        logger.info(f"\n{asset}:")
        
        # aggTrades: high frequency (1 per second)
        aggtrades = generate_aggtrades(asset, num_records=10000)
        save_parquet(aggtrades, asset, "aggTrades")
        
        # bookTicker: medium frequency (1 per 2 seconds)
        bookticker = generate_bookticker(asset, num_records=5000)
        save_parquet(bookticker, asset, "bookTicker")
        
        # fundingRate: low frequency (every 8 hours)
        funding = generate_funding_rate(asset, num_records=1000)
        save_parquet(funding, asset, "fundingRate")
        
        # metrics: 1 per hour
        metrics = generate_metrics(asset, num_records=1000)
        save_parquet(metrics, asset, "metrics")
    
    # Generate summary report
    logger.info("\n" + "=" * 80)
    logger.info("MOCK DATA GENERATION COMPLETE")
    logger.info("=" * 80)
    
    summary_lines = [
        "=" * 80,
        "MOCK DATA GENERATION SUMMARY",
        "=" * 80,
        f"\nGenerated: {datetime.utcnow().isoformat()}",
        f"Output Directory: {OUTPUT_DIR.absolute()}",
        "",
        "Data Generated:",
    ]
    
    for asset in ASSETS:
        summary_lines.append(f"\n{asset}:")
        for data_type in DATA_TYPES:
            summary_lines.append(f"  - {data_type}: 1 parquet file")
    
    summary_lines.extend([
        "",
        "Total Records Generated:",
        "  - aggTrades:   50,000 (10,000 per asset)",
        "  - bookTicker:  25,000 (5,000 per asset)",
        "  - fundingRate: 5,000 (1,000 per asset)",
        "  - metrics:     5,000 (1,000 per asset)",
        "  TOTAL:         85,000 records",
        "",
        "Status: ✓ Ready for Phase 2-4 Testing",
        "",
        "Next Steps:",
        "  1. python execute_phase2_feature_engineering.py",
        "  2. python execute_phase3_rl_training.py",
        "  3. python execute_phase4_feature_pruning.py",
        "",
        "=" * 80,
    ])
    
    summary = "\n".join(summary_lines)
    logger.info(summary)
    
    # Save summary
    summary_file = OUTPUT_DIR / "MOCK_DATA_SUMMARY.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)
    
    logger.info(f"\n✓ Summary saved to: {summary_file}")
    
    return 0


def main():
    """Main entry point."""
    try:
        return generate_all_mock_data()
    except Exception as e:
        logger.error(f"Error generating mock data: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
