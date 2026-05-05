#!/usr/bin/env python3
"""
Phase 1 Execution Driver: Binance Vision Data Acquisition
===========================================================

This script orchestrates the concurrent download of Binance Vision historical data
for Tier-1 assets (BTCUSDT, ETHUSDT, SOLUSDT, BTCETH, HYPEUSDT).

Execution: python execute_phase1_vision_scraper.py [start_date] [end_date]
Example:   python execute_phase1_vision_scraper.py 2024-01-01 2026-04-26

Runtime: ~1-2 hours with max_concurrent=5 (90-day window, 5 assets)
Output: Dataset/bn_vision_data/{asset}/{data_type}/YYYY-MM/*.parquet
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from quant_core.data_pipeline.vision_scraper import BinanceVisionScraper, TIER1_ASSETS

def main():
    """Execute Phase 1: Binance Vision concurrent scraper."""
    
    # Parse command-line arguments
    start_date = None
    end_date = None
    
    if len(sys.argv) > 1:
        start_date = sys.argv[1]
    if len(sys.argv) > 2:
        end_date = sys.argv[2]
    
    print("=" * 100)
    print("PHASE 1: BINANCE VISION DATA ACQUISITION")
    print("=" * 100)
    print(f"\nTimestamp: {datetime.utcnow().isoformat()}")
    print(f"Tier-1 Assets: {', '.join(TIER1_ASSETS)}")
    print(f"Data Types: aggTrades, bookTicker, fundingRate, metrics")
    print(f"Output Directory: Dataset/bn_vision_data/")
    print(f"Max Concurrent Workers: 5")
    print(f"Date Range: {start_date or '(90 days ago)'} to {end_date or '(today)'}")
    print()
    
    # Initialize scraper
    scraper = BinanceVisionScraper(
        output_dir="Dataset/bn_vision_data",
        assets=TIER1_ASSETS,
        data_types=["aggTrades", "bookTicker", "fundingRate", "metrics"],
        start_date=start_date,
        end_date=end_date,
        max_concurrent=5,
        timeout=30,
    )
    
    print(f"Scraper initialized: {len(scraper.assets)} assets × 4 data types = 20 download streams")
    print(f"Progress tracking: {scraper.progress_file}")
    print(f"Previously downloaded: {len(scraper.downloaded_files)} files")
    print()
    
    # Execute concurrent downloads
    print("=" * 100)
    print("STARTING CONCURRENT DOWNLOADS")
    print("=" * 100)
    print(f"\nStart Time: {datetime.utcnow().isoformat()}")
    print("Estimated Duration: 1-2 hours (network dependent)")
    print("Monitor Progress: Check Dataset/bn_vision_data/.download_progress.txt\n")
    
    try:
        results = asyncio.run(scraper.download_all())
        
        # Summarize results
        print("\n" + "=" * 100)
        print("DOWNLOAD RESULTS")
        print("=" * 100)
        
        total_files = 0
        total_rows = 0
        total_failures = 0
        
        for key, stats in results.items():
            asset, data_type = key.split("/")
            status = "✓" if stats["files_failed"] == 0 else "✗"
            print(
                f"{status} {asset:10s} / {data_type:12s}: "
                f"{stats['files_downloaded']:3d} files, "
                f"{stats['rows_total']:10,d} rows"
            )
            total_files += stats['files_downloaded']
            total_rows += stats['rows_total']
            total_failures += stats['files_failed']
        
        print()
        print(f"{'='*100}")
        print(f"Total: {total_files} files downloaded, {total_rows:,} rows ingested")
        if total_failures > 0:
            print(f"Failures: {total_failures} files (resume by re-running)")
        print(f"Completion Time: {datetime.utcnow().isoformat()}")
        print(f"{'='*100}\n")
        
        # Generate summary report
        print("Generating summary report...")
        summary = scraper.generate_summary_report()
        
        # Save summary to file
        summary_file = Path("Dataset/bn_vision_data/DOWNLOAD_SUMMARY.txt")
        with open(summary_file, "w") as f:
            f.write(summary)
        
        print(f"\n✓ Summary report saved to: {summary_file}")
        print(f"\n{'='*100}")
        print("PHASE 1 COMPLETE")
        print(f"{'='*100}")
        print(f"\nNext Steps:")
        print("1. Verify data integrity: Dataset/bn_vision_data/DOWNLOAD_SUMMARY.txt")
        print("2. Execute Phase 2: python execute_phase2_feature_engineering.py")
        print(f"\nStatus: ✓ READY FOR PHASE 2")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ ERROR during download: {e}")
        print(f"\nDiagnostics:")
        print(f"- Check network connectivity to data.binance.vision")
        print(f"- Verify ~80 GB free disk space in Dataset/")
        print(f"- Review logs above for specific failure details")
        print(f"\nTo resume interrupted downloads, re-run this script:")
        print(f"  python execute_phase1_vision_scraper.py {start_date or '2024-01-01'} {end_date or '2026-04-26'}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
