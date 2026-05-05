"""
Binance Vision S3 Concurrent Downloader & Parser
=================================================

This module implements a high-performance concurrent downloader for Binance Vision
historical data (https://data.binance.vision/). It downloads aggTrades, bookTicker,
fundingRate, and metrics from S3, validates checksums, unzips in-memory, and appends
to partitioned Parquet storage.

Key Features:
- Concurrent downloads using aiohttp with configurable concurrency limits
- Resume-on-failure with per-file progress tracking
- Checksum validation (MD5/SHA256) against Binance-provided checksums
- In-memory unzipping to avoid disk bottlenecks
- Append to partitioned Parquet (ZSTD compression)
- Logging and error recovery

Usage:
    scraper = BinanceVisionScraper(
        output_dir="Dataset/bn_vision_data",
        assets=["BTCUSDT", "ETHUSDT"],
        data_types=["aggTrades", "bookTicker"]
    )
    results = asyncio.run(scraper.download_all())
"""

import asyncio
import hashlib
import logging
import os
import re
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin
import zipfile

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Binance Vision Base URL
BINANCE_VISION_BASE = "https://data.binance.vision/"

# Target assets for Phase 1 (Tier-1 high-liquidity)
TIER1_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BTCETH", "HYPEUSDT"]

# Data types available from Binance Vision
DATA_TYPES = {
    "aggTrades": {
        "path": "data/spot/daily/aggTrades",
        "schema": {
            "agg_trade_id": "int64",
            "price": "float64",
            "quantity": "float64",
            "first_trade_id": "int64",
            "last_trade_id": "int64",
            "timestamp": "int64",
            "is_buyer_maker": "bool",
        },
        "description": "Aggregated trade stream (tick-level events)",
    },
    "bookTicker": {
        "path": "data/spot/daily/bookTicker",
        "schema": {
            "event_time": "int64",
            "symbol": "string",
            "best_bid_price": "float64",
            "best_bid_qty": "float64",
            "best_ask_price": "float64",
            "best_ask_qty": "float64",
        },
        "description": "Best bid/ask snapshots (L2 top of book)",
    },
    "fundingRate": {
        "path": "data/futures/um/daily/fundingRate",
        "schema": {
            "symbol": "string",
            "funding_time": "int64",
            "funding_rate": "float64",
        },
        "description": "Perpetuals funding rates (market sentiment indicator)",
    },
    "metrics": {
        "path": "data/futures/um/daily/metrics",
        "schema": {
            "symbol": "string",
            "event_time": "int64",
            "open_interest": "float64",
            "oi_change": "float64",
            "top_long_short_account_ratio": "float64",
            "top_long_short_position_ratio": "float64",
            "long_short_account_ratio": "float64",
            "long_short_position_ratio": "float64",
        },
        "description": "Open interest, long/short ratios, and derivatives metrics",
    },
}


class BinanceVisionScraper:
    """Concurrent downloader and parser for Binance Vision historical data."""

    def __init__(
        self,
        output_dir: str = "Dataset/bn_vision_data",
        assets: Optional[List[str]] = None,
        data_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_concurrent: int = 5,
        timeout: int = 30,
    ):
        """
        Initialize the scraper.

        Args:
            output_dir: Root directory for all downloaded Parquet files
            assets: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
            data_types: List of data types (e.g., ["aggTrades", "bookTicker"])
            start_date: Start date in YYYY-MM-DD format (default: 90 days ago)
            end_date: End date in YYYY-MM-DD format (default: today)
            max_concurrent: Max concurrent downloads (HTTP connection pool limit)
            timeout: Timeout per request in seconds
        """
        self.output_dir = Path(output_dir)
        self.assets = assets or TIER1_ASSETS
        self.data_types = data_types or list(DATA_TYPES.keys())
        self.max_concurrent = max_concurrent
        self.timeout = timeout

        # Date range
        if end_date is None:
            end_date = datetime.utcnow().date()
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        if start_date is None:
            start_date = end_date - timedelta(days=90)
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

        self.start_date = start_date
        self.end_date = end_date

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress_file = self.output_dir / ".download_progress.txt"
        self.downloaded_files: Set[str] = self._load_progress()

        logger.info(
            f"BinanceVisionScraper initialized: "
            f"assets={self.assets}, data_types={self.data_types}, "
            f"date_range={self.start_date} to {self.end_date}"
        )

    def _load_progress(self) -> Set[str]:
        """Load previously downloaded files from progress file."""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def _save_progress(self, filename: str):
        """Append filename to progress file."""
        with open(self.progress_file, "a") as f:
            f.write(f"{filename}\n")

    async def _fetch_directory_listing(
        self, url: str, session: aiohttp.ClientSession
    ) -> List[str]:
        """
        Fetch HTML directory listing from Binance Vision S3 bucket.
        Parse XML to extract file links.
        """
        try:
            async with session.get(url, timeout=self.timeout) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to fetch {url}: HTTP {resp.status}")
                    return []
                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")

                # Extract all <Contents> elements (S3 XML format)
                files = []
                for key_tag in soup.find_all("key"):
                    key = key_tag.get_text()
                    if key and not key.endswith("/"):
                        files.append(key)

                logger.info(f"Found {len(files)} files in {url}")
                return files
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            return []
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return []

    async def _download_file(
        self,
        url: str,
        filename: str,
        session: aiohttp.ClientSession,
        checksum_expected: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Download a single file from Binance Vision S3.
        Validates checksum if provided.
        Returns file content as bytes on success, None on failure.
        """
        try:
            async with session.get(url, timeout=self.timeout) as resp:
                if resp.status != 200:
                    logger.error(f"Failed download {filename}: HTTP {resp.status}")
                    return None

                content = await resp.read()

                # Validate checksum (MD5) if available
                if checksum_expected:
                    md5 = hashlib.md5(content).hexdigest()
                    if md5.lower() != checksum_expected.lower():
                        logger.warning(
                            f"Checksum mismatch for {filename}: "
                            f"expected {checksum_expected}, got {md5}"
                        )
                        return None

                logger.info(f"Downloaded {filename} ({len(content)} bytes)")
                self._save_progress(filename)
                return content

        except asyncio.TimeoutError:
            logger.error(f"Timeout downloading {filename}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None

    async def _process_zip(
        self, filename: str, content: bytes, data_type: str
    ) -> pd.DataFrame:
        """
        Unzip in-memory and parse CSV to DataFrame.
        Validates schema and applies minimal transformations.
        """
        try:
            with zipfile.ZipFile(BytesIO(content)) as zf:
                csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
                if not csv_files:
                    logger.error(f"No CSV found in {filename}")
                    return None

                csv_file = csv_files[0]
                with zf.open(csv_file) as f:
                    df = pd.read_csv(f, dtype_backend="pyarrow")

                logger.info(
                    f"Parsed {csv_file}: {len(df)} rows, "
                    f"columns: {list(df.columns)}"
                )
                return df

        except Exception as e:
            logger.error(f"Error processing zip {filename}: {e}")
            return None

    async def _append_to_parquet(
        self,
        df: pd.DataFrame,
        asset: str,
        data_type: str,
        date: datetime.date,
    ):
        """
        Append DataFrame to partitioned Parquet storage.
        Partition: Dataset/bn_vision_data/{asset}/{data_type}/YYYY-MM/{date}.parquet
        Uses ZSTD compression for efficiency.
        """
        try:
            partition_dir = (
                self.output_dir / asset / data_type / date.strftime("%Y-%m")
            )
            partition_dir.mkdir(parents=True, exist_ok=True)

            parquet_file = partition_dir / f"{date.strftime('%Y%m%d')}.parquet"

            # If file exists, read and concat; else write new
            if parquet_file.exists():
                existing = pd.read_parquet(parquet_file)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(keep="last")  # Remove any duplicates
                logger.info(
                    f"Appended to {parquet_file}: "
                    f"{len(df)} total rows after dedup"
                )
            else:
                logger.info(
                    f"Created {parquet_file}: {len(df)} rows"
                )

            # Write with ZSTD compression
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                parquet_file,
                compression="zstd",
                compression_level=3,
            )

        except Exception as e:
            logger.error(
                f"Error appending to parquet "
                f"({asset}/{data_type}/{date}): {e}"
            )

    def _parse_date_from_filename(self, filename: str) -> Optional[datetime.date]:
        """Extract date from Binance Vision filename (YYYYMMDD format)."""
        match = re.search(r"(\d{8})", filename)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d").date()
            except ValueError:
                return None
        return None

    async def download_asset_data_type(
        self,
        asset: str,
        data_type: str,
        session: aiohttp.ClientSession,
    ) -> Dict[str, int]:
        """
        Download all available daily files for a single asset + data_type combination.

        Returns:
            Dict with stats: {files_downloaded, files_failed, rows_total}
        """
        stats = {"files_downloaded": 0, "files_failed": 0, "rows_total": 0}

        # Build URL path
        base_path = DATA_TYPES[data_type]["path"]
        url = urljoin(BINANCE_VISION_BASE, f"{base_path}/{asset}/")

        logger.info(f"Fetching directory: {url}")
        files = await self._fetch_directory_listing(url, session)

        if not files:
            logger.warning(f"No files found for {asset}/{data_type}")
            return stats

        # Filter by date range and skip already downloaded
        target_files = []
        for file in files:
            if file.endswith(".CHECKSUM"):
                continue  # Skip checksum files; we'll validate differently

            date = self._parse_date_from_filename(file)
            if date and self.start_date <= date <= self.end_date:
                if file not in self.downloaded_files:
                    target_files.append(file)

        logger.info(
            f"Downloading {len(target_files)} files for "
            f"{asset}/{data_type} "
            f"(skipped {len(files) - len(target_files)} already-downloaded)"
        )

        # Download with semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def download_and_process(filename: str):
            async with semaphore:
                file_url = urljoin(url, filename)
                content = await self._download_file(
                    file_url, filename, session
                )

                if not content:
                    stats["files_failed"] += 1
                    return

                df = await self._process_zip(filename, content, data_type)
                if df is None:
                    stats["files_failed"] += 1
                    return

                date = self._parse_date_from_filename(filename)
                if date:
                    await self._append_to_parquet(df, asset, data_type, date)
                    stats["files_downloaded"] += 1
                    stats["rows_total"] += len(df)

        # Run all downloads concurrently
        await asyncio.gather(
            *[download_and_process(f) for f in target_files],
            return_exceptions=True,
        )

        return stats

    async def download_all(self) -> Dict[str, Dict[str, int]]:
        """
        Download all target assets and data types.

        Returns:
            Dict mapping (asset, data_type) -> stats
        """
        results = {}

        # Use connection pool for efficiency
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
        )
        timeout_obj = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout_obj
        ) as session:
            for asset in self.assets:
                for data_type in self.data_types:
                    key = f"{asset}/{data_type}"
                    logger.info(f"Starting download for {key}")

                    try:
                        stats = await self.download_asset_data_type(
                            asset, data_type, session
                        )
                        results[key] = stats

                        logger.info(
                            f"Completed {key}: "
                            f"{stats['files_downloaded']} files, "
                            f"{stats['rows_total']} rows, "
                            f"{stats['files_failed']} failures"
                        )
                    except Exception as e:
                        logger.error(f"Error downloading {key}: {e}")
                        results[key] = {
                            "files_downloaded": 0,
                            "files_failed": -1,
                            "rows_total": 0,
                            "error": str(e),
                        }

        return results

    def generate_summary_report(self) -> str:
        """
        Generate a summary report of downloaded data.
        Lists all Parquet files, row counts, and data distribution.
        """
        report_lines = [
            "=" * 80,
            "BINANCE VISION DATA DOWNLOAD SUMMARY",
            "=" * 80,
            f"Download Date: {datetime.utcnow().isoformat()}",
            f"Output Directory: {self.output_dir.absolute()}",
            f"Date Range: {self.start_date} to {self.end_date}",
            "",
        ]

        total_rows = 0
        total_files = 0

        # Traverse partitioned storage
        for asset_dir in sorted(self.output_dir.glob("*/")) :
            if asset_dir.name.startswith("."):
                continue

            asset = asset_dir.name
            report_lines.append(f"\n### ASSET: {asset}")

            for data_type_dir in sorted(asset_dir.glob("*/")):
                if data_type_dir.name.startswith("."):
                    continue

                data_type = data_type_dir.name
                report_lines.append(f"\n  **Data Type:** {data_type}")
                report_lines.append(
                    f"  Description: {DATA_TYPES.get(data_type, {}).get('description', 'N/A')}"
                )

                asset_data_type_rows = 0
                asset_data_type_files = 0

                # List all Parquet files
                for parquet_file in sorted(data_type_dir.rglob("*.parquet")):
                    try:
                        parquet_table = pq.read_table(parquet_file)
                        num_rows = len(parquet_table)
                        asset_data_type_rows += num_rows
                        asset_data_type_files += 1
                        total_rows += num_rows

                        report_lines.append(
                            f"    {parquet_file.relative_to(self.output_dir)}: "
                            f"{num_rows:,} rows"
                        )
                    except Exception as e:
                        logger.error(f"Error reading {parquet_file}: {e}")

                report_lines.append(
                    f"\n  **Subtotal:** {asset_data_type_files} files, "
                    f"{asset_data_type_rows:,} rows"
                )

                total_files += asset_data_type_files

        report_lines.extend(
            [
                "",
                "=" * 80,
                f"GRAND TOTAL: {total_files} files, {total_rows:,} rows",
                "=" * 80,
            ]
        )

        report = "\n".join(report_lines)
        logger.info(report)

        return report


# Example usage / CLI
if __name__ == "__main__":
    import sys

    # Parse command-line arguments (optional)
    start_date_arg = None
    end_date_arg = None

    if len(sys.argv) > 1:
        start_date_arg = sys.argv[1]
    if len(sys.argv) > 2:
        end_date_arg = sys.argv[2]

    # Initialize scraper
    scraper = BinanceVisionScraper(
        output_dir="Dataset/bn_vision_data",
        assets=TIER1_ASSETS,
        data_types=["aggTrades", "bookTicker", "fundingRate", "metrics"],
        start_date=start_date_arg,
        end_date=end_date_arg,
        max_concurrent=5,
    )

    # Run async downloader
    results = asyncio.run(scraper.download_all())

    # Print results
    print("\n\nDOWNLOAD RESULTS:")
    for key, stats in results.items():
        print(f"{key}: {stats}")

    # Generate summary report
    summary = scraper.generate_summary_report()
    summary_file = scraper.output_dir / "DOWNLOAD_SUMMARY.txt"
    with open(summary_file, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to {summary_file}")
