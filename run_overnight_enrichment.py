"""
Overnight enrichment launcher — Phase 5 Time Machine dataset.

Runs the full production ingest in resume mode (all already-written months
are skipped).  Safe to re-run at any time; adds only missing months.

Layers ingested:
  Parquet: fear_greed, tradfi_macro, on_chain, derivatives
  JSONL  : crypto_news, social_sentiment, hf_social, hf_news

Backfill note:
  SOL / BNB derivatives rows and SahandNZ / cvnberk JSONL months were
  added as one-time operations; see scripts/phase5/ for those scripts.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))

import asyncio
from src.phase5.environment.production.run_production_ingest import run_ingest, IngestConfig

cfg = IngestConfig(
    start="2013-01-01",
    end="2024-12-31",
    data_root=pathlib.Path("Dataset/phase5_time_machine_dataset"),
    resume=True,
    overwrite=False,
    layers=[
        "fear_greed",
        "tradfi_macro",
        "on_chain",
        "derivatives",
        "crypto_news",
        "social_sentiment",
        "hf_social",
        "hf_news",
    ],
)

asyncio.run(run_ingest(cfg))

# Print quality report at the end
from src.phase5.environment.production.chunked_storage import ChunkedLayerStorage
storage = ChunkedLayerStorage(cfg.data_root)
storage.print_quality_report()
