#!/usr/bin/env python3
"""
Phase 5 – Time Machine Dataset  |  Ingestion & Mock-Generation Pipeline
========================================================================
Populates ``Dataset/phase5_time_machine_dataset/`` with 10 synchronised
environmental data layers.

When live historical endpoints are unavailable the script mock-generates
a structurally-identical, historically-grounded slice for any requested
window.  The **Black Thursday** scenario (2020-03-11 → 2020-03-15) ships
as a named preset with realistic event data.

CLI
---
    # Generate the Black Thursday mock slice
    python src/phase5/environment/ingest_time_machine.py \\
        --mock --scenario black_thursday \\
        --data-root Dataset/phase5_time_machine_dataset

    # Generic mock for a custom window
    python src/phase5/environment/ingest_time_machine.py \\
        --mock --start 2022-11-06 --end 2022-11-14 \\
        --data-root Dataset/phase5_time_machine_dataset

Anti-leakage contract
---------------------
    * Every row carries ``timestamp_utc`` (int64, ms epoch, UTC).
    * No derived or look-ahead field is written.
    * The oracle in ``time_machine_oracle.py`` enforces strict  ``<= T``
      filtering before handing data to any agent.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import pathlib
import random
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

# Local schemas
from src.phase5.environment.schemas import (
    ALL_LAYER_KEYS,
    DERIVATIVES_COLUMNS,
    FEAR_GREED_COLUMNS,
    JSONL_LAYERS,
    LAYER_DIRS,
    ON_CHAIN_COLUMNS,
    PARQUET_LAYERS,
    TRADFI_MACRO_COLUMNS,
    CrisisHackEvent,
    CryptoNewsEvent,
    KOLFootprint,
    MacroGeopoliticalEvent,
    RegulatoryLegalEvent,
    SocialSentimentSnapshot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("time_machine.ingest")

# ─────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dt(date_str: str, time_str: str = "00:00:00") -> datetime:
    """Return a UTC-aware datetime from separate date and time strings."""
    return datetime.strptime(
        f"{date_str}T{time_str}", "%Y-%m-%dT%H:%M:%S"
    ).replace(tzinfo=timezone.utc)


def _ms(dt_obj: datetime) -> int:
    """Convert a UTC datetime to Unix milliseconds (int64)."""
    return int(dt_obj.timestamp() * 1_000)


def _ts(date_str: str, time_str: str = "00:00:00") -> int:
    """Convenience: return ms epoch for a date + time string."""
    return _ms(_dt(date_str, time_str))


def _iso(dt_obj: datetime) -> str:
    return dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")


def _uid() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Layer I/O  (saving Parquet and JSONL)
# ─────────────────────────────────────────────────────────────────────────────

class LayerIngestor:
    """Handles directory creation and serialisation for all 10 layers."""

    def __init__(self, data_root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(data_root).resolve()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for subdir in LAYER_DIRS.values():
            (self.root / subdir).mkdir(parents=True, exist_ok=True)
        log.info("Dataset directory tree ready at %s", self.root)

    def layer_path(self, layer_key: str) -> pathlib.Path:
        return self.root / LAYER_DIRS[layer_key]

    def save_jsonl(
        self,
        records: list[dict[str, Any]],
        layer_key: str,
        filename: str,
        *,
        append: bool = False,
    ) -> pathlib.Path:
        """Write a list of dicts to a JSONL file (one JSON object per line)."""
        out = self.layer_path(layer_key) / filename
        mode = "a" if append else "w"
        with out.open(mode, encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("Saved %d records → %s", len(records), out)
        return out

    def save_parquet(
        self,
        df: pd.DataFrame,
        layer_key: str,
        filename: str,
    ) -> pathlib.Path:
        """Write a DataFrame to a compressed Parquet file."""
        out = self.layer_path(layer_key) / filename
        df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
        log.info("Saved %d rows → %s", len(df), out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Mock Data Generator  –  Black Thursday preset (2020-03-11 … 2020-03-15)
# ─────────────────────────────────────────────────────────────────────────────

class BlackThursdayMockGenerator:
    """
    Generates historically-grounded mock data for the COVID-19 Black Thursday
    crypto crash window (2020-03-11 to 2020-03-15 UTC).

    Key historical anchors embedded:
        * BTC: $7,910 → $3,782 (−52 %) intraday low on Mar 12 → recovery to ~$5,400
        * ETH: $193 → $88 → recovery to ~$130
        * S&P 500: −4.9 % (Mar 11), −9.5 % (Mar 12), +9.3 % (Mar 13)
        * VIX: 54.5 → 75.5 (Mar 12 peak)
        * Fed: Emergency −100 bps cut announced Sunday Mar 15
        * MakerDAO Black Thursday: zero-bid DAI auctions → $8.32 M bad debt
        * >$1.2 B liquidated across perpetual swap venues on Mar 12
    """

    START_DATE = "2020-03-11"
    END_DATE   = "2020-03-15"

    # ── BTC price curve: hourly close prices for the 5-day window ────────────
    BTC_HOURLY = [
        # Mar 11  (UTC hours 0-23)
        7910, 7890, 7865, 7830, 7800, 7780, 7760, 7730, 7700, 7680,
        7650, 7620, 7590, 7560, 7510, 7470, 7430, 7400, 7380, 7360,
        7340, 7300, 7260, 7220,
        # Mar 12  (UTC hours 0-23) – Black Thursday
        7200, 7150, 7050, 6900, 6700, 6400, 6000, 5600, 5200, 4900,
        4600, 4300, 4050, 3900, 3782, 3850, 3950, 4050, 4200, 4400,
        4600, 4800, 4950, 5050,
        # Mar 13  (UTC hours 0-23)
        5050, 5100, 5200, 5300, 5400, 5500, 5600, 5650, 5700, 5750,
        5800, 5850, 5900, 5950, 6000, 6050, 6100, 6080, 6050, 6020,
        5990, 5960, 5940, 5920,
        # Mar 14  (UTC hours 0-23)
        5920, 5900, 5880, 5850, 5830, 5820, 5840, 5860, 5880, 5900,
        5870, 5840, 5810, 5790, 5800, 5820, 5840, 5850, 5860, 5870,
        5880, 5890, 5870, 5850,
        # Mar 15  (UTC hours 0-23)
        5850, 5820, 5780, 5750, 5700, 5650, 5600, 5550, 5500, 5480,
        5460, 5440, 5420, 5400, 5380, 5400, 5420, 5440, 5460, 5480,
        5500, 5520, 5540, 5560,
    ]  # 120 values

    # ── ETH price curve: hourly close ────────────────────────────────────────
    ETH_HOURLY = [
        # Mar 11
        193, 191, 189, 187, 185, 183, 181, 179, 177, 175,
        173, 171, 169, 167, 165, 163, 161, 159, 157, 155,
        153, 151, 149, 147,
        # Mar 12 – Black Thursday  (MakerDAO crisis adds extra pressure)
        145, 140, 133, 125, 118, 110, 102, 96, 90, 86,
        83, 81, 88, 92, 95, 98, 101, 104, 107, 110,
        113, 116, 119, 122,
        # Mar 13
        122, 124, 126, 128, 130, 132, 134, 136, 137, 138,
        138, 137, 136, 135, 134, 133, 132, 131, 130, 129,
        128, 127, 126, 125,
        # Mar 14
        125, 125, 126, 127, 128, 128, 129, 130, 131, 132,
        132, 131, 130, 129, 130, 131, 132, 133, 134, 134,
        133, 132, 131, 130,
        # Mar 15
        130, 129, 128, 127, 126, 125, 124, 123, 122, 121,
        120, 119, 118, 117, 116, 117, 118, 119, 120, 121,
        122, 123, 124, 125,
    ]  # 120 values

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        np.random.seed(seed)
        self._hours: list[datetime] = self._build_hourly_range()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_hourly_range(self) -> list[datetime]:
        start = _dt(self.START_DATE)
        return [start + timedelta(hours=h) for h in range(120)]

    def _noise(self, base: float, pct: float = 0.02) -> float:
        return base * (1 + self._rng.uniform(-pct, pct))

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 1  –  Macro-Geopolitical Events
    # ─────────────────────────────────────────────────────────────────────────

    def gen_layer1_macro_geopolitical(self) -> list[dict]:
        events: list[MacroGeopoliticalEvent] = [
            MacroGeopoliticalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-11", "22:30:00"),
                date_str="2020-03-11T22:30:00Z",
                title="WHO Declares COVID-19 a Global Pandemic",
                description=(
                    "The World Health Organization officially declared the COVID-19 "
                    "outbreak a global pandemic, the first since H1N1 in 2009. The "
                    "announcement triggered immediate sell-offs across all risk assets."
                ),
                source="WHO",
                event_type="pandemic",
                region="global",
                asset_class_impact=["crypto", "equities", "commodities", "bonds"],
                impact_score=-0.95,
                tags=["covid19", "pandemic", "risk_off", "black_swan"],
            ),
            MacroGeopoliticalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-11", "23:00:00"),
                date_str="2020-03-11T23:00:00Z",
                title="Trump Announces 30-Day Travel Ban from Europe",
                description=(
                    "President Trump announced a 30-day travel ban on arrivals from "
                    "Schengen Area countries, shocking global markets. Airlines, hotels "
                    "and hospitality stocks collapsed. Risk assets including crypto "
                    "accelerated their sell-off."
                ),
                source="White House",
                event_type="policy",
                region="USA",
                asset_class_impact=["equities", "commodities", "crypto"],
                impact_score=-0.85,
                tags=["covid19", "travel_ban", "trump", "risk_off"],
            ),
            MacroGeopoliticalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "14:00:00"),
                date_str="2020-03-12T14:00:00Z",
                title="S&P 500 Falls 9.5% – Largest Single-Day Drop Since 1987",
                description=(
                    "US equity markets recorded their worst single-day loss since "
                    "Black Monday 1987 as pandemic fears overwhelmed circuit-breaker "
                    "mechanisms.  Three Level 1 circuit breakers triggered during the "
                    "session. Oil simultaneously crashed on OPEC+ supply-war escalation, "
                    "creating a dual deflationary shock."
                ),
                source="NYSE / Bloomberg",
                event_type="crisis",
                region="USA",
                asset_class_impact=["equities", "crypto", "commodities"],
                impact_score=-0.98,
                tags=["black_thursday", "equity_crash", "circuit_breaker", "1987"],
            ),
            MacroGeopoliticalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "13:30:00"),
                date_str="2020-03-12T13:30:00Z",
                title="NYSE Circuit Breaker Level 1 Triggered (−7% S&P)",
                description=(
                    "Trading halted for 15 minutes after the S&P 500 fell 7% at "
                    "market open. Second halt triggered at −13%. Halt mechanism "
                    "prevented liquidity from entering markets and amplified the "
                    "correlated crypto sell-off."
                ),
                source="NYSE",
                event_type="crisis",
                region="USA",
                asset_class_impact=["equities", "crypto"],
                impact_score=-0.90,
                tags=["circuit_breaker", "halt", "nyse", "black_thursday"],
            ),
            MacroGeopoliticalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "14:30:00"),
                date_str="2020-03-12T14:30:00Z",
                title="Fed Announces $1.5T Repo Injection – Markets Unimpressed",
                description=(
                    "The New York Fed announced $1.5 trillion in repo operations over "
                    "the following weeks to address short-term dollar funding stress. "
                    "Markets initially rallied then reversed, with S&P eventually "
                    "closing −9.5 %. The repo bazooka failed to arrest the cascade."
                ),
                source="Federal Reserve Bank of New York",
                event_type="policy",
                region="USA",
                asset_class_impact=["equities", "bonds", "crypto"],
                impact_score=-0.30,
                tags=["fed", "repo", "qe", "liquidity_injection"],
            ),
            MacroGeopoliticalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-13", "13:30:00"),
                date_str="2020-03-13T13:30:00Z",
                title="S&P 500 Rebounds +9.3% on Trump 'National Emergency' Declaration",
                description=(
                    "President Trump declared a national emergency over COVID-19, "
                    "unlocking $50B in federal aid and triggering the largest single-day "
                    "S&P gain since October 2008.  Risk assets including crypto rebounded "
                    "sharply as liquidity fears temporarily eased."
                ),
                source="White House / Bloomberg",
                event_type="policy",
                region="USA",
                asset_class_impact=["equities", "crypto"],
                impact_score=0.70,
                tags=["trump", "national_emergency", "equity_rebound", "relief_rally"],
            ),
            MacroGeopoliticalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-15", "18:00:00"),
                date_str="2020-03-15T18:00:00Z",
                title="Federal Reserve Emergency Cut: −100 bps to 0–0.25 % + QE Restart",
                description=(
                    "In an extraordinary Sunday evening announcement the FOMC cut the "
                    "federal funds target rate by 100 basis points to 0–0.25 %, its "
                    "first inter-meeting emergency cut since 2008, and announced $700B "
                    "in new QE purchases.  Crypto markets initially sold off on the "
                    "'sell the news' reaction before stabilising."
                ),
                source="Federal Reserve",
                event_type="policy",
                region="USA",
                asset_class_impact=["bonds", "equities", "crypto", "commodities"],
                impact_score=0.40,
                tags=["fed", "rate_cut", "emergency_cut", "qe", "zirp"],
            ),
            MacroGeopoliticalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "05:00:00"),
                date_str="2020-03-12T05:00:00Z",
                title="Saudi Arabia Floods Oil Market – Brent Crude Drops to $31",
                description=(
                    "Following the breakdown of OPEC+ talks Russia and Saudi Arabia "
                    "both announced record production increases. Brent crude fell below "
                    "$31/bbl, its lowest since 2016. Oil-correlated assets collapsed "
                    "and the deflationary shock amplified Bitcoin's leveraged-unwind."
                ),
                source="Reuters",
                event_type="crisis",
                region="global",
                asset_class_impact=["commodities", "equities", "crypto"],
                impact_score=-0.80,
                tags=["oil_crash", "opec", "saudi", "russia", "deflation"],
            ),
        ]
        return [asdict(e) for e in events]

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 2  –  Crypto-Specific Project News
    # ─────────────────────────────────────────────────────────────────────────

    def gen_layer2_crypto_news(self) -> list[dict]:
        events: list[CryptoNewsEvent] = [
            CryptoNewsEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "08:00:00"),
                date_str="2020-03-12T08:00:00Z",
                coin_ticker="ETH",
                project_name="MakerDAO",
                category="protocol_upgrade",
                headline="MakerDAO Faces Systemic Undercollateralisation as ETH Crashes Below Liquidation Thresholds",
                body=(
                    "As ETH dropped through critical collateralisation thresholds, "
                    "MakerDAO's Keeper network became overwhelmed.  Network gas fees "
                    "spiked to >200 Gwei, preventing most keeper bots from executing "
                    "timely liquidation bids.  Zero-DAI bids began winning collateral "
                    "auctions, threatening the protocol's $100M+ in backed DAI."
                ),
                source="MakerDAO Forum",
                sentiment_score=-0.92,
                impact_tags=["defi_crisis", "undercollateral", "gas_spike", "maker"],
            ),
            CryptoNewsEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "09:15:00"),
                date_str="2020-03-12T09:15:00Z",
                coin_ticker="ETH",
                project_name="MakerDAO",
                category="hack",
                headline="Zero-Bid Auction Exploit: MakerDAO Accrues $8.32M Bad Debt in 2.5 Hours",
                body=(
                    "Due to network congestion and keeper failure, approximately $8.32M "
                    "worth of ETH collateral was liquidated for zero DAI.  The attackers "
                    "obtained ETH collateral for free, leaving MakerDAO with an "
                    "equivalent hole in its surplus buffer.  Emergency governance was "
                    "invoked and a MKR auction planned to cover the deficit."
                ),
                source="MakerDAO Governance",
                sentiment_score=-0.98,
                impact_tags=["exploit", "bad_debt", "zero_bid", "mkr_auction"],
            ),
            CryptoNewsEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "15:30:00"),
                date_str="2020-03-12T15:30:00Z",
                coin_ticker="BTC",
                project_name="BitMEX",
                category="market",
                headline="BitMEX XBTUSD Perpetual Halted Briefly Amid Extreme Volatility Spike",
                body=(
                    "BitMEX, then the largest crypto derivatives venue by open interest, "
                    "experienced an extended processing delay during peak volatility hours. "
                    "Traders were unable to add margin or close positions for approximately "
                    "25 minutes, contributing to cascading liquidations and preventing "
                    "demand-side intervention at the bottom."
                ),
                source="BitMEX Blog",
                sentiment_score=-0.88,
                impact_tags=["exchange_failure", "liquidation_cascade", "bitmex"],
            ),
            CryptoNewsEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "16:00:00"),
                date_str="2020-03-12T16:00:00Z",
                coin_ticker="MULTI",
                project_name="Multiple Exchanges",
                category="market",
                headline="Over $1.2B Liquidated on Crypto Derivatives in 24 Hours – Record at Time",
                body=(
                    "Coinglass (then Skew) data confirmed that liquidations across BitMEX, "
                    "Binance, OKEx, Deribit, and Huobi exceeded $1.2B within a 24-hour "
                    "window, shattering the previous record.  Long liquidations accounted "
                    "for ~78 % of the total, confirming the one-sided nature of the unwind."
                ),
                source="Skew Analytics",
                sentiment_score=-0.95,
                impact_tags=["liquidation_record", "long_wipeout", "derivatives"],
            ),
            CryptoNewsEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "20:00:00"),
                date_str="2020-03-12T20:00:00Z",
                coin_ticker="BTC",
                project_name="Bitcoin",
                category="market",
                headline="Bitcoin Marks its Worst 24h Percentage Drop Since 2013 at −53%",
                body=(
                    "BTC/USD reached an intraday low of $3,782 on Bitstamp, representing "
                    "a 53 % decline from the previous day open at ~$7,910. "
                    "On-chain data showed exchange inflows at an all-time high as "
                    "retail participants panic-sold.  Institutional buyers were largely "
                    "absent, with no GBTC premium recovery until March 13."
                ),
                source="Bitstamp / Glassnode",
                sentiment_score=-0.99,
                impact_tags=["price_record", "capitulation", "panic_sell"],
            ),
            CryptoNewsEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-13", "12:00:00"),
                date_str="2020-03-13T12:00:00Z",
                coin_ticker="BTC",
                project_name="Bitcoin",
                category="market",
                headline="Bitcoin Recovers to $5,500 – 'Buy-the-Dip' Narrative Begins",
                body=(
                    "Following the Black Thursday capitulation, Bitcoin recovered from "
                    "$3,782 to ~$5,500 within 24 hours as spot buyers returned. "
                    "Grayscale Bitcoin Trust (GBTC) resumed large accumulation. "
                    "Crypto Twitter began circulating 'generational buying opportunity' narratives."
                ),
                source="CoinDesk",
                sentiment_score=0.55,
                impact_tags=["relief_rally", "accumulation", "grayscale"],
            ),
            CryptoNewsEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-15", "19:30:00"),
                date_str="2020-03-15T19:30:00Z",
                coin_ticker="BTC",
                project_name="Bitcoin",
                category="market",
                headline="BTC Sells Off After Fed Emergency Cut as 'Sell the News' Hits Risk Assets",
                body=(
                    "Despite the Federal Reserve's surprise Sunday night −100 bps cut "
                    "to zero and $700B QE announcement, Bitcoin initially sold off from "
                    "$5,500 to $5,100, demonstrating its correlation with TradFi risk-off "
                    "sentiment during liquidity crises.  The 'digital gold' narrative took "
                    "a temporary credibility hit."
                ),
                source="CoinTelegraph",
                sentiment_score=-0.40,
                impact_tags=["sell_the_news", "fed_cut", "correlation", "risk_off"],
            ),
            CryptoNewsEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-11", "10:00:00"),
                date_str="2020-03-11T10:00:00Z",
                coin_ticker="BTC",
                project_name="Bitcoin",
                category="market",
                headline="Bitcoin Miner Capitulation Signals Begin Appearing in Hash Ribbon",
                body=(
                    "On-chain analysts flagged early miner capitulation signals in the "
                    "Hash Ribbon indicator, with the 30-day moving average of hash rate "
                    "crossing below the 60-day.  This historically marks deep bear market "
                    "floors and was interpreted as a long-term accumulation signal by "
                    "on-chain analysts despite short-term price collapse."
                ),
                source="Glassnode",
                sentiment_score=-0.30,
                impact_tags=["miner_capitulation", "hash_ribbon", "on_chain"],
            ),
        ]
        return [asdict(e) for e in events]

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 3  –  On-Chain Analytics  (Parquet, hourly)
    # ─────────────────────────────────────────────────────────────────────────

    def gen_layer3_on_chain(self) -> pd.DataFrame:
        rows: list[dict] = []

        # BTC hash rate context: approx 107 EH/s in early 2020
        BASE_HASH_RATE = 107.0
        # Difficulty approx 15.97T in Mar 2020
        BASE_DIFFICULTY = 15.97e12

        for i, dt in enumerate(self._hours):
            ts = _ms(dt)
            day = dt.day   # 11, 12, 13, 14, 15
            btc_price = self.BTC_HOURLY[i]
            eth_price = self.ETH_HOURLY[i]

            # Exchange inflow spikes dramatically on Mar 12
            if day == 12:
                inflow_btc = self._noise(8_000 + max(0, (18 - dt.hour) * 1_200), 0.15)
                inflow_eth = self._noise(65_000 + max(0, (18 - dt.hour) * 10_000), 0.15)
                whale_cnt  = self._rng.randint(180, 420)
                whale_vol  = self._noise(350_000_000, 0.20)
                sopr_btc   = self._noise(0.72, 0.05)   # sellers capitulating below cost
                sopr_eth   = self._noise(0.68, 0.05)
            elif day == 11:
                inflow_btc = self._noise(2_200, 0.10)
                inflow_eth = self._noise(18_000, 0.10)
                whale_cnt  = self._rng.randint(60, 100)
                whale_vol  = self._noise(90_000_000, 0.15)
                sopr_btc   = self._noise(0.92, 0.04)
                sopr_eth   = self._noise(0.90, 0.04)
            else:  # 13-15 recovery
                inflow_btc = self._noise(1_600 - (day - 13) * 100, 0.08)
                inflow_eth = self._noise(12_000 - (day - 13) * 800, 0.08)
                whale_cnt  = self._rng.randint(40, 80)
                whale_vol  = self._noise(60_000_000, 0.12)
                sopr_btc   = self._noise(0.85 + (day - 13) * 0.04, 0.03)
                sopr_eth   = self._noise(0.80 + (day - 13) * 0.04, 0.03)

            outflow_btc = self._noise(inflow_btc * 0.35, 0.10)
            outflow_eth = self._noise(inflow_eth * 0.35, 0.10)

            # Smart money actually accumulates post-crash
            smart_btc = (
                -self._noise(abs(inflow_btc - outflow_btc) * btc_price * 0.6, 0.10)
                if day <= 12 else
                self._noise(abs(outflow_btc - inflow_btc) * btc_price * 0.4, 0.10)
            )
            smart_eth = (
                -self._noise(abs(inflow_eth - outflow_eth) * eth_price * 0.6, 0.10)
                if day <= 12 else
                self._noise(abs(outflow_eth - inflow_eth) * eth_price * 0.4, 0.10)
            )

            nvt_btc = self._noise((btc_price * 18_600_000) / max(1, inflow_btc * btc_price * 24), 0.05)
            nvt_eth = self._noise((eth_price * 110_000_000) / max(1, inflow_eth * eth_price * 24), 0.05)
            active_btc = self._rng.randint(600_000, 950_000) if day == 12 else self._rng.randint(450_000, 700_000)
            active_eth = self._rng.randint(280_000, 420_000) if day == 12 else self._rng.randint(200_000, 320_000)

            hash_rate = self._noise(BASE_HASH_RATE * (1 - 0.005 * max(0, day - 12)), 0.02)

            rows.append({
                "timestamp_utc": ts,
                "symbol": "BTC",
                "exchange_inflow": round(inflow_btc, 2),
                "exchange_outflow": round(outflow_btc, 2),
                "net_exchange_flow": round(inflow_btc - outflow_btc, 2),
                "whale_transfer_count": whale_cnt,
                "whale_transfer_volume_usd": round(whale_vol, 0),
                "miner_hash_rate_eh_s": round(hash_rate, 3),
                "network_difficulty": round(BASE_DIFFICULTY * self._noise(1.0, 0.005), 0),
                "active_addresses": active_btc,
                "nvt_ratio": round(nvt_btc, 2),
                "sopr": round(sopr_btc, 4),
                "smart_money_net_flow_usd": round(smart_btc, 0),
            })
            rows.append({
                "timestamp_utc": ts,
                "symbol": "ETH",
                "exchange_inflow": round(inflow_eth, 2),
                "exchange_outflow": round(outflow_eth, 2),
                "net_exchange_flow": round(inflow_eth - outflow_eth, 2),
                "whale_transfer_count": int(whale_cnt * 0.6),
                "whale_transfer_volume_usd": round(whale_vol * 0.4, 0),
                "miner_hash_rate_eh_s": float("nan"),   # Ethereum PoW TH/s separate metric
                "network_difficulty": float("nan"),
                "active_addresses": active_eth,
                "nvt_ratio": round(nvt_eth, 2),
                "sopr": round(sopr_eth, 4),
                "smart_money_net_flow_usd": round(smart_eth, 0),
            })

        return pd.DataFrame(rows).astype({
            k: v for k, v in ON_CHAIN_COLUMNS.items()
            if k not in ("miner_hash_rate_eh_s", "network_difficulty", "nvt_ratio", "sopr",
                         "smart_money_net_flow_usd", "exchange_inflow", "exchange_outflow",
                         "net_exchange_flow", "whale_transfer_volume_usd")
        })

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 4  –  Social-Media Hype & Sentiment  (JSONL, hourly)
    # ─────────────────────────────────────────────────────────────────────────

    def gen_layer4_social_sentiment(self) -> list[dict]:
        platforms = ["twitter", "reddit_r_bitcoin", "reddit_r_crypto", "telegram"]
        coins     = ["BTC", "ETH"]
        records: list[SocialSentimentSnapshot] = []

        # Sample vocabulary pools per sentiment state
        crash_terms   = ["crash", "panic", "sell", "capitulation", "rekt", "liquidated",
                         "black thursday", "covid", "bear market", "wtf", "hodl rekt"]
        bounce_terms  = ["buy the dip", "accumulate", "bottom", "recovery", "hodl",
                         "generational buy", "golden opportunity", "dollar cost average"]
        neutral_terms = ["bitcoin", "crypto", "market", "price", "trading", "news"]

        for i, dt in enumerate(self._hours):
            ts  = _ms(dt)
            day = dt.day
            btc_price = self.BTC_HOURLY[i]

            # Sentiment dynamics
            if day == 12 and 6 <= dt.hour <= 20:
                base_sent = -0.85
                vol_pct   = 98.0
                top_terms = crash_terms
                bull_frac, bear_frac = 0.05, 0.78
            elif day == 12:
                base_sent = -0.70
                vol_pct   = 90.0
                top_terms = crash_terms[:6]
                bull_frac, bear_frac = 0.08, 0.70
            elif day == 11 and dt.hour >= 20:
                base_sent = -0.55
                vol_pct   = 80.0
                top_terms = crash_terms[:5] + neutral_terms[:3]
                bull_frac, bear_frac = 0.12, 0.60
            elif day == 13:
                base_sent = 0.10
                vol_pct   = 75.0
                top_terms = bounce_terms[:6] + neutral_terms[:2]
                bull_frac, bear_frac = 0.45, 0.30
            elif day >= 14:
                base_sent = 0.05
                vol_pct   = 60.0
                top_terms = bounce_terms[:4] + neutral_terms[:4]
                bull_frac, bear_frac = 0.40, 0.32
            else:
                base_sent = -0.15
                vol_pct   = 50.0
                top_terms = neutral_terms
                bull_frac, bear_frac = 0.35, 0.30

            for coin in coins:
                for platform in platforms:
                    mult = {"twitter": 1.5, "reddit_r_bitcoin": 1.0,
                            "reddit_r_crypto": 0.8, "telegram": 0.6}[platform]
                    raw = self._rng.randint(int(3000 * mult * (vol_pct / 100)),
                                            int(8000 * mult * (vol_pct / 100)))
                    sent = max(-1.0, min(1.0, base_sent + self._rng.uniform(-0.08, 0.08)))
                    bull = int(raw * bull_frac)
                    bear = int(raw * bear_frac)
                    neut = raw - bull - bear

                    records.append(SocialSentimentSnapshot(
                        snapshot_id=_uid(),
                        timestamp_utc=ts,
                        coin_ticker=coin,
                        platform=platform,
                        mention_count_1h=raw,
                        sentiment_score_1h=round(sent, 4),
                        volume_percentile_30d=round(vol_pct + self._rng.uniform(-3, 3), 2),
                        top_terms=self._rng.sample(top_terms, min(5, len(top_terms))),
                        raw_sample_count=raw,
                        bullish_count=bull,
                        bearish_count=bear,
                        neutral_count=max(0, neut),
                    ))

        return [asdict(r) for r in records]

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 5  –  TradFi Macro Liquidity  (Parquet, daily + events)
    # ─────────────────────────────────────────────────────────────────────────

    def gen_layer5_tradfi_macro(self) -> pd.DataFrame:
        # Daily close data (market days only; weekend entries mark close from Friday)
        rows = [
            # Mar 11  – S&P enters bear market territory
            {
                "timestamp_utc":       _ts("2020-03-11", "21:00:00"),  # 4 PM ET close
                "resolution":          "daily",
                "fed_funds_rate_bps":  125,   # already cut 50bps on Mar 3
                "fed_rate_change_bps": 0,
                "is_fomc_day":         False,
                "is_cpi_release_day":  False,
                "cpi_yoy_pct":         float("nan"),
                "ppi_yoy_pct":         float("nan"),
                "dxy_close":           96.26,
                "dxy_change_pct":      0.41,
                "sp500_close":         2741.38,
                "sp500_change_pct":    -4.89,
                "vix_close":           54.46,
                "nasdaq_close":        7952.05,
                "nasdaq_change_pct":   -4.70,
                "us_10y_yield":        0.871,
            },
            # Mar 12  – Black Thursday
            {
                "timestamp_utc":       _ts("2020-03-12", "21:00:00"),
                "resolution":          "daily",
                "fed_funds_rate_bps":  125,
                "fed_rate_change_bps": 0,
                "is_fomc_day":         False,
                "is_cpi_release_day":  False,
                "cpi_yoy_pct":         float("nan"),
                "ppi_yoy_pct":         float("nan"),
                "dxy_close":           97.45,
                "dxy_change_pct":      1.24,
                "sp500_close":         2480.64,
                "sp500_change_pct":    -9.51,
                "vix_close":           75.47,
                "nasdaq_close":        6904.59,
                "nasdaq_change_pct":   -9.43,
                "us_10y_yield":        0.707,
            },
            # Mar 13  – Biggest single-day S&P bounce since 2008
            {
                "timestamp_utc":       _ts("2020-03-13", "21:00:00"),
                "resolution":          "daily",
                "fed_funds_rate_bps":  125,
                "fed_rate_change_bps": 0,
                "is_fomc_day":         False,
                "is_cpi_release_day":  False,
                "cpi_yoy_pct":         float("nan"),
                "ppi_yoy_pct":         float("nan"),
                "dxy_close":           98.89,
                "dxy_change_pct":      1.48,
                "sp500_close":         2711.02,
                "sp500_change_pct":    9.29,
                "vix_close":           57.83,
                "nasdaq_close":        7874.88,
                "nasdaq_change_pct":   14.00,
                "us_10y_yield":        0.940,
            },
            # Mar 14 (Saturday) – carry-forward of Friday close
            {
                "timestamp_utc":       _ts("2020-03-14", "00:00:00"),
                "resolution":          "daily",
                "fed_funds_rate_bps":  125,
                "fed_rate_change_bps": 0,
                "is_fomc_day":         False,
                "is_cpi_release_day":  False,
                "cpi_yoy_pct":         float("nan"),
                "ppi_yoy_pct":         float("nan"),
                "dxy_close":           98.89,
                "dxy_change_pct":      0.00,
                "sp500_close":         2711.02,
                "sp500_change_pct":    0.00,
                "vix_close":           57.83,
                "nasdaq_close":        7874.88,
                "nasdaq_change_pct":   0.00,
                "us_10y_yield":        0.940,
            },
            # Mar 15 (Sunday) – FOMC emergency cut event
            {
                "timestamp_utc":       _ts("2020-03-15", "18:00:00"),
                "resolution":          "event",
                "fed_funds_rate_bps":  25,     # cut from 125 → 25 bps
                "fed_rate_change_bps": -100,
                "is_fomc_day":         True,
                "is_cpi_release_day":  False,
                "cpi_yoy_pct":         float("nan"),
                "ppi_yoy_pct":         float("nan"),
                "dxy_close":           98.89,
                "dxy_change_pct":      0.00,
                "sp500_close":         2711.02,  # futures not yet open
                "sp500_change_pct":    0.00,
                "vix_close":           57.83,
                "nasdaq_close":        7874.88,
                "nasdaq_change_pct":   0.00,
                "us_10y_yield":        0.740,
            },
        ]
        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 6  –  Derivatives Microstructure  (Parquet, hourly)
    # ─────────────────────────────────────────────────────────────────────────

    def gen_layer6_derivatives(self) -> pd.DataFrame:
        rows: list[dict] = []

        # BTC OI in USD: approx $1.5B before crash, collapses to ~$500M post-crash
        SYMBOLS = [
            ("BTCUSDT", self.BTC_HOURLY, 1_500_000_000, 1.0),
            ("ETHUSDT", self.ETH_HOURLY,   180_000_000, 0.15),
        ]

        for sym, prices, base_oi, oi_scale in SYMBOLS:
            prev_oi = base_oi
            for i, dt in enumerate(self._hours):
                ts    = _ms(dt)
                day   = dt.day
                price = prices[i]

                # OI dynamics: collapses on crash, slowly rebuilds
                if day == 12 and 6 <= dt.hour <= 20:
                    oi_mult = 1.0 - 0.04 * (dt.hour - 6)   # collapses hourly
                elif day == 12:
                    oi_mult = 0.35
                elif day == 11:
                    oi_mult = 1.0 - 0.01 * dt.hour
                else:
                    oi_mult = 0.35 + 0.03 * (i - 48)   # slow rebuild after crash
                oi = max(base_oi * 0.30, base_oi * min(1.0, oi_mult)) * oi_scale
                oi = self._noise(oi, 0.05)
                oi_change = oi - prev_oi
                prev_oi = oi

                # Funding rate: strongly negative during crash (bears dominate)
                if day == 12 and 6 <= dt.hour <= 20:
                    funding = self._noise(-0.003, 0.30)   # -0.30% per 8h
                elif day == 12:
                    funding = self._noise(-0.002, 0.20)
                elif day == 11 and dt.hour >= 20:
                    funding = self._noise(-0.0005, 0.20)
                elif day >= 13:
                    funding = self._noise(0.0001 * (day - 12), 0.30)
                else:
                    funding = self._noise(0.0001, 0.20)

                # Liquidations: enormous on Mar 12
                if day == 12 and 8 <= dt.hour <= 18:
                    long_liq  = self._noise(55_000_000 * oi_scale, 0.30)
                    short_liq = self._noise(3_000_000  * oi_scale, 0.30)
                elif day == 12:
                    long_liq  = self._noise(12_000_000 * oi_scale, 0.20)
                    short_liq = self._noise(1_500_000  * oi_scale, 0.20)
                elif day == 11 and dt.hour >= 20:
                    long_liq  = self._noise(4_000_000 * oi_scale, 0.20)
                    short_liq = self._noise(800_000   * oi_scale, 0.20)
                else:
                    long_liq  = self._noise(800_000  * oi_scale, 0.25)
                    short_liq = self._noise(600_000  * oi_scale, 0.25)

                total_liq = long_liq + short_liq
                ratio     = long_liq / total_liq if total_liq > 0 else 0.5

                # Basis: negative during crash (futures trade below spot = contango flip)
                if day == 12 and 8 <= dt.hour <= 18:
                    basis = self._noise(-0.015, 0.30)
                else:
                    basis = self._noise(0.002, 0.40)

                rows.append({
                    "timestamp_utc":          ts,
                    "symbol":                 sym,
                    "funding_rate":           round(funding, 6),
                    "oi_usd":                 round(oi, 0),
                    "oi_change_usd":          round(oi_change, 0),
                    "long_liquidations_usd":  round(long_liq, 0),
                    "short_liquidations_usd": round(short_liq, 0),
                    "total_liquidations_usd": round(total_liq, 0),
                    "liq_long_short_ratio":   round(ratio, 4),
                    "options_max_pain_usd":   float("nan"),
                    "put_call_ratio":         float("nan"),
                    "basis_pct":              round(basis, 6),
                })

        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 7  –  KOL Footprints
    # ─────────────────────────────────────────────────────────────────────────

    def gen_layer7_kol_footprints(self) -> list[dict]:
        kols: list[KOLFootprint] = [
            KOLFootprint(
                footprint_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "03:00:00"),
                kol_id="arthur_hayes",
                kol_name="Arthur Hayes",
                kol_role="ceo",
                platform="twitter",
                content=(
                    "The market is puking.  Our liquidation engine is working as designed. "
                    "XBTUSD funding has gone deeply negative.  This is what forced "
                    "deleveraging looks like.  Stay safe out there."
                ),
                coins_mentioned=["BTC"],
                estimated_reach=2_100_000,
                follower_count=740_000,
                market_reaction_1h_pct=-3.2,
                tags=["liquidation", "deleveraging", "bitmex"],
                sentiment="bearish",
            ),
            KOLFootprint(
                footprint_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "10:00:00"),
                kol_id="barry_silbert",
                kol_name="Barry Silbert",
                kol_role="investor",
                platform="twitter",
                content=(
                    "This is a generational buying opportunity for Bitcoin. "
                    "Grayscale is buying.  DCG is buying.  This is the most "
                    "important asymmetric bet of the decade. $BTC"
                ),
                coins_mentioned=["BTC"],
                estimated_reach=1_500_000,
                follower_count=310_000,
                market_reaction_1h_pct=2.1,
                tags=["accumulation", "grayscale", "institutional"],
                sentiment="bullish",
            ),
            KOLFootprint(
                footprint_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "13:00:00"),
                kol_id="cz_binance",
                kol_name="Changpeng Zhao (CZ)",
                kol_role="ceo",
                platform="twitter",
                content=(
                    "Binance is fully operational.  We are seeing record trading volume. "
                    "Our systems are healthy.  Focus on the long-term fundamentals. "
                    "#BNB #BTC #crypto"
                ),
                coins_mentioned=["BTC", "BNB"],
                estimated_reach=5_200_000,
                follower_count=2_100_000,
                market_reaction_1h_pct=1.5,
                tags=["exchange_health", "reassurance", "binance"],
                sentiment="neutral",
            ),
            KOLFootprint(
                footprint_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "16:00:00"),
                kol_id="bitmex_official",
                kol_name="BitMEX Official",
                kol_role="ceo",
                platform="twitter",
                content=(
                    "[SYSTEM UPDATE] BitMEX experienced infrastructure issues between "
                    "14:45 UTC and 15:10 UTC due to DDoS-like load.  All positions were "
                    "preserved.  We are investigating and will publish a post-mortem. "
                    "We apologise for any inconvenience."
                ),
                coins_mentioned=["BTC"],
                estimated_reach=3_100_000,
                follower_count=890_000,
                market_reaction_1h_pct=-5.8,
                tags=["downtime", "ddo", "bitmex", "trust_hit"],
                sentiment="fud",
            ),
            KOLFootprint(
                footprint_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "19:30:00"),
                kol_id="ryan_selkis",
                kol_name="Ryan Selkis",
                kol_role="analyst",
                platform="twitter",
                content=(
                    "Brutal.  BTC -53 % intraday.  ETH -60 % as MakerDAO implodes.  "
                    "The 'uncorrelated asset' narrative is completely dead.  "
                    "The question now: can any DeFi protocol survive a proper bear market?"
                ),
                coins_mentioned=["BTC", "ETH", "DAI", "MKR"],
                estimated_reach=900_000,
                follower_count=180_000,
                market_reaction_1h_pct=None,
                tags=["defi_crisis", "correlation", "bear_market"],
                sentiment="bearish",
            ),
            KOLFootprint(
                footprint_id=_uid(),
                timestamp_utc=_ts("2020-03-13", "09:00:00"),
                kol_id="plan_b",
                kol_name="PlanB",
                kol_role="analyst",
                platform="twitter",
                content=(
                    "S2F model intact.  The halving is in 60 days.  Every crash in "
                    "Bitcoin's history has been followed by new all-time highs.  "
                    "This is the last time you'll see sub-$6K Bitcoin. "
                    "LFG $BTC"
                ),
                coins_mentioned=["BTC"],
                estimated_reach=1_800_000,
                follower_count=650_000,
                market_reaction_1h_pct=4.3,
                tags=["s2f", "halving", "accumulation", "model"],
                sentiment="bullish",
            ),
            KOLFootprint(
                footprint_id=_uid(),
                timestamp_utc=_ts("2020-03-13", "17:00:00"),
                kol_id="willy_woo",
                kol_name="Willy Woo",
                kol_role="analyst",
                platform="twitter",
                content=(
                    "On-chain NVT Signal is at its lowest since December 2018 bottom. "
                    "Entity-adjusted transaction volume hit all-time highs yesterday as "
                    "weak hands capitulated.  This is textbook on-chain bottom formation. "
                    "Accumulating here."
                ),
                coins_mentioned=["BTC"],
                estimated_reach=1_100_000,
                follower_count=400_000,
                market_reaction_1h_pct=2.9,
                tags=["nvt", "on_chain", "bottom", "accumulation"],
                sentiment="bullish",
            ),
            KOLFootprint(
                footprint_id=_uid(),
                timestamp_utc=_ts("2020-03-15", "19:00:00"),
                kol_id="anthony_pompliano",
                kol_name="Anthony Pompliano",
                kol_role="influencer",
                platform="twitter",
                content=(
                    "The Fed just cut rates to zero and launched $700B QE.  "
                    "Bitcoin is the only money that cannot be debased by a central bank.  "
                    "If you don't own Bitcoin after tonight, you may not understand what "
                    "is happening to the global financial system.  #Bitcoin"
                ),
                coins_mentioned=["BTC"],
                estimated_reach=4_500_000,
                follower_count=1_600_000,
                market_reaction_1h_pct=-2.1,  # sell the news
                tags=["fed", "qe", "monetary_policy", "debasement"],
                sentiment="bullish",
            ),
        ]
        return [asdict(k) for k in kols]

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 8  –  Regulatory & Legal Milestones
    # ─────────────────────────────────────────────────────────────────────────

    def gen_layer8_regulatory_legal(self) -> list[dict]:
        events: list[RegulatoryLegalEvent] = [
            RegulatoryLegalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-12", "17:00:00"),
                jurisdiction="USA",
                regulator="CFTC",
                category="guidance",
                title="CFTC Issues Statement: Crypto Markets Fall Under Normal Oversight During COVID",
                description=(
                    "The CFTC issued a brief statement confirming that existing oversight "
                    "frameworks for crypto derivatives remain in force during the COVID-19 "
                    "national emergency.  No special exemptions granted.  BitMEX, which "
                    "was serving US clients in violation of CFTC rules, faces increased scrutiny."
                ),
                affected_coins=["BTC", "ETH"],
                outcome="pending",
                severity="medium",
                source_url="https://www.cftc.gov",
            ),
            RegulatoryLegalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-13", "15:00:00"),
                jurisdiction="EU",
                regulator="ECB",
                category="guidance",
                title="ECB: Crypto Assets Not Eligible as QE Collateral – COVID Relief Measures",
                description=(
                    "The European Central Bank confirmed that cryptocurrency assets would not "
                    "be accepted as collateral under its €750B Pandemic Emergency Purchase "
                    "Programme (PEPP), reinforcing the view that crypto remains outside the "
                    "traditional financial safety net during systemic crises."
                ),
                affected_coins=["BTC", "ETH"],
                outcome="approved",
                severity="low",
                source_url="https://www.ecb.europa.eu",
            ),
            RegulatoryLegalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-11", "16:00:00"),
                jurisdiction="USA",
                regulator="SEC",
                category="enforcement",
                title="SEC Delays ETF Decision Window Due to Market Conditions",
                description=(
                    "The SEC extended its review period for two pending spot Bitcoin ETF "
                    "applications citing extraordinary market conditions.  This continued "
                    "a pattern of ETF rejections that began in 2018 and would persist "
                    "until BlackRock's approval in January 2024."
                ),
                affected_coins=["BTC"],
                outcome="pending",
                severity="medium",
                source_url="https://www.sec.gov",
            ),
            RegulatoryLegalEvent(
                event_id=_uid(),
                timestamp_utc=_ts("2020-03-15", "20:00:00"),
                jurisdiction="USA",
                regulator="FinCEN",
                category="guidance",
                title="FinCEN Confirms AML/KYC Requirements Apply to Crypto Exchanges During Emergency",
                description=(
                    "FinCEN issued guidance confirming that cryptocurrency money service "
                    "businesses must maintain full AML and KYC compliance without exception "
                    "during the COVID-19 national emergency.  Stablecoin issuers specifically "
                    "called out as requiring vigilant monitoring of unusual transaction volumes."
                ),
                affected_coins=["USDT", "USDC", "DAI"],
                outcome="approved",
                severity="low",
                source_url="https://www.fincen.gov",
            ),
        ]
        return [asdict(e) for e in events]

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 9  –  Systemic Crises, Hacks & Exploits
    # ─────────────────────────────────────────────────────────────────────────

    def gen_layer9_crises_hacks(self) -> list[dict]:
        # MakerDAO Black Thursday timeline + BitMEX capacity event
        events: list[CrisisHackEvent] = [
            # ── MakerDAO – Pre-crisis accumulation ───────────────────────────
            CrisisHackEvent(
                incident_id="makerdao_blackthursday_T-12h",
                timestamp_utc=_ts("2020-03-11", "20:00:00"),
                incident_type="liquidity_crisis",
                protocol="MakerDAO",
                chain="Ethereum",
                amount_lost_usd=None,
                description=(
                    "ETH-backed CDPs approach minimum collateralisation ratio (150%) as "
                    "ETH slides to $155.  Keeper bots begin monitoring closely.  Gas prices "
                    "begin elevating above normal 10 Gwei baseline."
                ),
                stage="pre_attack",
                phase_tag="T-12h",
                severity="medium",
                recovery_status="unrecovered",
            ),
            CrisisHackEvent(
                incident_id="makerdao_blackthursday_T0",
                timestamp_utc=_ts("2020-03-12", "08:00:00"),
                incident_type="liquidity_crisis",
                protocol="MakerDAO",
                chain="Ethereum",
                amount_lost_usd=None,
                description=(
                    "ETH price crashes through 150% collateralisation threshold for "
                    "hundreds of CDPs simultaneously.  Liquidation queue floods the "
                    "Ethereum mempool.  Gas prices spike to 200+ Gwei as keeper bots "
                    "compete for priority.  The auction mechanism is overloaded."
                ),
                stage="attack",
                phase_tag="T+0h",
                severity="critical",
                recovery_status="unrecovered",
            ),
            CrisisHackEvent(
                incident_id="makerdao_blackthursday_T1h",
                timestamp_utc=_ts("2020-03-12", "09:00:00"),
                incident_type="exploit",
                protocol="MakerDAO",
                chain="Ethereum",
                amount_lost_usd=8_320_000,
                description=(
                    "A single keeper operator (address 0x...known as 'vault keeper 1') "
                    "recognises that gas congestion has prevented competitors from bidding.  "
                    "They execute a sequence of zero-DAI bids winning $8.32M in ETH "
                    "collateral at zero cost.  ~1,462 ETH seized for 0 DAI.  "
                    "The MakerDAO surplus buffer is wiped and the protocol accrues bad debt."
                ),
                stage="attack",
                phase_tag="T+1h",
                severity="critical",
                recovery_status="unrecovered",
            ),
            CrisisHackEvent(
                incident_id="makerdao_blackthursday_T3h",
                timestamp_utc=_ts("2020-03-12", "11:00:00"),
                incident_type="liquidity_crisis",
                protocol="MakerDAO",
                chain="Ethereum",
                amount_lost_usd=8_320_000,
                description=(
                    "MakerDAO governance forum emergency thread opened.  Core team confirms "
                    "$8.32M bad debt.  Emergency Executive Vote proposed to raise the "
                    "Stability Fee for ETH-A vaults and prepare for a MKR debt auction "
                    "to recapitalise the protocol.  ETH continues to fall."
                ),
                stage="discovery",
                phase_tag="T+3h",
                severity="critical",
                recovery_status="unrecovered",
            ),
            CrisisHackEvent(
                incident_id="makerdao_blackthursday_T6h",
                timestamp_utc=_ts("2020-03-12", "14:00:00"),
                incident_type="liquidity_crisis",
                protocol="MakerDAO",
                chain="Ethereum",
                amount_lost_usd=8_320_000,
                description=(
                    "Emergency governance vote passes to add USDC as collateral for the "
                    "first time, a controversial centralisation step.  Circuit breaker "
                    "deployed: Liquidation Penalty temporarily removed to reduce gas "
                    "competition.  Price oracle circuit breaker also activated to prevent "
                    "further cascade below $88 ETH."
                ),
                stage="escalation",
                phase_tag="T+6h",
                severity="critical",
                recovery_status="partial",
            ),
            CrisisHackEvent(
                incident_id="makerdao_blackthursday_T72h",
                timestamp_utc=_ts("2020-03-15", "08:00:00"),
                incident_type="liquidity_crisis",
                protocol="MakerDAO",
                chain="Ethereum",
                amount_lost_usd=8_320_000,
                description=(
                    "MakerDAO community reaches rough consensus to proceed with a "
                    "MKR dilution auction to cover the $8.32M bad debt.  "
                    "Flop auction (debt auction) parameters set: 50,000 MKR maximum "
                    "supply increase. Auction date set for March 19, 2020.  "
                    "USDC collateral type formally deployed on-chain."
                ),
                stage="post_mortem",
                phase_tag="T+72h",
                severity="high",
                recovery_status="partial",
            ),
            # ── BitMEX infrastructure event ───────────────────────────────────
            CrisisHackEvent(
                incident_id="bitmex_downtime_2020_03_12",
                timestamp_utc=_ts("2020-03-12", "14:45:00"),
                incident_type="exchange_failure",
                protocol="BitMEX",
                chain="Bitcoin",
                amount_lost_usd=None,
                description=(
                    "BitMEX XBTUSD perpetual market experiences a 25-minute processing "
                    "slowdown attributed to a 'hardware issue'. Traders unable to place "
                    "orders, add margin or close positions. At this point BTC was at ~$5,600 "
                    "falling toward $3,782.  The inability to buy or add margin amplified "
                    "the cascade and removed natural demand-side support."
                ),
                stage="attack",
                phase_tag="T+0h",
                severity="critical",
                recovery_status="full",
            ),
            CrisisHackEvent(
                incident_id="bitmex_downtime_2020_03_12_recovery",
                timestamp_utc=_ts("2020-03-12", "15:10:00"),
                incident_type="exchange_failure",
                protocol="BitMEX",
                chain="Bitcoin",
                amount_lost_usd=None,
                description=(
                    "BitMEX trading resumes.  At this point BTC has reached $4,200 on "
                    "its way to the $3,782 low.  Numerous traders have been liquidated "
                    "during the outage window.  Post-incident review will later claim "
                    "DDoS-like load rather than targeted attack."
                ),
                stage="recovery",
                phase_tag="T+0.4h",
                severity="critical",
                recovery_status="full",
            ),
            # ── Ethereum network congestion ───────────────────────────────────
            CrisisHackEvent(
                incident_id="eth_gas_congestion_blackthursday",
                timestamp_utc=_ts("2020-03-12", "08:30:00"),
                incident_type="liquidity_crisis",
                protocol="Ethereum Network",
                chain="Ethereum",
                amount_lost_usd=None,
                description=(
                    "Ethereum network gas prices spike from 10 Gwei baseline to 200+ Gwei "
                    "as the MakerDAO liquidation cascade floods the mempool.  Multiple DeFi "
                    "protocols (Compound, dYdX, Uniswap) become practically unusable for "
                    "retail participants.  This is an early precursor to EIP-1559 proposals."
                ),
                stage="attack",
                phase_tag="T+0h",
                severity="high",
                recovery_status="full",
            ),
        ]
        return [asdict(e) for e in events]

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 10  –  Retail Fear & Greed + Google Trends  (Parquet, daily)
    # ─────────────────────────────────────────────────────────────────────────

    def gen_layer10_fear_greed(self) -> pd.DataFrame:
        rows = [
            {
                "timestamp_utc":               _ts("2020-03-11"),
                "date_str":                    "2020-03-11",
                "fear_greed_index":            17,
                "fear_greed_label":            "extreme_fear",
                "google_trends_bitcoin":       62.0,
                "google_trends_crypto":        55.0,
                "google_trends_buy_crypto":    41.0,
                "google_trends_ethereum":      48.0,
                "google_trends_bitcoin_crash": 65.0,
                "google_trends_sell_bitcoin":  58.0,
                "retail_search_composite":     54.8,
            },
            {
                "timestamp_utc":               _ts("2020-03-12"),
                "date_str":                    "2020-03-12",
                "fear_greed_index":            8,    # absolute floor – extreme fear
                "fear_greed_label":            "extreme_fear",
                "google_trends_bitcoin":       100.0,
                "google_trends_crypto":        92.0,
                "google_trends_buy_crypto":    55.0,
                "google_trends_ethereum":      78.0,
                "google_trends_bitcoin_crash": 100.0,
                "google_trends_sell_bitcoin":  91.0,
                "retail_search_composite":     86.0,
            },
            {
                "timestamp_utc":               _ts("2020-03-13"),
                "date_str":                    "2020-03-13",
                "fear_greed_index":            11,
                "fear_greed_label":            "extreme_fear",
                "google_trends_bitcoin":       88.0,
                "google_trends_crypto":        80.0,
                "google_trends_buy_crypto":    73.0,
                "google_trends_ethereum":      65.0,
                "google_trends_bitcoin_crash": 72.0,
                "google_trends_sell_bitcoin":  61.0,
                "retail_search_composite":     73.2,
            },
            {
                "timestamp_utc":               _ts("2020-03-14"),
                "date_str":                    "2020-03-14",
                "fear_greed_index":            15,
                "fear_greed_label":            "extreme_fear",
                "google_trends_bitcoin":       75.0,
                "google_trends_crypto":        68.0,
                "google_trends_buy_crypto":    78.0,
                "google_trends_ethereum":      60.0,
                "google_trends_bitcoin_crash": 55.0,
                "google_trends_sell_bitcoin":  47.0,
                "retail_search_composite":     63.8,
            },
            {
                "timestamp_utc":               _ts("2020-03-15"),
                "date_str":                    "2020-03-15",
                "fear_greed_index":            13,
                "fear_greed_label":            "extreme_fear",
                "google_trends_bitcoin":       70.0,
                "google_trends_crypto":        63.0,
                "google_trends_buy_crypto":    82.0,  # FOMO buying dip rising
                "google_trends_ethereum":      55.0,
                "google_trends_bitcoin_crash": 48.0,  # crash term declining
                "google_trends_sell_bitcoin":  39.0,
                "retail_search_composite":     59.5,
            },
        ]
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Generic Window Mock Generator  (non-Black-Thursday windows)
# ─────────────────────────────────────────────────────────────────────────────

class GenericMockGenerator:
    """
    Generates synthetic (placeholder) data for any arbitrary window.
    Data is structurally correct but not historically grounded.
    Use only for system testing / schema validation.
    """

    def __init__(
        self,
        start: datetime,
        end: datetime,
        seed: int = 0,
    ) -> None:
        self.start = start
        self.end   = end
        self._rng  = random.Random(seed)
        np.random.seed(seed)

    def _hourly_range(self) -> list[datetime]:
        hours: list[datetime] = []
        cur = self.start
        while cur <= self.end:
            hours.append(cur)
            cur += timedelta(hours=1)
        return hours

    def gen_layer3_on_chain(self) -> pd.DataFrame:
        rows: list[dict] = []
        for dt in self._hourly_range():
            ts = _ms(dt)
            for sym in ("BTC", "ETH"):
                inflow  = abs(np.random.normal(2000, 400)) if sym == "BTC" else abs(np.random.normal(15000, 3000))
                outflow = inflow * self._rng.uniform(0.3, 0.7)
                rows.append({
                    "timestamp_utc": ts, "symbol": sym,
                    "exchange_inflow": round(inflow, 2),
                    "exchange_outflow": round(outflow, 2),
                    "net_exchange_flow": round(inflow - outflow, 2),
                    "whale_transfer_count": self._rng.randint(30, 120),
                    "whale_transfer_volume_usd": abs(np.random.normal(80_000_000, 20_000_000)),
                    "miner_hash_rate_eh_s": abs(np.random.normal(110, 5)) if sym == "BTC" else float("nan"),
                    "network_difficulty": abs(np.random.normal(16e12, 5e11)) if sym == "BTC" else float("nan"),
                    "active_addresses": self._rng.randint(400_000, 700_000) if sym == "BTC" else self._rng.randint(200_000, 400_000),
                    "nvt_ratio": abs(np.random.normal(35, 8)),
                    "sopr": abs(np.random.normal(1.01, 0.05)),
                    "smart_money_net_flow_usd": np.random.normal(0, 5_000_000),
                })
        return pd.DataFrame(rows)

    def gen_layer5_tradfi_macro(self) -> pd.DataFrame:
        rows: list[dict] = []
        cur_day = self.start.date()
        end_day = self.end.date()
        sp_val  = 4000.0
        dxy_val = 97.0
        while cur_day <= end_day:
            dt = datetime(cur_day.year, cur_day.month, cur_day.day, 21, 0, tzinfo=timezone.utc)
            sp_chg  = np.random.normal(0, 1.2)
            dxy_chg = np.random.normal(0, 0.3)
            sp_val  = sp_val * (1 + sp_chg / 100)
            dxy_val = dxy_val * (1 + dxy_chg / 100)
            rows.append({
                "timestamp_utc":       _ms(dt),
                "resolution":          "daily",
                "fed_funds_rate_bps":  25,
                "fed_rate_change_bps": 0,
                "is_fomc_day":         False,
                "is_cpi_release_day":  False,
                "cpi_yoy_pct":         float("nan"),
                "ppi_yoy_pct":         float("nan"),
                "dxy_close":           round(dxy_val, 2),
                "dxy_change_pct":      round(dxy_chg, 3),
                "sp500_close":         round(sp_val, 2),
                "sp500_change_pct":    round(sp_chg, 3),
                "vix_close":           round(abs(np.random.normal(22, 5)), 2),
                "nasdaq_close":        round(sp_val * 3.2, 2),
                "nasdaq_change_pct":   round(sp_chg * 1.1, 3),
                "us_10y_yield":        round(abs(np.random.normal(1.5, 0.2)), 3),
            })
            cur_day += timedelta(days=1)
        return pd.DataFrame(rows)

    def gen_layer6_derivatives(self) -> pd.DataFrame:
        rows: list[dict] = []
        for dt in self._hourly_range():
            ts = _ms(dt)
            for sym, base_oi in (("BTCUSDT", 5_000_000_000), ("ETHUSDT", 800_000_000)):
                rows.append({
                    "timestamp_utc":          ts,
                    "symbol":                 sym,
                    "funding_rate":           round(np.random.normal(0.0001, 0.0005), 6),
                    "oi_usd":                 abs(np.random.normal(base_oi, base_oi * 0.05)),
                    "oi_change_usd":          np.random.normal(0, base_oi * 0.01),
                    "long_liquidations_usd":  abs(np.random.normal(500_000, 300_000)),
                    "short_liquidations_usd": abs(np.random.normal(450_000, 280_000)),
                    "total_liquidations_usd": abs(np.random.normal(950_000, 550_000)),
                    "liq_long_short_ratio":   self._rng.uniform(0.4, 0.6),
                    "options_max_pain_usd":   float("nan"),
                    "put_call_ratio":         float("nan"),
                    "basis_pct":              round(np.random.normal(0.002, 0.003), 6),
                })
        return pd.DataFrame(rows)

    def gen_layer10_fear_greed(self) -> pd.DataFrame:
        rows: list[dict] = []
        cur_day = self.start.date()
        end_day = self.end.date()
        fg = 50
        while cur_day <= end_day:
            fg = max(5, min(95, fg + self._rng.randint(-8, 8)))
            if fg <= 25:
                label = "extreme_fear" if fg <= 10 else "fear"
            elif fg >= 75:
                label = "extreme_greed" if fg >= 90 else "greed"
            else:
                label = "neutral"
            goog_btc = min(100, abs(np.random.normal(45, 20)))
            rows.append({
                "timestamp_utc":               _ms(datetime(cur_day.year, cur_day.month, cur_day.day, tzinfo=timezone.utc)),
                "date_str":                    cur_day.isoformat(),
                "fear_greed_index":            fg,
                "fear_greed_label":            label,
                "google_trends_bitcoin":       round(goog_btc, 1),
                "google_trends_crypto":        round(goog_btc * 0.9, 1),
                "google_trends_buy_crypto":    round(goog_btc * 0.55, 1),
                "google_trends_ethereum":      round(goog_btc * 0.75, 1),
                "google_trends_bitcoin_crash": round(goog_btc * 0.3, 1),
                "google_trends_sell_bitcoin":  round(goog_btc * 0.25, 1),
                "retail_search_composite":     round(goog_btc * 0.7, 1),
            })
            cur_day += timedelta(days=1)
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(
    data_root: str,
    start: datetime,
    end: datetime,
    scenario: str | None = None,
    mock: bool = True,
) -> dict[str, pathlib.Path]:
    """
    Run the full ingestion pipeline.

    Parameters
    ----------
    data_root : str
        Root directory for the Time Machine dataset.
    start, end : datetime (UTC-aware)
        Window to generate/ingest data for.
    scenario : str | None
        Named preset.  Currently only ``"black_thursday"`` is supported.
    mock : bool
        If True, generate mock data instead of calling live APIs.

    Returns
    -------
    dict mapping layer_key → saved file Path.
    """
    ingestor = LayerIngestor(data_root)
    manifest: dict[str, pathlib.Path] = {}

    window_tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    is_bt = scenario == "black_thursday" or (
        start.date() == datetime(2020, 3, 11, tzinfo=timezone.utc).date()
    )

    if not mock:
        log.warning("Live ingestion not implemented – run with --mock flag.")
        return manifest

    log.info("Generating mock data  |  window=%s  |  scenario=%s", window_tag, scenario or "generic")

    if is_bt:
        gen_bt = BlackThursdayMockGenerator()

        # ── JSONL layers ──────────────────────────────────────────────────────
        manifest["macro_geopolitical"] = ingestor.save_jsonl(
            gen_bt.gen_layer1_macro_geopolitical(),
            "macro_geopolitical",
            f"events_{window_tag}.jsonl",
        )
        manifest["crypto_news"] = ingestor.save_jsonl(
            gen_bt.gen_layer2_crypto_news(),
            "crypto_news",
            f"events_{window_tag}.jsonl",
        )
        manifest["social_sentiment"] = ingestor.save_jsonl(
            gen_bt.gen_layer4_social_sentiment(),
            "social_sentiment",
            f"snapshots_{window_tag}.jsonl",
        )
        manifest["kol_footprints"] = ingestor.save_jsonl(
            gen_bt.gen_layer7_kol_footprints(),
            "kol_footprints",
            f"footprints_{window_tag}.jsonl",
        )
        manifest["regulatory_legal"] = ingestor.save_jsonl(
            gen_bt.gen_layer8_regulatory_legal(),
            "regulatory_legal",
            f"events_{window_tag}.jsonl",
        )
        manifest["crises_hacks"] = ingestor.save_jsonl(
            gen_bt.gen_layer9_crises_hacks(),
            "crises_hacks",
            f"incidents_{window_tag}.jsonl",
        )

        # ── Parquet layers ────────────────────────────────────────────────────
        manifest["on_chain"] = ingestor.save_parquet(
            gen_bt.gen_layer3_on_chain(),
            "on_chain",
            f"BTC_ETH_{window_tag}.parquet",
        )
        manifest["tradfi_macro"] = ingestor.save_parquet(
            gen_bt.gen_layer5_tradfi_macro(),
            "tradfi_macro",
            f"macro_{window_tag}.parquet",
        )
        manifest["derivatives"] = ingestor.save_parquet(
            gen_bt.gen_layer6_derivatives(),
            "derivatives",
            f"BTCUSDT_ETHUSDT_{window_tag}.parquet",
        )
        manifest["fear_greed"] = ingestor.save_parquet(
            gen_bt.gen_layer10_fear_greed(),
            "fear_greed",
            f"fg_{window_tag}.parquet",
        )

    else:
        # Generic window
        gen_g = GenericMockGenerator(start=start, end=end)

        manifest["on_chain"] = ingestor.save_parquet(
            gen_g.gen_layer3_on_chain(),
            "on_chain",
            f"BTC_ETH_{window_tag}.parquet",
        )
        manifest["tradfi_macro"] = ingestor.save_parquet(
            gen_g.gen_layer5_tradfi_macro(),
            "tradfi_macro",
            f"macro_{window_tag}.parquet",
        )
        manifest["derivatives"] = ingestor.save_parquet(
            gen_g.gen_layer6_derivatives(),
            "derivatives",
            f"BTCUSDT_ETHUSDT_{window_tag}.parquet",
        )
        manifest["fear_greed"] = ingestor.save_parquet(
            gen_g.gen_layer10_fear_greed(),
            "fear_greed",
            f"fg_{window_tag}.parquet",
        )

        for layer in ("macro_geopolitical", "crypto_news", "social_sentiment",
                      "kol_footprints", "regulatory_legal", "crises_hacks"):
            manifest[layer] = ingestor.save_jsonl(
                [],
                layer,
                f"events_{window_tag}.jsonl",
            )
            log.warning("No generic mock events for layer '%s' – wrote empty JSONL.", layer)

    log.info("Ingestion complete.  Files written: %d", len(manifest))
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5 Time Machine – data ingestion / mock generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mock",     action="store_true", default=True,
                   help="Generate mock data instead of live API calls (default: True)")
    p.add_argument("--scenario", default="black_thursday",
                   help="Named scenario preset.  'black_thursday' uses historically-grounded data.")
    p.add_argument("--start",    default="2020-03-11",
                   help="Start date YYYY-MM-DD (ignored when --scenario=black_thursday)")
    p.add_argument("--end",      default="2020-03-15",
                   help="End date YYYY-MM-DD (ignored when --scenario=black_thursday)")
    p.add_argument("--data-root", default="Dataset/phase5_time_machine_dataset",
                   help="Root directory for the Time Machine dataset")
    p.add_argument("--seed",     type=int, default=42, help="RNG seed")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(args.end,   "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    manifest = run_ingestion(
        data_root=args.data_root,
        start=start,
        end=end,
        scenario=args.scenario,
        mock=args.mock,
    )

    print("\n── Time Machine Dataset Manifest ──────────────────────────────")
    for layer, path in sorted(manifest.items()):
        print(f"  {layer:<28}  {path}")
    print("───────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
