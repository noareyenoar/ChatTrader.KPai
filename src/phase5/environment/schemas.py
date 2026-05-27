"""
Phase 5 – Time Machine Dataset
================================
Schema definitions for all 12 environmental data layers.

Quantitative layers (4)  → Apache Parquet  (column dtypes defined as COLUMNS dicts)
  on_chain, tradfi_macro, derivatives, fear_greed

Text / event layers (8)  → JSONL            (dataclasses define the field contract)
  macro_geopolitical, crypto_news, social_sentiment, kol_footprints,
  regulatory_legal, crises_hacks, hf_social, hf_news

Timestamp convention
--------------------
  ``timestamp_utc``  :  int64  –  Unix epoch in **milliseconds**, UTC.
                        Never store as a float; millisecond int64 is lossless
                        and sortable.  Derive with:
                        ``int(datetime.timestamp() * 1000)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Directory name mapping  (layer_key → subdirectory)
# ─────────────────────────────────────────────────────────────────────────────
LAYER_DIRS: dict[str, str] = {
    "macro_geopolitical": "macro_geopolitical",
    "crypto_news":        "crypto_news",
    "on_chain":           "on_chain",
    "social_sentiment":   "social_sentiment",
    "tradfi_macro":       "tradfi_macro",
    "derivatives":        "derivatives_microstructure",
    "kol_footprints":     "kol_footprints",
    "regulatory_legal":   "regulatory_legal",
    "crises_hacks":       "crises_hacks",
    "fear_greed":         "fear_greed",
    # HuggingFace enrichment layers (added Session 2)
    "hf_social":          "hf_social",
    "hf_news":            "hf_news",
}

# Layers stored as Parquet vs. JSONL
PARQUET_LAYERS: frozenset[str] = frozenset({
    "on_chain", "tradfi_macro", "derivatives", "fear_greed"
})
JSONL_LAYERS: frozenset[str] = frozenset({
    "macro_geopolitical", "crypto_news", "social_sentiment",
    "kol_footprints", "regulatory_legal", "crises_hacks",
    "hf_social", "hf_news",
})

ALL_LAYER_KEYS: tuple[str, ...] = tuple(LAYER_DIRS)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1  –  GLOBAL_MACRO_GEOPOLITICAL  (JSONL)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MacroGeopoliticalEvent:
    """A single macro-geopolitical event that shaped the market backdrop."""
    event_id:            str               # UUID-style unique ID
    timestamp_utc:       int               # ms epoch – *publication* time
    date_str:            str               # ISO-8601  e.g. "2020-03-11T22:30:00Z"
    title:               str
    description:         str
    source:              str               # "WHO" | "Reuters" | "Bloomberg" …
    event_type:          str               # pandemic|war|ban|crisis|policy|protest
    region:              str               # ISO-3166 alpha-3 or "global"
    asset_class_impact:  List[str]         # ["crypto","equities","bonds","commodities"]
    impact_score:        float             # -1.0 bearish … +1.0 bullish
    tags:                List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2  –  CRYPTO_SPECIFIC_PROJECT_NEWS  (JSONL)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CryptoNewsEvent:
    """A single cryptocurrency project news event."""
    event_id:       str
    timestamp_utc:  int
    date_str:       str
    coin_ticker:    str           # "BTC" | "ETH" | "MULTI"
    project_name:   str
    category:       str           # hard_fork|exchange_listing|hack|partnership|
                                  # tokenomics|protocol_upgrade|regulatory|market
    headline:       str
    body:           str
    source:         str
    sentiment_score: float        # -1.0 … +1.0
    impact_tags:    List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3  –  ON_CHAIN_ANALYTICS  (Parquet)
# ─────────────────────────────────────────────────────────────────────────────
ON_CHAIN_COLUMNS: dict[str, str] = {
    "timestamp_utc":              "int64",
    "symbol":                     "object",   # "BTC" | "ETH"
    "exchange_inflow":            "float64",  # coins moved onto exchanges
    "exchange_outflow":           "float64",  # coins moved off exchanges
    "net_exchange_flow":          "float64",  # inflow − outflow  (negative = accumulation)
    "whale_transfer_count":       "int64",    # transfers > $1M USD
    "whale_transfer_volume_usd":  "float64",
    "miner_hash_rate_eh_s":       "float64",  # EH/s  (BTC only)
    "network_difficulty":         "float64",  # raw difficulty (BTC only)
    "active_addresses":           "int64",
    "nvt_ratio":                  "float64",  # Network Value to Transactions
    "sopr":                       "float64",  # Spent Output Profit Ratio
    "smart_money_net_flow_usd":   "float64",  # large wallet net accumulation
}


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4  –  SOCIAL_MEDIA_HYPE_SENTIMENT  (JSONL)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SocialSentimentSnapshot:
    """Hourly social-media sentiment snapshot per platform per coin."""
    snapshot_id:          str
    timestamp_utc:        int
    coin_ticker:          str
    platform:             str    # twitter|reddit_r_bitcoin|reddit_r_crypto|telegram
    mention_count_1h:     int
    sentiment_score_1h:   float  # -1.0 … +1.0
    volume_percentile_30d: float # 0–100  (100 = highest 30-day activity)
    top_terms:            List[str]
    raw_sample_count:     int
    bullish_count:        int
    bearish_count:        int
    neutral_count:        int


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5  –  TRADFI_MACRO_LIQUIDITY  (Parquet)
# ─────────────────────────────────────────────────────────────────────────────
TRADFI_MACRO_COLUMNS: dict[str, str] = {
    "timestamp_utc":       "int64",
    "resolution":          "object",    # "daily" | "event"
    "fed_funds_rate_bps":  "int64",     # in basis points  e.g. 175 = 1.75 %
    "fed_rate_change_bps": "int64",     # 0 on non-FOMC days
    "is_fomc_day":         "bool",
    "is_cpi_release_day":  "bool",
    "cpi_yoy_pct":         "float64",   # NaN if not a release day
    "ppi_yoy_pct":         "float64",
    "dxy_close":           "float64",
    "dxy_change_pct":      "float64",
    "sp500_close":         "float64",
    "sp500_change_pct":    "float64",
    "vix_close":           "float64",
    "nasdaq_close":        "float64",
    "nasdaq_change_pct":   "float64",
    "us_10y_yield":        "float64",
}


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6  –  MARKET_DERIVATIVES_MICROSTRUCTURE  (Parquet)
# ─────────────────────────────────────────────────────────────────────────────
# Symbols with real funding-rate coverage in the dataset:
#   BTCUSDT  2019-09 →  (Binance perp launch + set3)
#   ETHUSDT  2019-09 →  (Binance perp launch + set3)
#   SOLUSDT  2020-08 →  (backfilled from scripts/phase5/salvage_derivatives.py)
#   BNBUSDT  2019-09 →  (backfilled from scripts/phase5/salvage_derivatives.py)
#   LINKUSDT future    (populated on full re-ingest from Binance Data Vision)
DERIVATIVES_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT",
)

DERIVATIVES_COLUMNS: dict[str, str] = {
    "timestamp_utc":           "int64",
    "symbol":                  "object",   # one of DERIVATIVES_SYMBOLS
    "funding_rate":            "float64",  # raw 8-hour funding rate (last in window)
    "oi_usd":                  "float64",  # open interest in USD (0.0 for backfilled rows)
    "oi_change_usd":           "float64",  # delta vs. previous period
    "long_liquidations_usd":   "float64",
    "short_liquidations_usd":  "float64",
    "total_liquidations_usd":  "float64",
    "liq_long_short_ratio":    "float64",  # long_liq / (long_liq + short_liq)
    "options_max_pain_usd":    "float64",  # NaN when no options data
    "put_call_ratio":          "float64",
    "basis_pct":               "float64",  # (futures − spot) / spot
    "is_synthetic":            "bool",     # True when row was imputed or backfilled
}


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 7  –  KEY_OPINION_LEADER_FOOTPRINTS  (JSONL)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class KOLFootprint:
    """A single public statement by a key opinion leader."""
    footprint_id:          str
    timestamp_utc:         int
    kol_id:                str
    kol_name:              str
    kol_role:              str            # ceo|analyst|influencer|protocol_lead|investor
    platform:              str            # twitter|blog|telegram|youtube
    content:               str            # verbatim or paraphrased quote
    coins_mentioned:       List[str]
    estimated_reach:       int            # approximate impressions
    follower_count:        int
    market_reaction_1h_pct: Optional[float]  # % price move in 1h after post
    tags:                  List[str] = field(default_factory=list)
    sentiment:             str = "neutral"  # bullish|bearish|neutral|fud|shill


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 8  –  REGULATORY_LEGAL_MILESTONES  (JSONL)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RegulatoryLegalEvent:
    event_id:       str
    timestamp_utc:  int
    jurisdiction:   str    # "USA" | "EU" | "CHN" | "global" …
    regulator:      str    # "SEC" | "CFTC" | "FinCEN" | "ECB" …
    category:       str    # enforcement|ban|approval|lawsuit|tax_policy|guidance
    title:          str
    description:    str
    affected_coins: List[str]
    outcome:        str    # pending|approved|rejected|settled|withdrawn
    severity:       str    # critical|high|medium|low
    source_url:     str = ""


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 9  –  SYSTEMIC_CRISES_HACKS_EXPLOITS  (JSONL)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CrisisHackEvent:
    incident_id:      str
    timestamp_utc:    int
    incident_type:    str           # hack|exploit|depeg|collapse|bridge_vulnerability|
                                    # exchange_failure|liquidity_crisis
    protocol:         str
    chain:            str
    amount_lost_usd:  Optional[float]
    description:      str
    stage:            str           # pre_attack|attack|discovery|escalation|
                                    # post_mortem|recovery
    phase_tag:        str           # "T+0h" | "T+3h" … for hour-by-hour tracking
    severity:         str           # critical|high|medium|low
    recovery_status:  str = "unrecovered"   # unrecovered|partial|full


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 10  –  RETAIL_BEHAVIORAL_FEAR_GREED  (Parquet)
# ─────────────────────────────────────────────────────────────────────────────
FEAR_GREED_COLUMNS: dict[str, str] = {
    "timestamp_utc":               "int64",
    "date_str":                    "object",
    "fear_greed_index":            "int64",   # 0 (extreme fear) … 100 (extreme greed)
    "fear_greed_label":            "object",  # extreme_fear|fear|neutral|greed|extreme_greed
    "google_trends_bitcoin":       "float64", # 0–100 normalised weekly peak
    "google_trends_crypto":        "float64",
    "google_trends_buy_crypto":    "float64",
    "google_trends_ethereum":      "float64",
    "google_trends_bitcoin_crash": "float64",
    "google_trends_sell_bitcoin":  "float64",
    "retail_search_composite":     "float64", # weighted average of above trends
}
