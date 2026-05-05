"""
agents/regime_detector.py
Regime Detector for ChatTrader.KPai Phase 5.

Identifies the current market state from OHLCV features and
model signals. Outputs a regime label used to weight analyst credibility.

Regimes:
  - TRENDING_UP
  - TRENDING_DOWN
  - RANGING
  - HIGH_VOLATILITY
  - LOW_VOLATILITY
  - BREAKOUT
  - REVERTING
  - UNKNOWN
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

REGIME_LABELS = [
    "TRENDING_UP",
    "TRENDING_DOWN",
    "RANGING",
    "HIGH_VOLATILITY",
    "LOW_VOLATILITY",
    "BREAKOUT",
    "REVERTING",
    "UNKNOWN",
]


class RegimeDetector:
    """
    Rule-based + signal-based market regime classifier.

    Uses rolling statistics from the market_context (provided by model_bridge)
    to classify the current market regime. The regime is then injected into
    every market_context dict that gets sent to analyst agents.

    When real trained models are available, this can be hot-swapped with
    a neural regime classifier without changing any downstream API.
    """

    def __init__(
        self,
        atr_multiplier_high: float = 1.5,
        atr_multiplier_low: float = 0.7,
        trend_slope_threshold: float = 0.001,
        bb_width_breakout: float = 0.04,
    ) -> None:
        self.atr_multiplier_high = atr_multiplier_high
        self.atr_multiplier_low = atr_multiplier_low
        self.trend_slope_threshold = trend_slope_threshold
        self.bb_width_breakout = bb_width_breakout
        self._last_regime: str = "UNKNOWN"

    def detect(self, market_context: Dict[str, Any]) -> str:
        """
        Classify current market regime from market_context features.

        Expected keys in market_context:
          features (dict):
            - atr_14: float
            - atr_mean: float           (rolling 50-bar mean of ATR)
            - price_slope_20: float
            - zscore_close_64: float
            - ema_spread: float
            - bb_width: float           (Bollinger Band width = 2*std/mean)

        Returns:
          str — one of REGIME_LABELS
        """
        features = market_context.get("features", {})
        if not features:
            logger.warning("RegimeDetector: no features in context, defaulting to UNKNOWN.")
            return "UNKNOWN"

        atr = features.get("atr_14", 0.0)
        atr_mean = features.get("atr_mean", atr)
        slope = features.get("price_slope_20", 0.0)
        zscore = features.get("zscore_close_64", 0.0)
        ema_spread = features.get("ema_spread", 0.0)
        bb_width = features.get("bb_width", 0.02)

        # ── Regime rules (priority order) ──────────────────────────────
        # 1. Breakout: BB width suddenly expands
        if bb_width > self.bb_width_breakout:
            regime = "BREAKOUT"

        # 2. High volatility: ATR significantly above its mean
        elif atr_mean > 0 and (atr / atr_mean) > self.atr_multiplier_high:
            regime = "HIGH_VOLATILITY"

        # 3. Low volatility: ATR significantly below mean
        elif atr_mean > 0 and (atr / atr_mean) < self.atr_multiplier_low:
            regime = "LOW_VOLATILITY"

        # 4. Strong uptrend: positive slope + positive EMA spread
        elif slope > self.trend_slope_threshold and ema_spread > 0:
            regime = "TRENDING_UP"

        # 5. Strong downtrend: negative slope + negative EMA spread
        elif slope < -self.trend_slope_threshold and ema_spread < 0:
            regime = "TRENDING_DOWN"

        # 6. Reverting: price is far from mean (extreme zscore) but slope is flat
        elif abs(zscore) > 1.8 and abs(slope) < self.trend_slope_threshold:
            regime = "REVERTING"

        # 7. Ranging: flat slope, narrow BB
        elif abs(slope) < self.trend_slope_threshold / 2:
            regime = "RANGING"

        else:
            regime = "UNKNOWN"

        self._last_regime = regime
        logger.debug("RegimeDetector: %s (slope=%.4f, atr_ratio=%.2f, zscore=%.2f)",
                     regime, slope, atr / atr_mean if atr_mean else 0, zscore)
        return regime

    @property
    def last_regime(self) -> str:
        return self._last_regime

    def regime_weights(self) -> Dict[str, float]:
        """
        Return archetype credibility multipliers for the current regime.
        Used by the Orchestrator to weight analyst votes.
        """
        weights = {
            "trend_follower": 0.5,
            "mean_reversion": 0.5,
            "scalping_microstructure": 0.5,
            "statistical_arbitrage": 0.5,
            "discretionary_multimodal": 0.5,
            "market_making_rl": 0.5,
        }

        regime_boost = {
            "TRENDING_UP": {"trend_follower": 0.9, "mean_reversion": 0.25, "discretionary_multimodal": 0.7},
            "TRENDING_DOWN": {"trend_follower": 0.9, "mean_reversion": 0.25, "discretionary_multimodal": 0.7},
            "RANGING": {"mean_reversion": 0.9, "statistical_arbitrage": 0.85, "market_making_rl": 0.85, "trend_follower": 0.2},
            "HIGH_VOLATILITY": {"scalping_microstructure": 0.85, "market_making_rl": 0.3, "trend_follower": 0.6},
            "LOW_VOLATILITY": {"market_making_rl": 0.9, "statistical_arbitrage": 0.8, "mean_reversion": 0.7},
            "BREAKOUT": {"scalping_microstructure": 0.85, "trend_follower": 0.75, "discretionary_multimodal": 0.7},
            "REVERTING": {"mean_reversion": 0.9, "statistical_arbitrage": 0.75, "market_making_rl": 0.65},
            "UNKNOWN": {},  # No boosts — all equal weight
        }

        overrides = regime_boost.get(self._last_regime, {})
        weights.update(overrides)
        return weights
