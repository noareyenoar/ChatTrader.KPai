"""
real_signal_bridge.py
Rule-based signal extraction from real market features — replaces MockModelBridge.

Each archetype uses domain-appropriate indicators derived from real price/volume data.
No random noise. Signals reflect actual market structure.

HOT-SWAP: Implements the same get_all_signals() contract as MockModelBridge.
When models pass validation, replace with ProductionModelBridge (loads checkpoint weights).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Signal confidence is clamped to [CONF_MIN, CONF_MAX] to avoid degenerate
# extreme values that might override the LLM debate
CONF_MIN = 0.35
CONF_MAX = 0.88


def _sigmoid(x: float) -> float:
    """Map any real value to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(v: float, lo: float = CONF_MIN, hi: float = CONF_MAX) -> float:
    return max(lo, min(hi, v))


def _direction(logit: float, threshold: float = 0.04) -> str:
    """Convert a normalised logit to LONG/SHORT/FLAT."""
    if logit > threshold:
        return "LONG"
    if logit < -threshold:
        return "SHORT"
    return "FLAT"


def _model_votes(archetype_logit: float, n: int = 3) -> list:
    """
    Simulate N model votes consistent with the ensemble logit.
    Each model gets slightly different noise so votes aren't identical.
    """
    votes = []
    for k in range(n):
        noise = (k - 1) * 0.06          # offsets: -0.06, 0, +0.06
        logit_k = archetype_logit + noise
        votes.append({
            "direction": _direction(logit_k),
            "logit": round(logit_k, 4),
            "confidence": round(_clamp(_sigmoid(abs(logit_k) * 3)), 4),
        })
    return votes


class RealSignalBridge:
    """
    Extracts archetype signals from real computed market features.

    Rules per archetype:
    ─────────────────────────────────────────────────────────────────
    trend_follower         : slope + ema_spread + momentum
    mean_reversion         : zscore mean-reversion + BB distance
    scalping_microstructure: OFI + short-term volatility + spread
    statistical_arbitrage  : zscore + spread_z + pair_correlation
    discretionary_multimodal: pattern_score + RSI + ema_spread
    market_making_rl       : inventory + vol_regime + spread_proxy
    ─────────────────────────────────────────────────────────────────
    """

    def __init__(self) -> None:
        logger.info(
            "RealSignalBridge initialized. "
            "Signals derived from real market features. "
            "Swap to ProductionModelBridge when models pass validation."
        )

    # ─────────────────────────────────────────────
    # Public interface (matches MockModelBridge)
    # ─────────────────────────────────────────────
    def get_all_signals(
        self,
        symbol: str,
        market_context: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Returns signals for all 6 archetypes from real features.
        """
        feat = market_context.get("features", {})
        if not feat:
            logger.warning("RealSignalBridge: no features in context — returning FLAT signals")
            return self._flat_signals()

        return {
            "trend_follower":           self._trend_signals(feat),
            "mean_reversion":           self._mean_reversion_signals(feat),
            "scalping_microstructure":  self._scalper_signals(feat),
            "statistical_arbitrage":    self._stat_arb_signals(feat),
            "discretionary_multimodal": self._discretionary_signals(feat),
            "market_making_rl":         self._market_maker_signals(feat),
        }

    def build_price_summary(self, features: Dict[str, Any]) -> str:
        """Human-readable feature summary for LLM prompts."""
        slope = features.get("price_slope_20", 0)
        zscore = features.get("zscore_close_64", 0)
        atr = features.get("atr_14", 0)
        ofi = features.get("ofi_proxy", 0)
        rsi = features.get("rsi_14", 50)
        ema_spread = features.get("ema_spread", 0)
        bb_width = features.get("bb_width", 0)

        return (
            f"Price slope: {slope:+.6f} | Z-score: {zscore:+.2f} | "
            f"ATR: {atr:.5f} | OFI: {ofi:+.3f} | RSI: {rsi:.1f} | "
            f"EMA spread: {ema_spread:+.5f} | BB width: {bb_width:.4f}"
        )

    # ─────────────────────────────────────────────
    # Archetype signal methods
    # ─────────────────────────────────────────────

    def _trend_signals(self, feat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trend Follower — LSTM/Transformer/TCN archetype.
        Primary: slope + ema_spread
        Secondary: RSI filter (not overbought/oversold)
        """
        slope = feat.get("price_slope_20", 0.0)
        ema_spread = feat.get("ema_spread", 0.0)
        rsi = feat.get("rsi_14", 50.0)

        # Composite logit: slope dominates, ema_spread confirms
        logit = slope * 800 + ema_spread * 200

        # RSI filter: dampen if overbought (>70) and long signal, or oversold (<30) and short
        if logit > 0 and rsi > 72:
            logit *= 0.5
        elif logit < 0 and rsi < 28:
            logit *= 0.5

        direction = _direction(logit, threshold=0.06)
        confidence = _clamp(_sigmoid(abs(logit) * 2.5))
        votes = _model_votes(logit)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "ensemble_logit": round(logit, 4),
            "model_votes": votes,
            "raw_outputs": {"slope": slope, "ema_spread": ema_spread, "rsi": rsi},
        }

    def _mean_reversion_signals(self, feat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mean Reversion — MLP/ResNet/GRN archetype.
        Primary: zscore (negative = buy low, positive = sell high)
        Secondary: BB distance confirmation
        """
        zscore = feat.get("zscore_close_64", 0.0)
        bb_dist = feat.get("bb_distance", 0.0)

        # Mean reversion: negative zscore = LONG (oversold), positive = SHORT (overbought)
        # Signal needs strong extension to be credible
        z_signal = -zscore * 0.4           # invert: buy low, sell high
        bb_signal = -bb_dist * 0.3         # same inversion

        logit = z_signal + bb_signal

        # Only trade at sufficient extension (|zscore| > 1.0)
        extension = abs(zscore)
        if extension < 0.8:
            logit *= 0.3                   # dampen near-zero signals

        direction = _direction(logit, threshold=0.05)
        confidence = _clamp(_sigmoid(extension * 0.8))
        votes = _model_votes(logit)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "ensemble_logit": round(logit, 4),
            "model_votes": votes,
            "raw_outputs": {"zscore": zscore, "bb_distance": bb_dist},
        }

    def _scalper_signals(self, feat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scalper — CNN/GRU microstructure archetype.
        Primary: OFI (order flow imbalance)
        Secondary: short-term spread, vol_regime
        """
        ofi = feat.get("ofi_proxy", 0.0)
        spread = feat.get("spread_proxy", 0.001)
        vol_regime = feat.get("vol_regime_code", 1.0)

        # Strong OFI imbalance drives scalper direction
        logit = ofi * 0.9

        # High vol regime lowers scalper conviction
        if vol_regime >= 2.0:
            logit *= 0.6

        # Very tight spread = cleaner signal
        spread_factor = max(0.4, 1.0 - spread * 100)
        confidence = _clamp(_sigmoid(abs(ofi) * 3) * spread_factor)
        direction = _direction(logit, threshold=0.04)
        votes = _model_votes(logit)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "ensemble_logit": round(logit, 4),
            "model_votes": votes,
            "raw_outputs": {"ofi": ofi, "spread": spread, "vol_regime": vol_regime},
        }

    def _stat_arb_signals(self, feat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Statistical Arbitrage — Autoencoder/GAT/LSTM archetype.
        Primary: spread_z_64 (z-score of spread proxy)
        Secondary: pair_correlation, zscore
        """
        spread_z = feat.get("spread_z_64", 0.0)
        pair_corr = feat.get("pair_correlation", 0.5)
        zscore = feat.get("zscore_close_64", 0.0)

        # Mean-reversion on spread z-score
        logit = -spread_z * 0.35 + (-zscore * 0.15)

        # Higher correlation = more reliable stat arb
        corr_factor = max(0.3, pair_corr)
        confidence = _clamp(_sigmoid(abs(spread_z) * 0.7) * corr_factor)
        direction = _direction(logit, threshold=0.04)
        votes = _model_votes(logit)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "ensemble_logit": round(logit, 4),
            "model_votes": votes,
            "raw_outputs": {"spread_z": spread_z, "pair_corr": pair_corr, "zscore": zscore},
        }

    def _discretionary_signals(self, feat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discretionary Multimodal — ViT/CNN chart pattern archetype.
        Primary: pattern_score (momentum proxy)
        Secondary: RSI divergence, fracdiff
        """
        pattern_score = feat.get("pattern_score", 0.0)
        rsi = feat.get("rsi_14", 50.0)
        fracdiff = feat.get("fracdiff_close_d04", 0.0)

        # Pattern-based directional signal
        logit = pattern_score * 6 + fracdiff * 2

        # RSI adds conviction at extremes
        if rsi > 65 and logit > 0:
            logit *= 1.2
        elif rsi < 35 and logit < 0:
            logit *= 1.2

        logit = max(-1.5, min(1.5, logit))  # cap before sigmoid
        confidence = _clamp(_sigmoid(abs(pattern_score) * 4))
        direction = _direction(logit, threshold=0.05)
        votes = _model_votes(logit)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "ensemble_logit": round(logit, 4),
            "model_votes": votes,
            "raw_outputs": {"pattern_score": pattern_score, "rsi": rsi, "fracdiff": fracdiff},
        }

    def _market_maker_signals(self, feat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Market Maker RL — PPO/SAC/DQN archetype.
        Primary: inventory_level (mean-revert inventory imbalance)
        Secondary: vol_regime (low vol = better for MM)
        """
        inventory = feat.get("inventory_level", 0.0)
        vol_regime = feat.get("vol_regime_code", 1.0)
        spread = feat.get("spread_proxy", 0.001)

        # MM wants to reduce inventory imbalance
        logit = -inventory * 0.7           # if long inventory: sell pressure

        # Low vol improves MM confidence
        if vol_regime <= 0:
            confidence_boost = 1.15
        elif vol_regime >= 2:
            confidence_boost = 0.70
        else:
            confidence_boost = 1.0

        # Wide spreads → better MM margins → more confident
        spread_boost = min(1.3, 1.0 + spread * 30)

        confidence = _clamp(_sigmoid(abs(inventory) * 2) * confidence_boost * spread_boost)
        direction = _direction(logit, threshold=0.05)
        votes = _model_votes(logit)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "ensemble_logit": round(logit, 4),
            "model_votes": votes,
            "raw_outputs": {"inventory": inventory, "vol_regime": vol_regime, "spread": spread},
        }

    # ─────────────────────────────────────────────
    # Fallback
    # ─────────────────────────────────────────────
    def _flat_signals(self) -> Dict[str, Any]:
        template = {
            "direction": "FLAT",
            "confidence": 0.30,
            "ensemble_logit": 0.0,
            "model_votes": [{"direction": "FLAT", "logit": 0.0, "confidence": 0.30}] * 3,
            "raw_outputs": {},
        }
        return {k: dict(template) for k in [
            "trend_follower", "mean_reversion", "scalping_microstructure",
            "statistical_arbitrage", "discretionary_multimodal", "market_making_rl",
        ]}
