"""
model_bridge.py
Mock Inference Layer — ChatTrader.KPai Phase 5

Provides realistically formatted mock signals for all 18 models.
HOT-SWAP: When real models finish training and pass validation gates,
replace `MockModelBridge` with `ProductionModelBridge` (drop-in API).

Architecture: 6 archetypes × 3 models = 18 models total
Each archetype returns a consensus signal dict that agents consume.

Signal contract per archetype:
{
  "direction": "LONG" | "SHORT" | "FLAT",
  "confidence": float [0.0 – 1.0],
  "model_votes": { model_name: {"direction": str, "logit": float}, ... },
  "ensemble_confidence": float,
  "raw_outputs": { ... }      # archetype-specific extras
}
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Mock signal distributions (per archetype)
# These mimic realistic model behavior without needing trained weights.
# Distributions are seeded per symbol+timestamp for reproducibility.
# ─────────────────────────────────────────────

_ARCHETYPE_MODEL_NAMES = {
    "trend_follower": ["LSTM_Trend_v1", "Transformer_Trend_v1", "TCN_Trend_v1"],
    "mean_reversion": ["MLP_MR_v1", "ResNet_MR_v1", "GRN_MR_v1"],
    "scalping_microstructure": ["CNN_Scalper_v1", "LinearAttn_Scalper_v1", "GRU_Scalper_v1"],
    "statistical_arbitrage": ["Autoencoder_StatArb_v1", "GAT_StatArb_v1", "LSTM_StatArb_v1"],
    "discretionary_multimodal": ["ViT_Disc_v1", "Multimodal_Disc_v1", "CNNChart_Disc_v1"],
    "market_making_rl": ["PPO_MM_v1", "SAC_MM_v1", "DQN_MM_v1"],
}

# Registry-derived validation stats (used to shape mock noise)
_ARCHETYPE_MOCK_ACCURACY = {
    "trend_follower": 0.506,          # ~coin-flip (still training)
    "mean_reversion": 0.517,
    "scalping_microstructure": 0.384, # Inverted signal pattern (known issue)
    "statistical_arbitrage": 0.513,
    "discretionary_multimodal": 0.383,
    "market_making_rl": 0.564,        # Best performer so far
}


def _direction_from_logit(logit: float) -> str:
    if logit > 0.1:
        return "LONG"
    elif logit < -0.1:
        return "SHORT"
    return "FLAT"


def _mock_logit(archetype: str, base_direction: str, noise_std: float = 0.3) -> float:
    """
    Generate a mock logit value biased toward base_direction.
    Accuracy is modulated by _ARCHETYPE_MOCK_ACCURACY.
    """
    acc = _ARCHETYPE_MOCK_ACCURACY.get(archetype, 0.51)
    bias = 0.5 if base_direction == "LONG" else (-0.5 if base_direction == "SHORT" else 0.0)
    # Scale bias by how far acc is from 0.5
    signal_strength = (acc - 0.5) * 4.0  # [−2, 2] range
    logit = bias * signal_strength + random.gauss(0, noise_std)
    return logit


class MockModelBridge:
    """
    Returns randomized but realistically formatted signals for all 18 models.

    Usage:
        bridge = MockModelBridge(seed=42)
        signals = bridge.get_all_signals(symbol="BTCUSDT", market_context={...})

    Returns dict keyed by archetype name, each containing the signal dict.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        logger.info("MockModelBridge initialized (seed=%s). "
                    "Hot-swap with ProductionModelBridge when models pass validation.", seed)

    def _mock_trend_signals(
        self, symbol: str, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        slope = features.get("price_slope_20", 0.0)
        ema_spread = features.get("ema_spread", 0.0)
        # Determine plausible base direction from raw features
        base = "LONG" if (slope > 0 and ema_spread > 0) else ("SHORT" if (slope < 0 and ema_spread < 0) else "FLAT")

        model_votes = {}
        logits = []
        for name in _ARCHETYPE_MODEL_NAMES["trend_follower"]:
            logit = _mock_logit("trend_follower", base)
            model_votes[name] = {"direction": _direction_from_logit(logit), "logit": round(logit, 4)}
            logits.append(logit)

        ensemble_logit = float(np.mean(logits))
        direction = _direction_from_logit(ensemble_logit)
        confidence = min(0.95, 0.5 + abs(ensemble_logit) * 0.3)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "model_votes": model_votes,
            "ensemble_logit": round(ensemble_logit, 4),
            "raw_outputs": {
                "price_slope_20": slope,
                "ema_spread": ema_spread,
                "atr_14": features.get("atr_14", 0.0),
            },
        }

    def _mock_mean_reversion_signals(
        self, symbol: str, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        zscore = features.get("zscore_close_64", 0.0)
        bb_dist = features.get("bb_distance", 0.0)
        # Reversal logic: if zscore is high positive → SHORT (revert down)
        base = "SHORT" if zscore > 1.5 else ("LONG" if zscore < -1.5 else "FLAT")

        model_votes = {}
        logits = []
        for name in _ARCHETYPE_MODEL_NAMES["mean_reversion"]:
            logit = _mock_logit("mean_reversion", base)
            model_votes[name] = {"direction": _direction_from_logit(logit), "logit": round(logit, 4)}
            logits.append(logit)

        ensemble_logit = float(np.mean(logits))
        direction = _direction_from_logit(ensemble_logit)
        confidence = min(0.95, 0.5 + min(abs(zscore) / 3.0, 0.45))

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "model_votes": model_votes,
            "ensemble_logit": round(ensemble_logit, 4),
            "raw_outputs": {
                "zscore_close_64": zscore,
                "bb_distance": bb_dist,
                "rsi_14": features.get("rsi_14", 50.0),
            },
        }

    def _mock_scalper_signals(
        self, symbol: str, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        ofi = features.get("ofi_proxy", 0.0)
        vol_regime = features.get("vol_regime_code", 1)
        base = "LONG" if ofi > 0.1 else ("SHORT" if ofi < -0.1 else "FLAT")

        model_votes = {}
        logits = []
        for name in _ARCHETYPE_MODEL_NAMES["scalping_microstructure"]:
            logit = _mock_logit("scalping_microstructure", base, noise_std=0.4)
            model_votes[name] = {"direction": _direction_from_logit(logit), "logit": round(logit, 4)}
            logits.append(logit)

        ensemble_logit = float(np.mean(logits))
        direction = _direction_from_logit(ensemble_logit)
        confidence = min(0.90, 0.4 + abs(ofi) * 0.5)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "model_votes": model_votes,
            "ensemble_logit": round(ensemble_logit, 4),
            "raw_outputs": {
                "ofi_proxy": ofi,
                "vol_regime_code": vol_regime,
                "spread_proxy": features.get("spread_proxy", 0.001),
            },
        }

    def _mock_stat_arb_signals(
        self, symbol: str, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        spread_z = features.get("spread_z_64", 0.0)
        base = "SHORT" if spread_z > 2.0 else ("LONG" if spread_z < -2.0 else "FLAT")

        model_votes = {}
        logits = []
        for name in _ARCHETYPE_MODEL_NAMES["statistical_arbitrage"]:
            logit = _mock_logit("statistical_arbitrage", base)
            model_votes[name] = {"direction": _direction_from_logit(logit), "logit": round(logit, 4)}
            logits.append(logit)

        ensemble_logit = float(np.mean(logits))
        direction = _direction_from_logit(ensemble_logit)
        confidence = min(0.90, 0.45 + min(abs(spread_z) / 4.0, 0.4))

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "model_votes": model_votes,
            "ensemble_logit": round(ensemble_logit, 4),
            "raw_outputs": {
                "spread_z_64": spread_z,
                "fracdiff_d04": features.get("fracdiff_close_d04", 0.0),
                "pair_correlation": features.get("pair_correlation", 0.7),
            },
        }

    def _mock_discretionary_signals(
        self, symbol: str, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        slope = features.get("price_slope_20", 0.0)
        pattern_score = features.get("pattern_score", random.uniform(-0.5, 0.5))
        base = "LONG" if pattern_score > 0.2 else ("SHORT" if pattern_score < -0.2 else "FLAT")

        model_votes = {}
        logits = []
        for name in _ARCHETYPE_MODEL_NAMES["discretionary_multimodal"]:
            logit = _mock_logit("discretionary_multimodal", base, noise_std=0.5)
            model_votes[name] = {"direction": _direction_from_logit(logit), "logit": round(logit, 4)}
            logits.append(logit)

        ensemble_logit = float(np.mean(logits))
        direction = _direction_from_logit(ensemble_logit)
        confidence = min(0.85, 0.4 + abs(pattern_score) * 0.4)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "model_votes": model_votes,
            "ensemble_logit": round(ensemble_logit, 4),
            "raw_outputs": {
                "pattern_score": round(pattern_score, 4),
                "chart_embedding_norm": round(random.uniform(0.4, 0.9), 4),
            },
        }

    def _mock_market_maker_signals(
        self, symbol: str, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        inventory = features.get("inventory_level", 0.0)
        volatility = features.get("atr_14", 0.002)
        # MM suggests reducing inventory if extreme
        base = "SHORT" if inventory > 0.5 else ("LONG" if inventory < -0.5 else "FLAT")

        model_votes = {}
        logits = []
        for name in _ARCHETYPE_MODEL_NAMES["market_making_rl"]:
            logit = _mock_logit("market_making_rl", base, noise_std=0.25)
            model_votes[name] = {"direction": _direction_from_logit(logit), "logit": round(logit, 4)}
            logits.append(logit)

        ensemble_logit = float(np.mean(logits))
        direction = _direction_from_logit(ensemble_logit)
        confidence = min(0.92, 0.55 + abs(inventory) * 0.3)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "model_votes": model_votes,
            "ensemble_logit": round(ensemble_logit, 4),
            "raw_outputs": {
                "inventory_level": inventory,
                "bid_offset": round(random.uniform(0.0001, 0.002), 6),
                "ask_offset": round(random.uniform(0.0001, 0.002), 6),
                "fill_probability": round(random.uniform(0.3, 0.8), 4),
            },
        }

    def get_all_signals(
        self,
        symbol: str,
        market_context: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate mock signals for all 6 archetypes.
        Returns a dict keyed by archetype name.

        This is the HOT-SWAP entry point:
        ProductionModelBridge.get_all_signals() has identical signature.
        """
        features = market_context.get("features", {})

        t0 = time.time()
        signals = {
            "trend_follower": self._mock_trend_signals(symbol, features),
            "mean_reversion": self._mock_mean_reversion_signals(symbol, features),
            "scalping_microstructure": self._mock_scalper_signals(symbol, features),
            "statistical_arbitrage": self._mock_stat_arb_signals(symbol, features),
            "discretionary_multimodal": self._mock_discretionary_signals(symbol, features),
            "market_making_rl": self._mock_market_maker_signals(symbol, features),
        }
        latency_ms = (time.time() - t0) * 1000
        logger.debug("MockModelBridge.get_all_signals: %.1f ms", latency_ms)

        return signals

    def build_price_summary(self, features: Dict[str, Any]) -> str:
        """Human-readable price action summary injected into LLM prompts."""
        slope = features.get("price_slope_20", 0.0)
        zscore = features.get("zscore_close_64", 0.0)
        atr = features.get("atr_14", 0.0)
        ema_spread = features.get("ema_spread", 0.0)

        direction_word = "rising" if slope > 0 else ("falling" if slope < 0 else "flat")
        extension = "overextended" if abs(zscore) > 1.8 else ("near fair value" if abs(zscore) < 0.5 else "mildly stretched")
        momentum_word = "positive" if ema_spread > 0 else ("negative" if ema_spread < 0 else "neutral")

        return (
            f"Price is {direction_word} (slope={slope:+.4f}), "
            f"{extension} (z-score={zscore:+.2f}), "
            f"momentum is {momentum_word} (EMA spread={ema_spread:+.4f}), "
            f"ATR={atr:.4f}."
        )
