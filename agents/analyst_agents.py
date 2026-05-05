"""
agents/analyst_agents.py
6 Archetype Analyst Agents for ChatTrader.KPai Phase 5.

Each analyst:
  1. Receives a market_context dict (from model_bridge + regime detector)
  2. Builds an LLM prompt with its unique persona and the quantitative signals
  3. Returns a structured EvidencePacket for the Orchestrator
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from agents.base_agent import BaseAgent, EvidencePacket, _parse_json_response

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Shared JSON contract for all analyst responses
# ─────────────────────────────────────────────
_ANALYST_RESPONSE_SCHEMA = """
Respond ONLY with this JSON (no prose, no markdown fences):
{{
    "direction": "LONG" | "SHORT" | "FLAT",
    "confidence": <float 0.0-1.0>,
    "regime_alignment": <float 0.0-1.0>,
    "thesis": "<1-3 sentence summary of your reasoning>"
}}
"""

_BASE_CONTEXT_TEMPLATE = """
Current market context:
- Symbol: {symbol}
- Timeframe: {timeframe}
- Current Regime: {regime}
- Your model signals: {signals}
- Recent price action summary: {price_summary}

Analyze and produce your trading stance.
""" + _ANALYST_RESPONSE_SCHEMA


def _build_evidence(
    agent: BaseAgent,
    archetype: str,
    market_context: Dict[str, Any],
    regime_alignment: float,
) -> EvidencePacket:
    """
    Common builder: calls LLM, parses response, constructs EvidencePacket.
    Falls back to model signals alone if LLM fails.
    """
    prompt = _BASE_CONTEXT_TEMPLATE.format(
        symbol=market_context.get("symbol", "BTCUSDT"),
        timeframe=market_context.get("timeframe", "1h"),
        regime=market_context.get("regime", "UNKNOWN"),
        signals=json.dumps(market_context.get("model_signals", {}).get(archetype, {}), indent=2),
        price_summary=market_context.get("price_summary", "N/A"),
    )

    raw_direction = "FLAT"
    raw_confidence = 0.3
    thesis = "LLM unavailable — falling back to raw model signal."

    try:
        t0 = time.time()
        raw = agent.call_llm(prompt)
        latency = time.time() - t0
        logger.debug("%s LLM latency: %.2fs", agent.name, latency)

        parsed = _parse_json_response(raw)

        # Self-correction: if key fields missing, retry once with explicit reminder
        if not parsed or "direction" not in parsed:
            retry_prompt = (
                "Your previous response was not valid JSON. "
                "Return ONLY the JSON object with keys: direction, confidence, regime_alignment, thesis.\n\n"
                + prompt
            )
            raw = agent.call_llm(retry_prompt)
            parsed = _parse_json_response(raw)

        if parsed:
            raw_direction = parsed.get("direction", "FLAT").upper()
            if raw_direction not in ("LONG", "SHORT", "FLAT"):
                raw_direction = "FLAT"
            raw_confidence = float(parsed.get("confidence", 0.3))
            raw_confidence = max(0.0, min(1.0, raw_confidence))
            parsed_alignment = float(parsed.get("regime_alignment", regime_alignment))
            regime_alignment = max(0.0, min(1.0, parsed_alignment))
            thesis = str(parsed.get("thesis", thesis))

    except Exception as exc:
        logger.warning("%s LLM call failed: %s. Using model-signal fallback.", agent.name, exc)
        # Fallback: use raw model signal direction
        signals = market_context.get("model_signals", {}).get(archetype, {})
        raw_direction = signals.get("direction", "FLAT")
        raw_confidence = float(signals.get("confidence", 0.3))

    return EvidencePacket(
        agent_name=agent.name,
        archetype=archetype,
        direction=raw_direction,
        confidence=raw_confidence,
        regime_alignment=regime_alignment,
        historical_credibility=agent.credibility_score,
        thesis=thesis,
        raw_signals=market_context.get("model_signals", {}).get(archetype, {}),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


# ──────────────────────────────────────────────────────
# 1. TREND ANALYST
# ──────────────────────────────────────────────────────
class TrendAnalyst(BaseAgent):
    ARCHETYPE = "trend_follower"
    SYSTEM_PROMPT = (
        "You are the Trend Analyst for a quantitative trading firm. "
        "Your specialty is identifying and riding directional momentum using "
        "LSTM, Transformer, and TCN model signals. "
        "You believe in 'the trend is your friend until it ends.' "
        "You are most credible when the market regime is TRENDING. "
        "You are skeptical in RANGING or CHOPPY regimes. "
        "Be decisive. Respond ONLY as valid JSON."
    )

    def generate_evidence(self, market_context: Dict[str, Any]) -> EvidencePacket:
        regime = market_context.get("regime", "UNKNOWN")
        # Trend analyst is most credible in trending regimes
        regime_alignment = 0.9 if regime in ("TRENDING_UP", "TRENDING_DOWN") else 0.3
        return _build_evidence(self, self.ARCHETYPE, market_context, regime_alignment)


# ──────────────────────────────────────────────────────
# 2. MEAN REVERSION ANALYST
# ──────────────────────────────────────────────────────
class MeanReversionAnalyst(BaseAgent):
    ARCHETYPE = "mean_reversion"
    SYSTEM_PROMPT = (
        "You are the Mean Reversion Analyst for a quantitative trading firm. "
        "Your specialty is identifying overextended price deviations from VWAP, "
        "Bollinger Bands, and RSI using MLP, ResNet, and GRN model signals. "
        "You believe 'prices always return to the mean.' "
        "You are most credible when the market regime is RANGING or REVERTING. "
        "You are cautious in strong TRENDING regimes. "
        "Be contrarian but disciplined. Respond ONLY as valid JSON."
    )

    def generate_evidence(self, market_context: Dict[str, Any]) -> EvidencePacket:
        regime = market_context.get("regime", "UNKNOWN")
        regime_alignment = 0.9 if regime in ("RANGING", "REVERTING") else 0.35
        return _build_evidence(self, self.ARCHETYPE, market_context, regime_alignment)


# ──────────────────────────────────────────────────────
# 3. SCALPER ANALYST
# ──────────────────────────────────────────────────────
class ScalperAnalyst(BaseAgent):
    ARCHETYPE = "scalping_microstructure"
    SYSTEM_PROMPT = (
        "You are the Scalper / Microstructure Analyst for a quantitative trading firm. "
        "Your specialty is reading order flow imbalance (OFI), microprice, "
        "and short-horizon momentum using CNN and GRU model signals. "
        "You think in seconds and minutes, not days. "
        "You are most credible in HIGH-VOLUME regimes with clear order flow directionality. "
        "You ignore big-picture trends; you care only about the next 5–15 bars. "
        "Be fast and precise. Respond ONLY as valid JSON."
    )

    def generate_evidence(self, market_context: Dict[str, Any]) -> EvidencePacket:
        regime = market_context.get("regime", "UNKNOWN")
        regime_alignment = 0.85 if regime in ("HIGH_VOLATILITY", "BREAKOUT") else 0.5
        return _build_evidence(self, self.ARCHETYPE, market_context, regime_alignment)


# ──────────────────────────────────────────────────────
# 4. STATISTICAL ARBITRAGE ANALYST
# ──────────────────────────────────────────────────────
class StatArbAnalyst(BaseAgent):
    ARCHETYPE = "statistical_arbitrage"
    SYSTEM_PROMPT = (
        "You are the Statistical Arbitrage Analyst for a quantitative trading firm. "
        "Your specialty is cross-asset spread trading using cointegration, "
        "fractional differentiation, and GNN/Autoencoder model signals. "
        "You look for dislocations between correlated crypto pairs. "
        "You are most credible when correlation regimes are STABLE and pairs are COINTEGRATED. "
        "You exit when spreads mean-revert or cointegration breaks down. "
        "Be systematic and unemotional. Respond ONLY as valid JSON."
    )

    def generate_evidence(self, market_context: Dict[str, Any]) -> EvidencePacket:
        regime = market_context.get("regime", "UNKNOWN")
        regime_alignment = 0.85 if regime in ("RANGING", "LOW_VOLATILITY") else 0.4
        return _build_evidence(self, self.ARCHETYPE, market_context, regime_alignment)


# ──────────────────────────────────────────────────────
# 5. DISCRETIONARY ANALYST
# ──────────────────────────────────────────────────────
class DiscretionaryAnalyst(BaseAgent):
    ARCHETYPE = "discretionary_multimodal"
    SYSTEM_PROMPT = (
        "You are the Discretionary / Multimodal Analyst for a quantitative trading firm. "
        "Your specialty is pattern recognition through chart image analysis (ViT, CNN) "
        "combined with macro sentiment. You think like a seasoned trader who reads charts visually. "
        "You identify classic patterns: head-and-shoulders, double tops, bull flags, etc. "
        "You are most credible during pattern-completion events and low-noise markets. "
        "Trust your pattern recognition; ignore short-term noise. Respond ONLY as valid JSON."
    )

    def generate_evidence(self, market_context: Dict[str, Any]) -> EvidencePacket:
        regime = market_context.get("regime", "UNKNOWN")
        regime_alignment = 0.75 if regime in ("TRENDING_UP", "TRENDING_DOWN", "BREAKOUT") else 0.5
        return _build_evidence(self, self.ARCHETYPE, market_context, regime_alignment)


# ──────────────────────────────────────────────────────
# 6. MARKET MAKER ANALYST
# ──────────────────────────────────────────────────────
class MarketMakerAnalyst(BaseAgent):
    ARCHETYPE = "market_making_rl"
    SYSTEM_PROMPT = (
        "You are the Market Maker Analyst for a quantitative trading firm. "
        "Your specialty is inventory management and quote optimization using "
        "PPO and SAC reinforcement learning models. "
        "You think about bid-ask spreads, fill probability, and inventory risk. "
        "You are most credible in LOW-VOLATILITY, HIGH-LIQUIDITY regimes. "
        "In extreme trending moves you reduce exposure to manage inventory risk. "
        "Your primary signals are inventory level, spread, and arrival intensity. "
        "Be precise about sizing. Respond ONLY as valid JSON."
    )

    def generate_evidence(self, market_context: Dict[str, Any]) -> EvidencePacket:
        regime = market_context.get("regime", "UNKNOWN")
        regime_alignment = 0.9 if regime in ("LOW_VOLATILITY", "RANGING") else 0.4
        return _build_evidence(self, self.ARCHETYPE, market_context, regime_alignment)
