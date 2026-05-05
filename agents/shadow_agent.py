"""
agents/shadow_agent.py
The Shadow Agent (Devil's Advocate) for ChatTrader.KPai Phase 5.

Role: Systematically attack analyst reasoning to prevent groupthink.
The Shadow Agent does NOT generate its own directional signal.
It returns a "rebuttal score" [0.0 – 1.0] indicating how strongly
the consensus should be challenged, plus a critique thesis.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from agents.base_agent import BaseAgent, EvidencePacket, _ollama_chat, _parse_json_response

logger = logging.getLogger(__name__)


class ShadowAgent(BaseAgent):
    """
    Devil's Advocate. Receives the full list of analyst EvidencePackets
    and the current consensus direction, then challenges the reasoning.
    """

    ARCHETYPE = "shadow"
    SYSTEM_PROMPT = (
        "You are the Shadow Agent — the Devil's Advocate of a quantitative trading firm. "
        "Your SOLE purpose is to stress-test the collective analyst consensus. "
        "You must find the weakest points, ignored risks, and potential invalidation scenarios. "
        "You do NOT generate a directional trade recommendation. "
        "You are only satisfied when you have found a genuine flaw. "
        "If the consensus is actually sound, admit it — but search hard first. "
        "Respond ONLY as valid JSON."
    )

    _CRITIQUE_SCHEMA = """
Respond ONLY with this JSON:
{
  "rebuttal_strength": <float 0.0-1.0>,
  "main_risk": "<one sentence: the biggest flaw in the consensus>",
  "invalidation_scenario": "<one sentence: what market condition would break this thesis>",
  "critique_thesis": "<2-3 sentences overall critique>"
}
Where rebuttal_strength = 0.0 means consensus is solid, 1.0 means consensus is deeply flawed.
"""

    def generate_evidence(self, market_context: Dict[str, Any]) -> EvidencePacket:
        """Not used — Shadow Agent uses critique_consensus instead."""
        raise NotImplementedError("Use ShadowAgent.critique_consensus() instead.")

    def critique_consensus(
        self,
        packets: List[EvidencePacket],
        consensus_direction: str,
        market_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyse the analyst evidence packets and return a critique dict.

        Returns:
            {
              "rebuttal_strength": float,
              "main_risk": str,
              "invalidation_scenario": str,
              "critique_thesis": str,
            }
        """
        # Build summary of analyst stances
        analyst_summary = "\n".join([
            f"- {p.agent_name} ({p.archetype}): {p.direction} @ confidence={p.confidence:.2f} | '{p.thesis}'"
            for p in packets
        ])

        prompt = (
            f"Current consensus: {consensus_direction}\n"
            f"Market regime: {market_context.get('regime', 'UNKNOWN')}\n"
            f"Symbol: {market_context.get('symbol', 'BTCUSDT')}\n\n"
            f"Analyst stances:\n{analyst_summary}\n\n"
            f"Attack this consensus. Find flaws, risks, and ignored factors.\n"
            + self._CRITIQUE_SCHEMA
        )

        rebuttal_strength = 0.5
        main_risk = "Unable to complete critique."
        invalidation_scenario = "Unknown."
        critique_thesis = "Shadow Agent LLM call failed."

        try:
            t0 = time.time()
            raw = self.call_llm(prompt)
            latency = time.time() - t0
            logger.debug("ShadowAgent LLM latency: %.2fs", latency)

            parsed = _parse_json_response(raw)

            # Self-correct if response malformed
            if not parsed or "rebuttal_strength" not in parsed:
                retry_prompt = (
                    "Your response was not valid JSON. "
                    "Return ONLY the JSON object with keys: rebuttal_strength, main_risk, "
                    "invalidation_scenario, critique_thesis.\n\n" + prompt
                )
                raw = self.call_llm(retry_prompt)
                parsed = _parse_json_response(raw)

            if parsed:
                rebuttal_strength = float(parsed.get("rebuttal_strength", 0.5))
                rebuttal_strength = max(0.0, min(1.0, rebuttal_strength))
                main_risk = str(parsed.get("main_risk", main_risk))
                invalidation_scenario = str(parsed.get("invalidation_scenario", invalidation_scenario))
                critique_thesis = str(parsed.get("critique_thesis", critique_thesis))

        except Exception as exc:
            logger.warning("ShadowAgent critique failed: %s", exc)

        return {
            "rebuttal_strength": rebuttal_strength,
            "main_risk": main_risk,
            "invalidation_scenario": invalidation_scenario,
            "critique_thesis": critique_thesis,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
