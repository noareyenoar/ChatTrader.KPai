"""
orchestration/debate_engine.py
Dual-Path Debate Engine — ChatTrader.KPai Phase 5

FAST PATH:  Bypasses LLM debate for scalping/low-latency scenarios.
            Direct signal aggregation. Target: < 200ms.

SLOW PATH:  Full LLM debate loop (6 analysts → shadow critique → orchestrator synthesis).
            Target: < 5 seconds. If consensus < 60%, outputs NO_TRADE.

Anti-Overthinking Rule:
    If Consensus Score < 0.60 during Slow Path → strict NO_TRADE output.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import EvidencePacket, _ollama_chat, _parse_json_response, DEFAULT_MODEL
from agents.analyst_agents import (
    TrendAnalyst, MeanReversionAnalyst, ScalperAnalyst,
    StatArbAnalyst, DiscretionaryAnalyst, MarketMakerAnalyst,
)
from agents.shadow_agent import ShadowAgent
from agents.portfolio_manager import PortfolioManager, RiskConfig, TradeOrder
from agents.regime_detector import RegimeDetector
from agents.journaler import Journaler
from agents.omni_log import OmniLog, STAGE_INPUT, STAGE_THINKING, STAGE_FEELING, STAGE_EVOLVING, STAGE_OUTPUT

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FAST_PATH_CONFIDENCE_THRESHOLD = 0.65   # Use fast path if scalper confidence > this
SLOW_PATH_CONSENSUS_MINIMUM = 0.60      # Anti-overthinking: NO_TRADE if below
FAST_PATH_TARGET_MS = 200
SLOW_PATH_TARGET_MS = 5000

_ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Orchestrator of a quantitative trading debate at a hedge fund.
You have received evidence packets from 6 specialist analysts and a shadow critique.
Your job: synthesize all evidence, make the final directional call, and assign a consensus score.
You must output ONLY valid JSON. If confidence is below 0.60, you MUST output FLAT direction.
"""

_ORCHESTRATOR_SCHEMA = """
Respond ONLY with this JSON:
{
  "direction": "LONG" | "SHORT" | "FLAT",
  "consensus_score": <float 0.0-1.0>,
  "final_thesis": "<2-3 sentences synthesizing the debate>",
  "dissenting_archetypes": ["<archetype name>", ...],
  "key_risk": "<one sentence: main risk to this call>"
}
"""


# ─────────────────────────────────────────────
# Debate Result container
# ─────────────────────────────────────────────
@dataclass
class DebateResult:
    session_id: str
    path: str                          # "FAST" or "SLOW"
    symbol: str
    timeframe: str
    regime: str
    packets: List[EvidencePacket]
    shadow_critique: Optional[Dict[str, Any]]
    orchestrator_decision: Dict[str, Any]
    trade_order: TradeOrder
    total_latency_ms: float
    fast_path_triggered: bool
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["packets"] = [p.to_dict() for p in self.packets]
        d["trade_order"] = self.trade_order.to_dict()
        return d


# ─────────────────────────────────────────────
# Debate Engine
# ─────────────────────────────────────────────
class DebateEngine:
    """
    Main entry point for the Phase 5 multi-agent system.

    Instantiate once; call run_debate() per market event.

    Parameters:
        risk_config: Portfolio Manager risk parameters
        ollama_model: Which Ollama model to use for LLM reasoning
        max_parallel_analysts: Concurrent analyst LLM calls (ThreadPoolExecutor)
    """

    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        ollama_model: str = DEFAULT_MODEL,
        max_parallel_analysts: int = 3,
        enable_journaling: bool = True,
        force_slow_path: bool = False,
        omni_log: Optional[OmniLog] = None,
    ) -> None:
        self.model = ollama_model
        self.max_parallel = max_parallel_analysts
        self.force_slow_path = force_slow_path  # Backtest: always run full debate

        # Agent roster
        self.analysts = [
            TrendAnalyst(),
            MeanReversionAnalyst(),
            ScalperAnalyst(),
            StatArbAnalyst(),
            DiscretionaryAnalyst(),
            MarketMakerAnalyst(),
        ]
        self.shadow = ShadowAgent()
        self.pm = PortfolioManager(risk_config=risk_config)
        self.regime_detector = RegimeDetector()
        self.journaler = Journaler() if enable_journaling else None
        self.omni = omni_log or OmniLog.get_instance()

        logger.info(
            "DebateEngine initialized | model=%s | analysts=%d | parallel=%d | force_slow=%s",
            self.model, len(self.analysts), self.max_parallel, self.force_slow_path,
        )

    # ─────────────────────────────────────────
    # Core: run debate
    # ─────────────────────────────────────────
    def run_debate(
        self,
        symbol: str,
        timeframe: str,
        market_context: Dict[str, Any],
        force_slow_path: bool = False,
    ) -> DebateResult:
        """
        Execute one full debate cycle (fast or slow path).

        market_context must contain:
          - features: dict of indicator values
          - model_signals: dict from RealSignalBridge / MockModelBridge get_all_signals()
          - price_summary: str (optional)
        """
        session_id = str(uuid.uuid4())[:8]
        t_start = time.time()
        bar_time = market_context.get("bar_time")

        # ── 1. Detect regime ─────────────────────────────────────────
        regime = self.regime_detector.detect(market_context)
        market_context["regime"] = regime
        market_context["symbol"] = symbol
        market_context["timeframe"] = timeframe

        # ── OmniLog: debate start ─────────────────────────────────────
        self.omni.log_message(
            kind="debate_start", from_agent="DebateEngine", to_agent="ALL",
            stage=STAGE_INPUT, session_id=session_id,
            content={"symbol": symbol, "timeframe": timeframe, "regime": regime,
                     "price_summary": market_context.get("price_summary", "")},
            bar_time=bar_time,
        )

        # ── 2. Route: fast or slow path ──────────────────────────────
        # self.force_slow_path (instance) or call-time force_slow_path
        _force_slow = force_slow_path or self.force_slow_path
        scalper_signals = market_context.get("model_signals", {}).get("scalping_microstructure", {})
        scalper_confidence = scalper_signals.get("confidence", 0.0)

        use_fast_path = (
            not _force_slow
            and timeframe in ("1m", "3m", "5m")
            and scalper_confidence >= FAST_PATH_CONFIDENCE_THRESHOLD
        )

        if use_fast_path:
            result = self._fast_path(session_id, symbol, timeframe, regime, market_context, t_start)
        else:
            result = self._slow_path(session_id, symbol, timeframe, regime, market_context, t_start)

        # ── 3. Journal the result ────────────────────────────────────
        if self.journaler:
            self.journaler.record_debate(
                session_id=session_id,
                symbol=symbol,
                timeframe=timeframe,
                regime=regime,
                packets=result.packets,
                shadow_critique=result.shadow_critique,
                orchestrator_decision=result.orchestrator_decision,
                trade_order=result.trade_order.to_dict(),
                path=result.path,
            )

        elapsed_ms = (time.time() - t_start) * 1000
        result.total_latency_ms = elapsed_ms
        result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        # ── OmniLog: final output ─────────────────────────────────────
        self.omni.log_message(
            kind="orchestrator_output", from_agent="DebateEngine", to_agent="PortfolioManager",
            stage=STAGE_OUTPUT, session_id=session_id,
            content={
                "path": result.path,
                "direction": result.trade_order.direction,
                "action": result.trade_order.action,
                "consensus": result.orchestrator_decision.get("consensus_score", 0),
                "position_size_pct": result.trade_order.position_size_pct,
                "latency_ms": round(elapsed_ms, 1),
            },
            bar_time=bar_time, latency_ms=elapsed_ms,
        )

        # ── 4. Latency gate logging ──────────────────────────────────
        if result.fast_path_triggered and elapsed_ms > FAST_PATH_TARGET_MS:
            logger.warning(
                "FAST PATH LATENCY FAILURE: %.0fms (target <200ms) | session=%s",
                elapsed_ms, session_id
            )
        elif not result.fast_path_triggered and elapsed_ms > SLOW_PATH_TARGET_MS:
            logger.warning(
                "SLOW PATH LATENCY FAILURE: %.0fms (target <5000ms) | session=%s",
                elapsed_ms, session_id
            )
        else:
            logger.info(
                "Debate complete | path=%s | %.0fms | direction=%s | session=%s",
                result.path, elapsed_ms,
                result.trade_order.direction, session_id
            )

        return result

    # ─────────────────────────────────────────
    # FAST PATH
    # ─────────────────────────────────────────
    def _fast_path(
        self,
        session_id: str,
        symbol: str,
        timeframe: str,
        regime: str,
        market_context: Dict[str, Any],
        t_start: float,
    ) -> DebateResult:
        """
        No LLM debate. Aggregate raw model signals directly.
        Scalper signal dominates. Use regime weights to adjust.
        """
        all_signals = market_context.get("model_signals", {})
        regime_weights = self.regime_detector.regime_weights()

        vote_score = {"LONG": 0.0, "SHORT": 0.0, "FLAT": 0.0}
        for archetype, sig in all_signals.items():
            weight = regime_weights.get(archetype, 0.5)
            conf = sig.get("confidence", 0.5)
            direction = sig.get("direction", "FLAT")
            vote_score[direction] += weight * conf

        consensus_direction = max(vote_score, key=vote_score.__getitem__)
        total_weight = sum(vote_score.values())
        consensus_score = vote_score[consensus_direction] / max(total_weight, 1e-8)

        # Build minimal evidence packets (no LLM)
        packets = []
        for analyst in self.analysts:
            archetype = analyst.ARCHETYPE
            sig = all_signals.get(archetype, {})
            pkt = EvidencePacket(
                agent_name=analyst.name,
                archetype=archetype,
                direction=sig.get("direction", "FLAT"),
                confidence=sig.get("confidence", 0.3),
                regime_alignment=regime_weights.get(archetype, 0.5),
                historical_credibility=analyst.credibility_score,
                thesis=f"[FAST PATH] Raw model signal: {sig.get('direction', 'FLAT')} @ {sig.get('confidence', 0.3):.2f}",
                raw_signals=sig,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            packets.append(pkt)

        orch_decision = {
            "direction": consensus_direction,
            "consensus_score": round(consensus_score, 4),
            "final_thesis": f"[FAST PATH] Signal aggregation: {consensus_direction} @ {consensus_score:.2f}",
            "dissenting_archetypes": [],
            "key_risk": "Fast path bypasses debate — use only for scalping timeframes.",
            "path": "FAST",
        }

        trade_order = self.pm.evaluate_and_size(
            orchestrator_direction=consensus_direction,
            consensus_score=consensus_score,
            packets=packets,
            market_context=market_context,
            shadow_critique=None,
        )

        return DebateResult(
            session_id=session_id,
            path="FAST",
            symbol=symbol,
            timeframe=timeframe,
            regime=regime,
            packets=packets,
            shadow_critique=None,
            orchestrator_decision=orch_decision,
            trade_order=trade_order,
            total_latency_ms=0.0,
            fast_path_triggered=True,
        )

    # ─────────────────────────────────────────
    # SLOW PATH (full LLM debate)
    # ─────────────────────────────────────────
    def _slow_path(
        self,
        session_id: str,
        symbol: str,
        timeframe: str,
        regime: str,
        market_context: Dict[str, Any],
        t_start: float,
    ) -> DebateResult:
        """
        Full debate: parallel analyst calls → shadow critique → orchestrator synthesis.
        """

        # ── Step 1: Parallel analyst evidence generation ─────────────
        packets = self._gather_analyst_evidence_parallel(market_context, session_id)

        # ── Step 2: Shadow critique ───────────────────────────────────
        # Quick consensus for shadow input
        directions = [p.direction for p in packets]
        raw_consensus = max(set(directions), key=directions.count)

        self.omni.log_message(
            kind="shadow_input", from_agent="DebateEngine", to_agent="ShadowAgent",
            stage=STAGE_FEELING, session_id=session_id,
            content={"raw_consensus": raw_consensus,
                     "analyst_directions": [{"agent": p.agent_name, "dir": p.direction, "conf": p.confidence}
                                            for p in packets]},
            bar_time=market_context.get("bar_time"),
        )
        shadow_critique = self.shadow.critique_consensus(packets, raw_consensus, market_context)
        self.omni.log_message(
            kind="shadow_output", from_agent="ShadowAgent", to_agent="Orchestrator",
            stage=STAGE_FEELING, session_id=session_id,
            content=shadow_critique,
            bar_time=market_context.get("bar_time"),
        )

        # ── Step 3: Orchestrator synthesis ───────────────────────────
        self.omni.log_message(
            kind="orchestrator_input", from_agent="ShadowAgent", to_agent="Orchestrator",
            stage=STAGE_THINKING, session_id=session_id,
            content={"n_packets": len(packets), "shadow_rebuttal": shadow_critique.get("rebuttal_strength", 0)},
            bar_time=market_context.get("bar_time"),
        )
        orch_decision = self._orchestrate(packets, shadow_critique, market_context)
        final_direction = orch_decision.get("direction", "FLAT")
        consensus_score = orch_decision.get("consensus_score", 0.0)

        # ── Anti-Overthinking Rule ────────────────────────────────────
        if consensus_score < SLOW_PATH_CONSENSUS_MINIMUM:
            logger.info(
                "Anti-overthinking triggered: consensus %.2f < %.2f -> NO_TRADE | session=%s",
                consensus_score, SLOW_PATH_CONSENSUS_MINIMUM, session_id,
            )
            orch_decision["direction"] = "FLAT"
            orch_decision["final_thesis"] = (
                f"[NO TRADE] Consensus score {consensus_score:.2f} below threshold "
                f"{SLOW_PATH_CONSENSUS_MINIMUM}. Analysis paralysis prevented."
            )
            final_direction = "FLAT"

        # ── Step 4: Portfolio Manager sizing ─────────────────────────
        self.omni.log_message(
            kind="pm_input", from_agent="Orchestrator", to_agent="PortfolioManager",
            stage=STAGE_THINKING, session_id=session_id,
            content={"direction": final_direction, "consensus": consensus_score},
            bar_time=market_context.get("bar_time"),
        )
        trade_order = self.pm.evaluate_and_size(
            orchestrator_direction=final_direction,
            consensus_score=consensus_score,
            packets=packets,
            market_context=market_context,
            shadow_critique=shadow_critique,
        )
        self.omni.log_message(
            kind="pm_output", from_agent="PortfolioManager", to_agent="Backtest",
            stage=STAGE_OUTPUT, session_id=session_id,
            content={"action": trade_order.action, "direction": trade_order.direction,
                     "size_pct": trade_order.position_size_pct, "reason": trade_order.reason},
            bar_time=market_context.get("bar_time"),
        )

        return DebateResult(
            session_id=session_id,
            path="SLOW",
            symbol=symbol,
            timeframe=timeframe,
            regime=regime,
            packets=packets,
            shadow_critique=shadow_critique,
            orchestrator_decision=orch_decision,
            trade_order=trade_order,
            total_latency_ms=0.0,
            fast_path_triggered=False,
        )

    def _gather_analyst_evidence_parallel(
        self, market_context: Dict[str, Any], session_id: str = ""
    ) -> List[EvidencePacket]:
        """Call all 6 analyst agents in parallel using a thread pool."""
        packets: List[EvidencePacket] = []
        bar_time = market_context.get("bar_time")

        def _call_analyst(analyst) -> EvidencePacket:
            self.omni.log_message(
                kind="analyst_input", from_agent="DebateEngine", to_agent=analyst.name,
                stage=STAGE_INPUT, session_id=session_id,
                content={"archetype": analyst.ARCHETYPE, "credibility": analyst.credibility_score,
                         "price_summary": market_context.get("price_summary", "")[:120]},
                bar_time=bar_time,
            )
            try:
                pkt = analyst.generate_evidence(market_context)
                self.omni.log_message(
                    kind="analyst_output", from_agent=analyst.name, to_agent="Orchestrator",
                    stage=STAGE_THINKING, session_id=session_id,
                    content={"direction": pkt.direction, "confidence": pkt.confidence,
                             "regime_alignment": pkt.regime_alignment,
                             "thesis": pkt.thesis[:120] if pkt.thesis else ""},
                    bar_time=bar_time,
                )
                return pkt
            except Exception as exc:
                logger.error("Analyst %s failed: %s", analyst.name, exc)
                self.omni.log_message(
                    kind="error", from_agent=analyst.name, to_agent="DebateEngine",
                    stage=STAGE_THINKING, session_id=session_id,
                    content={"error": str(exc)},
                    bar_time=bar_time,
                )
                return EvidencePacket(
                    agent_name=analyst.name,
                    archetype=analyst.ARCHETYPE,
                    direction="FLAT",
                    confidence=0.0,
                    regime_alignment=0.0,
                    historical_credibility=analyst.credibility_score,
                    thesis=f"ERROR: {exc}",
                    raw_signals={},
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {executor.submit(_call_analyst, a): a for a in self.analysts}
            for future in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    packets.append(future.result())
                except Exception as exc:
                    logger.error("Analyst future failed: %s", exc)

        return packets

    def _orchestrate(
        self,
        packets: List[EvidencePacket],
        shadow_critique: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Orchestrator synthesizes analyst debate into final direction."""
        regime_weights = self.regime_detector.regime_weights()

        # Build evidence summary for the LLM
        evidence_lines = []
        for p in packets:
            rw = regime_weights.get(p.archetype, 0.5)
            weighted_conf = p.confidence * p.regime_alignment * rw
            evidence_lines.append(
                f"- {p.agent_name} ({p.archetype}): {p.direction} | "
                f"conf={p.confidence:.2f} | regime_alignment={p.regime_alignment:.2f} | "
                f"regime_weight={rw:.2f} | weighted={weighted_conf:.3f}\n  thesis: {p.thesis}"
            )

        shadow_summary = (
            f"Shadow critique (rebuttal={shadow_critique.get('rebuttal_strength', 0):.2f}): "
            f"{shadow_critique.get('critique_thesis', 'N/A')}"
        )

        prompt = (
            f"Symbol: {market_context.get('symbol', '?')} | "
            f"Regime: {market_context.get('regime', 'UNKNOWN')} | "
            f"Timeframe: {market_context.get('timeframe', '?')}\n\n"
            f"Analyst evidence:\n" + "\n".join(evidence_lines) + "\n\n"
            f"{shadow_summary}\n\n"
            f"As Orchestrator, synthesize this debate into a final trading call.\n"
            f"If consensus is below 0.60, you MUST set direction = FLAT.\n"
            + _ORCHESTRATOR_SCHEMA
        )

        # Default fallback
        result = {
            "direction": "FLAT",
            "consensus_score": 0.0,
            "final_thesis": "Orchestrator LLM call failed — defaulting to FLAT.",
            "dissenting_archetypes": [],
            "key_risk": "LLM unavailable.",
        }

        try:
            raw = _ollama_chat(
                system_prompt=_ORCHESTRATOR_SYSTEM_PROMPT,
                user_message=prompt,
                model=self.model,
            )
            parsed = _parse_json_response(raw)

            # Self-correct if malformed
            if not parsed or "direction" not in parsed:
                retry = (
                    "Your response was not valid JSON. Return ONLY the JSON object with keys: "
                    "direction, consensus_score, final_thesis, dissenting_archetypes, key_risk.\n\n"
                    + prompt
                )
                raw = _ollama_chat(
                    system_prompt=_ORCHESTRATOR_SYSTEM_PROMPT,
                    user_message=retry,
                    model=self.model,
                )
                parsed = _parse_json_response(raw)

            if parsed:
                direction = parsed.get("direction", "FLAT").upper()
                if direction not in ("LONG", "SHORT", "FLAT"):
                    direction = "FLAT"
                result = {
                    "direction": direction,
                    "consensus_score": max(0.0, min(1.0, float(parsed.get("consensus_score", 0.0)))),
                    "final_thesis": str(parsed.get("final_thesis", "")),
                    "dissenting_archetypes": list(parsed.get("dissenting_archetypes", [])),
                    "key_risk": str(parsed.get("key_risk", "")),
                }

        except Exception as exc:
            logger.error("Orchestrator LLM call failed: %s", exc)
            # Compute weighted vote as fallback
            direction, score = self._weighted_vote_fallback(packets, regime_weights)
            result["direction"] = direction
            result["consensus_score"] = score
            result["final_thesis"] = f"[FALLBACK] Weighted vote: {direction} @ {score:.2f}"

        result["path"] = "SLOW"
        return result

    def _weighted_vote_fallback(
        self,
        packets: List[EvidencePacket],
        regime_weights: Dict[str, float],
    ) -> Tuple[str, float]:
        """Pure math fallback when LLM is unavailable."""
        votes: Dict[str, float] = {"LONG": 0.0, "SHORT": 0.0, "FLAT": 0.0}
        for p in packets:
            rw = regime_weights.get(p.archetype, 0.5)
            score = p.confidence * p.regime_alignment * rw * p.historical_credibility
            votes[p.direction] = votes.get(p.direction, 0.0) + score

        total = sum(votes.values())
        best = max(votes, key=votes.__getitem__)
        consensus = votes[best] / max(total, 1e-8)
        return best, consensus
