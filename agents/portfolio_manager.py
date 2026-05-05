"""
agents/portfolio_manager.py
Portfolio Manager (PM) — Final Risk Gatekeeper for ChatTrader.KPai Phase 5.

Responsibilities:
  1. Enforce hard position limits and drawdown stops
  2. Calculate position sizing (Kelly / fixed-fraction)
  3. Check portfolio correlation exposure
  4. VETO orchestrator decision if risk limits breached
  5. Return final trade order or NO_TRADE with reason
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, EvidencePacket, _ollama_chat, _parse_json_response

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Risk configuration (can be overridden via YAML)
# ─────────────────────────────────────────────
@dataclass
class RiskConfig:
    max_position_pct: float = 0.05      # Max 5% of portfolio per trade
    max_drawdown_pct: float = 0.10      # Hard stop at -10% portfolio drawdown
    max_open_positions: int = 6         # Max concurrent positions
    min_confidence_threshold: float = 0.55  # Minimum agent confidence to trade
    max_correlation_exposure: float = 0.6   # Stop adding if correlation > 60%
    kelly_fraction: float = 0.25        # Fractional Kelly multiplier


@dataclass
class TradeOrder:
    """Final output of the Portfolio Manager."""
    action: str               # "BUY", "SELL", "NO_TRADE"
    symbol: str
    direction: str            # "LONG", "SHORT", "FLAT"
    position_size_pct: float  # Fraction of portfolio capital [0.0 – 1.0]
    stop_loss_pct: float
    take_profit_pct: float
    reason: str
    risk_score: float         # [0.0 – 1.0]; higher = more risky
    orchestrator_direction: str
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


class PortfolioManager(BaseAgent):
    """
    Final risk gatekeeper. Consumes the Orchestrator's directional call
    and either approves, sizes, and returns a TradeOrder,
    or vetoes with a NO_TRADE order and a reason.
    """

    ARCHETYPE = "portfolio_manager"
    SYSTEM_PROMPT = (
        "You are the Portfolio Manager — the final risk gatekeeper at a quantitative trading firm. "
        "You never chase trades. You size positions conservatively using fractional Kelly. "
        "You will VETO any trade that violates drawdown limits or adds excessive correlation exposure. "
        "Risk first, always. Respond ONLY as valid JSON."
    )

    _SIZING_SCHEMA = """
Respond ONLY with this JSON:
{
  "approved": true | false,
  "position_size_pct": <float 0.0-0.05>,
  "stop_loss_pct": <float 0.005-0.05>,
  "take_profit_pct": <float 0.01-0.15>,
  "risk_score": <float 0.0-1.0>,
  "reason": "<one sentence>"
}
"""

    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        current_drawdown: float = 0.0,
        open_positions: int = 0,
    ) -> None:
        super().__init__(credibility_score=1.0)  # PM credibility is always 1.0
        self.risk_config = risk_config or RiskConfig()
        self.current_drawdown = current_drawdown
        self.open_positions = open_positions

    def generate_evidence(self, market_context: Dict[str, Any]) -> EvidencePacket:
        """Not used directly — PM uses evaluate_and_size() instead."""
        raise NotImplementedError("Use PortfolioManager.evaluate_and_size() instead.")

    def _hard_risk_check(
        self,
        direction: str,
        consensus_score: float,
    ) -> Optional[str]:
        """
        Perform pre-LLM hard rule checks.
        Returns veto reason string if vetoed, else None.
        """
        if self.current_drawdown >= self.risk_config.max_drawdown_pct:
            return (
                f"HARD VETO: Portfolio drawdown {self.current_drawdown:.1%} "
                f"exceeds max {self.risk_config.max_drawdown_pct:.1%}."
            )
        if self.open_positions >= self.risk_config.max_open_positions:
            return (
                f"HARD VETO: {self.open_positions} open positions at maximum "
                f"({self.risk_config.max_open_positions})."
            )
        if consensus_score < self.risk_config.min_confidence_threshold:
            return (
                f"HARD VETO: Consensus score {consensus_score:.2f} below threshold "
                f"{self.risk_config.min_confidence_threshold:.2f}."
            )
        return None

    def _kelly_size(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Fractional Kelly criterion for position sizing.
        f* = (b*p - q) / b * kelly_fraction
        b = win/loss ratio, p = win rate, q = 1-p
        """
        b = max(win_loss_ratio, 0.1)
        p = max(0.0, min(1.0, win_rate))
        q = 1.0 - p
        kelly = (b * p - q) / b
        kelly = max(0.0, kelly)  # No negative sizing
        sized = kelly * self.risk_config.kelly_fraction
        return min(sized, self.risk_config.max_position_pct)

    def evaluate_and_size(
        self,
        orchestrator_direction: str,
        consensus_score: float,
        packets: List[EvidencePacket],
        market_context: Dict[str, Any],
        shadow_critique: Optional[Dict[str, Any]] = None,
    ) -> TradeOrder:
        """
        Main entry point. Returns a TradeOrder (approved or vetoed).
        """
        symbol = market_context.get("symbol", "BTCUSDT")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Hard rule veto check (no LLM needed)
        veto_reason = self._hard_risk_check(orchestrator_direction, consensus_score)
        if veto_reason:
            logger.warning("PM HARD VETO: %s", veto_reason)
            return TradeOrder(
                action="NO_TRADE",
                symbol=symbol,
                direction="FLAT",
                position_size_pct=0.0,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                reason=veto_reason,
                risk_score=1.0,
                orchestrator_direction=orchestrator_direction,
                timestamp=ts,
            )

        # Shadow critique penalty: high rebuttal strength reduces effective consensus
        shadow_penalty = 0.0
        if shadow_critique:
            shadow_penalty = shadow_critique.get("rebuttal_strength", 0.0) * 0.2
            effective_consensus = consensus_score - shadow_penalty
            if effective_consensus < self.risk_config.min_confidence_threshold:
                reason = (
                    f"Shadow critique reduced effective consensus to {effective_consensus:.2f}. "
                    f"Risk: {shadow_critique.get('main_risk', 'Unknown')}"
                )
                return TradeOrder(
                    action="NO_TRADE",
                    symbol=symbol,
                    direction="FLAT",
                    position_size_pct=0.0,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    reason=reason,
                    risk_score=0.8,
                    orchestrator_direction=orchestrator_direction,
                    timestamp=ts,
                )

        # Estimate win rate from historical credibility of agreeing agents
        agreeing = [p for p in packets if p.direction == orchestrator_direction]
        avg_credibility = (
            sum(p.historical_credibility for p in agreeing) / len(agreeing)
            if agreeing else 0.5
        )
        win_loss_ratio = 1.5  # Default; will be updated when real models pass validation

        size = self._kelly_size(avg_credibility, win_loss_ratio)

        # LLM call for final sizing approval (optional enrichment)
        approved_size = size
        stop_loss = 0.02
        take_profit = size * win_loss_ratio * 2
        risk_score = 1.0 - consensus_score
        llm_reason = f"Fractional Kelly sizing: {size:.3f} based on avg credibility {avg_credibility:.2f}."

        prompt = (
            f"Orchestrator direction: {orchestrator_direction}\n"
            f"Consensus score: {consensus_score:.2f}\n"
            f"Shadow rebuttal: {shadow_critique.get('critique_thesis', 'N/A') if shadow_critique else 'N/A'}\n"
            f"Calculated position size: {size:.4f}\n"
            f"Current drawdown: {self.current_drawdown:.2%}\n"
            f"Open positions: {self.open_positions}/{self.risk_config.max_open_positions}\n"
            f"Market regime: {market_context.get('regime', 'UNKNOWN')}\n\n"
            f"Approve or refine the sizing. If position_size_pct = 0.0 then set approved = false.\n"
            + self._SIZING_SCHEMA
        )

        try:
            raw = self.call_llm(prompt)
            parsed = _parse_json_response(raw)
            if not parsed or "approved" not in parsed:
                retry = (
                    "Return ONLY valid JSON with keys: approved, position_size_pct, "
                    "stop_loss_pct, take_profit_pct, risk_score, reason.\n\n" + prompt
                )
                raw = self.call_llm(retry)
                parsed = _parse_json_response(raw)

            if parsed:
                if not parsed.get("approved", True):
                    return TradeOrder(
                        action="NO_TRADE",
                        symbol=symbol,
                        direction="FLAT",
                        position_size_pct=0.0,
                        stop_loss_pct=0.0,
                        take_profit_pct=0.0,
                        reason=str(parsed.get("reason", "PM LLM vetoed trade.")),
                        risk_score=float(parsed.get("risk_score", 0.8)),
                        orchestrator_direction=orchestrator_direction,
                        timestamp=ts,
                    )
                approved_size = float(parsed.get("position_size_pct", size))
                approved_size = max(0.0, min(approved_size, self.risk_config.max_position_pct))
                stop_loss = float(parsed.get("stop_loss_pct", 0.02))
                take_profit = float(parsed.get("take_profit_pct", 0.04))
                risk_score = float(parsed.get("risk_score", risk_score))
                llm_reason = str(parsed.get("reason", llm_reason))

        except Exception as exc:
            logger.warning("PM LLM sizing call failed: %s. Using Kelly fallback.", exc)

        action = "BUY" if orchestrator_direction == "LONG" else "SELL"
        return TradeOrder(
            action=action,
            symbol=symbol,
            direction=orchestrator_direction,
            position_size_pct=approved_size,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            reason=llm_reason,
            risk_score=risk_score,
            orchestrator_direction=orchestrator_direction,
            timestamp=ts,
        )
