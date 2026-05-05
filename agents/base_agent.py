"""
agents/base_agent.py
Base class for all ChatTrader.KPai agents.
All agents share: Ollama LLM access, evidence packet generation,
credibility tracking, and retry/self-correction logic.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Ollama client configuration
# ─────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"   # Fastest available model for trading latency
LLM_TIMEOUT = 60               # seconds per request
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5            # exponential base (seconds)


# ─────────────────────────────────────────────
# Core data structure: Evidence Packet
# ─────────────────────────────────────────────
@dataclass
class EvidencePacket:
    """
    Standardized signal packet emitted by every analyst agent.
    The Orchestrator ingests these to run the debate.
    """
    agent_name: str
    archetype: str
    direction: str           # "LONG", "SHORT", or "FLAT"
    confidence: float        # [0.0 – 1.0]
    regime_alignment: float  # how well current regime suits this archetype [0.0 – 1.0]
    historical_credibility: float  # rolling win rate in current regime [0.0 – 1.0]
    thesis: str              # plain-English reasoning summary (1–3 sentences)
    raw_signals: Dict[str, Any] = field(default_factory=dict)  # model output dict
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidencePacket":
        return cls(**d)

    def weighted_score(self) -> float:
        """Composite credibility-weighted confidence signal."""
        return self.confidence * self.regime_alignment * self.historical_credibility


# ─────────────────────────────────────────────
# Ollama call helper (shared by all agents)
# ─────────────────────────────────────────────
def _ollama_chat(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> str:
    """
    Single Ollama /api/chat call. Returns the assistant message content.
    Raises RuntimeError after MAX_RETRIES exhausted.
    """
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 4096},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"]
            return content
        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF ** attempt
            logger.warning("Ollama timeout (attempt %d/%d). Retrying in %.1fs...", attempt, MAX_RETRIES, wait)
            if attempt < MAX_RETRIES:
                time.sleep(wait)
        except Exception as exc:
            wait = RETRY_BACKOFF ** attempt
            logger.warning("Ollama error (attempt %d/%d): %s. Retrying in %.1fs...", attempt, MAX_RETRIES, exc, wait)
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    raise RuntimeError(f"Ollama call failed after {MAX_RETRIES} attempts.")


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from an LLM response.
    Handles markdown code fences and stray prose.
    Returns {} on failure so callers can self-correct.
    """
    # Strip markdown fences
    text = raw.strip()
    if "```" in text:
        start = text.find("{", text.find("```"))
        end = text.rfind("}") + 1
        text = text[start:end]
    else:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failure: %s | raw=%r", e, raw[:200])
        return {}


# ─────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────
class BaseAgent:
    """
    Abstract base for all 9 agents in the ChatTrader.KPai multi-agent system.
    Subclasses override:
      - ARCHETYPE: str
      - SYSTEM_PROMPT: str
      - generate_evidence(market_context) -> EvidencePacket
    """

    ARCHETYPE: str = "base"
    SYSTEM_PROMPT: str = (
        "You are a quantitative trading analyst. "
        "Return responses ONLY as valid JSON."
    )
    MODEL: str = DEFAULT_MODEL

    def __init__(self, credibility_score: float = 0.5) -> None:
        self.credibility_score = credibility_score        # Updated by Journaler
        self._call_count = 0
        self._correct_count = 0

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def call_llm(self, user_message: str) -> str:
        """Call Ollama with this agent's system prompt."""
        return _ollama_chat(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=user_message,
            model=self.MODEL,
        )

    def update_credibility(self, was_correct: bool) -> None:
        """Bayesian-style credibility update (exponential moving average)."""
        self._call_count += 1
        if was_correct:
            self._correct_count += 1
        # EMA with α = 0.1
        target = 1.0 if was_correct else 0.0
        self.credibility_score = 0.9 * self.credibility_score + 0.1 * target

    def generate_evidence(self, market_context: Dict[str, Any]) -> EvidencePacket:
        """Override in subclasses."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name} credibility={self.credibility_score:.3f}>"
