"""
agents/omni_log.py
Thread-safe Omni Interaction Log — mirrors AgentAGIv2/omni_log.py pattern.

Records agent-to-agent messages, debate stages, and growth events.
Used by the Streamlit dashboard for real-time Glass Brain and Omni Feed display.

Event kinds:
  debate_start      — new session begins
  analyst_input     — analyst receives context
  analyst_output    — analyst emits EvidencePacket
  shadow_input      — shadow receives consensus
  shadow_output     — shadow emits critique
  orchestrator_input  — orchestrator receives all packets
  orchestrator_output — orchestrator emits final direction
  pm_input          — portfolio manager receives directive
  pm_output         — portfolio manager emits TradeOrder
  credibility_update  — agent credibility score updated after trade closes
  trade_open        — backtest opens a position
  trade_close       — backtest closes a position (includes PnL)
  error             — any exception captured
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TZ_UTC7 = timezone(timedelta(hours=7))
_LOCK = threading.Lock()

# Stage constants (Glass Brain 5-stage observability)
STAGE_INPUT     = "INPUT"
STAGE_THINKING  = "THINKING"
STAGE_FEELING   = "FEELING"    # used by Shadow/PM for critique/risk appraisal
STAGE_EVOLVING  = "EVOLVING"   # credibility updates
STAGE_OUTPUT    = "OUTPUT"

_DEFAULT_DIR = Path("agents/omni_log")


def _now_iso() -> str:
    return datetime.now(TZ_UTC7).isoformat()


class OmniLog:
    """
    Thread-safe append-only event log for the debate system.

    Two backing files (rotated daily):
      omni_messages.jsonl  — all agent interaction messages
      growth_log.jsonl     — credibility/evolution events only
    """

    _instance: Optional["OmniLog"] = None

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self.log_dir = log_dir or _DEFAULT_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._today = self._date_str()
        self._msg_file = self._msg_path()
        self._growth_file = self._growth_path()

    # ── Singleton factory ─────────────────────────────────────────────
    @classmethod
    def get_instance(cls, log_dir: Optional[Path] = None) -> "OmniLog":
        if cls._instance is None:
            cls._instance = cls(log_dir)
        return cls._instance

    # ── Internal ──────────────────────────────────────────────────────
    def _date_str(self) -> str:
        return datetime.now(TZ_UTC7).strftime("%Y-%m-%d")

    def _msg_path(self) -> Path:
        return self.log_dir / f"{self._today}-omni-messages.jsonl"

    def _growth_path(self) -> Path:
        return self.log_dir / f"{self._today}-growth-log.jsonl"

    def _rotate(self) -> None:
        today = self._date_str()
        if today != self._today:
            self._today = today
            self._msg_file = self._msg_path()
            self._growth_file = self._growth_path()

    def _append(self, filepath: Path, record: Dict[str, Any]) -> None:
        with _LOCK:
            self._rotate()
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Public write API ──────────────────────────────────────────────
    def log_message(
        self,
        kind: str,
        from_agent: str,
        to_agent: str,
        stage: str,
        session_id: str,
        content: Any,
        bar_time: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Log an agent interaction event."""
        record = {
            "timestamp": _now_iso(),
            "kind": kind,
            "from": from_agent,
            "to": to_agent,
            "stage": stage,
            "session_id": session_id,
            "content": content,
        }
        if bar_time:
            record["bar_time"] = bar_time
        if latency_ms is not None:
            record["latency_ms"] = round(latency_ms, 1)

        self._append(self._msg_file, record)

    def log_growth(
        self,
        agent_name: str,
        session_id: str,
        old_credibility: float,
        new_credibility: float,
        regime: str,
        was_profitable: bool,
        error_decomp: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a credibility evolution event."""
        record = {
            "timestamp": _now_iso(),
            "kind": "credibility_update",
            "stage": STAGE_EVOLVING,
            "agent": agent_name,
            "session_id": session_id,
            "old_credibility": round(old_credibility, 4),
            "new_credibility": round(new_credibility, 4),
            "delta": round(new_credibility - old_credibility, 4),
            "regime": regime,
            "was_profitable": was_profitable,
            "error_decomp": error_decomp or {},
        }
        self._append(self._growth_file, record)
        self._append(self._msg_file, record)

    def log_trade(
        self,
        kind: str,        # "trade_open" or "trade_close"
        session_id: str,
        symbol: str,
        direction: str,
        price: float,
        size_pct: float,
        pnl: Optional[float] = None,
        regime: Optional[str] = None,
    ) -> None:
        """Log a simulated trade event."""
        record = {
            "timestamp": _now_iso(),
            "kind": kind,
            "stage": STAGE_OUTPUT,
            "session_id": session_id,
            "symbol": symbol,
            "direction": direction,
            "price": price,
            "size_pct": size_pct,
        }
        if pnl is not None:
            record["pnl"] = round(pnl, 6)
        if regime:
            record["regime"] = regime

        self._append(self._msg_file, record)

    # ── Public read API ───────────────────────────────────────────────
    def read_messages(self, limit: int = 100, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return recent messages from today's log, newest last."""
        self._rotate()
        results: List[Dict[str, Any]] = []
        try:
            lines = self._msg_file.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                try:
                    rec = json.loads(line)
                    if session_id and rec.get("session_id") != session_id:
                        continue
                    results.append(rec)
                    if len(results) >= limit:
                        break
                except json.JSONDecodeError:
                    pass
        except FileNotFoundError:
            pass
        return list(reversed(results))

    def read_growth_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent growth events (credibility updates), newest last."""
        self._rotate()
        results: List[Dict[str, Any]] = []
        try:
            lines = self._growth_file.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                try:
                    results.append(json.loads(line))
                    if len(results) >= limit:
                        break
                except json.JSONDecodeError:
                    pass
        except FileNotFoundError:
            pass
        return list(reversed(results))

    def read_all_days(self, last_n_days: int = 3, limit: int = 200) -> List[Dict[str, Any]]:
        """Read messages across multiple days (for history panel)."""
        files = sorted(self.log_dir.glob("*-omni-messages.jsonl"), reverse=True)[:last_n_days]
        results: List[Dict[str, Any]] = []
        for f in files:
            try:
                for line in f.read_text(encoding="utf-8").splitlines():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            except FileNotFoundError:
                pass
        return sorted(results, key=lambda r: r.get("timestamp", ""))[-limit:]
