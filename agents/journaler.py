"""
agents/journaler.py
Journaler Agent for ChatTrader.KPai Phase 5.

Responsibilities:
  1. Persist every debate session to JSONL (hot memory)
  2. Append actual trade outcomes and error attribution
  3. Update agent credibility scores based on outcomes
  4. Provide ground-truth recall for pre-trade hypothesis checking
  5. Enforce the Loss Decomposition: Signal_Error + Decision_Error + Execution_Error
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base_agent import EvidencePacket

logger = logging.getLogger(__name__)

TZ_UTC7 = timezone(timedelta(hours=7))
JOURNAL_DIR = Path("agents/journal")


def _now_iso() -> str:
    return datetime.now(TZ_UTC7).isoformat()


class Journaler:
    """
    Persistent memory layer for the multi-agent debate system.

    Journal structure (JSONL):
      Each line = one debate session record.
      After trade closes, outcome is appended via update_outcome().
    """

    def __init__(self, journal_dir: Optional[Path] = None) -> None:
        self.journal_dir = journal_dir or JOURNAL_DIR
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._session_file = self._get_daily_file()

    def _get_daily_file(self) -> Path:
        today = datetime.now(TZ_UTC7).strftime("%Y-%m-%d")
        return self.journal_dir / f"{today}-debate-journal.jsonl"

    def _ensure_daily_file(self) -> None:
        """Rotate to today's file if date has changed."""
        expected = self._get_daily_file()
        if expected != self._session_file:
            self._session_file = expected

    def record_debate(
        self,
        session_id: str,
        symbol: str,
        timeframe: str,
        regime: str,
        packets: List[EvidencePacket],
        shadow_critique: Optional[Dict[str, Any]],
        orchestrator_decision: Dict[str, Any],
        trade_order: Dict[str, Any],
        path: str = "SLOW",
    ) -> None:
        """
        Write a complete debate session record to the journal.
        Called immediately after every debate cycle.
        """
        self._ensure_daily_file()

        record = {
            "session_id": session_id,
            "timestamp": _now_iso(),
            "symbol": symbol,
            "timeframe": timeframe,
            "regime": regime,
            "path": path,
            "analyst_evidence": [p.to_dict() for p in packets],
            "shadow_critique": shadow_critique,
            "orchestrator_decision": orchestrator_decision,
            "trade_order": trade_order,
            "outcome": None,           # Filled later by update_outcome()
            "error_decomposition": None,
        }

        with open(self._session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Journaler: recorded session %s -> %s", session_id, self._session_file.name)

    def update_outcome(
        self,
        session_id: str,
        actual_pnl: float,
        was_profitable: bool,
        signal_error: float,
        decision_error: float,
        execution_error: float,
        notes: str = "",
    ) -> bool:
        """
        Find a session record by ID and append the actual outcome + error attribution.
        Uses the current day's file first, then searches recent files.

        Loss = Signal_Error + Decision_Error + Execution_Error
        All error values are in [0.0 – 1.0] (0 = no error, 1 = full error).
        """
        error_decomp = {
            "actual_pnl": actual_pnl,
            "was_profitable": was_profitable,
            "signal_error": signal_error,
            "decision_error": decision_error,
            "execution_error": execution_error,
            "total_loss_score": signal_error + decision_error + execution_error,
            "notes": notes,
            "recorded_at": _now_iso(),
        }

        # Search recent journal files (last 7 days)
        files = sorted(self.journal_dir.glob("*-debate-journal.jsonl"), reverse=True)[:7]

        for filepath in files:
            lines = filepath.read_text(encoding="utf-8").splitlines()
            updated_lines = []
            found = False
            for line in lines:
                try:
                    rec = json.loads(line)
                    if rec.get("session_id") == session_id:
                        rec["outcome"] = {"actual_pnl": actual_pnl, "was_profitable": was_profitable}
                        rec["error_decomposition"] = error_decomp
                        found = True
                except json.JSONDecodeError:
                    pass
                updated_lines.append(json.dumps(rec, ensure_ascii=False))

            if found:
                filepath.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
                logger.info("Journaler: updated outcome for session %s", session_id)
                return True

        logger.warning("Journaler: session_id %s not found in recent journals.", session_id)
        return False

    def recall_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return the most recent debate session records (newest first).
        Used by agents before a debate to review past performance.
        """
        self._ensure_daily_file()
        records: List[Dict[str, Any]] = []

        files = sorted(self.journal_dir.glob("*-debate-journal.jsonl"), reverse=True)[:3]
        for filepath in files:
            try:
                lines = filepath.read_text(encoding="utf-8").splitlines()
                for line in reversed(lines):
                    if len(records) >= limit:
                        break
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            except FileNotFoundError:
                pass
            if len(records) >= limit:
                break

        return records[:limit]

    def get_agent_error_summary(self, agent_name: str, last_n: int = 50) -> Dict[str, Any]:
        """
        Summarize a specific agent's recent performance for credibility updates.
        Returns: { wins, losses, avg_signal_error, avg_decision_error }
        """
        records = self.recall_recent(last_n)
        wins = 0
        losses = 0
        signal_errors = []
        decision_errors = []

        for rec in records:
            outcome = rec.get("outcome")
            error_decomp = rec.get("error_decomposition")
            if not outcome:
                continue

            # Check if this agent participated in the debate
            for pkt in rec.get("analyst_evidence", []):
                if pkt.get("agent_name") == agent_name:
                    orch_dir = rec.get("orchestrator_decision", {}).get("direction", "FLAT")
                    agent_dir = pkt.get("direction", "FLAT")
                    agent_agreed = agent_dir == orch_dir

                    if outcome.get("was_profitable"):
                        if agent_agreed:
                            wins += 1
                        # Agent was right to agree
                    else:
                        if agent_agreed:
                            losses += 1
                        # Agent was wrong to agree

                    if error_decomp:
                        signal_errors.append(error_decomp.get("signal_error", 0))
                        decision_errors.append(error_decomp.get("decision_error", 0))

        return {
            "agent_name": agent_name,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / max(wins + losses, 1),
            "avg_signal_error": sum(signal_errors) / max(len(signal_errors), 1),
            "avg_decision_error": sum(decision_errors) / max(len(decision_errors), 1),
            "sample_count": wins + losses,
        }
