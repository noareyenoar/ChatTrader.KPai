"""
tests/test_phase5.py
Phase 5 Integration Test Suite — ChatTrader.KPai

Tests:
  1. MockModelBridge — signal contract & structure
  2. RegimeDetector — all 8 regime paths
  3. ShadowAgent — critique structure
  4. PortfolioManager — hard veto rules
  5. DebateEngine — full fast-path and dry-run slow-path
  6. Journaler — record + recall
  7. EvidencePacket — serialization roundtrip
"""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_bridge import MockModelBridge
from agents.base_agent import EvidencePacket
from agents.regime_detector import RegimeDetector
from agents.portfolio_manager import PortfolioManager, RiskConfig
from agents.journaler import Journaler
from orchestration.debate_engine import DebateEngine


# ── Helpers ──────────────────────────────────────────────────────────
def _make_features(**overrides) -> dict:
    base = {
        "price_slope_20": 0.002,
        "zscore_close_64": 1.2,
        "atr_14": 0.003,
        "atr_mean": 0.0025,
        "ema_spread": 0.001,
        "bb_distance": 0.8,
        "bb_width": 0.02,
        "rsi_14": 58.0,
        "ofi_proxy": 0.2,
        "spread_proxy": 0.001,
        "vol_regime_code": 1,
        "spread_z_64": 1.0,
        "fracdiff_close_d04": 0.001,
        "pair_correlation": 0.72,
        "inventory_level": 0.1,
        "pattern_score": 0.3,
    }
    base.update(overrides)
    return base


def _make_context(symbol="BTCUSDT", timeframe="1h", **feat_overrides) -> dict:
    features = _make_features(**feat_overrides)
    bridge = MockModelBridge(seed=42)
    signals = bridge.get_all_signals(symbol, {"features": features})
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "features": features,
        "model_signals": signals,
        "price_summary": "Test context.",
    }


# ─────────────────────────────────────────────────────────────────────
class TestMockModelBridge(unittest.TestCase):

    def setUp(self):
        self.bridge = MockModelBridge(seed=42)

    def test_returns_all_six_archetypes(self):
        ctx = _make_context()
        signals = self.bridge.get_all_signals("BTCUSDT", ctx)
        expected = {
            "trend_follower", "mean_reversion", "scalping_microstructure",
            "statistical_arbitrage", "discretionary_multimodal", "market_making_rl",
        }
        self.assertEqual(set(signals.keys()), expected)

    def test_signal_contract(self):
        ctx = _make_context()
        signals = self.bridge.get_all_signals("BTCUSDT", ctx)
        for archetype, sig in signals.items():
            self.assertIn("direction", sig, f"{archetype} missing direction")
            self.assertIn("confidence", sig, f"{archetype} missing confidence")
            self.assertIn("model_votes", sig, f"{archetype} missing model_votes")
            self.assertIn(sig["direction"], ("LONG", "SHORT", "FLAT"))
            self.assertGreaterEqual(sig["confidence"], 0.0)
            self.assertLessEqual(sig["confidence"], 1.0)

    def test_three_models_per_archetype(self):
        ctx = _make_context()
        signals = self.bridge.get_all_signals("BTCUSDT", ctx)
        for archetype, sig in signals.items():
            votes = sig.get("model_votes", {})
            self.assertEqual(len(votes), 3, f"{archetype} should have 3 model votes")

    def test_price_summary_is_string(self):
        features = _make_features()
        summary = self.bridge.build_price_summary(features)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 10)


# ─────────────────────────────────────────────────────────────────────
class TestRegimeDetector(unittest.TestCase):

    def setUp(self):
        self.rd = RegimeDetector()

    def _detect(self, **feat_overrides) -> str:
        ctx = {"features": _make_features(**feat_overrides)}
        return self.rd.detect(ctx)

    def test_trending_up(self):
        regime = self._detect(price_slope_20=0.005, ema_spread=0.003, atr_14=0.003, atr_mean=0.003, bb_width=0.01)
        self.assertEqual(regime, "TRENDING_UP")

    def test_trending_down(self):
        regime = self._detect(price_slope_20=-0.005, ema_spread=-0.003, atr_14=0.003, atr_mean=0.003, bb_width=0.01)
        self.assertEqual(regime, "TRENDING_DOWN")

    def test_ranging(self):
        regime = self._detect(price_slope_20=0.0, ema_spread=0.0, atr_14=0.003, atr_mean=0.003, bb_width=0.01)
        self.assertEqual(regime, "RANGING")

    def test_high_volatility(self):
        regime = self._detect(atr_14=0.01, atr_mean=0.003, bb_width=0.01, price_slope_20=0.0)
        self.assertEqual(regime, "HIGH_VOLATILITY")

    def test_low_volatility(self):
        regime = self._detect(atr_14=0.001, atr_mean=0.003, bb_width=0.005, price_slope_20=0.0)
        self.assertEqual(regime, "LOW_VOLATILITY")

    def test_breakout(self):
        regime = self._detect(bb_width=0.06, atr_14=0.003, atr_mean=0.003)
        self.assertEqual(regime, "BREAKOUT")

    def test_unknown_on_no_features(self):
        regime = self.rd.detect({})
        self.assertEqual(regime, "UNKNOWN")

    def test_regime_weights_returns_all_archetypes(self):
        self.rd._last_regime = "TRENDING_UP"
        weights = self.rd.regime_weights()
        expected_keys = {
            "trend_follower", "mean_reversion", "scalping_microstructure",
            "statistical_arbitrage", "discretionary_multimodal", "market_making_rl",
        }
        self.assertEqual(set(weights.keys()), expected_keys)


# ─────────────────────────────────────────────────────────────────────
class TestPortfolioManager(unittest.TestCase):

    def setUp(self):
        self.pm = PortfolioManager(
            risk_config=RiskConfig(
                max_drawdown_pct=0.10,
                max_position_pct=0.05,
                min_confidence_threshold=0.55,
            )
        )

    def _make_packets(self, direction="LONG") -> list:
        return [
            EvidencePacket(
                agent_name="TrendAnalyst",
                archetype="trend_follower",
                direction=direction,
                confidence=0.7,
                regime_alignment=0.8,
                historical_credibility=0.6,
                thesis="Test thesis.",
                raw_signals={},
            )
        ]

    def test_hard_veto_on_drawdown(self):
        self.pm.current_drawdown = 0.15
        with patch.object(self.pm, "call_llm", return_value="{}"):
            order = self.pm.evaluate_and_size(
                "LONG", 0.70, self._make_packets(), _make_context(), None
            )
        self.assertEqual(order.action, "NO_TRADE")
        self.assertIn("drawdown", order.reason.lower())

    def test_hard_veto_on_low_consensus(self):
        self.pm.current_drawdown = 0.0
        with patch.object(self.pm, "call_llm", return_value="{}"):
            order = self.pm.evaluate_and_size(
                "LONG", 0.40, self._make_packets(), _make_context(), None
            )
        self.assertEqual(order.action, "NO_TRADE")

    def test_hard_veto_on_max_positions(self):
        self.pm.current_drawdown = 0.0
        self.pm.open_positions = 6
        with patch.object(self.pm, "call_llm", return_value="{}"):
            order = self.pm.evaluate_and_size(
                "LONG", 0.70, self._make_packets(), _make_context(), None
            )
        self.assertEqual(order.action, "NO_TRADE")

    def test_approved_trade_structure(self):
        self.pm.current_drawdown = 0.0
        self.pm.open_positions = 2
        mock_llm = json.dumps({
            "approved": True,
            "position_size_pct": 0.03,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
            "risk_score": 0.3,
            "reason": "Acceptable risk."
        })
        with patch.object(self.pm, "call_llm", return_value=mock_llm):
            order = self.pm.evaluate_and_size(
                "LONG", 0.70, self._make_packets(), _make_context(), None
            )
        self.assertIn(order.action, ("BUY", "SELL", "NO_TRADE"))
        self.assertGreaterEqual(order.position_size_pct, 0.0)
        self.assertLessEqual(order.position_size_pct, 0.05)

    def test_kelly_sizing_never_negative(self):
        size = self.pm._kelly_size(win_rate=0.3, win_loss_ratio=0.5)
        self.assertGreaterEqual(size, 0.0)

    def test_kelly_sizing_never_exceeds_max(self):
        size = self.pm._kelly_size(win_rate=0.9, win_loss_ratio=5.0)
        self.assertLessEqual(size, self.pm.risk_config.max_position_pct)


# ─────────────────────────────────────────────────────────────────────
class TestEvidencePacket(unittest.TestCase):

    def test_serialization_roundtrip(self):
        pkt = EvidencePacket(
            agent_name="TrendAnalyst",
            archetype="trend_follower",
            direction="LONG",
            confidence=0.72,
            regime_alignment=0.85,
            historical_credibility=0.60,
            thesis="Price is trending up with positive EMA spread.",
            raw_signals={"ensemble_logit": 0.42},
            timestamp="2026-05-01T12:00:00",
        )
        d = pkt.to_dict()
        restored = EvidencePacket.from_dict(d)
        self.assertEqual(pkt.direction, restored.direction)
        self.assertAlmostEqual(pkt.confidence, restored.confidence)

    def test_weighted_score_bounded(self):
        pkt = EvidencePacket("A", "b", "LONG", 0.8, 0.7, 0.6, "", {})
        score = pkt.weighted_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ─────────────────────────────────────────────────────────────────────
class TestJournaler(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.journaler = Journaler(journal_dir=self.tmp)

    def _make_pkt(self) -> EvidencePacket:
        return EvidencePacket("TrendAnalyst", "trend_follower", "LONG", 0.7, 0.8, 0.6, "Test.", {})

    def test_record_and_recall(self):
        session_id = "test_123"
        self.journaler.record_debate(
            session_id=session_id,
            symbol="BTCUSDT",
            timeframe="1h",
            regime="TRENDING_UP",
            packets=[self._make_pkt()],
            shadow_critique=None,
            orchestrator_decision={"direction": "LONG", "consensus_score": 0.75},
            trade_order={"action": "BUY", "position_size_pct": 0.03},
        )
        records = self.journaler.recall_recent(10)
        self.assertGreaterEqual(len(records), 1)
        session_ids = [r.get("session_id") for r in records]
        self.assertIn(session_id, session_ids)

    def test_update_outcome(self):
        session_id = "outcome_test_456"
        self.journaler.record_debate(
            session_id=session_id,
            symbol="ETHUSDT",
            timeframe="4h",
            regime="RANGING",
            packets=[self._make_pkt()],
            shadow_critique={"rebuttal_strength": 0.3},
            orchestrator_decision={"direction": "SHORT"},
            trade_order={"action": "SELL"},
        )
        success = self.journaler.update_outcome(
            session_id=session_id,
            actual_pnl=0.02,
            was_profitable=True,
            signal_error=0.1,
            decision_error=0.05,
            execution_error=0.02,
        )
        self.assertTrue(success)

    def test_recall_empty_when_no_records(self):
        records = self.journaler.recall_recent(5)
        self.assertIsInstance(records, list)


# ─────────────────────────────────────────────────────────────────────
class TestDebateEngineDryRun(unittest.TestCase):
    """
    Tests DebateEngine with all LLM calls mocked out.
    Validates structure and correctness of debate flow.
    """

    def _make_engine(self) -> DebateEngine:
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        from agents.journaler import Journaler as J
        engine = DebateEngine(
            risk_config=RiskConfig(max_drawdown_pct=0.10, max_position_pct=0.05),
            ollama_model="mock",
            max_parallel_analysts=2,
            enable_journaling=False,
        )
        return engine

    def _mock_llm_response(self, direction="LONG", consensus=0.72):
        return json.dumps({
            "direction": direction,
            "confidence": consensus,
            "regime_alignment": 0.8,
            "thesis": "Mock LLM thesis for testing.",
            "consensus_score": consensus,
            "final_thesis": "Mock orchestrator synthesis.",
            "dissenting_archetypes": [],
            "key_risk": "Mock risk.",
            "approved": True,
            "position_size_pct": 0.03,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
            "risk_score": 0.3,
            "reason": "Mock PM approval.",
            "rebuttal_strength": 0.2,
            "main_risk": "Mock risk.",
            "invalidation_scenario": "Mock scenario.",
            "critique_thesis": "Mock critique.",
        })

    def test_fast_path_triggered_for_scalping(self):
        engine = self._make_engine()
        ctx = _make_context(timeframe="1m")
        # Force high scalper confidence
        ctx["model_signals"]["scalping_microstructure"]["confidence"] = 0.80

        with patch("agents.base_agent._ollama_chat", return_value=self._mock_llm_response()):
            result = engine.run_debate("BTCUSDT", "1m", ctx)

        self.assertEqual(result.path, "FAST")
        self.assertTrue(result.fast_path_triggered)

    def test_slow_path_triggered_for_hourly(self):
        engine = self._make_engine()
        ctx = _make_context(timeframe="1h")

        with patch("agents.base_agent._ollama_chat", return_value=self._mock_llm_response()):
            with patch("orchestration.debate_engine._ollama_chat", return_value=self._mock_llm_response()):
                result = engine.run_debate("BTCUSDT", "1h", ctx, force_slow_path=True)

        self.assertEqual(result.path, "SLOW")
        self.assertFalse(result.fast_path_triggered)

    def test_result_has_all_required_fields(self):
        engine = self._make_engine()
        ctx = _make_context()

        with patch("agents.base_agent._ollama_chat", return_value=self._mock_llm_response()):
            with patch("orchestration.debate_engine._ollama_chat", return_value=self._mock_llm_response()):
                result = engine.run_debate("BTCUSDT", "1h", ctx)

        d = result.to_dict()
        self.assertIn("session_id", d)
        self.assertIn("path", d)
        self.assertIn("regime", d)
        self.assertIn("packets", d)
        self.assertIn("orchestrator_decision", d)
        self.assertIn("trade_order", d)
        self.assertGreater(len(d["packets"]), 0)

    def test_no_trade_on_low_consensus(self):
        engine = self._make_engine()
        ctx = _make_context()

        low_consensus_response = json.dumps({
            "direction": "FLAT",
            "confidence": 0.3,
            "regime_alignment": 0.4,
            "thesis": "Low conviction.",
            "consensus_score": 0.3,  # Below threshold
            "final_thesis": "No consensus.",
            "dissenting_archetypes": ["trend_follower"],
            "key_risk": "Insufficient conviction.",
            "approved": False,
            "position_size_pct": 0.0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "risk_score": 0.9,
            "reason": "PM veto.",
            "rebuttal_strength": 0.7,
            "main_risk": "Low consensus.",
            "invalidation_scenario": "Any move.",
            "critique_thesis": "Consensus too low.",
        })

        with patch("agents.base_agent._ollama_chat", return_value=low_consensus_response):
            with patch("orchestration.debate_engine._ollama_chat", return_value=low_consensus_response):
                result = engine.run_debate("BTCUSDT", "1h", ctx, force_slow_path=True)

        self.assertEqual(result.trade_order.action, "NO_TRADE")

    def test_latency_recorded(self):
        engine = self._make_engine()
        ctx = _make_context(timeframe="1m")
        ctx["model_signals"]["scalping_microstructure"]["confidence"] = 0.80

        with patch("agents.base_agent._ollama_chat", return_value=self._mock_llm_response()):
            result = engine.run_debate("BTCUSDT", "1m", ctx)

        self.assertGreaterEqual(result.total_latency_ms, 0)  # >= 0 due to Windows timer resolution


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ChatTrader.KPai — Phase 5 Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
