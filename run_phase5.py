"""
run_phase5.py
CLI runner for ChatTrader.KPai Phase 5 — Multi-Agent Debate System.

Usage:
    python run_phase5.py --symbol BTCUSDT --timeframe 1h
    python run_phase5.py --symbol ETHUSDT --timeframe 5m --slow
    python run_phase5.py --symbol BTCUSDT --timeframe 1h --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows to handle Unicode characters in output
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── path setup ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_bridge import MockModelBridge
from orchestration.debate_engine import DebateEngine
from agents.portfolio_manager import RiskConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase5_runner")


def _random_features(symbol: str) -> dict:
    """Generate random but plausible market features for demo/testing."""
    random.seed(hash(symbol) % 10000)
    slope = random.uniform(-0.008, 0.008)
    return {
        "price_slope_20": slope,
        "zscore_close_64": random.uniform(-2.5, 2.5),
        "atr_14": random.uniform(0.001, 0.012),
        "atr_mean": random.uniform(0.002, 0.008),
        "ema_spread": slope * random.uniform(0.3, 1.5),
        "bb_distance": random.uniform(-1.5, 1.5),
        "bb_width": random.uniform(0.005, 0.05),
        "rsi_14": random.uniform(30, 70),
        "ofi_proxy": random.uniform(-0.5, 0.5),
        "spread_proxy": random.uniform(0.0005, 0.002),
        "vol_regime_code": random.randint(0, 2),
        "spread_z_64": random.uniform(-2.0, 2.0),
        "fracdiff_close_d04": slope * 0.5,
        "pair_correlation": random.uniform(0.5, 0.9),
        "inventory_level": random.uniform(-0.8, 0.8),
        "pattern_score": random.uniform(-0.6, 0.6),
    }


def build_context(symbol: str, timeframe: str, bridge: MockModelBridge) -> dict:
    features = _random_features(symbol)
    model_signals = bridge.get_all_signals(symbol, {"features": features})
    price_summary = bridge.build_price_summary(features)
    return {
        "features": features,
        "model_signals": model_signals,
        "price_summary": price_summary,
    }


def print_result(result) -> None:
    """Pretty-print debate result to console."""
    r = result.to_dict()
    order = r.get("trade_order", {})
    orch = r.get("orchestrator_decision", {})
    shadow = r.get("shadow_critique") or {}

    SEP = "─" * 60

    print(f"\n{SEP}")
    print(f"  DEBATE SESSION: {r['session_id']}")
    print(f"  Symbol: {r['symbol']} | TF: {r['timeframe']} | Path: {r['path']}")
    print(f"  Regime: {r['regime']}")
    print(f"  Latency: {r['total_latency_ms']:.0f}ms")
    print(SEP)

    print("\n  [ANALYST EVIDENCE]")
    for pkt in r.get("packets", []):
        conf_bar = "█" * int(pkt['confidence'] * 10) + "░" * (10 - int(pkt['confidence'] * 10))
        print(f"    {pkt['agent_name']:35s} {pkt['direction']:5s} [{conf_bar}] {pkt['confidence']:.0%}")
        print(f"      → {pkt['thesis'][:80]}…" if len(pkt.get('thesis', '')) > 80 else f"      → {pkt.get('thesis', '')}")

    if shadow:
        print(f"\n  [SHADOW CRITIQUE] rebuttal={shadow.get('rebuttal_strength', 0):.0%}")
        print(f"    Risk: {shadow.get('main_risk', 'N/A')}")
        print(f"    Invalidation: {shadow.get('invalidation_scenario', 'N/A')}")

    print(f"\n  [ORCHESTRATOR]")
    print(f"    Direction:  {orch.get('direction', '?')}")
    print(f"    Consensus:  {orch.get('consensus_score', 0):.0%}")
    print(f"    Thesis:     {orch.get('final_thesis', '')[:120]}")
    print(f"    Key Risk:   {orch.get('key_risk', '')}")

    print(f"\n  [PORTFOLIO MANAGER]")
    print(f"    Action:     {order.get('action', '?')}")
    print(f"    Direction:  {order.get('direction', '?')}")
    print(f"    Size:       {order.get('position_size_pct', 0):.2%} of portfolio")
    print(f"    Stop Loss:  {order.get('stop_loss_pct', 0):.2%}")
    print(f"    Take Prof:  {order.get('take_profit_pct', 0):.2%}")
    print(f"    Risk Score: {order.get('risk_score', 0):.0%}")
    print(f"    Reason:     {order.get('reason', '')[:100]}")
    print(f"\n{SEP}\n")


def main():
    parser = argparse.ArgumentParser(description="ChatTrader.KPai Phase 5 — Debate Runner")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--timeframe", default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--slow", action="store_true", help="Force slow path (full LLM debate)")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls, use mock only")
    parser.add_argument("--model", default="qwen3.5:4b", help="Ollama model (default: qwen3.5:4b)")
    parser.add_argument("--output", default=None, help="Save result JSON to file")
    args = parser.parse_args()

    logger.info("Phase 5 Debate Runner starting...")
    logger.info("Symbol: %s | TF: %s | Path: %s | Model: %s",
                args.symbol, args.timeframe,
                "SLOW (forced)" if args.slow else "AUTO",
                "MOCK (dry-run)" if args.dry_run else args.model)

    bridge = MockModelBridge(seed=42)

    if args.dry_run:
        # Dry run: no Ollama calls, pure model signal aggregation
        logger.info("DRY RUN: LLM calls disabled. Using weighted vote fallback.")
        def _noop(*a, **kw):
            raise RuntimeError("Dry run: Ollama disabled.")
        # Patch the module-level function (used by BaseAgent.call_llm via analysts/shadow/PM)
        import agents.base_agent as ba
        ba._ollama_chat = _noop
        # Also patch debate_engine's own local binding (used directly in _orchestrate)
        import orchestration.debate_engine as de_module
        de_module._ollama_chat = _noop

    engine = DebateEngine(
        risk_config=RiskConfig(),
        ollama_model=args.model,
        max_parallel_analysts=3,
    )

    context = build_context(args.symbol, args.timeframe, bridge)

    result = engine.run_debate(
        symbol=args.symbol,
        timeframe=args.timeframe,
        market_context=context,
        force_slow_path=args.slow,
    )

    print_result(result)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        logger.info("Result saved to %s", out_path)


if __name__ == "__main__":
    main()
