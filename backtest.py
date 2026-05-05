"""
backtest.py
Historical backtest using real aggTrade Parquet data.
Forces SLOW PATH (full LLM debate) on every bar.
Implements recursive learning via automatic error decomposition on trade close.

Usage:
    python backtest.py --parquet Dataset/binance_vision_real/BTCUSDT/aggTrades/2025-10/20251010.parquet
    python backtest.py --parquet ... --timeframe 5m --max-bars 50 --model qwen3.5:4b
    python backtest.py --parquet ... --dry-run   # no Ollama, weighted-vote fallback

Autonomous Debug Protocol:
    If an exception occurs at runtime, it is caught, logged, and the offending bar is
    skipped. Structural bugs (import errors, etc.) must be fixed before re-running.
    Run with --check-only to do a single-bar smoke test before full execution.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path setup ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_feeder import DataFeeder
from real_signal_bridge import RealSignalBridge
from orchestration.debate_engine import DebateEngine
from agents.portfolio_manager import RiskConfig
from agents.journaler import Journaler
from agents.omni_log import OmniLog, STAGE_EVOLVING, STAGE_OUTPUT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest.log", encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger("backtest")


# ─────────────────────────────────────────────
# Position tracking
# ─────────────────────────────────────────────
@dataclass
class Position:
    session_id: str
    symbol: str
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    size_pct: float
    entry_bar: int
    entry_time: str
    stop_loss_pct: float
    take_profit_pct: float
    regime: str
    packets_snapshot: List[Dict] = field(default_factory=list)

    def pnl(self, exit_price: float) -> float:
        """Raw PnL as fraction of entry price."""
        if self.direction == "LONG":
            return (exit_price - self.entry_price) / self.entry_price
        elif self.direction == "SHORT":
            return (self.entry_price - exit_price) / self.entry_price
        return 0.0

    def stop_hit(self, current_price: float) -> bool:
        p = self.pnl(current_price)
        return p <= -abs(self.stop_loss_pct)

    def tp_hit(self, current_price: float) -> bool:
        p = self.pnl(current_price)
        return p >= abs(self.take_profit_pct)


# ─────────────────────────────────────────────
# Backtest state
# ─────────────────────────────────────────────
@dataclass
class BacktestStats:
    total_bars: int = 0
    debates_run: int = 0
    trades_opened: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    flat_decisions: int = 0
    total_pnl: float = 0.0
    peak_equity: float = 1.0
    current_equity: float = 1.0
    max_drawdown: float = 0.0
    errors_caught: int = 0
    credibility_updates: int = 0

    @property
    def win_rate(self) -> float:
        if self.trades_closed == 0:
            return 0.0
        return self.wins / self.trades_closed

    @property
    def net_pnl_pct(self) -> float:
        return self.total_pnl * 100


class BacktestEngine:
    """
    Orchestrates the full backtest loop:
      1. DataFeeder → real market_context
      2. RealSignalBridge → archetype signals
      3. DebateEngine (SLOW PATH forced) → DebateResult
      4. Simulated position management
      5. On trade close → error decomposition → recursive credibility update
    """

    def __init__(
        self,
        parquet_path: str,
        timeframe: str = "5m",
        symbol: str = "BTCUSDT",
        ollama_model: str = "qwen3.5:4b",
        max_bars: Optional[int] = None,
        dry_run: bool = False,
        max_open_positions: int = 1,    # Conservative: 1 at a time for backtest clarity
        freeze_credibility: bool = False,           # True → skip credibility updates (VAL/TEST phase)
        preload_credibilities: Optional[Dict[str, float]] = None,  # Load saved weights
    ) -> None:
        self.freeze_credibility = freeze_credibility
        self.symbol = symbol
        self.timeframe = timeframe
        self.dry_run = dry_run
        self.stats = BacktestStats()
        self.open_positions: List[Position] = []
        self._preload_credibilities = preload_credibilities or {}

        # Journaler and OmniLog
        self.journaler = Journaler()
        self.omni = OmniLog.get_instance()

        # Components
        logger.info("Initializing DataFeeder ...")
        self.feeder = DataFeeder(parquet_path, timeframe=timeframe, symbol=symbol, max_bars=max_bars)

        logger.info("Initializing RealSignalBridge ...")
        self.bridge = RealSignalBridge()

        logger.info("Initializing DebateEngine (force_slow_path=True) ...")

        if dry_run:
            logger.info("DRY RUN: Patching LLM calls to offline fallback.")
            import agents.base_agent as ba
            import orchestration.debate_engine as de_mod
            def _noop(*a, **kw):
                raise RuntimeError("Dry run: Ollama disabled.")
            ba._ollama_chat = _noop
            de_mod._ollama_chat = _noop

        self.engine = DebateEngine(
            risk_config=RiskConfig(
                max_position_pct=0.05,
                max_drawdown_pct=0.10,
                max_open_positions=max_open_positions,
                kelly_fraction=0.25,
            ),
            ollama_model=ollama_model,
            max_parallel_analysts=3,
            enable_journaling=True,
            force_slow_path=True,       # Always full LLM debate
            omni_log=self.omni,
        )

        # Apply pre-loaded credibility weights if provided
        if self._preload_credibilities:
            for analyst in self.engine.analysts:
                if analyst.name in self._preload_credibilities:
                    analyst.credibility_score = self._preload_credibilities[analyst.name]
                    logger.info(
                        "Loaded credibility for %s: %.4f",
                        analyst.name, analyst.credibility_score,
                    )
        if self.freeze_credibility:
            logger.info("freeze_credibility=True — credibility scores will NOT update during this run.")

        # Per-agent learning trace used by split-learn validation/test critique loops.
        self.agent_learning_stats: Dict[str, Dict[str, Any]] = {
            a.name: {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "by_regime": {},
            }
            for a in self.engine.analysts
        }

    # ─────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────
    def run(self) -> BacktestStats:
        logger.info(
            "Starting backtest | %s @ %s | %d usable bars | dry_run=%s",
            self.symbol, self.timeframe, self.feeder.usable_bars, self.dry_run,
        )

        for bar_idx, market_context in self.feeder.iterate():
            self.stats.total_bars += 1
            bar_time = market_context.get("bar_time", "")
            close_price = market_context.get("close_price", 0.0)

            # ── 1. Check open positions for SL/TP ────────────────────
            self._check_positions(close_price, bar_idx, bar_time)

            # ── 2. Check portfolio drawdown ───────────────────────────
            dd = 1.0 - self.stats.current_equity / self.stats.peak_equity
            if dd > 0.10:
                logger.warning("Max drawdown %.2f%% reached — skipping debate this bar", dd * 100)
                continue

            # ── 3. Skip if max positions open ────────────────────────
            if len(self.open_positions) >= self.engine.pm.risk_config.max_open_positions:
                continue

            # ── 4. Build market_context with real signals ─────────────
            try:
                model_signals = self.bridge.get_all_signals(self.symbol, market_context)
                market_context["model_signals"] = model_signals
                if "price_summary" not in market_context:
                    market_context["price_summary"] = self.bridge.build_price_summary(
                        market_context["features"]
                    )

                # ── 5. Run debate ─────────────────────────────────────
                t0 = time.time()
                result = self.engine.run_debate(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    market_context=market_context,
                )
                latency = (time.time() - t0) * 1000
                self.stats.debates_run += 1

                logger.info(
                    "[%s] bar=%d action=%s dir=%s consensus=%.0f%% size=%.2f%% latency=%.0fms",
                    bar_time[:16], bar_idx,
                    result.trade_order.action,
                    result.trade_order.direction,
                    result.orchestrator_decision.get("consensus_score", 0) * 100,
                    result.trade_order.position_size_pct * 100,
                    latency,
                )

                # ── 6. Open position if trade signal ─────────────────
                action = result.trade_order.action
                if action in ("BUY", "SELL") and result.trade_order.position_size_pct > 0:
                    self._open_position(result, bar_idx, close_price, bar_time)
                else:
                    self.stats.flat_decisions += 1

            except KeyboardInterrupt:
                logger.info("Backtest interrupted by user.")
                break
            except Exception as exc:
                self.stats.errors_caught += 1
                logger.error(
                    "Error on bar %d (%s): %s\n%s",
                    bar_idx, bar_time, exc, traceback.format_exc(),
                )
                # Autonomous debug protocol: log and continue (don't crash)
                self.omni.log_message(
                    kind="error", from_agent="Backtest", to_agent="Backtest",
                    stage="ERROR", session_id="runtime",
                    content={"bar": bar_idx, "error": str(exc), "traceback": traceback.format_exc()[-500:]},
                    bar_time=bar_time,
                )
                continue

        # Close any remaining open positions at last bar
        self._close_all_open_positions()

        self._print_summary()
        self._save_report()
        return self.stats

    # ─────────────────────────────────────────
    # Position management
    # ─────────────────────────────────────────
    def _open_position(self, result: Any, bar_idx: int, price: float, bar_time: str) -> None:
        order = result.trade_order
        direction = order.direction

        pos = Position(
            session_id=result.session_id,
            symbol=self.symbol,
            direction=direction,
            entry_price=price,
            size_pct=order.position_size_pct,
            entry_bar=bar_idx,
            entry_time=bar_time,
            stop_loss_pct=max(order.stop_loss_pct, 0.005),
            take_profit_pct=max(order.take_profit_pct, 0.01),
            regime=result.regime,
            packets_snapshot=[p.to_dict() for p in result.packets],
        )
        self.open_positions.append(pos)
        self.stats.trades_opened += 1

        self.omni.log_trade(
            kind="trade_open", session_id=result.session_id,
            symbol=self.symbol, direction=direction,
            price=price, size_pct=order.position_size_pct,
            regime=result.regime,
        )
        logger.info("OPENED %s @ %.2f | SL=%.3f%% TP=%.3f%% session=%s",
                    direction, price, pos.stop_loss_pct * 100,
                    pos.take_profit_pct * 100, result.session_id)

    def _check_positions(self, current_price: float, bar_idx: int, bar_time: str) -> None:
        remaining = []
        for pos in self.open_positions:
            if pos.stop_hit(current_price):
                self._close_position(pos, current_price, bar_idx, bar_time, "STOP_LOSS")
            elif pos.tp_hit(current_price):
                self._close_position(pos, current_price, bar_idx, bar_time, "TAKE_PROFIT")
            else:
                remaining.append(pos)
        self.open_positions = remaining

    def _close_all_open_positions(self) -> None:
        """Close all remaining positions at end of backtest."""
        if not self.open_positions:
            return
        logger.info("Closing %d remaining positions at end of data.", len(self.open_positions))
        for pos in list(self.open_positions):
            # Use entry price as exit (no next bar available at EOF)
            self._close_position(pos, pos.entry_price, -1, "EOF", "EXPIRED")
        self.open_positions = []

    def _close_position(
        self,
        pos: Position,
        exit_price: float,
        bar_idx: int,
        bar_time: str,
        reason: str,
    ) -> None:
        raw_pnl = pos.pnl(exit_price)
        was_profitable = raw_pnl > 0
        pnl_sized = raw_pnl * pos.size_pct

        # Update equity curve
        self.stats.current_equity += pnl_sized
        self.stats.peak_equity = max(self.stats.peak_equity, self.stats.current_equity)
        dd = 1.0 - self.stats.current_equity / self.stats.peak_equity
        self.stats.max_drawdown = max(self.stats.max_drawdown, dd)
        self.stats.total_pnl += pnl_sized
        self.stats.trades_closed += 1
        if was_profitable:
            self.stats.wins += 1
        else:
            self.stats.losses += 1

        logger.info(
            "CLOSED %s | reason=%s | entry=%.2f exit=%.2f | pnl=%.4f%% | equity=%.4f",
            pos.direction, reason, pos.entry_price, exit_price,
            raw_pnl * 100, self.stats.current_equity,
        )

        self.omni.log_trade(
            kind="trade_close", session_id=pos.session_id,
            symbol=self.symbol, direction=pos.direction,
            price=exit_price, size_pct=pos.size_pct,
            pnl=pnl_sized, regime=pos.regime,
        )

        # ── Recursive Learning: Error Decomposition ───────────────────
        self._recursive_learning(pos, raw_pnl, was_profitable, exit_price)

    # ─────────────────────────────────────────
    # Recursive learning
    # ─────────────────────────────────────────
    def _recursive_learning(
        self,
        pos: Position,
        raw_pnl: float,
        was_profitable: bool,
        exit_price: float,
    ) -> None:
        """
        Automatic error decomposition + credibility update on trade close.

        Loss = Signal_Error + Decision_Error + Execution_Error

        Signal_Error:    fraction of loss attributable to model signals being wrong.
                         Measured as disagreement ratio among analyst packets.
        Decision_Error:  fraction attributable to Orchestrator/Shadow over-ruling signals.
                         Measured as divergence between weighted signal vote and final direction.
        Execution_Error: remaining fraction (slippage, timing, SL placement).
        """
        if not pos.packets_snapshot:
            return

        # ── Compute signal_error (analyst disagreement) ──────────────
        n = len(pos.packets_snapshot)
        if n > 0:
            directions = [p["direction"] for p in pos.packets_snapshot]
            majority = max(set(directions), key=directions.count)
            agreement = directions.count(majority) / n
            signal_error = round(1.0 - agreement, 3)          # 0 = unanimous, 1 = random
        else:
            signal_error = 0.5

        # ── Compute decision_error (final direction vs majority signal) ─
        if pos.direction != majority and not was_profitable:
            # Orchestrator overruled majority signal and was wrong
            decision_error = round(min(0.6, abs(raw_pnl) * 5), 3)
        elif pos.direction == majority and not was_profitable:
            # Signals were right direction but still lost — model signal noise
            decision_error = round(min(0.3, abs(raw_pnl) * 3), 3)
        else:
            decision_error = 0.0

        # ── Execution error = residual ────────────────────────────────
        total_loss = signal_error + decision_error
        execution_error = round(max(0.0, min(0.4, abs(raw_pnl) * 8 - total_loss * 0.5)), 3)

        notes = (
            f"exit_price={exit_price:.2f} raw_pnl={raw_pnl:.4f} "
            f"majority_signal={majority} actual_dir={pos.direction}"
        )

        # ── Write to journal ──────────────────────────────────────────
        self.journaler.update_outcome(
            session_id=pos.session_id,
            actual_pnl=raw_pnl,
            was_profitable=was_profitable,
            signal_error=signal_error,
            decision_error=decision_error,
            execution_error=execution_error,
            notes=notes,
        )

        # ── Update analyst credibility scores ─────────────────────────
        for analyst in self.engine.analysts:
            archetype = analyst.ARCHETYPE
            # Find this analyst's packet
            pkt = next((p for p in pos.packets_snapshot if p.get("archetype") == archetype), None)
            if pkt:
                # Was this analyst's signal aligned with the actual profitable direction?
                analyst_direction = pkt.get("direction", "FLAT")
                if was_profitable and analyst_direction == pos.direction:
                    correct = True
                elif not was_profitable and analyst_direction != pos.direction:
                    correct = True      # analyst was right, debaters overruled
                else:
                    correct = False

                self._record_agent_learning(
                    agent_name=analyst.name,
                    regime=pos.regime,
                    correct=correct,
                )

                old_cred = analyst.credibility_score
                if not self.freeze_credibility:
                    analyst.update_credibility(correct)
                    new_cred = analyst.credibility_score
                    self.omni.log_growth(
                        agent_name=analyst.name,
                        session_id=pos.session_id,
                        old_credibility=old_cred,
                        new_credibility=new_cred,
                        regime=pos.regime,
                        was_profitable=was_profitable,
                        error_decomp={
                            "signal_error": signal_error,
                            "decision_error": decision_error,
                            "execution_error": execution_error,
                        },
                    )
                    self.stats.credibility_updates += 1
                else:
                    # Frozen: still log the would-be update for analysis
                    new_cred = old_cred  # no change
                    logger.debug(
                        "[FROZEN] %s credibility stays %.4f (was_correct=%s)",
                        analyst.name, old_cred, correct,
                    )

    def _record_agent_learning(self, agent_name: str, regime: str, correct: bool) -> None:
        """Track per-agent correctness by regime for downstream self-critique."""
        st = self.agent_learning_stats.setdefault(
            agent_name,
            {"total": 0, "correct": 0, "incorrect": 0, "by_regime": {}},
        )
        st["total"] += 1
        if correct:
            st["correct"] += 1
        else:
            st["incorrect"] += 1

        regime_key = regime or "UNKNOWN"
        rst = st["by_regime"].setdefault(
            regime_key,
            {"total": 0, "correct": 0, "incorrect": 0},
        )
        rst["total"] += 1
        if correct:
            rst["correct"] += 1
        else:
            rst["incorrect"] += 1

    # ─────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────
    def _print_summary(self) -> None:
        s = self.stats
        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info("Bars processed:     %d", s.total_bars)
        logger.info("Debates run:        %d", s.debates_run)
        logger.info("Trades opened:      %d", s.trades_opened)
        logger.info("Trades closed:      %d", s.trades_closed)
        logger.info("Win rate:           %.1f%%", s.win_rate * 100)
        logger.info("Net PnL:            %.4f%%", s.net_pnl_pct)
        logger.info("Max Drawdown:       %.2f%%", s.max_drawdown * 100)
        logger.info("Final Equity:       %.4f", s.current_equity)
        logger.info("Flat decisions:     %d", s.flat_decisions)
        logger.info("Errors caught:      %d", s.errors_caught)
        logger.info("Credibility updates:%d", s.credibility_updates)
        logger.info("=" * 60)
        for a in self.engine.analysts:
            logger.info("  %-30s credibility=%.4f", a.name, a.credibility_score)
        logger.info("=" * 60)

    def get_credibilities(self) -> Dict[str, float]:
        """Export current per-agent credibility scores."""
        return {a.name: a.credibility_score for a in self.engine.analysts}

    def get_regime_stats(self) -> Dict[str, Dict]:
        """Export per-regime per-analyst packet stats collected during run."""
        return getattr(self, "_regime_stats", {})

    def get_agent_learning_stats(self) -> Dict[str, Dict[str, Any]]:
        """Export per-agent and per-regime correctness stats with computed accuracy."""
        out: Dict[str, Dict[str, Any]] = {}
        for agent_name, st in self.agent_learning_stats.items():
            total = int(st.get("total", 0))
            correct = int(st.get("correct", 0))
            by_regime = st.get("by_regime", {})

            regime_out: Dict[str, Dict[str, Any]] = {}
            for regime, rst in by_regime.items():
                r_total = int(rst.get("total", 0))
                r_correct = int(rst.get("correct", 0))
                regime_out[regime] = {
                    "total": r_total,
                    "correct": r_correct,
                    "incorrect": int(rst.get("incorrect", 0)),
                    "accuracy": (r_correct / r_total) if r_total > 0 else 0.0,
                }

            out[agent_name] = {
                "total": total,
                "correct": correct,
                "incorrect": int(st.get("incorrect", 0)),
                "accuracy": (correct / total) if total > 0 else 0.0,
                "by_regime": regime_out,
            }
        return out

    def _save_report(self) -> None:
        out = {
            "stats": asdict(self.stats),
            "agent_credibilities": {
                a.name: a.credibility_score for a in self.engine.analysts
            },
        }
        report_path = Path("backtest_report.json")
        report_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Report saved to %s", report_path)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="ChatTrader.KPai Historical Backtest")
    parser.add_argument(
        "--parquet",
        default="Dataset/binance_vision_real/BTCUSDT/aggTrades/2025-10/20251010.parquet",
        help="Path to aggTrade Parquet file",
    )
    parser.add_argument("--timeframe", default="5m", help="Bar timeframe (default: 5m)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--model", default="qwen3.5:4b", help="Ollama model")
    parser.add_argument("--max-bars", type=int, default=None, help="Limit bars for testing")
    parser.add_argument("--dry-run", action="store_true", help="Skip Ollama — weighted-vote fallback")
    parser.add_argument("--check-only", action="store_true", help="Run 1 bar only (smoke test)")
    args = parser.parse_args()

    if args.check_only:
        args.max_bars = 1
        args.dry_run = True
        logger.info("CHECK ONLY mode: running 1 bar dry-run smoke test.")

    engine = BacktestEngine(
        parquet_path=args.parquet,
        timeframe=args.timeframe,
        symbol=args.symbol,
        ollama_model=args.model,
        max_bars=args.max_bars,
        dry_run=args.dry_run,
    )
    engine.run()


if __name__ == "__main__":
    main()
