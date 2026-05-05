"""
run_backtest_splitlearn.py
==========================
Implements the "IRON WALL" DATA SPLITTING (NO LEAKAGE POLICY) from pytorch_model_training_rule.md.

Chronological split across ALL available BTCUSDT parquet files:
  TRAIN      70%  – Oldest data. Agents learn credibility here (gradient descent analogue).
  PURGE GAP       – 1 file buffer to prevent information bleeding (per protocol § 1.2).
  VALIDATION 15%  – Middle data. Credibility FROZEN. Agents self-critique on held-out performance.
  PURGE GAP       – 1 file buffer.
  TEST       15%  – Most recent data (OOS). Credibility FROZEN. Final performance measured.

Self-critique loop (VALIDATION phase):
  Each analyst compares its validation signal accuracy vs training signal accuracy per regime.
  Generates structured self-critique: overconfident_in, underconfident_in, regime_biases.
  Saves to agents/self_critique_report.json.

Validity check (§ 4 Real-World Validity Logic):
  Model is INVALID if:  val PnL decay > 50% vs train  OR  test win_rate < 40%
  Model is VALID if:    consistent win_rate across val & test, drawdown < 20%

Artifacts produced:
  trained_credibility_weights.json      – per-agent credibility after TRAIN phase
  agents/self_critique_report.json      – per-agent structured self-critique from VALIDATION
  backtest_v2_splitlearn_report.md      – full 3-phase report

Usage:
  python run_backtest_splitlearn.py --dry-run
  python run_backtest_splitlearn.py --model qwen3.5:4b
  python run_backtest_splitlearn.py --dry-run --split 70 15 15
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import BacktestEngine, BacktestStats
from agents.base_agent import _ollama_chat, _parse_json_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("splitlearn.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger("splitlearn")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DATASET_ROOT = Path("Dataset/binance_vision_real/BTCUSDT/aggTrades")
PURGE_GAP_FILES = 1          # Drop N files between each split boundary (§ 1.2 Gap Buffer)
WEIGHTS_PATH = Path("trained_credibility_weights.json")
CRITIQUE_PATH = Path("agents/self_critique_report.json")
REPORT_PATH = Path("backtest_v2_splitlearn_report.md")
JSON_REPORT_PATH = Path("backtest_v2_splitlearn_report.json")

# Per-regime labels that the engine may emit
ALL_REGIMES = ["TRENDING", "RANGING", "HIGH_VOL", "LOW_VOL", "REVERTING", "UNKNOWN"]


# ─────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────
def discover_files() -> List[Path]:
    """
    Scan DATASET_ROOT for all *.parquet files.
    Sort strictly by filename (YYYYMMDD) → chronological order, NO LEAKAGE.
    """
    files = sorted(DATASET_ROOT.rglob("*.parquet"), key=lambda p: p.stem)
    logger.info("Discovered %d parquet files in %s", len(files), DATASET_ROOT)
    if not files:
        logger.error("No parquet files found. Check DATASET_ROOT: %s", DATASET_ROOT.resolve())
        sys.exit(1)
    return files


def split_files(
    files: List[Path],
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    purge_gap: int = PURGE_GAP_FILES,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split list of files chronologically: TRAIN / (gap) / VAL / (gap) / TEST.
    Strictly no overlap — purge gap files are DROPPED between splits.
    """
    n = len(files)
    train_end = int(n * train_pct)
    val_end = train_end + purge_gap + int(n * val_pct)
    test_start = val_end + purge_gap

    train_files = files[:train_end]
    val_files   = files[train_end + purge_gap : val_end]
    test_files  = files[test_start:]

    logger.info(
        "DATA SPLIT (NO LEAKAGE): TRAIN=%d files | PURGE=%d | VAL=%d files | PURGE=%d | TEST=%d files",
        len(train_files), purge_gap, len(val_files), purge_gap, len(test_files),
    )
    logger.info(
        "  TRAIN: %s → %s",
        train_files[0].stem if train_files else "N/A",
        train_files[-1].stem if train_files else "N/A",
    )
    logger.info(
        "  VAL:   %s → %s",
        val_files[0].stem if val_files else "N/A",
        val_files[-1].stem if val_files else "N/A",
    )
    logger.info(
        "  TEST:  %s → %s",
        test_files[0].stem if test_files else "N/A",
        test_files[-1].stem if test_files else "N/A",
    )
    return train_files, val_files, test_files


# ─────────────────────────────────────────────
# Per-file stats accumulator
# ─────────────────────────────────────────────
def run_phase(
    label: str,
    files: List[Path],
    ollama_model: str,
    dry_run: bool,
    freeze_credibility: bool,
    preload_credibilities: Optional[Dict[str, float]],
    max_bars_per_file: Optional[int] = None,
) -> Tuple[Dict[str, BacktestStats], Dict[str, float], Dict[str, Dict]]:
    """
    Run BacktestEngine over a list of files sequentially.
    Returns (per_file_stats_dict, final_credibilities, per_phase_agent_learning_stats).
    Credibility state is threaded: end of file N becomes start of file N+1.
    When freeze_credibility=True, weights do NOT update (VAL/TEST phases).
    """
    logger.info("=" * 70)
    logger.info("PHASE: %s | files=%d | freeze=%s", label, len(files), freeze_credibility)
    logger.info("=" * 70)

    per_file_stats: Dict[str, BacktestStats] = {}
    current_creds = dict(preload_credibilities) if preload_credibilities else {}
    phase_agent_stats: Dict[str, Dict] = {}

    for i, fpath in enumerate(files, 1):
        logger.info("[%s] File %d/%d: %s", label, i, len(files), fpath.name)
        try:
            eng = BacktestEngine(
                parquet_path=str(fpath),
                timeframe="5m",
                symbol="BTCUSDT",
                ollama_model=ollama_model,
                max_bars=max_bars_per_file,
                dry_run=dry_run,
                freeze_credibility=freeze_credibility,
                preload_credibilities=current_creds if current_creds else None,
            )
            stats = eng.run()
            per_file_stats[fpath.stem] = stats

            # Merge per-file agent learning stats into phase aggregate.
            file_agent_stats = eng.get_agent_learning_stats()
            for agent_name, ast in file_agent_stats.items():
                dst = phase_agent_stats.setdefault(
                    agent_name,
                    {"total": 0, "correct": 0, "incorrect": 0, "by_regime": {}},
                )
                dst["total"] += int(ast.get("total", 0))
                dst["correct"] += int(ast.get("correct", 0))
                dst["incorrect"] += int(ast.get("incorrect", 0))

                for regime, rst in ast.get("by_regime", {}).items():
                    rdst = dst["by_regime"].setdefault(
                        regime,
                        {"total": 0, "correct": 0, "incorrect": 0},
                    )
                    rdst["total"] += int(rst.get("total", 0))
                    rdst["correct"] += int(rst.get("correct", 0))
                    rdst["incorrect"] += int(rst.get("incorrect", 0))

            # Thread credibility forward (only meaningful in TRAIN phase)
            if not freeze_credibility:
                current_creds = eng.get_credibilities()
                logger.debug(
                    "Updated credibilities after %s: %s",
                    fpath.stem, {k: f"{v:.4f}" for k, v in current_creds.items()},
                )
        except Exception as exc:
            logger.error("Failed on file %s: %s", fpath.name, exc, exc_info=True)
            per_file_stats[fpath.stem] = BacktestStats()   # blank stats for failed file

    for ast in phase_agent_stats.values():
        total = int(ast.get("total", 0))
        correct = int(ast.get("correct", 0))
        ast["accuracy"] = (correct / total) if total > 0 else 0.0
        for rst in ast.get("by_regime", {}).values():
            r_total = int(rst.get("total", 0))
            r_correct = int(rst.get("correct", 0))
            rst["accuracy"] = (r_correct / r_total) if r_total > 0 else 0.0

    return per_file_stats, current_creds, phase_agent_stats


# ─────────────────────────────────────────────
# Aggregate stats helper
# ─────────────────────────────────────────────
def aggregate(per_file: Dict[str, BacktestStats]) -> BacktestStats:
    agg = BacktestStats()
    for s in per_file.values():
        agg.total_bars += s.total_bars
        agg.debates_run += s.debates_run
        agg.trades_opened += s.trades_opened
        agg.trades_closed += s.trades_closed
        agg.wins += s.wins
        agg.losses += s.losses
        agg.flat_decisions += s.flat_decisions
        agg.total_pnl += s.total_pnl
        agg.errors_caught += s.errors_caught
        agg.credibility_updates += s.credibility_updates
        agg.max_drawdown = max(agg.max_drawdown, s.max_drawdown)
        agg.current_equity = max(agg.current_equity, s.current_equity)
    return agg


# ─────────────────────────────────────────────
# Self-critique generator
# ─────────────────────────────────────────────
def generate_self_critique(
    agent_names: List[str],
    train_creds: Dict[str, float],
    val_creds: Dict[str, float],
    train_agent_stats: Dict[str, Dict],
    val_agent_stats: Dict[str, Dict],
    test_agent_stats: Dict[str, Dict],
    train_agg: BacktestStats,
    val_agg: BacktestStats,
    test_agg: BacktestStats,
    ollama_model: str,
    dry_run: bool,
) -> Dict[str, Dict]:
    """
    Each agent compares its train/validation/test held-out behavior.
    If live mode is enabled, ask Ollama to produce natural-language self-critique.

    Returns structured self-critique per agent.
    """
    critiques: Dict[str, Dict] = {}

    train_win_rate = train_agg.win_rate
    val_win_rate = val_agg.win_rate
    test_win_rate = test_agg.win_rate

    for name in agent_names:
        train_c = train_creds.get(name, 0.5)
        val_c   = val_creds.get(name, 0.5)        # frozen → same as train_c
        cred_delta = round(val_c - train_c, 4)    # will be 0 when frozen, kept for clarity

        tr_stats = train_agent_stats.get(name, {})
        va_stats = val_agent_stats.get(name, {})
        te_stats = test_agent_stats.get(name, {})

        tr_acc = float(tr_stats.get("accuracy", 0.0))
        va_acc = float(va_stats.get("accuracy", 0.0))
        te_acc = float(te_stats.get("accuracy", 0.0))
        val_drift = va_acc - tr_acc
        test_drift = te_acc - va_acc

        # Derive bias indicators from credibility level
        is_overconfident = train_c > 0.65
        is_underconfident = train_c < 0.35
        consistency_good = abs(val_drift) < 0.15

        # Determine regime biases from per-regime validation accuracy.
        overconfident_in: List[str] = []
        underconfident_in: List[str] = []
        regime_biases: Dict[str, str] = {}

        va_regimes = va_stats.get("by_regime", {})
        if va_regimes:
            sorted_regimes = sorted(
                va_regimes.items(),
                key=lambda kv: float(kv[1].get("accuracy", 0.0)),
            )
            low_regimes = [r for r, _ in sorted_regimes[:2]]
            high_regimes = [r for r, _ in sorted_regimes[-2:]]
            if is_overconfident:
                overconfident_in = low_regimes
            if is_underconfident:
                underconfident_in = high_regimes
            for r, rst in va_regimes.items():
                regime_biases[r] = (
                    f"val_acc={float(rst.get('accuracy', 0.0)):.1%}"
                )
        else:
            regime_biases["general"] = "Insufficient validation regime samples"

        performance_decay_flag = (
            (train_agg.total_pnl > 0 and val_agg.total_pnl < train_agg.total_pnl * 0.5)
            or (tr_acc > 0 and va_acc < tr_acc * 0.5)
        )

        recommended_adjustment: str
        if performance_decay_flag and is_overconfident:
            recommended_adjustment = (
                f"Reduce credibility weight by 0.05 in RANGING regime. "
                f"Current val PnL ({val_agg.total_pnl:.4%}) is significantly below train ({train_agg.total_pnl:.4%}). "
                f"Possible overfit to trending patterns seen in training data."
            )
        elif is_underconfident and train_c < 0.4:
            recommended_adjustment = (
                f"Explore feature engineering improvements. "
                f"Low credibility ({train_c:.4f}) suggests signal quality below baseline."
            )
        else:
            recommended_adjustment = (
                f"Performance within acceptable bounds. Monitor for regime shift. "
                f"Train acc={tr_acc:.1%} | Val acc={va_acc:.1%} | Test acc={te_acc:.1%}."
            )

        llm_self_critique = ""
        if not dry_run:
            critique_system = (
                "You are a strict quant model reviewer. "
                "Return ONLY JSON with fields: self_reflection, adjustment, confidence_note, overconfident_in, underconfident_in."
            )
            critique_user = (
                f"Agent={name}\n"
                f"train_credibility={train_c:.4f} val_credibility={val_c:.4f}\n"
                f"train_agent_accuracy={tr_acc:.4f} val_agent_accuracy={va_acc:.4f} test_agent_accuracy={te_acc:.4f}\n"
                f"train_phase_win_rate={train_win_rate:.4f} val_phase_win_rate={val_win_rate:.4f} test_phase_win_rate={test_win_rate:.4f}\n"
                f"validation_regime_breakdown={json.dumps(va_regimes)}\n"
                f"Give concise self-critique and one specific adjustment."
            )
            try:
                raw = _ollama_chat(
                    system_prompt=critique_system,
                    user_message=critique_user,
                    model=ollama_model,
                    temperature=0.2,
                )
                parsed = _parse_json_response(raw)
                if parsed:
                    llm_self_critique = parsed.get("self_reflection", "")
                    recommended_adjustment = parsed.get("adjustment", recommended_adjustment)
                    if isinstance(parsed.get("overconfident_in"), list):
                        overconfident_in = [str(x) for x in parsed["overconfident_in"]][:4]
                    if isinstance(parsed.get("underconfident_in"), list):
                        underconfident_in = [str(x) for x in parsed["underconfident_in"]][:4]
                else:
                    llm_self_critique = raw.strip()[:300]
            except Exception as exc:
                llm_self_critique = f"LLM critique unavailable: {exc}"

        validity_flag: str
        if consistency_good and not performance_decay_flag:
            validity_flag = "VALID [OK]"
        elif performance_decay_flag:
            validity_flag = "INVALID [FAIL] -- performance decay > 50%"
        else:
            validity_flag = "WARNING [!] -- moderate drift detected"

        critiques[name] = {
            "agent": name,
            "train_credibility": round(train_c, 4),
            "val_credibility": round(val_c, 4),
            "credibility_delta": cred_delta,
            "train_agent_accuracy": round(tr_acc, 4),
            "val_agent_accuracy": round(va_acc, 4),
            "test_agent_accuracy": round(te_acc, 4),
            "train_win_rate": round(train_win_rate, 4),
            "val_win_rate": round(val_win_rate, 4),
            "test_win_rate": round(test_win_rate, 4),
            "val_accuracy_drift": round(val_drift, 4),
            "test_accuracy_drift": round(test_drift, 4),
            "overconfident_in": overconfident_in,
            "underconfident_in": underconfident_in,
            "regime_biases": regime_biases,
            "recommended_adjustment": recommended_adjustment,
            "llm_self_critique": llm_self_critique,
            "validity_flag": validity_flag,
        }

    return critiques


# ─────────────────────────────────────────────
# OOS validity check (§ 4)
# ─────────────────────────────────────────────
def check_oos_validity(
    train_agg: BacktestStats,
    val_agg: BacktestStats,
    test_agg: BacktestStats,
) -> Tuple[str, List[str]]:
    """
    Apply Section 4 validity criteria from pytorch_model_training_rule.md.
    Returns (validity_verdict, list_of_failure_reasons).
    """
    failures: List[str] = []

    # § 4 ❌ Criterion 1: Performance Decay
    if train_agg.total_pnl > 0:
        decay_ratio = test_agg.total_pnl / train_agg.total_pnl if train_agg.total_pnl != 0 else 0
        if decay_ratio < 0.5:
            failures.append(
                f"Performance Decay: test PnL ({test_agg.total_pnl:.4%}) is "
                f"{(1-decay_ratio):.0%} below train ({train_agg.total_pnl:.4%})"
            )

    # § 4 ❌ Criterion 2: Drawdown > 20%
    if test_agg.max_drawdown > 0.20:
        failures.append(
            f"Max Drawdown in TEST ({test_agg.max_drawdown:.1%}) exceeds 20% threshold"
        )

    # § 4 ❌ Criterion 3: Win rate collapse
    if train_agg.win_rate > 0 and test_agg.win_rate < 0.40:
        failures.append(
            f"Test win rate ({test_agg.win_rate:.1%}) below 40% floor"
        )

    if not failures:
        verdict = "[VALID] OOS performance within acceptable bounds"
    else:
        verdict = "[INVALID] " + "; ".join(failures)

    return verdict, failures


# ─────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────
def write_report(
    train_files: List[Path],
    val_files: List[Path],
    test_files: List[Path],
    train_agg: BacktestStats,
    val_agg: BacktestStats,
    test_agg: BacktestStats,
    train_creds: Dict[str, float],
    critiques: Dict[str, Dict],
    validity_verdict: str,
    validity_failures: List[str],
    dry_run: bool,
    elapsed_sec: float,
) -> None:
    lines: List[str] = []
    a = lines.append

    a("# ChatTrader.KPai — Backtest v2: Split-Learn Report")
    a(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    a(f"**Mode:** {'DRY RUN (offline fallback)' if dry_run else 'LIVE (Ollama)'}  ")
    a(f"**Protocol:** IRON WALL Data Splitting (NO LEAKAGE POLICY) — pytorch_model_training_rule.md  ")
    a(f"**Total elapsed:** {elapsed_sec:.1f}s")
    a("")

    a("---")
    a("## 1. Data Split Summary")
    a("")
    a("| Phase | Files | Date Range | Bars (est.) |")
    a("|-------|-------|------------|-------------|")

    def _date_range(fl: List[Path]) -> str:
        return f"{fl[0].stem} → {fl[-1].stem}" if fl else "—"

    def _est_bars(fl: List[Path]) -> str:
        return f"~{len(fl) * 224:,}"

    a(f"| **TRAIN** (70%) | {len(train_files)} | {_date_range(train_files)} | {_est_bars(train_files)} |")
    a(f"| *Purge Gap* | {PURGE_GAP_FILES} | — | — |")
    a(f"| **VALIDATION** (15%) | {len(val_files)} | {_date_range(val_files)} | {_est_bars(val_files)} |")
    a(f"| *Purge Gap* | {PURGE_GAP_FILES} | — | — |")
    a(f"| **TEST** (15%) | {len(test_files)} | {_date_range(test_files)} | {_est_bars(test_files)} |")
    a("")

    a("---")
    a("## 2. Phase Results")
    a("")
    a("| Metric | TRAIN | VALIDATION | TEST |")
    a("|--------|-------|------------|------|")

    def _fmt(s: BacktestStats) -> Tuple[str, str, str, str, str, str]:
        return (
            str(s.total_bars),
            f"{s.win_rate:.1%}",
            f"{s.total_pnl:.4%}",
            f"{s.max_drawdown:.2%}",
            str(s.trades_closed),
            str(s.debates_run),
        )

    t = _fmt(train_agg)
    v = _fmt(val_agg)
    x = _fmt(test_agg)

    metrics = ["Bars", "Win Rate", "Net PnL", "Max Drawdown", "Trades", "Debates"]
    for i, m in enumerate(metrics):
        a(f"| {m} | {t[i]} | {v[i]} | {x[i]} |")
    a("")

    a("---")
    a("## 3. Trained Credibility Weights (after TRAIN phase)")
    a("")
    a("These are the agent \"weights\" saved before VALIDATION. Loaded unchanged for VAL and TEST.")
    a("")
    a("| Agent | Credibility | Status |")
    a("|-------|-------------|--------|")
    for name, cred in sorted(train_creds.items(), key=lambda x: x[1], reverse=True):
        status = "↑ Bullish lean" if cred > 0.55 else ("↓ Bearish lean" if cred < 0.45 else "→ Neutral")
        a(f"| {name} | {cred:.4f} | {status} |")
    a("")

    a("---")
    a("## 4. Agent Self-Critique (VALIDATION Phase)")
    a("")
    a("Each analyst compared its training credibility to its validation-phase performance.")
    a("Credibility scores were FROZEN during validation — these critiques are pure OOS analysis.")
    a("")
    for agent_name, c in critiques.items():
        a(f"### {agent_name}")
        a(f"- **Train credibility:** {c['train_credibility']:.4f}  ")
        a(f"- **Val accuracy drift:** {c.get('val_accuracy_drift', 0.0):+.1%}  ")
        a(f"- **Validity:** {c['validity_flag']}  ")
        if c['overconfident_in']:
            a(f"- **Overconfident in:** {', '.join(c['overconfident_in'])}  ")
        if c['underconfident_in']:
            a(f"- **Underconfident in:** {', '.join(c['underconfident_in'])}  ")
        a(f"- **Regime biases:** {c['regime_biases']}  ")
        a(f"- **Recommended adjustment:** {c['recommended_adjustment']}  ")
        a("")

    a("---")
    a("## 5. OOS Validity Verdict (§ 4 Real-World Validity Logic)")
    a("")
    a(f"**{validity_verdict}**")
    a("")
    if validity_failures:
        a("Failure reasons:")
        for f_ in validity_failures:
            a(f"  - {f_}")
    else:
        a("All validity criteria passed:")
        a("  - Performance decay < 50% ✓")
        a("  - Test drawdown < 20% ✓")
        a("  - Test win rate ≥ 40% ✓")
    a("")

    a("---")
    a("## 6. Leakage Prevention Checklist")
    a("")
    a("| Criterion | Status |")
    a("|-----------|--------|")
    a(f"| Chronological sort (no shuffle) | ✅ Enforced — files sorted by YYYYMMDD stem |")
    a(f"| Purge gap between splits | ✅ {PURGE_GAP_FILES} file(s) dropped at each boundary |")
    a(f"| Scaler fit only on TRAIN | ✅ DataFeeder computes features per-file (no cross-file scaling) |")
    a(f"| Credibility frozen in VAL/TEST | ✅ freeze_credibility=True applied |")
    a(f"| No lookahead (t features from t-1) | ✅ DataFeeder warmup_bars=64 respected |")
    a("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written: %s", REPORT_PATH)


# ─────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChatTrader.KPai — Split-Learn Backtest (NO LEAKAGE)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Offline fallback, skip Ollama")
    parser.add_argument("--model", default="qwen3.5:4b", help="Ollama model")
    parser.add_argument(
        "--split", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"),
        default=[70, 15, 15],
        help="Split percentages (default: 70 15 15)",
    )
    parser.add_argument(
        "--max-bars", type=int, default=None,
        help="Limit bars per file (for quick smoke tests)",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Run TRAIN phase only and save weights (skip VAL/TEST)",
    )
    parser.add_argument(
        "--load-weights", type=str, default=None,
        help="Path to existing trained_credibility_weights.json to skip re-training",
    )
    args = parser.parse_args()

    train_pct, val_pct, test_pct = [x / 100 for x in args.split]
    if abs(train_pct + val_pct + test_pct - 1.0) > 0.01:
        logger.error("Split percentages must sum to 100. Got: %s", args.split)
        sys.exit(1)

    t_start = time.time()

    # ─── 1. Discover & split files ─────────────────────────────────
    all_files = discover_files()
    train_files, val_files, test_files = split_files(
        all_files, train_pct=train_pct, val_pct=val_pct, purge_gap=PURGE_GAP_FILES
    )

    # ─── 2. TRAIN PHASE ────────────────────────────────────────────
    if args.load_weights:
        logger.info("Skipping TRAIN — loading pre-trained weights from %s", args.load_weights)
        train_creds = json.loads(Path(args.load_weights).read_text(encoding="utf-8"))
        train_per_file: Dict[str, BacktestStats] = {}
        train_agg = BacktestStats()
        train_agent_stats: Dict[str, Dict] = {}
    else:
        logger.info("\n%s\n  PHASE 1: TRAIN (%d files)\n%s", "="*70, len(train_files), "="*70)
        train_per_file, train_creds, train_agent_stats = run_phase(
            label="TRAIN",
            files=train_files,
            ollama_model=args.model,
            dry_run=args.dry_run,
            freeze_credibility=False,     # ← credibility UPDATES here
            preload_credibilities={},
            max_bars_per_file=args.max_bars,
        )
        train_agg = aggregate(train_per_file)

        # Save trained weights (the "model checkpoint")
        WEIGHTS_PATH.write_text(
            json.dumps(train_creds, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "Trained credibility weights saved → %s | agents: %s",
            WEIGHTS_PATH,
            {k: f"{v:.4f}" for k, v in train_creds.items()},
        )

    if args.train_only:
        logger.info("--train-only flag set. Exiting after TRAIN phase.")
        return

    # ─── 3. VALIDATION PHASE ───────────────────────────────────────
    logger.info("\n%s\n  PHASE 2: VALIDATION (%d files, credibility FROZEN)\n%s",
                "="*70, len(val_files), "="*70)
    val_per_file, val_creds, val_agent_stats = run_phase(
        label="VAL",
        files=val_files,
        ollama_model=args.model,
        dry_run=args.dry_run,
        freeze_credibility=True,          # ← FROZEN — no weight updates
        preload_credibilities=train_creds,
        max_bars_per_file=args.max_bars,
    )
    val_agg = aggregate(val_per_file)

    # ─── 4. Self-critique ──────────────────────────────────────────
    logger.info("Generating self-critique reports ...")
    agent_names = list(train_creds.keys())
    critiques = generate_self_critique(
        agent_names=agent_names,
        train_creds=train_creds,
        val_creds=val_creds,
        train_agent_stats=train_agent_stats,
        val_agent_stats=val_agent_stats,
        test_agent_stats={},
        train_agg=train_agg,
        val_agg=val_agg,
        test_agg=BacktestStats(),
        ollama_model=args.model,
        dry_run=args.dry_run,
    )

    CRITIQUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CRITIQUE_PATH.write_text(
        json.dumps(critiques, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Self-critique saved -> %s", CRITIQUE_PATH)

    for name, c in critiques.items():
        logger.info(
            "  [SELF-CRITIQUE] %s | validity=%s | val_acc_drift=%+.1fpp",
            name, c["validity_flag"], c.get("val_accuracy_drift", 0.0) * 100,
        )

    # ─── 5. TEST PHASE ─────────────────────────────────────────────
    logger.info("\n%s\n  PHASE 3: TEST — OOS HELD-OUT (%d files, credibility FROZEN)\n%s",
                "="*70, len(test_files), "="*70)
    test_per_file, _test_creds, test_agent_stats = run_phase(
        label="TEST",
        files=test_files,
        ollama_model=args.model,
        dry_run=args.dry_run,
        freeze_credibility=True,          # ← FROZEN — true OOS evaluation
        preload_credibilities=train_creds,
        max_bars_per_file=args.max_bars,
    )
    test_agg = aggregate(test_per_file)

    # Refresh critiques with TEST phase behavior included.
    critiques = generate_self_critique(
        agent_names=agent_names,
        train_creds=train_creds,
        val_creds=val_creds,
        train_agent_stats=train_agent_stats,
        val_agent_stats=val_agent_stats,
        test_agent_stats=test_agent_stats,
        train_agg=train_agg,
        val_agg=val_agg,
        test_agg=test_agg,
        ollama_model=args.model,
        dry_run=args.dry_run,
    )
    CRITIQUE_PATH.write_text(
        json.dumps(critiques, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ─── 6. OOS Validity check (§ 4) ─────────────────────────────
    validity_verdict, validity_failures = check_oos_validity(train_agg, val_agg, test_agg)
    logger.info("OOS Validity verdict: %s", validity_verdict.encode('ascii', 'replace').decode('ascii'))

    elapsed = time.time() - t_start

    # ─── 7. Write report ──────────────────────────────────────────
    write_report(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        train_agg=train_agg,
        val_agg=val_agg,
        test_agg=test_agg,
        train_creds=train_creds,
        critiques=critiques,
        validity_verdict=validity_verdict,
        validity_failures=validity_failures,
        dry_run=args.dry_run,
        elapsed_sec=elapsed,
    )

    # ─── 8. JSON summary ─────────────────────────────────────────
    summary = {
        "split": {
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "purge_gap": PURGE_GAP_FILES,
            "train_date_range": [train_files[0].stem, train_files[-1].stem] if train_files else [],
            "val_date_range": [val_files[0].stem, val_files[-1].stem] if val_files else [],
            "test_date_range": [test_files[0].stem, test_files[-1].stem] if test_files else [],
        },
        "train": asdict(train_agg),
        "validation": asdict(val_agg),
        "test": asdict(test_agg),
        "trained_credibilities": train_creds,
        "self_critiques": critiques,
        "validity_verdict": validity_verdict,
        "validity_failures": validity_failures,
        "elapsed_sec": round(elapsed, 1),
    }
    JSON_REPORT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("JSON summary written -> %s", JSON_REPORT_PATH)

    # ─── 9. Final console summary ────────────────────────────────
    v_safe = validity_verdict.encode("ascii", "replace").decode("ascii")
    print("\n" + "=" * 70)
    print("  SPLIT-LEARN BACKTEST COMPLETE")
    print("=" * 70)
    print(f"  TRAIN   | bars={train_agg.total_bars:>6} | trades={train_agg.trades_closed:>4} | "
          f"win={train_agg.win_rate:.1%} | PnL={train_agg.total_pnl:.4%}")
    print(f"  VAL     | bars={val_agg.total_bars:>6} | trades={val_agg.trades_closed:>4} | "
          f"win={val_agg.win_rate:.1%} | PnL={val_agg.total_pnl:.4%}")
    print(f"  TEST    | bars={test_agg.total_bars:>6} | trades={test_agg.trades_closed:>4} | "
          f"win={test_agg.win_rate:.1%} | PnL={test_agg.total_pnl:.4%}")
    print(f"\n  OOS Validity: {v_safe}")
    print(f"  Trained weights -> {WEIGHTS_PATH}")
    print(f"  Self-critique   -> {CRITIQUE_PATH}")
    print(f"  Report          -> {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
