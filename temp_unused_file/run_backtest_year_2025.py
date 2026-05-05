from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from backtest import BacktestEngine

DATA_ROOT = Path("Dataset/binance_vision_real/BTCUSDT/aggTrades")
JOURNAL_DIR = Path("agents/journal")
OMNI_DIR = Path("agents/omni_log")
REPORT_PATH = Path("backtest_v1_summary_report.md")

DATE_FILE_RE = re.compile(r"(2025\d{4})\.parquet$")
WORD_RE = re.compile(r"[a-zA-Z]{4,}")
STOPWORDS = {
    "with", "that", "from", "this", "into", "when", "your", "will", "have", "were", "they",
    "their", "about", "under", "then", "than", "while", "market", "price", "trade", "trading",
    "regime", "signal", "model", "using", "based", "near", "over", "into", "only", "very",
}


def _discover_2025_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.parquet"):
        m = DATE_FILE_RE.search(p.name)
        if m:
            files.append(p)
    return sorted(files)


def _read_jsonl_files(directory: Path, pattern: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for fp in sorted(directory.glob(pattern)):
        try:
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except FileNotFoundError:
            continue
    return records


def _snapshot_ids(records: List[Dict[str, Any]], key: str) -> set:
    return {r.get(key) for r in records if r.get(key)}


def _extract_keywords(theses: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
    ctr: Counter = Counter()
    for t in theses:
        for w in WORD_RE.findall((t or "").lower()):
            if w in STOPWORDS:
                continue
            ctr[w] += 1
    return ctr.most_common(top_n)


def _safe_avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _render_report(
    run_started_at: str,
    run_finished_at: str,
    files: List[Path],
    per_day: List[Dict[str, Any]],
    journal_new: List[Dict[str, Any]],
    omni_new: List[Dict[str, Any]],
    growth_new: List[Dict[str, Any]],
    dry_run: bool,
    model: str,
) -> str:
    total = {
        "bars": sum(d["stats"].get("total_bars", 0) for d in per_day),
        "debates": sum(d["stats"].get("debates_run", 0) for d in per_day),
        "trades_opened": sum(d["stats"].get("trades_opened", 0) for d in per_day),
        "trades_closed": sum(d["stats"].get("trades_closed", 0) for d in per_day),
        "wins": sum(d["stats"].get("wins", 0) for d in per_day),
        "losses": sum(d["stats"].get("losses", 0) for d in per_day),
        "flat": sum(d["stats"].get("flat_decisions", 0) for d in per_day),
        "errors": sum(d["stats"].get("errors_caught", 0) for d in per_day),
        "cred_updates": sum(d["stats"].get("credibility_updates", 0) for d in per_day),
        "net_pnl": sum(d["stats"].get("total_pnl", 0.0) for d in per_day),
    }
    win_rate = (total["wins"] / total["trades_closed"]) if total["trades_closed"] else 0.0

    trade_open = [m for m in omni_new if m.get("kind") == "trade_open"]
    trade_close = [m for m in omni_new if m.get("kind") == "trade_close"]
    close_pnls = [float(m.get("pnl", 0.0)) for m in trade_close if m.get("pnl") is not None]

    stage_counter = Counter(m.get("stage", "UNKNOWN") for m in omni_new)
    kind_counter = Counter(m.get("kind", "UNKNOWN") for m in omni_new)

    regime_counter = Counter(r.get("regime", "UNKNOWN") for r in journal_new)
    path_counter = Counter(r.get("path", "UNKNOWN") for r in journal_new)

    signal_errors: List[float] = []
    decision_errors: List[float] = []
    execution_errors: List[float] = []
    total_loss_scores: List[float] = []
    for rec in journal_new:
        ed = rec.get("error_decomposition") or {}
        if not ed:
            continue
        signal_errors.append(float(ed.get("signal_error", 0.0)))
        decision_errors.append(float(ed.get("decision_error", 0.0)))
        execution_errors.append(float(ed.get("execution_error", 0.0)))
        total_loss_scores.append(float(ed.get("total_loss_score", 0.0)))

    # Agent team theme summary
    per_agent_dirs: Dict[str, Counter] = defaultdict(Counter)
    per_agent_conf: Dict[str, List[float]] = defaultdict(list)
    per_agent_thesis: Dict[str, List[str]] = defaultdict(list)

    for rec in journal_new:
        for pkt in rec.get("analyst_evidence", []):
            agent = pkt.get("agent_name", pkt.get("archetype", "UNKNOWN"))
            per_agent_dirs[agent][pkt.get("direction", "FLAT")] += 1
            try:
                per_agent_conf[agent].append(float(pkt.get("confidence", 0.0)))
            except Exception:
                pass
            per_agent_thesis[agent].append(str(pkt.get("thesis", "")))

    # Growth summary
    growth_by_agent: Dict[str, List[float]] = defaultdict(list)
    growth_profitable: Dict[str, int] = defaultdict(int)
    growth_total: Dict[str, int] = defaultdict(int)
    for g in growth_new:
        agent = g.get("agent") or g.get("agent_name") or "UNKNOWN"
        delta = float(g.get("delta", 0.0))
        growth_by_agent[agent].append(delta)
        growth_total[agent] += 1
        if g.get("was_profitable"):
            growth_profitable[agent] += 1

    lines: List[str] = []
    lines.append("# Backtest V1 Summary Report")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- Run started: {run_started_at}")
    lines.append(f"- Run finished: {run_finished_at}")
    lines.append(f"- Symbol: BTCUSDT")
    lines.append(f"- Timeframe: 5m")
    lines.append(f"- Model: {model}")
    lines.append(f"- Dry run mode: {dry_run}")
    lines.append(f"- Available 2025 data files processed: {len(files)}")
    lines.append("")
    lines.append("### Data Coverage (2025 files found in workspace)")
    for f in files:
        lines.append(f"- {f.as_posix()}")

    lines.append("")
    lines.append("## Aggregate Performance")
    lines.append(f"- Total bars processed: {total['bars']}")
    lines.append(f"- Debates run: {total['debates']}")
    lines.append(f"- Trades opened: {total['trades_opened']}")
    lines.append(f"- Trades closed: {total['trades_closed']}")
    lines.append(f"- Win/Loss: {total['wins']}/{total['losses']} (win rate: {win_rate:.2%})")
    lines.append(f"- Net PnL (sum sized): {total['net_pnl']:.6f} ({total['net_pnl'] * 100:.4f}%)")
    lines.append(f"- Flat decisions: {total['flat']}")
    lines.append(f"- Runtime errors caught: {total['errors']}")
    lines.append(f"- Credibility updates: {total['cred_updates']}")

    lines.append("")
    lines.append("## Per-Day Backtest Stats")
    lines.append("| Date File | Bars | Debates | Opened | Closed | Wins | Losses | Net PnL | Errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in per_day:
        s = d["stats"]
        lines.append(
            f"| {d['file_name']} | {s.get('total_bars',0)} | {s.get('debates_run',0)} | {s.get('trades_opened',0)} | "
            f"{s.get('trades_closed',0)} | {s.get('wins',0)} | {s.get('losses',0)} | {s.get('total_pnl',0.0):.6f} | {s.get('errors_caught',0)} |"
        )

    lines.append("")
    lines.append("## Glass Brain / Omni Debate Log Summary")
    lines.append(f"- New Omni events captured in this run: {len(omni_new)}")
    lines.append(f"- trade_open events: {len(trade_open)}")
    lines.append(f"- trade_close events: {len(trade_close)}")
    lines.append(f"- Closed-trade PnL sum from Omni log: {sum(close_pnls):.6f}")
    lines.append("- Stage distribution:")
    for stage, cnt in stage_counter.most_common():
        lines.append(f"  - {stage}: {cnt}")
    lines.append("- Event kind distribution (top):")
    for kind, cnt in kind_counter.most_common(15):
        lines.append(f"  - {kind}: {cnt}")

    lines.append("")
    lines.append("## Journal Summary")
    lines.append(f"- New journal sessions in this run: {len(journal_new)}")
    lines.append("- Regime distribution:")
    for regime, cnt in regime_counter.most_common():
        lines.append(f"  - {regime}: {cnt}")
    lines.append("- Path distribution:")
    for path, cnt in path_counter.most_common():
        lines.append(f"  - {path}: {cnt}")
    lines.append("- Error decomposition averages (sessions with closed outcomes only):")
    lines.append(f"  - avg signal_error: {_safe_avg(signal_errors):.4f}")
    lines.append(f"  - avg decision_error: {_safe_avg(decision_errors):.4f}")
    lines.append(f"  - avg execution_error: {_safe_avg(execution_errors):.4f}")
    lines.append(f"  - avg total_loss_score: {_safe_avg(total_loss_scores):.4f}")

    lines.append("")
    lines.append("## Agent Team Themes (from debate packets)")
    if not per_agent_dirs:
        lines.append("- No analyst packets found in new journal records.")
    else:
        for agent in sorted(per_agent_dirs.keys()):
            dirs = per_agent_dirs[agent]
            major_dir, major_cnt = (dirs.most_common(1)[0] if dirs else ("FLAT", 0))
            avg_conf = _safe_avg(per_agent_conf.get(agent, []))
            top_words = _extract_keywords(per_agent_thesis.get(agent, []), top_n=6)
            theme = ", ".join([w for w, _ in top_words]) if top_words else "n/a"
            lines.append(f"- {agent}")
            lines.append(f"  - dominant direction: {major_dir} ({major_cnt} packets)")
            lines.append(f"  - direction mix: LONG={dirs.get('LONG',0)}, SHORT={dirs.get('SHORT',0)}, FLAT={dirs.get('FLAT',0)}")
            lines.append(f"  - avg confidence: {avg_conf:.4f}")
            lines.append(f"  - recurring thesis terms: {theme}")

    lines.append("")
    lines.append("## Agent Growth (Credibility Evolution)")
    lines.append(f"- New growth events: {len(growth_new)}")
    if not growth_by_agent:
        lines.append("- No credibility update events were emitted in this run.")
    else:
        lines.append("| Agent | Updates | Net Delta | Avg Delta | Profitable-linked Updates |")
        lines.append("|---|---:|---:|---:|---:|")
        for agent in sorted(growth_by_agent.keys()):
            deltas = growth_by_agent[agent]
            lines.append(
                f"| {agent} | {growth_total[agent]} | {sum(deltas):+.4f} | {_safe_avg(deltas):+.4f} | {growth_profitable[agent]} |"
            )

    lines.append("")
    lines.append("## Notes")
    lines.append("- This report summarizes all *available* 2025 BTCUSDT parquet files currently present in the workspace.")
    lines.append("- If additional 2025 daily files are added later, re-run this script to expand coverage.")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full BTCUSDT 2025 backtest and write v1 summary report")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--dry-run", action="store_true", help="Disable Ollama calls and use fallback path")
    args = parser.parse_args()

    files = _discover_2025_files(DATA_ROOT)
    if not files:
        raise SystemExit("No 2025 BTCUSDT parquet files found under Dataset/binance_vision_real/BTCUSDT/aggTrades")

    run_started_at = datetime.now().isoformat(timespec="seconds")

    # Snapshots to isolate newly created records from this run
    journal_before = _read_jsonl_files(JOURNAL_DIR, "*-debate-journal.jsonl")
    journal_before_ids = _snapshot_ids(journal_before, "session_id")

    omni_before = _read_jsonl_files(OMNI_DIR, "*-omni-messages.jsonl")
    growth_before = _read_jsonl_files(OMNI_DIR, "*-growth-log.jsonl")
    omni_before_len = len(omni_before)
    growth_before_len = len(growth_before)

    per_day: List[Dict[str, Any]] = []

    for fp in files:
        engine = BacktestEngine(
            parquet_path=str(fp),
            timeframe=args.timeframe,
            symbol="BTCUSDT",
            ollama_model=args.model,
            max_bars=None,
            dry_run=args.dry_run,
        )
        stats = engine.run()
        per_day.append(
            {
                "file": str(fp),
                "file_name": fp.name,
                "stats": asdict(stats),
                "agent_credibilities": {a.name: a.credibility_score for a in engine.engine.analysts},
            }
        )

    run_finished_at = datetime.now().isoformat(timespec="seconds")

    journal_after = _read_jsonl_files(JOURNAL_DIR, "*-debate-journal.jsonl")
    journal_new = [r for r in journal_after if r.get("session_id") and r.get("session_id") not in journal_before_ids]

    omni_after = _read_jsonl_files(OMNI_DIR, "*-omni-messages.jsonl")
    growth_after = _read_jsonl_files(OMNI_DIR, "*-growth-log.jsonl")

    omni_new = omni_after[omni_before_len:]
    growth_new = growth_after[growth_before_len:]

    report_md = _render_report(
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        files=files,
        per_day=per_day,
        journal_new=journal_new,
        omni_new=omni_new,
        growth_new=growth_new,
        dry_run=args.dry_run,
        model=args.model,
    )
    REPORT_PATH.write_text(report_md, encoding="utf-8")

    # Save machine-readable artifact as well
    Path("backtest_v1_summary_report.json").write_text(
        json.dumps(
            {
                "run_started_at": run_started_at,
                "run_finished_at": run_finished_at,
                "files": [str(f) for f in files],
                "per_day": per_day,
                "new_records": {
                    "journal": len(journal_new),
                    "omni": len(omni_new),
                    "growth": len(growth_new),
                },
                "dry_run": args.dry_run,
                "model": args.model,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Processed {len(files)} file(s).")
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
