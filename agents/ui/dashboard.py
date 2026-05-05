"""
agents/ui/dashboard.py
Phase 6 Glass Brain Dashboard — ChatTrader.KPai
Inspired by AgentAGIv2 5-stage observability console.

Run with:
    streamlit run agents/ui/dashboard.py

New in Phase 6:
  - Glass Brain Live Console: 5-stage colored badges (INPUT/THINKING/FEELING/EVOLVING/OUTPUT)
  - Omni Interaction Log: live debate-feed from OmniLog JSONL
  - Autonomous Evolution Stream: per-agent credibility changes with regime + error decomp
  - RealSignalBridge (replaces MockModelBridge) — signals derived from real features
  - Backtest-ready context builder alongside manual slider input
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── path setup ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd

from real_signal_bridge import RealSignalBridge
from orchestration.debate_engine import DebateEngine
from agents.portfolio_manager import RiskConfig
from agents.journaler import Journaler
from agents.omni_log import OmniLog, STAGE_INPUT, STAGE_THINKING, STAGE_FEELING, STAGE_EVOLVING, STAGE_OUTPUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatTrader.KPai — Glass Brain Console",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Glass Brain stage color map ──────────────────────────────────────
STAGE_COLORS: Dict[str, str] = {
    STAGE_INPUT:    "#1e3a5f",   # deep blue
    STAGE_THINKING: "#1a3a1a",   # deep green
    STAGE_FEELING:  "#3a1a2e",   # deep purple
    STAGE_EVOLVING: "#2a1a0a",   # deep amber
    STAGE_OUTPUT:   "#0a2a2a",   # deep teal
}
STAGE_LABELS: Dict[str, str] = {
    STAGE_INPUT:    "INPUT",
    STAGE_THINKING: "THINKING",
    STAGE_FEELING:  "FEELING",
    STAGE_EVOLVING: "EVOLVING",
    STAGE_OUTPUT:   "OUTPUT",
}
STAGE_ICONS: Dict[str, str] = {
    STAGE_INPUT:    "📥",
    STAGE_THINKING: "🧠",
    STAGE_FEELING:  "💜",
    STAGE_EVOLVING: "🔄",
    STAGE_OUTPUT:   "📤",
}

# ── kind icon map ─────────────────────────────────────────────────────
KIND_ICONS: Dict[str, str] = {
    "debate_start":       "🎬",
    "analyst_input":      "📥",
    "analyst_output":     "🔬",
    "shadow_input":       "👹",
    "shadow_output":      "⚡",
    "orchestrator_input": "🧠",
    "orchestrator_output":"📊",
    "pm_input":           "💼",
    "pm_output":          "✅",
    "trade_open":         "🟢",
    "trade_close":        "🔴",
    "error":              "❌",
}

# ────────────────────────────────────────────────────────────────────
# Session-state initialization
# ────────────────────────────────────────────────────────────────────
if "debate_history" not in st.session_state:
    st.session_state.debate_history = []
if "engine" not in st.session_state:
    st.session_state.engine = None
if "bridge" not in st.session_state:
    st.session_state.bridge = RealSignalBridge()
if "omni" not in st.session_state:
    st.session_state.omni = OmniLog.get_instance()
if "stage_events" not in st.session_state:
    st.session_state.stage_events: List[Dict] = []   # live stage trace for current debate

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
DIRECTION_COLORS = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "⚪"}
REGIME_ICONS = {
    "TRENDING_UP": "📈", "TRENDING_DOWN": "📉",
    "RANGING": "↔️", "HIGH_VOLATILITY": "⚡",
    "LOW_VOLATILITY": "😴", "BREAKOUT": "💥",
    "REVERTING": "🔄", "UNKNOWN": "❓",
}


def _conf_bar(conf: float) -> str:
    filled = int(conf * 10)
    return "█" * filled + "░" * (10 - filled) + f" {conf:.0%}"


def _stage_badge(stage: str) -> str:
    """Return HTML-colored badge string for a stage."""
    color = STAGE_COLORS.get(stage, "#333")
    label = STAGE_LABELS.get(stage, stage)
    icon = STAGE_ICONS.get(stage, "•")
    return f'<span style="background:{color};color:#eee;padding:2px 8px;border-radius:4px;font-size:0.8em;font-family:monospace">{icon} {label}</span>'


def _render_glass_brain_stage_card(stage: str, events: List[Dict]) -> None:
    """Render a colored expandable card for one pipeline stage."""
    bg = STAGE_COLORS.get(stage, "#333")
    label = STAGE_LABELS.get(stage, stage)
    icon = STAGE_ICONS.get(stage, "•")
    header_html = (
        f'<div style="background:{bg};padding:6px 12px;border-radius:6px;margin:4px 0">'
        f'<b style="color:#eee;font-family:monospace">{icon} {label}</b>'
        f'<span style="color:#aaa;font-size:0.8em;margin-left:12px">{len(events)} event(s)</span>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    if events:
        with st.expander(f"View {label} events", expanded=False):
            for ev in events[-10:]:   # last 10 per stage
                kind = ev.get("kind", "?")
                frm = ev.get("from", ev.get("from_agent", "?"))
                to = ev.get("to", ev.get("to_agent", "?"))
                icon2 = KIND_ICONS.get(kind, "•")
                content = ev.get("content", {})
                if isinstance(content, dict):
                    content_str = json.dumps(content, ensure_ascii=False)[:200]
                else:
                    content_str = str(content)[:200]
                st.caption(
                    f"{icon2} **{kind}** | {frm} → {to} | {content_str}"
                )


def _build_manual_context(
    symbol: str,
    timeframe: str,
    slope: float,
    zscore: float,
    atr: float,
    ema_spread: float,
    ofi: float,
    inventory: float,
) -> Dict[str, Any]:
    features = {
        "price_slope_20": slope,
        "zscore_close_64": zscore,
        "atr_14": atr,
        "atr_mean": atr * 0.85,
        "ema_spread": ema_spread,
        "bb_distance": zscore * 0.5,
        "bb_width": abs(zscore) * 0.01 + 0.01,
        "rsi_14": 50.0 + zscore * 10.0,
        "ofi_proxy": ofi,
        "spread_proxy": 0.001,
        "vol_regime_code": 1 if atr < 0.003 else (2 if atr > 0.006 else 1),
        "spread_z_64": zscore * 0.8,
        "fracdiff_close_d04": slope * 0.5,
        "pair_correlation": 0.72,
        "inventory_level": inventory,
        "pattern_score": slope * 2.0,
    }

    bridge: RealSignalBridge = st.session_state.bridge
    model_signals = bridge.get_all_signals(symbol, {"features": features})
    price_summary = bridge.build_price_summary(features)

    return {
        "features": features,
        "model_signals": model_signals,
        "price_summary": price_summary,
        "symbol": symbol,
        "timeframe": timeframe,
        "bar_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _get_or_create_engine(ollama_model: str, max_dd: float, max_pos: float, force_slow: bool) -> DebateEngine:
    """Create or recreate engine when settings change."""
    key = f"{ollama_model}_{max_dd}_{max_pos}_{force_slow}"
    if st.session_state.engine is None or st.session_state.get("_engine_key") != key:
        risk = RiskConfig(
            max_drawdown_pct=max_dd,
            max_position_pct=max_pos,
        )
        st.session_state.engine = DebateEngine(
            risk_config=risk,
            ollama_model=ollama_model,
            max_parallel_analysts=3,
            force_slow_path=force_slow,
            omni_log=st.session_state.omni,
        )
        st.session_state["_engine_key"] = key
    return st.session_state.engine


# ────────────────────────────────────────────────────────────────────
# SIDEBAR — Controls + Live Perception Feed
# ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Glass Brain Console")
    st.caption("ChatTrader.KPai Phase 6")

    st.subheader("Market Parameters")
    symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AAVEUSDT"], index=0)
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    force_slow = st.checkbox("Force SLOW path", value=True,
                             help="Force full LLM debate on every bar (required for learning loop)")

    st.subheader("Feature Sliders (Manual Override)")
    slope = st.slider("Price Slope (20-bar)", -0.01, 0.01, 0.002, step=0.0005, format="%.4f")
    zscore = st.slider("Z-Score Close (64-bar)", -3.0, 3.0, 0.5, step=0.1)
    atr = st.slider("ATR-14", 0.001, 0.015, 0.003, step=0.001, format="%.4f")
    ema_spread = st.slider("EMA Spread", -0.005, 0.005, 0.001, step=0.0005, format="%.4f")
    ofi = st.slider("OFI Proxy", -1.0, 1.0, 0.1, step=0.05)
    inventory = st.slider("MM Inventory", -1.0, 1.0, 0.0, step=0.05)

    st.divider()
    st.subheader("Risk Configuration")
    max_dd = st.slider("Max Drawdown %", 0.05, 0.25, 0.10, step=0.01, format="%.0%%")
    max_pos = st.slider("Max Position %", 0.01, 0.10, 0.05, step=0.005, format="%.1%%")

    st.divider()
    st.subheader("LLM Configuration")
    ollama_model = st.selectbox(
        "Ollama Model",
        ["qwen3.5:4b", "qwen3.5:9b", "llama3.1:8b", "deepseek-r1:8b", "gemma4:e2b"],
        index=0,
        help="qwen3.5:4b is fastest (~1-2s/call)."
    )

    st.divider()
    run_btn = st.button("▶ Run Debate", type="primary", use_container_width=True)
    clear_btn = st.button("🗑 Clear History", use_container_width=True)
    refresh_omni_btn = st.button("🔄 Refresh OmniLog", use_container_width=True)

    # ── Live Perception Feed (sidebar OmniLog preview) ────────────────
    st.divider()
    st.subheader("📡 Live Perception Feed")
    _omni: OmniLog = st.session_state.omni
    recent_msgs = _omni.read_messages(limit=8)
    for msg in recent_msgs[::-1]:   # newest last = bottom of feed
        kind = msg.get("kind", "?")
        icon = KIND_ICONS.get(kind, "•")
        frm = msg.get("from", msg.get("from_agent", "?"))
        stage = msg.get("stage", "?")
        st.caption(f"{icon} `{frm}` [{stage}]")

if clear_btn:
    st.session_state.debate_history = []
    st.session_state.stage_events = []
    st.rerun()

# ────────────────────────────────────────────────────────────────────
# MAIN — Header
# ────────────────────────────────────────────────────────────────────
st.title("🧠 ChatTrader.KPai — Glass Brain Console")
st.caption("Phase 6 | Real signals active | Recursive learning loop | OmniLog enabled")

# ── Status bar ───────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Analysts", "6", "Online")
col2.metric("Signal Bridge", "RealSignalBridge", "Feature-based")
col3.metric("Ollama", ollama_model, "Connected")
col4.metric("Slow Path", "Forced" if force_slow else "Auto", "debate mode")
col5.metric("OmniLog", "Active", "JSONL event bus")

st.divider()

# ────────────────────────────────────────────────────────────────────
# RUN DEBATE
# ────────────────────────────────────────────────────────────────────
if run_btn:
    # Clear stage events for new debate
    st.session_state.stage_events = []

    with st.spinner(f"Running {'SLOW' if force_slow else 'AUTO'} path debate for {symbol}..."):
        market_ctx = _build_manual_context(symbol, timeframe, slope, zscore, atr, ema_spread, ofi, inventory)
        engine = _get_or_create_engine(ollama_model, max_dd, max_pos, force_slow)

        t0 = time.time()
        try:
            result = engine.run_debate(
                symbol=symbol,
                timeframe=timeframe,
                market_context=market_ctx,
            )
            st.session_state.debate_history.insert(0, result.to_dict())

            # Snapshot OmniLog events for this session into stage_events
            session_id = result.session_id
            all_msgs = st.session_state.omni.read_messages(limit=200)
            st.session_state.stage_events = [
                m for m in all_msgs if m.get("session_id") == session_id
            ]
        except Exception as exc:
            st.error(f"Debate engine error: {exc}")
            logger.exception("Debate run failed")
            st.stop()


# ────────────────────────────────────────────────────────────────────
# GLASS BRAIN — 5-Stage Pipeline Console
# ────────────────────────────────────────────────────────────────────
if st.session_state.stage_events:
    st.subheader("🧠 Glass Brain — Pipeline Stage Trace")
    stage_event_map: Dict[str, List[Dict]] = {
        STAGE_INPUT:    [],
        STAGE_THINKING: [],
        STAGE_FEELING:  [],
        STAGE_EVOLVING: [],
        STAGE_OUTPUT:   [],
    }
    for ev in st.session_state.stage_events:
        stage = ev.get("stage", STAGE_INPUT)
        if stage in stage_event_map:
            stage_event_map[stage].append(ev)
        else:
            stage_event_map[STAGE_INPUT].append(ev)

    brain_cols = st.columns(5)
    for i, (stage, col) in enumerate(zip(
        [STAGE_INPUT, STAGE_THINKING, STAGE_FEELING, STAGE_EVOLVING, STAGE_OUTPUT],
        brain_cols,
    )):
        with col:
            bg = STAGE_COLORS[stage]
            lbl = STAGE_LABELS[stage]
            icon = STAGE_ICONS[stage]
            n = len(stage_event_map[stage])
            st.markdown(
                f'<div style="background:{bg};padding:8px;border-radius:8px;text-align:center">'
                f'<div style="font-size:1.4em">{icon}</div>'
                f'<div style="color:#eee;font-family:monospace;font-weight:bold">{lbl}</div>'
                f'<div style="color:#aaa;font-size:0.85em">{n} events</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if n > 0:
                with st.expander("View", expanded=False):
                    for ev in stage_event_map[stage][-8:]:
                        kind = ev.get("kind", "?")
                        frm = ev.get("from", ev.get("from_agent", "?"))
                        to = ev.get("to", ev.get("to_agent", "?"))
                        content = ev.get("content", {})
                        cstr = json.dumps(content, ensure_ascii=False)[:160] if isinstance(content, dict) else str(content)[:160]
                        icon2 = KIND_ICONS.get(kind, "•")
                        st.caption(f"{icon2} **{kind}**  \n`{frm}` → `{to}`  \n{cstr}")
    st.divider()

# ────────────────────────────────────────────────────────────────────
# DISPLAY LATEST DEBATE RESULT
# ────────────────────────────────────────────────────────────────────
if st.session_state.debate_history:
    latest = st.session_state.debate_history[0]
    order = latest.get("trade_order", {})
    orch = latest.get("orchestrator_decision", {})
    shadow = latest.get("shadow_critique") or {}

    st.subheader(f"Latest Debate — Session {latest.get('session_id', '?')} | {latest.get('timestamp', '')}")

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    direction = order.get("direction", "FLAT")
    regime = latest.get("regime", "UNKNOWN")
    path = latest.get("path", "?")
    latency = latest.get("total_latency_ms", 0)
    consensus = orch.get("consensus_score", 0)

    kc1.metric("Final Action", f"{DIRECTION_COLORS.get(direction,'')} {order.get('action','?')}")
    kc2.metric("Regime", f"{REGIME_ICONS.get(regime,'')} {regime}")
    kc3.metric("Path", path, f"{'✓' if latency < (200 if path=='FAST' else 5000) else '⚠ SLOW'} {latency:.0f}ms")
    kc4.metric("Consensus Score", f"{consensus:.0%}")
    kc5.metric("Position Size", f"{order.get('position_size_pct', 0):.1%}")

    st.divider()

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("🔬 Analyst Evidence Packets")
        packets = latest.get("packets", [])
        for pkt in packets:
            archetype = pkt.get("archetype", "?")
            pkt_dir = pkt.get("direction", "FLAT")
            pkt_conf = pkt.get("confidence", 0.0)
            ra = pkt.get("regime_alignment", 0.0)
            cred = pkt.get("historical_credibility", 0.5)

            badge = DIRECTION_COLORS.get(pkt_dir, "⚪")
            with st.expander(
                f"{badge} **{pkt.get('agent_name')}** — {pkt_dir} "
                f"| conf={pkt_conf:.0%} | cred={cred:.0%}",
                expanded=False,
            ):
                c1, c2, c3 = st.columns(3)
                c1.progress(pkt_conf, text=f"Confidence: {_conf_bar(pkt_conf)}")
                c2.progress(ra, text=f"Regime fit: {_conf_bar(ra)}")
                c3.progress(cred, text=f"Credibility: {_conf_bar(cred)}")
                st.caption(f"**Thesis:** {pkt.get('thesis', 'N/A')}")
                raw = pkt.get("raw_signals", {})
                if raw:
                    st.json(raw, expanded=False)

        st.subheader("🧠 Orchestrator Synthesis")
        st.info(f"**Direction:** {orch.get('direction','?')} | **Consensus:** {orch.get('consensus_score',0):.0%}")
        st.write(orch.get("final_thesis", ""))
        if orch.get("key_risk"):
            st.warning(f"Key Risk: {orch.get('key_risk')}")
        if orch.get("dissenting_archetypes"):
            st.caption(f"Dissenting: {', '.join(orch.get('dissenting_archetypes', []))}")

    with right_col:
        st.subheader("👹 Shadow Agent Critique")
        if shadow:
            rb = shadow.get("rebuttal_strength", 0.0)
            st.progress(rb, text=f"Rebuttal strength: {_conf_bar(rb)}")
            st.write(f"**Main Risk:** {shadow.get('main_risk', 'N/A')}")
            st.write(f"**Invalidation:** {shadow.get('invalidation_scenario', 'N/A')}")
            st.caption(shadow.get("critique_thesis", ""))
        else:
            st.caption("No shadow critique (Fast Path)")

        st.subheader("💼 Portfolio Manager Decision")
        action_color = {"BUY": "🟢", "SELL": "🔴", "NO_TRADE": "⚪"}.get(order.get("action", ""), "⚪")
        st.metric("Action", f"{action_color} {order.get('action','?')}")
        if order.get("action") != "NO_TRADE":
            st.write(f"**Size:** {order.get('position_size_pct', 0):.2%} of portfolio")
            st.write(f"**Stop Loss:** {order.get('stop_loss_pct', 0):.2%}")
            st.write(f"**Take Profit:** {order.get('take_profit_pct', 0):.2%}")
            st.write(f"**Risk Score:** {order.get('risk_score', 0):.0%}")
        else:
            st.warning(f"**Reason:** {order.get('reason', 'N/A')}")

        st.subheader("⏱ Latency")
        lat = latest.get("total_latency_ms", 0)
        target = 200 if path == "FAST" else 5000
        pct = min(lat / target, 1.0)
        st.metric(
            f"{path} Path Latency",
            f"{lat:.0f} ms",
            f"Target: <{target}ms {'✓' if lat < target else '⚠ EXCEEDED'}",
            delta_color="normal" if pct < 0.7 else "inverse",
        )
        st.progress(min(pct, 1.0))

    st.divider()

    st.subheader("📊 Model Signal Overview")
    rows = []
    for pkt in packets:
        mv = pkt.get("raw_signals", {}).get("model_votes", {})
        for model_name, vote in mv.items():
            rows.append({
                "Model": model_name,
                "Archetype": pkt.get("archetype", "?"),
                "Direction": vote.get("direction", "FLAT"),
                "Logit": vote.get("logit", 0.0),
                "Agent Dir": pkt.get("direction", "FLAT"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ────────────────────────────────────────────────────────────────────
# OMNI INTERACTION LOG — Debate Feed
# ────────────────────────────────────────────────────────────────────
st.subheader("📡 Omni Interaction Log — Debate Feed")

omni_msgs = st.session_state.omni.read_messages(limit=50)
if omni_msgs:
    omni_rows = []
    for msg in reversed(omni_msgs):  # newest first
        content = msg.get("content", {})
        if isinstance(content, dict):
            content_str = json.dumps(content, ensure_ascii=False)[:200]
        else:
            content_str = str(content)[:200]
        stage = msg.get("stage", "?")
        kind = msg.get("kind", "?")
        omni_rows.append({
            "Time": msg.get("timestamp", "")[-8:],
            "Session": msg.get("session_id", ""),
            "Stage": stage,
            "Kind": kind,
            "From": msg.get("from", msg.get("from_agent", "?")),
            "To": msg.get("to", msg.get("to_agent", "?")),
            "Content": content_str,
        })
    df_omni = pd.DataFrame(omni_rows)
    st.dataframe(df_omni, use_container_width=True, hide_index=True)
else:
    st.caption("No OmniLog entries yet. Run a debate to populate.")

st.divider()

# ────────────────────────────────────────────────────────────────────
# AUTONOMOUS EVOLUTION STREAM — Credibility Changes
# ────────────────────────────────────────────────────────────────────
st.subheader("🔄 Autonomous Evolution Stream")

growth_entries = st.session_state.omni.read_growth_log(limit=100)
if growth_entries:
    growth_rows = []
    for entry in reversed(growth_entries):  # newest first
        old_c = entry.get("old_credibility", 0.5)
        new_c = entry.get("new_credibility", 0.5)
        delta = new_c - old_c
        errd = entry.get("error_decomp", {})
        growth_rows.append({
            "Time": entry.get("timestamp", "")[-8:],
            "Agent": entry.get("agent_name", "?"),
            "Regime": entry.get("regime", "?"),
            "Old Cred": f"{old_c:.4f}",
            "New Cred": f"{new_c:.4f}",
            "Delta": f"{'▲' if delta >= 0 else '▼'} {delta:+.4f}",
            "Profitable": "✓" if entry.get("was_profitable") else "✗",
            "Signal Err": f"{errd.get('signal_error', 0):.3f}",
            "Decision Err": f"{errd.get('decision_error', 0):.3f}",
            "Session": entry.get("session_id", ""),
        })
    df_growth = pd.DataFrame(growth_rows)
    st.dataframe(df_growth, use_container_width=True, hide_index=True)

    # Credibility summary by agent (latest value)
    if st.session_state.engine:
        st.subheader("Current Agent Credibility Scores")
        cred_cols = st.columns(6)
        for i, analyst in enumerate(st.session_state.engine.analysts):
            cred_cols[i % 6].metric(
                analyst.name.replace("Analyst", ""),
                f"{analyst.credibility_score:.4f}",
            )
else:
    st.caption("No evolution data yet. Run backtest.py to populate credibility updates.")

st.divider()

# ────────────────────────────────────────────────────────────────────
# DEBATE HISTORY TABLE
# ────────────────────────────────────────────────────────────────────
if len(st.session_state.debate_history) > 1:
    st.subheader(f"📜 Debate History ({len(st.session_state.debate_history)} sessions)")
    history_rows = []
    for rec in st.session_state.debate_history:
        o = rec.get("trade_order", {})
        orch_r = rec.get("orchestrator_decision", {})
        history_rows.append({
            "Session": rec.get("session_id", "?"),
            "Time": rec.get("timestamp", "?"),
            "Symbol": rec.get("symbol", "?"),
            "TF": rec.get("timeframe", "?"),
            "Regime": rec.get("regime", "?"),
            "Path": rec.get("path", "?"),
            "Direction": orch_r.get("direction", "?"),
            "Action": o.get("action", "?"),
            "Consensus": f"{orch_r.get('consensus_score', 0):.0%}",
            "Size": f"{o.get('position_size_pct', 0):.1%}",
            "Latency (ms)": f"{rec.get('total_latency_ms', 0):.0f}",
        })
    st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "ChatTrader.KPai Phase 6 | RealSignalBridge active | Glass Brain + OmniLog + Evolution Stream | "
    "Run `python backtest.py` for recursive learning loop | "
    f"Ollama: {ollama_model} @ http://localhost:11434"
)
