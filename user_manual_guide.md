# ChatTrader.KPai Phase 5 — User Manual Guide

**System:** Multi-Agent Debate System  
**Version:** Phase 5 (Mock Inference)  
**Date:** 2026-05-01  
**Ollama:** Required — `qwen3.5:4b` (default)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [File Hierarchy](#2-file-hierarchy)
3. [Prerequisites & Installation](#3-prerequisites--installation)
4. [Running the System](#4-running-the-system)
5. [Streamlit Dashboard Guide](#5-streamlit-dashboard-guide)
6. [CLI Runner Guide](#6-cli-runner-guide)
7. [Interpreting Agent Debate Logs](#7-interpreting-agent-debate-logs)
8. [Understanding the Visualization](#8-understanding-the-visualization)
9. [Agent Behavior Reference](#9-agent-behavior-reference)
10. [Risk Configuration](#10-risk-configuration)
11. [Switching to Production Models](#11-switching-to-production-models)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. System Overview

The Phase 5 system is a 10-agent multi-agent trading debate system. Given a symbol, timeframe, and market features, it:

1. Generates mock signals for all 18 neural network models (6 archetypes × 3 models)
2. Detects the current market regime (Trending, Ranging, High-Vol, etc.)
3. Routes to **FAST PATH** (direct signal aggregation, < 200ms) or **SLOW PATH** (full LLM debate, target < 5s)
4. Runs 6 analyst agents in parallel using your local Ollama instance
5. Has a Shadow Agent attack the consensus
6. Orchestrator synthesizes the final direction
7. Portfolio Manager sizes the position (or vetoes with NO_TRADE)
8. Persists everything to a daily JSONL journal

**Mock inference is active.** All 18 models return realistically formatted but randomized signals. When model training completes and models pass validation gates, simply swap `MockModelBridge` → `ProductionModelBridge`.

---

## 2. File Hierarchy

```
ChatTrader.KPai/
├── model_bridge.py                  # Mock inference layer (hot-swappable)
├── run_phase5.py                    # CLI debate runner
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py                # BaseAgent class + Ollama client + EvidencePacket
│   ├── analyst_agents.py            # 6 archetype analysts
│   ├── shadow_agent.py              # Devil's Advocate
│   ├── portfolio_manager.py         # Risk gatekeeper + Kelly sizing
│   ├── regime_detector.py           # Market state classifier
│   ├── journaler.py                 # JSONL persistence + error decomposition
│   ├── journal/                     # Auto-created — daily JSONL debate logs
│   └── ui/
│       ├── __init__.py
│       └── dashboard.py             # Streamlit visualization dashboard
│
├── orchestration/
│   ├── __init__.py
│   └── debate_engine.py             # Dual-path engine (FAST + SLOW)
│
└── tests/
    └── test_phase5.py               # 28 unit tests (all passing)
```

---

## 3. Prerequisites & Installation

### 3.1 Ollama Setup

Ollama must be running before launching the debate system.

**Start Ollama** (if not already running):
```powershell
ollama serve
```

**Verify available models:**
```powershell
ollama list
```

Recommended models (fastest → highest quality):
| Model | Speed | Quality | Use Case |
|---|---|---|---|
| `qwen3.5:4b` | Fastest (~1-2s/call) | Good | Default — scalping & swing |
| `phi4-mini-reasoning:3.8b` | Fast (~1.5s/call) | Very Good | Reasoning tasks |
| `qwen3.5:9b` | Medium (~3-4s/call) | Excellent | High-stakes decisions |
| `llama3.1:8b` | Medium (~2-3s/call) | Excellent | Balanced |

**Pull a model** (if not installed):
```powershell
ollama pull qwen3.5:4b
```

### 3.2 Python Environment

Activate the project virtual environment:
```powershell
cd D:\kp_ai_agent\ChatTrader.KPai
.\.venv\Scripts\Activate.ps1
```

Install Phase 5 dependencies:
```powershell
pip install requests streamlit pandas
```

**Verify Ollama connectivity:**
```powershell
python -c "import requests; r=requests.get('http://localhost:11434/api/tags',timeout=5); print([m['name'] for m in r.json()['models']])"
```

---

## 4. Running the System

### 4.1 Quick Start — Dry Run (No Ollama needed)

Tests the full pipeline using only mock model signals and weighted-vote fallback (no LLM):

```powershell
cd D:\kp_ai_agent\ChatTrader.KPai
.\.venv\Scripts\python.exe run_phase5.py --symbol BTCUSDT --timeframe 1h --dry-run
```

Expected output:
```
────────────────────────────────────────────────────────────
  DEBATE SESSION: a3f1b2c4
  Symbol: BTCUSDT | TF: 1h | Path: SLOW
  Regime: TRENDING_UP
  Latency: 12ms
────────────────────────────────────────────────────────────

  [ANALYST EVIDENCE]
    TrendAnalyst                      LONG  [██████░░░░] 62%
      → [FAST PATH] Raw model signal: LONG @ 0.62
    ...
  [PORTFOLIO MANAGER]
    Action:     BUY
    Size:       2.14% of portfolio
    ...
```

### 4.2 Full LLM Debate (Slow Path)

Runs the complete multi-agent debate with Ollama:

```powershell
# Default: 1h timeframe, auto path selection
.\.venv\Scripts\python.exe run_phase5.py --symbol BTCUSDT --timeframe 1h

# Force full debate loop regardless of timeframe
.\.venv\Scripts\python.exe run_phase5.py --symbol ETHUSDT --timeframe 1h --slow

# Save result to JSON
.\.venv\Scripts\python.exe run_phase5.py --symbol BTCUSDT --timeframe 4h --output result.json

# Use higher-quality model
.\.venv\Scripts\python.exe run_phase5.py --symbol SOLUSDT --timeframe 1h --model qwen3.5:9b
```

### 4.3 Streamlit Dashboard

```powershell
.\.venv\Scripts\streamlit.exe run agents/ui/dashboard.py
```

Then open: **http://localhost:8501**

### 4.4 Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase5.py -v
```

---

## 5. Streamlit Dashboard Guide

### Layout Overview

```
┌─ SIDEBAR ──────────────────┐  ┌─ MAIN PANEL ─────────────────────────────────────┐
│ Market Parameters           │  │ Status Bar: Models | Agents | Ollama | Journal    │
│   Symbol, Timeframe         │  ├──────────────────────────────────────────────────┤
│   Force SLOW Path           │  │ Latest Debate KPIs:                               │
│                             │  │   Action | Regime | Path/Latency | Consensus | Size│
│ Feature Sliders             │  ├──────────────────────────────────────────────────┤
│   Price Slope               │  │ LEFT: Analyst Evidence Packets                    │
│   Z-Score                   │  │   ► TrendAnalyst: LONG [██████] 72%               │
│   ATR-14                    │  │   ► MeanReversionAnalyst: SHORT [████] 45%        │
│   EMA Spread                │  │   ► ... (6 analysts expandable)                   │
│   OFI Proxy                 │  │                                                   │
│   MM Inventory              │  │   Orchestrator Synthesis box                      │
│                             │  │   (direction, consensus, thesis, key risk)        │
│ Risk Configuration          │  ├──────────────────────────────────────────────────┤
│   Max Drawdown %            │  │ RIGHT: Shadow Agent Critique                      │
│   Max Position %            │  │   Rebuttal strength bar                           │
│                             │  │   Main Risk, Invalidation Scenario               │
│ LLM Configuration           │  │                                                   │
│   Model Selection           │  │   Portfolio Manager Decision                      │
│                             │  │   Action, Size, Stop Loss, Take Profit            │
│ ▶ Run Debate                │  │                                                   │
│ 🗑 Clear History            │  │   Latency Gauge                                   │
└─────────────────────────────┘  ├──────────────────────────────────────────────────┤
                                  │ Model Signal Overview (dataframe)                 │
                                  ├──────────────────────────────────────────────────┤
                                  │ Debate History Table                              │
                                  └──────────────────────────────────────────────────┘
```

### Key Controls

| Control | Effect |
|---|---|
| **Symbol** | Which crypto pair to simulate |
| **Timeframe** | Determines FAST vs SLOW path eligibility |
| **Force SLOW path** | Disables fast-path optimization, always runs full LLM debate |
| **Feature Sliders** | Manually set market indicators to test different regimes |
| **Run Debate** | Execute one full debate cycle |
| **Clear History** | Reset the session history table |
| **Max Drawdown %** | Portfolio Manager hard stop threshold |
| **Max Position %** | Maximum size per trade (Kelly cap) |
| **Ollama Model** | Which local LLM to use for reasoning |

### Interpreting the Direction Badge Colors

- 🟢 **LONG** — System recommends buying
- 🔴 **SHORT** — System recommends selling  
- ⚪ **FLAT** — No trade (either low consensus, risk veto, or anti-overthinking rule)

---

## 6. CLI Runner Guide

```
python run_phase5.py [OPTIONS]

Options:
  --symbol SYMBOL      Trading pair (default: BTCUSDT)
  --timeframe TF       Timeframe: 1m, 3m, 5m, 15m, 1h, 4h, 1d (default: 1h)
  --slow               Force SLOW path (full LLM debate)
  --dry-run            Skip Ollama — use weighted vote fallback only
  --model MODEL        Ollama model name (default: qwen3.5:4b)
  --output FILE.json   Save full debate result to JSON file
```

### Example: Quick regime test

```powershell
# Test ranging regime (flat slope, near zero zscore)
python run_phase5.py --symbol BTCUSDT --timeframe 4h --dry-run

# Test trending up (high slope and ema_spread via feature sliders in dashboard)
python run_phase5.py --symbol ETHUSDT --timeframe 1h --slow
```

---

## 7. Interpreting Agent Debate Logs

### 7.1 Journal Files

Location: `agents/journal/YYYY-MM-DD-debate-journal.jsonl`

Each line is a complete JSON debate session record. Example fields:

```jsonc
{
  "session_id": "a3f1b2c4",
  "timestamp": "2026-05-01T15:30:00+07:00",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "regime": "TRENDING_UP",
  "path": "SLOW",
  "analyst_evidence": [
    {
      "agent_name": "TrendAnalyst",
      "archetype": "trend_follower",
      "direction": "LONG",
      "confidence": 0.72,
      "regime_alignment": 0.90,
      "historical_credibility": 0.55,
      "thesis": "Strong uptrend with positive EMA spread and momentum..."
    }
    // ... 5 more analysts
  ],
  "shadow_critique": {
    "rebuttal_strength": 0.35,
    "main_risk": "BTC is near resistance at ATH, potential rejection",
    "invalidation_scenario": "A close below the 20-period EMA would invalidate the thesis",
    "critique_thesis": "The trend is intact but conviction is moderate..."
  },
  "orchestrator_decision": {
    "direction": "LONG",
    "consensus_score": 0.74,
    "final_thesis": "5 of 6 analysts agree on a bullish bias...",
    "dissenting_archetypes": ["mean_reversion"],
    "key_risk": "Approaching resistance zone"
  },
  "trade_order": {
    "action": "BUY",
    "direction": "LONG",
    "position_size_pct": 0.031,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.05,
    "risk_score": 0.26,
    "reason": "Fractional Kelly sizing: 0.031 based on avg credibility 0.55"
  },
  "outcome": null,           // Filled after trade closes
  "error_decomposition": null
}
```

### 7.2 Key Metrics to Monitor

| Field | What it means | Action if poor |
|---|---|---|
| `consensus_score` | How much agents agree. < 0.60 → NO_TRADE | Normal. Anti-overthinking working. |
| `shadow_critique.rebuttal_strength` | 0 = solid consensus, 1 = very flawed | > 0.7 means genuine concern. Review key_risk. |
| `position_size_pct` | Fraction of portfolio. Expect 0.01–0.05 | If 0.0 → PM vetoed. Read `reason`. |
| `historical_credibility` | Agent's rolling win rate | Falls below 0.40 = agent unreliable in this regime |
| `regime_alignment` | How well this archetype fits current regime | Naturally varies. Low for off-regime agents. |
| `path` | FAST or SLOW | FAST = no LLM reasoning used. |

### 7.3 Error Decomposition (Post-Trade)

After a trade closes, call from Python:
```python
from agents.journaler import Journaler
j = Journaler()
j.update_outcome(
    session_id="a3f1b2c4",
    actual_pnl=-0.015,           # -1.5% loss
    was_profitable=False,
    signal_error=0.7,            # Main failure was model signal
    decision_error=0.2,          # Orchestrator also made poor call
    execution_error=0.1,         # Minor slippage
    notes="Stopped out at support breakout. Regime shifted."
)
```

This satisfies: **Loss = Signal_Error + Decision_Error + Execution_Error**

---

## 8. Understanding the Visualization

### 8.1 Confidence Bar Interpretation

```
[██████████] 100%  — Extremely high conviction
[███████░░░]  70%  — Good conviction
[█████░░░░░]  50%  — Neutral / uncertain
[███░░░░░░░]  30%  — Low conviction (near noise)
[█░░░░░░░░░]  10%  — Nearly random signal
```

### 8.2 Latency Gauge

- **Green** (< 70% of target): System performing well
- **Orange** (70–100% of target): Approaching limit
- **Red** (> 100% of target): Latency target exceeded — documented failure

For FAST PATH: target = 200ms  
For SLOW PATH: target = 5000ms

### 8.3 Regime Icons

| Icon | Regime | Best Agents |
|---|---|---|
| 📈 | TRENDING_UP | Trend Follower, Discretionary |
| 📉 | TRENDING_DOWN | Trend Follower, Discretionary |
| ↔️ | RANGING | Mean Reversion, Stat Arb, Market Maker |
| ⚡ | HIGH_VOLATILITY | Scalper |
| 😴 | LOW_VOLATILITY | Market Maker, Stat Arb |
| 💥 | BREAKOUT | Scalper, Trend Follower |
| 🔄 | REVERTING | Mean Reversion, Stat Arb |
| ❓ | UNKNOWN | All agents equal weight |

---

## 9. Agent Behavior Reference

### When an agent outputs FLAT

Agents default to FLAT when:
1. Their regime alignment is low (wrong market type for their strategy)
2. Their model signals are near zero (logit ≈ 0)
3. LLM reasoning concludes insufficient evidence

### Anti-Overthinking Rule

If the Orchestrator's `consensus_score < 0.60`, the system **strictly outputs FLAT** (NO_TRADE). This is by design. The system should only trade when there is real conviction. "Analysis paralysis is penalized" — a NO_TRADE is a valid output, not a failure.

### Shadow Agent Rebuttal Strength Guide

| Strength | Meaning |
|---|---|
| 0.0–0.2 | Shadow finds consensus sound — high-quality trade setup |
| 0.2–0.4 | Minor concerns — trade can proceed with awareness |
| 0.4–0.6 | Significant risks — PM will reduce sizing |
| 0.6–0.8 | Serious flaws — PM may veto |
| 0.8–1.0 | Consensus is deeply flawed — expect NO_TRADE |

---

## 10. Risk Configuration

### Default Settings

```python
RiskConfig(
    max_position_pct=0.05,           # Max 5% of portfolio per trade
    max_drawdown_pct=0.10,           # Hard stop at -10% portfolio drawdown
    max_open_positions=6,            # Max concurrent positions
    min_confidence_threshold=0.55,   # Minimum consensus to trade
    max_correlation_exposure=0.6,    # Not yet enforced in Phase 5
    kelly_fraction=0.25,             # Fractional Kelly multiplier
)
```

### Configuring in Dashboard

Use the sidebar sliders:
- **Max Drawdown %** — If your portfolio is down this amount, all new trades are vetoed
- **Max Position %** — Kelly criterion output is capped at this fraction

### Configuring Programmatically

```python
from agents.portfolio_manager import RiskConfig
from orchestration.debate_engine import DebateEngine

engine = DebateEngine(
    risk_config=RiskConfig(
        max_position_pct=0.03,    # Tighter: 3% max
        max_drawdown_pct=0.05,    # Tighter: 5% drawdown stop
        kelly_fraction=0.15,      # More conservative Kelly
    ),
    ollama_model="qwen3.5:9b",    # Higher quality LLM
)
```

---

## 11. Switching to Production Models

When Phase 4 training completes and models pass validation gates (Sharpe > 1.2, Accuracy > 55%, etc.):

### Step 1: Add `ProductionModelBridge` to `model_bridge.py`

```python
class ProductionModelBridge:
    """Hot-swap replacement for MockModelBridge."""
    
    def __init__(self, registry_path: str = "model_registry.json"):
        import json
        self.registry = json.loads(open(registry_path).read())
        self._load_models()
    
    def get_all_signals(self, symbol: str, market_context: dict) -> dict:
        # Load real model weights, run inference, return same signal dict structure
        ...
```

### Step 2: Switch in `run_phase5.py`

```python
# Change this line:
bridge = MockModelBridge(seed=42)
# To:
bridge = ProductionModelBridge(registry_path="model_registry.json")
```

### Step 3: Verify with dry-run

```powershell
python run_phase5.py --symbol BTCUSDT --timeframe 1h --dry-run --output test_production.json
```

Check that signal structure matches expected schema (direction, confidence, model_votes, ensemble_logit, raw_outputs).

### Step 4: Seed credibility scores

Update each analyst's initial `credibility_score` based on their model's validation Sharpe ratio:
```python
# In DebateEngine.__init__()
self.analysts[0].credibility_score = 0.72  # TrendAnalyst — based on LSTM Sharpe
```

---

## 12. Troubleshooting

### "Connection refused" — Ollama not running
```powershell
# Start Ollama service
ollama serve

# Verify it's up
curl http://localhost:11434/api/tags
```

### Slow path exceeds 5 seconds

**Root cause:** Sequential LLM calls for shadow + orchestrator + PM.  
**Workaround:** Use `--dry-run` for testing, or switch to `qwen3.5:4b` (fastest model).  
**Documented failure:** Phase 5 implementation report Section 3.2.

### All agents returning FLAT

Check:
1. Are features within realistic ranges? (slope: ±0.01, zscore: ±3.0)
2. Is consensus score above 0.60? (anti-overthinking threshold)
3. Is shadow rebuttal_strength > 0.6? (shadow veto kicking in)
4. Is drawdown above 10%? (PM hard veto)

Use `--dry-run` to isolate LLM issues from logic issues.

### JSON parse errors in logs

This is expected occasionally (LLMs sometimes output prose instead of JSON). The self-correction retry logic handles this automatically. If retries also fail, the system falls back to raw model signal direction.

Check `logging.WARNING` output for: `"JSON parse failure:"` messages.

### Tests failing after code changes

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase5.py -v --tb=long
```

Key invariants that tests enforce:
- `MockModelBridge` returns all 6 archetypes with exactly 3 model votes each
- `PortfolioManager` never exceeds `max_position_pct`
- Kelly sizing never returns negative values
- `NO_TRADE` is output when drawdown, position limit, or consensus thresholds are breached
- Fast path triggers only for scalping timeframes with high scalper confidence
