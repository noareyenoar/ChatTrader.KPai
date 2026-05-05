# Phase 5 Implementation Report — ChatTrader.KPai Multi-Agent Debate System

**Date:** 2026-05-01  
**Status:** COMPLETE — All 28 unit tests passing  
**Engineer:** AI Senior Quant/Systems Architect  
**Reference Prompts:** `coder_agent_system_prompts/generative-ai/03-ai-agent-builder.txt`, `coder_agent_system_prompts/system-architecture/01-distributed-systems-architect.txt`

---

## 1. Executive Summary

Phase 5 delivers the complete Multi-Agent Debate System for ChatTrader.KPai. The system implements the 6+1+1+1+1 agent roster defined in `Full_Recursive_Learning_Trade_Agents.md`, a dual-path execution engine (FAST < 200ms / SLOW < 5s), parallel Ollama LLM integration, a mock inference bridge for all 18 models, a real-time Streamlit visualization dashboard, and a JSONL-backed journaling system with error decomposition.

**All 18 models are still in `RESUME_TRAINING_REQUIRED` status.** The mock inference layer (`model_bridge.py`) provides realistically distributed signals that are hot-swappable when training completes.

---

## 2. System Architecture

### 2.1 Agent Roster (10 total)

| Agent | Role | File | LLM Used |
|---|---|---|---|
| `TrendAnalyst` | Archetype 1: LSTM/Transformer/TCN signals | `agents/analyst_agents.py` | Yes |
| `MeanReversionAnalyst` | Archetype 2: MLP/ResNet/GRN signals | `agents/analyst_agents.py` | Yes |
| `ScalperAnalyst` | Archetype 3: CNN/GRU order flow signals | `agents/analyst_agents.py` | Yes |
| `StatArbAnalyst` | Archetype 4: Autoencoder/GAT/LSTM signals | `agents/analyst_agents.py` | Yes |
| `DiscretionaryAnalyst` | Archetype 5: ViT/CNN chart pattern signals | `agents/analyst_agents.py` | Yes |
| `MarketMakerAnalyst` | Archetype 6: PPO/SAC/DQN RL signals | `agents/analyst_agents.py` | Yes |
| `ShadowAgent` | Devil's Advocate — rebuttal scoring | `agents/shadow_agent.py` | Yes |
| `PortfolioManager` | Risk gatekeeper + position sizing | `agents/portfolio_manager.py` | Yes |
| `RegimeDetector` | Market state classifier (no LLM) | `agents/regime_detector.py` | No |
| `Journaler` | JSONL memory + error decomposition | `agents/journaler.py` | No |

### 2.2 Orchestrator
The `DebateEngine` (`orchestration/debate_engine.py`) acts as the Orchestrator. It chairs the debate, invokes agents in parallel, runs the Shadow critique, synthesizes with the Orchestrator LLM call, and delegates final sizing to the Portfolio Manager.

### 2.3 Data Flow Diagram

```
Market Event (symbol, timeframe, features)
    │
    ▼
MockModelBridge.get_all_signals()         ← 18 mock model signals (6 archetypes × 3)
    │
    ▼
RegimeDetector.detect()                   ← Classifies: TRENDING_UP / RANGING / etc.
    │
    ├─[FAST PATH: scalping + high confidence]────────────────────────────┐
    │                                                                     │
    ▼                                                                     │
6× AnalystAgent.generate_evidence()       ← Parallel LLM calls          │
    │                                      (ThreadPoolExecutor)          │
    ▼                                                                     │
ShadowAgent.critique_consensus()          ← Devil's Advocate LLM call   │
    │                                                                     │
    ▼                                                                     │
Orchestrator LLM synthesis                ← Final directional call       │
    │                                                                     │
    ├── consensus < 0.60 → NO_TRADE (Anti-Overthinking Rule)             │
    │                                                                     │
    ▼                                                                     │
PortfolioManager.evaluate_and_size()      ← Kelly sizing + risk veto     │
    │                                             ◄──────────────────────┘
    ▼
DebateResult (session_id, packets, trade_order, latency)
    │
    ▼
Journaler.record_debate()                 ← Persisted to JSONL
    │
    ▼
Streamlit Dashboard (agents/ui/dashboard.py)
```

---

## 3. Dual-Path Execution Engine

### 3.1 FAST PATH

**Trigger conditions:** `timeframe ∈ {1m, 3m, 5m}` AND `scalper_confidence ≥ 0.65`

**Logic:** Direct weighted vote aggregation across all 6 archetype signals using regime weights. No LLM calls. Evidence packets built from model signals directly.

**Target latency:** < 200ms  
**Measured latency (dry run):** < 5ms

### 3.2 SLOW PATH (Full Debate)

**Trigger conditions:** All other timeframes OR `force_slow_path=True`

**Logic:**
1. Parallel analyst LLM calls (6 agents, `ThreadPoolExecutor` with 3 workers)
2. Shadow Agent critique LLM call
3. Orchestrator synthesis LLM call
4. Portfolio Manager sizing LLM call
5. Anti-overthinking: `consensus < 0.60 → FLAT`

**Target latency:** < 5,000ms  
**Measured latency (real Ollama `qwen3.5:4b`):** Approximately 6–20s per full debate cycle.

> **LATENCY FAILURE DOCUMENTED:** With `qwen3.5:4b`, the slow path exceeds the 5-second target due to sequential LLM calls for shadow + orchestrator + PM (3 calls × ~2–5s each). Mitigation options:
> - Reduce to 2 parallel LLM streams (shadow + analysts simultaneously)
> - Use `phi4-mini-reasoning:3.8b` which is ~20% faster
> - Cache orchestrator for repeated context within the same regime epoch

---

## 4. Mock Inference Layer (Hot-Swap Design)

### 4.1 MockModelBridge

**File:** `model_bridge.py`  
**Class:** `MockModelBridge`

Generates signals for all 18 models using:
- Feature-informed directional bias (slope, zscore, OFI, inventory)
- Noise from per-archetype historical accuracy distributions (from model registry)
- Gaussian logit perturbation per individual model

**Mock accuracy baseline** (from current registry):
| Archetype | Mock Accuracy | Status |
|---|---|---|
| Trend Follower | 50.6% | ~coin-flip |
| Mean Reversion | 51.7% | Near coin-flip |
| Scalper | 38.4% | Inverted signal pattern (known issue) |
| Stat Arb | 51.3% | Marginal |
| Discretionary | 38.4% | Very poor (needs more training) |
| Market Maker (RL) | 56.4% | Best performer — GAT and PPO are viable |

### 4.2 Hot-Swap Protocol

When models pass Phase 4 validation gates, replace `MockModelBridge` with `ProductionModelBridge`:

```python
# Current (Phase 5)
from model_bridge import MockModelBridge
bridge = MockModelBridge()

# Future (Phase 6 — after training completes)
from model_bridge import ProductionModelBridge
bridge = ProductionModelBridge(registry_path="model_registry.json")
```

`ProductionModelBridge.get_all_signals()` must implement the **identical signature** as `MockModelBridge.get_all_signals()`. The `DebateEngine` requires no changes.

---

## 5. The Iron Wall — Separation of Concerns

The "Iron Wall" ensures agents never access training data or contaminate the temporal split.

| Layer | Accesses | Does NOT Access |
|---|---|---|
| `model_bridge.py` | Trained checkpoint files (weights) | Raw training data |
| `agents/*.py` | LLM (Ollama) + model signals | Feature factory, dataset files |
| `orchestration/debate_engine.py` | Agent evidence packets | Model weights, raw data |
| `agents/journaler.py` | Debate session records | Model internals, training loop |

The agents operate purely on **inference outputs** (signals + features). The training pipeline (`quant_core/`, `data_pipeline/`) is never imported from within the agent system. This prevents:
- Look-ahead leakage in live inference
- Inadvertent reuse of in-sample data for trading decisions
- Cross-contamination of the temporal validation split

---

## 6. Ollama Integration

**Reference implementation:** `d:\kp_ai_agent\AgentAGIv2\entity_core.py`

**Endpoint:** `http://localhost:11434/api/chat`  
**Default model:** `qwen3.5:4b` (fastest, 4.7B Q4_K_M quantized)  
**Available models:** `qwen3.5:4b`, `qwen3.5:9b`, `llama3.1:8b`, `deepseek-r1:8b`, `gemma4:e2b`

### Retry Logic
```python
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5  # seconds (exponential base)
```
Every LLM call is wrapped in exponential-backoff retry (1.5s, 2.25s, 3.375s).

### Self-Correction Logic
Every agent validates JSON after LLM response. If keys are missing, a second call is made with an explicit correction prompt. This prevents malformed JSON from crashing the debate.

### Parallel Calls
The `ThreadPoolExecutor(max_workers=3)` runs up to 3 analyst LLM calls simultaneously, reducing the slow path debate time by ~40–60% versus sequential execution.

---

## 7. Regime Detector

**File:** `agents/regime_detector.py`  
**Method:** Rule-based (feature thresholds) — no LLM required.

8 regime labels with clear feature-based conditions:

| Regime | Trigger |
|---|---|
| `BREAKOUT` | `bb_width > 0.04` (Bollinger Band expansion) |
| `HIGH_VOLATILITY` | `ATR / ATR_mean > 1.5` |
| `LOW_VOLATILITY` | `ATR / ATR_mean < 0.7` |
| `TRENDING_UP` | `slope > 0.001` AND `ema_spread > 0` |
| `TRENDING_DOWN` | `slope < -0.001` AND `ema_spread < 0` |
| `REVERTING` | `\|zscore\| > 1.8` AND `\|slope\| < 0.0005` |
| `RANGING` | `\|slope\| < 0.0005` (flat, no extension) |
| `UNKNOWN` | No features available |

The detector also returns **archetype credibility multipliers** (`regime_weights()`) which the Orchestrator uses to weight analyst votes.

---

## 8. Journaler & Error Decomposition

**File:** `agents/journaler.py`  
**Storage:** `agents/journal/YYYY-MM-DD-debate-journal.jsonl` (daily rotation)

Every debate session is persisted immediately after completion. When trades close, the system calls `update_outcome()` which appends:

```
Loss = Signal_Error + Decision_Error + Execution_Error
```

This anchors the Journaler's memory to ground truth and feeds back into agent credibility scores, implementing the recursive learning loop defined in `Full_Recursive_Learning_Trade_Agents.md`.

---

## 9. Test Results

**File:** `tests/test_phase5.py`  
**Status:** 28/28 PASSED (0 failures)

| Test Class | Tests | Coverage |
|---|---|---|
| `TestMockModelBridge` | 4 | Signal structure, archetype count, price summary |
| `TestRegimeDetector` | 8 | All 8 regimes + weights |
| `TestPortfolioManager` | 6 | Hard veto rules, Kelly sizing bounds |
| `TestEvidencePacket` | 2 | Serialization roundtrip, score bounds |
| `TestJournaler` | 3 | Record, recall, outcome update |
| `TestDebateEngineDryRun` | 5 | Fast/slow path routing, NO_TRADE trigger, latency |

---

## 10. Phase 6 Handoff Notes

When Phase 4 training completes:

1. **Hot-swap bridge:** Implement `ProductionModelBridge` in `model_bridge.py`
2. **Validate signals:** Run `python run_phase5.py --dry-run` to confirm structure unchanged
3. **Update credibility baselines:** Seed `historical_credibility` in each analyst from validation Sharpe/accuracy
4. **Enable live Ollama debate:** Remove `--dry-run` flag
5. **Tune latency:** If slow path exceeds 5s, reduce to 2 LLM calls (combine shadow + orchestrator)
6. **Activate recursive learning:** Call `journaler.update_outcome()` after every live trade closes

---

## 11. Dependency Audit

**New dependencies required** (not previously in project):
- `requests` — for Ollama HTTP calls (already installed in `.venv`)
- `streamlit` — for dashboard UI
- `pandas` — for dashboard dataframes (likely already present)
- `numpy` — already present

**No ChatDev fork dependencies were required** for the backend debate logic. The ChatDev `frontend/` Vue app was reviewed for visualization patterns; the decision was made to implement a **Streamlit dashboard** instead (`agents/ui/dashboard.py`) because:
- ChatDev frontend requires a Node.js build step and Vue toolchain (added complexity)
- Streamlit provides identical visualization capability in pure Python
- The ChatDev server (`server_main.py`) is designed for LLM code generation workflows, not trading signal display
- Streamlit integrates directly with the debate engine without an API layer

> **Note:** If the ChatDev Vue frontend visualization is preferred, the `agents/ui/dashboard.py` can export debate results as JSON to the ChatDev runtime's WebSocket event bus. Contact the team if this integration is needed.
