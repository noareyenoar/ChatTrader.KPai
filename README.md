# ChatTrader.KPai

**An institutional-grade AI quantitative trading system built around a Multi-Agent Debate engine.**

Rather than a single model making a trade call, ChatTrader.KPai simulates a team of specialists — each with a different trading philosophy — that argue, challenge each other, and only commit capital when there is genuine consensus. The architecture mirrors how a real trading desk operates: analysts present theses, a devil's advocate attacks them, and a portfolio manager controls risk before any order is placed.

> **Status:** Phase 4 in progress — 18 neural network models training across 6 archetypes. Phase 5 (Multi-Agent Debate) begins when training gates are met.

---

## How it works

The system runs as a deterministic six-stage pipeline:

```
Binance Historical Data
        ↓
Feature Factory          ← archetype-specific tensors, no lookahead
        ↓
18 Neural Network Models ← 6 trading archetypes × 3 architectures each
        ↓
Multi-Agent Debate       ← 6 Analysts + 1 Shadow Agent + 1 Portfolio Manager
        ↓
Vectorized Backtest      ← with realistic fees, slippage, and drawdown gates
        ↓
Recursive Learning       ← agents read their own past failures before each debate
```

Every stage is designed to be independently auditable. A model that passes statistical gates but fails the backtest is rejected. An agent that ignores a correct Shadow Agent warning is penalized in future debates.

---

## The Six Trading Archetypes

Each archetype represents a distinct market hypothesis. Three neural network architectures are trained per archetype, giving 18 models total.

| Archetype | Market Hypothesis | Neural Architectures |
|---|---|---|
| **Trend Follower** | Price momentum persists across regimes | LSTM, Transformer, TCN |
| **Mean Reversion** | Over-extended prices return to their mean | MLP, ResNet, GRN |
| **Scalper** | Order flow imbalance predicts short-term direction | CNN, Linear Attention, GRU |
| **Statistical Arbitrage** | Related assets have predictable spread relationships | Autoencoder, GAT, LSTM |
| **Discretionary** | Price action patterns repeat, context matters | ViT, CNNChart, Multimodal |
| **Market Maker** | Profit comes from spread, not direction | PPO, SAC, DQN (RL) |

Two additional models sit outside the archetype grid:

- **TG-MNN** — predicts wave structure (magnitude, duration, Markov state) rather than price direction
- **APV-PLN** — uses an Oracle Teacher (LUPI framework) to model the price-volume relationship as a probability span

---

## Multi-Agent Debate (Phase 5)

Once models are trained, they are handed to a team of agents running on a local LLM via Ollama.

**The cast:**

- **6 Analysts** — one per archetype. Each presents a signal, a confidence score, and a written thesis for why the trade makes sense from their perspective.
- **1 Shadow Agent** — the devil's advocate. Its only job is to find the weakest assumption in the current thesis and attack it. Exists to prevent groupthink.
- **1 Portfolio Manager** — the final gatekeeper. Controls position sizing, checks cross-model correlation, and can veto the Orchestrator if risk limits are breached.
- **1 Orchestrator** — chairs the debate and makes the directional call.
- **1 Regime Detector** — identifies the current market state (trending, ranging, volatile) and adjusts analyst credibility weights accordingly.

**Two execution paths based on timeframe:**

- **Fast Path** (scalping): Bypasses the full debate. If signal confidence exceeds threshold, the Portfolio Manager executes directly. Target latency < 200ms.
- **Slow Path** (swing/position): Full debate loop. If consensus score falls below 60%, the Orchestrator outputs `NO TRADE`. Analysis paralysis is penalized, not rewarded.

**Recursive Learning:**
Every closed trade is decomposed into `Signal Error + Decision Error + Execution Error`. If the Orchestrator ignored a correct Shadow Agent warning, the Orchestrator's credibility on that logic path is penalized. Agents are required to read their own logged failures before the next debate on a similar setup.

---

## Data Anti-Leakage Policy ("Iron Wall")

This project treats look-ahead bias as a disqualifying defect, not a warning.

- **Chronological split:** 70% train / 15% validation / 15% test. No random shuffling.
- **Purge gap:** A buffer equal to the maximum prediction horizon is removed between each split boundary to prevent information bleeding.
- **Scaler leakage:** Scalers are fitted exclusively on the training set. Validation and test sets are transformed using the fitted parameters only — never refitted.
- **Walk-forward validation:** Used for regime-sensitive architectures (Trend Following) to prevent models from memorizing a single market regime.

The `IronWallSplitter` in `data_pipeline/splitter.py` enforces these constraints and raises an exception if any timestamp ordering violation is detected.

---

## Production Readiness Gates

A model is not considered trained until it clears all four gates simultaneously on the held-out test set, with transaction costs applied:

| Gate | Threshold |
|---|---|
| Net Sharpe Ratio | > 1.2 (after 0.04% commission + 1-2 tick slippage) |
| Profit Factor | > 1.5 |
| Max Drawdown | < 20% |
| Directional Accuracy | > 55% |

Models that pass statistical gates are also subjected to a Monte Carlo stress test: 1,000 random shuffles of the trade sequence. The 95th percentile worst-case drawdown must remain below 20%.

---

## Current Training Status (Phase 4)

> Last updated: May 2026

| Model | Archetype | OOS Test Sharpe | Status |
|---|---|---|---|
| GAT_StatArb_v1 | Statistical Arbitrage | 1.97 | Training — evaluator fix in progress |
| Autoencoder_StatArb_v1 | Statistical Arbitrage | 1.45 | Training — evaluator fix in progress |
| TG-MNN | Wave Structure | State Acc 52.4% | Passing |
| MLP_MR_v1 | Mean Reversion | Prior run: 1.48 | Re-evaluation pending |
| SAC_MM_v1 | Market Making | Prior run: 2.05 | Re-evaluation pending |
| All others | Various | Below gate | Retraining in progress |

**Current blockers being resolved:**
- Trend Following: switching from fixed split to Walk-Forward Validation to fix regime overfitting
- Scalper: DirectML operator compatibility fix (ELU → SiLU) applied, retraining
- Discretionary: label threshold inconsistency between training and evaluation corrected, retraining
- StatArb: Sharpe metric computation corrected from reconstruction error to spread-trading P&L
- Market Making evaluator: replacing directional proxy with proper episode-based simulation

---

## Hardware Setup

Developed on a single consumer machine using AMD DirectML:

| Component | Spec |
|---|---|
| GPU | AMD Radeon RX 6750 (DirectML via `torch-directml`) |
| Python | 3.11.9 |
| PyTorch | 2.4.1 |
| DirectML plugin | 0.2.5 |
| Training backend | DirectML for most models; CPU for StatArb Autoencoder (DML is slower for this arch) |

Notable constraint: DirectML does not support all PyTorch operators natively. Several operators fall back to CPU mid-forward-pass, which silently corrupts gradient flow. The codebase explicitly avoids `aten::elu` and `aten::log_sigmoid_forward` in favor of DML-native equivalents.

---

## Project Structure

```
ChatTrader.KPai/
├── data_pipeline/          # Feature factory, Iron Wall splitter, quality gates
│   ├── features.py
│   ├── splitter.py         # IronWallSplitter + PurgedWalkForwardSplitter
│   └── gpu_utils.py
├── quant_core/             # 18 model architectures + training loops
│   ├── train_*_phase4.py   # Entry points per archetype
│   ├── *_models.py         # Architecture definitions
│   ├── *_data.py           # Data loading + label generation
│   └── *_training.py       # Training loops
├── agents/                 # Phase 5 multi-agent debate system
│   ├── analyst_agents.py
│   ├── shadow_agent.py
│   ├── portfolio_manager.py
│   └── regime_detector.py
├── orchestration/
│   └── debate_engine.py
├── configs/                # YAML configs per archetype
├── models/checkpoints/     # Saved model weights
├── evaluate_all_checkpoints.py   # OOS evaluation entry point
└── backtest.py
```

---

## Roadmap

- **Phase 1–2** ✅ Data pipeline, feature factory, Iron Wall splitter
- **Phase 3** ✅ Quality gates, feature validation, data integrity reports
- **Phase 4** 🔄 18 model training — in progress
- **Phase 5** ⬜ Multi-agent debate via Ollama local LLM
- **Phase 6** ⬜ End-to-end simulation with fees, slippage, recursive learning loop

---

## Design Philosophy

Most ML trading systems fail in production for one of three reasons: data leakage during development, overfitting to a single market regime, or models that look good in isolation but make conflicting decisions when combined.

This project tries to address all three explicitly:

- Leakage is prevented by architecture (splitter raises exceptions, not warnings)
- Regime overfitting is addressed by Walk-Forward Validation and a Regime Detector that adjusts agent weights in real time
- Conflicting signals are resolved by structured debate with a devil's advocate and a portfolio manager that controls for cross-model correlation

The system is not designed to predict the market. It is designed to make decisions that are robust to being wrong.

---

*Built on PyTorch · Local LLM via Ollama · Binance Futures data*
