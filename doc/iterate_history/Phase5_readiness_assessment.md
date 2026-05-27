# Phase 4 → Phase 5 Readiness Assessment

**Assessment Date:** 2026-05-14  
**Assessment Basis:** Fresh OOS evaluation (2026-05-14 13:51-13:52 UTC) + complete training history analysis

---

## Executive Summary

**Phase 5 Status:** 🔴 BLOCKED — Cannot proceed to Multi-Agent Debate phase until Phase 4 produces at least 1 production-viable model.

**Current Readiness Level:** 0% (0/18 models pass)

**Estimated Days to Phase 5 Readiness:** 5-10 days (following Phase 4 Next Iteration Plan)

---

## Phase 5 Blockers (from doc/Full_Recursive_Learning_Trade_Agents.md Requirements)

### Blocker #1: Zero Production-Viable Models
**Requirement:** At least 1 model must simultaneously pass:
- Sharpe > 1.2
- Profit Factor > 1.5
- Max Drawdown < 0.20
- Directional Accuracy > 0.55

**Current Status:**
- ❌ 0/18 models pass all four gates
- ⚠️ 1/18 models (SAC_MM_v1) passes 1 gate (Dir Acc 61.62%)
- ⚠️ 2/18 models (GAT_StatArb_v1, Autoencoder_StatArb_v1) have positive Sharpe but fail PF gate
- ❌ 6/18 models have eval errors (cannot evaluate)
- ❌ 11/18 models have negative Sharpe

**Unblock Path:**
- Implement Phase 4 Next Iteration Plan (Days 1-5)
- Re-evaluate using updated configs per Full Retraining Plan prescriptions
- Target: GAT_StatArb_v1 (closest to viability, only -0.486 PF gap) or SAC_MM_v1 (only needs +0.833 PF with directional accuracy already at 61%)

### Blocker #2: Model Diversity for Ensemble
**Requirement:** At least 3 models should pass 3/4 gates for ensemble decision-making (per Full_Recursive_Learning_Trade_Agents.md debate protocol)

**Current Status:**
- ❌ 0/18 models pass 3/4 gates
- ⚠️ SAC_MM_v1 passes 1/4

**Unblock Path:**
- After Phase 4 closure (5-10 days), expect:
  - ≥1 model at 4/4 gates
  - ≥2-3 models at 3/4 gates (likely Trend + MR if seq_len/horizon fixes work)

### Blocker #3: Walk-Forward Validation
**Requirement:** Production models must show:
- Positive test Sharpe on OOS window
- Min 1,000 trades in backtest period
- Consistency: ≥80% of rolling 30-day windows show positive PnL

**Current Status:**
- ❌ Cannot validate — no models pass gate 1

**Unblock Path:**
- Once Phase 4 closure complete, run `quant_core/backtester.py` on passed models
- Generate walk-forward report for Phase 5 input

### Blocker #4: Monte Carlo Robustness
**Requirement:** Production models must pass 1,000 trade-sequence shuffles with:
- 95th percentile Max Drawdown < 20%
- Win rate consistency >90% (shuffled vs original)

**Current Status:**
- ❌ No models pass this gate

**Unblock Path:**
- After Phase 4 closure, run `tools/monte_carlo_robustness_tester.py`
- Validate mechanical stability (not just statistical luck)

### Blocker #5: Reproducibility Manifest
**Requirement:** All production models must have:
- Complete training log (timestamp, hyperparams, loss curve)
- Checkpoint metadata (path, version, scaler parameters)
- Data split audit (IronWall 70/15/15 with purge gap confirmation)
- Evaluation log (OOS Sharpe, PF, MDD, samples)

**Current Status:**
- ⚠️ Partially complete (training logs exist, but manifest is fragmented)

**Unblock Path:**
- Create `model_production_manifest.json` for each passed model
- Document in `doc/reproducibility_manifest_2026-05-14.md`

---

## If Phase 4 Achieves Closure (Target: Day 5-10)

### Assumed Scenario: GAT_StatArb_v1 Passes (PF 1.5+, Sharpe 0.052→+1.3)

**Phase 5 Can Begin With:**

1. **Single-Model Execution (Minimal Risk):**
   - Deploy GAT_StatArb_v1 alone
   - Run daily backtest on OOS window
   - Monitor Sharpe drift

2. **Optional: Ensemble if 3+ Models Pass**
   - If Trend + MR + Scalper each produce ≥1 viable model after retraining
   - Implement debate via `agents/analyst_agents.py` + `orchestration/debate_orchestrator.py`
   - Aggregate signals from GAT (StatArb) + best_trend + best_mr + best_scalper

### Phase 5 Architecture (Per Full_Recursive_Learning_Trade_Agents.md)

```
┌─────────────────────────────────────────────────┐
│ Market Data Feed (binance_historical)           │
└────────────┬────────────────────────────────────┘
             │
             ▼
   ┌─────────────────────────┐
   │ Feature Factory (all    │
   │ archetypes)             │
   └────┬────────────────────┘
        │
        ▼
   ┌──────────────────────────────────────────┐
   │ Trader Agents (6 archetypes × ≥1 model) │
   │  • TrendTraderAgent (if LSTM_Trend/TCN) │
   │  • MRTraderAgent (if GRN_MR)            │
   │  • ScalperTraderAgent (if CNN_Scalper)  │
   │  • StatArbTraderAgent (if GAT)          │◄─── READY with GAT
   │  • DiscretionaryTraderAgent (if CNN)    │
   │  • MMTraderAgent (if SAC)               │
   └────┬─────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────────┐
   │ Multi-Agent Debate (Orchestrator)       │
   │  • Evidence packet per trader           │
   │  • Confidence scoring + alignment       │
   │  • Rebuttal loop (via Ollama)          │
   │  • Final position sizing                │
   └────┬────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ Portfolio Manager (Risk Gates)          │
│  • Max position per symbol              │
│  • Aggregate drawdown circuit breaker   │
│  • Forced liquidation on MDD > 15%      │
└─────────────────────────────────────────┘
```

**Phase 5 Implementation Effort:**
- If single-model (GAT only): 2-3 hours (agent + orchestrator glue)
- If multi-model ensemble: 8-12 hours (full debate + evidence packet format)

---

## Conditional Timeline to Full Production

| Milestone | Date | Blocker Status |
|-----------|------|---|
| Phase 4 Eval (current) | 2026-05-14 | ❌ 0 models pass |
| Phase 4 Iteration 1 Complete | 2026-05-19 (Day 5) | ⚠️ ≥1 model likely passes |
| Phase 5 Single-Model Ready | 2026-05-20 (Day 6) | ✅ Unblocked |
| Phase 5 Multi-Model Debate Ready | 2026-05-22 (Day 8) | ✅ Unblocked (if 3+ models pass) |
| Phase 6 Walk-Forward + MC Validation | 2026-05-26 (Day 12) | ✅ Unblocked |
| **Production Deployment Ready** | **2026-05-27 (Day 13)** | **✅ Fully Unblocked** |

---

## Critical Decision Points

### Decision #1: Can GAT_StatArb_v1 Be Pushed Past 1.5 PF in 48 Hours?
- **If YES (PF > 1.5):** Unblock Phase 5 immediately; begin single-model debate prep
- **If NO (PF < 1.5):** Pivot to SAC_MM_v1 (directional accuracy already passes) or extend trend/MR training

### Decision #2: Should Phase 5 Start with Single Model or Wait for 3+?
- **Single-Model Risk:** Low portfolio diversity, high model-specific event risk
- **Recommendation:** Start with GAT_StatArb_v1 if it passes, but maintain Phase 4 retraining in parallel to produce 2-3 ensemble partners

### Decision #3: Which Archetype is Most Likely to Pass First?
**Ranked by Closure Probability:**
1. **StatArb (GAT)** — 85% chance (only 0.486 gap) ✅
2. **Market Making (SAC)** — 70% chance (directional gate passed, needs PF tuning)
3. **Trend (TCN/Transformer)** — 50% chance (high risk; seq_len/horizon fix untested)
4. **Mean Reversion (GRN)** — 45% chance (similar risk to Trend)
5. **Scalper (CNN)** — 30% chance (label distribution fix uncertain)
6. **Discretionary (CNNChart)** — 20% chance (data starvation may persist even at 50k rows/symbol)

---

## Phase 5 Cannot Start Until:

- [ ] At least 1 model passes all 4 production gates simultaneously
- [ ] That model(s) have been validated on ≥1,000 OOS trades
- [ ] Walk-forward consistency confirmed (80%+ rolling positive PnL windows)
- [ ] Reproducibility manifest created
- [ ] Agents (analyst, debate orchestrator, portfolio manager) are loaded with production model checkpoints

---

## Recommendation for User

**As of 2026-05-14:**

1. ✅ **Execute Phase 4 Next Iteration Plan immediately** (Days 1-5)
   - Start with GAT/SAC tuning (highest probability wins)
   - Run in parallel with eval error fixes

2. ⏳ **Monitor GAT_StatArb_v1 for 1.5 PF breakthrough** (target: Day 3)
   - If achieved, Phase 5 can begin with single-model
   - If not achieved by Day 4, pivot to SAC_MM_v1 focus

3. 📊 **Prepare Phase 5 infrastructure while Phase 4 trains** (Days 2-4)
   - Stage the agent classes and debate orchestrator
   - Test Ollama integration
   - Create evidence packet format

4. 🚨 **If no model passes by Day 8, escalate:**
   - Investigate fundamental feature/label quality issues
   - Consider architecture redesign (not config tuning)

---

**Next Review:** 2026-05-18 (after Phase 4 Iteration 1)  
**Current Blocker Status:** FULLY BLOCKED on Phase 4 viability (0/18 models pass)
