# Gap Analysis Report

Date: 2026-05-08  
Scope baseline: doc/master_plan.md (target future state)  
Repository: ChatTrader.KPai

## 1. Executive Summary

This project has strong implementation depth in Phases 3-5 and partial implementation in Phase 6, but it is not yet at the complete-state defined in master_plan.md.

High-level completion estimate (against master plan checklist items):
- Phase 3 (Data Engineering): 6/7 complete, 1 partial
- Phase 4 (Quant Core): 4/6 complete, 2 partial
- Phase 5 (Multi-Agent Debate + Ollama): 5/5 complete
- Phase 6 (End-to-End Simulation): 1/5 complete, 3 partial, 1 missing

Overall program state:
- Functional R&D platform: Yes
- Production-hard and HA-ready system: No
- Five-nines design readiness: No (major SLO/DR/operational gaps)

---

## 2. Assessment Method

This analysis maps each master_plan.md checkbox to actual repository implementation and runtime behavior.

Primary evidence sources:
- data_pipeline/* (quality gate, splitter, features, reporting, run_pipeline)
- quant_core/* (data, models, training, validation policy, train entrypoints)
- agents/* and orchestration/debate_engine.py
- backtest.py, model_bridge.py, real_signal_bridge.py
- tests/test_phase3_iron_wall.py, tests/test_phase5.py
- configs/*.yaml

---

## 3. Phase-by-Phase GAP Matrix

## Phase 3: Data Engineering Pipeline

Target checklist from master plan:
1. Build manifest-aware quality gate.
2. Enforce missing-bar threshold rejection (>5%).
3. Implement Iron Wall splitter with purge gap.
4. Implement vectorized feature transforms for all archetypes.
5. Create feature registry and data integrity report generation.
6. Add distribution visualization exports for key features.
7. Add unit tests for leakage and split correctness.

Status:
- Complete:
  - Manifest-aware quality gate implemented in data_pipeline/quality_gate.py.
  - Missing-bar ratio rejection implemented (`max_missing_ratio`, default 5%).
  - IronWall splitter with purge gap implemented in data_pipeline/splitter.py.
  - Feature registry + integrity report generation implemented in data_pipeline/run_pipeline.py.
  - Distribution plots implemented in data_pipeline/reporting.py.
  - Leakage/split correctness tests implemented in tests/test_phase3_iron_wall.py.
- Partial:
  - "Vectorized feature transforms for all archetypes" is only partially complete in Phase 3 pipeline. Current run_pipeline integrates trend/mean_reversion/stat_arb features. Scalper/discretionary feature paths exist as helpers but are not fully unified as first-class pipeline outputs in the same report flow.

Gap severity: Medium

Required closure:
- Extend data_pipeline/run_pipeline.py to output archetype-complete feature products (including scalper microstructure tensors and discretionary-ready artifacts) with lineage metadata.

---

## Phase 4: Quant Core (18 Models)

Target checklist:
1. Implement 3 model classes per archetype (18 total).
2. Add YAML-driven model/training configs.
3. Integrate early stopping (patience=10) and checkpointing.
4. Log train/val/lr to TensorBoard each epoch.
5. Add random-noise sanity propagation test per architecture.
6. Produce model_registry.json with validation audit flags.

Status:
- Complete:
  - 18-model architecture is implemented across quant_core/*_models.py and train entrypoints.
  - YAML-driven config is implemented via configs/*.yaml and quant_core/train_*_phase4.py.
  - Checkpointing and early stopping are implemented broadly in training modules.
  - model_registry output with validation fields and `is_valid` flags is implemented via append_registry/write_*_registry paths.
- Partial:
  - Early stopping does not consistently enforce the master-plan-specific `patience=10`; it is configurable and currently often >10. Functionally present, specification drift exists.
  - TensorBoard coverage is strong, but not uniformly equivalent across all training paradigms (supervised per-epoch detail is better than RL-style loops).
  - Random-noise sanity checks exist in supervised trainers, but are not uniformly explicit as architecture-specific propagation tests in all RL flows.

Gap severity: Medium

Required closure:
- Create a single compliance contract for Phase 4 training loops:
  - Mandatory telemetry schema (train/val/lr/event timing).
  - Mandatory sanity test hook for every architecture path (including RL).
  - Explicit spec alignment policy for patience baseline vs config overrides.

---

## Phase 5: Multi-Agent Debate + Ollama

Target checklist:
1. Implement TraderAgent with model inference hooks.
2. Implement evidence packet.
3. Implement orchestrator debate loop with rebuttal and final sizing.
4. Add retry/self-correct logic for malformed Ollama outputs.
5. Enable parallel LLM calls.

Status:
- Complete:
  - Analyst agents and signal bridge integration implemented (agents/analyst_agents.py, model_bridge.py, real_signal_bridge.py).
  - EvidencePacket implemented in agents/base_agent.py.
  - Debate loop + shadow critique + PM sizing implemented in orchestration/debate_engine.py.
  - Retry and self-correction implemented in base agent/orchestrator/PM parsing flows.
  - Parallel analyst calls implemented with ThreadPoolExecutor in debate engine.

Gap severity: Low (feature-complete vs master plan)

Residual risk:
- Heavy coupling to single local Ollama endpoint with limited fault isolation and no multi-provider fallback policy.

---

## Phase 6: End-to-End Simulation

Target checklist:
1. Build simulation runner: data -> features -> inference -> debate -> trade.
2. Integrate slippage, fees, and latency in backtest loop.
3. Compute Sharpe, profit factor, max drawdown, regime breakdown.
4. Validate OOS consistency rules and invalidate failing models.
5. Produce full run report and reproducibility manifest.

Status:
- Complete:
  - End-to-end simulation runner implemented in backtest.py (pipeline chain exists and executes).
- Partial:
  - Latency is measured and logged, but slippage/fee modeling is not integrated as explicit transaction-cost model in realized PnL path.
  - Max drawdown is computed; however Sharpe/profit factor/regime breakdown are not all present in final backtest report payload.
  - OOS validation exists in separate evaluator pipeline, but invalidation is not enforced as a mandatory gate in live/backtest execution path.
- Missing:
  - Reproducibility manifest (code version, config hash, dataset snapshot id, environment fingerprint, seed lineage) is absent from run outputs.

Gap severity: High

Required closure:
- Introduce a formal execution-grade backtest accounting layer:
  - Explicit fee/slippage/latency execution model.
  - Performance suite: Sharpe, PF, MDD, exposure, turnover, regime metrics.
  - Pre-run and in-run model gating with hard invalidation.
  - Reproducibility manifest emitted per run.

---

## 4. Cross-Cutting Gaps (HA / Reliability Lens)

This section applies high-availability engineering principles to current architecture.

### 4.1 SLO/SLI/Error-Budget Program (Missing)

Current state:
- No explicit service-level objectives for inference, debate, signal publication, or backtest processing reliability.
- No error-budget policy that governs release risk.

Gap:
- Without SLOs, reliability decisions are ad hoc and cannot be prioritized mathematically.

Action:
- Define SLOs now:
  - Debate path availability SLO (e.g., 99.9% successful decision cycles).
  - Latency SLO per timeframe (FAST < 200ms, SLOW < 5s as objective not just logging).
  - Signal pipeline freshness SLO.

### 4.2 Single Points of Failure (High)

Current SPOFs:
- Single local Ollama endpoint (`http://localhost:11434`).
- Local filesystem as primary state store for journals/reports/checkpoints.
- Single-process execution path for debate/backtest orchestration.

Gap:
- Any host failure or local service failure halts the system.

Action:
- Add pluggable inference backends (primary + fallback).
- External durable state store and artifact store.
- Process supervision and restart policy.

### 4.3 Failover and DR Testing (Missing)

Current state:
- Retry exists at request level.
- No explicit failover drills, game days, or DR playbooks with measured RTO/RPO.

Gap:
- Untested failover path is operationally unreliable.

Action:
- Establish DR matrix and tests:
  - Ollama outage drill
  - Checkpoint corruption drill
  - Dataset source unavailable drill
  - Node restart and resume tests

### 4.4 Blast Radius and Graceful Degradation (Partial)

Current state:
- FAST/SLOW path routing and dry-run fallback exist.
- Error handling often logs and continues.

Gap:
- No formal subsystem isolation boundaries or policy-driven degraded mode transitions.

Action:
- Introduce explicit degradation states:
  - LLM unavailable -> deterministic weighted vote mode with tightened risk caps.
  - Model invalidation -> archetype quarantine.
  - Data quality breach -> symbol quarantine.

### 4.5 Operational Readiness (Partial)

Current state:
- Multiple docs exist; runtime logs are present.
- No runbook set with owner, trigger, action, verification checkpoints.

Gap:
- Incident response depends on operator memory.

Action:
- Create runbooks for top 10 incident classes and on-call escalation criteria.

---

## 5. Priority Remediation Plan (Delta to Complete State)

## P0 (Immediate, 1-3 days)

1. Phase 6 accounting hardening:
- Add fee/slippage modeling in backtest PnL.
- Add Sharpe/PF + regime breakdown to report outputs.

2. Reproducibility manifest:
- Include git commit, config digest, dataset manifest digest, seed, Python/torch versions.

3. Gating enforcement:
- Mandatory pre-run model validity check from model_registry and evaluation policy.

## P1 (Short term, 1-2 weeks)

1. Phase 3 completion:
- Promote scalper/discretionary feature outputs to first-class pipeline artifacts.

2. Phase 4 compliance normalization:
- Unified telemetry and sanity-test hooks across all training routines.
- Clarify and enforce patience policy baseline.

3. Reliability controls:
- Backend fallback chain for LLM inference.
- Persistent execution journal with recovery checkpoints.

## P2 (Medium term, 2-6 weeks)

1. HA program bootstrap:
- Define SLIs/SLOs and error budgets.
- Add availability dashboards.

2. DR and resilience validation:
- Game days/tabletop drills with measured RTO/RPO outcomes.
- Fault-injection test harness for critical dependencies.

3. Blast radius controls:
- Symbol-level and archetype-level circuit breakers.
- Graceful degradation policy engine.

---

## 6. Completion Gate Definition (Recommended)

Project should be declared "master-plan complete" only when all are true:

1. Every Phase 3-6 checklist item is either complete or superseded by an accepted architectural decision record.
2. Backtest/execution outputs include full metric suite and reproducibility manifest.
3. Model gating is enforced pre-trade with hard invalidation.
4. SLO/SLI/error-budget policy exists and is observed in operations.
5. Failover and DR drills have been executed and documented with objective pass criteria.

---

## 7. Final Conclusion

The codebase is beyond prototype: it is a capable quant research/trading platform with strong anti-leakage controls, model training infrastructure, and multi-agent orchestration. The largest remaining gap to the master plan is Phase 6 execution realism and production reliability discipline.

In high-availability terms: design intent is solid, but operational resilience and measurable service reliability are not yet engineered to production standards.
