# 30-4-26 Forward Work Plan

## Mission
Primary objective: **All 18 models must pass strict validation and demonstrate real-profit potential in realistic trading conditions**, not only proxy metrics.

This plan is based on:
- Source-code audit (training, data split, evaluation, RL environment, orchestrator)
- Fresh re-validation from latest checkpoints (run on 2026-04-30)
- Cross-check against training working logs and registry artifacts

---

## 1) What Was Verified (Data-Based)

### 1.1 Fresh evaluation executed
- Command executed:
  - `d:/kp_ai_agent/ChatTrader.KPai/.venv/Scripts/python.exe evaluate_all_checkpoints.py`
- Evaluation used real checkpoint files under `models/checkpoints` and rebuilt `model_registry.json`.

### 1.2 Current factual status from fresh run
- Total models evaluated: 18
- Passed by current evaluator gate (`sharpe > 1.0` and `accuracy > 0.52`): **4 models**
  - `GAT_StatArb_v1`
  - `PPO_MM_v1`
  - `SAC_MM_v1`
  - `DQN_MM_v1`
- Failed / resume required: **14 models**
- `Transformer_Trend_v1` failed to load in evaluator due to positional embedding shape mismatch (checkpoint seq len 96 vs evaluator model init seq len 64)

### 1.3 Artifact recency sanity check
Latest checkpoint timestamps confirm active updates today for:
- Trend (LSTM/Transformer/TCN)
- Mean Reversion (MLP/ResNet/GRN)
- Scalper (CNN and LinearAttn epochs up to 21)

Older artifacts remain for:
- Market Maker RL and StatArb (last updated earlier)

### 1.4 Training-log consistency check
`doc/training_more_27-4/27-04-2026_plan_REVISED_workingLog.md` confirms:
- MR models repeatedly converge around 51-52% accuracy but report very high proxy Sharpe in training logs
- Scalper models remain around ~40% validation accuracy range
- Historical RL logs include episodes with repeated low reward behaviors in prior runs

---

## 2) Critical Findings from Real Code (Root Cause Layer)

## 2.1 Metric and gate inconsistency across the system
- `pytorch_model_training_rule.md` demands strict gates (Sharpe > 1.2, Acc > 55%, PF > 1.5, MaxDD < 20%).
- `execute_all_phases.py` uses looser KPI gate (`Sharpe > 1.0` and `Acc > 0.52`).
- `evaluate_all_checkpoints.py` also uses looser thresholds and simplified metric construction.

Impact:
- A model can be marked PASSED in one place but still fail production policy.

## 2.2 Backtest/evaluation logic is not yet production-realistic
In training modules (`mean_reversion_training.py`, `stat_arb_training.py`, `trend_training.py`):
- PnL is computed with sign-proxy logic (example pattern: prediction sign x target sign/value), not order-level execution.
- No symbol-specific fees/slippage/liquidity constraints in the core metric path.

In `evaluate_all_checkpoints.py`:
- Returns are synthetic and normalized (`ret_sign * 0.001`) instead of execution-derived PnL.
- RL models can produce nearly identical metrics because evaluator maps outputs to simplified direction labels.
- Max drawdown sign convention differs from training code (negative drawdown in evaluator vs positive fraction in training utilities).

Impact:
- High risk of metric illusion and non-transferable OOS performance.

## 2.3 Drawdown implementation inconsistency
- `quant_core/shared_training.py` uses equity from cumulative PnL with clipping and positive drawdown ratio.
- `evaluate_all_checkpoints.py` computes drawdown from compounded synthetic returns and reports negative values.

Impact:
- DD values are not comparable across pipelines; pass/fail interpretation can drift.

## 2.4 Transformer evaluation bug
- Evaluator initializes `Transformer_Trend_v1` with seq length 64 while trained config/checkpoint uses 96.
- This causes architecture load failure and removes one model from fair evaluation.

## 2.5 Label/noise design remains weak for several archetypes
Observed from code and behavior:
- Trend and MR targets are still mostly directional next-horizon proxies with weak edge density.
- Scalper still struggles with noisy short-horizon labels and insufficient signal-to-noise ratio.
- Discretionary has severe sample scarcity and remains underpowered.

## 2.6 RL environment quality improved, but evaluation still weakly coupled to execution reality
- `market_maker_env.py` includes inventory, impact, fill probability, and drawdown termination.
- But evaluator does not fully validate RL by replaying environment-level execution economics consistently with training objective.

---

## 3) Decision: What Is Deployable Right Now

## 3.1 Strict recommendation
- **Do not deploy full multi-archetype system yet**.
- Conditional-only deployment candidate:
  - Market Maker RL (small capital, strict risk cap, continuous monitoring)
- All other archetypes remain in retraining / redesign state.

## 3.2 Capital protection policy (immediate)
- Start at 1-2% risk budget for any live MM pilot.
- Hard kill-switch on:
  - Intraday DD > 2%
  - Rolling Sharpe < 0.5
  - Feature drift breach
  - Exchange connectivity anomaly

---

## 4) Forward Work Plan

## Phase A (0-24h): Integrity First (must complete before any new training claims)
1. **Unify gate definitions**
   - Single source of truth for production gates.
   - Remove mixed thresholds (`1.0/0.52` vs `1.2/0.55`).
2. **Fix evaluator architecture loading**
   - Infer Transformer seq length from checkpoint positional embedding shape.
3. **Standardize drawdown conventions**
   - Use one DD definition and sign across all modules.
4. **Freeze metric schema**
   - Registry must always store: metric definition version, gate version, data slice metadata.
5. **Add metric unit tests**
   - PF, Sharpe, DD deterministic tests with known vectors and edge cases.

Exit criteria:
- Re-run evaluator successfully for all 18 models with no architecture-load failure.
- Same model gets same verdict across trainer/evaluator when fed identical predictions.

## Phase B (24-72h): Realistic Backtest Engine (highest ROI)
1. **Build execution-grade PnL simulator module** used by all archetypes
   - Order type, spread crossing, fees, slippage, latency, partial fill, position limits.
2. **Replace sign-proxy PnL in validation paths** with execution PnL.
3. **Per-archetype trade log schema**
   - signal -> order -> fill -> pnl -> inventory -> risk flags.
4. **Cross-validation for simulator realism**
   - Invariants:
     - high Sharpe should not coexist with chronically weak PF unless trade distribution explains it.
     - DD cannot be structurally impossible.

Exit criteria:
- Backtest outputs for at least one archetype are reproducible and execution-consistent.
- Metrics become economically interpretable and stable across reruns.

## Phase C (72h+): Signal Quality Upgrade
1. **Label redesign (signal-centric)**
   - Thresholded and volatility-adjusted targets.
   - Abstain/no-trade label for low-edge regime.
2. **Regime-aware training**
   - Train/eval split by trend/chop/high-vol/low-vol slices.
3. **Walk-forward protocol migration**
   - Replace fixed single split as primary pass criterion.
4. **Multi-seed + hyperparameter search**
   - 3-5 seeds and Optuna for each archetype.
5. **Discretionary dataset expansion**
   - Raise sample count toward >=500k effective samples before expecting production-grade generalization.

Exit criteria:
- At least one non-RL archetype passes strict gates on walk-forward and execution-grade backtest.

## Phase D: Scale to "All Models Must PASS"
1. Lock one archetype at a time.
2. Promote only after paper trading confirmation.
3. Integrate ensemble only after each component is independently profitable.
4. Final integrated validation under portfolio-level risk and correlation constraints.

Exit criteria:
- 18/18 pass strict gates.
- Paper/live shadow run confirms positive net PnL after costs.

---

## 5) Concrete Task Board (Next 7 Days)

### D1-D2
- Patch evaluator seq-len loading for Transformer
- Normalize DD computation and sign
- Implement metric consistency tests
- Regenerate `model_registry.json` with versioned metric metadata

### D2-D4
- Implement shared execution simulator module
- Integrate simulator into Trend/MR/StatArb validation first
- Compare old proxy metrics vs new execution metrics

### D4-D5
- Redesign labels for Trend + MR + Scalper
- Add abstain/no-trade filtering and confidence thresholding

### D5-D7
- Run walk-forward + multi-seed for one pilot archetype (recommended: StatArb or Trend)
- Decide go/no-go for broader retraining rollout

---

## 6) Risk Register (Blocking Risks)
1. Metric mismatch across modules can produce false PASS.
2. Synthetic evaluator returns can overstate transferability to live execution.
3. Label noise can dominate training despite longer epochs.
4. RL policy quality may be misread if not validated on execution-consistent environment replay.
5. Mixed checkpoint directories (`checkpoints` vs `checkpoints_verify`) can contaminate conclusions.

---

## 7) Definition of Done (Program Level)
A model (and eventually all 18 models) is considered production-ready only if:
1. Passes unified strict gates (single gate source).
2. Passes walk-forward robustness checks.
3. Passes execution-grade backtest with realistic costs.
4. Shows stable paper-trading behavior over required sample size.
5. Has complete audit trail (dataset version, feature schema, checkpoint hash, code version, gate version).

---

## 8) Current Bottom Line (as of 2026-04-30 re-validation)
- Foundation/process direction is strong.
- Current model performance is still not enough for full production deployment.
- Most urgent leverage is **evaluation realism + metric integrity** before additional brute-force retraining.
- Immediate target is to make every pass/fail decision economically trustworthy first; then optimize models.
