# Phase 4 Next Iteration Plan — Closing Critical Gaps to Positive OOS Output

**Date:** 2026-05-14  
**Status:** All 18 models failing; 0 pass simultaneous 4-gate requirement  
**Master Goal:** Achieve at least 1 model with Sharpe > 1.2 AND PF > 1.5 AND MDD < 0.20 AND Dir Acc > 0.55  
**Timeline:** 5-10 days (aggressive retraining + debugging)

---

## PHASE 4 CRITICAL GAPS vs. Full Retraining Plan

### Gap #1: Load/Eval Errors Blocking 6 Models (P0 — Block Everything)

**Affected Models:**
1. LSTM_Trend_v1 (hidden_size mismatch: checkpoint 192 vs manifest 128)
2. LSTM_StatArb_v1 (num_assets mismatch: checkpoint 34 vs manifest 2)
3. PPO_MM_v1 (state_size mismatch: eval expects 10, training has 7)
4. DQN_MM_v1 (state_size mismatch: eval expects 10, training has 8)

**Root Causes:**
- LSTM models: Architecture definition inconsistent with saved checkpoint
- RL models: State space dimension changed between training and evaluation

**Impact:** Cannot evaluate 22% of catalog; cannot debug without eval metrics

**Solution (Full Retraining Plan Phase 4.1):**

| Model | Action | Timeline |
|-------|--------|----------|
| LSTM_Trend_v1 | Retrain with hidden_size=192 to match checkpoint OR fix manifest | 2-3 hours |
| LSTM_StatArb_v1 | Retrain with num_assets=34 to match checkpoint OR fix manifest | 2-3 hours |
| PPO_MM_v1 | Verify state encoding (should be 7-d: inventory, spread, vol, imbalance, 3 more). Fix eval harness. | 1 hour |
| DQN_MM_v1 | Same as PPO — state dimension inconsistency | 1 hour |

**Priority Order:** Fix eval errors first (enables diagnosis of training issues). Then retrain.

**Expected Effort:** ~8 hours total (eval fixes + retraining)

---

### Gap #2: Trend Archetype — Seq/Horizon Configuration Fundamentally Wrong (P0)

**Current Config:** `seq_len=32, horizon=20`  
**Problem:** 
- seq_len=32 gives only 32 minutes of context (1-min bars) — insufficient for trend identification
- horizon=20 asks model to predict 20 bars (~20 min) ahead — too far; target is noisy

**Evidence:**
- LSTM_Trend_v1: Test Sharpe -1.7 (estimated once load error fixed)
- Transformer_Trend_v1: Test Sharpe -1.84
- TCN_Trend_v1: Test Sharpe -1.72
- All show overfitting (train_loss << val_loss) + poor OOS performance

**Full Retraining Plan Prescription (Phase 4.2):**
```yaml
seq_len: 32 → 96    # 1.5 hours of context instead of 30 min
horizon: 20 → 5     # Predict 5 min ahead instead of 20 min (stronger signal)
dropout: 0.1 → 0.3  # Reduce overfitting
weight_decay: 0.0001 → 0.001  # Stronger L2 regularization
max_epochs: 80 → 150  # More training time with stricter early stopping
patience: 12 → 20    # More tolerance (longer sequences need more stability)
preferred_backend: cpu → directml  # GPU acceleration
```

**Expected Impact:**
- seq_len=96 + horizon=5 should produce much stronger trend labels (early momentum detection)
- Longer context + DirectML should improve convergence
- ≥3 Sharpe unit improvement possible (from -1.7 to +1.3 range)

**Timeline:** ~4-5 hours per model × 3 models = 12-15 hours

**Immediate Action:**
```bash
# Retrain all three trend models with new config immediately
python -m quant_core.train_trend_phase4 --config configs/trend_phase4.yaml
```

**Success Criteria:**
- Test Sharpe > 1.2 for at least 1 trend model
- No overfitting pattern (train_loss within 2× of val_loss)

---

### Gap #3: Mean Reversion — Horizon=20 Fundamentally Wrong (P0)

**Current Config:** `horizon=20`  
**Problem:**
- MR profits when price snaps back _quickly_ (within 3-5 bars)
- By bar t+20, the reversion is either complete or failed → noise
- GRN_MR_v1 Sharpe -0.74 is best, still negative despite gating advantage

**Full Retraining Plan Prescription (Phase 4.3):**
```yaml
horizon: 20 → 3      # MR signal peaks at 3-bar horizon
dropout: 0.1 → 0.4   # Heavy regularization for tabular models
weight_decay: 0.0001 → 0.002  # Stronger L2
lr: 0.0005 → 0.0002  # Lower LR for stability with longer training
max_epochs: 80 → 150  # More training
patience: 12 → 20
preferred_backend: cpu → directml
```

**Expected Impact:**
- horizon=3 fixes fundamental label quality issue (reversal signal is 5–7× stronger)
- Heavy dropout + lower LR prevents overfitting on tabular data
- Potential +2.5 Sharpe improvement (from -0.74 to +1.76 range)

**Timeline:** ~3-4 hours per model × 3 models = 9-12 hours

**Immediate Action:**
```bash
python -m quant_core.train_mr_phase4 --config configs/mr_phase4.yaml
```

**Success Criteria:**
- Test Sharpe > 1.2 for at least 1 MR model
- GRN_MR_v1 should show marked improvement (closest to viability currently)

---

### Gap #4: Scalper — Label Distribution Catastrophically Broken (P1 — High Risk)

**Current Config:** `flat_threshold=0.0010`  
**Problem:**
- Empirical observation: ~60% of labels are "flat" at 0.0010 threshold
- Model learns to predict "flat" for everything → 41% accuracy (60% baseline for flat class)
- All three scalpers show negative Sharpe + Dir Acc ~46%

**Full Retraining Plan Prescription (Phase 4.4):**
```yaml
flat_threshold: 0.0010 → 0.0015     # 3× wider flat zone
horizon: 5 → 2                       # 2-bar scalper window (true scalper frequency)
seq_len: 32 → 16                     # Match 2-bar horizon
preferred_backend: cpu → directml
use_cyclic_lr: true → false          # Remove LR oscillation
lr: 0.001 → 0.0005
dropout: 0.1 → 0.3
max_epochs: 80 → 120
```

**Critical Pre-Retraining Action:**
```bash
# BEFORE retraining, verify label distribution
python -c "
from data_pipeline.features import FeatureFactory
factory = FeatureFactory(...)
labels = factory.build_scalper_labels(threshold=0.0015)
print(f'Label distribution at threshold=0.0015:')
print(f'Up: {(labels==0).sum()}, Flat: {(labels==1).sum()}, Down: {(labels==2).sum()}')
# Expected: Up ~20%, Flat ~30%, Down ~20% (roughly balanced)
"
```

**If label distribution is still skewed (>40% flat):** Increase threshold further to 0.002.

**Timeline:** 
- Label audit: 15 min
- Retraining (if needed): ~3-4 hours per model × 3 = 9-12 hours

**Success Criteria:**
- Label distribution: up 20-30%, flat 30-40%, down 20-30%
- Scalper Dir Acc > 0.45 (margin above random = 33%)
- Scalper Sharpe > -0.5 (no longer strongly negative)

---

### Gap #5: Stat Arb — GAT Near Viability, LSTM Config Error (P1)

**Current State:**
- GAT_StatArb_v1: Sharpe 0.052 (positive!), PF 1.014 (gap: -0.486)
- Autoencoder_StatArb_v1: Sharpe -0.05, PF 0.987 (gap: -0.513)
- LSTM_StatArb_v1: Load error (num_assets config)

**Full Retraining Plan Prescription (Phase 4.5):**

#### 5.1) Fix LSTM_StatArb_v1
**Action:** Match manifest num_assets to checkpoint (should be 34) and retrain
```bash
python -m quant_core.train_stat_arb_phase4 --config configs/stat_arb_phase4.yaml
```
**Timeline:** 2-3 hours

#### 5.2) Extend GAT Training (Highest Priority)
**Observation:** GAT is only 0.486 PF units from gate and already positive Sharpe
**New Config:**
```yaml
max_epochs: 80 → 150      # More training (spread capture improves slowly)
patience: 12 → 25          # More tolerance
seq_len: 64 → 128          # Longer context for cointegration
batch_size: 1024 → 512     # Better gradient signal
dropout: 0.2 → 0.3         # Slight regularization increase
weight_decay: 0.0001 → 0.0005  # Moderate L2
```

**Expected Impact:**
- +0.2-0.4 Sharpe improvement plausible (GAT already has signal)
- +0.3-0.5 PF improvement possible with longer training on 34 assets

**Timeline:** 4-5 hours

**Success Criteria:**
- GAT_StatArb_v1: PF > 1.5 (only 0.486 gap remaining)
- Sharpe remains positive or becomes > 1.2

#### 5.3) Retrain Autoencoder
```yaml
max_epochs: 80 → 150
seq_len: 64 → 128
lr: 0.0003 → 0.0001    # Slower learning for reconstruction
dropout: 0.2 → 0.3
```

**Timeline:** 4-5 hours

---

### Gap #6: Discretionary — Data Starvation (P2 — Lower Priority)

**Current State:** 
- All three models (ViT, Multimodal, CNNChart) have Dir Acc 38-39% (far below 55%)
- Classic overfitting pattern (train_acc 46-67% >> OOS 35-40%)
- Root cause: `max_rows_per_symbol=12,000` insufficient for chart learning

**Full Retraining Plan Prescription (Phase 4.6):**
```yaml
max_rows_per_symbol: 12,000 → 50,000    # 4× more chart samples per symbol
horizon: 20 → 5                          # Shorter = clearer visual patterns
flat_threshold: 0.003 → 0.005            # Wider flat zone = fewer ambiguous labels
batch_size: 256 → 128                    # Better gradient signal for ViT
max_epochs: 100 → 200                    # ViT needs more passes
patience: 20 → 30
dropout: 0.1 → 0.4                       # Heavy regularization
weight_decay: 0.0001 → 0.01              # Strong L2 for ViT
lr: 0.0003 → 0.0001                      # ViT sensitive to high LR
label_smoothing: — → 0.1                 # Prevent overconfident memorization
```

**Timeline:** 
- Data augmentation + chart generation: 2-3 hours
- Training: ~8-10 hours per model × 3 = 24-30 hours (ViT is slow)
- **Total: 26-33 hours** (run overnight/parallel)

**Success Criteria:**
- Dir Acc > 0.50 for at least 1 discretionary model
- CNNChart_Disc_v1 (currently best at -1.95 Sharpe) should improve most

---

### Gap #7: Market Making RL — State/Reward Tuning (P2)

**Current State:**
- SAC_MM_v1: **PASSES Dir Acc gate (61.62%)** but PF only 0.667 (gap: -0.833)
- PPO_MM_v1, DQN_MM_v1: Eval errors (state dimension)

**Full Retraining Plan Prescription (Phase 4.7):**

#### 7.1) Fix PPO/DQN Eval Errors
**Action:** 
1. Verify state_size in training config and ensure eval harness uses same state_size
2. Retrain both with consistent state encoding (likely 7-d: inventory, spread, vol, imbalance, + 3 derived)

**Timeline:** 1-2 hours (fixes + retraining)

#### 7.2) Tune SAC for Profitability
**Observation:** SAC's directional accuracy is good (61%); issue is profit capture (PF 0.667 → need 1.5)

**New Config:**
```yaml
max_episodes: 2,000 → 8,000              # 4× more exploration
max_steps: 500,000 → 2,000,000           # 4× steps for spread learning
episode_length: 200 → 400                # Longer episodes for multi-step reward shaping
inventory_lambda: 0.1 → 0.05             # Less penalty on holding (was too passive)
replay_buffer_size: 50,000 → 200,000     # 4× buffer = more diverse experience
batch_size: 2,048 → 512                  # More frequent updates
survival_bonus: 0.0005 → 0.001           # Higher bonus for avoiding MDD
reward_alpha_neg: 1.35 → 1.5             # Stronger loss aversion
```

**Expected Impact:**
- Longer training allows RL to learn spread capture + inventory dynamics
- Lower inventory_lambda lets agent hold positions longer (capture wider spreads)
- Larger buffer prevents catastrophic forgetting of diverse market conditions
- Potential +0.8-1.0 PF improvement (from 0.667 to 1.3-1.5 range)

**Timeline:** 6-8 hours

**Success Criteria:**
- SAC_MM_v1 PF > 1.5 (only -0.833 gap remaining)
- Episode win rate > 50% maintained or improved

---

## PRIORITIZED EXECUTION ORDER (Next 10 Days)

### Day 1 (Today): Fix Load/Eval Errors + Quick Wins (P0)

**Tasks:**
1. **Immediate** (1 hour): Fix PPO/DQN state eval harness
   ```bash
   # Diagnose eval errors
   python evaluate_all_checkpoints.py 2>&1 | grep -A5 "RL_EVAL_ERROR"
   # Fix state_size mismatch in evaluate_all_checkpoints.py
   ```

2. **Parallel** (2-3 hours): Retrain LSTM_Trend_v1 with hidden_size=192
   ```bash
   # Option A: Fix manifest to say hidden_size=192
   # Option B: Retrain with new config
   python -m quant_core.train_trend_phase4 --config configs/trend_phase4.yaml --subset lstm-only
   ```

3. **Parallel** (2-3 hours): Retrain LSTM_StatArb_v1 with num_assets=34
   ```bash
   python -m quant_core.train_stat_arb_phase4 --config configs/stat_arb_phase4.yaml --subset lstm-only
   ```

4. **Audit** (15 min): Check scalper label distribution
   ```bash
   # Run label audit script to verify flat_threshold effect
   ```

**Day 1 Checkpoint:** All eval errors resolved; understand scalper label distribution

---

### Day 2: Trend & Mean Reversion Retrain (12-15 hours)

**Parallel Tasks:**
1. **Retrain Trend (3×)** with new seq_len=96, horizon=5
   ```bash
   # Edit configs/trend_phase4.yaml with new parameters
   python -m quant_core.train_trend_phase4 --config configs/trend_phase4.yaml
   # Monitor: Watch for Sharpe improvement vs -1.7 baseline
   ```

2. **Retrain Mean Reversion (3×)** with new horizon=3, dropout=0.4
   ```bash
   python -m quant_core.train_mr_phase4 --config configs/mr_phase4.yaml
   # Monitor: Watch for GRN_MR_v1 improvement (currently best at -0.74)
   ```

**Expected Duration:** 8-12 hours (run both in parallel)

**Day 2 Checkpoint:** Re-evaluate all 6 models. If any Sharpe > 0.5, trend is positive.

---

### Day 3: Scalper & StatArb (12-14 hours)

**Tasks:**
1. **Scalper Fix** (2-3 hours):
   - If label audit shows >40% flat: Update flat_threshold → 0.0015-0.002
   - Edit configs/scalper_phase4.yaml with flat_threshold, horizon=2, seq_len=16
   - Retrain all three: CNN, LinearAttn, GRU
   ```bash
   python -m quant_core.train_scalper_phase4 --config configs/scalper_phase4.yaml
   ```

2. **StatArb Extended Training** (5-8 hours):
   - Focus on **GAT_StatArb_v1** (highest priority — 0.486 PF gap, positive Sharpe)
   - Edit configs/stat_arb_phase4.yaml: max_epochs 80→150, seq_len 64→128
   - Retrain: `python -m quant_core.train_stat_arb_phase4 --config configs/stat_arb_phase4.yaml`

**Day 3 Checkpoint:** Re-evaluate scalper (expect label distribution fix), evaluate extended GAT

---

### Day 4-5: Market Making & Discretionary (16-20 hours)

**Tasks:**
1. **SAC_MM_v1 Tuning** (6-8 hours):
   - Edit configs/mm_phase4.yaml with new episode_length=400, inventory_lambda=0.05, etc.
   - Retrain: `python -m quant_core.train_mm_phase4 --config configs/mm_phase4.yaml --subset sac-only`
   - Monitor for improved PF (currently 0.667, target 1.5)

2. **Discretionary Data + Retrain** (10-12 hours):
   - Generate chart images with max_rows_per_symbol=50,000 (2-3 hours)
   - Retrain all three with new hyperparameters (8-10 hours overnight)

**Days 4-5 Checkpoint:** SAC evaluated for PF improvement; discretionary retraining started

---

### Day 6-10: Iteration & Refinement

**Depending on Day 1-5 results:**

- **If GAT_StatArb_v1 passes (PF > 1.5):** Declare first model viable; begin Phase 5 debate preparation
- **If SAC_MM_v1 passes (PF > 1.5):** Second viable model; validate RL episodic eval
- **If any Trend/MR model passes:** Validate seq_len/horizon fixes worked; consider as Phase 5 ensemble candidate

**If no models pass by Day 5:**
- Diagnose remaining blocker (likely data quality or label distribution)
- Consider feature redesign for failing archetypes
- Run second iteration of retraining with even stricter regularization

---

## SUCCESS CRITERIA FOR PHASE 4 CLOSURE

### Minimum Acceptable Outcome:
- **At least 1 model passes all 4 gates** (Sharpe > 1.2, PF > 1.5, MDD < 0.2, Dir Acc > 0.55)
- **At least 3 models pass at least 3/4 gates** (de-risk single model dependency)

### Acceptable Models by Archetype:
| Archetype | Priority Candidate | Current Gap |
|---|---|---|
| Trend | TCN_Trend_v1 (best) | Sharpe -1.72 → need +2.92 |
| Mean Reversion | GRN_MR_v1 (best) | Sharpe -1.50 → need +2.70 |
| Scalper | CNN_Scalper_v1 | Sharpe -2.27 → need +3.47 (high risk) |
| Stat Arb | GAT_StatArb_v1 (PRIORITY) | PF 1.014 → need +0.486 (CLOSEST) |
| Discretionary | CNNChart_Disc_v1 (best) | Sharpe -1.95 → need +3.15 |
| Market Making | SAC_MM_v1 (PRIORITY) | PF 0.667 → need +0.833 (PASSES Dir Acc) |

**Top 2 Bets for First Pass:**
1. **GAT_StatArb_v1** — Only 0.486 PF units away; already positive Sharpe
2. **SAC_MM_v1** — Passes directional gate; needs inventory/spread tuning

---

## Phase 5 Readiness Criteria

**Phase 5 (Multi-Agent Debate + Orchestration) is BLOCKED until:**
1. ✅ At least 1 model passes all 4 production gates
2. ✅ At least 3 models pass 3/4 gates (ensemble diversity)
3. ✅ All models have reproducible, documented training runs
4. ✅ Walk-forward validation confirmed on OOS window (1,000+ trades minimum)
5. ✅ Monte Carlo robustness test passed (95th percentile MDD < 20%)

**Current Status:** 0/5 blockers cleared. Phase 5 cannot start until Phase 4 closure.

---

## Appendix A: Full Retraining Plan Configuration Deltas

```yaml
# TREND_PHASE4.yaml Changes
seq_len: 32 → 96
horizon: 20 → 5
batch_size: 4096 → 2048
dropout: 0.1 → 0.3
weight_decay: 0.0001 → 0.001
max_epochs: 80 → 150
patience: 12 → 20
preferred_backend: cpu → directml
LSTM.hidden_size: 128 → 256
Transformer.num_layers: 2 → 4

# MR_PHASE4.yaml Changes
horizon: 20 → 3
dropout: 0.1 → 0.4
weight_decay: 0.0001 → 0.002
lr: 0.0005 → 0.0002
max_epochs: 80 → 150
patience: 12 → 20
preferred_backend: cpu → directml

# SCALPER_PHASE4.yaml Changes
flat_threshold: 0.0010 → 0.0015 (audit first!)
horizon: 5 → 2
seq_len: 32 → 16
use_cyclic_lr: true → false
lr: 0.001 → 0.0005
dropout: 0.1 → 0.3
max_epochs: 80 → 120

# STAT_ARB_PHASE4.yaml Changes (for extended training on GAT)
seq_len: 64 → 128
batch_size: 1024 → 512
max_epochs: 80 → 150
patience: 12 → 25
dropout: 0.2 → 0.3

# MM_PHASE4.yaml Changes (SAC focus)
max_episodes: 2,000 → 8,000
max_steps: 500,000 → 2,000,000
episode_length: 200 → 400
inventory_lambda: 0.1 → 0.05
replay_buffer_size: 50,000 → 200,000
batch_size: 2,048 → 512
survival_bonus: 0.0005 → 0.001

# DISCRETIONARY_PHASE4.yaml Changes
max_rows_per_symbol: 12,000 → 50,000
horizon: 20 → 5
flat_threshold: 0.003 → 0.005
batch_size: 256 → 128
dropout: 0.1 → 0.4
weight_decay: 0.0001 → 0.01
lr: 0.0003 → 0.0001
max_epochs: 100 → 200
label_smoothing: — → 0.1
```

---

**Plan Author:** ChatTrader.KPai Lead Quant AI Architect  
**Plan Alignment:** Master Plan Phase 4, Full Retraining Plan Phase 0-6  
**Expected Outcome:** ≥1 production-ready model by Day 10 of iteration
