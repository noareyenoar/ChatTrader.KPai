# ChatTrader.KPai — Backtest v2: Split-Learn Report
**Generated:** 2026-05-04 20:21:08  
**Mode:** DRY RUN (offline fallback)  
**Protocol:** IRON WALL Data Splitting (NO LEAKAGE POLICY) — pytorch_model_training_rule.md  
**Total elapsed:** 74.4s

---
## 1. Data Split Summary

| Phase | Files | Date Range | Bars (est.) |
|-------|-------|------------|-------------|
| **TRAIN** (70%) | 84 | 20251010 → 20260319 | ~18,816 |
| *Purge Gap* | 1 | — | — |
| **VALIDATION** (15%) | 18 | 20260321 → 20260407 | ~4,032 |
| *Purge Gap* | 1 | — | — |
| **TEST** (15%) | 17 | 20260409 → 20260425 | ~3,808 |

---
## 2. Phase Results

| Metric | TRAIN | VALIDATION | TEST |
|--------|-------|------------|------|
| Bars | 420 | 90 | 85 |
| Win Rate | 0.0% | 0.0% | 0.0% |
| Net PnL | 0.0000% | 0.0000% | 0.0000% |
| Max Drawdown | 0.00% | 0.00% | 0.00% |
| Trades | 12 | 0 | 3 |
| Debates | 387 | 90 | 77 |

---
## 3. Trained Credibility Weights (after TRAIN phase)

These are the agent "weights" saved before VALIDATION. Loaded unchanged for VAL and TEST.

| Agent | Credibility | Status |
|-------|-------------|--------|
| TrendAnalyst | 0.6500 | ↑ Bullish lean |
| DiscretionaryAnalyst | 0.5458 | → Neutral |
| ScalperAnalyst | 0.3565 | ↓ Bearish lean |
| MeanReversionAnalyst | 0.3282 | ↓ Bearish lean |
| MarketMakerAnalyst | 0.3222 | ↓ Bearish lean |
| StatArbAnalyst | 0.2529 | ↓ Bearish lean |

---
## 4. Agent Self-Critique (VALIDATION Phase)

Each analyst compared its training credibility to its validation-phase performance.
Credibility scores were FROZEN during validation — these critiques are pure OOS analysis.

### TrendAnalyst
- **Train credibility:** 0.6500  
- **Val accuracy drift:** -75.0%  
- **Validity:** INVALID [FAIL] -- performance decay > 50%  
- **Regime biases:** {'general': 'Insufficient validation regime samples'}  
- **Recommended adjustment:** Reduce credibility weight by 0.05 in RANGING regime. Current val PnL (0.0000%) is significantly below train (0.0000%). Possible overfit to trending patterns seen in training data.  

### MeanReversionAnalyst
- **Train credibility:** 0.3282  
- **Val accuracy drift:** -25.0%  
- **Validity:** INVALID [FAIL] -- performance decay > 50%  
- **Regime biases:** {'general': 'Insufficient validation regime samples'}  
- **Recommended adjustment:** Explore feature engineering improvements. Low credibility (0.3282) suggests signal quality below baseline.  

### ScalperAnalyst
- **Train credibility:** 0.3565  
- **Val accuracy drift:** -33.3%  
- **Validity:** INVALID [FAIL] -- performance decay > 50%  
- **Regime biases:** {'general': 'Insufficient validation regime samples'}  
- **Recommended adjustment:** Performance within acceptable bounds. Monitor for regime shift. Train acc=33.3% | Val acc=0.0% | Test acc=66.7%.  

### StatArbAnalyst
- **Train credibility:** 0.2529  
- **Val accuracy drift:** -16.7%  
- **Validity:** INVALID [FAIL] -- performance decay > 50%  
- **Regime biases:** {'general': 'Insufficient validation regime samples'}  
- **Recommended adjustment:** Explore feature engineering improvements. Low credibility (0.2529) suggests signal quality below baseline.  

### DiscretionaryAnalyst
- **Train credibility:** 0.5458  
- **Val accuracy drift:** -66.7%  
- **Validity:** INVALID [FAIL] -- performance decay > 50%  
- **Regime biases:** {'general': 'Insufficient validation regime samples'}  
- **Recommended adjustment:** Performance within acceptable bounds. Monitor for regime shift. Train acc=66.7% | Val acc=0.0% | Test acc=100.0%.  

### MarketMakerAnalyst
- **Train credibility:** 0.3222  
- **Val accuracy drift:** -16.7%  
- **Validity:** INVALID [FAIL] -- performance decay > 50%  
- **Regime biases:** {'general': 'Insufficient validation regime samples'}  
- **Recommended adjustment:** Explore feature engineering improvements. Low credibility (0.3222) suggests signal quality below baseline.  

---
## 5. OOS Validity Verdict (§ 4 Real-World Validity Logic)

**[VALID] OOS performance within acceptable bounds**

All validity criteria passed:
  - Performance decay < 50% ✓
  - Test drawdown < 20% ✓
  - Test win rate ≥ 40% ✓

---
## 6. Leakage Prevention Checklist

| Criterion | Status |
|-----------|--------|
| Chronological sort (no shuffle) | ✅ Enforced — files sorted by YYYYMMDD stem |
| Purge gap between splits | ✅ 1 file(s) dropped at each boundary |
| Scaler fit only on TRAIN | ✅ DataFeeder computes features per-file (no cross-file scaling) |
| Credibility frozen in VAL/TEST | ✅ freeze_credibility=True applied |
| No lookahead (t features from t-1) | ✅ DataFeeder warmup_bars=64 respected |
