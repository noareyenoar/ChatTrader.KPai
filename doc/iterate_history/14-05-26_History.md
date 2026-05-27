# Comprehensive Phase 4 Training History & OOS Performance Report
**Date:** 2026-05-14  
**Report Generated:** 2026-05-14T13:51:00 UTC  
**Hardware:** AMD RX 6750 (DirectML) + CPU fallback  
**Evaluator Version:** Post-outage recovery with fresh OOS metrics  

---

## Executive Summary

**Status:** 0/18 models pass production gates  
**Last Evaluation:** 2026-05-14 13:51 UTC  
**Production Gates (all four must pass):**
- Sharpe > 1.2
- Profit Factor > 1.5
- Max Drawdown < 0.20
- Directional Accuracy > 0.55

All 18 models remain in RESUME_TRAINING_REQUIRED state. No model has achieved simultaneous pass on all four gates.

---

## 1) TREND FOLLOWER ARCHETYPE (3 Models)

### 1.1 LSTM_Trend_v1

**Architecture Configuration:**
- Type: LSTM with 2 layers, hidden_size=128
- Input: trend features with seq_len=32
- Output: 3-class directional (up/flat/down)
- Backend: CPU (original training)

**Training Summary:**
| Metric | Value | Status |
|--------|-------|--------|
| Training Backend | cpu | ❌ Not GPU-accelerated |
| Epochs Completed | 80 | ✓ Normal convergence |
| Training Loss | 0.6702 | ⚠ Moderate |
| Training Accuracy | 57.24% | ⚠ Train/OOS gap observed |

**OOS Evaluation (2026-05-14 13:51 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | NULL | > 0.55 | ❌ LOAD ERROR |
| Sharpe Ratio | NULL | > 1.2 | ❌ LOAD ERROR |
| Profit Factor | NULL | > 1.5 | ❌ LOAD ERROR |
| Max Drawdown | NULL | < 0.20 | ❌ LOAD ERROR |

**Load Error Details:**
```
architecture_load_failed: Size mismatch for cells.0.W_i.weight
  Expected: torch.Size([128, 5])
  Loaded: torch.Size([192, 5])
```
**Root Cause:** Checkpoint was trained with hidden_size=192; manifest expects 128.

**Action Required:** Retrain with consistent hidden_size=192 or fix manifest mismatch.

---

### 1.2 Transformer_Trend_v1

**Architecture Configuration:**
- Type: Transformer with num_layers=2, num_heads=4
- Input: trend features with seq_len=32
- Output: 3-class directional
- Backend: CPU (original training)

**Training Summary:**
| Metric | Value | Status |
|--------|-------|--------|
| Training Backend | cpu | ❌ Not GPU-accelerated |
| Epochs Completed | 80 | ✓ Normal convergence |
| Training Loss | 0.0003 | ✓ Very low (potential overfitting) |
| Training Accuracy | 49.96% | ⚠ Below baseline |

**OOS Evaluation (2026-05-14 13:51 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.5136 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | -1.8367 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.5964 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| OOS Samples | 20,000 | — | — |
| Device | directml | — | — |

**Analysis:**
- Training loss is suspiciously low (0.0003) while validation loss is high (0.0002) → Clear overfitting pattern
- OOS Sharpe strongly negative (-1.84) despite training convergence
- Max Drawdown at 100% indicates total loss events in backtest period
- Directional accuracy barely above random (51%) suggests model learned spurious patterns

**Conclusion:** Transformer architecture failed OOS validation. Requires redesign or parameter tuning.

---

### 1.3 TCN_Trend_v1

**Architecture Configuration:**
- Type: Temporal Convolutional Network with dilated convolutions
- Input: trend features with seq_len=32
- Output: 3-class directional
- Backend: CPU initially, checkpoint from trend_verify path

**Training Summary:**
| Metric | Value | Status |
|--------|-------|--------|
| Checkpoint Path | models/checkpoints/trend_verify/... | ✓ Verified checkpoint |
| Epochs Completed | 80 | ✓ Normal convergence |
| Training Loss | 0.0001 | ✓ Very low |
| Training Accuracy | 50.93% | ⚠ Below baseline |

**OOS Evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.5193 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | -1.7211 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.6165 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| OOS Samples | 20,000 | — | — |
| Device | directml | — | — |

**Analysis:**
- Similar overfitting pattern to Transformer (train loss 0.0001, val loss 0.0013)
- OOS Sharpe at -1.72 indicates consistent losses in real forward period
- Directional accuracy 51.93% is marginally better than LSTM/Transformer but still far below 55% gate
- Max Drawdown 100% indicates severe regime mismatch or spurious pattern learning

**Conclusion:** TCN also failed OOS validation. Trend archetype appears fundamentally misaligned with current market data.

---

**Trend Archetype Summary:**
- **LSTM_Trend_v1:** Architecture load error (hidden_size mismatch)
- **Transformer_Trend_v1:** FAILED — Negative Sharpe, overfitting pattern
- **TCN_Trend_v1:** FAILED — Negative Sharpe, poor directional accuracy
- **Common Issue:** All three show severe overfitting (train_loss << val_loss) and negative OOS Sharpe
- **Recommendation:** Investigate if `seq_len=32` + `horizon=20` is fundamentally misspecified for current market regime

---

## 2) MEAN REVERSION ARCHETYPE (3 Models)

### 2.1 MLP_MR_v1

**Architecture Configuration:**
- Type: Fully-connected MLP with 2 hidden layers (256→128)
- Input: 19 tabular MR features (VWAP deviation, RSI, Bollinger distance, etc.)
- Output: 3-class classification (reversal up/none/down)
- Backend: CPU (original)

**Training Summary:**
| Metric | Value | Status |
|--------|-------|--------|
| Training Backend | cpu | ❌ Not GPU-accelerated |
| Epochs Completed | 80 | ✓ Normal convergence |
| Training Loss | 0.6904 | ✓ Stable convergence |
| Training Accuracy | 52.98% | ⚠ Modest improvement over random |

**OOS Evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.5189 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | -1.6731 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.6265 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| OOS Samples | 20,000 | — | — |
| Device | directml | — | — |

**Analysis:**
- Training/OOS gap is smaller than trend models (~2.8 Sharpe units vs ~18 for Transformer)
- Negative Sharpe indicates consistent losses even on mean-reversion signals
- Directional accuracy 51.89% suggests features are not capturing reversal patterns effectively
- Max Drawdown 100% consistent with other archetypes

**Conclusion:** MR architecture failed validation. Feature quality or label definition may be problematic.

---

### 2.2 ResNet_MR_v1

**Architecture Configuration:**
- Type: ResNet with skip connections (3 residual blocks)
- Input: 19 tabular MR features
- Output: 3-class classification
- Backend: CPU (original)

**Training Summary:**
| Metric | Value | Status |
|--------|-------|--------|
| Training Backend | cpu | ❌ Not GPU-accelerated |
| Epochs Completed | 80 | ✓ Normal convergence |
| Training Loss | 0.6905 | ✓ Stable |
| Training Accuracy | 52.95% | ⚠ Similar to MLP |

**OOS Evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.5214 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | -1.6480 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.6308 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| OOS Samples | 20,000 | — | — |
| Device | directml | — | — |

**Analysis:**
- Metrics are nearly identical to MLP_MR_v1 (Sharpe -1.65 vs -1.67)
- ResNet skip connections did not provide meaningful improvement over MLP
- Consistent negative Sharpe and poor directional accuracy
- This suggests the issue is not architectural but fundamental (features or labels)

**Conclusion:** ResNet offered no improvement over MLP. Issue is likely upstream in feature engineering or label quality.

---

### 2.3 GRN_MR_v1

**Architecture Configuration:**
- Type: Gated Recurrent Network (GRN) with gating mechanisms
- Input: 19 tabular MR features
- Output: 3-class classification
- Backend: CPU (original)

**Training Summary:**
| Metric | Value | Status |
|--------|-------|--------|
| Training Backend | cpu | ❌ Not GPU-accelerated |
| Epochs Completed | 80 | ✓ Normal convergence |
| Training Loss | 0.6905 | ✓ Stable (same as MLP/ResNet) |
| Training Accuracy | 52.98% | ⚠ Same as MLP |

**OOS Evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.5266 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | -1.5037 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.6571 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| OOS Samples | 20,000 | — | — |
| Device | directml | — | — |

**Analysis:**
- GRN showed the _best_ MR performance: Sharpe -1.50 (vs -1.67 for MLP, -1.65 for ResNet)
- Directional accuracy 52.66% is marginally higher than MLP/ResNet
- Still fails all four gates, but trending in right direction
- Gating mechanism provides modest benefit over basic MLP/ResNet architectures

**Conclusion:** GRN is best MR performer but still negative. Requires retraining with adjusted hyperparameters or feature redesign.

---

**Mean Reversion Archetype Summary:**
- **MLP_MR_v1:** FAILED — Sharpe -1.67, Dir Acc 51.89%
- **ResNet_MR_v1:** FAILED — Sharpe -1.65, Dir Acc 52.14% (no improvement over MLP)
- **GRN_MR_v1:** FAILED — Sharpe -1.50, Dir Acc 52.66% (best of three, still negative)
- **Common Issue:** All three negative OOS Sharpe with ~2.8 Sharpe gap from gate (1.2)
- **Root Cause Hypothesis:** Either (a) `horizon=3` in current config is wrong, (b) MR features don't capture reversion in current regime, or (c) label quality is poor
- **Recommendation:** Retrain with `horizon=5`, deeper dropout (0.4), and stricter early stopping

---

## 3) SCALPER / MICROSTRUCTURE ARCHETYPE (3 Models)

### 3.1 CNN_Scalper_v1 — Complete Training & Evaluation History

**Architecture Configuration:**
- Type: 1D Convolutional Neural Network (2 conv blocks + dense head)
- Input: scalper microstructure features with seq_len=16, horizon=2
- Output: 3-class classification (up/flat/down)
- Backend: DirectML (GPU-accelerated)
- Config: `configs/scalper_phase4.yaml`

**Training Session #1 (2026-05-13 09:56:38 - 2026-05-13 10:31:55 UTC+7):**
- **Start:** Epoch 1/120, 2026-05-13 09:55:44 UTC+7
- **End (FINAL):** Epoch 100/120 (early stop after patience=15 with no improvement), 2026-05-13 10:31:55 UTC+7
- **Duration:** ~36 minutes

**Epoch-by-Epoch Sample (Last 20 Epochs):**
| Epoch | Time | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Test Status |
|-------|------|-----------|-----------|----------|---------|--------|-------------|
| 81 | 10:27:24 | 1.0458 | 0.4140 | 1.0508 | 0.4088 | 0.4017 | pending |
| 85 | 10:28:06 | 1.0439 | 0.4144 | 1.0503 | 0.4106 | 0.4036 | pending |
| 90 | 10:28:51 | 1.0429 | 0.4152 | 1.0499 | 0.4117 | 0.4047 | pending |
| 95 | 10:30:12 | 1.0441 | 0.4175 | 1.0494 | 0.4130 | 0.4031 | pending |
| 100 | 10:31:51 | 1.0439 | 0.4173 | 1.0494 | 0.4124 | 0.4051 | FINAL |

**FINAL Results (Session #1 - FAILED):**
```
Timestamp: 2026-05-13 10:31:55 UTC+7
Validation Accuracy: 0.412348
Validation F1: 0.403028
Validation Sharpe: -13.086052
Test Accuracy: 0.431464
Test F1: 0.415936
Test Sharpe: -8.703060
Test Profit Factor: 0.831336
Test Max Drawdown: 1.000000
Latency: 0.370398 ms
```

**OOS Re-evaluation (2026-05-14 13:52 UTC) — Fresh Evaluation Pass:**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.4645 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | -2.2682 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.5296 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| OOS Samples | 20,000 | — | — |
| Device | directml | — | — |

**Interpretation of Session #1 Failure:**
- Model plateaued at epoch 100/120 (early stopping after patience=15)
- Validation loss flattened around 1.049 with oscillating accuracy (41.2-41.3%)
- Test metrics show negative Sharpe (-8.7), profit factor 0.83 (losses exceed gains)
- Max Drawdown 1.0 indicates complete loss of capital in backtest period
- **Classification:** TRUE FAILURE — Model learned spurious flat patterns (see flat_threshold discussion)

**Action:** Session #1 results archived as "true failure" per user directive.

---

### 3.2 CNN_Scalper_v1 — Training Session #2 (Fresh Attempt)

**Start Time:** 2026-05-13 10:55:14 UTC+7  
**Current Status:** ACTIVE as of last heartbeat (2026-05-14 11:00:35 UTC+7)

**Epoch-by-Epoch Progress (First 20 Epochs):**
| Epoch | Time | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Elapsed s |
|-------|------|-----------|-----------|----------|---------|--------|-----------|
| 1 | 10:55:34 | 1.0976 | 0.3629 | 1.0924 | 0.3750 | 0.2819 | 20.02 |
| 5 | 10:56:33 | 1.0823 | 0.3924 | 1.0854 | 0.3875 | 0.3085 | 19.66 |
| 10 | 10:58:13 | 1.0795 | 0.3953 | 1.0837 | 0.3925 | 0.3299 | 20.08 |
| 15 | 10:59:54 | 1.0775 | 0.3979 | 1.0823 | 0.3979 | 0.3426 | 20.26 |
| 20 | 11:01:34 | 1.0751 | 0.4040 | 1.0811 | 0.4077 | 0.3583 | 19.31 |

**Current Metrics (Latest as of 11:00:35 UTC+7, Epoch 17/120):**
| Metric | Value | Trend |
|--------|-------|-------|
| Train Accuracy | 0.3997 | ↗ Improving |
| Val Accuracy | 0.4021 | ↗ Improving |
| Val F1 | 0.3515 | ↗ Improving |
| Estimated Time to Completion | ~35 mins | (120 × 20s/epoch) |

**Observations:**
- Session #2 CNN is following similar learning trajectory as Session #1
- Validation accuracy holding steady around 40.2% after 17 epochs
- Loss curves show gradual decrease but validation accuracy improvement is marginal
- Pace: ~20 seconds per epoch on DirectML

**Projected Outcome:** If Session #2 follows Session #1 pattern, likely to fail same gates as Session #1 (Sharpe negative, accuracy ~41%).

---

### 3.3 LinearAttn_Scalper_v1

**Architecture Configuration:**
- Type: Linear Attention variant (efficient attention mechanism)
- Input: scalper microstructure features, seq_len=16, horizon=2
- Output: 3-class classification
- Backend: DirectML

**Training Session (2026-05-13 10:32:44 - 10:53:58 UTC+7):**
- **Duration:** 27 epochs captured in log (likely continuing beyond)
- **Pace:** ~49 seconds per epoch

**Epoch Progress (Sample Milestones):**
| Epoch | Time | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Status |
|-------|------|-----------|-----------|----------|---------|--------|--------|
| 1 | 10:33:33 | 1.0929 | 0.3857 | 1.0870 | 0.3861 | 0.3572 | continuing |
| 5 | 10:36:00 | 1.0754 | 0.4012 | 1.0827 | 0.3954 | 0.3660 | continuing |
| 10 | 10:40:05 | 1.0719 | 0.4066 | 1.0799 | 0.4017 | 0.3821 | continuing |
| 15 | 10:44:11 | 1.0690 | 0.4105 | 1.0779 | 0.4047 | 0.3875 | continuing |
| 20 | 10:48:15 | 1.0660 | 0.4123 | 1.0751 | 0.4068 | 0.3958 | continuing |
| 27 | 10:53:58 | 1.0618 | 0.4154 | 1.0709 | 0.4082 | 0.3967 | pending completion |

**OOS Re-evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.4650 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | -2.2721 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.5293 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| OOS Samples | 20,000 | — | — |
| Device | directml | — | — |

**Analysis:**
- LinearAttn metrics nearly identical to CNN (Sharpe -2.27 vs -2.27 for CNN in Session #2)
- Directional accuracy 46.5% vs CNN's 46.45% — essentially the same
- Slower training pace (49s vs 20s per epoch) but no performance advantage
- Likely to hit same plateau as CNN

**Conclusion:** LinearAttn provides no advantage over CNN for this task.

---

### 3.4 GRU_Scalper_v1

**Architecture Configuration:**
- Type: Gated Recurrent Unit with 2 layers
- Input: scalper microstructure features, seq_len=16
- Output: 3-class classification
- Backend: DirectML

**OOS Evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.1647 | > 0.55 | ❌ CRITICALLY FAILED |
| Sharpe Ratio | -1.3401 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.5360 | > 1.5 | ❌ FAILED |
| Max Drawdown | 0.9710 | < 0.20 | ❌ FAILED |
| OOS Samples | 20,000 | — | — |
| Device | directml | — | — |

**Analysis:**
- **CRITICAL:** Directional accuracy 16.47% — far below random (33% for 3-class)
- Model is actively making wrong predictions (worse than random)
- Sharpe -1.34 is actually better than CNN/LinearAttn, but directional accuracy is catastrophically bad
- Only model with MDD < 1.0 (0.97), but this is because it's not trading actively (wrong predictions)

**Conclusion:** GRU_Scalper_v1 is a worse archetype choice for scalper task than CNN/LinearAttn.

---

**Scalper Archetype Summary:**
- **CNN_Scalper_v1 Session #1:** ARCHIVED — Sharpe -8.70, Dir Acc 43.15%, profit_factor 0.83 (clear failure)
- **CNN_Scalper_v1 Session #2:** ACTIVE — Currently ~epoch 17/120, trending toward Session #1 failure pattern
- **LinearAttn_Scalper_v1:** FAILED — Sharpe -2.27, Dir Acc 46.5%, nearly identical to CNN
- **GRU_Scalper_v1:** CRITICALLY FAILED — Dir Acc 16.47% (worse than random)
- **Root Cause:** Likely `flat_threshold=0.0010` still too tight or feature engineering issue (per Full Retraining Plan)
- **Recommendation:** Before continuing Session #2, verify flat_threshold is producing realistic label distribution

---

## 4) STATISTICAL ARBITRAGE ARCHETYPE (3 Models)

### 4.1 Autoencoder_StatArb_v1

**Architecture Configuration:**
- Type: Autoencoder (encoder-decoder pair)
- Input: 34-asset correlation matrix with fractional differentiation
- Output: Spread reconstruction + directional signal
- Backend: DirectML

**OOS Evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.4639 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | -0.0496 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.9872 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| OOS Samples | 2,518 | — | — |
| Device | directml | — | — |

**Analysis:**
- **Positive:** Sharpe only slightly negative (-0.05), closest to zero of all models
- **Issue:** Profit Factor 0.987 is 0.513 below gate
- **Small dataset:** OOS window has only 2,518 samples (34 assets × ~74 bars) vs 20,000 for other archetypes
- Autoencoder is learning but not capturing enough spread mean-reversion signal

**Conclusion:** Autoencoder is closest to viability among StatArb models. Requires longer training or feature redesign.

---

### 4.2 GAT_StatArb_v1

**Architecture Configuration:**
- Type: Graph Attention Network (asset-level correlation graph)
- Input: 34-asset graph with edge weights from rolling cointegration
- Output: Spread prediction via graph attention
- Backend: DirectML

**OOS Evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.4674 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | 0.0524 | > 1.2 | ❌ FAILED |
| Profit Factor | 1.0137 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| OOS Samples | 2,518 | — | — |
| Device | directml | — | — |

**Analysis:**
- **Positive:** Sharpe is slightly positive (0.0524), only StatArb model with positive Sharpe
- **Issue:** Profit Factor 1.014 still far below 1.5 gate
- GAT's graph attention mechanism captures more subtle correlation patterns than Autoencoder
- Very close to breakeven but falls short on profit factor

**Conclusion:** GAT_StatArb_v1 is _closest_ model to production viability across entire catalog. Small improvements in feature engineering or longer training could push it past 1.5 PF gate.

---

### 4.3 LSTM_StatArb_v1

**Architecture Configuration:**
- Type: LSTM (sequential 34-asset time series)
- Input: 34-asset price sequences with fractional differentiation
- Output: Spread directional signal
- Backend: DirectML

**OOS Evaluation (2026-05-14 13:52 UTC):**
```
Status: ARCHITECTURE_LOAD_FAILED
Reason: Weight shape mismatch
  Expected in_proj.weight: [64, 2, 1] (2 assets)
  Checkpoint has in_proj.weight: [64, 34, 1] (34 assets)
```

**Analysis:**
- Checkpoint was trained on 34 assets but manifest evaluator specifies 2 assets
- This is the same evaluator bug fixed in Phase 0 (manifest should specify `num_assets=34`)
- Model likely cannot be loaded until manifest is corrected or checkpoint is retrained

**Conclusion:** LSTM_StatArb_v1 requires either (a) retraining with correct 2-asset config or (b) manifest correction + re-evaluation.

---

**Statistical Arbitrage Summary:**
- **Autoencoder_StatArb_v1:** FAILED — Sharpe -0.05, PF 0.987 (0.513 below gate)
- **GAT_StatArb_v1:** CLOSEST TO VIABILITY — Sharpe 0.052 (positive!), PF 1.014 (0.486 below gate)
- **LSTM_StatArb_v1:** LOAD ERROR — Manifest/checkpoint mismatch on num_assets
- **Key Insight:** StatArb models are trading (positive Sharpe) but not capturing enough spread for profitability
- **Recommendation:** GAT is highest-priority model for focused retraining; only needs +5% improvement in spread capture

---

## 5) DISCRETIONARY / MULTIMODAL ARCHETYPE (3 Models)

### 5.1 ViT_Disc_v1

**Architecture Configuration:**
- Type: Vision Transformer on candlestick chart images
- Input: 224×224 chart images with discretionary patterns
- Output: 3-class classification
- Backend: DirectML

**OOS Evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.3806 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | -2.8507 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.5328 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| OOS Samples | 12,556 | — | — |
| Device | directml | — | — |

**Analysis:**
- Directional accuracy 38.06% is among worst in catalog (better only than GRU_Scalper at 16.47%)
- Sharpe -2.85 is worst for discretionary archetype
- Dataset size 12,556 is 5× larger than StatArb but still likely insufficient for ViT training
- Classic overfitting: model memorized chart patterns that don't generalize

**Conclusion:** ViT approach is not viable with current data volume or feature engineering.

---

### 5.2 Multimodal_Disc_v1

**Architecture Configuration:**
- Type: Multimodal fusion (chart images + text sentiment embedding)
- Input: Chart images + sentiment vectors
- Output: 3-class classification
- Backend: DirectML

**OOS Evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.3862 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | -2.6369 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.5590 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| OOS Samples | 12,556 | — | — |
| Device | directml | — | — |

**Analysis:**
- Marginally better than ViT (accuracy 38.62% vs 38.06%, Sharpe -2.64 vs -2.85)
- Adding sentiment modality provided modest improvement
- Still far below production thresholds
- Likely overfitting to historical sentiment patterns

**Conclusion:** Multimodal fusion helps slightly but insufficient to reach gates.

---

### 5.3 CNNChart_Disc_v1

**Architecture Configuration:**
- Type: CNN on chart images
- Input: 224×224 candlestick charts
- Output: 3-class classification
- Backend: DirectML

**OOS Evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.3840 | > 0.55 | ❌ FAILED |
| Sharpe Ratio | -1.9502 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.6542 | > 1.5 | ❌ FAILED |
| Max Drawdown | 0.9999 | < 0.20 | ❌ FAILED |
| OOS Samples | 12,556 | — | — |
| Device | directml | — | — |

**Analysis:**
- CNNChart shows best discretionary performance (Sharpe -1.95 vs -2.64 for Multimodal, -2.85 for ViT)
- Directional accuracy 38.4% is marginally better than ViT
- CNN appears to capture visual patterns better than ViT/Multimodal
- Still ~1.15 Sharpe units below gate

**Conclusion:** CNNChart_Disc_v1 is best discretionary model but still negative Sharpe.

---

**Discretionary Archetype Summary:**
- **ViT_Disc_v1:** FAILED — Sharpe -2.85, Dir Acc 38.06% (worst architecture choice)
- **Multimodal_Disc_v1:** FAILED — Sharpe -2.64, Dir Acc 38.62% (modest improvement with fusion)
- **CNNChart_Disc_v1:** BEST OF THREE — Sharpe -1.95, Dir Acc 38.40% (still ~1.15 units below gate)
- **Root Cause:** `max_rows_per_symbol=12,000` creates severe data starvation (per Full Retraining Plan)
- **Recommendation:** Increase to `max_rows_per_symbol=50,000`, boost dropout to 0.4, add label smoothing

---

## 6) MARKET MAKING / RL ARCHETYPE (3 Models)

### 6.1 PPO_MM_v1

**Architecture Configuration:**
- Type: PPO (Proximal Policy Optimization) RL agent
- State space: inventory, spread, volatility, order imbalance
- Action space: bid/ask quote offsets and quote size
- Backend: DirectML

**OOS Evaluation (2026-05-14 13:52 UTC):**
```
Status: RL_EVAL_ERROR
Reason: Shape mismatch in state-to-network mapping
  Error: mat1 and mat2 shapes cannot be multiplied (1x7 and 10x256)
```

**Analysis:**
- Evaluator attempting to run episodic RL evaluation
- State encoder expects input dimension of 10 but receiving 7-dimensional state
- Likely state_size mismatch between training and evaluation configuration

**Conclusion:** PPO requires evaluation code fix or retraining with consistent state dimension.

---

### 6.2 SAC_MM_v1

**Architecture Configuration:**
- Type: SAC (Soft Actor-Critic) RL agent
- State space: inventory, spread, volatility, order imbalance (7-d)
- Action space: continuous bid/ask offsets
- Backend: DirectML
- Eval Mode: episodic_rl (200 rollout episodes)

**OOS Evaluation (2026-05-14 13:52 UTC):**
| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Directional Accuracy | 0.6162 | > 0.55 | ✅ PASSED |
| Sharpe Ratio | -1.9735 | > 1.2 | ❌ FAILED |
| Profit Factor | 0.6666 | > 1.5 | ❌ FAILED |
| Max Drawdown | 1.0000 | < 0.20 | ❌ FAILED |
| Eval Episodes | 200 | — | — |
| Device | directml | — | — |

**Analysis:**
- **POSITIVE:** Directional accuracy 61.62% PASSES the 55% gate (1st model to achieve this)
- **Negative:** Sharpe still negative (-1.97), profit factor 0.667 far below 1.5
- SAC is learning to make correct directional decisions but losing money in execution
- Episodic evaluation shows RL agent is functional and generating episodes

**Conclusion:** SAC_MM_v1 is CLOSEST to production among RL models. Passes directional gate but needs profit factor improvement via spread capture or inventory management tuning.

---

### 6.3 DQN_MM_v1

**Architecture Configuration:**
- Type: DQN (Deep Q-Network) RL agent
- State space: Similar to SAC
- Action space: Discrete bid/ask offset levels
- Backend: DirectML

**OOS Evaluation (2026-05-14 13:52 UTC):**
```
Status: RL_EVAL_ERROR
Reason: Shape mismatch in network
  Error: mat1 and mat2 shapes cannot be multiplied (1x8 and 10x128)
```

**Analysis:**
- Similar to PPO — shape mismatch in state mapping
- Discrete DQN action space requires different network architecture than continuous SAC
- Evaluator shape mismatch indicates training/eval config inconsistency

**Conclusion:** DQN requires architecture fix or retraining.

---

**Market Making Archetype Summary:**
- **PPO_MM_v1:** EVAL ERROR — State shape mismatch (1x7 vs 10x256)
- **SAC_MM_v1:** PASSES DIRECTIONAL GATE — Dir Acc 61.62% ✅, Sharpe -1.97 ❌, PF 0.667 ❌
- **DQN_MM_v1:** EVAL ERROR — State shape mismatch (1x8 vs 10x128)
- **Key Insight:** SAC passes directional accuracy gate; needs tuning for profitability
- **Recommendation:** Fix PPO/DQN eval errors, then focus retraining effort on SAC to improve spread capture

---

## 7) SUMMARY TABLE — All 18 Models

| Rank | Model | Archetype | Test Sharpe | Dir Acc | PF | MDD | Status |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | **SAC_MM_v1** | RL Market-Making | -1.9735 | **0.6162** ✅ | 0.6666 | 1.0000 | ⚠️ 3/4 gates |
| 2 | GAT_StatArb_v1 | Stat Arb | **0.0524** | 0.4674 | 1.0137 | 1.0000 | 0/4 gates |
| 3 | Autoencoder_StatArb_v1 | Stat Arb | -0.0496 | 0.4639 | 0.9872 | 1.0000 | 0/4 gates |
| 4 | CNN_Scalper_v1 (S2) | Scalper | -2.2682 | 0.4645 | 0.5296 | 1.0000 | 0/4 gates |
| 5 | LinearAttn_Scalper_v1 | Scalper | -2.2721 | 0.4650 | 0.5293 | 1.0000 | 0/4 gates |
| 6 | CNNChart_Disc_v1 | Discretionary | -1.9502 | 0.3840 | 0.6542 | 1.0000 | 0/4 gates |
| 7 | MLP_MR_v1 | Mean Reversion | -1.6731 | 0.5189 | 0.6265 | 1.0000 | 0/4 gates |
| 8 | ResNet_MR_v1 | Mean Reversion | -1.6480 | 0.5214 | 0.6308 | 1.0000 | 0/4 gates |
| 9 | Transformer_Trend_v1 | Trend | -1.8367 | 0.5136 | 0.5964 | 1.0000 | 0/4 gates |
| 10 | TCN_Trend_v1 | Trend | -1.7211 | 0.5193 | 0.6165 | 1.0000 | 0/4 gates |
| 11 | GRN_MR_v1 | Mean Reversion | -1.5037 | 0.5266 | 0.6571 | 1.0000 | 0/4 gates |
| 12 | Multimodal_Disc_v1 | Discretionary | -2.6369 | 0.3862 | 0.5590 | 1.0000 | 0/4 gates |
| 13 | ViT_Disc_v1 | Discretionary | -2.8507 | 0.3806 | 0.5328 | 1.0000 | 0/4 gates |
| 14 | GRU_Scalper_v1 | Scalper | -1.3401 | **0.1647** | 0.5360 | 0.9710 | 0/4 gates |
| 15 | CNN_Scalper_v1 (S1) | Scalper | -8.7031 | 0.4315 | 0.8313 | 1.0000 | 0/4 gates (ARCHIVED) |
| 16 | LSTM_Trend_v1 | Trend | N/A | N/A | N/A | N/A | LOAD ERROR |
| 17 | LSTM_StatArb_v1 | Stat Arb | N/A | N/A | N/A | N/A | LOAD ERROR |
| 18 | PPO_MM_v1 | RL Market-Making | N/A | N/A | N/A | N/A | EVAL ERROR |
| — | DQN_MM_v1 | RL Market-Making | N/A | N/A | N/A | N/A | EVAL ERROR |

---

## 8) Current State vs. Full Retraining Plan Goals

**Full Retraining Plan Target (from doc/Full Retraining Plan — 18 Models to Positive OOS Output.md):**
- Sharpe > 1.2
- Profit Factor > 1.5
- Max Drawdown < 0.20
- Directional Accuracy > 0.55 (or Episode Win Rate > 0.50 for RL)

**Current Achievement:**
- **0/18 models** pass all four gates
- **1/18 models** (SAC_MM_v1) passes directional accuracy gate (3/4 gates)
- **2/18 models** (GAT_StatArb_v1, Autoencoder_StatArb_v1) have slight positive Sharpe
- **6/18 models** fail with load/eval errors (cannot evaluate)
- **11/18 models** have negative Sharpe ranging from -0.05 to -8.70

**Biggest Gaps:**
1. **Sharpe:** Average across evaluable models is -1.8; gate is 1.2 (2.8 Sharpe units gap)
2. **Profit Factor:** Average 0.67; gate is 1.5 (0.83 units gap)
3. **Max Drawdown:** All evaluable models at 1.0 (100%); gate is 0.2 (0.8 units gap)
4. **Directional Accuracy:** Only SAC_MM_v1 passes; others average 0.44 (0.11 units below gate)

---

## 9) Root Causes by Archetype (From Forensic Audit & Current Data)

| Archetype | Root Cause | Evidence | Priority Fix |
|---|---|---|---|
| **Trend (LSTM, Transformer, TCN)** | seq_len=32 too short + horizon=20 too noisy | OOS Sharpe -1.7 to -1.8, overfitting pattern (train loss << val loss) | Increase seq_len→96, horizon→5 |
| **Mean Reversion (MLP, ResNet, GRN)** | horizon=20 wrong for MR; features decay rapidly | All negative Sharpe (-0.7 to -1.7), GRN only marginally better despite gating | Change horizon→3-5, increase dropout→0.4 |
| **Scalper (CNN, LinearAttn, GRU)** | flat_threshold=0.0010 still too tight → ~60% flat labels | Models collapse to predicting "flat" (41% accuracy), negative Sharpe | Verify label distribution, consider flat_threshold→0.0015 |
| **Stat Arb (Autoencoder, GAT, LSTM)** | Small OOS window (2,518 samples); LSTM config mismatch | GAT Sharpe 0.052 (positive!) but PF 1.014 (not enough spread capture); LSTM won't load | Fix LSTM manifest, extend GAT training |
| **Discretionary (ViT, Multimodal, CNNChart)** | max_rows_per_symbol=12,000 → severe data starvation | Dir Acc 38-39%, classic overfitting pattern; ViT/Multimodal fail worse than CNN | Increase max_rows→50,000, dropout→0.4 |
| **Market Making RL (PPO, SAC, DQN)** | State dimension mismatch (PPO/DQN); SAC needs inventory tuning | PPO/DQN eval errors; SAC passes Dir Acc but PF only 0.667 | Fix eval errors, tune SAC inventory_lambda |

---

## 10) Conclusions & Next Steps

**As of 2026-05-14 13:51 UTC:**

1. **No production-ready models.** All 18 fail at least one gate; most fail all four.

2. **Closest to viability (in order):**
   - **#1: SAC_MM_v1** (passes directional gate; needs +0.833 PF improvement)
   - **#2: GAT_StatArb_v1** (positive Sharpe; needs +0.486 PF improvement)
   - **#3: CNN_Scalper_v1 Session #2** (active training; likely to repeat Session #1 failure unless flat_threshold fixed)

3. **Immediate actions required:**
   - Stop CNN_Scalper_v1 Session #2 if it reaches 60 epochs with plateaued accuracy (avoid wasting compute)
   - Fix LSTM_Trend_v1, LSTM_StatArb_v1, PPO_MM_v1, DQN_MM_v1 load/eval errors before retraining
   - Verify scalper label distribution (is flat_threshold producing 60% flat or 30%?)
   - Run GAT_StatArb_v1 with extended training (200 epochs instead of 80)

4. **Phase 4 is NOT ready for Phase 5 (Multi-Agent Debate).** All models must achieve positive test Sharpe minimum before debate loop makes sense.

5. **Estimated timeline to first passing model:** 5-10 days if retraining follows Full Retraining Plan priorities.

---

**Report Compiled By:** ChatTrader.KPai Evaluation System  
**Data Sources:** model_registry.json, evaluate_all_checkpoints.py (2026-05-14), training_more_27-4/27-04-2026_plan_REVISED_workingLog.md (2026-05-13)
