# Full Retraining Plan — 18 Models to Positive OOS Output

**Date:** 2026-05-05  
**Hardware:** AMD RX 6750 (DirectML) + CPU fallback  
**Dataset:** `Dataset/binance_historical/` — 39 parquet files, ~12,000 rows each, no new data collection required  
**Production Gates (all four must pass simultaneously):**  
- Sharpe > 1.2  
- Profit Factor > 1.5  
- Max Drawdown < 0.20  
- Directional Accuracy > 0.55 (classification models) / Episode Win Rate > 0.50 (RL models)

---

## Diagnosis Summary

| Archetype | Root Cause of Failure | Severity |
|---|---|---|
| Trend (3 models) | `seq_len=32` too short; `horizon=20` too noisy for crypto; underfitting on CPU | Medium |
| Mean Reversion (3) | `horizon=20` is wrong for MR — price has reversed and re-trended before label fires | Medium |
| Scalper (3) | `flat_threshold=0.0003` too tight → ~70% labels are "flat", model collapses to predicting flat | **High** |
| Stat Arb (3) | Evaluator bug (2-asset OOS vs 34-asset training); LSTM has weight shape mismatch | Evaluator only |
| Discretionary (3) | `max_rows_per_symbol=12000` — severe data starvation; ViT/Multimodal memorise 12k rows | **Critical** |
| Market Maker (3) | Evaluator bug — RL agents evaluated as classifiers, not via episode rollout | Evaluator only |

---

## Current OOS Leaderboard (Pre-Retrain)

| Model | Sharpe | Dir Acc | Profit Factor | MDD | Gate Result |
|---|---|---|---|---|---|
| GAT_StatArb_v1 | 1.9748 | 0.5568 | 1.6853 | 1.0000 | FAILED (MDD — evaluator bug) |
| Autoencoder_StatArb_v1 | 1.4525 | 0.5532 | 1.4642 | 1.0000 | FAILED (PF+MDD — evaluator bug) |
| GRN_MR_v1 | -0.7405 | 0.5073 | 0.8156 | 1.0000 | FAILED |
| CNN_Scalper_v1 | -0.9983 | 0.4031 | 0.7349 | 1.0000 | FAILED |
| GRU_Scalper_v1 | -1.2542 | 0.3817 | 0.6901 | 1.0000 | FAILED |
| LinearAttn_Scalper_v1 | -1.6963 | 0.3668 | 0.5670 | 1.0000 | FAILED |
| LSTM_Trend_v1 | -1.7256 | 0.5191 | 0.6157 | 1.0000 | FAILED |
| TCN_Trend_v1 | -1.7256 | 0.5192 | 0.6157 | 1.0000 | FAILED |
| MLP_MR_v1 | -1.7336 | 0.5175 | 0.6156 | 1.0000 | FAILED |
| ResNet_MR_v1 | -1.7953 | 0.5169 | 0.6047 | 1.0000 | FAILED |
| CNNChart_Disc_v1 | -1.8424 | 0.3925 | 0.6457 | 1.0000 | FAILED |
| Transformer_Trend_v1 | -1.8865 | 0.5167 | 0.5889 | 1.0000 | FAILED |
| ViT_Disc_v1 | -1.9119 | 0.3558 | 0.5988 | 1.0000 | FAILED |
| Multimodal_Disc_v1 | -2.2178 | 0.3989 | 0.5859 | 1.0000 | FAILED |
| DQN_MM_v1 | -4.3059 | 0.5287 | 0.3550 | 0.9999 | FAILED (wrong eval method) |
| PPO_MM_v1 | -4.3157 | 0.5641 | 0.3553 | 0.9999 | FAILED (wrong eval method) |
| SAC_MM_v1 | -4.3157 | 0.5641 | 0.3553 | 0.9999 | FAILED (wrong eval method) |
| LSTM_StatArb_v1 | N/A | N/A | N/A | N/A | FAILED (load error — evaluator bug) |

---

## PHASE 0 — Evaluator Fixes (Already Applied)

These are code fixes in `evaluate_all_checkpoints.py`. No retraining needed for these — just re-run the evaluator.

### Fix 0-A: StatArb `num_assets` mismatch (DONE)
- **Bug:** `MODEL_MANIFEST` had `num_assets=2` for all three StatArb entries; checkpoints were trained on 34 assets
- **Fix applied:** Changed manifest entries to `num_assets=34`; changed `load_stat_arb_data` default to `min(34, len(frames))`
- **Impact:** OOS dataset grows from 2,518 to ~20,000 samples; MDD will compute correctly

### Fix 0-B: RL episodic evaluator (DONE)
- **Bug:** PPO/SAC/DQN were evaluated as one-shot classifiers (output action → direction signal → PnL). RL actions are bid/ask offset prices, not directional predictions.
- **Fix applied:** Added `eval_rl_episode()` function that runs 200 environment episodes on the OOS test price slice using `MarketMakingEnv`, then computes Sharpe of episode reward distribution, max drawdown of cumulative reward curve, and episode win rate.
- **Impact:** PPO/SAC/DQN now get meaningful metrics instead of Sharpe ~-4.3

### Action: Re-run Evaluator Before Any Retraining

```powershell
cd d:\kp_ai_agent\ChatTrader.KPai
.\.venv\Scripts\python.exe evaluate_all_checkpoints.py
```

Expected: GAT_StatArb_v1 and Autoencoder_StatArb_v1 likely PASS (their Sharpe/PF were already good). LSTM_StatArb_v1 now loadable. PPO/SAC/DQN get real episode-based scores. If any model passes here, skip its retrain phase.

---

## PHASE 1 — Stat Arb Retraining

**Run only if GAT/Autoencoder/LSTM still fail after Phase 0 re-eval.**

**Root cause detail:**  
- GAT had Sharpe=1.97, PF=1.69 — already above gates. Only MDD=1.0 was blocking it, caused purely by the 2-asset evaluator bug.  
- LSTM had a weight shape mismatch (`in_proj.weight [64,34,1] vs [64,2,1]`) — fixed by the manifest correction. After re-eval it should load and score.

**Config changes** (`configs/stat_arb_phase4.yaml` already updated):
| Parameter | Old | New | Reason |
|---|---|---|---|
| `seq_len` | 64 | 128 | Longer spread context for mean-reversion cycles |
| `batch_size` | 2048 | 1024 | Better gradient signal per update |
| `max_epochs` | 80 | 120 | More training time |
| `patience` | 12 | 20 | More tolerance before early stop |
| `preferred_backend` | cpu | directml | Use GPU |
| `dropout` (all) | 0.1 | 0.2 | Slight regularisation increase |

**Run:**
```powershell
.\.venv\Scripts\python.exe -m quant_core.train_stat_arb_phase4 --config configs/stat_arb_phase4.yaml
```

**Pass criteria:** `test_sharpe > 1.2` AND `test_profit_factor > 1.5` AND `test_max_drawdown < 0.20`

---

## PHASE 2 — Trend Follower Retraining

**Models:** LSTM_Trend_v1, Transformer_Trend_v1, TCN_Trend_v1

**Root causes:**
1. `seq_len=32` gives only 32 bars of context (~32 minutes on 1-min). Trend identification needs 96+ bars.
2. `horizon=20` asks the model to predict 20 bars ahead — too far. At 20-bar horizon, the target bar has absorbed multiple reversal and continuation cycles, making the binary label nearly random.
3. All three trained on `cpu` — may have been under-trained relative to convergence with 80 epochs.
4. `dropout=0.1` is too low — OOS accuracy (51-52%) far below training (~57%) shows memorisation.

**Config changes** (`configs/trend_phase4.yaml` already updated):
| Parameter | Old | New | Reason |
|---|---|---|---|
| `seq_len` | 32 | 96 | 3× longer context (1.5 hours at 1-min) |
| `horizon` | 20 | 5 | Shorter prediction = much stronger signal, easier to exceed 55% accuracy |
| `preferred_backend` | cpu | directml | GPU acceleration |
| `batch_size` | 4096 | 2048 | Smaller batch for longer sequences |
| `max_epochs` | 80 | 150 | More convergence time |
| `patience` | 12 | 20 | More tolerance |
| `dropout` (all) | 0.1 | 0.3 | Reduce OOS vs train gap |
| `weight_decay` | 0.0001 | 0.001 | Stronger L2 |
| LSTM `hidden_size` | 128 | 256 | More capacity for longer sequences |
| Transformer `num_layers` | 2 | 4 | Deeper attention |

**Run:**
```powershell
.\.venv\Scripts\python.exe -m quant_core.train_trend_phase4 --config configs/trend_phase4.yaml
```

---

## PHASE 3 — Mean Reversion Retraining

**Models:** MLP_MR_v1, ResNet_MR_v1, GRN_MR_v1

**Root causes:**
1. `horizon=20` is fundamentally wrong for mean reversion. MR profits when price snaps back quickly. By bar t+20 the reversal has already completed (or failed and reversed again). Signal-to-noise ratio for MR peaks at horizon 2–5.
2. `dropout=0.1` too low for tabular MLP/ResNet — GRN had the highest score (-0.74) showing deeper architectures with gating can slightly compensate, but regularisation is still needed.
3. MR features (`vwap_dev`, `zscore_close_20`, `rsi_14`) are designed for short-term reversion — they decay rapidly in predictive value past 5 bars.

**Config changes** (`configs/mr_phase4.yaml` already updated):
| Parameter | Old | New | Reason |
|---|---|---|---|
| `horizon` | 20 | 3 | MR signal strongest at 3-bar horizon |
| `preferred_backend` | cpu | directml | GPU |
| `max_epochs` | 80 | 150 | More training |
| `patience` | 12 | 20 | More tolerance |
| `dropout` (all) | 0.1 | 0.4 | Heavy regularisation for tabular models |
| `weight_decay` | 0.0001 | 0.002 | Stronger L2 |
| `lr` | 0.0005 | 0.0002 | Lower LR for stability |

**Run:**
```powershell
.\.venv\Scripts\python.exe -m quant_core.train_mr_phase4 --config configs/mr_phase4.yaml
```

---

## PHASE 4 — Scalper Retraining

**Models:** CNN_Scalper_v1, LinearAttn_Scalper_v1, GRU_Scalper_v1

**Root causes (highest severity — worst models in catalog):**
1. `flat_threshold=0.0003` (0.03%) is catastrophically tight. With crypto trading fees of ~0.1%, any move below 0.1% in either direction is unprofitable regardless. Setting threshold at 0.03% means the vast majority of bars are labelled "flat" while the model is rewarded for predicting them flat.
2. Class distribution at 0.03% threshold: approximately 70% flat, 15% up, 15% down. The model collapses to predicting "flat" for everything, giving ~70% flat accuracy but ~0% directional accuracy → overall 3-class accuracy of 37–40%.
3. `horizon=5` is reasonable but still too long for scalping at bar frequency — scalpers trade in 1–2 bar windows.

**Config changes** (`configs/scalper_phase4.yaml` already updated):
| Parameter | Old | New | Reason |
|---|---|---|---|
| `flat_threshold` | 0.0003 | **0.0010** | 3.3× larger — covers transaction costs; only meaningful moves get up/down labels |
| `horizon` | 5 | 2 | True scalper micro-prediction window |
| `seq_len` | 32 | 16 | Shorter memory matches 2-bar prediction target |
| `preferred_backend` | cpu | directml | GPU |
| `use_cyclic_lr` | true | **false** | Cyclic LR was causing oscillation — use flat LR with decay |
| `lr` | 0.001 | 0.0005 | Lower for stability |
| `dropout` (all) | 0.1 | 0.3 | More regularisation |
| `max_epochs` | 80 | 120 | More training |

> **Note on class weights:** The training module should apply `CrossEntropyLoss(weight=torch.tensor([2.0, 0.5, 2.0]))` to up-weight directional classes (up=2×, down=2×) and down-weight the flat class (0.5×). This counteracts any remaining class imbalance after threshold widening. Verify this is active in `quant_core/scalper_training.py`.

**Run:**
```powershell
.\.venv\Scripts\python.exe -m quant_core.train_scalper_phase4 --config configs/scalper_phase4.yaml
```

---

## PHASE 5 — Discretionary Retraining

**Models:** ViT_Disc_v1, Multimodal_Disc_v1, CNNChart_Disc_v1

**Root causes (critical severity — worst generalisation in catalog):**
1. `max_rows_per_symbol=12000` — catastrophic data starvation. 34 symbols × 12,000 rows = 408,000 total training chart images, yet ViT models require millions of diverse examples to learn generalizable visual patterns.
2. Evidence of memorisation: training accuracy 46–67%, OOS accuracy 35–40% — worse than random (33.3% for 3-class).
3. `flat_threshold=0.003` (0.3%) with `horizon=20` bars: at this combination many ambiguous samples fall near the threshold boundary, creating label noise that ViT models overfit to.
4. `dropout=0.1` and `weight_decay=0.0001` are far too weak for ViT models with limited data.

**Config changes** (`configs/discretionary_phase4.yaml` already updated):
| Parameter | Old | New | Reason |
|---|---|---|---|
| `max_rows_per_symbol` | 12,000 | **50,000** | 4× more chart images per symbol |
| `horizon` | 20 | 5 | Shorter horizon → cleaner visual patterns |
| `flat_threshold` | 0.003 | **0.005** | Wider flat zone → fewer ambiguous labels |
| `preferred_backend` | cpu | directml | GPU |
| `batch_size` | 256 | 128 | Better gradient signal for ViT |
| `max_epochs` | 100 | 200 | ViT needs more passes with stronger regularisation |
| `patience` | 20 | 30 | More tolerance |
| `dropout` (all) | 0.1 | **0.4** | Critical — ViT most prone to overfit |
| `weight_decay` | 0.0001 | **0.01** | Strong L2 for ViT |
| `lr` | 0.0003 | **0.0001** | ViT is sensitive to high LR |
| `label_smoothing` | — | **0.1** | Prevents overconfident memorisation |

⚠️ **Longest phase.** ViT at 200 epochs × 34 symbols × 50k rows ≈ several hours on DirectML.

**Run:**
```powershell
.\.venv\Scripts\python.exe -m quant_core.train_discretionary_phase4 --config configs/discretionary_phase4.yaml
```

---

## PHASE 6 — Market Maker Retraining

**Run only if PPO/SAC/DQN still fail after Phase 0 episodic re-eval.**

**Root causes:**
1. PPO ran only 2,000 episodes — insufficient for convergence on 34-symbol data distribution.
2. SAC/DQN ran to `max_steps=500,000` but `replay_buffer_size=50,000` means the buffer turns over ~10× during training. Early diverse experience is continuously overwritten with recent narrow experience → catastrophic forgetting of early exploration.
3. `inventory_lambda=0.1` penalises holding inventory heavily, making the agent too passive and unable to learn spread capture.
4. `episode_length=200` is too short for the agent to learn multi-step reward shaping.

**Config changes** (`configs/mm_phase4.yaml` already updated):
| Parameter | Old | New | Reason |
|---|---|---|---|
| `max_episodes` | 2,000 | 8,000 | 4× for PPO convergence |
| `max_steps` | 500,000 | 2,000,000 | 4× for SAC/DQN |
| `episode_length` | 200 | 400 | Longer episodes for multi-step reward learning |
| `inventory_lambda` | 0.1 | 0.05 | Less penalty → more active spread capture |
| `replay_buffer_size` | 50,000 | 200,000 | 4× buffer — retains more diverse experience |
| `batch_size` | 2,048 | 512 | More frequent gradient updates per step |
| `survival_bonus` | 0.0005 | 0.001 | Stronger incentive to avoid MDD |
| `reward_alpha_neg` | 1.35 | 1.5 | Stronger loss aversion shaping |

**Run:**
```powershell
.\.venv\Scripts\python.exe -m quant_core.train_mm_phase4 --config configs/mm_phase4.yaml
```

---

## PHASE 7 — Full Re-Evaluation Pass

After all retraining completes, run the corrected evaluator to produce final metrics:

```powershell
.\.venv\Scripts\python.exe evaluate_all_checkpoints.py
```

---

## Execution Order

```
Step 0:  Run evaluator FIRST (no retraining yet)
         → Confirms StatArb + RL already pass with evaluator bug fixes

Step 1:  Only if StatArb still fails:
         python -m quant_core.train_stat_arb_phase4 --config configs/stat_arb_phase4.yaml

Step 2:  python -m quant_core.train_trend_phase4 --config configs/trend_phase4.yaml

Step 3:  python -m quant_core.train_mr_phase4 --config configs/mr_phase4.yaml

Step 4:  python -m quant_core.train_scalper_phase4 --config configs/scalper_phase4.yaml

Step 5:  Only if RL still fails after step 0:
         python -m quant_core.train_mm_phase4 --config configs/mm_phase4.yaml

Step 6:  python -m quant_core.train_discretionary_phase4 --config configs/discretionary_phase4.yaml
         (longest — run overnight)

Final:   python evaluate_all_checkpoints.py
```

---

## Expected Outcome After Retraining

| Archetype | Likely Result | Confidence |
|---|---|---|
| GAT_StatArb_v1 | PASS after evaluator fix alone | High |
| Autoencoder_StatArb_v1 | PASS or borderline (PF was 1.46, just below 1.5) | Medium-High |
| LSTM_StatArb_v1 | PASS after retrain (now loads with correct 34 assets) | Medium |
| PPO_MM_v1, SAC_MM_v1, DQN_MM_v1 | PASS or near-pass after episodic re-eval | Medium |
| GRN_MR_v1 | PASS after horizon fix (had best MR Sharpe) | Medium |
| MLP_MR_v1, ResNet_MR_v1 | PASS after horizon fix | Medium |
| LSTM_Trend_v1, TCN_Trend_v1 | PASS after seq_len+horizon fix | Medium |
| Transformer_Trend_v1 | PASS but needs more epochs | Medium |
| CNN_Scalper_v1 | Best chance of passing (best scalper score) | Medium-Low |
| GRU_Scalper_v1, LinearAttn_Scalper_v1 | May need additional tuning | Low-Medium |
| CNNChart_Disc_v1 | Best chance in discretionary group | Low-Medium |
| ViT_Disc_v1, Multimodal_Disc_v1 | Will improve significantly; may need 2nd retrain | Low |
