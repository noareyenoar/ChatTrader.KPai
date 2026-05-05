# User Manual: Model Catalog (Phase 4)

This manual describes all 18 models in the trading stack:
- 6 archetypes
- 3 models per archetype
- shared data governance (Iron Wall split)
- DirectML-safe implementation constraints for AMD GPUs

The document is written for both discretionary traders and AI agents that orchestrate model selection.

---

## 1) System Overview

### 1.1 Objective
Build a multi-archetype model suite where each archetype targets a distinct market behavior:
- Trend continuation
- Mean reversion
- Microstructure/tick-direction scalp
- Statistical convergence/divergence
- Visual/discretionary chart interpretation
- Inventory-aware market making

### 1.2 Execution Backend
Primary backend in this workspace is AMD DirectML on Windows.

Important compatibility constraints that shape architecture choices:
- Avoid fused recurrent kernels that trigger unsupported ops on DirectML.
- Avoid fused Transformer encoder kernels on DirectML.
- For DirectML optimizer stability, SGD fallback is used where Adam/AdamW kernels trigger unsupported operations.
- Checkpoint loading is done via CPU map_location for reliability.

### 1.3 Data Leakage Prevention
All supervised archetypes use strict chronological splitting with purge gap:
- Train: 70%
- Validation: 15%
- Test: 15%
- Purge gap bars between train-val and val-test boundaries
- No random shuffle across time

This is the Iron Wall policy and is mandatory.

---

## 2) Shared Input Contracts

### 2.1 Common Symbol Universe
Accepted symbols come from data quality gate screening.

### 2.2 Common Feature Engineering Utilities
FeatureFactory provides:
- Trend features: log_return, zscore_close_64, ema_spread, atr_14, price_slope_20
- Mean-reversion features: vwap_dev, bb_distance, zscore_close_20, rsi_14, rsi_div_5
- Stat-arb features: fracdiff_close_d04, spread_z_64
- Train-only scaling utilities to avoid leakage

### 2.3 Registry Contract
Each trained model appends one entry into model_registry.json, including:
- architecture_name
- archetype
- checkpoint paths
- validation/test metrics
- backend metadata
- is_valid decision

---

## 3) Archetype Catalog (All 18 Models)

## 3.1 Trend Follower Archetype

### Model 1: Trend_LSTM_v1
- Core idea: temporal continuation with recurrent memory.
- Input: [Batch, Seq_Len, Feature_Dim]
- Output: regression logit/score for directional continuation.
- DirectML detail: implemented with LSTMCell stack to avoid unsupported fused LSTM path.
- Best regime: persistent directional moves with stable momentum.

### Model 2: Trend_Transformer_v1
- Core idea: sequence attention over trend context.
- Input: [Batch, Seq_Len, Feature_Dim]
- Output: directional score.
- DirectML detail: manual multi-head attention and transformer blocks, avoiding fused transformer encoder op.
- Best regime: medium-term trend with non-local dependencies.

### Model 3: Trend_TCN_v1
- Core idea: dilated temporal convolutions capture multi-scale trend patterns.
- Input: [Batch, Seq_Len, Feature_Dim]
- Output: directional score.
- Best regime: smooth trend with local and mid-horizon structure.

Typical validation lens:
- Directional accuracy
- Sharpe
- Max drawdown
- Profit factor

---

## 3.2 Mean Reversion Archetype

Input feature set (tabular, 5 dims):
- vwap_dev
- bb_distance
- zscore_close_20
- rsi_14
- rsi_div_5

Target:
- Binary class (upward reversion vs downward continuation)

### Model 4: MLP_MR_v1
- Core idea: deep tabular nonlinearity for reversal classification.
- Input: [Batch, 5]
- Output: one logit converted to 2-class decision.
- Activation design: DirectML-safe activations.
- Best regime: clean overextension snaps with limited cross-asset complexity.

### Model 5: ResNet_MR_v1
- Core idea: residual tabular network to improve gradient flow at depth.
- Input: [Batch, 5]
- Output: binary reversal decision.
- Best regime: noisy reversions requiring deeper interaction modeling.

### Model 6: GRN_MR_v1
- Core idea: gated residual feature routing, akin to learned feature selection.
- Input: [Batch, 5]
- Output: binary reversal decision.
- Best regime: weak-signal environments where selective gating helps SNR.

MR validity focus:
- Accuracy > 0.55
- Precision at reversal windows > 0.60
- Sharpe > 1.2

---

## 3.3 Scalper / Microstructure Archetype

Input feature tensor:
- Shape: [Batch, Seq_Len=32, 5]
- Features:
  - ofi_proxy
  - microprice_dev
  - spread_pct
  - log_return
  - vol_imbalance

Target:
- 3-class tick-direction style label: down, flat, up

### Model 7: CNN_Scalper_v1
- Core idea: fast 1D conv stack for local microstructure motifs.
- Output: 3 logits.
- Best regime: bursty short-horizon directional pockets.

### Model 8: LinearAttn_Scalper_v1
- Core idea: linearized attention for O(N) sequence mixing.
- Output: 3 logits.
- DirectML-safe kernel map replaces unsupported variants.
- Best regime: longer microstructure context under latency constraints.

### Model 9: GRU_Scalper_v1
- Core idea: bidirectional recurrent modeling of short-term order-flow evolution.
- Output: 3 logits.
- DirectML detail: manual GRU-cell formulation to avoid fused GRU backend failures.
- Best regime: asymmetric buildup then release patterns.

Scalper metrics:
- Accuracy
- F1 macro
- Inference latency (ms)

---

## 3.4 Statistical Arbitrage Archetype

Input:
- Multi-asset aligned sequence
- Shape: [Batch, Seq_Len, Num_Assets]

Target:
- Spread Z-score style regression objective

### Model 10: Autoencoder_StatArb_v1
- Core idea: latent representation + reconstruction error as dislocation signal.
- Output: spread forecast plus reconstruction dynamics.
- DirectML detail: manual recurrent cells for encoder/decoder stability.
- Best regime: correlation structure drifts and temporary dislocations.

### Model 11: GAT_StatArb_v1
- Core idea: graph attention over asset relationships.
- Output: spread regression score.
- Best regime: cluster-wise co-movement with changing graph topology.

### Model 12: LSTM_StatArb_v1
- Core idea: recurrent spread evolution modeling over fractionally-differenced series.
- Output: spread regression score.
- DirectML detail: manual LSTM-cell implementation to avoid fused LSTM kernel failures.
- Best regime: smooth convergence after transient divergence.

Stat-arb metrics:
- MAE
- Tracking error
- Sharpe proxy
- Max drawdown proxy
- Profit factor proxy

---

## 3.5 Discretionary / Multimodal Archetype

Input image:
- Shape: [Batch, 4, 32, 32]
- Rasterized chart channels from OHLC windows

Optional tabular branch:
- Shape: [Batch, 5]
- Momentum/statistical descriptors

Target:
- 3-class directional class

### Model 13: ViT_Disc_v1
- Core idea: patch embedding + transformer blocks for chart pattern semantics.
- Output: 3 logits.
- DirectML detail: manual MHA path (no fused transformer encoder).
- Best regime: visual pattern recognition scenarios (breakouts, formations).

### Model 14: Multimodal_Disc_v1
- Core idea: fuse CNN image embedding with tabular momentum embedding.
- Output: 3 logits.
- Best regime: when both shape context and numeric momentum matter.

### Model 15: CNNChart_Disc_v1
- Core idea: lightweight residual CNN for fast chart scanning.
- Output: 3 logits.
- Best regime: broad universe scan with strict latency budget.

Discretionary metrics:
- Accuracy
- F1 macro
- Latency

---

## 3.6 Market Making RL Archetype

Environment summary:
- Avellaneda-Stoikov-inspired simulator on historical close stream
- State includes inventory, price change, spread proxy, volatility, time progress, pnl, position value
- Reward balances pnl and inventory risk penalty

### Model 16: PPO_MM_v1
- Core idea: actor-critic policy gradient with clipped objective.
- Action: continuous bid/ask offsets.
- Best regime: stable online adaptation with moderate action smoothness.

### Model 17: SAC_MM_v1
- Core idea: entropy-regularized off-policy control with twin critics.
- Action: continuous bid/ask offsets.
- Best regime: noisy fill dynamics requiring exploratory robustness.

### Model 18: DQN_MM_v1
- Core idea: discrete spread-level control with dueling Q architecture.
- Action: tight/medium/wide discrete quote choices.
- Best regime: constrained policy sets where discrete controls are operationally preferred.

RL validity lens:
- Mean episode reward
- Reward stability (std)
- Reward Sharpe-style trend
- Drawdown of reward trajectory

---

## 4) Decision Workflow for Live Use

Use this top-down selection logic:

1. Regime detection first
- Trending: prioritize Trend archetype.
- Oscillating around fair value: prioritize Mean Reversion.
- Very short horizon / microstructure-sensitive: prioritize Scalper.
- Multi-asset relative value dislocation: prioritize Stat Arb.
- Human-pattern or multimodal interpretation need: prioritize Discretionary.
- Inventory + spread quoting objective: use Market Making RL.

2. Model narrowing inside archetype
- Need highest interpretability and speed: prefer simpler CNN/TCN/MLP variants.
- Need richer sequence context: prefer Transformer/attention/recurrent variants.
- Need fusion of modalities: prefer Multimodal discretionary.

3. Confidence and risk gating
- Require model is_valid where defined.
- Cross-check Sharpe, drawdown, and precision/F1 against deployment thresholds.
- Reject models that pass accuracy but fail risk profile.

4. Ensemble fallback
- If top model confidence is low, blend with second-best within same archetype.
- If archetype confidence conflicts with regime classifier, reduce position size.

---

## 5) Input/Output Quick Reference

### Supervised binary models (MR)
- Input: [B, F]
- Output: [B, 1] logit, converted to 2-class decision

### Supervised multiclass models (Scalper, Discretionary)
- Input: sequence or image (+ optional tabular)
- Output: [B, 3] logits

### Supervised regression models (Trend/StatArb depending head)
- Input: [B, T, F] or [B, T, A]
- Output: [B, 1] regression score

### RL policies (Market Maker)
- Input: [B, state_dim]
- Output:
  - PPO/SAC: continuous action parameters
  - DQN: Q-values over discrete actions

---

## 6) Validation and Production Criteria

Use per-archetype criteria, then portfolio-level checks:

- Trend: directional quality + risk-adjusted return
- Mean Reversion: accuracy + precision on high-conviction reversal calls
- Scalper: class quality + latency ceiling
- Stat Arb: MAE/tracking error + risk metrics
- Discretionary: F1 macro + stability on out-of-sample windows
- Market Maker: positive mean reward with acceptable volatility of reward

Hard reject conditions:
- Leakage suspicion
- Empty or degenerate validation split
- Excessive drawdown despite nominal accuracy
- Backend fallback warnings in strict mode if they imply unstable deployment behavior

---

## 7) Operational Notes for AMD DirectML

When running strict warning mode:
- Any DirectML fallback warning can escalate to failure.
- Prefer primitive ops and manually defined recurrent/attention cells.
- Use backend-aware optimizer selection to avoid unsupported kernels.

Practical implications:
- Training may prioritize compatibility over maximal theoretical efficiency.
- Architecture simplifications are intentional for deterministic reproducibility.

---

## 8) Glossary

- OHLCV: Open, High, Low, Close, Volume market bars.
- Sharpe ratio: return-to-volatility efficiency measure.
- Max drawdown: worst peak-to-trough equity decline.
- Profit factor: gross profit divided by gross loss.
- Fractional differentiation: transformation preserving memory while improving stationarity.
- Iron Wall split: strict chronological split with purge gaps preventing temporal leakage.
- DirectML: Microsoft GPU acceleration backend used here for AMD hardware.
- Tracking error: deviation from target spread behavior in stat-arb context.
- F1 macro: class-balanced harmonic mean of precision and recall.

---

## 9) Recommended Deployment Pattern

1. Run nightly retraining per archetype with Iron Wall policy.
2. Update registry and keep only validated checkpoints for promotion.
3. At inference, run regime filter first, then archetype-specific model stack.
4. Apply risk overlay to position sizing regardless of model confidence.
5. Log prediction, confidence, realized outcome, and drift indicators for continual monitoring.

This concludes the model catalog for all 18 Phase 4 architectures.
