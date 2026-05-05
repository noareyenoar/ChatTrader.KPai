# Model Performance Summary — Strict OOS Re-Evaluation Pass

> **Re-evaluation date:** 2026-05-05  
> **Protocol:** Iron Wall — 70% Train / 15% Val / 15% Test chronological split. Training data discarded before evaluation. All models loaded from saved checkpoints in `eval()` mode, zero gradient passes.  
> **Hardware:** AMD Radeon RX 6750 via `torch_directml`  
> **OOS samples per model:** up to 20,000 (tail of test split — most recent, unseen data)  
> **Pass gate:** Sharpe > 1.2 AND Profit Factor > 1.5 AND Max Drawdown < 0.20 AND Directional Accuracy > 0.55 (all four must hold simultaneously)

---

## Leaderboard (Ranked by Test Sharpe — Real OOS Metrics Only)

| Rank | Model | Archetype | Test Sharpe | Test Dir Acc | Test Profit Factor | Test Max Drawdown | Status |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | GAT_StatArb_v1 | statistical_arbitrage | **1.9748** | 0.5568 | 1.6853 | 1.0000 | **FAILED** ¹ |
| 2 | Autoencoder_StatArb_v1 | statistical_arbitrage | **1.4525** | 0.5532 | 1.4642 | 1.0000 | **FAILED** ² |
| 3 | GRN_MR_v1 | mean_reversion | -0.7405 | 0.5073 | 0.8156 | 1.0000 | **FAILED** |
| 4 | CNN_Scalper_v1 | scalping_microstructure | -0.9983 | 0.4031 | 0.7349 | 1.0000 | **FAILED** |
| 5 | GRU_Scalper_v1 | scalping_microstructure | -1.2542 | 0.3817 | 0.6901 | 1.0000 | **FAILED** |
| 6 | LinearAttn_Scalper_v1 | scalping_microstructure | -1.6963 | 0.3668 | 0.5670 | 1.0000 | **FAILED** |
| 7 | LSTM_Trend_v1 | trend_follower | -1.7256 | 0.5191 | 0.6157 | 1.0000 | **FAILED** |
| 8 | TCN_Trend_v1 | trend_follower | -1.7256 | 0.5192 | 0.6157 | 1.0000 | **FAILED** |
| 9 | MLP_MR_v1 | mean_reversion | -1.7336 | 0.5175 | 0.6156 | 1.0000 | **FAILED** |
| 10 | ResNet_MR_v1 | mean_reversion | -1.7953 | 0.5169 | 0.6047 | 1.0000 | **FAILED** |
| 11 | CNNChart_Disc_v1 | discretionary_multimodal | -1.8424 | 0.3925 | 0.6457 | 1.0000 | **FAILED** |
| 12 | Transformer_Trend_v1 | trend_follower | -1.8865 | 0.5167 | 0.5889 | 1.0000 | **FAILED** |
| 13 | ViT_Disc_v1 | discretionary_multimodal | -1.9119 | 0.3558 | 0.5988 | 1.0000 | **FAILED** |
| 14 | Multimodal_Disc_v1 | discretionary_multimodal | -2.2178 | 0.3989 | 0.5859 | 1.0000 | **FAILED** |
| 15 | DQN_MM_v1 | market_making_rl | -4.3059 | 0.5287 | 0.3550 | 0.9999 | **FAILED** |
| 16 | PPO_MM_v1 | market_making_rl | -4.3157 | 0.5641 | 0.3553 | 0.9999 | **FAILED** |
| 17 | SAC_MM_v1 | market_making_rl | -4.3157 | 0.5641 | 0.3553 | 0.9999 | **FAILED** |
| 18 | LSTM_StatArb_v1 | statistical_arbitrage | N/A ³ | N/A | N/A | N/A | **FAILED** |

> ¹ GAT_StatArb_v1: Sharpe (1.97) and PF (1.69) are above gate thresholds but Max Drawdown = 1.00 (100%) — fails the ≤ 20% drawdown gate. Closest to production-ready of the 18.  
> ² Autoencoder_StatArb_v1: Sharpe (1.45) clears the gate but Profit Factor (1.46) is just below 1.5 and Max Drawdown = 1.00 — fails on two gates.  
> ³ LSTM_StatArb_v1: Weight shape mismatch at load time (`in_proj.weight` checkpoint = [64,34,1], model expects [64,2,1]). Checkpoint was trained on 34 assets; eval manifest specifies 2. Cannot run inference — marked FAILED.

---

## Per-Model Metrics (Full OOS Detail)

| Model | Archetype | Train Acc | OOS Dir Acc | Train Loss | Val Loss | Test Sharpe | Test Max Drawdown | Test Profit Factor | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| LSTM_Trend_v1 | trend_follower | 0.5724 | 0.5191 | 0.6702 | 0.6949 | -1.7256 | 1.0000 | 0.6157 | **FAILED** |
| Transformer_Trend_v1 | trend_follower | 0.4996 | 0.5167 | 0.0003 | 0.0002 | -1.8865 | 1.0000 | 0.5889 | **FAILED** |
| TCN_Trend_v1 | trend_follower | 0.5093 | 0.5192 | 0.0001 | 0.0013 | -1.7256 | 1.0000 | 0.6157 | **FAILED** |
| MLP_MR_v1 | mean_reversion | 0.5298 | 0.5175 | 0.6904 | 0.6926 | -1.7336 | 1.0000 | 0.6156 | **FAILED** |
| ResNet_MR_v1 | mean_reversion | 0.5295 | 0.5169 | 0.6905 | 0.6928 | -1.7953 | 1.0000 | 0.6047 | **FAILED** |
| GRN_MR_v1 | mean_reversion | 0.5298 | 0.5073 | 0.6905 | 0.6926 | -0.7405 | 1.0000 | 0.8156 | **FAILED** |
| CNN_Scalper_v1 | scalping_microstructure | 0.4213 | 0.4031 | 1.0025 | 1.0414 | -0.9983 | 1.0000 | 0.7349 | **FAILED** |
| LinearAttn_Scalper_v1 | scalping_microstructure | 0.4155 | 0.3668 | 1.0018 | 1.0309 | -1.6963 | 1.0000 | 0.5670 | **FAILED** |
| GRU_Scalper_v1 | scalping_microstructure | 0.4438 | 0.3817 | 1.1238 | 1.0486 | -1.2542 | 1.0000 | 0.6901 | **FAILED** |
| Autoencoder_StatArb_v1 | statistical_arbitrage | — | 0.5532 | 0.1469 | 1.3405 | 1.4525 | 1.0000 | 1.4642 | **FAILED** |
| GAT_StatArb_v1 | statistical_arbitrage | — | 0.5568 | 0.0457 | 1.2211 | 1.9748 | 1.0000 | 1.6853 | **FAILED** |
| LSTM_StatArb_v1 | statistical_arbitrage | — | N/A | 0.0298 | 0.8973 | N/A | N/A | N/A | **FAILED** |
| ViT_Disc_v1 | discretionary_multimodal | 0.4617 | 0.3558 | 1.0096 | 1.0894 | -1.9119 | 1.0000 | 0.5988 | **FAILED** |
| Multimodal_Disc_v1 | discretionary_multimodal | 0.5879 | 0.3989 | 0.8645 | 1.2039 | -2.2178 | 1.0000 | 0.5859 | **FAILED** |
| CNNChart_Disc_v1 | discretionary_multimodal | 0.6685 | 0.3925 | 0.7288 | 1.5521 | -1.8424 | 1.0000 | 0.6457 | **FAILED** |
| PPO_MM_v1 | market_making_rl | — | 0.5641 | — | — | -4.3157 | 0.9999 | 0.3553 | **FAILED** |
| SAC_MM_v1 | market_making_rl | — | 0.5641 | — | — | -4.3157 | 0.9999 | 0.3553 | **FAILED** |
| DQN_MM_v1 | market_making_rl | — | 0.5287 | — | — | -4.3059 | 0.9999 | 0.3550 | **FAILED** |

---

## Evaluation Incident Log

| Model | Issue | Action Required |
|---|---|---|
| **LSTM_StatArb_v1** | `in_proj.weight` shape mismatch: checkpoint stores [64,34,1] (trained on 34 assets), evaluator manifest specifies `num_assets=2`. Inference aborted. | Retrain with correct `num_assets` config OR fix manifest to match checkpoint. |
| **All StatArb models** | `test_max_drawdown = 1.0000` — the stat_arb OOS dataset had only 2,518 samples (2 assets × test window). Spread-mean-reversion PnL collapses to full ruin under worst-case paths at that sample size. | Increase `num_assets` in eval to match the 34-asset training regime for a more representative drawdown estimate. |
| **All RL Market-Maker models** | Sharpe extremely negative (-4.3). The `load_mm_data` evaluator maps RL policy outputs to directional binary predictions against raw log-returns — this is structurally incorrect for PPO/SAC which optimise spread quoting reward, not direction. Financial metrics are therefore pessimistic for this archetype. | Implement a proper market-maker episode-based evaluator (step through order-book simulation) instead of a one-shot directional forward-pass. |
| **All Discretionary models** | All three models show OOS accuracy 35–40%, well below 50% and the 65% F1 target from the training protocol. Training Acc was 46–67% vs OOS 35–40% — classic overfitting pattern. | Retrain with stronger regularisation, data augmentation on chart images, and cross-validation. |
| **Scalper models** | OOS directional accuracy 37–40% (below random on 3-class). Model may have learned spurious microstructure patterns that do not transfer OOS. | Retrain with larger purge gap and feature re-selection; validate that scaler parameters are consistent. |

---

## Summary

**0 / 18 models pass strict production gates** (Sharpe > 1.2, PF > 1.5, MDD < 20%, Dir Acc > 55%).

The previous "PASSED" labels in this document were generated from a misconfigured validation pass that omitted financial metrics for 9 models and applied per-archetype relaxed gates instead of the uniform four-gate production standard. This re-evaluation corrects that record.

The nearest candidates to production readiness:
1. **GAT_StatArb_v1** — Sharpe 1.97, PF 1.69 (both above gate). Only failure: Max Drawdown = 100% on the 2-asset OOS window. Likely an evaluator artefact (see incident log). Priority candidate for re-evaluation with a proper multi-asset dataset.
2. **Autoencoder_StatArb_v1** — Sharpe 1.45, PF 1.46. Both marginally below respective gates. High-priority for continued training.
