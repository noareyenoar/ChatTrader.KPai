# Model Performance Summary (Full Training Sweep)

## Leaderboard

| Rank | Model | Archetype | Test Sharpe | Val Accuracy | Status |
|---:|---|---|---:|---:|---|
| 1 | Multimodal_Disc_v1 | discretionary_multimodal | 3.5399 | 0.7522 | PASSED |
| 2 | TCN_Trend_v1 | trend_follower | 2.2660 | 0.5201 | REVIEW |
| 3 | SAC_MM_v1 | market_making_rl | 0.0043 | n/a | PASSED |
| 4 | DQN_MM_v1 | market_making_rl | 0.0008 | n/a | PASSED |
| 5 | ViT_Disc_v1 | discretionary_multimodal | 0.0000 | 0.7508 | PASSED |
| 6 | CNNChart_Disc_v1 | discretionary_multimodal | 0.0000 | 0.7508 | PASSED |
| 7 | PPO_MM_v1 | market_making_rl | -0.0302 | n/a | FAILED |
| 8 | LSTM_Trend_v1 | trend_follower | -0.6479 | 0.5212 | FAILED |
| 9 | Transformer_Trend_v1 | trend_follower | -2.3524 | 0.5136 | FAILED |
| 10 | GRU_Scalper_v1 | scalping_microstructure | -8.0631 | 0.4101 | FAILED |
| 11 | CNN_Scalper_v1 | scalping_microstructure | -8.7031 | 0.4123 | FAILED |
| 12 | LinearAttn_Scalper_v1 | scalping_microstructure | -10.2714 | 0.4131 | FAILED |
| 13 | MLP_MR_v1 | mean_reversion | n/a | n/a | PASSED |
| 14 | ResNet_MR_v1 | mean_reversion | n/a | n/a | PASSED |
| 15 | GRN_MR_v1 | mean_reversion | n/a | n/a | PASSED |
| 16 | Autoencoder_StatArb_v1 | statistical_arbitrage | n/a | n/a | PASSED |
| 17 | GAT_StatArb_v1 | statistical_arbitrage | n/a | n/a | PASSED |
| 18 | LSTM_StatArb_v1 | statistical_arbitrage | n/a | n/a | PASSED |
| 19 | TG_MNN_v1 | trend_follower | n/a | n/a | PASSED |
| 20 | APV_PLN_v1 | apv_pln | n/a | n/a | PASSED |

## Per-Model Metrics

| Model | Archetype | Train Accuracy | Val Accuracy | Train Loss | Val Loss | Test Sharpe | Test Max Drawdown | Test Profit Factor | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| LSTM_Trend_v1 | trend_follower | 0.5783 | 0.5212 | 0.4620 | 0.7101 | -0.6479 | 36.4705 | 0.9902 | FAILED |
| Transformer_Trend_v1 | trend_follower | 0.5281 | 0.5136 | 0.4837 | 0.6919 | -2.3524 | 5.1012 | 0.9648 | FAILED |
| TCN_Trend_v1 | trend_follower | 0.5601 | 0.5201 | 0.4758 | 0.7036 | 2.2660 | 23.9107 | 1.0351 | REVIEW |
| MLP_MR_v1 | mean_reversion | 0.5456 | n/a | 0.6875 | 0.6829 | n/a | n/a | n/a | PASSED |
| ResNet_MR_v1 | mean_reversion | 0.5473 | n/a | 0.6868 | 0.6822 | n/a | n/a | n/a | PASSED |
| GRN_MR_v1 | mean_reversion | 0.5462 | n/a | 0.6872 | 0.6823 | n/a | n/a | n/a | PASSED |
| CNN_Scalper_v1 | scalping_microstructure | 0.4173 | 0.4123 | 1.0439 | 1.0494 | -8.7031 | 1.0000 | 0.8313 | FAILED |
| LinearAttn_Scalper_v1 | scalping_microstructure | 0.4208 | 0.4131 | 1.0477 | 1.0564 | -10.2714 | 1.0000 | 0.8079 | FAILED |
| GRU_Scalper_v1 | scalping_microstructure | 0.4163 | 0.4101 | 1.0661 | 1.0778 | -8.0631 | 1.0000 | 0.8396 | FAILED |
| Autoencoder_StatArb_v1 | statistical_arbitrage | n/a | n/a | 0.6189 | 0.5700 | n/a | n/a | n/a | PASSED |
| GAT_StatArb_v1 | statistical_arbitrage | n/a | n/a | 0.4767 | 0.7718 | n/a | n/a | n/a | PASSED |
| LSTM_StatArb_v1 | statistical_arbitrage | n/a | n/a | 0.3084 | 0.7670 | n/a | n/a | n/a | PASSED |
| ViT_Disc_v1 | discretionary_multimodal | 0.7119 | 0.7508 | 0.7801 | 0.7215 | 0.0000 | 0.0000 | 1.0000 | PASSED |
| Multimodal_Disc_v1 | discretionary_multimodal | 0.7128 | 0.7522 | 0.7588 | 0.7190 | 3.5399 | 1.0000 | 1.1703 | PASSED |
| CNNChart_Disc_v1 | discretionary_multimodal | 0.7119 | 0.7508 | 0.7719 | 0.7184 | 0.0000 | 0.0000 | 1.0000 | PASSED |
| PPO_MM_v1 | market_making_rl | n/a | n/a | n/a | n/a | -0.0302 | 1.0000 | n/a | FAILED |
| SAC_MM_v1 | market_making_rl | n/a | n/a | n/a | n/a | 0.0043 | 1.0000 | n/a | PASSED |
| DQN_MM_v1 | market_making_rl | n/a | n/a | n/a | n/a | 0.0008 | 0.0000 | n/a | PASSED |
| TG_MNN_v1 | trend_follower | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASSED |
| APV_PLN_v1 | apv_pln | n/a | n/a | n/a | n/a | n/a | n/a | n/a | PASSED |

## Notes

- Train Accuracy is shown as n/a when not logged by that model family.
- Val Loss is sourced from TensorBoard val/loss when available; otherwise from registry validation fields.
- FAILED status is applied when Test Sharpe < 0 or Test Accuracy < 50%.
