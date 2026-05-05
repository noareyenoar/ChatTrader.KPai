# Model Performance Summary (Full Training Sweep)

## Leaderboard

| Rank | Model | Archetype | Test Sharpe | Val Accuracy | Status |
|---:|---|---|---:|---:|---|
| 1 | LSTM_StatArb_v1 | statistical_arbitrage | 9.9906 | n/a | PASSED |
| 2 | Autoencoder_StatArb_v1 | statistical_arbitrage | 9.4245 | n/a | PASSED |
| 3 | Multimodal_Disc_v1 | discretionary_multimodal | 4.0228 | 0.4337 | FAILED |
| 4 | CNNChart_Disc_v1 | discretionary_multimodal | 3.0833 | 0.4148 | FAILED |
| 5 | PPO_MM_v1 | market_making_rl | 0.0695 | n/a | PASSED |
| 6 | SAC_MM_v1 | market_making_rl | 0.0058 | n/a | PASSED |
| 7 | DQN_MM_v1 | market_making_rl | 0.0054 | n/a | PASSED |
| 8 | GAT_StatArb_v1 | statistical_arbitrage | -0.6031 | n/a | FAILED |
| 9 | ViT_Disc_v1 | discretionary_multimodal | -7.3585 | 0.3995 | FAILED |
| 10 | LSTM_Trend_v1 | trend_follower | n/a | n/a | PASSED |
| 11 | Transformer_Trend_v1 | trend_follower | n/a | n/a | PASSED |
| 12 | TCN_Trend_v1 | trend_follower | n/a | n/a | PASSED |
| 13 | MLP_MR_v1 | mean_reversion | n/a | n/a | PASSED |
| 14 | ResNet_MR_v1 | mean_reversion | n/a | n/a | PASSED |
| 15 | GRN_MR_v1 | mean_reversion | n/a | n/a | PASSED |
| 16 | CNN_Scalper_v1 | scalping_microstructure | n/a | n/a | PASSED |
| 17 | LinearAttn_Scalper_v1 | scalping_microstructure | n/a | n/a | PASSED |
| 18 | GRU_Scalper_v1 | scalping_microstructure | n/a | n/a | PASSED |

## Per-Model Metrics

| Model | Archetype | Train Accuracy | Val Accuracy | Train Loss | Val Loss | Test Sharpe | Test Max Drawdown | Test Profit Factor | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| LSTM_Trend_v1 | trend_follower | 0.5724 | n/a | 0.6702 | 0.6949 | n/a | n/a | n/a | PASSED |
| Transformer_Trend_v1 | trend_follower | 0.4996 | n/a | 0.0003 | 0.0002 | n/a | n/a | n/a | PASSED |
| TCN_Trend_v1 | trend_follower | 0.5093 | n/a | 0.0001 | 0.0013 | n/a | n/a | n/a | PASSED |
| MLP_MR_v1 | mean_reversion | 0.5298 | n/a | 0.6904 | 0.6926 | n/a | n/a | n/a | PASSED |
| ResNet_MR_v1 | mean_reversion | 0.5295 | n/a | 0.6905 | 0.6928 | n/a | n/a | n/a | PASSED |
| GRN_MR_v1 | mean_reversion | 0.5298 | n/a | 0.6905 | 0.6926 | n/a | n/a | n/a | PASSED |
| CNN_Scalper_v1 | scalping_microstructure | 0.4213 | n/a | 1.0025 | 1.0414 | n/a | n/a | n/a | PASSED |
| LinearAttn_Scalper_v1 | scalping_microstructure | 0.4155 | n/a | 1.0018 | 1.0309 | n/a | n/a | n/a | PASSED |
| GRU_Scalper_v1 | scalping_microstructure | 0.4438 | n/a | 1.1238 | 1.0486 | n/a | n/a | n/a | PASSED |
| Autoencoder_StatArb_v1 | statistical_arbitrage | n/a | n/a | 0.1469 | 1.3405 | 9.4245 | 0.8898 | 1.0907 | PASSED |
| GAT_StatArb_v1 | statistical_arbitrage | n/a | n/a | 0.0457 | 1.2211 | -0.6031 | 1.0000 | 0.9945 | FAILED |
| LSTM_StatArb_v1 | statistical_arbitrage | n/a | n/a | 0.0298 | 0.8973 | 9.9906 | 0.8803 | 1.0964 | PASSED |
| ViT_Disc_v1 | discretionary_multimodal | 0.4617 | 0.3995 | 1.0096 | 1.0894 | -7.3585 | 1.0000 | 0.9246 | FAILED |
| Multimodal_Disc_v1 | discretionary_multimodal | 0.5879 | 0.4337 | 0.8645 | 1.2039 | 4.0228 | 1.0000 | 1.0421 | FAILED |
| CNNChart_Disc_v1 | discretionary_multimodal | 0.6685 | 0.4148 | 0.7288 | 1.5521 | 3.0833 | 1.0000 | 1.0326 | FAILED |
| PPO_MM_v1 | market_making_rl | n/a | n/a | n/a | n/a | 0.0695 | 0.0000 | n/a | PASSED |
| SAC_MM_v1 | market_making_rl | n/a | n/a | n/a | n/a | 0.0058 | 1.0000 | n/a | PASSED |
| DQN_MM_v1 | market_making_rl | n/a | n/a | n/a | n/a | 0.0054 | 1.0000 | n/a | PASSED |

## Notes

- Train Accuracy is shown as n/a when not logged by that model family.
- Val Loss is sourced from TensorBoard val/loss when available; otherwise from registry validation fields.
- FAILED status is applied when Test Sharpe < 0 or Test Accuracy < 50%.
