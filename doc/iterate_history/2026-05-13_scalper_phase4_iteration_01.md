# Iteration History

- Timestamp: 2026-05-13 00:00 local
- Scope: Start Scalper Phase 4 training
- Objective: Evaluate scalper viability, keep a heartbeat on training progress, and decide whether to iterate or archive failures based on validation/backtest/testing results.
- Config: configs/scalper_phase4.yaml
- Planned execution: python -m quant_core.train_scalper_phase4 --config configs/scalper_phase4.yaml
- Notes: Phase 4 scalper uses seq_len=16, horizon=2, flat_threshold=0.0010, dropout=0.3, max_epochs=120, patience=15.
- Timestamp: 2026-05-13 04:10 UTC+7
- Work done: Launched live Scalper Phase 4 run, confirmed active process, captured current CNN progress at epoch 7/120, and estimated remaining runtime at roughly 2-3 hours for the full scalper suite.
- Timestamp: 2026-05-13 04:11 UTC+7
- Work done: Heartbeat confirmed the scalper run is still advancing; CNN moved to epoch 11/120 with checkpoint val_loss=1.08332, while LinearAttn remains at its prior checkpoint metadata. ETA remains roughly 2-3 hours for the full scalper suite unless early-stop triggers sooner.
- Timestamp: 2026-05-13 04:16 UTC+7
- Work done: Working log now shows CNN_Scalper_v1 at epoch 14/120 with val_loss=1.082735 and a steadier pace of about 43s/epoch. Revised ETA: about 65-75 minutes for CNN and 3-5 hours for the full scalper suite if no early-stop interruption occurs.
- Timestamp: 2026-05-13 10:31 UTC+7
- Work done: CNN_Scalper_v1 completed and failed validation/backtest gates (test_sharpe=-8.703060, PF=0.831336, MDD=1.000000). Marked CNN as a true failure to archive; LinearAttn remains the active scalper stage with latest checkpoint epoch 27/120 and val_loss=1.070889.
- Timestamp: 2026-05-13 10:59 UTC+7
- Work done: Re-checked the live tail and confirmed the current active scalper pass is CNN_Scalper_v1 at epoch 15/120 with val_loss=1.082319. The previously archived CNN failure remains closed; this new CNN pass is still in progress and should be watched for a repeat failure or a late recovery.
- Timestamp: 2026-05-13 11:00 UTC+7
- Work done: Live heartbeat advanced again to CNN_Scalper_v1 epoch 16/120 with val_loss=1.082140 and ~20s/epoch cadence. The current pass is still active, while the earlier CNN failure remains archived separately.
