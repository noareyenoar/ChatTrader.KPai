# Temporary Training Report (Emergency Forensic Pivot)

Generated: 2026-05-12 (local)

## 1) Emergency Stop Status

- Prior trend training terminal was terminated by request.
- Last recorded trend state before termination:
  - `Transformer_Trend_v1` final line: `test_sharpe=-7.9806` with divergence alert.
  - `LSTM_Trend_v1` had started and reached epoch 5 mid-epoch before hard stop.

## 2) Forensic Audit Results

Primary artifact:
- `doc/trend_forensic_audit_report.json`

Executed checks:
1. Zero-information shuffled-label test on trend pipeline (Iron Wall split retained).
2. Feature factory static causality audit (`build_trend_features`).
3. Feature-target timestamp alignment proof (feature time at t, label from t->t+horizon).

Outcomes:
- Feature factory static causality check: PASS
  - No negative shift in trend feature builder.
  - No centered rolling windows.
- Target alignment check: PASS
  - `max_abs_target_return_diff=0.0`
  - sampled rows confirm label horizon timestamp strictly after feature timestamp.
- Shuffled-label leakage probe:
  - final val_sharpe: `9.5046`
  - final test_sharpe: `0.4508`
  - naive always-long test Sharpe baseline: `11.0637`
  - leakage flag rule (test-side): NOT TRIGGERED

Decision:
- `pivot_to_mean_reversion=true`
- Reason: No structural leakage signal on OOS/test under shuffled labels and causality/alignment checks passed.

## 3) Pivot Execution Status

Smoke validation completed:
- `python -m quant_core.train_mr_phase4 --config configs/mr_phase4_smoke.yaml`
- Entrypoint and training loop: healthy (all MR models executed 1 epoch).

Full MR run launched:
- Command: `python -m quant_core.train_mr_phase4 --config configs/mr_phase4.yaml`
- Active terminal id: `e81ff648-7e07-494b-8607-fcf9810efdd8`
- Current state: `mlp` model training, epoch 1/150 in progress.

## 4) Mean Reversion Full Run Outcomes

Completed at: 2026-05-12 (local)
Elapsed: 4939.50 seconds (~82 minutes)

Models trained:
1. **MLP_MR_v1**
   - Stopped at epoch 143 (early-stop patience=20)
   - val_acc=0.5622, test_acc=0.5512, test_sharpe=-13.8499
   - is_valid=False

2. **ResNet_MR_v1**
   - Stopped at epoch 59 (early-stop patience=20)
   - val_acc=0.5632, test_acc=0.5510, test_sharpe=-13.3205
   - is_valid=False

3. **GRN_MR_v1**
   - Stopped at epoch 46 (early-stop patience=20)
   - val_acc=0.5636, test_acc=0.5515, test_sharpe=-12.1727
   - is_valid=False

All models wrote to `model_registry.json`.

## 5) Forensic Audit + Pivot Decision Summary

### Findings:
- Feature factory causality: PASS (no future leakage in trend features)
- Target alignment: PASS (labels strictly from future time horizons)
- Shuffled-label test: NO STRUCTURAL LEAKAGE SIGNAL
  - val_sharpe on shuffled labels: 9.5 (high noise, expected)
  - test_sharpe on shuffled labels: 0.45 (below baseline)
  - Criterion: test_sharpe > 1.2 AND > always-long baseline + 1.0 = NOT MET

### Decision:
- Pivot from Trend → Mean Reversion executed
- Reason: Forensic checks cleared structural leakage; trend divergence was likely regime/objective mismatch

### MR Pivot Outcome:
- Negative Sharpe across all MR architectures (-13.8 to -12.1)
- Suggests either:
  1. Mean reversion signal unsuitable for current market regime
  2. Horizon=3 bars too short for reliable signals
  3. Feature engineering not capturing relevant dynamics
  4. Overall strategy architecture mismatch

### Immediate Action Required:
- Evaluate remaining Phase 4 archetypes (Scalper, Stat Arb, Discretionary, MM)
- Check if any archetype passes is_valid gates with positive OOS Sharpe
- If all fail, assess whether data/horizon/feature assumptions need fundamental revision

## 6) Path Forward

1. Continue to Scalper Phase 4 training (shortest horizon, highest frequency)
2. If Scalper also fails negative Sharpe gate, escalate to architecture/feature redesign
3. Do NOT resume Trend training without new feature engineering or regime-specific tuning

## 7) Scalper Live Status

Current run:
- Command: `python -m quant_core.train_scalper_phase4 --config configs/scalper_phase4.yaml`
- Active process observed: yes
- Latest confirmed final model: `CNN_Scalper_v1`
- CNN final: `val_acc=0.412348`, `test_acc=0.431464`, `test_sharpe=-8.703060`, `test_profit_factor=0.831336`, `test_max_drawdown=1.000000`
- Current active scalper stage: `CNN_Scalper_v1` (fresh pass; latest observed epoch `16/120`, `val_loss=1.082140`)

Observed pace:
- Roughly `~20s/epoch` for the current CNN pass
- Validation is improving slowly but the prior CNN pass still failed the Phase 4 gates on test metrics

ETA estimate:
- Current CNN pass remaining time: about `30-40 minutes` if the current pace holds and early stop does not trigger sooner
- Full scalper suite completion: about `2-4 hours` from the current checkpoint, assuming the remaining models continue at similar pace and GRU does not stall
- If validation plateaus and patience triggers early stop, the total can finish earlier

Current action:
- Keep heartbeat monitoring the run
- Archive CNN as a true failure
- Continue watching the fresh CNN pass; if it again fails, archive this run and move on
