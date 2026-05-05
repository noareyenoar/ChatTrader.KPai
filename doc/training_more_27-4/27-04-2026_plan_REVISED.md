# Phase 4.1 Revised Plan — Strategic Pivot (ROI-First)

Generated: 2026-04-28
Owner: Head of Quant Research & Optimization Architect

## Executive Decision
Brute-force continuation is suspended for non-convergent groups. Current strategy shifts from epoch accumulation to targeted redesign and controlled re-entry.

## Triage Groups

### Group 1 — Near-Gate (Continue with fine-tuning)
Models:
- Transformer_Trend_v1
- TCN_Trend_v1
- LSTM_Trend_v1
- MLP_MR_v1
- ResNet_MR_v1

Applied changes:
- Trend config: `patience=15`, `lr=0.0005`
- Mean Reversion config: `patience=15`, `lr=0.0005`

Files changed:
- configs/trend_phase4.yaml
- configs/mr_phase4.yaml

### Group 2 — Broken (Scalper)
Models:
- CNN_Scalper_v1
- LinearAttn_Scalper_v1
- GRU_Scalper_v1

Actions:
- Cease retraining until signal inversion/collapse audit is closed.
- Audit class distribution, prediction collapse-to-flat behavior, and CE weighting path.
- If collapse persists (>90% flat), enforce class re-balancing on training tensors before next sweep.

### Group 3 — Broken (Discretionary)
Models:
- ViT_Disc_v1
- Multimodal_Disc_v1
- CNNChart_Disc_v1

Actions:
- No resume from current checkpoints.
- Pivot path: replace ViT-first path with simpler backbone or raise chart resolution after synchronization checks.
- Verify multi-instrument alignment and timestamp integrity before new training run.

## Safety Protocols (Implemented)

### Training hard gate
- `execute_all_phases.py` now blocks all batch training unless:
  - `SCALPER_SIGNAL_INVERSION_FIXED.flag` exists in repository root.
- This prevents accidental brute-force retries before scalper root cause is closed.

### Batch cool-down
- `execute_all_phases.py` now enforces a 300s (5-minute) cool-down between batch runs.

## Registry Reset (Implemented)
Group 2 + Group 3 entries are reset in `model_registry.json` with:
- `status: RESET_REQUIRED_PHASE41`
- `reason: phase41_pivot_non_convergent_group`

## Audit Evidence (Completed)

### Scalper collapse audit
Evidence file:
- doc/training_more_27-4/phase41_scalper_audit.json

Key findings:
- `CNN_Scalper_v1` validation prediction ratio: `[0.000259, 0.999741, 0.0]` with `flat_collapse_flag=true`.
- `LinearAttn_Scalper_v1` and `GRU_Scalper_v1` do not show flat-class collapse under the same audit slice.
- Gate remains closed because at least one production candidate still exhibits collapse behavior.

### Feature correlation + RF pruning audit
Evidence file:
- doc/training_more_27-4/phase41_feature_revalidation.json

Bottom-30% prune candidates from quick RF importance pass:
- Scalper feature set: `vol_imbalance`, `vol_regime_code`, `ofi_proxy`, `price_velocity_5`
- Discretionary tab feature set: `price_slope_20`, `log_return`

Status:
- Correlation and RF re-validation step is complete and reproducible.
- Pruning candidates are approved for next config revision batch (not yet injected into production configs in this step).

## Tactical Work Queue (Phase 4.1)
1. Scalper collapse remediation in training path (CNN branch) using audit findings.
2. Discretionary data validation: synchronization and regime separability checks.
3. Inject approved bottom-30% feature-prune candidates into next experimental configs.
4. Run 10-trial Optuna search for Group 2 and Group 3 starting points (DirectML-compatible settings).
5. Re-open training gate only after collapse fix is confirmed in fresh validation diagnostics and flag is created.

## Re-open Criteria
All must be true before creating `SCALPER_SIGNAL_INVERSION_FIXED.flag`:
- Scalper collapse mechanism identified and fixed in code.
- Validation prediction distribution no longer dominated by a single class.
- Group 2/3 replacement configs pass smoke-eval sanity.
- HPO baseline candidates selected.
