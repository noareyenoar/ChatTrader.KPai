# Full Run User Guide - Multi-Instrument Upgrade

## 1) Environment Setup

### Prerequisites
- **OS**: Windows 11 (PowerShell or Command Prompt)
- **Python**: 3.11.9+ (venv configured)
- **GPU**: AMD RX 6750 (DirectML supported)
- **Disk**: ~500 GB (for 365-day multi-instrument data + models)

### Activate Virtual Environment

```powershell
# Navigate to project root
cd d:\kp_ai_agent\ChatTrader.KPai

# Activate venv
.\.venv\Scripts\Activate.ps1
```

### Install/Verify Dependencies

```powershell
# Core ML/Data packages
pip install torch torch-directml pandas pyarrow aiohttp pyyaml scikit-learn

# Development (optional)
pip install jupyter notebook ipython

# Verify GPU support
python -c "import torch_directml; print(f'DirectML device: {torch_directml.device()}')"
```

---

## 2) Run Full Upgrade (One Command)

```powershell
# Default: 365 days, 16 concurrent download threads
d:/kp_ai_agent/ChatTrader.KPai/.venv/Scripts/python.exe execute_multi_instrument_upgrade.py

# Customizable
d:/kp_ai_agent/ChatTrader.KPai/.venv/Scripts/python.exe execute_multi_instrument_upgrade.py --days 365 --concurrency 16
```

### What This Does (Automatically)

1. **Phase 1: Multi-Instrument Data Ingestion** (~30-60 min)
   - Downloads Futures UM/CM (aggTrades, fundingRate, metrics) for 8+ symbols
   - Downloads Options volatility (BVOL) for BTC/ETH
   - Validates all files with SHA256 checksums
   - Outputs: Dataset/{spot,futures,options}

2. **Phase 2: Feature Engineering** (~5 min)
   - Triggered only if Phase 1 downloaded new files
   - Builds tick bars (1000-trade window) and volume bars (100 BTC)
   - Extracts OFI, VPIN, spread dynamics
   - Generates GARCH Monte Carlo paths and HMM regime scenarios
   - Creates triple-barrier labels (LONG/FLAT/SHORT)
   - Multi-instrument fusion: spot/futures basis, funding spread, OI delta, BVOL state
   - Outputs: Dataset/processed/{tick_bars,volume_bars,microstructure,synthetic,labels,multi_instrument}

3. **Phase 3: RL Training with Curriculum** (~10 min)
   - Triggered only if Phase 1 downloaded new files
   - Hierarchical RL environment training
   - Outputs: models/checkpoints/{phase3_*}

4. **Phase 4: Full 18-Model Sweep** (~2-4 hours)
   - Trains all 18 models (6 archetypes × 3 variants):
     - Trend Followers: LSTM, Transformer, TCN
     - Mean Reversion: MLP, ResNet, GRN
     - Scalping: CNN, LinearAttn, GRU
     - Stat Arb: Autoencoder, GAT, LSTM
     - Discretionary: ViT, Multimodal, CNNChart
     - Market Making RL: PPO, SAC, DQN
   - Runs in parallel on GPU (DirectML backend)
   - Outputs: models/checkpoints/{phase4_*}, model_registry.json

5. **Strict Gate Enforcement & Retry Loop** (~1-3 hours per retry)
   - Checks each model: Sharpe > 1.0 AND Accuracy > 52%
   - Auto-retrains failing archetypes with tuned hyperparameters
   - Max 3 retry rounds per archetype
   - Stops when all 18 models pass or max retries exhausted
   - **Updates from AMD RX 6750 as primary accelerator (DirectML)**

6. **Report Generation**
   - Outputs: MULTI_INSTRUMENT_STRICT_PASS_REPORT.md, RUN_USER_GUIDE_MULTI_INSTRUMENT.md

---

## 3) Key Output Artifacts

### Phase 1
```
Dataset/multi_instrument/
  ├── PHASE1_MULTI_INSTRUMENT_SUMMARY.json     # Download stats (downloaded, skipped, failed)
  └── common_timestamp_index.parquet           # Synchronized timestamp index across symbols
```

### Phase 2
```
Dataset/processed/
  ├── PHASE2_SUMMARY.txt                      # Feature engineering summary
  ├── tick_bars/BTCUSDT_tick_*.parquet        # 1000-trade bars for all assets
  ├── volume_bars/BTCUSDT_volume_*.parquet    # 100 BTC volume bars
  ├── microstructure/BTCUSDT_ofi_vpin_*.parquet
  ├── synthetic/                               # GARCH paths, HMM regimes
  ├── labels/BTCUSDT_triple_barrier_*.parquet  # LONG/FLAT/SHORT labels
  └── multi_instrument/
      └── BTCUSDT_multi_instrument_state.parquet # Fused spot/futures/options features
```

### Phase 3
```
models/checkpoints/
  └── phase3_*.pt                              # RL agent checkpoints
```

### Phase 4 & Final
```
model_registry.json                            # Full model metadata (Sharpe, Accuracy, status)
model_performance_summary.md                   # Human-readable results
MULTI_INSTRUMENT_STRICT_PASS_REPORT.md         # This report (auto-updated)
RUN_USER_GUIDE_MULTI_INSTRUMENT.md             # This guide
```

---

## 4) Monitoring Progress

### While Running

```powershell
# Watch logs (live)
Get-Content -Path "Dataset\processed\PHASE2_SUMMARY.txt" -Tail 20 -Wait

# Check active Python processes
Get-Process python

# View model registry updates (real-time)
Get-Item model_registry.json | Select-Object LastWriteTime
Get-Content model_registry.json | ConvertFrom-Json | Select-Object -First 5
```

### After Completion

```powershell
# Count models passing strict gates
$reg = Get-Content model_registry.json | ConvertFrom-Json
$passing = $reg | Where-Object {
    $_.validation.test_sharpe -gt 1.0 -and
    $_.validation.test_directional_accuracy -gt 0.52 -and
    $_.validation.status -eq "PASSED"
}
Write-Host "Passing models: $($passing.Count) / 18"

# View Phase 1 summary
Get-Content Dataset\multi_instrument\PHASE1_MULTI_INSTRUMENT_SUMMARY.json | ConvertFrom-Json
```

---

## 5) Strict KPI Gates

Every model is evaluated on:
- **Sharpe Ratio** > 1.0 (risk-adjusted returns)
- **Directional Accuracy** > 52% (better than coin flip)
- **Status** = "PASSED" (no errors in training/testing)

If any model fails, it is auto-retuned up to 3 times:
- Learning rate: 0.8x per round
- Batch size: 0.8x per round  
- Epochs: +4 per round
- Layers: +1 per round
- Hidden size: +32 per round

All training uses **AMD RX 6750 DirectML** as the primary backend.

---

## 6) Resume or Debug

### Re-run From Current State
```powershell
# Automatically picks up from latest checkpoint/config
d:/kp_ai_agent/ChatTrader.KPai/.venv/Scripts/python.exe execute_multi_instrument_upgrade.py --days 365 --concurrency 16
```

### Check Current Model Performance
```powershell
$reg = Get-Content model_registry.json | ConvertFrom-Json
$reg | Where-Object { $_.validation.test_sharpe -gt 1.0 } | Format-Table architecture_name, @{N="Sharpe"; E={ $_.validation.test_sharpe }}, @{N="Acc"; E={ $_.validation.test_directional_accuracy }}
```

### Manual Phase Execution
```powershell
# Phase 1 only
python execute_phase1_multi_instrument.py --days 365 --concurrency 16

# Phase 2 only (if new data exists)
python execute_phase2_feature_engineering.py

# Phase 3 only (if new data exists)
python execute_phase3_rl_training.py

# Phase 4 sweep
python tools/run_full_phase4_sweep.py

# Phase 4 finalization (aggregate results)
python tools/finalize_phase4_results.py
```

---

## 7) GPU / Hardware Notes

### AMD RX 6750 Configuration
- **Backend**: DirectML (torch-directml package)
- **Auto-Enabled**: Phase 4 configs force `preferred_backend: directml`
- **Performance**: ~2x faster than CPU for 18-model parallel training

### Verify GPU Usage
```powershell
# During training, check GPU load
Get-Process python | Measure-Object

# If DirectML not detected, install
pip install torch-directml --upgrade
```

### Fallback to CPU
If DirectML unavailable, training falls back to CPU automatically (slower, ~4-6 hours for full sweep).

---

## 8) Troubleshooting

### Issue: "Phase 1 timed out"
- Normal for 365-day + 18 symbols (~15K tasks)
- Background process continues; check progress with:
  ```powershell
  Get-Item Dataset\multi_instrument\PHASE1_MULTI_INSTRUMENT_SUMMARY.json | Select-Object LastWriteTime
  ```
- Re-run same command to continue from latest state

### Issue: "Strict pass not achieved after 3 retries"
- Model limits may be inherent to data/architecture
- Edit configs manually:
  ```powershell
  notepad configs\trend_phase4.yaml
  ```
- Increase `max_epochs`, `hidden_size`, or `num_layers`
- Re-run orchestrator to apply tuned configs

### Issue: "GPU memory error"
- Reduce Phase 4 parallel load:
  - Edit `tools/run_full_phase4_sweep.py`, lower `MAX_PARALLEL_MODELS`
  - Or reduce `batch_size` in phase4 YAML configs

### Issue: "Missing BVOL or funding data"
- Some futures daily data unavailable on source
- Script falls back to monthly aggregates automatically
- No user action required

---

## 9) Expected Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1 (Data) | 30-60 min | Download 15K+ files, validate SHA256 |
| Phase 2 (Features) | 5 min | Only if Phase 1 new downloads |
| Phase 3 (RL) | 10 min | Only if Phase 1 new downloads |
| Phase 4 (Sweep) | 2-4 hours | 18 models parallel on DirectML |
| Strict Gate + Retry | 0-9 hours | 0-3 rounds × 6 archetypes |
| **Total (Full Run)** | **3-14 hours** | Depends on Phase 1 + retry need |

---

## 10) Contact & Support

For issues or feature requests:
1. Check `Dataset/processed/PHASE2_SUMMARY.txt` for feature engineering status
2. Inspect `model_registry.json` for training metrics
3. Review logs in `models/tensorboard/` (TensorBoard summaries)
4. Examine failed checkpoints in `models/checkpoints/`

---

**Generated**: 2026-04-26T17:20:00Z  
**Version**: Multi-Instrument Upgrade v1.0  
**GPU Target**: AMD RX 6750 DirectML  
**Data Window**: 365 days (2025-04-26 to 2026-04-26)
