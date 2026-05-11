# SYSTEM INTEGRATION & EXECUTION CHECKLIST

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     AUTO-GENERATED SCHEDULE SYSTEM              │
│                                                                  │
│  Input Data:                                                     │
│  ├─ Benchmarks: 6 archetypes × 2 backends (CPU/DirectML)        │
│  ├─ Configs: 6 archetype config files with max_epochs           │
│  └─ Convergence Factors: per-archetype difficulty multipliers   │
│                                                                  │
│  Processing:                                                     │
│  └─ Generate ETA = (smoke_time × epochs) × convergence_factor   │
│                                                                  │
│  Outputs:                                                        │
│  ├─ training_schedule.yaml                                      │
│  ├─ Schedule ASCII table with timeline                          │
│  ├─ First-pass and all-pass time estimates                      │
│  └─ Pass probability per archetype                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓ (YAML)
┌─────────────────────────────────────────────────────────────────┐
│                    LIVE MONITOR (Continuous)                    │
│                                                                  │
│  Reads:                                                          │
│  ├─ training_schedule.yaml (predicted timeline)                 │
│  ├─ working_log.md (live epoch/episode metrics from trainers)   │
│  └─ Current system time                                         │
│                                                                  │
│  Calculates:                                                     │
│  ├─ Progress per model (% complete)                             │
│  ├─ Speed metrics (samples/sec or episodes/min)                 │
│  ├─ Elapsed time vs predicted time                              │
│  ├─ ETA drift = (actual / predicted - 1) × 100%                 │
│  └─ Backend speed recommendations                               │
│                                                                  │
│  Displays (every 10 seconds):                                    │
│  ├─ Live Training Progress (18 models)                          │
│  ├─ Schedule & ETA Tracking (6 archetypes)                      │
│  ├─ Completion Timeline                                         │
│  └─ Backend Speed Analysis                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
         ↑ (monitors)             ↓ (launches)
         │                        │
    working_log.md       ┌─────────────────────────────────────────┐
    (appended by         │       TRAINING EXECUTOR (Sequential)    │
     trainers)           │                                          │
                         │  For each archetype in order:           │
                         │  1. Patch config with optimal backend   │
                         │  2. Launch trainer module               │
                         │  3. Capture output                      │
                         │  4. Wait for completion                 │
                         │  5. Record elapsed time                 │
                         │                                          │
                         │  Trainer behavior:                       │
                         │  ├─ Load 34 symbols (50k-900k rows)    │
                         │  ├─ Train 3 models in parallel          │
                         │  ├─ Append progress to working_log      │
                         │  └─ Save checkpoints                    │
                         │                                          │
                         └─────────────────────────────────────────┘
                                    ↓ (checks)
                         ┌─────────────────────────────────────────┐
                         │    EVALUATION (After all archetypes)    │
                         │                                          │
                         │  Run: evaluate_all_checkpoints.py       │
                         │  Check: model_registry.json             │
                         │  Gates: Sharpe>1.2, PF>1.5, MDD<0.2    │
                         │  Output: Pass/Fail status per model     │
                         │                                          │
                         └─────────────────────────────────────────┘
                                    ↓ (if failures)
                         ┌─────────────────────────────────────────┐
                         │        RETRY LOOP (Automated)           │
                         │                                          │
                         │  • Identify failed models               │
                         │  • Adjust hyperparameters               │
                         │  • Regenerate schedule for failures     │
                         │  • Re-execute training                  │
                         │  • Re-evaluate                          │
                         │  • Repeat until all 18 pass             │
                         │                                          │
                         └─────────────────────────────────────────┘
```

## Step-by-Step Setup & Execution

### Step 1: Verify Prerequisites
```bash
# Check Python environment
python --version  # Should be 3.9+
pip list | grep -E "torch|yaml|numpy|pandas"  # Verify key packages

# Check config files exist
ls configs/*_phase4.yaml  # Should list 6 files

# Check data directory
ls Dataset/binance_historical/manifest.json  # Should exist
```

### Step 2: Generate Schedule (5 minutes)
```bash
cd d:\kp_ai_agent\ChatTrader.KPai

python tools/training_schedule_generator.py \
    --start-time "2026-05-09T06:00:00" \
    --output doc/training_schedule.yaml

# Output:
# ✓ Generates doc/training_schedule.yaml
# ✓ Prints 6-archetype schedule with times
# ✓ Shows first-pass and all-pass estimates
```

### Step 3: Validate Schedule
```bash
# Check schedule was created
cat doc/training_schedule.yaml | head -20

# Verify it shows:
# • All 6 archetypes
# • 3 models per archetype
# • Backend assignment (CPU or DirectML)
# • Start/end times
# • Expected pass probability
```

### Step 4: Launch Monitor (Terminal 2)
```bash
# In a second terminal, go to same directory
cd d:\kp_ai_agent\ChatTrader.KPai

# Start live monitor (refreshes every 10 seconds)
python tools/training_monitor_with_schedule.py

# Monitor will show:
# ✓ Live progress table (18 models)
# ✓ Schedule timeline
# ✓ Completion estimates
# Note: Will show "No training data yet" until Step 5 launches

# To see once and exit:
python tools/training_monitor_with_schedule.py --once
```

### Step 5: Execute Training (Terminal 1)
```bash
# Back in original terminal
python tools/training_schedule_executor.py \
    --schedule doc/training_schedule.yaml

# What happens:
# • Reads schedule YAML
# • For each archetype (in order):
#   1. Patches config with best backend (CPU or DirectML)
#   2. Launches: python -m quant_core.train_*_phase4 --config ...
#   3. Trainer loads 34 symbols
#   4. Trainer starts 3 models in parallel
#   5. Each model appends progress to working_log.md
#   6. Monitor reads these appends every 10 seconds
# • When archetype completes, moves to next
# • Produces execution summary at end

# Expected output:
# ✓ discretionary: 36.2 minutes
# ✓ market_maker:  13.9 minutes
# ✓ mean_reversion: 16.1 minutes
# ✓ scalper:       32.2 minutes
# ✓ stat_arb:      34.8 minutes
# ✓ trend:         77.5 minutes
# Total: 3.51 hours
```

### Step 6: Monitor Progress (Terminal 2)
```
Terminal 2 will continuously show:

[Every 10 seconds updates...]

LIVE TRAINING PROGRESS
├─ 18 models with epoch count and backend
├─ Speed metrics (samples/sec)
└─ Elapsed time per model

SCHEDULE & ETA TRACKING
├─ Discretionary: 36.2m predicted, 54.2m actual → 50% SLOWER ✗
├─ Market Maker:  13.9m predicted, 14.0m actual → 1% slower (OK)
├─ [continues for all archetypes...]
└─ Shows which are RUNNING, PENDING, or COMPLETED

COMPLETION TIMELINE
├─ Estimated First Pass: 07:24
├─ Estimated All-Pass: 09:24
└─ Execution Mode: SEQUENTIAL

BACKEND SPEED ANALYSIS
└─ Shows actual vs predicted speed, recommendations
```

### Step 7: After Training Completes (3.5 hours)
```bash
# When executor finishes all 6 archetypes
# Run evaluation
python evaluate_all_checkpoints.py

# Output:
# • model_registry.json updated
# • Pass/Fail status for each model
# • Sharpe, PF, MDD, DirAcc metrics
# • Summary: X/18 PASS

# If < 18/18 pass:
# • Regenerate schedule for failed models
# • Run executor again on failures
# • Re-evaluate

# Repeat until all 18 pass
```

## Configuration Reference

### Benchmark Data (Measured 1-epoch timings)
```yaml
Trend:        DirectML=20.66s CPU=21.11s   → Use DirectML (1% faster)
Mean Reversion: CPU=4.60s    DirectML=4.88s → Use CPU (6% faster)
Scalper:      CPU=10.07s     DirectML=14.75s → Use CPU (46% faster)
Stat Arb:     CPU=13.38s     DirectML=20.93s → Use CPU (56% faster)
Discretionary: DirectML=6.03s CPU=7.15s     → Use DirectML (19% faster)
Market Maker: CPU=17.37s     DirectML=32.53s → Use CPU (87% faster)
```

### Convergence Factors (Difficulty multipliers)
```yaml
Trend:         1.5  # Strong convergence, strict gate (Sharpe>1.2)
Mean Reversion: 1.4  # Quick convergence, noise-sensitive
Scalper:       1.6  # Very tight gate, may need tuning
Stat Arb:      1.3  # Close to passing historically
Discretionary: 1.8  # Hardest gate (F1>0.65), 200 epochs
Market Maker:  1.2  # RL learns fast
```

### Expected Timing Formulas
```
For Supervised Models:
  ETA = smoke_time_seconds × max_epochs × convergence_factor
  Example Trend: 20.66s × 150 × 1.5 = 4648.5s ≈ 77.5 minutes

For RL Models (Market Maker):
  ETA = smoke_time_seconds × (full_episodes / smoke_episodes) × convergence_factor
  Example MM: 17.37s × (8000 / 200) × 1.2 = 833.76s ≈ 13.9 minutes
```

## Files Checklist

### Created Files
- [x] tools/training_schedule_generator.py (394 lines)
- [x] tools/training_monitor_with_schedule.py (312 lines)
- [x] tools/training_schedule_executor.py (398 lines)
- [x] tools/quick_start_training.py (141 lines)
- [x] doc/training_schedule.yaml (generated)
- [x] doc/QUICK_REFERENCE.md (this file's sibling)
- [x] doc/TRAINING_SCHEDULE_GUIDE.md (600+ lines)
- [x] doc/SCHEDULE_IMPLEMENTATION_SUMMARY.md
- [x] doc/EXECUTION_WALKTHROUGH.md (800+ lines)
- [x] doc/SESSION_COMPLETION_SUMMARY.md

### Existing Files (Used)
- configs/discretionary_phase4.yaml
- configs/market_maker_phase4.yaml
- configs/mean_reversion_phase4.yaml
- configs/scalper_phase4.yaml
- configs/stat_arb_phase4.yaml
- configs/trend_phase4.yaml
- doc/training_more_27-4/27-04-2026_plan_REVISED_workingLog.md
- quant_core/train_*_phase4.py (6 trainer modules)
- evaluate_all_checkpoints.py

## Failure Recovery

### If Training Crashes
```bash
# Restart training (auto-resumes from checkpoint)
python tools/training_schedule_executor.py --schedule doc/training_schedule.yaml

# Or just the failed archetype
python -m quant_core.train_ARCHETYPE_phase4 --config configs/ARCHETYPE_phase4.yaml
```

### If Model Fails Gates
```bash
# After evaluation shows failures:
# 1. Adjust hyperparameters in config (lr, dropout, patience, etc.)
# 2. Regenerate schedule for failed archetypes only
# 3. Re-execute and re-evaluate

python tools/training_schedule_generator.py --output doc/training_schedule_retry.yaml
python tools/training_schedule_executor.py --schedule doc/training_schedule_retry.yaml
python evaluate_all_checkpoints.py
```

### If Monitor Shows Wrong Times
```bash
# Regenerate schedule with correct start time
python tools/training_schedule_generator.py \
    --start-time "2026-05-09T10:00:00" \
    --output doc/training_schedule.yaml

# Restart monitor
python tools/training_monitor_with_schedule.py
```

## Verification Checklist (Before Running)

- [ ] Schedule generated: `ls doc/training_schedule.yaml`
- [ ] Schedule contains 6 archetypes: `grep "^  [a-z_]*:" doc/training_schedule.yaml | wc -l` (should be 6)
- [ ] All config files exist: `ls configs/*_phase4.yaml` (should list 6)
- [ ] Working log path exists: `ls doc/training_more_27-4/`
- [ ] Trainers importable: `python -c "from quant_core import train_trend_phase4"` (no error)
- [ ] Monitor can import: `python -c "import yaml"` (should succeed)

## Performance Monitoring

### Key Metrics to Watch
```
Samples/sec (supervised models):
  • Target: 400-1200 samp/s
  • If < 200: Possible memory pressure, batch cap in effect
  • If > 1500: Excellent performance

Episodes/min (RL models):
  • Target: 100-150 ep/min
  • If < 50: Possible training instability
  • If > 200: Excellent performance

ETA Drift:
  • ±20%: Normal, on-track
  • -50%: Running 50% faster, ahead of schedule (good!)
  • +100%: Running 100% slower, behind schedule (may need tuning)

Pass Rate on First Attempt:
  • Target: 49% pessimistic, 72% realistic, 80% optimistic
  • If > 70%: Ahead of expected, may pass all in 4-5 hours
  • If < 40%: Behind, may need 8+ hours with retries
```

## Next Steps

1. ✓ Review this file and QUICK_REFERENCE.md
2. ✓ Run: `python tools/training_schedule_generator.py`
3. ✓ Verify: `cat doc/training_schedule.yaml | head -30`
4. Launch monitor: `python tools/training_monitor_with_schedule.py` (Terminal 2)
5. Execute: `python tools/training_schedule_executor.py` (Terminal 1)
6. Watch monitor for progress (Terminal 2)
7. Wait for completion (3.5 hours)
8. Evaluate: `python evaluate_all_checkpoints.py`
9. If needed, repeat steps 2-8 for retry loop

---

**You're ready to go!** This system will guide you through training all 18 models to completion.
