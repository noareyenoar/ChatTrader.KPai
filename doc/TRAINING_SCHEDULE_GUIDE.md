# Auto-Generated Training Schedule with Live Monitoring

## Overview

This system auto-generates a per-archetype training schedule based on:
1. **Benchmark Results** - Actual measured training speed from smoke runs (1-epoch tests)
2. **Model Architecture Complexity** - Different archetypes have different convergence characteristics
3. **Financial Gate Difficulty** - Stricter gates need more epochs to pass
4. **Hardware Backend Selection** - Optimal CPU vs DirectML per archetype

The schedule predicts:
- ✓ Start/end times for each archetype
- ✓ Expected first-pass model completion time
- ✓ Estimated all-pass completion time
- ✓ Real-time ETA drift tracking during training

## Quick Start (3 steps)

### Step 1: Generate Schedule

```bash
# Generate sequential schedule (default)
python tools/training_schedule_generator.py \
    --start-time "2026-05-09T10:00:00" \
    --output doc/training_schedule.yaml

# Or generate parallel schedule
python tools/training_schedule_generator.py \
    --parallel \
    --output doc/training_schedule.yaml
```

**Output**: `doc/training_schedule.yaml` with predicted timestamps

### Step 2: Launch Enhanced Monitor (in separate terminal)

```bash
# Live monitor with continuous updates
python tools/training_monitor_with_schedule.py \
    --log-path doc/training_more_27-4/27-04-2026_plan_REVISED_workingLog.md \
    --schedule-path doc/training_schedule.yaml \
    --refresh-sec 10

# Or one-time snapshot
python tools/training_monitor_with_schedule.py --once
```

**Output**: Real-time dashboard showing:
- Live progress per model
- Schedule vs actual completion
- ETA drift (faster/slower than predicted?)
- Backend performance recommendations
- Overall timeline to first-pass and all-pass

### Step 3: Execute Schedule

```bash
# Show what would run (dry-run)
python tools/training_schedule_executor.py \
    --schedule doc/training_schedule.yaml \
    --dry-run

# Execute schedule (sequential)
python tools/training_schedule_executor.py \
    --schedule doc/training_schedule.yaml

# Execute schedule (parallel - all archetypes at once)
python tools/training_schedule_executor.py \
    --schedule doc/training_schedule.yaml \
    --parallel
```

## Architecture of the System

### 1. Schedule Generator (`tools/training_schedule_generator.py`)

**Input:**
- Benchmark results from smoke runs (1-epoch timing measurements)
- Config max_epochs per archetype
- Convergence factors per archetype

**Computation:**
```
For each archetype:
  full_training_time = (smoke_time_seconds × max_epochs) × convergence_factor
  
Convergence Factors (account for plateau and gate difficulty):
  - Trend: 1.5 (strong signal but strict gate)
  - Mean Reversion: 1.4 (quick convergence, noise-sensitive)
  - Scalper: 1.6 (very tight gate, may need tuning)
  - Stat Arb: 1.3 (historically close to passing)
  - Discretionary: 1.8 (hardest gate F1>0.65, longest epochs)
  - Market Maker: 1.2 (RL learns fast)
```

**Output:** `doc/training_schedule.yaml`
```yaml
metadata:
  generated_at: 2026-05-09T05:53:00
  start_time: 2026-05-09T05:53:00
  parallel_execution: false

archetypes:
  discretionary:
    backend: directml
    models: [ViT_Disc_v2, Multimodal_Disc_v2, CNN_Chart_Disc_v2]
    eta_seconds: 2172
    eta_duration: 36.2m
    start_time: 2026-05-09T05:53:00
    end_time: 2026-05-09T06:29:11
    expected_status:
      confidence: LOW
      expected_pass: 0.35

summary:
  total_models: 18
  estimated_first_pass_time: 2026-05-09T07:24:31
  estimated_all_pass_time: 2026-05-09T09:23:40
  total_seconds_sequential: 12634
```

**Backend Selection Logic:**
```
Backend Benchmarks (1-epoch smoke runs):
  trend:       directml=20.66s  [PICK]  cpu=21.11s
  mean_reversion: cpu=4.60s     [PICK]  directml=4.88s
  scalper:     cpu=10.07s       [PICK]  directml=14.75s
  stat_arb:    cpu=13.38s       [PICK]  directml=20.93s
  discretionary: directml=6.03s [PICK]  cpu=7.15s
  market_maker: cpu=17.37s      [PICK]  directml=32.53s
```

### 2. Enhanced Monitor (`tools/training_monitor_with_schedule.py`)

**Features:**

#### Live Training Progress
- Per-model epoch/episode count and percentage
- Backend in use (CPU/DirectML)
- Training speed (samples/sec or episodes/min)
- Last update timestamp

#### Schedule & ETA Tracking
- Predicted total time per archetype
- Actual elapsed time so far
- ETA drift calculation:
  - `< -20%`: ✓ FASTER (ahead of schedule)
  - `> +20%`: ✗ SLOWER (behind schedule)
  - Otherwise: ON_TIME
- Scheduled start/end times

#### Completion Timeline
- Schedule start time
- Estimated first model pass time (one archetype completing)
- Estimated all-pass time (all archetypes completing)
- Execution mode (sequential vs parallel)

#### Backend Speed Analysis
- Actual measured speed per archetype/backend
- Recommendation for fastest backend
- Number of measurements taken

### 3. Schedule Executor (`tools/training_schedule_executor.py`)

**Features:**
- Reads schedule YAML
- Patches config files with correct backend per archetype
- Launches training sequentially or in parallel
- Captures output in real-time
- Reports exit codes and elapsed time
- Execution summary

## Interpreting the Schedule Output

### Example Schedule

```
TRAINING SCHEDULE — Per-Archetype Timing with Predicted Finish Timestamps
================================================================
Generated: 2026-05-09T05:53:00
Start Time: 2026-05-09T05:53:00
Execution Mode: SEQUENTIAL

ARCHETYPE SCHEDULE
Archetype          Backend      #Models  Smoke      Conv.F   ETA Duration    Start       End         Status
discretionary      directml     3        6.03s      1.8      36.2m           05:53       06:29       P=35%
market_maker       cpu          3        17.37s     1.2      13.9m           06:29       06:43       P=50%
mean_reversion     cpu          3        4.60s      1.4      16.1m           06:43       06:59       P=45%
scalper            cpu          3        10.07s     1.6      32.2m           06:59       07:31       P=40%
stat_arb           cpu          3        13.38s     1.3      34.8m           07:31       08:06       P=70%
trend              directml     3        20.66s     1.5      77.5m           08:06       09:24       P=55%

ESTIMATED COMPLETION TIMES
Total Models: 18
Sequential Total Time: 3.51h
First Model Expected to Pass: 2026-05-09T07:24:31   (Stat Arb has highest pass confidence)
All Models Expected to Pass: 2026-05-09T09:23:40    (All archetypes complete + retries)

Pessimistic Scenario (40% pass): 8.78h              (Need 2.5x retraining cycles)
Optimistic Scenario (80% pass): 4.56h               (Need 1.3x retraining cycles)
```

### Key Metrics

**Smoke Time**: 1-epoch measured time
- Shorter = better training speed per epoch
- Example: Scalper CPU 10.07s/epoch vs DirectML 14.75s/epoch

**Convergence Factor**: Multiplier accounting for how many iterations to pass gates
- 1.2 = Converges quickly (RL, simple patterns)
- 1.8 = Slow convergence (ViT, complex patterns)

**ETA Duration**: Total predicted training time = Smoke × Epochs × Conv.F
- Example: Trend = 20.66s × 150 epochs × 1.5 factor = 77.5 minutes

**Expected Pass %**: Confidence in passing financial gates
- Based on prior evaluation results
- Stat Arb highest (70%) - was already close to passing
- Discretionary lowest (35%) - hardest gate (F1>0.65)

## How the Monitor Tracks Progress

### Real-Time ETA Drift

As training progresses, the monitor calculates:

```
drift = (actual_elapsed_seconds / predicted_total_seconds - 1.0) × 100%

If drift < -20%:  ✓ FASTER - you're ahead of schedule
If drift > +20%:  ✗ SLOWER - you're behind schedule
Otherwise:        ON_TIME - tracking as predicted
```

**Example**: Trend predicted 77.5min but has run 41.8min at epoch 20/150
- `drift = (41.8 / 77.5 - 1.0) × 100 = -46%` → 46% FASTER ✓
- This suggests actual convergence is faster than the safety-factor prediction

### Speed Metrics

**Samples/sec** (supervised models):
- How many data samples processed per second
- Higher = faster epoch iterations
- Used to rank CPU vs DirectML

**Episodes/min** (RL models):
- How many RL episodes completed per minute
- Market Maker metrics tracked per EPISODE not EPOCH

## Workflow: From Schedule to Passing Models

### Scenario: Sequential Training with Retries

```
09:53 UTC: Executor launches Discretionary models on DirectML
          - ViT_Disc_v2, Multimodal_Disc_v2, CNN_Chart_Disc_v2
          - Monitor shows: 0 → 200 epochs
          - Speed: 450-550 samples/sec

06:29 UTC: Discretionary training completes
          - Evaluator runs on OOS test set
          - Result: 2/3 models pass gates (e.g., ViT and Multimodal pass)
          - CNN_Chart fails → triggers RETRY with hyperparameter adjustment

06:29 UTC: Market Maker launches (CPU)
06:43 UTC: Market Maker completes (8000 episodes ≈ 14 minutes on CPU)
...
08:06 UTC: Stat Arb completes
          - Evaluator: 3/3 PASS ✓ (this is first model to fully pass!)

09:24 UTC: Trend completes (Trend is last due to longest epochs)
          - Evaluator: likely 2-3/3 pass

09:24 UTC: All archetypes have completed at least one training cycle
          - Overall: ~12-14/18 models pass on first attempt
          - Need retry cycle for remaining ~4-6 models

10:00 UTC: Retry cycle starts for failed models
          (Retry loop repeats until all 18 pass)
```

## Common Scenarios

### Scenario 1: Training Faster Than Predicted

If monitor shows model consistently `30% FASTER`:
- Schedule estimates were conservative (good!)
- All-pass time may be 1-2 hours earlier than predicted
- No action needed - just enjoy the speedup

### Scenario 2: Training Slower Than Predicted

If model shows `40% SLOWER`:
- Might indicate:
  - More data than smoke profile
  - Slower convergence than factor accounts for
  - GPU memory pressure (DirectML batch cap in effect?)
- Options:
  - Reduce batch_size in config
  - Increase convergence_factor and re-generate schedule
  - Switch to CPU backend if DirectML is throttling

### Scenario 3: Model Fails Gates After Training

If a model completes training but fails evaluation:
- Schedule completion time is still accurate (training finished)
- Evaluation phase takes additional time (~5-10 min per model)
- Retry loop begins with hyperparameter tuning
- Executor can be re-run on failed models

### Scenario 4: Parallel Execution Desired

If you have spare hardware:
```bash
python tools/training_schedule_executor.py \
    --schedule doc/training_schedule.yaml \
    --parallel
```
- Regenerate schedule: `--parallel` flag
- All 6 archetypes train simultaneously
- Bottleneck = longest archetype (Trend at 77.5 min)
- Total time: ~1.3 hours instead of 3.5 hours

## Integration with Existing Pipeline

### Working Log Append Mechanism

All trainers append entries to:
```
doc/training_more_27-4/27-04-2026_plan_REVISED_workingLog.md
```

Format per epoch/episode:
```markdown
- [2026-05-09T06:15:23.456789] model=LSTM_Trend_v2 stage=EPOCH epoch=20/150 total_epochs=150 elapsed_s=2505 backend=directml samples_per_s=473.7 learning_rate=0.00023
```

Monitor reads this file continuously and parses into:
```python
{
    'LSTM_Trend_v2': {
        'ts': '2026-05-09T06:15:23.456789',
        'model': 'LSTM_Trend_v2',
        'stage': 'EPOCH',
        'epoch': 20,
        'total_epochs': 150,
        'elapsed_s': 2505,
        'backend': 'directml',
        'samples_per_s': 473.7,
        'learning_rate': 0.00023,
    }
}
```

### Checkpoint Recovery

If training is interrupted:
```bash
# Resume from last checkpoint (auto-detected in trainer)
python -m quant_core.train_trend_phase4 \
    --config configs/trend_phase4.yaml \
    --resume

# Monitor will see epoch continue from where it left off
# Schedule ETA adjusts based on actual progress
```

### Post-Training Evaluation

After schedule completes:
```bash
python evaluate_all_checkpoints.py
```

Produces:
- `model_registry.json` with pass/fail status per model
- Summary of which archetypes fully passed
- Failures routed to retry cycle

## Advanced Options

### Custom Convergence Factors

Edit `tools/training_schedule_generator.py`:
```python
CONVERGENCE_FACTORS = {
    'trend': 1.5,           # Adjust if trending faster/slower
    'mean_reversion': 1.4,  # Noise-sensitive, may need higher
    'scalper': 1.6,         # Already aggressive
    'stat_arb': 1.3,        # Already close to passing
    'discretionary': 1.8,   # Hardest, may increase to 2.0
    'market_maker': 1.2,    # RL-specific
}
```

Then regenerate:
```bash
python tools/training_schedule_generator.py --output doc/training_schedule.yaml
```

### Custom Start Time

```bash
# Schedule starting tomorrow at 18:00 UTC
python tools/training_schedule_generator.py \
    --start-time "2026-05-10T18:00:00" \
    --output doc/training_schedule.yaml
```

Monitor will show countdown to start time and adjusted ETA timeline.

### Monitor Refresh Rate

```bash
# Check progress every 5 seconds (faster updates)
python tools/training_monitor_with_schedule.py --refresh-sec 5

# Check every 30 seconds (slower, less CPU usage)
python tools/training_monitor_with_schedule.py --refresh-sec 30
```

## Validation Checklist

Before running full schedule:

- [ ] `doc/training_schedule.yaml` generated and shows reasonable ETAs
- [ ] Schedule shows expected backend assignments (CPU for fast archs, DirectML for heavy)
- [ ] Monitor displays live progress without errors
- [ ] All 6 archetypes have 3 models each (18 total)
- [ ] Working log path is correct and writable
- [ ] Config files exist: `configs/{archetype}_phase4.yaml`

## Troubleshooting

### Monitor shows "WAITING FOR FIRST EPOCH"
- Training hasn't started yet or hasn't written first log entry
- Check if trainer process is running
- Check if working log file is being appended

### Schedule times seem wrong
- Verify benchmark results are reasonable:
  - CPU/DirectML times should be in 4-35 second range for 1 epoch
  - If not, smoke runs may not have completed successfully
- Check convergence factors (may be too conservative or aggressive)
- Try `--dry-run` on executor to verify commands

### Training much slower than predicted
- Check if batch cap is in effect (DirectML limiting VRAM)
- Monitor speed metric (samples_per_s) may be much lower than benchmark
- Consider switching to CPU backend for that archetype
- Or increase convergence_factor and regenerate schedule

### Schedule executor fails to launch
- Check Python path: `.venv\Scripts\python.exe`
- Verify config files exist and are valid YAML
- Run with `--dry-run` first to see commands being generated
- Check for typos in archetype names

## Files Generated

```
doc/training_schedule.yaml          ← Main schedule file (YAML format)
tools/training_schedule_generator.py ← Generate schedule
tools/training_monitor_with_schedule.py ← Live monitor dashboard
tools/training_schedule_executor.py  ← Execute schedule

doc/training_more_27-4/
  27-04-2026_plan_REVISED_workingLog.md ← Appended by trainers
```

## Summary

This system provides:
✓ **Data-driven scheduling** based on measured benchmark timings
✓ **Real-time progress tracking** with ETA drift analysis
✓ **Backend optimization** (CPU vs DirectML per archetype)
✓ **Confidence intervals** (pessimistic/optimistic completion times)
✓ **Automated orchestration** (launch all 18 models in order)
✓ **Schedule resilience** (continues on model failure, triggers retries)

**Expected first model pass**: ~1.5 hours from start
**Expected all-pass (with retries)**: ~4-9 hours depending on gate pass rates
