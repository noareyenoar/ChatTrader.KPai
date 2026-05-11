# AUTO-GENERATED TRAINING SCHEDULE — QUICK REFERENCE

## 📋 System Overview

A complete training orchestration system that:
1. **Generates** per-archetype schedules with predicted timestamps
2. **Monitors** real-time training progress with ETA tracking
3. **Executes** sequential training of all 18 models automatically

## 🚀 Quick Start (Copy & Paste)

### Terminal 1: Generate Schedule & Execute Training
```bash
cd d:\kp_ai_agent\ChatTrader.KPai

# Option A: All-in-one command (asks for confirmation between steps)
python tools/quick_start_training.py

# Option B: Individual commands (more control)
python tools/training_schedule_generator.py --output doc/training_schedule.yaml
python tools/training_schedule_executor.py --schedule doc/training_schedule.yaml
```

### Terminal 2 (Separate): Launch Live Monitor
```bash
cd d:\kp_ai_agent\ChatTrader.KPai

# Continuous live dashboard (refreshes every 10 seconds)
python tools/training_monitor_with_schedule.py

# Or one-time snapshot
python tools/training_monitor_with_schedule.py --once
```

## 📊 Generated Schedule

### Timeline Overview
```
START: 2026-05-09 05:53:00 UTC

05:53 → 06:29 (36m)   Discretionary  (DirectML)  3 models
06:29 → 06:43 (14m)   Market Maker   (CPU)      3 models
06:43 → 06:59 (16m)   Mean Rev       (CPU)      3 models
06:59 → 07:31 (32m)   Scalper        (CPU)      3 models
07:31 → 08:06 (35m)   Stat Arb       (CPU)      3 models
08:06 → 09:24 (78m)   Trend          (DirectML) 3 models

TOTAL: 3.51 hours (18 models trained sequentially)
```

### Estimated Completion Times
```
First Model Expected Pass: 07:24 (1.5 hours from start)
↳ Highest confidence: Stat Arb (70% expected pass)

All Models Expected Pass: 09:24 (3.5 hours from start)
↳ Plus retries: 4.5-8.8 hours depending on pass rate

Pass Rate Estimates (1st attempt):
  Pessimistic (40% pass): 8.78h total
  Optimistic (80% pass):  4.56h total
  Expected (49% pass):    6.2h total
```

## 🎯 What Each Script Does

### 1. training_schedule_generator.py
**Input:**
- Benchmark timings from smoke runs (1-epoch tests)
- Config max_epochs per archetype
- Convergence factors

**Output:**
- `doc/training_schedule.yaml` with complete timeline
- Console printout with ASCII table
- First-pass and all-pass time estimates

**Run:** `python tools/training_schedule_generator.py`

### 2. training_monitor_with_schedule.py
**Displays (updates every 10 seconds):**
- Live training progress per model (epoch/episode count)
- Backend in use (CPU/DirectML)
- Training speed (samples/sec or episodes/min)
- Schedule timeline vs actual progress
- ETA drift analysis (faster/slower than predicted?)
- Backend speed recommendations

**Run:** `python tools/training_monitor_with_schedule.py`

### 3. training_schedule_executor.py
**Does:**
- Reads schedule YAML
- For each archetype:
  1. Patches config with optimal backend
  2. Launches trainer module
  3. Captures output in real-time
  4. Records elapsed time
- Prints execution summary

**Run:** `python tools/training_schedule_executor.py --schedule doc/training_schedule.yaml`

## 📈 Monitor Dashboard Example

```
╔════════════════════════════════════════════════════════════════════════════╗
║ TRAINING PROGRESS MONITOR WITH SCHEDULE TRACKING                          ║
╚════════════════════════════════════════════════════════════════════════════╝
Current Time: 2026-05-09T06:15:30

┌─ LIVE TRAINING PROGRESS ──────────────────────────────────────────────────┐
│ Archetype      Model              Stage     Progress      Backend   Speed  │
├──────────────────────────────────────────────────────────────────────────┤
│ discretionary  ViT_Disc_v2        EPOCH     45/200 (22%)   directml  521 s/s
│ discretionary  Multimodal_Disc_v2 EPOCH     43/200 (21%)   directml  499 s/s
│ discretionary  CNN_Chart_Disc_v2  EPOCH     41/200 (20%)   directml  512 s/s
└──────────────────────────────────────────────────────────────────────────┘

┌─ SCHEDULE & ETA TRACKING ─────────────────────────────────────────────────┐
│ Archetype      Status   Est. Total  Actual    Drift         ETA End    │
├──────────────────────────────────────────────────────────────────────────┤
│ discretionary  RUNNING  36.2m       54.2m     50% SLOWER ✗  06:29      │
│ market_maker   PENDING  13.9m       -         -             06:43      │
│ mean_reversion PENDING  16.1m       -         -             06:59      │
│ scalper        PENDING  32.2m       -         -             07:31      │
│ stat_arb       PENDING  34.8m       -         -             08:06      │
│ trend          PENDING  77.5m       -         -             09:24      │
└──────────────────────────────────────────────────────────────────────────┘

┌─ COMPLETION TIMELINE ─────────────────────────────────────────────────────┐
│ Estimated First Pass: 07:24  (one model passing all gates)
│ Estimated All-Pass:   09:24  (all models passing all gates)
│ Total Sequential Time: 3.5 hours
│ Execution Mode: SEQUENTIAL
└──────────────────────────────────────────────────────────────────────────┘
```

## 🎮 Common Workflows

### Scenario 1: Normal Training (Most Common)
```bash
# Terminal 1
python tools/quick_start_training.py

# Terminal 2 (when ready)
python tools/training_monitor_with_schedule.py

# Wait for "All Models Expected to Pass" time
# Then run evaluation
python evaluate_all_checkpoints.py
```

### Scenario 2: Dry-Run (See What Would Run)
```bash
python tools/training_schedule_executor.py --schedule doc/training_schedule.yaml --dry-run
```

### Scenario 3: Custom Start Time
```bash
# Schedule training for tomorrow 18:00 UTC
python tools/training_schedule_generator.py \
    --start-time "2026-05-10T18:00:00" \
    --output doc/training_schedule.yaml
```

### Scenario 4: Parallel Execution (All Archetypes at Once)
```bash
# Regenerate schedule for parallel mode
python tools/training_schedule_generator.py --parallel

# Execute (all 6 archetypes launch simultaneously)
python tools/training_schedule_executor.py --schedule doc/training_schedule.yaml --parallel
```

## 📋 Files Generated/Used

### Schedule System Files
```
tools/
  ├── training_schedule_generator.py       ✓ Created (394 lines)
  ├── training_monitor_with_schedule.py    ✓ Created (312 lines)
  ├── training_schedule_executor.py        ✓ Created (398 lines)
  └── quick_start_training.py              ✓ Created (141 lines)

doc/
  ├── training_schedule.yaml               ✓ Generated (YAML)
  ├── TRAINING_SCHEDULE_GUIDE.md           ✓ Created (600+ lines)
  ├── SCHEDULE_IMPLEMENTATION_SUMMARY.md   ✓ Created
  ├── EXECUTION_WALKTHROUGH.md             ✓ Created (800+ lines)
  └── SESSION_COMPLETION_SUMMARY.md        ✓ Created
```

### Existing Files (Read by System)
```
configs/*_phase4.yaml                      ← Trainer configs
doc/training_more_27-4/27-04-2026_plan_REVISED_workingLog.md ← Live logs
```

## 🔍 How to Read the Monitor Output

### Progress Column (EPOCH section)
- `45/200 (22%)` = at epoch 45 of max 200, which is 22% complete
- `8000/8000 (100%)` = at episode 8000 of max 8000 (complete)

### Speed Metrics
- `521 samp/s` = 521 samples processed per second (supervised models)
- `125 ep/m` = 125 episodes per minute (RL models like Market Maker)

### ETA Drift
- `46% FASTER ✓` = Training completed 46% faster than predicted (GOOD)
- `50% SLOWER ✗` = Training 50% slower than predicted (may need tuning)
- `0% (on-track)` = Tracking as predicted (GOOD)

### Backend Recommendation
- Shows which backend is fastest per archetype
- Example: `discretionary → directml: 510.7 samp/s`

## ⏱️ Expected Timings

### Per Archetype (from generated schedule)
| Archetype | Backend | Models | ETA | Confidence |
|-----------|---------|--------|-----|------------|
| Discretionary | DirectML | 3 | 36m | Low (35%) |
| Market Maker | CPU | 3 | 14m | Medium (50%) |
| Mean Reversion | CPU | 3 | 16m | Medium (45%) |
| Scalper | CPU | 3 | 32m | Low (40%) |
| Stat Arb | CPU | 3 | 35m | High (70%) |
| Trend | DirectML | 3 | 78m | Medium (55%) |
| **TOTAL** | - | **18** | **3.5h** | **49% avg** |

### Pass Rate Predictions
```
Optimistic (80% 1st-attempt pass): 4.5h total
Expected (49% 1st-attempt pass):   6.2h total (with retries)
Pessimistic (40% 1st-attempt pass): 8.8h total
```

## ✅ Validation Checklist

Before launching training:
- [ ] Schedule YAML generated: `doc/training_schedule.yaml`
- [ ] Schedule shows 6 archetypes × 3 models = 18 total
- [ ] Backend assignments match benchmark winners
- [ ] Times seem reasonable (hours, not days)
- [ ] All configs exist: `configs/*_phase4.yaml`
- [ ] Working log is writable: `doc/training_more_27-4/27-04-2026_plan_REVISED_workingLog.md`

## 🔗 Related Documentation

- **Full User Guide**: See `doc/TRAINING_SCHEDULE_GUIDE.md`
- **Implementation Summary**: See `doc/SCHEDULE_IMPLEMENTATION_SUMMARY.md`
- **Detailed Walkthrough**: See `doc/EXECUTION_WALKTHROUGH.md` for step-by-step example with outputs
- **Session Summary**: See `doc/SESSION_COMPLETION_SUMMARY.md`

## 🎓 Learning the System

### Beginner: Just Want to Run It
1. Copy the "Quick Start" commands above
2. Open 2 terminals
3. Run schedule generator in terminal 1
4. Run monitor in terminal 2
5. Run executor in terminal 1
6. Watch monitor update in real-time

### Intermediate: Want to Customize
1. Read `TRAINING_SCHEDULE_GUIDE.md` sections on custom convergence factors
2. Edit convergence factors in `training_schedule_generator.py`
3. Regenerate schedule
4. Adjust start time with `--start-time` flag
5. Regenerate if needed

### Advanced: Want to Understand Architecture
1. Read `EXECUTION_WALKTHROUGH.md` architecture diagram
2. Study the three main scripts (generator, monitor, executor)
3. Look at how monitor integrates with trainer working logs
4. Understand ETA drift calculation formula

## 🆘 Troubleshooting

**"No backend metrics yet"**
- Training hasn't started or first epoch hasn't completed
- Monitor needs at least one epoch log entry to show speed

**"Training slower than predicted"**
- First epochs are often slower than average
- Monitor will show acceleration as training progresses
- If consistently 50%+ slower, may need batch size adjustment

**"Schedule times seem wrong"**
- Run with `--dry-run` to verify commands being generated
- Check if configs exist and are valid YAML
- Verify benchmark timings in generator script match what you measured

For more help, see "Troubleshooting" section in `TRAINING_SCHEDULE_GUIDE.md`

---

**Ready to start?** Copy the Quick Start commands above and run them! 🚀
