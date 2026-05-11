#!/usr/bin/env python3
"""Extract 3-day training progress snapshot."""

import re
from pathlib import Path
from datetime import datetime

log_path = Path('doc/training_more_27-4/27-04-2026_plan_REVISED_workingLog.md')
lines = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()

# Extract all EPOCH entries
epochs_dict = {}
for line in lines:
    if 'stage=EPOCH' in line and 'epoch=' in line:
        match = re.search(r'model=(\S+).*epoch=(\d+)/(\d+).*elapsed_s=(\d+\.\d+)', line)
        if match:
            model = match.group(1)
            current = int(match.group(2))
            total = int(match.group(3))
            elapsed = float(match.group(4))
            epochs_dict[model] = (current, total, elapsed)

# Extract FINAL entries  
finals_dict = {}
for line in lines:
    if 'stage=FINAL' in line:
        match = re.search(r'model=(\S+).*test_sharpe=([0-9.\-e]+).*test_profit_factor=([0-9.\-e]+).*test_max_drawdown=([0-9.\-e]+)', line)
        if match:
            model = match.group(1)
            sharpe = float(match.group(2))
            pf = float(match.group(3))
            mdd = float(match.group(4))
            finals_dict[model] = (sharpe, pf, mdd)

print('=' * 80)
print('⏱️  3-DAY TRAINING PROGRESS REPORT (May 9 14:51 → May 11 08:11)')
print('=' * 80)
print()

# Show in-progress models
if epochs_dict:
    print('🔄 IN PROGRESS MODELS')
    print('-' * 80)
    for model in sorted(epochs_dict.keys()):
        current, total, elapsed = epochs_dict[model]
        pct = current * 100 / total
        time_per_epoch = elapsed / 60  # convert to minutes
        remaining = total - current
        eta_minutes = remaining * time_per_epoch
        eta_hours = eta_minutes / 60
        print(f'  {model:<25} Epoch {current:3d}/{total} ({pct:5.1f}%)')
        print(f'    ├─ Time/Epoch: {time_per_epoch:6.1f} min')
        print(f'    └─ ETA: {eta_hours:6.1f} hours ({eta_hours/24:.1f} days)')
    print()

# Show completed models
if finals_dict:
    print('✅ COMPLETED MODELS')
    print('-' * 80)
    passed = 0
    for model in sorted(finals_dict.keys()):
        sharpe, pf, mdd = finals_dict[model]
        gates_pass = sharpe > 1.2 and pf > 1.5 and mdd < 0.2
        status = '✓ PASS' if gates_pass else '✗ FAIL'
        if gates_pass:
            passed += 1
        print(f'  {model:<25} {status}  Sharpe={sharpe:7.2f} PF={pf:5.2f} MDD={mdd:5.2f}')
    print(f'\n  Gate Compliance: {passed}/{len(finals_dict)} passed')
else:
    print('⏳ NO COMPLETED MODELS YET')
    if epochs_dict:
        print('  First model will complete after current training finishes')
        lstm_time = epochs_dict.get('LSTM_Trend_v1')
        if lstm_time:
            current, total, elapsed = lstm_time
            remaining = total - current
            eta_hours = (remaining * elapsed / 60) / 60
            print(f'  LSTM_Trend_v1 ETA: ~{eta_hours:.1f} hours ({eta_hours/24:.1f} days)')

print()
print('=' * 80)
