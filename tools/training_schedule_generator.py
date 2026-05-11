#!/usr/bin/env python3
"""
Auto-generate per-archetype training schedule with predicted finish timestamps.

Inputs:
  - Benchmark results from smoke runs (1-epoch timing)
  - Config files with max_epochs per archetype
  - Convergence and gate difficulty factors

Output:
  - Schedule YAML with [archetype | models | backend | start_time | predicted_end_time | expected_result]
  - Sequential or parallel execution plan
  - ETA for first pass + all-pass
"""

import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import sys

# Benchmark results from previous measurements (smoke = 1 epoch)
BENCHMARKS = {
    'trend': {'directml': 20.66, 'cpu': 21.11},
    'mean_reversion': {'directml': 4.88, 'cpu': 4.60},
    'scalper': {'directml': 14.75, 'cpu': 10.07},
    'stat_arb': {'directml': 20.93, 'cpu': 13.38},
    'discretionary': {'directml': 6.03, 'cpu': 7.15},
    'market_maker': {'directml': 32.53, 'cpu': 17.37},
}

# Best backend per archetype (from benchmarks)
BEST_BACKEND = {
    'trend': 'directml',  # 20.66 < 21.11
    'mean_reversion': 'cpu',  # 4.60 < 4.88
    'scalper': 'cpu',  # 10.07 < 14.75
    'stat_arb': 'cpu',  # 13.38 < 20.93
    'discretionary': 'directml',  # 6.03 < 7.15
    'market_maker': 'cpu',  # 17.37 < 32.53
}

# Config epochs per archetype
CONFIG_EPOCHS = {
    'trend': 150,
    'mean_reversion': 150,
    'scalper': 120,
    'stat_arb': 120,
    'discretionary': 200,
    'market_maker': ('episodes', 8000),  # (type, count)
}

# For market maker: smoke_episodes = 200, full_episodes = 8000
MM_SMOKE_EPISODES = 200
MM_FULL_EPISODES = 8000

# Convergence and gate difficulty multipliers
# These account for:
#   - Convergence plateau (valuation Sharpe may plateau before max_epochs)
#   - Strict financial gates (Sharpe>1.2, PF>1.5, MDD<0.2 are harder than 55% accuracy)
#   - Overfitting risk (later epochs may not improve OOS metrics)
CONVERGENCE_FACTORS = {
    'trend': 1.5,           # Strong convergence but gate is strict
    'mean_reversion': 1.4,  # Can converge quickly but noise-sensitive
    'scalper': 1.6,         # Very tight gate, may need tuning iterations
    'stat_arb': 1.3,        # Close to passing historically
    'discretionary': 1.8,   # Hardest gate (F1>0.65), longest epochs (200)
    'market_maker': 1.2,    # RL learns fast, episodes scale linearly
}

# Models per archetype (v2 model names)
MODELS_PER_ARCHETYPE = {
    'trend': ['LSTM_Trend_v2', 'Transformer_Trend_v2', 'TCN_Trend_v2'],
    'mean_reversion': ['MLP_MR_v2', 'ResNet_MR_v2', 'GRN_MR_v2'],
    'scalper': ['CNN_Scalper_v2', 'LinearAttn_Scalper_v2', 'GRU_Scalper_v2'],
    'stat_arb': ['Autoencoder_StatArb_v2', 'GAT_StatArb_v2', 'LSTM_StatArb_v2'],
    'discretionary': ['ViT_Disc_v2', 'Multimodal_Disc_v2', 'CNN_Chart_Disc_v2'],
    'market_maker': ['PPO_MM_v2', 'SAC_MM_v2', 'DQN_MM_v2'],
}


def compute_eta_seconds(archetype: str, backend: str) -> float:
    """
    Compute estimated training time in seconds.
    
    Formula:
      full_time = (smoke_time_seconds × num_epochs) × convergence_factor
    
    For market maker (episodes-based), adjust the multiplier differently.
    """
    smoke_time = BENCHMARKS[archetype][backend]
    
    if archetype == 'market_maker':
        # MM: smoke=200 episodes, full=8000 episodes
        # Time scales linearly with episodes
        num_episodes_ratio = MM_FULL_EPISODES / MM_SMOKE_EPISODES
        convergence_factor = CONVERGENCE_FACTORS[archetype]
        return smoke_time * num_episodes_ratio * convergence_factor
    else:
        # Supervised models: smoke=1 epoch, full=config_epochs
        num_epochs = CONFIG_EPOCHS[archetype]
        convergence_factor = CONVERGENCE_FACTORS[archetype]
        return smoke_time * num_epochs * convergence_factor


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def generate_schedule(
    start_time: datetime,
    parallel: bool = False
) -> Dict:
    """
    Generate training schedule with predicted timestamps.
    
    Args:
        start_time: UTC datetime to start training
        parallel: If True, train all archetypes in parallel; else sequential
    
    Returns:
        Dict with schedule entries, ETAs, and metadata
    """
    schedule = {
        'metadata': {
            'generated_at': datetime.utcnow().isoformat(),
            'start_time': start_time.isoformat(),
            'parallel_execution': parallel,
            'benchmark_source': 'smoke_runs_27-04-2026',
        },
        'archetypes': {},
        'timeline': [],
        'summary': {
            'total_models': 0,
            'total_seconds_sequential': 0.0,
            'total_seconds_parallel': 0.0,
            'estimated_first_pass_time': None,
            'estimated_all_pass_time': None,
        }
    }
    
    current_time = start_time
    all_etas = {}  # Track all ETA endpoints for parallel case
    
    for archetype in sorted(BEST_BACKEND.keys()):
        backend = BEST_BACKEND[archetype]
        eta_seconds = compute_eta_seconds(archetype, backend)
        models = MODELS_PER_ARCHETYPE[archetype]
        
        schedule['summary']['total_models'] += len(models)
        
        # For sequential: each archetype runs after previous
        # For parallel: all start at same time, end time is max(all etas)
        
        if parallel:
            arch_start = start_time
            arch_end = start_time + timedelta(seconds=eta_seconds)
            all_etas[archetype] = arch_end
        else:
            arch_start = current_time
            arch_end = arch_start + timedelta(seconds=eta_seconds)
            current_time = arch_end
        
        schedule['archetypes'][archetype] = {
            'backend': backend,
            'models': models,
            'smoke_time_seconds': BENCHMARKS[archetype][backend],
            'convergence_factor': CONVERGENCE_FACTORS[archetype],
            'eta_seconds': eta_seconds,
            'eta_duration': format_duration(eta_seconds),
            'start_time': arch_start.isoformat(),
            'end_time': arch_end.isoformat(),
            'expected_status': _predict_status(archetype),
        }
        
        schedule['timeline'].append({
            'archetype': archetype,
            'start': arch_start.isoformat(),
            'end': arch_end.isoformat(),
            'duration': format_duration(eta_seconds),
            'models': len(models),
            'backend': backend,
        })
        
        schedule['summary']['total_seconds_sequential'] += eta_seconds
    
    # Compute all-pass time
    if parallel:
        schedule['summary']['total_seconds_parallel'] = max(
            (end - start_time).total_seconds()
            for end in all_etas.values()
        )
        all_pass_time = start_time + timedelta(
            seconds=schedule['summary']['total_seconds_parallel']
        )
    else:
        schedule['summary']['total_seconds_parallel'] = schedule['summary']['total_seconds_sequential']
        all_pass_time = current_time
    
    schedule['summary']['estimated_all_pass_time'] = all_pass_time.isoformat()
    
    # Estimate first pass (pessimistic: longest archetype, but 80% success rate)
    # Assume ~60% of models pass on first attempt (2/3), need tuning for ~40%
    # First pass ETA = max_eta + (2 retry cycles × mean_eta / 6 models)
    max_eta = max(
        schedule['archetypes'][a]['eta_seconds']
        for a in BEST_BACKEND.keys()
    )
    mean_eta = sum(
        schedule['archetypes'][a]['eta_seconds']
        for a in BEST_BACKEND.keys()
    ) / len(BEST_BACKEND)
    
    # Retry cost: if 60% pass, 40% need one retry cycle
    retry_cost = mean_eta * 0.40
    first_pass_seconds = max_eta + retry_cost
    first_pass_time = start_time + timedelta(seconds=first_pass_seconds)
    schedule['summary']['estimated_first_pass_time'] = first_pass_time.isoformat()
    
    return schedule


def _predict_status(archetype: str) -> Dict:
    """
    Predict expected status for an archetype based on prior runs.
    
    Returns a dict with confidence levels and reasoning.
    """
    # Based on prior evaluation results
    prior_results = {
        'trend': {
            'confidence': 'MEDIUM',
            'reason': 'Froze at val_sharpe=6.29 epoch 1 with 2-asset subset; full 34-asset may help',
            'expected_pass': 0.55,
        },
        'mean_reversion': {
            'confidence': 'MEDIUM',
            'reason': 'Not yet retrained v2; noise sensitivity high',
            'expected_pass': 0.45,
        },
        'scalper': {
            'confidence': 'LOW',
            'reason': 'Not yet retrained v2; flat_threshold change is significant',
            'expected_pass': 0.40,
        },
        'stat_arb': {
            'confidence': 'MEDIUM-HIGH',
            'reason': 'Autoencoder/GAT close to passing (Sharpe 2.08, PF 1.73); need MDD fix',
            'expected_pass': 0.70,
        },
        'discretionary': {
            'confidence': 'LOW',
            'reason': 'Not yet retrained v2; hardest gate (F1>0.65); longest epochs (200)',
            'expected_pass': 0.35,
        },
        'market_maker': {
            'confidence': 'LOW',
            'reason': 'RL obs_dim mismatch in v1; 8000 episodes is long convergence',
            'expected_pass': 0.50,
        },
    }
    
    return prior_results.get(archetype, {
        'confidence': 'UNKNOWN',
        'reason': 'No prior data',
        'expected_pass': 0.5,
    })


def save_schedule(schedule: Dict, output_path: Path) -> None:
    """Save schedule to YAML file."""
    output_path.write_text(
        yaml.dump(schedule, default_flow_style=False, sort_keys=False),
        encoding='utf-8'
    )


def print_schedule(schedule: Dict) -> None:
    """Print schedule in human-readable format."""
    meta = schedule['metadata']
    print("\n" + "="*120)
    print("TRAINING SCHEDULE — Per-Archetype Timing with Predicted Finish Timestamps")
    print("="*120)
    print(f"Generated: {meta['generated_at']}")
    print(f"Start Time: {meta['start_time']}")
    print(f"Execution Mode: {'PARALLEL' if meta['parallel_execution'] else 'SEQUENTIAL'}")
    print()
    
    print("ARCHETYPE SCHEDULE")
    print("-"*120)
    print(f"{'Archetype':<18} {'Backend':<12} {'#Models':<8} {'Smoke':<10} {'Conv.F':<8} {'ETA Duration':<15} {'Start':<20} {'End':<20} {'Status':<20}")
    print("-"*120)
    
    for arch in sorted(schedule['archetypes'].keys()):
        info = schedule['archetypes'][arch]
        status = info['expected_status']
        status_str = f"P={status.get('expected_pass', 0)*100:.0f}%"
        
        print(
            f"{arch:<18} {info['backend']:<12} {len(info['models']):<8} "
            f"{info['smoke_time_seconds']:<10.2f}s {info['convergence_factor']:<8.1f} "
            f"{info['eta_duration']:<15} "
            f"{info['start_time'][11:19]:<20} {info['end_time'][11:19]:<20} {status_str:<20}"
        )
    
    print()
    print("TIMELINE SUMMARY")
    print("-"*120)
    for entry in schedule['timeline']:
        print(
            f"{entry['archetype']:<18} {entry['start'][11:19]} → {entry['end'][11:19]} "
            f"({entry['duration']:<10}) {entry['models']} models @ {entry['backend']}"
        )
    
    print()
    print("ESTIMATED COMPLETION TIMES")
    print("-"*120)
    summary = schedule['summary']
    print(f"Total Models: {summary['total_models']}")
    print(f"Sequential Total Time: {format_duration(summary['total_seconds_sequential'])}")
    print(f"Parallel Total Time: {format_duration(summary['total_seconds_parallel'])}")
    print(f"")
    print(f"First Model Expected to Pass: {summary['estimated_first_pass_time']}")
    print(f"All Models Expected to Pass: {summary['estimated_all_pass_time']}")
    print(f"")
    
    # Confidence analysis
    expected_pass_rates = [
        schedule['archetypes'][a]['expected_status']['expected_pass']
        for a in schedule['archetypes'].keys()
    ]
    avg_expected = sum(expected_pass_rates) / len(expected_pass_rates)
    print(f"Average Expected Pass Rate Across Archetypes: {avg_expected*100:.1f}%")
    print(f"Pessimistic Scenario (40% pass): {format_duration(summary['total_seconds_sequential'] * 2.5)}")
    print(f"Optimistic Scenario (80% pass): {format_duration(summary['total_seconds_sequential'] * 1.3)}")
    print()
    print("="*120)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate per-archetype training schedule with predicted finish timestamps"
    )
    parser.add_argument(
        '--start-time',
        type=str,
        default=None,
        help="Start time in ISO format (default: now UTC)"
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help="Generate parallel schedule (all archetypes concurrent) instead of sequential"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='doc/training_schedule.yaml',
        help="Output schedule file path"
    )
    
    args = parser.parse_args()
    
    # Parse start time
    if args.start_time:
        try:
            start = datetime.fromisoformat(args.start_time)
        except ValueError:
            print(f"ERROR: Invalid ISO datetime format: {args.start_time}", file=sys.stderr)
            return 1
    else:
        start = datetime.utcnow()
    
    # Generate schedule
    schedule = generate_schedule(start, parallel=args.parallel)
    
    # Save to YAML
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_schedule(schedule, output_path)
    
    # Print to console
    print_schedule(schedule)
    
    print(f"\n✓ Schedule saved to: {output_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
