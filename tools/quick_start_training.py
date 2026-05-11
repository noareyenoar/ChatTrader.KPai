#!/usr/bin/env python3
"""
Quick-start launcher: Generates schedule, launches monitor, and executes training.
Usage: python tools/quick_start_training.py [--monitor-only] [--schedule-only] [--execute-only]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd_list, description):
    """Run a command and report results."""
    print(f"\n{'='*100}")
    print(f"LAUNCHING: {description}")
    print(f"{'='*100}")
    print(f"Command: {' '.join(cmd_list)}\n")
    
    try:
        result = subprocess.run(cmd_list, cwd=Path.cwd())
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quick-start training schedule system"
    )
    parser.add_argument(
        "--schedule-only",
        action="store_true",
        help="Only generate schedule, don't launch monitor or execute"
    )
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="Only launch monitor, assume schedule already exists"
    )
    parser.add_argument(
        "--execute-only",
        action="store_true",
        help="Only execute schedule, assume schedule already exists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would execute without running"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel execution mode"
    )
    
    args = parser.parse_args()
    
    # Determine which steps to run
    do_schedule = not (args.monitor_only or args.execute_only)
    do_monitor = not (args.schedule_only or args.execute_only)
    do_execute = not (args.schedule_only or args.monitor_only)
    
    python_exe = sys.executable
    
    # Step 1: Generate Schedule
    if do_schedule:
        schedule_cmd = [
            python_exe,
            "tools/training_schedule_generator.py",
            "--output", "doc/training_schedule.yaml",
        ]
        if args.parallel:
            schedule_cmd.append("--parallel")
        
        if not run_command(schedule_cmd, "SCHEDULE GENERATOR"):
            print("ERROR: Failed to generate schedule")
            return 1
        
        print("✓ Schedule generated successfully")
    
    # Step 2: Launch Monitor (in background)
    if do_monitor:
        print(f"\n{'='*100}")
        print("NEXT: Launch Enhanced Monitor in a separate terminal/window")
        print(f"{'='*100}")
        print("\nLinux/Mac:")
        print("  python tools/training_monitor_with_schedule.py &")
        print("\nWindows (PowerShell):")
        print("  Start-Process python -ArgumentList 'tools/training_monitor_with_schedule.py'")
        print("\nOr copy-paste this command:")
        print(f"  {python_exe} tools/training_monitor_with_schedule.py")
        print("\nThe monitor will display:")
        print("  - Live training progress per model")
        print("  - Scheduled timeline vs actual progress")
        print("  - ETA tracking with drift analysis")
        print("  - Backend speed recommendations")
    
    # Step 3: Execute Schedule
    if do_execute:
        print(f"\n{'='*100}")
        print("READY TO EXECUTE TRAINING SCHEDULE")
        print(f"{'='*100}")
        print("\nStarting training in 5 seconds...")
        print("(Press Ctrl+C to cancel)\n")
        
        for i in range(5, 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1)
        
        execute_cmd = [
            python_exe,
            "tools/training_schedule_executor.py",
            "--schedule", "doc/training_schedule.yaml",
        ]
        if args.dry_run:
            execute_cmd.append("--dry-run")
        if args.parallel:
            execute_cmd.append("--parallel")
        
        if not run_command(execute_cmd, "SCHEDULE EXECUTOR"):
            print("ERROR: Training execution failed")
            return 1
        
        print("\n✓ Training execution completed")
    
    print(f"\n{'='*100}")
    print("QUICK START COMPLETE")
    print(f"{'='*100}")
    print("\nNext steps:")
    print("1. Monitor the progress in the monitor terminal")
    print("2. After training completes, run evaluation:")
    print(f"   {python_exe} evaluate_all_checkpoints.py")
    print("3. Check model_registry.json for pass/fail status")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
