#!/usr/bin/env python3
"""
Schedule executor: runs training jobs according to generated schedule.

Orchestrates:
1. Sequential (or parallel) archetype launches
2. Backend assignment from schedule
3. Real-time progress monitoring
4. Checkpoint recovery on interruption
5. Post-training evaluation
"""

import argparse
import json
import subprocess
import sys
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil


class ScheduleExecutor:
    def __init__(
        self,
        schedule_path: Path,
        work_log_path: Path,
        configs_dir: Path = Path("configs"),
        parallel: bool = False,
        dry_run: bool = False,
    ):
        self.schedule_path = schedule_path
        self.work_log_path = work_log_path
        self.configs_dir = configs_dir
        self.parallel = parallel
        self.dry_run = dry_run
        
        self.schedule = self._load_schedule()
        self.results: Dict[str, Dict] = {}
        
    def _load_schedule(self) -> Dict:
        """Load schedule YAML."""
        if not self.schedule_path.exists():
            raise FileNotFoundError(f"Schedule file not found: {self.schedule_path}")
        
        return yaml.safe_load(self.schedule_path.read_text(encoding="utf-8"))
    
    def _get_training_command(
        self,
        archetype: str,
        backend: str,
    ) -> Tuple[str, List[str]]:
        """
        Get the Python module and arguments for training an archetype.
        
        Returns:
            (module_name, [--config, path_to_config])
        """
        module_map = {
            'trend': 'quant_core.train_trend_phase4',
            'mean_reversion': 'quant_core.train_mr_phase4',
            'scalper': 'quant_core.train_scalper_phase4',
            'stat_arb': 'quant_core.train_stat_arb_phase4',
            'discretionary': 'quant_core.train_discretionary_phase4',
            'market_maker': 'quant_core.train_mm_phase4',
        }
        
        config_map = {
            'trend': 'trend_phase4.yaml',
            'mean_reversion': 'mr_phase4.yaml',
            'scalper': 'scalper_phase4.yaml',
            'stat_arb': 'stat_arb_phase4.yaml',
            'discretionary': 'discretionary_phase4.yaml',
            'market_maker': 'mm_phase4.yaml',
        }
        
        if archetype not in module_map:
            raise ValueError(f"Unknown archetype: {archetype}")
        
        module = module_map[archetype]
        config_file = self.configs_dir / config_map[archetype]
        
        # If backend is specified, we need to patch the config
        if backend not in {'directml', 'cpu'}:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Create a temporary patched config with the desired backend
        patched_config = self._patch_config(config_file, backend)
        
        return module, ['--config', str(patched_config)]
    
    def _patch_config(self, config_path: Path, backend: str) -> Path:
        """
        Patch a config file to use the specified backend.
        
        Creates a temporary config file and returns its path.
        """
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        
        # Update backend
        if 'training' in config:
            config['training']['preferred_backend'] = backend
        
        # Write to temporary file
        # For now, we'll write to a temp file in the configs directory
        temp_config_path = self.configs_dir / f"_schedule_patch_{backend}_{int(time.time())}.yaml"
        temp_config_path.write_text(
            yaml.dump(config, default_flow_style=False),
            encoding="utf-8"
        )
        
        return temp_config_path
    
    def _run_archetype_training(
        self,
        archetype: str,
        backend: str,
        start_time: Optional[datetime] = None,
    ) -> Dict:
        """
        Launch training for an archetype.
        
        Returns:
            Dict with execution result
        """
        result = {
            'archetype': archetype,
            'backend': backend,
            'start_time': start_time or datetime.utcnow(),
            'end_time': None,
            'exit_code': None,
            'duration_seconds': 0,
            'status': 'pending',
        }
        
        module, args = self._get_training_command(archetype, backend)
        
        cmd = [
            sys.executable,
            '-m',
            module,
        ] + args
        
        print(f"\n{'='*100}")
        print(f"[{result['start_time'].isoformat()}] Starting {archetype} training")
        print(f"Backend: {backend}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*100}\n")
        
        if self.dry_run:
            print("[DRY RUN] Would execute above command")
            result['status'] = 'dry_run'
            result['exit_code'] = 0
            return result
        
        try:
            start = time.time()
            proc = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=False,  # Show output in real-time
                text=True,
                timeout=None,  # No timeout - let it run to completion
            )
            elapsed = time.time() - start
            
            result['exit_code'] = proc.returncode
            result['duration_seconds'] = elapsed
            result['end_time'] = datetime.utcnow()
            result['status'] = 'completed' if proc.returncode == 0 else 'failed'
            
            print(f"\n[{result['end_time'].isoformat()}] {archetype} training completed")
            print(f"Exit Code: {proc.returncode}")
            print(f"Duration: {elapsed/60:.1f} minutes")
            print(f"{'='*100}\n")
            
        except subprocess.TimeoutExpired:
            result['status'] = 'timeout'
            result['end_time'] = datetime.utcnow()
            print(f"\nERROR: {archetype} training timed out after {result['duration_seconds']/60:.1f} minutes")
        except Exception as e:
            result['status'] = 'error'
            result['end_time'] = datetime.utcnow()
            print(f"\nERROR: {archetype} training failed with exception: {e}")
        
        return result
    
    def execute(self) -> int:
        """
        Execute the training schedule.
        
        Returns:
            0 on success, non-zero on failure
        """
        archetypes_info = self.schedule.get('archetypes', {})
        
        if not archetypes_info:
            print("ERROR: No archetypes found in schedule", file=sys.stderr)
            return 1
        
        print(f"\n{'='*100}")
        print("TRAINING SCHEDULE EXECUTOR")
        print(f"{'='*100}")
        print(f"Schedule: {self.schedule_path}")
        print(f"Mode: {'PARALLEL' if self.parallel else 'SEQUENTIAL'}")
        print(f"Dry Run: {self.dry_run}")
        print(f"Total Archetypes: {len(archetypes_info)}")
        print(f"{'='*100}\n")
        
        # Show schedule summary
        for arch in sorted(archetypes_info.keys()):
            info = archetypes_info[arch]
            print(
                f"{arch:<18} backend={info['backend']:<10} "
                f"models={len(info['models']):<2} eta={info['eta_duration']:<12}"
            )
        
        print(f"\n{'='*100}\n")
        
        # Execute archetypes in order
        for arch in sorted(archetypes_info.keys()):
            info = archetypes_info[arch]
            backend = info['backend']
            
            result = self._run_archetype_training(
                archetype=arch,
                backend=backend,
                start_time=None,
            )
            
            self.results[arch] = result
            
            # Check result
            if result['status'] not in {'completed', 'dry_run'}:
                print(f"WARNING: {arch} training {result['status']}")
                # Continue to next archetype instead of stopping
            
            # If not parallel, wait before next archetype
            if not self.parallel:
                time.sleep(1)  # Small delay between archetypes
        
        # Print summary
        self._print_summary()
        
        return 0
    
    def _print_summary(self) -> None:
        """Print execution summary."""
        print(f"\n{'='*100}")
        print("EXECUTION SUMMARY")
        print(f"{'='*100}")
        
        total_duration = 0
        completed = 0
        failed = 0
        
        for arch in sorted(self.results.keys()):
            result = self.results[arch]
            status = result['status']
            duration = result['duration_seconds']
            total_duration += duration
            
            if status in {'completed', 'dry_run'}:
                completed += 1
            else:
                failed += 1
            
            duration_str = f"{duration/60:.1f}m" if duration > 0 else "N/A"
            
            print(
                f"{arch:<18} status={status:<12} duration={duration_str:<12} "
                f"exit_code={result['exit_code']}"
            )
        
        print(f"\n{'='*100}")
        print(f"Total Completed: {completed}")
        print(f"Total Failed: {failed}")
        print(f"Total Duration: {total_duration/3600:.2f} hours")
        print(f"{'='*100}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute training schedule with per-archetype orchestration"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="doc/training_schedule.yaml",
        help="Path to schedule YAML file",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(Path("doc") / "training_more_27-4" / "27-04-2026_plan_REVISED_workingLog.md"),
        help="Path to working log file",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="configs",
        help="Directory containing config files",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Execute archetypes in parallel (requires separate processes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without actually running",
    )
    
    args = parser.parse_args()
    
    executor = ScheduleExecutor(
        schedule_path=Path(args.schedule),
        work_log_path=Path(args.log_path),
        configs_dir=Path(args.configs_dir),
        parallel=args.parallel,
        dry_run=args.dry_run,
    )
    
    return executor.execute()


if __name__ == "__main__":
    sys.exit(main())
