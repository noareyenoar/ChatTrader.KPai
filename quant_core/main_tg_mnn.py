"""Main entry point for TG-MNN Phase 4 training.

Orchestrates:
1. Configuration loading from YAML
2. Data preparation and validation
3. Model training with proper seeding
4. Checkpoint management
5. Evaluation and reporting
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from quant_core.train_tg_mnn_phase4 import (
    train_tg_mnn,
    set_global_seed,
    TGMNNTrainingResult,
)
from quant_core.tg_mnn_validation import WaveValidationReporter


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str, symbols: list[str] | None = None) -> None:
    """
    Train TG-MNN model end-to-end.

    Args:
        config_path: Path to YAML configuration file
        symbols: Optional list of symbols to use
    """
    # Load config
    config = load_config(config_path)
    print(f"[TG-MNN] Loaded configuration from {config_path}")

    # Extract relevant settings
    reproducibility = config.get('reproducibility', {})
    seed = reproducibility.get('seed', 42)
    set_global_seed(seed)
    print(f"[TG-MNN] Set global seed: {seed}")

    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})

    data_dir = Path(data_cfg.get('input_dir', 'Dataset/binance_historical'))
    output_dir = Path(data_cfg.get('output_dir', 'models/checkpoints/tg_mnn'))

    # Override with CLI symbols if provided
    if symbols is None:
        symbols = data_cfg.get('symbols', [])
        if not symbols:
            symbols = None  # Auto-discover

    print(f"[TG-MNN] Data directory: {data_dir}")
    print(f"[TG-MNN] Output directory: {output_dir}")

    # Prepare training config dict
    train_config = {
        'seq_len': data_cfg.get('seq_len', 50),
        'hidden_dim': model_cfg.get('hidden_dim', 64),
        'num_layers': model_cfg.get('num_backbone_layers', 3),
        'batch_size': training_cfg.get('batch_size', 32),
        'learning_rate': training_cfg.get('learning_rate', 1e-3),
        'max_epochs': training_cfg.get('max_epochs', 50),
        'early_stopping_patience': training_cfg.get('early_stopping_patience', 10),
        'use_mixed_precision': training_cfg.get('use_mixed_precision', True),
        'preferred_backend': training_cfg.get('preferred_backend', 'auto'),
    }

    print(f"[TG-MNN] Training config: {train_config}")

    # Train model
    start_time = time.time()
    result = train_tg_mnn(
        data_dir=data_dir,
        output_dir=output_dir,
        config=train_config,
        symbols=symbols,
    )
    elapsed = time.time() - start_time
    print(f"[TG-MNN] Training completed in {elapsed:.1f}s")

    # Save result metadata
    result_path = output_dir / "training_result.json"
    result_dict = {
        'model_name': result.model_name,
        'checkpoint_dir': result.checkpoint_dir,
        'train_loss': result.train_loss,
        'val_loss': result.val_loss,
        'test_loss': result.test_loss,
        'test_state_acc': result.test_state_acc,
        'is_valid': result.is_valid,
        'backend': result.backend,
        'cuda_used': result.cuda_used,
        'training_time_sec': elapsed,
    }

    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"[TG-MNN] Saved result metadata: {result_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TG-MNN TRAINING SUMMARY")
    print("=" * 70)
    print(f"Model: {result.model_name}")
    print(f"Checkpoint: {result.checkpoint_dir}")
    print(f"Training Loss: {result.train_loss:.6f}")
    print(f"Validation Loss: {result.val_loss:.6f}")
    print(f"Test Loss: {result.test_loss:.6f}")
    print(f"Test State Accuracy: {result.test_state_acc:.4f}")
    print(f"Model Valid (acc > 0.45): {result.is_valid}")
    print(f"Backend: {result.backend}")
    print(f"Total Time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TG-MNN (Temporal-Gradient Markov Neural Network)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tg_mnn_phase4.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs='+',
        default=None,
        help="Symbols to train on (space-separated); if not provided, auto-discover",
    )
    args = parser.parse_args()

    main(args.config, args.symbols)
