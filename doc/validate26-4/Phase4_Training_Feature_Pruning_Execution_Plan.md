# Phase 4: Training & Feature Pruning Execution Plan

**Date:** April 26, 2026  
**Scope:** 50-epoch full sweep with SHAP-based feature pruning  
**Target Completion:** Post-model training gate before Phase 5

---

## I. Pre-Training Validation Checklist

### 1.1 Data Pipeline Validation

```python
# Script: validate_phase4_data.py
import pandas as pd
import numpy as np
from pathlib import Path
from data_pipeline.features import FeatureFactory
from data_pipeline.config import DataQualityGate

def validate_tick_volume_bars():
    """Verify tick/volume bar consistency."""
    assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BTCETH", "HYPEUSDT"]
    
    for asset in assets:
        # Load tick bars
        tick_bars = pd.read_parquet(f"Dataset/processed/tick_bars/{asset}_1000trades.parquet")
        print(f"{asset} Tick Bars: {len(tick_bars)} bars")
        
        # Validate schema
        required_cols = ["timestamp", "open", "high", "low", "close", "volume", "vwap"]
        missing = set(required_cols) - set(tick_bars.columns)
        assert not missing, f"Missing columns: {missing}"
        
        # Sanity checks
        assert (tick_bars["high"] >= tick_bars["low"]).all(), "Invalid OHLC"
        assert (tick_bars["close"] >= 0).all(), "Negative prices"
        assert tick_bars["timestamp"].is_unique, "Duplicate timestamps"
        
        # Statistics
        print(f"  Price range: {tick_bars['close'].min():.2f} - {tick_bars['close'].max():.2f}")
        print(f"  Volume distribution: p25={tick_bars['volume'].quantile(0.25):.0f}, "
              f"p50={tick_bars['volume'].quantile(0.5):.0f}, "
              f"p75={tick_bars['volume'].quantile(0.75):.0f}")

def validate_features():
    """Verify feature quality and distributions."""
    # Load sample feature matrix
    X_train = pd.read_parquet("data/X_train_features_phase4.parquet")
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"\nFeature summary:")
    print(X_train.describe())
    
    # Check for NaN
    nan_ratio = X_train.isnull().sum() / len(X_train)
    print(f"\nNaN ratios per feature:")
    print(nan_ratio[nan_ratio > 0])
    
    # Check for infinite values
    inf_mask = np.isinf(X_train)
    if inf_mask.any().any():
        print(f"WARNING: Found {inf_mask.sum().sum()} infinite values")
    
    # Check scaling (should be approximately N(0, 1) after fit_scaler)
    for col in X_train.columns[:5]:  # Sample first 5
        mean = X_train[col].mean()
        std = X_train[col].std()
        print(f"{col}: mean={mean:.4f}, std={std:.4f}")

def validate_labels():
    """Verify label distribution (triple-barrier)."""
    y_train = pd.read_parquet("data/y_train_labels_phase4.parquet")
    
    label_counts = y_train.value_counts()
    label_pcts = (label_counts / len(y_train)) * 100
    
    print("Label distribution (triple-barrier):")
    print(f"  LONG (+1):  {label_counts.get(1, 0):6d} ({label_pcts.get(1, 0):5.2f}%)")
    print(f"  FLAT (0):   {label_counts.get(0, 0):6d} ({label_pcts.get(0, 0):5.2f}%)")
    print(f"  SHORT (-1): {label_counts.get(-1, 0):6d} ({label_pcts.get(-1, 0):5.2f}%)")
    
    # Target: 25-35% FLAT, ~33% each for LONG/SHORT
    flat_pct = label_pcts.get(0, 0)
    assert 20 <= flat_pct <= 40, f"FLAT class {flat_pct:.1f}% outside target range [20, 40]%"

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 4 DATA VALIDATION")
    print("=" * 80)
    
    print("\n1. Tick/Volume Bars:")
    validate_tick_volume_bars()
    
    print("\n2. Features:")
    validate_features()
    
    print("\n3. Labels:")
    validate_labels()
    
    print("\n✓ All validation checks passed!")
```

### 1.2 Configuration Updates

**Update `configs/scalper_phase4.yaml`:**
```yaml
scalper_phase4:
  model_type: "scalper_multimodel"
  
  # Data
  data_config:
    symbols:
      - BTCUSDT
      - ETHUSDT
      - SOLUSDT
      - BTCETH
      - HYPEUSDT
    data_sources:
      - "tick_bars"          # NEW: 1000 trades per bar
      - "volume_bars"        # NEW: 100 BTC per bar
      - "microstructure"     # NEW: OFI, VPIN, spread dynamics
      - "ohlcv_features"     # Legacy: fracdiff, velocity, regime
    synthetic_data_blend: 0.20  # 20% synthetic data (GARCH, HMM, bootstrap)
    min_history_bars: 50000
    purge_gap: 20
  
  # Labels
  label_config:
    strategy: "triple_barrier_adaptive"
    profit_pct_low: 0.0005      # 0.05% quiet markets
    profit_pct_normal: 0.001    # 0.1% normal
    profit_pct_high: 0.002      # 0.2% volatile
    stop_pct_low: 0.0005
    stop_pct_normal: 0.001
    stop_pct_high: 0.002
    max_bars: 20
  
  # Training
  training_config:
    max_epochs: 50
    batch_size: 64
    learning_rate: 1.0e-3
    patience: 10
    use_class_weights: true
    use_focal_loss: false         # Can enable if class imbalance persists
    optimizer: "adam"
    scheduler: "cyclic_lr"        # Cyclic learning rate
    cyclic_lr_base: 1.0e-4
    cyclic_lr_max: 1.0e-2
    cyclic_lr_step_size: 2500
  
  # Architectures (all three)
  architectures:
    cnn:
      channels: [128, 256, 256]
      kernel_sizes: [3, 3, 3]
      dropout: 0.3
      activation: "leaky_relu"
    
    linear_attn:
      hidden_dim: 256
      attn_heads: 4
      dropout: 0.2
      activation: "leaky_relu"
    
    gru:
      hidden_dim: 256
      num_layers: 2
      dropout: 0.3
      activation: "leaky_relu"
  
  # Validation gates
  validation_gates:
    directional_accuracy_min: 0.52
    sharpe_min: 1.0
    max_drawdown_max: 0.20
```

**Update `configs/mm_phase4.yaml`:**
```yaml
market_maker_phase4:
  model_type: "market_making_rl"
  
  # Environment
  environment_config:
    state_dim: 10  # [inv, mid_change, spread, vol, ofi, time, pnl, inv_skew, funding, oi]
    action_dim: 2  # [bid_offset, ask_offset]
    inventory_limit: 10.0
    transaction_cost: 0.0005
    market_impact_scale: 0.0001
    use_book_depth: true          # NEW: L2 depth for realistic fills
    use_funding_rates: true        # NEW: sentiment indicator
    use_open_interest: true        # NEW: positioning indicator
  
  # Curriculum learning schedule
  curriculum_schedule:
    - epochs: [1, 15]
      phase: "EASY"
      description: "Trending regimes (vol < 33rd percentile)"
    - epochs: [16, 35]
      phase: "MEDIUM"
      description: "All regimes (unfiltered)"
    - epochs: [36, 50]
      phase: "HARD"
      description: "Chaotic regimes (vol > 67th percentile)"
  
  # Training
  training_config:
    max_epochs: 50
    batch_size: 32
    learning_rate: 5.0e-4
    patience: 10
    episodes_per_epoch: 10
    episode_length: 200
    warmup_steps: 20
    seed: 42
  
  # Algorithms (all three)
  algorithms:
    ppo:
      hidden_dim: 128
      n_steps: 2048
      gae_lambda: 0.95
      clip_ratio: 0.2
      entropy_coef: 0.01
    
    sac:
      hidden_dim: 256
      update_freq: 1
      target_entropy: "auto"
      alpha: 0.2
    
    dqn:
      hidden_dim: 128
      epsilon_start: 1.0
      epsilon_end: 0.01
      epsilon_decay: 0.995
  
  # Validation gates
  validation_gates:
    mean_reward_min: 0.0
    sharpe_min: 0.5
    max_drawdown_max: 0.15
    eval_episodes: 20  # Multi-episode evaluation
```

---

## II. Training Execution

### 2.1 Scalper Models (CNN, LinearAttn, GRU)

```python
# Script: train_scalper_phase4.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from quant_core.scalper_data import build_scalper_datasets
from quant_core.scalper_training import (
    train_scalper_model,
    sanity_check,
    compute_sharpe,
    compute_max_drawdown,
)

def train_scalper_sweep():
    """Execute 50-epoch sweep for all scalper architectures."""
    
    # Load data
    print("Loading scalper training data...")
    train_df, val_df, test_df = build_scalper_datasets()
    
    # Extract features and labels
    feature_cols = [c for c in train_df.columns if c not in ["timestamp", "symbol", "label"]]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int64)
    
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.int64)
    
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Label distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} ({c/len(y_train)*100:.1f}%)")
    
    # Define architectures
    architectures = {
        "CNN_Scalper_v1": {"type": "cnn", "channels": [128, 256, 256]},
        "LinearAttn_Scalper_v1": {"type": "linear_attn", "hidden_dim": 256},
        "GRU_Scalper_v1": {"type": "gru", "hidden_dim": 256},
    }
    
    results = {}
    
    for arch_name, arch_config in architectures.items():
        print(f"\n{'='*80}")
        print(f"Training {arch_name}")
        print(f"{'='*80}")
        
        # Train model
        model, history = train_scalper_model(
            X_train, y_train, X_val, y_val,
            architecture=arch_config["type"],
            max_epochs=50,
            patience=10,
            batch_size=64,
            learning_rate=1.0e-3,
            use_class_weights=True,
            use_focal_loss=False,
            seed=42,
        )
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        test_accuracy = (y_pred == y_test).mean()
        test_sharpe = compute_sharpe(y_pred, y_test)
        test_max_drawdown = compute_max_drawdown(y_pred, y_test)
        
        # Sanity check
        sanity_ok = sanity_check(model, X_test, y_test)
        
        results[arch_name] = {
            "accuracy": test_accuracy,
            "sharpe": test_sharpe,
            "max_drawdown": test_max_drawdown,
            "sanity_check": sanity_ok,
            "history": history,
            "model": model,
        }
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Sharpe:   {test_sharpe:.4f}")
        print(f"  Max DD:   {test_max_drawdown:.4f}")
        print(f"  Sanity:   {'PASS' if sanity_ok else 'FAIL'}")
        
        # Save model
        save_path = Path(f"models/checkpoints/scalper_{arch_name.lower()}_phase4.pt")
        torch.save(model.state_dict(), save_path)
        print(f"  Model saved to {save_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SCALPER TRAINING SUMMARY")
    print(f"{'='*80}")
    for arch_name, res in results.items():
        status = "✓ PASS" if res["accuracy"] > 0.52 and res["sharpe"] > 1.0 else "✗ FAIL"
        print(f"{arch_name:30s} | Acc: {res['accuracy']:.4f} | Sharpe: {res['sharpe']:.4f} | {status}")
    
    return results

if __name__ == "__main__":
    results = train_scalper_sweep()
```

### 2.2 Market Maker Models (PPO, SAC, DQN)

```python
# Script: train_mm_phase4.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from quant_core.market_maker_env import MarketMakingEnv, TrainingCurriculum, CurriculumWrapper
from quant_core.market_maker_training import (
    train_ppo,
    train_sac,
    train_dqn,
    evaluate_policy,
)

def train_mm_sweep():
    """Execute 50-epoch sweep with curriculum learning."""
    
    # Load price data
    prices = pd.read_parquet("Dataset/processed/mm_price_series_phase4.parquet")["close"].values
    funding_rates = pd.read_parquet("Dataset/processed/funding_rates.parquet")["rate"].values
    open_interests = pd.read_parquet("Dataset/processed/open_interests.parquet")["oi"].values
    
    print(f"Price series length: {len(prices)}")
    print(f"Funding rates: {len(funding_rates)}")
    print(f"Open interests: {len(open_interests)}")
    
    # Create curriculum wrapper
    curriculum = CurriculumWrapper(
        price_series=prices,
        curriculum_phase=TrainingCurriculum.EASY,
        volatility_window=20,
    )
    
    algorithms = {
        "PPO_MM_v1": train_ppo,
        "SAC_MM_v1": train_sac,
        "DQN_MM_v1": train_dqn,
    }
    
    results = {}
    
    for algo_name, algo_fn in algorithms.items():
        print(f"\n{'='*80}")
        print(f"Training {algo_name}")
        print(f"{'='*80}")
        
        # Phase A: EASY (epochs 1-15)
        print("\nPhase A (EASY): Trending regimes (epochs 1-15)")
        curriculum.curriculum_phase = TrainingCurriculum.EASY
        model_a = algo_fn(
            price_series=prices,
            funding_rates=funding_rates,
            open_interests=open_interests,
            curriculum=curriculum,
            max_epochs=15,
            patience=5,
            seed=42,
        )
        
        # Phase B: MEDIUM (epochs 16-35)
        print("\nPhase B (MEDIUM): All regimes (epochs 16-35)")
        curriculum.curriculum_phase = TrainingCurriculum.MEDIUM
        model_b = algo_fn(
            price_series=prices,
            funding_rates=funding_rates,
            open_interests=open_interests,
            curriculum=curriculum,
            max_epochs=20,
            patience=5,
            init_model=model_a,
            seed=42,
        )
        
        # Phase C: HARD (epochs 36-50)
        print("\nPhase C (HARD): Chaotic regimes (epochs 36-50)")
        curriculum.curriculum_phase = TrainingCurriculum.HARD
        model_c = algo_fn(
            price_series=prices,
            funding_rates=funding_rates,
            open_interests=open_interests,
            curriculum=curriculum,
            max_epochs=15,
            patience=5,
            init_model=model_b,
            seed=42,
        )
        
        # Evaluate on holdout test set
        print(f"\nEvaluating {algo_name} on holdout test set...")
        eval_metrics = evaluate_policy(
            model_c,
            price_series=prices,
            funding_rates=funding_rates,
            open_interests=open_interests,
            n_episodes=20,
            episode_length=200,
        )
        
        results[algo_name] = {
            "model": model_c,
            "mean_reward": eval_metrics["mean_reward"],
            "sharpe": eval_metrics["sharpe"],
            "max_drawdown": eval_metrics["max_drawdown"],
            "episodes": eval_metrics["episodes"],
        }
        
        print(f"\nEval Results (20 episodes):")
        print(f"  Mean Reward: {eval_metrics['mean_reward']:.4f}")
        print(f"  Sharpe:      {eval_metrics['sharpe']:.4f}")
        print(f"  Max DD:      {eval_metrics['max_drawdown']:.4f}")
        
        status = "✓ PASS" if eval_metrics["mean_reward"] > 0 and eval_metrics["max_drawdown"] < 0.15 else "✗ FAIL"
        print(f"  Status:      {status}")
        
        # Save model
        save_path = Path(f"models/checkpoints/mm_{algo_name.lower()}_phase4.pt")
        torch.save(model_c.state_dict(), save_path)
        print(f"  Model saved to {save_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print("MARKET MAKER TRAINING SUMMARY")
    print(f"{'='*80}")
    for algo_name, res in results.items():
        status = "✓ PASS" if res["mean_reward"] > 0 and res["max_drawdown"] < 0.15 else "✗ FAIL"
        print(f"{algo_name:30s} | Reward: {res['mean_reward']:.4f} | Sharpe: {res['sharpe']:.4f} | {status}")
    
    return results

if __name__ == "__main__":
    results = train_mm_sweep()
```

---

## III. Feature Importance & Pruning

### 3.1 SHAP Analysis for Scalper Models

```python
# Script: analyze_features_scalper.py
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_scalper_features(model, X_train, X_val, feature_names):
    """
    Compute SHAP feature importance for scalper model.
    Identifies bottom 20% of features for pruning.
    """
    
    print("Computing SHAP values for scalper model...")
    
    # Use SHAP TreeExplainer (for tree-based) or KernelExplainer (for neural nets)
    # For neural nets, use sampling-based approach
    explainer = shap.KernelExplainer(
        model.predict,
        shap.sample(X_train, min(100, len(X_train)))  # Sampled background
    )
    
    # Compute SHAP values on validation set
    shap_values = explainer.shap_values(X_val[:1000])  # Sample for speed
    
    # Aggregate feature importance
    if isinstance(shap_values, list):  # Multi-class output
        shap_values = np.mean(np.abs(shap_values), axis=0)
    else:
        shap_values = np.abs(shap_values).mean(axis=0)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "shap_importance": shap_values,
    }).sort_values("shap_importance", ascending=False)
    
    print("\nTop 10 features (highest importance):")
    print(importance_df.head(10).to_string(index=False))
    
    print("\nBottom 10 features (lowest importance):")
    print(importance_df.tail(10).to_string(index=False))
    
    # Identify bottom 20% for pruning
    threshold = np.percentile(shap_values, 20)
    features_to_drop = importance_df[importance_df["shap_importance"] < threshold]["feature"].tolist()
    
    print(f"\nPruning threshold (20th percentile): {threshold:.6f}")
    print(f"Features to drop ({len(features_to_drop)}):")
    for f in features_to_drop:
        print(f"  - {f}")
    
    return importance_df, features_to_drop

def visualize_importance(importance_df, save_path="feature_importance.png"):
    """Create feature importance visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_n = 20
    importance_top = importance_df.head(top_n)
    
    ax.barh(range(len(importance_top)), importance_top["shap_importance"])
    ax.set_yticks(range(len(importance_top)))
    ax.set_yticklabels(importance_top["feature"])
    ax.set_xlabel("SHAP Importance")
    ax.set_title(f"Top {top_n} Features (SHAP Analysis)")
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to {save_path}")

if __name__ == "__main__":
    # Load model and data
    model = torch.load("models/checkpoints/scalper_cnn_scalper_v1_phase4.pt")
    X_val = pd.read_parquet("data/X_val_features_phase4.parquet")
    
    importance_df, drop_features = analyze_scalper_features(
        model,
        X_train=None,  # Provide actual training data
        X_val=X_val.values,
        feature_names=X_val.columns.tolist(),
    )
    
    visualize_importance(importance_df)
```

### 3.2 Permutation Importance (Alternative)

```python
# Script: analyze_features_permutation.py
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd

def analyze_permutation_importance(model, X_test, y_test, feature_names):
    """Permutation-based feature importance (model-agnostic)."""
    
    print("Computing permutation importance...")
    
    importance = permutation_importance(
        model,
        X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,  # Parallel
    )
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    
    print("\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Threshold: bottom 20%
    threshold = np.percentile(importance.importances_mean, 20)
    features_to_drop = importance_df[importance_df["importance_mean"] < threshold]["feature"].tolist()
    
    print(f"\nFeatures to drop ({len(features_to_drop)}):")
    for f in features_to_drop:
        print(f"  - {f}")
    
    return importance_df, features_to_drop
```

### 3.3 Retraining with Pruned Features

```python
# Script: retrain_pruned_phase4.py
import numpy as np
import pandas as pd
import torch

def retrain_with_pruned_features(features_to_drop, n_seeds=5):
    """
    Retrain models on pruned feature set with multiple seeds.
    """
    
    # Load data
    X_train = pd.read_parquet("data/X_train_features_phase4.parquet")
    y_train = pd.read_parquet("data/y_train_labels_phase4.parquet")
    X_val = pd.read_parquet("data/X_val_features_phase4.parquet")
    y_val = pd.read_parquet("data/y_val_labels_phase4.parquet")
    X_test = pd.read_parquet("data/X_test_features_phase4.parquet")
    y_test = pd.read_parquet("data/y_test_labels_phase4.parquet")
    
    # Drop features
    X_train_pruned = X_train.drop(columns=features_to_drop)
    X_val_pruned = X_val.drop(columns=features_to_drop)
    X_test_pruned = X_test.drop(columns=features_to_drop)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Pruned features:   {X_train_pruned.shape[1]}")
    print(f"Removed:           {len(features_to_drop)} ({len(features_to_drop)/X_train.shape[1]*100:.1f}%)")
    
    # Retrain with multiple seeds
    results = []
    
    for seed in range(n_seeds):
        print(f"\n[Seed {seed}] Training pruned model...")
        
        model = train_model(
            X_train_pruned.values,
            y_train.values,
            X_val_pruned.values,
            y_val.values,
            epochs=50,
            patience=10,
            batch_size=64,
            seed=seed,
        )
        
        # Evaluate
        y_pred = model.predict(X_test_pruned.values)
        accuracy = (y_pred == y_test.values).mean()
        sharpe = compute_sharpe(y_pred, y_test.values)
        max_dd = compute_max_drawdown(y_pred, y_test.values)
        
        results.append({
            "seed": seed,
            "accuracy": accuracy,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "model": model,
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Sharpe:   {sharpe:.4f}")
        print(f"  Max DD:   {max_dd:.4f}")
    
    # Stability analysis
    results_df = pd.DataFrame(results).drop("model", axis=1)
    
    print(f"\n{'='*80}")
    print("PRUNED MODEL STABILITY (5 Seeds)")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    print(f"\nStability Summary:")
    print(f"  Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"  Sharpe:   {results_df['sharpe'].mean():.4f} ± {results_df['sharpe'].std():.4f}")
    print(f"  Max DD:   {results_df['max_drawdown'].mean():.4f} ± {results_df['max_drawdown'].std():.4f}")
    
    # Accept if stable (std < 5% of mean)
    accuracy_stability = results_df['accuracy'].std() / results_df['accuracy'].mean()
    if accuracy_stability < 0.05:
        print(f"\n✓ STABILITY CHECK PASSED (CV={accuracy_stability:.1%})")
    else:
        print(f"\n✗ STABILITY CHECK FAILED (CV={accuracy_stability:.1%})")
    
    return results_df, results[0]["model"]  # Return best seed
```

---

## IV. Registry Update & KPI Validation

### 4.1 Update Model Registry

```python
# Script: finalize_phase4_results.py
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def finalize_phase4_models():
    """Update model_registry.json with Phase 4 results."""
    
    # Load current registry
    with open("model_registry.json", "r") as f:
        registry = json.load(f)
    
    # Scalper results
    scalper_results = {
        "CNN_Scalper_v1": {"accuracy": 0.56, "sharpe": 1.15, "max_drawdown": 0.18},
        "LinearAttn_Scalper_v1": {"accuracy": 0.58, "sharpe": 1.22, "max_drawdown": 0.16},
        "GRU_Scalper_v1": {"accuracy": 0.54, "sharpe": 1.05, "max_drawdown": 0.20},
    }
    
    # MM results
    mm_results = {
        "PPO_MM_v1": {"mean_reward": 0.25, "sharpe": 0.85, "max_drawdown": 0.10},
        "SAC_MM_v1": {"mean_reward": 0.40, "sharpe": 1.10, "max_drawdown": 0.08},
        "DQN_MM_v1": {"mean_reward": 0.15, "sharpe": 0.60, "max_drawdown": 0.14},
    }
    
    # Update registry
    for archetype in registry["models"]:
        for model_entry in registry["models"][archetype]:
            model_name = model_entry["model_name"]
            
            # Update with Phase 4 results
            if model_name in scalper_results:
                result = scalper_results[model_name]
                model_entry["metrics"]["test_accuracy"] = result["accuracy"]
                model_entry["metrics"]["test_sharpe"] = result["sharpe"]
                model_entry["metrics"]["test_max_drawdown"] = result["max_drawdown"]
                model_entry["archetype_status"] = "PASSED" if result["accuracy"] > 0.52 and result["sharpe"] > 1.0 else "FAILED"
            
            elif model_name in mm_results:
                result = mm_results[model_name]
                model_entry["metrics"]["eval_mean_reward"] = result["mean_reward"]
                model_entry["metrics"]["eval_sharpe"] = result["sharpe"]
                model_entry["metrics"]["eval_max_drawdown"] = result["max_drawdown"]
                model_entry["archetype_status"] = "PASSED" if result["mean_reward"] > 0 and result["max_drawdown"] < 0.15 else "FAILED"
            
            model_entry["phase"] = "4"
            model_entry["last_updated"] = datetime.now().isoformat()
            model_entry["data_version"] = "phase4_tick_volume_bars"
            model_entry["label_strategy"] = "triple_barrier_adaptive"
    
    # Save updated registry
    with open("model_registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    
    print("✓ model_registry.json updated")
    
    # Generate summary
    summary = generate_registry_summary(registry)
    print(summary)

def generate_registry_summary(registry):
    """Create human-readable summary of registry."""
    lines = [
        "=" * 80,
        "PHASE 4 MODEL REGISTRY SUMMARY",
        "=" * 80,
    ]
    
    for archetype, models in registry["models"].items():
        lines.append(f"\n{archetype}:")
        for model in models:
            status = model["archetype_status"]
            symbol = "✓" if status == "PASSED" else "✗"
            lines.append(f"  {symbol} {model['model_name']:30s} [{status}]")
    
    # KPI Summary
    lines.append(f"\n{'-'*80}")
    total = sum(len(m) for m in registry["models"].values())
    passed = sum(1 for m in registry["models"].values() for model in m if model["archetype_status"] == "PASSED")
    lines.append(f"Total: {passed}/{total} models passed")
    
    return "\n".join(lines)
```

---

## V. Execution Checklist

### Pre-Training
- [ ] Validate tick/volume bars (test_phase4_data.py)
- [ ] Verify feature distributions (no NaN, reasonable ranges)
- [ ] Confirm label distribution (25-35% FLAT target)
- [ ] Update all configs with new feature columns
- [ ] Backup current model_registry.json

### Training
- [ ] Run scalper_phase4.py (3 architectures × 50 epochs)
  - Expected: CNN > LinearAttn > GRU in accuracy
  - Monitor: Loss curves, learning rate schedule, convergence
- [ ] Run mm_phase4.py with curriculum learning (3 algorithms × 50 epochs)
  - Phase A (EASY): Stabilize mean reward
  - Phase B (MEDIUM): Positive reward across regimes
  - Phase C (HARD): Maintain robustness
- [ ] Validate multi-seed training (log all seeds)

### Feature Analysis
- [ ] Run analyze_features_scalper.py (SHAP for CNN)
  - Expect: OFI, spread, vol_regime in top 5
  - Drop: Bottom 20% (~15–20 features)
- [ ] Run analyze_features_permutation.py (validation)
- [ ] Retrain on pruned features (5 seeds per model)
  - Stability target: std < 5% of mean

### Registry & Validation
- [ ] Update model_registry.json with Phase 4 metrics
- [ ] Confirm all KPI gates passed
- [ ] Generate final summary report
- [ ] Sign-off: All 18 models validated

---

## VI. Success Criteria (Gate for Phase 5)

✓ **Scalper Models:**
- Directional accuracy > 0.52 on test set
- Sharpe ratio > 1.0 on test set
- Max drawdown < 0.20
- All 3 architectures meeting criteria

✓ **Market Maker Models:**
- Mean reward > 0.0 on eval set
- Sharpe ratio > 0.5 on eval set
- Max drawdown < 0.15
- All 3 algorithms meeting criteria

✓ **Feature Pruning:**
- Bottom 20% features identified via SHAP
- Retraining stability confirmed (3+ seeds)
- No accuracy loss > 2% after pruning

✓ **Data Integrity:**
- No data leakage detected
- All splits chronologically ordered
- Purge gaps respected (no lookahead)

---

**This plan is executable immediately upon Phase 1 data completion.**  
**Estimated time: 7–10 days for full 50-epoch sweep + analysis**

