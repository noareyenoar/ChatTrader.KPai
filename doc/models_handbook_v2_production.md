# ChatTrader.KPai — Models Handbook V2
## Production-Grade Specification & Deployment Manual

**Version:** 2.1 (Production)  
**Release Date:** April 30, 2026 | **Updated:** May 19, 2026 (APV-PLN added)  
**Scope:** All 18 neural network models + APV-PLN (Archetype VII) + execution, risk, and ensemble systems  
**Audience:** Traders, quant engineers, AI agents, compliance, and risk management  
**Status:** CRITICAL SYSTEM UPGRADE — From research prototype to institutional deployment

---

## TABLE OF CONTENTS

1. [Version 2 Overview & Breaking Changes](#1-version-2-overview--breaking-changes)
2. [System Architecture & Data Flow](#2-system-architecture--data-flow)
3. [Data Infrastructure & Feature Management](#3-data-infrastructure--feature-management)
4. [Training Pipeline & Hyperparameter Optimization](#4-training-pipeline--hyperparameter-optimization)
5. [The Iron Wall: Temporal Validation Protocol](#5-the-iron-wall-temporal-validation-protocol)
6. [Evaluation Metrics & Validation Gates](#6-evaluation-metrics--validation-gates)
7. [Execution Layer Specification](#7-execution-layer-specification)
8. [Risk Engine & Portfolio Control](#8-risk-engine--portfolio-control)
9. [Ensemble System & Dynamic Weighting](#9-ensemble-system--dynamic-weighting)
10. [RL Environment Specification (Market Maker)](#10-rl-environment-specification-market-maker)
11. [Live Trading Conditions & Safeguards](#11-live-trading-conditions--safeguards)
12. [Monitoring, Drift Detection & Retraining](#12-monitoring-drift-detection--retraining)
13. [Failure Mode Analysis Per Archetype](#13-failure-mode-analysis-per-archetype)
14. [API & Interface Standardization](#14-api--interface-standardization)
15. [Archetype-Specific Procedures](#15-archetype-specific-procedures)
16. [Operational Procedures & Checklists](#16-operational-procedures--checklists)

---

## 1. VERSION 2 OVERVIEW & BREAKING CHANGES

### 1.1 What's New in V2

ChatTrader.KPai V2 transitions from **research prototype (V1)** to **production-grade system**. Critical upgrades:

| Component | V1 | V2 | Impact |
|---|---|---|---|
| **Validation** | Fixed 70/15/15 split | Walk-forward rolling windows | Regime-robust performance measurement |
| **Execution** | Signal output only | Full order generation + execution | Actionable trading orders |
| **Risk** | None | Portfolio-level constraints | Loss prevention |
| **Ensemble** | Example code | Learnable weights + regime switching | Better signal quality |
| **Monitoring** | Offline batch | Real-time performance tracking | Early drift detection |
| **Failure Modes** | Undocumented | Comprehensive per-archetype analysis | Operator awareness |
| **Hyperparameter Search** | Manual | Automated (Optuna/Random) | Better model fit |
| **Multi-Seed Training** | Single run | 3–5 seeds with variance | Robustness confirmation |

### 1.2 Backwards Compatibility

**V2 is NOT backwards compatible with V1 model files.**

Existing V1 checkpoints must be:
1. Reloaded and re-evaluated under V2 walk-forward protocol
2. Re-trained with V2 hyperparameter specifications
3. Re-validated against V2 gates

V1 models should not be deployed alongside V2 models without explicit compatibility testing.

### 1.3 Validation Gate Upgrades

**V1 Gates** (5 criteria, all must pass):
- Sharpe > 1.2
- Accuracy > 55%
- Profit Factor > 1.5
- Max Drawdown < 20%
- OOS/Val Sharpe Decay < 50%

**V2 Gates** (expanded):
- **Sharpe > 1.2** (unchanged)
- **Accuracy > 55%** (unchanged for supervised models; N/A for RL)
- **Profit Factor > 1.5** (unchanged)
- **Max Drawdown < 20%** (unchanged)
- **OOS/Val Sharpe Decay < 50%** (unchanged)
- **NEW: Walk-Forward Sharpe Variance < 30%** (regime robustness)
- **NEW: Calmar Ratio > 1.0** (return per drawdown risk)
- **NEW: Sortino Ratio > 1.5** (return per downside volatility)
- **NEW: Win Rate > 45%** (directional frequency on trades)
- **NEW: Payoff Ratio > 1.0** (avg win size > avg loss size)

---

## 2. SYSTEM ARCHITECTURE & DATA FLOW

### 2.1 End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIVE MARKET DATA STREAM                      │
│         (Binance WebSocket: OHLCV + Level 2 Order Book)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────┐
        │    FEATURE EXTRACTION & ENGINEERING     │
        │  (FeatureFactory, real-time indicators) │
        └────────────┬────────────────────────────┘
                     │
        ┌────────────▼────────────────────────────────────────┐
        │           INFERENCE LAYER (Parallel)                 │
        ├──────────────────────────────────────────────────────┤
        │ Trend      │ Mean-Rev  │ Scalper  │ StatArb │ Disc  │
        │ (3 models) │ (3 models)│(3 models)│(3 models)(3 models)
        │ ┌────────┐ │ ┌───────┐ │ ┌──────┐ │ ┌────┐  │ ┌───┐  │
        │ │Ensemble│ │ │Ensemble│ │Ensemble │ │Ens.│  │ │Ens│  │
        │ └────────┘ │ └───────┘ │ └──────┘ │ └────┘  │ └───┘  │
        │   Signal   │   Signal  │  Signal  │ Signal  │ Signal│
        └────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼────────────────────────────────────────┐
        │      MARKET MAKER RL LAYER (Continuous)             │
        │  (PPO + SAC + DQN consensus quote generation)       │
        └────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────────────────┐
        │       ENSEMBLE AGGREGATION & CONFLICT RESOLUTION      │
        │   (Weighted voting, regime-based weighting)          │
        └────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────────────────┐
        │         RISK ENGINE & POSITION LIMITS                 │
        │  (Max DD guard, correlation limits, leverage cap)    │
        └────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────▼─────────────────────────────────────────┐
        │    EXECUTION ENGINE & ORDER GENERATION                │
        │  (Position sizing, order type selection, TWAP/VWAP)  │
        └────────────┬─────────────────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────────┐
        │     BROKER API / EXCHANGE GATEWAY        │
        │  (Binance Futures, Spot, Options)       │
        └────────────┬────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────────┐
        │       POSITION MANAGEMENT               │
        │  (Fill tracking, slippage measurement)  │
        └────────────┬────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────────┐
        │    PERFORMANCE MONITORING & FEEDBACK    │
        │  (PnL tracking, drift detection, logs)  │
        └─────────────────────────────────────────┘
```

### 2.2 System Components

| Component | Responsibility | Failure Behavior |
|---|---|---|
| **Feature Factory** | Real-time indicator computation | Falls back to cached features; alerts ops |
| **Inference Engine** | Model inference, ensemble voting | Halts signals; queries manual override |
| **Risk Engine** | Position limit enforcement | Rejects order; logs violation |
| **Execution Engine** | Order generation + submission | Retries 3x; falls back to market order |
| **Position Manager** | Fill tracking, P&L | Reconciles with exchange; alerts if mismatch |
| **Monitor** | Performance tracking, drift | Auto-retraining trigger if drift >threshold |

---

## 3. DATA INFRASTRUCTURE & FEATURE MANAGEMENT

### 3.1 Dataset Versioning

Every model is trained on a **versioned dataset snapshot**. Dataset versioning prevents:
- Retraining on different historical data
- Feature drift from preprocessing changes
- Inconsistent comparisons across retraining cycles

**Dataset Version Schema**:
```
Dataset_20260430_v1
├── ID: SHA256(symbol_list + data_hash + feature_schema)
├── symbol_list: [AAVEUSDT, ADAUSDT, ..., ZECUSDT] (34 symbols)
├── date_range: [2024-01-01, 2026-04-30]
├── ohlcv_source: binance_historical (parquet)
├── feature_schema_version: 2.1
├── min_history_bars: 50000
├── purge_gap_bars: 20
├── train/val/test split: 70/15/15 (chronological)
└── checksum: verified before each training run
```

**Implementation**:
```python
# Before training, verify dataset
dataset_id = compute_dataset_id(symbols, data_paths, feature_schema)
assert dataset_id in APPROVED_DATASETS, f"Unknown dataset {dataset_id}"

# Log training dataset for audit trail
log_training_metadata(
    model_name=model_name,
    dataset_id=dataset_id,
    dataset_version=dataset_version,
    timestamp=datetime.now()
)
```

### 3.2 Feature Version Control

Features are explicitly versioned in `data_pipeline/features.py`:

```python
FEATURE_SCHEMA_VERSION = "2.1"  # Increment when features change

TREND_FEATURES_v2_1 = [
    ("log_return", "Causal, 1-bar return"),
    ("zscore_close_64", "64-bar z-score of close"),
    ("ema_spread_12_26", "Difference of 12/26 EMAs"),
    ("atr_14", "14-bar Average True Range"),
    ("price_slope_20", "Linear regression slope, 20 bars"),
]

# If adding features, increment version and update:
# 1. FEATURE_SCHEMA_VERSION
# 2. TREND_FEATURES_v2_2 (new name)
# 3. FeatureFactory methods
# 4. Training config validation
```

**Feature Drift Detection Hook**:
```python
def detect_feature_drift(live_features, train_feature_stats, threshold=0.05):
    """
    Compare live feature distributions to training statistics.
    Returns: drift_detected (bool), drift_metrics (dict)
    """
    for feature_name, live_data in live_features.items():
        train_mean, train_std = train_feature_stats[feature_name]
        live_mean, live_std = np.mean(live_data), np.std(live_data)
        
        kl_div = kl_divergence(live_data, train_data)  # KL divergence
        if kl_div > threshold:
            alert(f"Feature drift detected: {feature_name} (KL={kl_div:.4f})")
            return True, {"feature": feature_name, "kl_divergence": kl_div}
    
    return False, {}
```

### 3.3 Data Quality Validation

**Pre-Training Checks**:
```python
def validate_training_data(train_df, val_df, test_df, config):
    """
    Verify data integrity, no leakage, label distribution.
    Raises: DataValidationError if any check fails.
    """
    
    # Check 1: No temporal overlap
    assert train_df.timestamp.max() < val_df.timestamp.min(), "Data leakage: train/val overlap"
    assert val_df.timestamp.max() < test_df.timestamp.min(), "Data leakage: val/test overlap"
    
    # Check 2: No NaN/Inf
    assert not train_df.isnull().any().any(), "NaN values in training set"
    assert np.all(np.isfinite(train_df.values)), "Inf values in training set"
    
    # Check 3: Label distribution (should not be heavily imbalanced)
    if 'target' in train_df.columns:
        value_counts = train_df['target'].value_counts()
        ratios = value_counts / len(train_df)
        imbalance = np.max(ratios) / np.min(ratios)
        assert imbalance < 5.0, f"Severe class imbalance: {imbalance:.1f}x"
    
    # Check 4: Sufficient samples
    min_samples = config.get('min_train_samples', 100000)
    assert len(train_df) >= min_samples, f"Insufficient train samples: {len(train_df)} < {min_samples}"
    
    print(f"✓ Data validation passed: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
```

---

## 4. TRAINING PIPELINE & HYPERPARAMETER OPTIMIZATION

### 4.1 Multi-Seed Training

Every model is trained 3 times with different random seeds to:
1. Estimate variance in final performance
2. Select the best seed's model for deployment
3. Compute confidence intervals on metrics

```python
TRAINING_SEEDS = [42, 123, 456]  # Fixed for reproducibility

def train_model_multi_seed(config, seeds=TRAINING_SEEDS):
    results = {}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        result = train_single_seed(config, seed)
        results[seed] = result
    
    # Select best seed by validation Sharpe
    best_seed = max(results.keys(), key=lambda s: results[s]['val_sharpe'])
    
    return results[best_seed], {
        'seeds': TRAINING_SEEDS,
        'mean_sharpe': np.mean([r['val_sharpe'] for r in results.values()]),
        'std_sharpe': np.std([r['val_sharpe'] for r in results.values()]),
        'best_seed': best_seed
    }
```

### 4.2 Hyperparameter Search (Optuna + Random)

**Search Space Per Archetype**:

| Component | Search Range | Default |
|---|---|---|
| **Learning Rate** | [0.00001, 0.01] | 0.001 |
| **Batch Size** | [32, 64, 128, 256, 512, 1024] | 1024 |
| **Hidden Dimension** | [32, 64, 128, 256, 512] | varies |
| **Dropout Rate** | [0.0, 0.1, 0.2, 0.3, 0.5] | 0.1 |
| **Optimizer** | [Adam, AdamW, SGD, RMSprop] | Adam |
| **Activation** | [ReLU, GELU, LeakyReLU, ELU] | ReLU |
| **Gradient Clip** | [0.5, 1.0, 5.0, None] | 1.0 |
| **Weight Decay** | [0.0, 0.00001, 0.0001, 0.001] | 0.0 |

**Optuna Search Configuration**:
```python
def create_optuna_study(archetype, n_trials=200):
    """
    Create an Optuna study for hyperparameter optimization.
    n_trials=200 takes ~8 hours on 4 parallel workers.
    """
    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024])
        dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        activation = trial.suggest_categorical('activation', ['ReLU', 'GELU', 'LeakyReLU', 'ELU'])
        
        # Train model with proposed hyperparameters
        config = make_config(archetype, lr=lr, batch_size=batch_size, dropout=dropout, activation=activation)
        result = train_model_multi_seed(config, seeds=[42])  # Seed=42 for speed
        
        # Optuna maximizes objective; return validation Sharpe
        return result['val_sharpe']
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, n_jobs=4)  # 4 parallel workers
    
    return study.best_trial.params
```

**Runtime Estimate**:
- Single training run: 4–8 hours (varies by archetype)
- 200 trials with 4 workers: ~8 hours total
- Hyperparameter search per archetype: **8 hours**
- Full system (6 archetypes): **48 hours**

### 4.3 Early Stopping & Patience

```python
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4, metric='val_sharpe', mode='max'):
        """
        patience: epochs without improvement before stopping
        delta: minimum change to qualify as improvement
        metric: which validation metric to monitor
        mode: 'max' (sharpe, accuracy) or 'min' (loss)
        """
        self.patience = patience
        self.delta = delta
        self.metric = metric
        self.mode = mode
        self.best_value = -np.inf if mode == 'max' else np.inf
        self.counter = 0
    
    def __call__(self, current_value):
        if self.mode == 'max':
            improved = current_value > self.best_value + self.delta
        else:
            improved = current_value < self.best_value - self.delta
        
        if improved:
            self.best_value = current_value
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
            return False
```

### 4.4 Failure Detection

Training halts immediately if:
1. **NaN Loss**: Model outputs NaN or Inf (numerical instability)
2. **Divergence**: Loss increases consistently for 5 epochs (learning rate too high)
3. **Gradient Explosion**: Gradient norm >100 (clip or reduce LR)
4. **OOM**: CUDA out of memory (reduce batch size or model dimension)

```python
def check_training_health(loss, gradient_norm, epoch, max_epochs):
    """Raises TrainingFailureError if health checks fail."""
    
    if np.isnan(loss) or np.isinf(loss):
        raise TrainingFailureError(f"NaN/Inf loss at epoch {epoch}")
    
    if gradient_norm > 100:
        raise TrainingFailureError(f"Gradient explosion at epoch {epoch}: norm={gradient_norm:.1f}")
    
    if epoch > 5 and loss > self.previous_loss * 1.1:  # 10% increase
        self.divergence_count += 1
        if self.divergence_count >= 5:
            raise TrainingFailureError(f"Loss diverging at epoch {epoch}")
```

---

## 5. THE IRON WALL: TEMPORAL VALIDATION PROTOCOL

### 5.1 Walk-Forward Validation (WFV)

**Purpose**: Detect regime shifts and overfitting to specific historical periods.

**Schema**:
```
Walk 1: [0-45%] train → [45-58%] val → [58-70%] test
Walk 2: [5-50%] train → [50-63%] val → [63-75%] test
Walk 3: [10-55%] train → [55-68%] val → [68-80%] test
Walk 4: [15-60%] train → [60-73%] val → [73-85%] test
Walk 5: [20-65%] train → [65-78%] val → [78-90%] test
Walk 6: [25-70%] train → [70-83%] val → [83-95%] test
Walk 7: [30-75%] train → [75-88%] val → [88-100%] test
```

**Implementation**:
```python
def walk_forward_validation(data, n_walks=7, train_pct=0.65, val_pct=0.13, test_pct=0.22):
    """
    Generate walk-forward train/val/test splits.
    Each walk slides forward by ~10% of total data.
    """
    total_len = len(data)
    walk_stride = int(total_len / (n_walks + 1))
    
    walks = []
    for i in range(n_walks):
        start_idx = i * walk_stride
        train_end = int(start_idx + train_pct * total_len)
        val_end = int(start_idx + (train_pct + val_pct) * total_len)
        test_end = int(start_idx + (train_pct + val_pct + test_pct) * total_len)
        
        train = data[start_idx:train_end]
        val = data[train_end:val_end]
        test = data[val_end:test_end]
        
        walks.append({
            'walk': i,
            'train': train,
            'val': val,
            'test': test,
            'date_range': (data.iloc[start_idx]['timestamp'], data.iloc[test_end-1]['timestamp'])
        })
    
    return walks

def evaluate_walk_forward(model, walks):
    """
    Train on each walk and collect metrics.
    """
    walk_results = []
    
    for walk in walks:
        # Train on this walk's training set
        result = train_and_evaluate(model, walk['train'], walk['val'], walk['test'])
        walk_results.append(result)
    
    # Aggregate across walks
    mean_sharpe = np.mean([r['test_sharpe'] for r in walk_results])
    std_sharpe = np.std([r['test_sharpe'] for r in walk_results])
    
    # Check variance
    if std_sharpe / mean_sharpe > 0.30:  # >30% variance
        alert(f"High Sharpe variance across walks: {std_sharpe:.3f} (std)")
        return False  # Fail robustness check
    
    return True, walk_results
```

### 5.2 Regime-Stratified Evaluation

Models are evaluated separately for each market regime to identify blind spots:

```python
REGIMES = {
    'trending_up': lambda df: df['log_return'].rolling(20).mean() > 0.001,
    'trending_down': lambda df: df['log_return'].rolling(20).mean() < -0.001,
    'mean_revert': lambda df: abs(df['zscore_close_64']) > 1.5,
    'high_vol': lambda df: df['atr_14'].rolling(20).mean() > df['atr_14'].quantile(0.75),
    'low_vol': lambda df: df['atr_14'].rolling(20).mean() < df['atr_14'].quantile(0.25),
    'sideways': lambda df: ~(trending_up | trending_down | mean_revert),
}

def evaluate_by_regime(model, test_df, predictions):
    """
    Separate test-set performance by regime.
    """
    regime_results = {}
    
    for regime_name, regime_filter in REGIMES.items():
        mask = regime_filter(test_df)
        regime_test = test_df[mask]
        regime_preds = predictions[mask]
        
        if len(regime_test) > 100:  # Only evaluate if sufficient samples
            metrics = compute_metrics(regime_test, regime_preds)
            regime_results[regime_name] = metrics
    
    return regime_results
```

---

## 6. EVALUATION METRICS & VALIDATION GATES

### 6.1 Comprehensive Metrics Library

| Metric | Formula | V2 Gate | Interpretation |
|---|---|---|---|
| **Sharpe Ratio** | μ_PnL / σ_PnL × √(252×24×12) | >1.2 | Risk-adjusted return, annualized |
| **Calmar Ratio** | μ_PnL / MaxDD | >1.0 | Return per drawdown magnitude |
| **Sortino Ratio** | μ_PnL / σ_downside | >1.5 | Return per downside volatility only |
| **Profit Factor** | ∑wins / ∑losses | >1.5 | Gross profitability |
| **Win Rate** | (# wins) / (# trades) | >45% | Frequency of positive trades |
| **Payoff Ratio** | avg_win_size / avg_loss_size | >1.0 | Average win > average loss |
| **Max Drawdown** | (peak_to_trough) / peak | <20% | Worst equity decline |
| **Directional Accuracy** | (# correct pred) / (# total) | >55% | Trend prediction correctness (supervised) |
| **MAE** | mean( |pred - actual| ) | archetype-specific | Regression error (StatArb) |
| **F1-Score** | 2×(P×R)/(P+R) | >0.55 | Classification balance (precision+recall) |
| **Walk-Forward Sharpe Variance** | σ(sharpes across 7 walks) / μ(sharpes) | <30% | Regime robustness |

### 6.2 Validation Gate Logic

```python
def evaluate_model(model, test_df, predictions, config, archetype):
    """
    Comprehensive evaluation against all V2 gates.
    Returns: (is_valid, metrics_dict, failure_reasons)
    """
    metrics = {}
    failures = []
    
    # Compute all metrics
    metrics['sharpe'] = compute_sharpe(predictions['pnl'])
    metrics['calmar'] = compute_calmar(predictions['pnl'], predictions['drawdown'])
    metrics['sortino'] = compute_sortino(predictions['pnl'])
    metrics['profit_factor'] = compute_profit_factor(predictions['trades'])
    metrics['win_rate'] = compute_win_rate(predictions['trades'])
    metrics['payoff_ratio'] = compute_payoff_ratio(predictions['trades'])
    metrics['max_drawdown'] = compute_max_drawdown(predictions['equity_curve'])
    
    if archetype in ['trend', 'mean_reversion', 'scalper', 'discretionary']:
        metrics['accuracy'] = compute_accuracy(test_df, predictions)
    if archetype == 'stat_arb':
        metrics['mae'] = compute_mae(test_df, predictions)
    if archetype == 'discretionary':
        metrics['f1'] = compute_f1(test_df, predictions)
    
    # Gate checks
    if metrics['sharpe'] < 1.2:
        failures.append(f"Sharpe {metrics['sharpe']:.2f} < 1.2")
    if metrics['profit_factor'] < 1.5:
        failures.append(f"Profit Factor {metrics['profit_factor']:.2f} < 1.5")
    if metrics['max_drawdown'] > 0.20:
        failures.append(f"Max Drawdown {metrics['max_drawdown']:.1%} > 20%")
    if metrics['calmar'] < 1.0:
        failures.append(f"Calmar {metrics['calmar']:.2f} < 1.0")
    if metrics['sortino'] < 1.5:
        failures.append(f"Sortino {metrics['sortino']:.2f} < 1.5")
    if metrics['win_rate'] < 0.45:
        failures.append(f"Win Rate {metrics['win_rate']:.1%} < 45%")
    if metrics['payoff_ratio'] < 1.0:
        failures.append(f"Payoff Ratio {metrics['payoff_ratio']:.2f} < 1.0")
    
    # Archetype-specific gates
    if archetype in ['trend', 'mean_reversion', 'scalper', 'discretionary']:
        if metrics.get('accuracy', 0.5) < 0.55:
            failures.append(f"Accuracy {metrics.get('accuracy', 0):.1%} < 55%")
    if archetype == 'stat_arb':
        if metrics.get('mae', 1000) > 0.05:
            failures.append(f"MAE {metrics.get('mae', 1000):.4f} > 0.05")
    
    is_valid = len(failures) == 0
    
    return is_valid, metrics, failures
```

---

## 7. EXECUTION LAYER SPECIFICATION

### 7.1 Signal-to-Order Transformation

**Input**: Model signals (continuous predictions or class probabilities)  
**Output**: Executable limit/market orders

```python
class ExecutionEngine:
    def __init__(self, exchange_api, position_manager, risk_engine):
        self.exchange = exchange_api
        self.positions = position_manager
        self.risk = risk_engine
    
    def signal_to_order(self, signal, current_price, symbol, timestamp):
        """
        Convert model signal to executable order.
        
        signal: dict with keys:
            - 'archetype': str (trend, mr, scalper, etc.)
            - 'direction': float (-1, 0, +1 or continuous)
            - 'confidence': float (0–1)
            - 'target_size': float (units)
        
        Returns: dict with keys:
            - 'order_id': str
            - 'symbol': str
            - 'side': 'BUY' | 'SELL'
            - 'quantity': float
            - 'order_type': 'LIMIT' | 'MARKET' | 'TWAP' | 'VWAP'
            - 'limit_price': float (if LIMIT)
            - 'time_in_force': 'GTC' | 'IOC' | 'FOK'
            - 'timestamp': datetime
        """
        
        # Step 1: Check confidence minimum
        min_confidence = self.risk.get_min_confidence(signal['archetype'])
        if signal['confidence'] < min_confidence:
            return None  # Signal too weak; skip
        
        # Step 2: Compute position size
        position_size = self.compute_position_size(
            signal['direction'],
            signal['confidence'],
            current_price,
            symbol
        )
        
        # Step 3: Check risk constraints
        if not self.risk.position_within_limits(symbol, position_size):
            return None  # Would violate position limits
        
        # Step 4: Determine order type
        if signal['archetype'] == 'scalper':
            order_type = 'LIMIT'  # Scalper needs tight timing
            limit_price = self._compute_limit_price(current_price, signal['direction'])
            time_in_force = 'IOC'  # Immediate or cancel
        elif signal['archetype'] == 'market_maker':
            order_type = 'LIMIT'  # Market maker always limits
            limit_price = self._compute_market_maker_price(signal, current_price)
            time_in_force = 'GTC'  # Good till cancel
        else:
            # Trend, Mean Rev, StatArb, Discretionary: can use market
            if abs(signal['confidence'] - 0.5) > 0.3:  # High confidence → market
                order_type = 'MARKET'
            else:
                order_type = 'LIMIT'
                limit_price = self._compute_limit_price(current_price, signal['direction'])
            time_in_force = 'GTC'
        
        # Step 5: Generate order
        order = {
            'order_id': self._generate_order_id(),
            'symbol': symbol,
            'side': 'BUY' if signal['direction'] > 0 else 'SELL',
            'quantity': abs(position_size),
            'order_type': order_type,
            'limit_price': limit_price if order_type == 'LIMIT' else None,
            'time_in_force': time_in_force,
            'timestamp': timestamp,
            'signal_archetype': signal['archetype'],
            'signal_confidence': signal['confidence'],
        }
        
        return order
    
    def compute_position_size(self, direction, confidence, current_price, symbol):
        """
        Position sizing: Kelly Criterion variant.
        
        size = (edge × account_pct) / RR_ratio
        where edge = win_rate - loss_rate
        """
        base_size = self.risk.max_position_size(symbol)  # Max units per symbol
        confidence_scalar = (confidence - 0.5) * 2  # Scale to [0, 1]
        
        # Kelly fraction: f* = (2*p - 1) / b, capped at 20% Kelly for safety
        p = 0.45 + confidence_scalar * 0.1  # Confidence → win probability
        risk_reward = 1.5  # Payoff ratio
        kelly_fraction = (2 * p - 1) / risk_reward
        kelly_fraction = np.clip(kelly_fraction, 0, 0.2)  # Cap at 20% Kelly
        
        position_size = base_size * kelly_fraction
        return position_size
    
    def _compute_limit_price(self, current_price, direction):
        """Limit price: bid down if buying, offer up if selling."""
        spread_bps = 5  # 5 basis points from mid
        spread_amount = current_price * spread_bps / 10000
        
        if direction > 0:  # Buy limit: bid down
            return current_price - spread_amount
        else:  # Sell limit: offer up
            return current_price + spread_amount
    
    def _compute_market_maker_price(self, signal, current_price):
        """Market maker: quote width based on inventory."""
        inventory_level = signal.get('inventory_level', 0)  # -1 (short) to +1 (long)
        
        # If long: offer higher to sell
        # If short: bid lower to buy
        bias = inventory_level * 10  # bps
        spread_bps = 20  # 20 bps base spread
        
        limit_price = current_price + (bias - spread_bps/2) / 10000 * current_price
        return limit_price
```

### 7.2 Order Execution & Fill Tracking

```python
class OrderExecutor:
    def __init__(self, exchange_api):
        self.exchange = exchange_api
        self.pending_orders = {}
    
    def execute_order(self, order, max_retries=3):
        """
        Submit order to exchange with retry logic.
        """
        for attempt in range(max_retries):
            try:
                # Submit order to exchange
                result = self.exchange.submit_order(
                    symbol=order['symbol'],
                    side=order['side'],
                    quantity=order['quantity'],
                    order_type=order['order_type'],
                    limit_price=order.get('limit_price'),
                    time_in_force=order['time_in_force'],
                    client_order_id=order['order_id']
                )
                
                order['exchange_order_id'] = result['order_id']
                order['status'] = 'PENDING'
                order['submission_timestamp'] = datetime.now()
                
                self.pending_orders[result['order_id']] = order
                
                return order
            
            except ExchangeException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Final retry failed; escalate
                    alert(f"Order submission failed after {max_retries} retries: {e}")
                    order['status'] = 'FAILED'
                    return order
    
    def track_fills(self):
        """
        Poll exchange for fill updates on pending orders.
        """
        for order_id, order in list(self.pending_orders.items()):
            status = self.exchange.order_status(order_id)
            
            if status['status'] == 'FILLED':
                order['status'] = 'FILLED'
                order['filled_quantity'] = status['filled_qty']
                order['average_fill_price'] = status['avg_price']
                order['fill_timestamp'] = status['fill_time']
                
                # Record slippage: difference between intended limit and actual fill
                if order['order_type'] == 'LIMIT':
                    slippage = abs(order['average_fill_price'] - order['limit_price'])
                    order['slippage'] = slippage
                
                del self.pending_orders[order_id]
            
            elif status['status'] == 'CANCELLED':
                order['status'] = 'CANCELLED'
                del self.pending_orders[order_id]
            
            elif status['status'] == 'REJECTED':
                order['status'] = 'REJECTED'
                del self.pending_orders[order_id]
```

### 7.3 Latency Requirements

| Archetype | Max Latency | Implementation |
|---|---|---|
| **Market Maker** | 50 ms | Real-time vectorized inference, async order submission |
| **Scalper** | 100 ms | Low-latency feature computation, IOC orders |
| **Trend Follower** | 500 ms | Batch inference every 5 minutes |
| **Mean Reversion** | 500 ms | Batch inference every 5 minutes |
| **StatArb** | 1000 ms | Batch inference every 10 minutes |
| **Discretionary** | 1000 ms | Batch inference every 10 minutes |

---

## 8. RISK ENGINE & PORTFOLIO CONTROL

### 8.1 Position Limits & Constraints

```python
class RiskEngine:
    def __init__(self, config):
        self.max_leverage = config['max_leverage']  # Default: 3.0×
        self.max_position_per_symbol = config['max_position_per_symbol']  # 5% of account per symbol
        self.max_correlated_exposure = config['max_correlated_exposure']  # 20% to highly correlated assets
        self.max_portfolio_dd = config['max_portfolio_dd']  # 25% max drawdown
        self.max_intraday_loss = config['max_intraday_loss']  # 2% stop-loss per day
    
    def check_position_limits(self, symbol, quantity, current_price):
        """
        Verify proposed position satisfies all constraints.
        Returns: (allowed, reason_if_denied)
        """
        
        # Check 1: Position size per symbol
        notional = quantity * current_price
        max_notional = self.account_equity * self.max_position_per_symbol
        if notional > max_notional:
            return False, f"Position ${notional:.0f} exceeds max ${max_notional:.0f}"
        
        # Check 2: Leverage constraint
        new_total_notional = self._compute_total_notional() + notional
        if new_total_notional > self.account_equity * self.max_leverage:
            return False, f"Leverage would be {new_total_notional / self.account_equity:.1f}x, max is {self.max_leverage:.1f}x"
        
        # Check 3: Correlation limit
        if self._is_highly_correlated(symbol):
            correlated_exposure = self._compute_correlated_exposure(symbol)
            if correlated_exposure + notional > self.account_equity * self.max_correlated_exposure:
                return False, f"Correlated exposure ${correlated_exposure + notional:.0f} exceeds ${self.account_equity * self.max_correlated_exposure:.0f}"
        
        # Check 4: Portfolio drawdown guard
        if self._portfolio_drawdown() > self.max_portfolio_dd:
            return False, f"Portfolio drawdown {self._portfolio_drawdown():.1%} exceeds max {self.max_portfolio_dd:.1%}"
        
        # Check 5: Intraday loss limit
        if self._intraday_loss() > self.max_intraday_loss:
            return False, f"Intraday loss {self._intraday_loss():.1%} exceeds max {self.max_intraday_loss:.1%}"
        
        return True, None
    
    def _is_highly_correlated(self, symbol):
        """Check if symbol is correlated (>0.7) with other held positions."""
        for held_symbol in self.positions.keys():
            correlation = self.compute_correlation(symbol, held_symbol)
            if correlation > 0.7:
                return True
        return False
    
    def emergency_stop(self):
        """
        Halt all trading immediately.
        Triggered when:
        - Portfolio DD > 50%
        - Intraday loss > 5%
        - Model drift detected + Sharpe <0.5 recently
        """
        for symbol in list(self.positions.keys()):
            self.close_position(symbol)  # Liquidate all positions at market
        
        self.trading_halted = True
        alert("EMERGENCY STOP: All positions closed", severity='CRITICAL')
```

### 8.2 Kill-Switch Conditions

Trading is automatically halted if any of:

1. **Portfolio Drawdown > 50%** (stop total losses)
2. **Daily Loss > 5%** (stop intraday bleeding)
3. **Model Sharpe < 0.5 (rolling 50 trades)** (stop deteriorating models)
4. **Feature Drift KL Divergence > 0.10** (stop predictions in new regime)
5. **Exchange Connectivity Lost** (stop blind trading)
6. **Manual Operator Command** (always honored)

```python
def check_kill_switch_conditions():
    """
    Executed every 1 minute.
    """
    conditions_triggered = []
    
    if portfolio_drawdown() > 0.50:
        conditions_triggered.append("Portfolio DD > 50%")
    
    if intraday_loss() > 0.05:
        conditions_triggered.append("Intraday Loss > 5%")
    
    if rolling_sharpe(windows=50) < 0.5:
        conditions_triggered.append("Model Sharpe < 0.5 (50-trade window)")
    
    if feature_drift() > 0.10:
        conditions_triggered.append(f"Feature Drift KL > 0.10")
    
    if not exchange_api.is_connected():
        conditions_triggered.append("Exchange Disconnected")
    
    if manual_stop_requested():
        conditions_triggered.append("Manual Stop Requested")
    
    if conditions_triggered:
        trigger_emergency_stop(conditions_triggered)
        return True
    
    return False
```

---

## 9. ENSEMBLE SYSTEM & DYNAMIC WEIGHTING

### 9.1 Ensemble Architecture

Each archetype (except RL) runs 3 models in parallel. Ensemble output combines predictions:

```
          Trend Input
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
  LSTM  Transformer  TCN
     │        │        │
     └────────┼────────┘
              │
        ┌─────▼────────────────┐
        │ ENSEMBLE AGGREGATION  │
        ├───────────────────────┤
        │ Weighted voting:      │
        │ w_LSTM=0.3            │
        │ w_Transformer=0.35    │
        │ w_TCN=0.35            │
        │                       │
        │ output = weighted avg │
        │ confidence = entropy  │
        └─────┬────────────────┘
              │
          Ensemble
          Signal
```

### 9.2 Learnable Ensemble Weights

Weights are initialized from validation Sharpe but adapt based on live performance:

```python
class EnsembleWeighting:
    def __init__(self, archetype, model_names):
        self.archetype = archetype
        self.model_names = model_names  # ['LSTM', 'Transformer', 'TCN']
        self.weights = {model: 1.0 / len(model_names) for model in model_names}
        self.performance_history = {model: [] for model in model_names}
    
    def initialize_from_validation(self, val_metrics_dict):
        """
        Set initial weights proportional to validation Sharpe.
        weights ∝ exp(sharpe / temperature)
        """
        sharpes = [val_metrics_dict[model]['val_sharpe'] for model in self.model_names]
        
        if all(s > 0 for s in sharpes):  # All models are profitable
            # Softmax weighting
            temperatures = 1.0
            unnormalized = np.exp(np.array(sharpes) / temperatures)
            self.weights = {
                model: w for model, w in zip(self.model_names, unnormalized / unnormalized.sum())
            }
        else:
            # Use equal weights if some models are unprofitable
            self.weights = {model: 1.0 / len(self.model_names) for model in self.model_names}
    
    def update_weights(self, recent_trades, window=50):
        """
        Adapt weights based on recent performance.
        Run every 100 trades.
        """
        # Compute recent Sharpe for each model's signals
        for model in self.model_names:
            model_trades = [t for t in recent_trades if t['signal_from_model'] == model]
            
            if len(model_trades) > 10:
                sharpe = compute_sharpe([t['pnl'] for t in model_trades])
                self.performance_history[model].append(sharpe)
        
        # Reweight proportional to recent Sharpe
        recent_sharpes = {
            model: np.mean(self.performance_history[model][-5:])  # Last 5 periods
            for model in self.model_names
        }
        
        # Update weights: moving average of 90% old + 10% new
        new_weights = {}
        for model in self.model_names:
            sharpe = recent_sharpes[model]
            if sharpe > 0:
                new_weight = np.exp(sharpe)
            else:
                new_weight = 0.1  # Penalize negative Sharpe
            new_weights[model] = new_weight
        
        # Normalize
        total = sum(new_weights.values())
        new_weights = {model: w / total for model, w in new_weights.items()}
        
        # Smooth: 90% old + 10% new
        for model in self.model_names:
            self.weights[model] = 0.9 * self.weights[model] + 0.1 * new_weights[model]
    
    def aggregate_predictions(self, predictions_dict):
        """
        Combine predictions from 3 models.
        
        predictions_dict: {
            'LSTM': [0.6, -0.2, 0.4, ...],
            'Transformer': [0.55, -0.15, 0.42, ...],
            'TCN': [0.58, -0.18, 0.39, ...]
        }
        
        Returns: (ensemble_signal, confidence, ensemble_entropy)
        """
        
        # Weighted average
        ensemble = np.zeros_like(predictions_dict['LSTM'])
        for model in self.model_names:
            ensemble += self.weights[model] * np.array(predictions_dict[model])
        
        # Confidence: inverse of entropy (agreement among models)
        predictions_array = np.array([
            predictions_dict[model] for model in self.model_names
        ])  # Shape: (3, N)
        
        # Disagreement: std across models per sample
        disagreement = np.std(predictions_array, axis=0)  # Shape: (N,)
        confidence = 1.0 / (1.0 + disagreement)  # Inverse relationship
        
        # Entropy: measure of disagreement
        entropy = -np.sum([
            self.weights[model] * np.abs(predictions_dict[model] - ensemble) 
            for model in self.model_names
        ], axis=0)
        
        return ensemble, confidence, entropy
```

### 9.3 Conflict Resolution

When models disagree strongly, ensemble confidence drops and signal size shrinks:

```python
def resolve_conflicts(ensemble_signal, confidence, max_disagreement_threshold=0.5):
    """
    If models strongly disagree, reduce signal size.
    """
    
    # Map confidence to position size multiplier
    if confidence > 0.7:
        size_multiplier = 1.0  # Full size
    elif confidence > 0.5:
        size_multiplier = 0.7  # Reduced size
    else:
        size_multiplier = 0.3  # Very cautious
    
    # Also: if prediction is weak (ensemble signal near 0), reduce size
    signal_magnitude = abs(ensemble_signal)
    if signal_magnitude < 0.2:
        size_multiplier *= 0.5
    
    return ensemble_signal * size_multiplier
```

---

## 10. RL ENVIRONMENT SPECIFICATION (MARKET MAKER)

### 10.1 Simulated Order Book

```python
class SimulatedOrderBook:
    """
    Realistic order book simulation for market maker training.
    Includes: stochastic fill probabilities, queue position, partial fills.
    """
    
    def __init__(self, mid_price, spread_bps=20, depth=5):
        self.mid_price = mid_price
        self.bid_ask_spread = spread_bps / 10000  # Convert bps to decimal
        self.bid_price = mid_price - (mid_price * self.bid_ask_spread / 2)
        self.ask_price = mid_price + (mid_price * self.bid_ask_spread / 2)
        
        # Order book depth: how many levels deep
        self.depth = depth
        
        # Current orders at each price level
        self.bids = {}  # price → quantity
        self.asks = {}  # price → quantity
        
        # Queue position: how deep in the queue is our order?
        self.bid_queue_position = 0
        self.ask_queue_position = 0
    
    def submit_limit_order(self, side, price, quantity):
        """
        Submit a limit order.
        side: 'BUY' | 'SELL'
        
        Returns: immediately if not fully filled, partial fill possible.
        """
        if side == 'BUY':
            self.bids[price] = quantity
            # Queue position: how many other bids are ahead?
            self.bid_queue_position = len([p for p in self.bids.keys() if p > price])
        else:  # SELL
            self.asks[price] = quantity
            self.ask_queue_position = len([p for p in self.asks.keys() if p < price])
    
    def cancel_order(self, side, price):
        """Cancel order at given price level."""
        if side == 'BUY':
            if price in self.bids:
                del self.bids[price]
        else:
            if price in self.asks:
                del self.asks[price]
    
    def simulate_fill(self, side, queue_position, order_age_seconds=1):
        """
        Stochastic order fill probability.
        
        Factors:
        - Queue position: deeper queue → lower fill probability
        - Market volatility: high vol → higher fill probability (more aggressive orders flow)
        - Order age: older orders → higher fill probability (decay of better orders)
        
        Returns: (is_filled, fill_percentage)
        """
        
        # Base fill probability based on queue position
        # Position 0 (top of book): 80% chance per second
        # Position 1: 60%, Position 2: 40%, etc.
        base_fill_prob = max(0.1, 0.9 - 0.2 * queue_position)
        
        # Adjust for order age (persistence helps)
        age_factor = 1.0 + 0.01 * order_age_seconds
        
        # Adjust for market volatility (high vol = more fills)
        vol_factor = 1.0 + 0.5 * self.volatility
        
        final_fill_prob = np.clip(base_fill_prob * age_factor * vol_factor, 0, 1)
        
        # Partial fill: can be 50%, 75%, or 100%
        if np.random.random() < final_fill_prob:
            partial_pct = np.random.choice([0.5, 0.75, 1.0], p=[0.2, 0.3, 0.5])
            return True, partial_pct
        else:
            return False, 0.0
```

### 10.2 Market Maker Reward Structure

```python
class MMRewardComputer:
    """
    Reward function for market maker RL.
    """
    
    def __init__(self, config):
        self.config = config
        self.pnl_scale = config['pnl_reward_scale']  # 0.001 (small scale)
        self.inventory_penalty = config['inventory_penalty']  # 100
        self.survival_bonus = config['survival_bonus']  # 0.01 per step
        self.drawdown_termination = config['max_episode_drawdown']  # -0.5 (terminate if -50% from peak)
    
    def compute_reward(self, state, action, next_state):
        """
        Reward = PnL + Inventory Penalty + Survival Bonus
        
        state: (inventory, bid_spread, ask_spread, volatility, ...)
        action: (bid_price_offset, ask_price_offset, target_inventory)
        next_state: (new_inventory, new_bid_spread, ..., new_pnl)
        """
        
        # Extract PnL (assumes next_state includes cumulative PnL)
        pnl = next_state['pnl'] - state['pnl']
        pnl_reward = pnl * self.pnl_scale
        
        # Inventory penalty: penalize holding large positions
        inventory = next_state['inventory']
        inventory_penalty = -self.inventory_penalty * abs(inventory) ** 1.5
        
        # Survival bonus: small reward per step (encourages long episodes)
        survival_bonus = self.survival_bonus
        
        total_reward = pnl_reward + inventory_penalty + survival_bonus
        
        return total_reward, {
            'pnl_reward': pnl_reward,
            'inventory_penalty': inventory_penalty,
            'survival_bonus': survival_bonus,
        }
    
    def check_episode_termination(self, episode_pnl, peak_pnl):
        """
        Terminate episode if drawdown exceeds threshold.
        """
        drawdown = (episode_pnl - peak_pnl) / (abs(peak_pnl) + 1e-8)
        
        if drawdown < self.drawdown_termination:
            return True, f"Drawdown {drawdown:.1%} exceeds limit"
        
        return False, None
```

---

## 11. LIVE TRADING CONDITIONS & SAFEGUARDS

### 11.1 Minimum Confidence Threshold

Models must exceed confidence threshold to generate orders:

```python
MINIMUM_CONFIDENCE_THRESHOLDS = {
    'trend': 0.55,           # Trend signals must be 55%+ confident
    'mean_reversion': 0.60,  # MR signals higher bar (harder to predict)
    'scalper': 0.52,         # Scalper: very tight, so 52% is meaningful
    'stat_arb': 0.65,        # StatArb: multi-asset complexity
    'discretionary': 0.60,   # Disc: pattern recognition uncertainty
    'market_maker': 0.45,    # MM: always quotes, minimum confidence OK
}

def filter_by_confidence(signal, archetype):
    """Only generate order if confidence exceeds threshold."""
    min_conf = MINIMUM_CONFIDENCE_THRESHOLDS[archetype]
    
    if signal['confidence'] < min_conf:
        return None  # Suppress signal
    
    return signal
```

### 11.2 Signal Expiration

Signals are valid for a limited time window, after which they are discarded:

```python
SIGNAL_EXPIRATION_WINDOWS = {
    'scalper': 5,           # Scalper signals valid for 5 seconds
    'trend': 300,           # Trend signals valid for 5 minutes
    'mean_reversion': 300,  # MR signals valid for 5 minutes
    'stat_arb': 600,        # StatArb signals valid for 10 minutes
    'discretionary': 600,   # Disc signals valid for 10 minutes
    'market_maker': 1,      # MM quotes update continuously
}

def check_signal_expiration(signal, archetype, current_time):
    """
    Discard signal if older than expiration window.
    """
    age_seconds = (current_time - signal['timestamp']).total_seconds()
    expiration_seconds = SIGNAL_EXPIRATION_WINDOWS[archetype]
    
    if age_seconds > expiration_seconds:
        return None  # Signal expired
    
    return signal
```

### 11.3 Model Consensus Requirements

For high-risk positions, multiple archetypes must agree:

```python
def require_multi_archetype_consensus(signals, min_archetypes=2):
    """
    Large position sizes require agreement from multiple archetypes.
    
    Example: A SELL signal is valid only if:
    - Trend model says SELL
    - AND Mean Reversion says SELL
    
    This prevents single-archetype false signals from causing large losses.
    """
    
    sell_signals = sum(1 for s in signals if s['direction'] < 0)
    buy_signals = sum(1 for s in signals if s['direction'] > 0)
    
    consensus_direction = 'SELL' if sell_signals >= min_archetypes else 'BUY' if buy_signals >= min_archetypes else None
    
    return consensus_direction, {
        'sell_count': sell_signals,
        'buy_count': buy_signals,
        'consensus_required': min_archetypes,
    }
```

---

## 12. MONITORING, DRIFT DETECTION & RETRAINING

### 12.1 Real-Time Performance Tracking

Every trade is logged with:

```python
class TradeLog:
    def record_trade(self, signal, order, fill, pnl):
        """
        Log every trade for monitoring and retraining.
        """
        record = {
            'timestamp': datetime.now(),
            'symbol': order['symbol'],
            'signal_archetype': signal['archetype'],
            'signal_direction': signal['direction'],
            'signal_confidence': signal['confidence'],
            'order_id': order['order_id'],
            'order_type': order['order_type'],
            'entry_price': fill['average_fill_price'],
            'exit_price': None,  # Set when position closes
            'pnl': None,
            'pnl_pct': None,
            'hold_seconds': None,
            'model_name': signal.get('model_name'),  # Which model generated signal
        }
        
        self.log.append(record)
        self.db.insert(record)
```

### 12.2 Feature Drift Detection

```python
class DriftDetector:
    def __init__(self, train_feature_stats):
        self.train_mean = train_feature_stats['mean']
        self.train_std = train_feature_stats['std']
        self.rolling_window = deque(maxlen=1000)  # Last 1000 inference batches
    
    def compute_drift(self, batch_features):
        """
        Compute KL divergence of live features vs training distribution.
        """
        live_mean = np.mean(batch_features, axis=0)
        live_std = np.std(batch_features, axis=0)
        
        # KL divergence for normal distributions
        kl = 0.5 * (
            (live_std ** 2 + (live_mean - self.train_mean) ** 2) / (self.train_std ** 2)
            - 1
            + 2 * np.log(self.train_std / (live_std + 1e-8))
        )
        
        return np.mean(kl)
    
    def check_drift(self, batch_features, threshold=0.10):
        """
        Check if current batch shows significant feature drift.
        Returns: (drift_detected, drift_score)
        """
        drift = self.compute_drift(batch_features)
        self.rolling_window.append(drift)
        
        rolling_drift = np.mean(self.rolling_window)
        
        if rolling_drift > threshold:
            return True, rolling_drift
        
        return False, rolling_drift
```

### 12.3 Automatic Retraining Trigger

Retraining is triggered when:

1. **Model Sharpe < 0.5 (50-trade window)**: Performance deteriorated
2. **Feature Drift > 0.10**: Market regime shifted
3. **Win Rate < 40% (100-trade window)**: Directional predictions failing
4. **Scheduled**: Every 30 days (regardless of performance)

```python
def check_retraining_requirements():
    """
    Check all conditions; return True if retraining needed.
    """
    
    # Condition 1: Sharpe deterioration
    recent_sharpe = compute_rolling_sharpe(window=50)
    if recent_sharpe < 0.5:
        return True, "Sharpe < 0.5"
    
    # Condition 2: Feature drift
    drift_detected, drift_score = drift_detector.check_drift(latest_batch)
    if drift_detected:
        return True, f"Feature drift: {drift_score:.3f}"
    
    # Condition 3: Win rate collapse
    recent_win_rate = compute_rolling_win_rate(window=100)
    if recent_win_rate < 0.40:
        return True, f"Win rate < 40%: {recent_win_rate:.1%}"
    
    # Condition 4: Scheduled retraining
    if (datetime.now() - last_retrain_date).days > 30:
        return True, "Scheduled (30 days)"
    
    return False, None
```

### 12.4 Retraining Procedure

```python
def trigger_retraining(archetype, reason):
    """
    Halt trading on this archetype, retrain, revalidate, redeploy.
    """
    
    print(f"[RETRAINING] {archetype}: {reason}")
    
    # Step 1: Pause live trading for this archetype
    disable_archetype_signals(archetype)
    
    # Step 2: Collect fresh training data
    recent_data = collect_recent_trades(lookback_days=30)
    
    # Step 3: Split data
    train, val, test = split_chronological(recent_data, [0.7, 0.15, 0.15])
    
    # Step 4: Retrain with hyperparameters from previous best
    best_hyperparams = load_best_hyperparams(archetype)
    result = train_model_multi_seed(config, seeds=[42, 123, 456])
    
    # Step 5: Evaluate on walk-forward
    wfv_results = walk_forward_validation(result['model'], [train, val, test])
    
    # Step 6: Gate check
    is_valid, metrics, failures = evaluate_model(result['model'], test, config, archetype)
    
    if is_valid:
        print(f"✓ Retraining successful: {archetype} passed all gates")
        # Deploy retrained model
        deploy_model(archetype, result['model'])
        enable_archetype_signals(archetype)
    else:
        print(f"✗ Retraining failed: {archetype}")
        # Keep old model, alert operations
        alert(f"Retraining failed for {archetype}: {failures}", severity='HIGH')
        # Fall back to manual trading
```

---

## 13. FAILURE MODE ANALYSIS PER ARCHETYPE

### 13.1 Trend Follower

**Known Blind Spots**:
- **Range-bound markets**: Cannot profit in sideways consolidation
- **Whipsaws**: False breakouts cause stop-losses
- **Gap risk**: Overnight gaps skip stop-loss levels

**Mitigation**:
- Filter signals in low-volatility periods (ATR < 20th percentile)
- Require multi-day trend confirmation before opening position
- Reduce position size during high-news-flow periods

**When to Disable**:
- Rolling win rate < 45% for 200+ trades
- Sharpe < 0.5 for 50+ trades
- In predicted "mean-reversion" regime (opposite of trend)

### 13.2 Mean Reversion

**Known Blind Spots**:
- **Breakout markets**: Reversals fail when trend continues
- **Gaps through bands**: Bollinger bands invalidated on large moves
- **Low liquidity**: Slippage eats profits on small positions

**Mitigation**:
- Only trade reversals if trend is weak (price slope < median)
- Scale position size inverse to distance from bands
- Require volatility > 15th percentile (avoid thin trading)

**When to Disable**:
- Trend strength > 0.7 (detected by trend model)
- Sharpe < 0.5 for 50+ trades
- Profit factor < 1.2 for 100+ trades

### 13.3 Scalper

**Known Blind Spots**:
- **Latency disadvantage**: Against colocated market makers
- **Tick size limits**: Profit per trade often < 1 basis point
- **Fill slippage**: Bid-ask spread erodes most gains

**Mitigation**:
- Only scalp liquid instruments (BTC, ETH, top altcoins)
- Require probability of profitability > 52% (not 50%)
- Halt during wide-spread periods (spread > 2× median)

**When to Disable**:
- Win rate < 50% for 500+ trades
- Average slippage > 1 basis point per trade
- Spread > 10 basis points (illiquid market)

### 13.4 Statistical Arbitrage

**Known Blind Spots**:
- **Correlation breakdown**: Pairs diverge in crisis
- **Funding rate blows**: Perpetual futures funding makes hedges expensive
- **Limited pair universe**: Only 34 symbols, limited diversification

**Mitigation**:
- Monitor spread of best pair; alert if >2× recent median
- Check funding rate before opening; skip if >0.1% per 8h
- Require correlation stability (ρ changing <5% per hour)

**When to Disable**:
- MAE > 0.1 (mean absolute error on spread prediction)
- Sharpe < 1.0 for 50+ trades (lower bar due to complexity)
- Correlation metric breaks (ρ < 0.5)

### 13.5 Discretionary

**Known Blind Spots**:
- **Insufficient training data**: Only 12K samples (vs 2M for others)
- **Pattern specificity**: Chart patterns don't repeat exactly
- **No order-flow context**: Candlesticks alone insufficient

**Mitigation**:
- Require very high confidence (>70%) before trading
- Only trade high-probability patterns (support/resistance breaks)
- Pair with other archetypes (consensus voting)

**When to Disable**:
- Accuracy < 50% for 100+ trades (worse than coin-flip)
- Sharpe < 0.0 for 50+ trades (consistent losses)
- Always disabled until retraining with more data

### 13.6 Market Maker

**Known Blind Spots**:
- **Flash crashes**: Large moves can bankrupt MM inventory
- **Asymmetric fills**: Usually buy at ask, sell at bid (short the spread)
- **Model risk**: RL policy can diverge from training environment

**Mitigation**:
- Maximum position size: 100 units per symbol
- Stop-loss if inventory > 200 units or <-200 units
- Revert to symmetric spreads on wide vol spikes

**When to Disable**:
- Mean episode reward < 0.5 for 10+ episodes (unprofitable)
- Max drawdown per episode > 10% (too risky)
- PnL correlation with BTC price > 0.8 (model misbehaving)

---

## 14. API & INTERFACE STANDARDIZATION

### 14.1 Model Inference Interface

All models expose a standard inference API:

```python
class ModelInterface:
    def __init__(self, weights_path: str, config: dict, backend: str = 'directml'):
        self.model = torch.load(weights_path)
        self.config = config
        self.device = resolve_device(backend)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, features: np.ndarray) -> dict:
        """
        Unified prediction interface.
        
        Input:
            features: (batch_size, feature_dim) or (feature_dim,)
        
        Returns:
            {
                'prediction': float or (batch_size,),  # Raw model output
                'confidence': float or (batch_size,),  # 0-1 confidence
                'latency_ms': float,  # Inference time in milliseconds
                'version': str,  # Model version
            }
        """
        
        # Ensure input is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Convert to tensor
        x = torch.from_numpy(features).float().to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            output = self.model(x)
        latency_ms = (time.time() - start_time) * 1000
        
        # Process output
        if isinstance(output, tuple):  # (logits, confidence)
            logits, confidence = output
            prediction = logits.argmax(dim=1) if logits.shape[1] > 1 else logits.squeeze()
            confidence = confidence.cpu().numpy()
        else:
            prediction = output
            confidence = np.ones(len(prediction)) * 0.5  # Default: medium confidence
        
        return {
            'prediction': prediction.cpu().numpy(),
            'confidence': confidence,
            'latency_ms': latency_ms,
            'version': self.config.get('model_version', 'unknown'),
        }
    
    def get_config(self) -> dict:
        """Return model hyperparameters and design info."""
        return self.config
```

### 14.2 Signal Output Schema

All archetype signals conform to this schema:

```python
@dataclass
class Signal:
    timestamp: datetime
    archetype: str  # 'trend', 'mean_reversion', 'scalper', 'stat_arb', 'discretionary', 'market_maker'
    symbol: str  # BTCUSDT, ETHUSDT, etc.
    
    # Prediction
    direction: float  # -1 (SELL), 0 (NEUTRAL), +1 (BUY); or continuous [-1, 1]
    confidence: float  # 0–1 confidence in prediction
    
    # Model attribution
    model_name: str  # e.g., 'LSTM_Trend_v1'
    model_version: str  # e.g., '2.0'
    
    # Ensemble info
    ensemble_size: int  # Number of models aggregated
    ensemble_entropy: float  # 0=unanimous, 1=maximum disagreement
    
    # Metadata
    features_version: str  # e.g., '2.1'
    backend: str  # 'directml', 'cuda', 'cpu'
    inference_latency_ms: float  # Latency of model inference
    
    # Optional fields
    stop_loss: float = None  # Suggested stop-loss level
    take_profit: float = None  # Suggested take-profit level
    position_size_suggestion: float = None  # Units to trade
    
    def to_json(self) -> str:
        """Serialize to JSON for logging/transmission."""
        return json.dumps({
            'timestamp': self.timestamp.isoformat(),
            'archetype': self.archetype,
            'symbol': self.symbol,
            'direction': float(self.direction),
            'confidence': float(self.confidence),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'ensemble_size': self.ensemble_size,
            'ensemble_entropy': float(self.ensemble_entropy),
            'features_version': self.features_version,
            'backend': self.backend,
            'inference_latency_ms': float(self.inference_latency_ms),
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
            'position_size_suggestion': float(self.position_size_suggestion) if self.position_size_suggestion else None,
        })
```

### 14.3 Execution Order Interface

Orders generated from signals conform to:

```python
@dataclass
class ExecutionOrder:
    order_id: str  # Unique ID
    symbol: str
    side: Literal['BUY', 'SELL']
    quantity: float
    order_type: Literal['MARKET', 'LIMIT', 'TWAP', 'VWAP', 'IOC']
    limit_price: float = None  # For LIMIT orders
    time_in_force: Literal['GTC', 'IOC', 'FOK'] = 'GTC'
    
    # Traceability
    signal_id: str  # Links back to Signal
    archetype: str
    
    # Risk management
    stop_loss: float = None
    take_profit: float = None
    max_position_hours: int = 24  # Auto-close after X hours
    
    # Execution constraints
    max_slippage_bps: int = 10  # Reject if slippage > 10 bps
    min_fill_percentage: float = 0.8  # Need at least 80% fill
    
    def to_order_message(self) -> dict:
        """Prepare for exchange API submission."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'type': self.order_type,
            'quantity': self.quantity,
            'price': self.limit_price if self.order_type == 'LIMIT' else None,
            'timeInForce': self.time_in_force,
            'clientOrderId': self.order_id,
            'stopPrice': self.stop_loss,
            'takeProfit': self.take_profit,
        }
```

---

## 15. ARCHETYPE-SPECIFIC PROCEDURES

### 15.1 Trend Follower Deployment

```
1. Pre-deployment validation:
   - Walk-forward Sharpe variance < 30%
   - Accuracy > 55% on all 7 walks
   - Max drawdown < 20%

2. Live deployment parameters:
   - Position size: Kelly Criterion, capped at 5% of account
   - Stop-loss: 2× ATR below entry
   - Take-profit: 3× ATR above entry
   - Time limit: Auto-close after 4 hours

3. Monitoring:
   - Track rolling win rate (100 trades)
   - Check Sharpe (50-trade window)
   - Alert if accuracy < 52% (degradation)

4. Disable conditions:
   - Sharpe < 0.5 (50 trades)
   - Win rate < 45% (200 trades)
   - Market detected as "mean-reversion"
```

### 15.2 StatArb Deployment

```
1. Pre-deployment validation:
   - MAE < 0.05 on walk-forward
   - Profit factor > 1.8
   - Max drawdown < 15% (stricter than others)

2. Live deployment parameters:
   - Pair selection: Only correlations >0.8
   - Spread limit: Trade only if spread < 2× median
   - Position size: 50 units long + 50 units short per pair
   - Funding rate check: Skip if > 0.1% per 8h

3. Monitoring:
   - Track spread divergence
   - Monitor funding rates
   - Alert if correlation < 0.7

4. Disable conditions:
   - Correlation breakdown (ρ < 0.5)
   - MAE > 0.15 (spread prediction failing)
   - Spread > 5× median (illiquid)
```

### 15.3 Market Maker Deployment

```
1. Pre-deployment validation:
   - All 3 RL agents (PPO, SAC, DQN) passing
   - Mean episode reward > 0.5
   - Max drawdown < 10% per episode

2. Live deployment parameters:
   - Quote width: 20 bps from mid
   - Max inventory: ±100 units per symbol
   - Revert to symmetric quotes on volatility spikes
   - Refresh quotes every 1 second

3. Monitoring:
   - Track inventory evolution
   - Monitor PnL per minute
   - Check spread win rate (% of quotes filled on winning side)

4. Disable conditions:
   - Mean reward < 0.2 (10 episodes)
   - Max drawdown > 15% (too risky)
   - Inventory hit hard stops (±200 units)
```

---

## 16. OPERATIONAL PROCEDURES & CHECKLISTS

### 16.1 Pre-Deployment Checklist

```
[ ] All 18 models trained with V2 hyperparameters
[ ] All models pass walk-forward validation (7 walks)
[ ] All models pass all 5 primary gates + 5 extended gates
[ ] Ensemble weights initialized from validation Sharpe
[ ] Risk engine configured with appropriate limits
[ ] Feature drift detector trained on training dataset statistics
[ ] Exchange API connectivity verified
[ ] Order submission and fill tracking tested in paper trading
[ ] Emergency stop mechanism tested
[ ] Monitoring and alerting configured
[ ] Operational runbooks reviewed by team
[ ] Compliance sign-off obtained
```

### 16.2 Daily Operations

```
**Morning (9:00 AM)**:
- [ ] Check overnight drift metrics
- [ ] Review any alerts/triggered kill-switches
- [ ] Verify all models loaded and ready
- [ ] Run feature validation (no NaN/Inf)

**During Trading (9:30 AM - 4:00 PM)**:
- [ ] Monitor position sizes every 5 minutes
- [ ] Log all signals and fills
- [ ] Check rolling Sharpe/win-rate every 100 trades
- [ ] Alert if Sharpe < 0.5 or win-rate < 45%

**End-of-Day (4:00 PM - 5:00 PM)**:
- [ ] Close all positions at market if needed
- [ ] Reconcile PnL with exchange
- [ ] Archive today's logs
- [ ] Update model registry with daily results
- [ ] Check if retraining triggered

**Weekly (Friday Close)**:
- [ ] Compile performance report (Sharpe, Profit Factor, Drawdown)
- [ ] Review failure modes/anomalies
- [ ] Update hyperparameter search if needed
- [ ] Backup all model checkpoints
```

### 16.3 Retraining Trigger Checklist

When retraining is initiated:

```
[ ] Pause live signals for this archetype
[ ] Collect recent 30-day trade data
[ ] Run hyperparameter search (if drift detected)
[ ] Retrain with best hyperparams + multi-seed
[ ] Run walk-forward validation (7 walks)
[ ] Verify all gates passed
[ ] Update ensemble weights
[ ] Run live validation (paper trading) for 1 day
[ ] Deploy if all checks pass
[ ] Resume live signals
[ ] Monitor closely for 100 trades
```

---

## CONCLUSION

ChatTrader.KPai V2 is a production-ready, institutional-grade system for multi-archetype quantitative trading. Key differentiators:

1. **Robustness**: Walk-forward validation, regime-stratified testing, drift detection
2. **Safety**: Comprehensive risk engine, kill-switches, position limits
3. **Transparency**: Detailed failure mode analysis, explainable ensemble weighting
4. **Autonomy**: Automated retraining, monitoring, and performance tracking
5. **Auditability**: Full logging, signal-to-execution traceability, compliance hooks

The system is designed to operate autonomously while remaining under human control. All critical decisions (retraining, emergency stop, archetype disable) are logged and auditable.

**Deployment readiness**: Market Maker RL models are immediately deployable (3/3 passed gates). Remaining archetypes require extended hyperparameter search and walk-forward validation (48–72 hours estimated).

---

**Report Prepared**: April 30, 2026  
**Status**: PRODUCTION SPECIFICATION FINALIZED  
**Next Action**: Initiate extended training pipeline with V2 specifications


---

## APV-PLN — Archetype VII: Probabilistic Trend & Regime Learner

> **Added to V2 Handbook:** May 19, 2026

### Model Specification

| Field | Value |
|---|---|
| **Model Name** | Adaptive Price-Volume Probabilistic Learner Network (APV-PLN) |
| **Short Name** | APV-PLN |
| **Type / Archetype** | Probabilistic Trend & Regime Learner |
| **Training Type** | Privileged Information Distillation (Oracle-Teacher / LUPI) |
| **Module** | \quant_core.apv_pln_models.APVPLNModel\ |
| **Launcher** | \python -m quant_core.train_apv_pln_phase4\ |
| **Config** | \configs/apv_pln_phase4.yaml\ |
| **Checkpoint Root** | \models/checkpoints/apv_pln/APV_PLN_<variant>/\ |
| **Registry Key** | \pv_pln\ |

---

### Input Specifications

**Price Stream** \[B, seq_len=32, 5]
| Feature | Description |
|---|---|
| \log_return\ | log(close_t / close_{t-1}) |
| \zscore_close_64\ | Rolling z-score of close, window 64 |
| \ema_spread\ | EMA(12) - EMA(26) |
| \tr_14\ | Average True Range, window 14 |
| \price_slope_20\ | (close - close.shift(20)) / 20 |

**Volume Stream** \[B, seq_len=32, 5]
| Feature | Description |
|---|---|
| \log_volume\ | log1p(volume) |
| \olume_zscore_64\ | Rolling z-score of log_volume, window 64 |
| \	aker_buy_ratio\ | taker_buy_base / volume (buying pressure) |
| \wap_deviation\ | (close - VWAP) / (ATR + eps) |
| \ol_imbalance\ | (vol_up - vol_dn) / (vol_up + vol_dn + eps) |

**Oracle Stream** \[B, horizon=5, 2]\ — **Training Only (LUPI)**

| Feature | Description |
|---|---|
| \log_return\ | Future bar log-return (privileged) |
| \log_volume\ | Future bar log-volume (privileged) |

All features strictly causal. Scaler fit on training split only.

---

### Output Specifications

**Discrete Probability Distribution** \[B, num_bins=51]
- **51 bins** covering the 0.5th-99.5th percentile of training forward returns
- Typical bounds: bin_min approx -0.030, bin_max approx +0.032 (5-bar horizon, 5-min Binance OHLCV)
- Bin width: approx 0.0012 per bin
- Bin centres saved in \in_meta.pt\ alongside model weights
- **Expected return** = (softmax(logits) x bin_centres).sum()
- **Direction signal** = sign(expected_return)
- **Confidence** = max(softmax(logits))

---

### Architecture

Price CNN (2x Conv1D+LN+LReLU) + Volume CNN (2x Conv1D+LN+LReLU) with cross-attention between streams, adaptive gate, and 51-bin head.
Oracle Teacher: ManualLSTM over 5 future bars -> 51-bin softmax (train only).
Loss: L = alpha*CE(student_logits, y_bin) + beta*T^2*KL(student/T || oracle/T); alpha=beta=0.5, T=2.0

---

### Oracle Isolation Contract

| Phase | Oracle Teacher | Loss | Batch Shape |
|---|---|---|---|
| train | CALLED | alpha*CE + beta*KL | (x_price, x_volume, y_bin, x_oracle) |
| val | NEVER CALLED | CE only | (x_price, x_volume, y_bin) |
| test | NEVER CALLED | CE only | (x_price, x_volume, y_bin) |

---

### Validation Gates

| Metric | Gate |
|---|---|
| Test Sharpe Ratio | >= 1.2 |
| Test Directional Accuracy | >= 55% |
| Test Profit Factor | >= 1.5 |
| Test Max Drawdown | <= 20% |
| Val->Test Sharpe Decay | < 50% |
| Val CE Loss | < 3.9 (log(51) = pure noise) |

---

### Model Variants

| Variant | cnn_channels | nhead | dropout | Params |
|---|---|---|---|---|
| APV_PLN_v1 | 64 | 4 | 0.15 | ~200K |
| APV_PLN_v2 | 128 | 4 | 0.20 | ~750K |
| APV_PLN_v3 | 64 | 8 | 0.25 | ~210K |

---

### Running

\\ash
# Smoke test
python -m quant_core.train_apv_pln_phase4 --config configs/apv_pln_phase4_smoke.yaml

# Full training
python -m quant_core.train_apv_pln_phase4 --config configs/apv_pln_phase4.yaml
\
---

**Section Added**: May 19, 2026 | **Status**: CODE VERIFIED, SMOKE TEST PASSED, ORACLE ISOLATION CONFIRMED
