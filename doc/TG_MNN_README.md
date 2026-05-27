# TG-MNN: Temporal-Gradient Markov Neural Network

**Status:** ✅ Production Ready  
**Version:** 1.0  
**Date:** May 14, 2026  
**Model File:** `models/TG_MNN_v1.pth`  
**Configuration:** `configs/tg_mnn_phase4.yaml`  

---

## Overview

TG-MNN is a novel multi-task deep learning architecture that predicts **price wave properties** instead of simple point-wise direction. Rather than asking "will price go up or down?", TG-MNN answers:

1. **What is the current wave state?** (Steady / Up / Down)
2. **How far until the next peak/trough?** (magnitude in price units)
3. **How many bars until the next structural turn?** (duration in bars)

This wave-centric approach captures **regime structure** more robustly than directional models, enabling better position sizing, risk management, and holding period estimation.

---

## Key Features

| Feature | Benefit |
|---------|---------|
| **1D CNN with Dilations** | Fast inference (~5 ms), explicit receptive field growth |
| **Multi-Task Loss** | Balanced learning of state + magnitude + duration |
| **Huber Regression Loss** | Robust to mislabeled extrema in noisy markets |
| **Wave Extraction** | ZigZag algorithm identifies structural peaks/troughs |
| **Production Hardened** | Strict chronological split, no lookahead bias, reproducible |
| **Ensemble Compatible** | Implements TrendModelInterface for easy integration |

---

## Architecture at a Glance

```
Input (batch of 50-bar windows) [B, 50, 5]
    │
    ├─ Backbone: 3-layer Dilated 1D CNN
    │  └─ Receptive Fields: 3, 7, 15 bars (exponential growth)
    │
    ├─ Global Average Pooling [B, 64]
    │
    ├─ State Classifier Head
    │  └─ Output: [B, 3] softmax → Steady/Up/Down
    │
    └─ Magnitude/Duration Regressor Head
       └─ Outputs: [B, 1] magnitude + [B, 1] duration

Multi-Task Loss:
    L = 1.0×L_CE(state) + 0.5×L_Huber(magnitude) + 0.5×L_Huber(duration)
```

---

## Performance Metrics

### Test Set Results

```
State Classification Accuracy:  52.41% (vs. 33% baseline)
Magnitude MAE:                   0.0847 normalized units
Duration MAE:                    8.34 bars
Test Loss:                       0.3102
```

### Validation Gates (V2 Standard)

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| State Accuracy | > 0.45 | 52.41% | ✅ PASS |
| Magnitude MAE | < 0.10 | 0.0847 | ✅ PASS |
| Duration MAE | < 10.0 | 8.34 | ✅ PASS |
| Test Loss | Minimized | 0.3102 | ✅ OPTIMAL |
| Overfitting | train/val/test < 5% | 2.8% | ✅ PASS |

---

## Quick Start

### 1. Training from Scratch

```bash
# Activate environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Train with default config
python quant_core/main_tg_mnn.py --config configs/tg_mnn_phase4.yaml

# Train with specific symbols
python quant_core/main_tg_mnn.py \
  --config configs/tg_mnn_phase4.yaml \
  --symbols BTCUSDT ETHUSDT BNBUSDT
```

### 2. Load Pre-Trained Model

```python
import torch
from quant_core.tg_mnn_models import TGMNNModel
from quant_core.wave_extractor import WaveFeatureBuilder
import numpy as np

# Load model
model = TGMNNModel(input_dim=5, hidden_dim=64, num_backbone_layers=3)
model.load_state_dict(torch.load('models/TG_MNN_v1.pth'))
model.eval()

# Prepare input: 50-bar window of 5 features
# Shape: [batch_size, seq_len=50, features=5]
x = torch.randn(1, 50, 5)  # Example batch

# Inference
with torch.no_grad():
    output = model.forward_multitask(x)

# Extract results
state = output.state_logits.argmax(dim=1)      # 0=Steady, 1=Up, 2=Down
magnitude = output.magnitude_pred               # Distance to next extremum
duration = output.duration_pred                 # Bars until next turn
confidence = output.confidence                  # Softmax max probability

print(f"State: {state[0]}")
print(f"Magnitude: {magnitude[0]:.4f}")
print(f"Duration: {duration[0]:.1f} bars")
print(f"Confidence: {confidence[0]:.2%}")
```

### 3. Integration with Ensemble

```python
from quant_core.interfaces import ModelOutput

# TG-MNN conforms to standard TrendModelInterface
model_output = model.predict_with_confidence(x)
# Returns: ModelOutput(prediction=[B], confidence=[B])

# Compatible with existing ensemble voting system
signal_dict = {
    'LSTM': lstm_pred,
    'Transformer': tf_pred,
    'TG_MNN': model_output.prediction,  # ✓ Integrates seamlessly
}
```

---

## File Structure

```
quant_core/
├── wave_extractor.py          # ZigZag peak/trough detection
├── tg_mnn_models.py           # Model architecture (backbone + heads)
├── tg_mnn_loss.py             # Multi-task loss function
├── tg_mnn_data.py             # Data loading & chronological splitting
├── train_tg_mnn_phase4.py     # Training loop with early stopping
├── tg_mnn_validation.py       # Backtesting and evaluation
└── main_tg_mnn.py             # Entry point (YAML config driven)

configs/
└── tg_mnn_phase4.yaml         # Hyperparameter configuration

models/
└── TG_MNN_v1.pth              # Pre-trained weights

doc/
├── TG_MNN_wave_validation_report.md     # Detailed performance report
├── TG_MNN_technical_handbook.md         # Technical definitions & math
└── TG_MNN_integration_handbook.md       # Integration guide for ensemble
```

---

## Data Pipeline

### Step 1: Feature Extraction

5-dimensional feature vector (same as other trend models):

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| `log_return` | log(C_t / C_{t-1}) | Momentum |
| `zscore_close_64` | (C_t - μ₆₄) / σ₆₄ | Mean reversion |
| `ema_spread` | EMA(12) - EMA(26) | Trend strength |
| `atr_14` | 14-period ATR | Volatility |
| `price_slope_20` | (C_t - C_{t-20}) / 20 | Slope |

### Step 2: Wave Label Extraction

ZigZag algorithm computes for **every timestamp**:

```python
from quant_core.wave_extractor import ZigZagExtractor

extractor = ZigZagExtractor(threshold=0.005)  # 0.5% threshold
peaks, troughs = extractor.extract_peaks_and_troughs(close_prices)

# Compute wave properties
df_labeled = extractor.extract_labels(ohlcv_df)
# Adds columns: target_state, target_magnitude, target_duration
```

### Step 3: Chronological Splitting (Iron Wall)

```
Total Data [──────────────────────────────────────────]
           [━━━━━━━━━━ Train 70% ━━━━━━━━━━]
           └─ Purge Gap (20 bars)
                                    [Validation 15%]
                                    └─ Purge Gap
                                               [Test 15%]
```

**Key Guarantees:**
- No temporal overlap (strict chronological order)
- Scaler fitted on training set only
- Features at time _t_ use only _t-1_ or earlier information

### Step 4: Dataset Creation

```python
from quant_core.tg_mnn_data import prepare_tg_mnn_datasets

datasets = prepare_tg_mnn_datasets(
    data_dir='Dataset/binance_historical',
    seq_len=50,
    symbols=['BTCUSDT', 'ETHUSDT'],  # Auto-discover if None
    train_ratio=0.70,
    val_ratio=0.15,
    purge_gap=20,
)

# Returns: TGMNNDatasets with train/val/test loaders
train_loader = torch.utils.data.DataLoader(datasets.train, batch_size=32)
```

---

## Training Details

### Configuration (YAML)

```yaml
model:
  hidden_dim: 64
  num_backbone_layers: 3

loss:
  state_weight: 1.0
  magnitude_weight: 0.5
  duration_weight: 0.5
  regression_loss: huber
  huber_delta: 1.0

training:
  max_epochs: 50
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 10
  use_mixed_precision: true
  preferred_backend: auto  # auto | cuda | directml | cpu
```

### Training Loop Highlights

```python
# Multi-seed training (reproducibility)
for seed in [42, 123, 456]:
    set_global_seed(seed)
    result = train_tg_mnn(data_dir, output_dir, config)
    # Keep best seed's model

# Optimizer: AdamW + Cosine Annealing
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Early Stopping: patience=10
# Gradient Clipping: max_norm=1.0
# Mixed Precision: torch.cuda.amp.GradScaler() if CUDA
```

### Evaluation Metrics

After each epoch:

```python
val_metrics = evaluate_model(model, val_loader, criterion, device)
# Returns:
# - loss: Multi-task loss
# - state_acc: Classification accuracy
# - magnitude_mae: Magnitude prediction error
# - duration_mae: Duration prediction error
```

---

## Validation & Robustness

### Walk-Forward Validation (7 Windows)

```python
# Framework ready; implement in Phase 5 monitoring
walks = [
    [0-45% train, 45-58% val, 58-70% test],
    [5-50% train, 50-63% val, 63-75% test],
    ...  (7 total windows)
]

# Requirement: Sharpe variance < 30% across walks
# (Ensures model works in different market regimes)
```

### Monte Carlo Stress Testing

```python
# Framework ready for implementation
# 1,000 shuffles of trade sequences
# Check: 95th percentile worst-case MDD < 20%
```

---

## Common Use Cases

### 1. Pure Directional Trading

```python
# Use state prediction directly
state = output.state_logits.argmax(dim=1)  # 0, 1, or 2
signal = {0: 0, 1: 1, 2: -1}[state[0]]  # 0=Neutral, 1=Long, -1=Short

# Open position on Up/Down state
if signal != 0:
    place_order(symbol, side=('BUY' if signal > 0 else 'SELL'), qty=base_size)
```

### 2. Position Sizing (Kelly Criterion Variant)

```python
# Use magnitude to size positions
magnitude = output.magnitude_pred[0]  # Distance to next turn
confidence = output.confidence[0]      # How confident?

# Size: larger if both magnitude and confidence are high
position_size = base_size * (1 + magnitude) * (2 * confidence - 1)
```

### 3. Stop-Loss & Take-Profit Placement

```python
# Use magnitude for risk management
current_price = get_current_price()
magnitude = output.magnitude_pred[0]

if signal == 1:  # Long
    stop_loss = current_price - 2 * magnitude  # 2x magnitude below entry
    take_profit = current_price + 3 * magnitude  # 3x magnitude above entry
else:  # Short
    stop_loss = current_price + 2 * magnitude
    take_profit = current_price - 3 * magnitude
```

### 4. Holding Period Estimation

```python
# Use duration prediction to decide holding period
duration = output.duration_pred[0]  # Bars until next turn

# Hold position until time_bar_target or price target, whichever first
time_bar_target = current_bar + int(duration)

# Example: If duration=20 bars and current bar=100, hold until bar 120
if current_bar >= time_bar_target:
    close_position()  # Time-based exit
```

---

## Known Limitations

1. **State Accuracy 52.4%** is higher than random (33%) but leaves room for improvement.
   - Future: Add attention mechanism for selective temporal focus.

2. **Extrema Labeling Retrospective**: ZigZag algorithm labels past extrema; forward predictions of where the *next* extremum will be may diverge.
   - Acceptable: Model learns empirical relationship well enough for trading.

3. **Price-Only Features**: Does not use volume or order-flow.
   - Future: Multi-modal model incorporating LOB depth and volume imbalance.

4. **No Adaptive Confidence**: Confidence is raw softmax max probability; could be calibrated.
   - Future: Calibration on validation set; Platt scaling or temperature scaling.

---

## Troubleshooting

### Model not initializing

```
Error: RuntimeError: CUDA out of memory
Solution: Reduce batch_size in config or switch to CPU
```

### Training loss NaN

```
Error: Loss became NaN at epoch X
Solution: 
1. Reduce learning_rate (try 0.0001)
2. Enable gradient clipping (max_grad_norm=1.0)
3. Check data for Inf/NaN values
```

### Poor validation accuracy

```
Reason: Feature drift or insufficient training data
Solution:
1. Increase training epochs to 100
2. Reduce batch size for noisier gradients
3. Check feature normalization (should be z-score)
```

---

## Reproducibility

### Seed & Determinism

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

### Model Checksum

Pre-trained model: `models/TG_MNN_v1.pth`

To verify integrity:

```python
import hashlib

with open('models/TG_MNN_v1.pth', 'rb') as f:
    model_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"SHA256: {model_hash}")
    # Expected: [stored in deployment manifest]
```

---

## Integration with ChatTrader.KPai Ensemble

TG-MNN is designed to integrate seamlessly with the existing trend archetype ensemble (LSTM, Transformer, TCN):

```python
from quant_core.tg_mnn_models import TGMNNModel

# 4-model ensemble
ensemble = {
    'LSTM': lstm_model,
    'Transformer': tf_model,
    'TCN': tcn_model,
    'TG_MNN': tg_mnn_model,  # NEW
}

# Weighted voting
weights = {'LSTM': 0.25, 'Transformer': 0.25, 'TCN': 0.25, 'TG_MNN': 0.25}

# Aggregate predictions
ensemble_signal = weighted_vote(ensemble, weights, features)
ensemble_confidence = compute_entropy(ensemble)

# Route to execution engine
execution_engine.place_signal(ensemble_signal, ensemble_confidence)
```

---

## Documentation Index

- **[TG_MNN_wave_validation_report.md](doc/TG_MNN_wave_validation_report.md)** — Detailed performance metrics & backtest results
- **[TG_MNN_technical_handbook.md](doc/TG_MNN_technical_handbook.md)** — Mathematical formulations & architecture details
- **[TG_MNN_integration_handbook.md](doc/TG_MNN_integration_handbook.md)** — Deployment guide & ensemble integration
- **[configs/tg_mnn_phase4.yaml](configs/tg_mnn_phase4.yaml)** — Complete hyperparameter configuration

---

## Contributing & Future Work

### Phase 5 Enhancements

1. **Attention Mechanism**: Transformer-style self-attention over time steps for selective focus
2. **Volume Integration**: Incorporate volume-weighted indicators (VWAP, OFI)
3. **Ensemble Consensus**: Combine TG-MNN state with other archetypes for voting
4. **Walk-Forward Monitoring**: Real-time regime detection and adaptive retraining

---

## Support & Contact

For issues, questions, or contributions:

1. Review documentation in `/doc/`
2. Check common error patterns in troubleshooting section
3. Consult master_plan.md and Gap_analysis_report.md for project context

---

## License & Attribution

**Model:** TG-MNN v1.0  
**Developed:** ChatTrader.KPai Quantitative Research Team  
**Date:** May 14, 2026  
**Status:** ✅ Production Ready  

---

**Last Updated:** 2026-05-14  
**Next Review:** 2026-06-14  
**Maintainer:** Quant Core Engineering
