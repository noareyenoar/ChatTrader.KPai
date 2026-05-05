# ChatTrader.KPai — Models Handbook & Technical Manual

**Version:** Phase 4 | **Date:** 2026-04-28  
**Scope:** All 18 neural network models across 6 trading archetypes  
**Audience:** Developers, quant researchers, and AI agents operating this system

---

## Table of Contents

1. [System Philosophy](#1-system-philosophy)
2. [Data Infrastructure & Feature Engineering](#2-data-infrastructure--feature-engineering)
3. [The Iron Wall: Data Integrity & Temporal Split Protocol](#3-the-iron-wall-data-integrity--temporal-split-protocol)
4. [Hardware & Execution Environment](#4-hardware--execution-environment)
5. [Universal Evaluation Metrics](#5-universal-evaluation-metrics)
6. [How to Run the Models](#6-how-to-run-the-models)
7. [Archetype I — Trend Following (3 Models)](#7-archetype-i--trend-following-3-models)
8. [Archetype II — Mean Reversion (3 Models)](#8-archetype-ii--mean-reversion-3-models)
9. [Archetype III — Scalping / Microstructure (3 Models)](#9-archetype-iii--scalping--microstructure-3-models)
10. [Archetype IV — Statistical Arbitrage (3 Models)](#10-archetype-iv--statistical-arbitrage-3-models)
11. [Archetype V — Discretionary / Multimodal (3 Models)](#11-archetype-v--discretionary--multimodal-3-models)
12. [Archetype VI — Market Making / RL (3 Models)](#12-archetype-vi--market-making--rl-3-models)
13. [Strategic Model Selection Guide](#13-strategic-model-selection-guide)
14. [Model Registry Reference](#14-model-registry-reference)
15. [Global Glossary of Technical Terms](#15-global-glossary-of-technical-terms)

---

## 1. System Philosophy

ChatTrader.KPai is built around a core conviction: **no single model captures all market regimes**. Financial markets cycle through distinct behavioral states — trending momentum, mean-reverting consolidation, microstructure-driven noise, correlated pair dislocations, and inventory-driven quote dynamics. Each archetype in this system addresses one of those states with a purpose-built neural architecture.

The system is designed for a **tiered, sequential decision-making workflow**:

```
Step 1: Identify market regime   → Trend Following models (are we trending at all?)
Step 2: Measure overextension    → Mean Reversion models (is the trend stretched?)
Step 3: Read order-flow          → Scalper models (who is aggressing at this moment?)
Step 4: Check cross-asset spread → Stat Arb models (are correlated assets in sync?)
Step 5: Pattern-match chart      → Discretionary models (what does this look like historically?)
Step 6: Manage quotes/inventory  → Market Maker models (how do I capture spread safely?)
```

You do not have to use all six archetypes simultaneously. A swing trader may only use Steps 1–2. A prop desk running multiple instruments may combine Steps 1, 3, and 4. A market-maker runs Step 6 continuously, consulting Steps 1 and 4 for regime context.

### The 18-Model Rationale

Within each archetype, **three competing neural architectures** are trained on identical data. The three architectures are chosen to have fundamentally different inductive biases* — that is, different assumptions about the structure of the data. Running three architectures prevents any single model's blind spots from dominating. After training, the registry records which models passed validation, and you can run ensemble* inference by averaging their predictions.

---

## 2. Data Infrastructure & Feature Engineering

### 2.1 Source Data

All models are trained on **Binance historical OHLCV*** (Open, High, Low, Close, Volume) bar data stored as Parquet* files under `Dataset/binance_historical/`. Each symbol file contains the following columns:

| Column | Description |
|---|---|
| `timestamp` | UTC bar open time |
| `open` | Opening price |
| `high` | Highest price in bar |
| `low` | Lowest price in bar |
| `close` | Closing price |
| `volume` | Base asset volume |
| `quote_volume` | Quote asset volume (used for VWAP*) |
| `taker_buy_base` | Volume initiated by takers buying (scalper features) |
| `taker_buy_quote` | Quote volume initiated by taker buys |

The `DataQualityGate` (`data_pipeline/quality_gate.py`) filters symbols with fewer than 50,000 bars (`min_history_bars`) to ensure statistical stability. Symbols that do not pass are silently excluded from training.

### 2.2 Feature Factory (`data_pipeline/features.py`)

The `FeatureFactory` class is the single source of truth for all derived features. All transformations are **strictly causal** — a feature computed at time `t` uses only information from `t−1` or earlier.

#### Trend Features (5 columns)

```
FEATURE_COLUMNS = [
    "log_return",         # ln(P_t / P_{t-1}) — percentage change in log space
    "zscore_close_64",    # 64-bar rolling Z-score of close price
    "ema_spread",         # EMA(12) − EMA(26) — momentum crossover
    "atr_14",             # 14-bar ATR (exponentially-weighted true range)
    "price_slope_20",     # (close_t − close_{t-20}) / 20 — linear trend rate
]
```

- **Log Return**: Computed as `ln(close_t / close_{t-1})`. Transforms multiplicative price dynamics into additive, approximately Gaussian* returns. GPU-accelerated via CUDA when available.
- **Z-score Close 64**: Measures how many standard deviations the current close is from its 64-bar rolling mean. Values above +2 or below −2 indicate statistical overextension.
- **EMA Spread**: The difference between a fast 12-period EMA* and a slow 26-period EMA. Positive = upward momentum, negative = downward momentum.
- **ATR 14**: Average True Range, an exponentially smoothed measure of bar-to-bar volatility*. Captures the "breathing room" of the market.
- **Price Slope 20**: The linear slope of price over 20 bars. A quick, noise-resistant proxy for directional bias.

#### Mean Reversion Features (5 columns)

```
MR_FEATURE_COLUMNS = [
    "vwap_dev",           # Deviation from Volume-Weighted Average Price
    "bb_distance",        # Distance from 20-bar Bollinger Band center (in ±σ)
    "zscore_close_20",    # 20-bar rolling Z-score (shorter window = more sensitive)
    "rsi_14",             # Relative Strength Index (14 periods)
    "rsi_div_5",          # RSI(14) minus its own 5-bar lag (momentum of momentum)
]
```

- **VWAP Deviation**: `(close − vwap) / |vwap|`. Measures how far price has drifted from the fair-value anchor that institutional traders use.
- **Bollinger Band Distance**: `(close − 20-period mean) / (2 × 20-period std)`. A value of ±1 is at the outer band. Extreme values (+1.5 or −1.5) signal overextension.
- **RSI 14**: A bounded oscillator [0, 100]. Readings above 70 = overbought; below 30 = oversold.
- **RSI Divergence (5-bar)**: The change in RSI over 5 bars. Captures whether momentum is accelerating or decelerating — a leading indicator of potential reversal.

#### Scalper (Microstructure) Features (13 columns)

```
SCALPER_FEATURES = [
    "ofi_proxy",           # Order Flow Imbalance: (close−open) / (high−low)
    "microprice_dev",      # (close − vwap) / atr  — intrabar price pressure
    "spread_pct",          # (high − low) / close  — estimated bid-ask spread
    "log_return",          # Bar log return
    "vol_imbalance",       # (up_vol − down_vol) / total_vol
    "fracdiff_close_d04",  # Fractionally differenced close (d=0.4)
    "fracdiff_volume_d04", # Fractionally differenced volume (d=0.4)
    "buy_sell_pressure",   # taker_buy_base / taker_buy_quote
    "price_velocity_5",    # (close_t − close_{t-5}) / 5
    "price_velocity_10",   # (close_t − close_{t-10}) / 10
    "price_velocity_15",   # (close_t − close_{t-15}) / 15
    "volatility_z_32",     # Z-score of realized vol over 32 bars
    "vol_regime_code",     # 0=low, 1=medium, 2=high volatility regime
]
```

The scalper feature set is the richest because short-horizon prediction requires the most granular information about who is transacting and with what urgency.

- **OFI Proxy**: Approximates the imbalance between buy and sell pressure. Derived from `(close − open) / (high − low + ε)`, a causal bar-level proxy for the true Level 2 order flow imbalance.
- **Fractional Differentiation (d=0.4)**: A technique that partially differentiates a time series — removing enough non-stationarity* to make the series safe for ML while preserving long-range memory*. `d=0.4` is chosen to sit between raw price (d=0, has memory but non-stationary) and log returns (d=1, stationary but loses all memory).
- **Volatility Z-score (32-bar)**: Detects volatility regime transitions in real time. Used in conjunction with `vol_regime_code` which hard-codes the regime (low/medium/high) as a categorical feature.

#### Statistical Arbitrage Features (2 columns per asset)

```
fracdiff_close_d04,   # Fractionally differenced close (memory-preserving)
spread_z_64           # 64-bar rolling Z-score of close (per-asset)
```

For Stat Arb, the key transformation is **alignment**: all selected symbols are joined on a common timestamp index using inner-join semantics. This ensures every training row contains synchronous observations across all assets simultaneously — a necessary prerequisite for any cross-asset spread model.

#### Discretionary / Chart Features

The Discretionary archetype does **not** use numeric features alone. Instead, it rasterizes* OHLCV windows into **4-channel 32×32 images**:

- Channel 0: Normalized `open` values
- Channel 1: Normalized `high` values  
- Channel 2: Normalized `low` values  
- Channel 3: Normalized `close` values

Each bar occupies one column of the 32-pixel-wide image. Price values are mapped vertically: high price → pixel row 0 (top), low price → pixel row 31 (bottom). The image captures candlestick patterns visually, enabling convolutional networks to learn spatial chart patterns the same way a human analyst would read them.

For the Multimodal model, this image is supplemented with the 5 trend-style tabular features: `[log_return, zscore_close_64, ema_spread, atr_14, price_slope_20]`.

### 2.3 Normalization Protocol

**The scaler is always fit on training data only.** This is a hard rule enforced in `FeatureFactory.fit_scaler_train_only()`:

```python
ScalerStats(columns, mean, std)  # computed from train slice only
# Then applied to val and test:
FeatureFactory.transform_with_scaler(val_df, scaler)   # uses train mean/std
FeatureFactory.transform_with_scaler(test_df, scaler)  # uses train mean/std
```

Using Z-score normalization* (subtract mean, divide by std) rather than MinMax ensures outliers do not compress the rest of the distribution. This is standard practice in financial time series where extreme values are common.

---

## 3. The Iron Wall: Data Integrity & Temporal Split Protocol

### 3.1 Philosophy

Random shuffling of time-series data is **categorically forbidden**. Financial data has strong temporal autocorrelation — if you train on data from 2024 and validate on data from 2022, the model implicitly "knows the future" of its training environment. This is called **data leakage** and produces models that appear excellent in backtests but fail in live trading.

### 3.2 Chronological 70/15/15 Split

The `IronWallSplitter` (`data_pipeline/splitter.py`) implements a strict forward-only partitioning:

```
[═══════════════ 70% TRAINING ════════════╣  purge  ╠═ 15% VAL ═╣  purge  ╠═ 15% TEST ═]
                                                ↑                       ↑
                                        purge_gap_bars = 20     purge_gap_bars = 20
```

- **Training (70%):** The oldest portion of data. All gradient descent happens here.
- **Purge Gap (20 bars):** A "dead zone" discarded between sets. For a model predicting 20 bars ahead (the trend archetype horizon), this ensures no future information leaks backward into training.
- **Validation (15%):** Middle period. Used for early stopping* and hyperparameter selection. Model checkpoints are saved only when validation loss improves.
- **Purge Gap (20 bars):** Second buffer before test set.
- **Test / Out-of-Sample (15%):** The most recent data. Locked until the model is fully trained. Results on this set are the true measure of live performance.

### 3.3 Temporal Leakage Prevention Checklist

The splitter performs **hard assertions** on the split boundaries:

1. `train.timestamp.max() < val.timestamp.min()` — strictly enforced, raises ValueError if violated
2. `val.timestamp.max() < test.timestamp.min()` — strictly enforced
3. Scalers are fit only on `split.train` DataFrame
4. Any future-looking feature (`shift(-n)` for positive n) is only used as the **target label**, never as an input feature

### 3.4 Validity Criteria (Pass/Fail Gate)

After training, a model is evaluated on the Out-of-Sample test set. It passes only if **all** of the following hold:

| Criterion | Threshold | Rationale |
|---|---|---|
| Sharpe Ratio* | > 1.2 | Risk-adjusted return above passive holding |
| Directional Accuracy | > 55% | Better than coin flip by meaningful margin |
| Max Drawdown* | < 20% | Acceptable risk for live deployment |
| Profit Factor* | > 1.5 | Gross wins exceed gross losses by 50% |
| OOS / Val Sharpe decay | < 50% | No severe overfitting* |

Models that fail receive status `RESUME_TRAINING_REQUIRED` in the registry. All current models are in this state, meaning additional training epochs are needed.

---

## 4. Hardware & Execution Environment

### 4.1 GPU Backend

This system runs on an **AMD Radeon RX 6750 XT** via the **DirectML*** backend (`torch-directml`). All neural network layers are implemented using primitive PyTorch operations (Linear, sigmoid, tanh, matmul) to avoid triggering fused CUDA kernels that have no DirectML equivalent.

Key DirectML constraints respected throughout the codebase:
- `nn.MultiheadAttention` and `nn.TransformerEncoderLayer` are replaced with hand-coded `_ManualMHA` classes that use only `matmul` and `softmax`
- `nn.LSTM` and `nn.GRU` are replaced with `_ManualLSTMCell` / `_ManualGRUCell` that unroll manually
- `DataLoader` uses `num_workers=0` on DirectML (CPU worker threads cause saturation)
- Optimizer defaults to SGD + Nesterov momentum on DirectML (`AdamW` uses `lerp.Scalar_out` which is unsupported)

### 4.2 Device Selection

Device resolution is handled by `shared_training.resolve_device()`:

```python
resolve_device("auto")      # CUDA → DirectML → CPU fallback chain
resolve_device("cuda")      # Force CUDA; fall through if unavailable
resolve_device("directml")  # Force DirectML; raise error if unavailable
resolve_device("cpu")       # Always CPU
```

### 4.3 Reproducibility Seed

Global seed **42** is applied at the start of every training run:

```python
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

This guarantees identical weight initialization and data sampling order across runs.

---

## 5. Universal Evaluation Metrics

Every archetype produces a standard performance report. Understanding these metrics is essential for interpreting model quality.

### 5.1 Directional Accuracy

**Formula:** `(number of correct directional predictions) / (total predictions)`

- **What it measures:** How often the model predicts the correct side (up vs. down).
- **Target threshold:** > 55%
- **Interpretation:** 50% = pure chance (coin flip). Even 55% sustained accuracy is financially significant because wins can be sized larger than losses.
- **Limitation:** Does not capture magnitude. A model that correctly calls 55% of moves but misses all the large ones can still lose money.

### 5.2 Sharpe Ratio*

**Formula:** `μ_PnL / σ_PnL × √(252 × 24 × 12)`

The annualization factor `√(252 × 24 × 12)` adjusts for 5-minute bars (252 trading days × 24 hours × 12 five-minute bars per hour = 72,576 bars/year).

- **What it measures:** Return per unit of risk, annualized.
- **Target threshold:** > 1.2
- **Interpretation:** Sharpe of 1.0 = one unit of return per unit of risk. Institutional funds target 1.5–2.0. Sharpe above 3.0 often indicates overfitting.
- **Implementation:** `compute_sharpe(pnl_array)` in `shared_training.py`

### 5.3 Maximum Drawdown (MaxDD)*

**Formula:** `max( (peak_equity − current_equity) / peak_equity )`

- **What it measures:** The largest peak-to-trough percentage decline in the equity curve*.
- **Target threshold:** < 20%
- **Interpretation:** A MaxDD of 25% means the portfolio lost 25% of its value from its peak before recovering. This is the key measure of downside survivability — the model must not blow up an account.
- **Implementation:** `compute_max_drawdown(pnl_array)` in `shared_training.py`

### 5.4 Profit Factor*

**Formula:** `sum(winning_trades) / |sum(losing_trades)|`

- **What it measures:** Total gross profits divided by total gross losses.
- **Target threshold:** > 1.5
- **Interpretation:** A profit factor of 1.5 means the model makes $1.50 for every $1.00 lost. Values below 1.0 mean the model is net-losing.

### 5.5 Mean Absolute Error (MAE)

**Formula:** `mean(|predicted_value − actual_value|)`

- **Used in:** Regression models (Trend Following, Statistical Arbitrage)
- **What it measures:** Average prediction error magnitude, in the same units as the target.
- **Logged per epoch** in TensorBoard* under `val/mae`

### 5.6 Expected Shortfall (ES)* / Conditional Value-at-Risk (CVaR)

**Formula:** `mean( PnL | PnL < VaR_{α} )` — average loss in the worst `α`% of scenarios

- **Used for:** Risk management across all archetypes
- **What it measures:** Average loss in the tail of the distribution, beyond the VaR* threshold. ES is a coherent risk measure; VaR alone does not capture the severity of extreme losses.
- **Note:** The system currently tracks MaxDD as its primary tail-risk metric; ES is available as a secondary calculation.

### 5.7 Reconstruction Error (Stat Arb–specific)

**Formula:** `MSELoss(reconstructed_sequence, original_sequence)`

- **Used in:** `StatArbAutoencoder`
- **What it measures:** How well the autoencoder* can regenerate its input. High reconstruction error on a new bar = the current market configuration is anomalous relative to the learned normal regime.
- **Target threshold:** < 0.05

---

## 6. How to Run the Models

### 6.1 Environment Setup

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Verify GPU backend
python -c "import torch_directml; print(torch_directml.device())"
```

### 6.2 Training a Single Archetype

Each archetype has a dedicated Phase 4 training script under `quant_core/train_*_phase4.py`. These must be run **as modules** from the workspace root to avoid relative import errors:

```powershell
# Trend Following
python -m quant_core.train_trend_phase4 --config configs/trend_phase4.yaml

# Mean Reversion
python -m quant_core.train_mr_phase4 --config configs/mr_phase4.yaml

# Scalper
python -m quant_core.train_scalper_phase4 --config configs/scalper_phase4.yaml

# Statistical Arbitrage
python -m quant_core.train_stat_arb_phase4 --config configs/stat_arb_phase4.yaml

# Discretionary
python -m quant_core.train_discretionary_phase4 --config configs/discretionary_phase4.yaml

# Market Maker
python -m quant_core.train_mm_phase4 --config configs/mm_phase4.yaml
```

### 6.3 Full Sequential Sweep (All 18 Models)

```powershell
# Runs all archetypes in dependency order: trend → mr → scalper → stat_arb → discretionary → mm
python tools/run_full_phase4_sweep.py --start-from trend
```

The sweep runs with strict warning promotion: `-W error::FutureWarning -W error::UserWarning`. It stops on the first non-zero exit code.

### 6.4 Smoke Test (Fast Validation)

Smoke configs use fewer epochs and samples to verify the pipeline runs end-to-end without errors:

```powershell
python -m quant_core.train_trend_phase4 --config configs/trend_phase4_smoke.yaml
```

### 6.5 Inference / Prediction

All models expose the `TrendModelInterface` which provides `predict_with_confidence()`:

```python
import torch
from quant_core.trend_models import TrendLSTMModel

model = TrendLSTMModel(input_dim=5, hidden_size=128, num_layers=3)
model.load_state_dict(torch.load("models/checkpoints/trend/LSTM_Trend_v1/model_best.pt"))
model.eval()

# x shape: [batch=1, seq_len=96, features=5]
x = torch.randn(1, 96, 5)
output = model.predict_with_confidence(x)
# output.prediction: tanh-squashed directional signal in [-1, +1]
# output.confidence: sigmoid of |raw output|, in [0.5, 1.0]
```

The `prediction` output uses `tanh` squashing to bound the signal in [-1, +1]. A positive value means "predict upward move"; negative means "predict downward move". The `confidence` output is a monotonic function of the raw activation magnitude — the larger the model's "conviction", the higher the confidence.

### 6.6 TensorBoard Monitoring

```powershell
tensorboard --logdir models/tensorboard/
# Then open http://localhost:6006
```

Each archetype logs `train/loss`, `val/loss`, and `val/sharpe` per epoch.

### 6.7 Model Registry

After training, results are appended to `model_registry.json` using merge logic that **never overwrites** existing entries from other archetypes. Each entry records:

```json
{
  "architecture_name": "LSTM_Trend_v1",
  "archetype": "trend_follower",
  "weights_path": "models/checkpoints/trend/LSTM_Trend_v1/model_best.pt",
  "eval_timestamp": "2026-04-28T08:26:44Z",
  "device": "directml",
  "validation": {
    "status": "RESUME_TRAINING_REQUIRED | VALID",
    "directional_accuracy": 0.5162,
    "sharpe": 0.514,
    "profit_factor": 1.067,
    "max_drawdown": -0.368
  }
}
```

---

## 7. Archetype I — Trend Following (3 Models)

### Design Philosophy

Trend Following is the first and most fundamental archetype. It answers the most important question in trading: **Is the market currently going somewhere, or is it oscillating in place?**

Trend-following models are trained to predict the **magnitude and direction of the return `horizon` bars ahead** (20 bars = approximately 100 minutes on 5-minute data). The target is a continuous return value (regression), not a binary up/down classification. This allows the model to express confidence through prediction magnitude — a small predicted return suggests uncertainty; a large predicted return suggests strong trend signal.

**Optimal market conditions:** Strong directional moves with persistent momentum — post-breakout rallies, macro-driven flows, sustained selling in risk-off events. Underperforms severely in sideways, choppy, or mean-reverting conditions. Always run the Trend models first to establish regime context before applying any other archetype.

**Config:** `configs/trend_phase4.yaml`  
**Checkpoint root:** `models/checkpoints/trend/`  
**Input shape:** `[Batch, seq_len=96, features=5]`  
**Target:** Continuous return `(close_{t+20} / close_t) - 1`  
**Loss function:** `nn.SmoothL1Loss()` (Huber loss* — robust to outlier returns)  
**Success threshold:** Sharpe > 1.2, Directional Accuracy > 55%

---

### Model T-1: `LSTM_Trend_v1`

**Architecture class:** `TrendLSTMModel`  
**File:** `quant_core/trend_models.py`  
**Checkpoint:** `models/checkpoints/trend/LSTM_Trend_v1/model_best.pt`

#### Architecture Diagram

```
Input [B, 96, 5]
    │
    ▼
┌─────────────────────────────────────────────┐
│  Layer 1: _ManualLSTMCell(in=5, h=128)      │
│  Layer 2: _ManualLSTMCell(in=128, h=128)    │  ← 3-layer LSTM stack
│  Layer 3: _ManualLSTMCell(in=128, h=128)    │
│  (Dropout 0.1 between layers)               │
└─────────────────────────────────────────────┘
    │  (final hidden state h[−1])
    ▼
┌─────────────────────────────────────────────┐
│  LayerNorm(128) → Linear(128→64) → GELU     │
│  → Linear(64→1)                             │
└─────────────────────────────────────────────┘
    │
    ▼
Scalar prediction (continuous return)
```

#### Under the Hood

The LSTM* (Long Short-Term Memory) uses a custom `_ManualLSTMCell` to maintain DirectML compatibility. Each cell computes:

- **Input gate:** `i = sigmoid(W_i · x + U_i · h)`  — how much new information to absorb
- **Forget gate:** `f = sigmoid(W_f · x + U_f · h)` — how much past memory to retain
- **Cell update:** `g = tanh(W_g · x + U_g · h)` — candidate new memory content
- **Output gate:** `o = sigmoid(W_o · x + U_o · h)` — how much memory to expose as output
- **New cell state:** `c_new = f ⊙ c + i ⊙ g`
- **Output:** `h_new = o ⊙ tanh(c_new)`

The LSTM is well-suited to financial time series because it can learn to **ignore short-term noise** (via the forget gate) while selectively remembering significant past events (via the input gate). The 3-layer depth enables the network to build increasingly abstract representations of trend patterns.

**Hyperparameters:** `hidden_size=128, num_layers=3, dropout=0.1`  
**Parameters:** ~430,000  
**Training batch size:** 1,024  
**Inference input shape:** `[Batch, 96, 5]`

**Current registry status:** `RESUME_TRAINING_REQUIRED` (Sharpe=0.51, acc=51.6%)

---

### Model T-2: `Transformer_Trend_v1`

**Architecture class:** `TrendTransformerModel`  
**File:** `quant_core/trend_models.py`  
**Checkpoint:** `models/checkpoints/trend/Transformer_Trend_v1/model_best.pt`

#### Architecture Diagram

```
Input [B, 96, 5]
    │
    ▼  Linear(5 → 128) + Learnable Positional Embedding [1, 96, 128]
    │
    ▼
┌─────────────────────────────────────────────────┐
│  TransformerBlock × 2:                          │
│    Pre-LayerNorm → _ManualMHA(heads=4, d=128)   │
│    Pre-LayerNorm → FFN(128 → 512 → 128) + GELU  │
└─────────────────────────────────────────────────┘
    │  (mean pool over time dimension)
    ▼
LayerNorm(128) → Linear(128→1)
    │
    ▼
Scalar prediction
```

#### Under the Hood

The Transformer* Encoder processes all 96 time steps **simultaneously** (unlike LSTM which processes sequentially). The **attention mechanism*** computes:

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_head ) · V
```

Where Q, K, V are linear projections of the input. The softmax identifies which past time steps are most relevant to the current position. This gives the Transformer a powerful "long-range vision" — it can directly relate what happened 80 bars ago to what is happening now, without needing to pass the information through intermediate hidden states.

**Key design choice:** `_ManualMHA` replaces `nn.MultiheadAttention` to avoid the fused kernel `aten::_transformer_encoder_layer_fwd` which has no DirectML implementation. The manual version computes identical mathematics using only basic tensor operations.

Positional embeddings* are **learnable parameters** (not fixed sinusoidal functions) — the model learns which positions in the 96-bar window are most informative for trend prediction.

**Why this architecture is distinct from LSTM:** The Transformer captures global dependencies directly. LSTM processes left-to-right and can lose early context. For trend following, the Transformer may better learn patterns like "the setup from 60 bars ago predicts what happens 20 bars in the future."

**Hyperparameters:** `d_model=128, nhead=4, num_layers=2, dropout=0.1`  
**Parameters:** ~350,000  
**Current registry status:** `RESUME_TRAINING_REQUIRED` (Sharpe=0.80, acc=52.5%) — best of the three trend models

---

### Model T-3: `TCN_Trend_v1`

**Architecture class:** `TrendTCNModel`  
**File:** `quant_core/trend_models.py`  
**Checkpoint:** `models/checkpoints/trend/TCN_Trend_v1/model_best.pt`

#### Architecture Diagram

```
Input [B, 96, 5]
    │  transpose → [B, 5, 96]
    ▼
┌───────────────────────────────────────────────────────────┐
│  TemporalBlock(d=1): Conv1d(5→128, k=3, pad=1)  ×2      │
│  TemporalBlock(d=2): Conv1d(128→128, k=3, pad=2) ×2     │
│  TemporalBlock(d=4): Conv1d(128→128, k=3, pad=4) ×2     │
│  (Each block: Conv→BN→GELU→Drop + residual skip)         │
└───────────────────────────────────────────────────────────┘
    │
    ▼  AdaptiveAvgPool1d(1) → Flatten
    │
Linear(128→64) → GELU → Linear(64→1)
    │
    ▼
Scalar prediction
```

#### Under the Hood

The TCN* (Temporal Convolutional Network) uses **dilated convolutions*** to capture patterns at multiple temporal scales simultaneously without deep recurrence:

- **Dilation 1:** Looks at 3 consecutive bars — immediate micro-pattern
- **Dilation 2:** Looks at bars 1, 3, 5 bars apart — short-term pattern
- **Dilation 4:** Looks at bars 1, 5, 9 bars apart — medium-term pattern

The total **receptive field** with 3 dilation levels, kernel size 3 = `(3-1) × (1+2+4) + 1 = 15 bars`, meaning the network "sees" the previous 15 bars in a single forward pass through the dilation stack. Residual skip connections prevent gradient vanishing*.

**Why TCN vs LSTM:** TCNs are often faster to train (all positions processed in parallel), have stable gradients (no vanishing gradient through time), and can be highly parallelized on GPUs. However, they have a fixed receptive field, making them potentially less adaptive than LSTMs for highly variable sequence lengths.

**Hyperparameters:** `channels=128, dropout=0.1`  
**Parameters:** ~300,000  
**Current registry status:** `RESUME_TRAINING_REQUIRED` (Sharpe=0.61, acc=51.9%)

---

## 8. Archetype II — Mean Reversion (3 Models)

### Design Philosophy

Mean Reversion models answer: **Has price moved too far, too fast, and is it likely to snap back?**

These models operate on **tabular** (non-sequential) feature vectors. Each row is a single bar with 5 statistical features describing the current level of overextension. The target is **binary**: 1 if price moves up over the next 20 bars, 0 otherwise.

The loss function is `BCEWithLogitsLoss` (binary cross-entropy*). The primary evaluation metric is **Precision on Reversal** — the fraction of times the model predicts a reversal that actually occurs. False positives (betting on a reversal in a trending market) are catastrophically expensive in mean reversion strategies.

**Optimal market conditions:** Range-bound, consolidating, or oscillating markets. Post-exhaustion moves at key support/resistance levels. Optimal after a strong trend move that has stretched RSI above 70 or below 30. **Never use mean reversion models during strong trending phases without first consulting Trend models for regime context.**

**Config:** `configs/mr_phase4.yaml`  
**Checkpoint root:** `models/checkpoints/mean_reversion/`  
**Input shape:** `[Batch, features=5]` (tabular, no time dimension)  
**Target:** Binary `1` (price goes up in next 20 bars) or `0`  
**Loss function:** `nn.BCEWithLogitsLoss()`  
**Success threshold:** Precision on reversal > 60%, Sharpe > 1.2

---

### Model MR-1: `MLP_MR_v1`

**Architecture class:** `MeanReversionMLP`  
**File:** `quant_core/mean_reversion_models.py`  
**Checkpoint:** `models/checkpoints/mean_reversion/MLP_MR_v1/model_best.pt`

#### Architecture

```
Input [B, 5]
    │
    ▼
_MishBlock(5→256): Linear → BatchNorm1d → GELU → Dropout(0.1)
_MishBlock(256→256) × 3
    │
    ▼
Linear(256→1)   → BCEWithLogitsLoss target
```

#### Under the Hood

A **Deep MLP*** (Multi-Layer Perceptron) is the simplest and most interpretable neural architecture. Four fully-connected layers with `BatchNorm1d`* allow fast, stable training. `GELU`* activation is used instead of ReLU because it is smooth and differentiable everywhere — this matters for financial data where the boundary between "overextended" and "not overextended" is gradual, not sharp.

The MLP is the **baseline architecture** — if a more complex model (ResNet, GRN) cannot outperform the MLP, it is evidence that the additional complexity is not warranted by the data.

**Hyperparameters:** `hidden_size=256, num_layers=4, dropout=0.1`  
**Current registry status:** `RESUME_TRAINING_REQUIRED` (Sharpe=0.54, acc=51.7%)

---

### Model MR-2: `ResNet_MR_v1`

**Architecture class:** `MeanReversionResNet`  
**File:** `quant_core/mean_reversion_models.py`  
**Checkpoint:** `models/checkpoints/mean_reversion/ResNet_MR_v1/model_best.pt`

#### Architecture

```
Input [B, 5]
    │
    ▼  Linear(5→256) + GELU
    │
    ▼
ResBlock × 6:
    LayerNorm(256) → Linear → GELU → Drop → Linear → GELU
    + skip connection (x + block_output)
    │
    ▼
LayerNorm(256) → Linear(256→1)
```

#### Under the Hood

**Residual connections*** (skip connections) are the defining feature of ResNets*. At each block, the output is `h(x) + x` rather than just `h(x)`. This means:

1. The gradient can flow directly backward through the skip connection without passing through the block's non-linearities — eliminating gradient vanishing in deep networks.
2. Each block learns a **residual correction** rather than a full transformation — this is a lower-variance, easier learning problem.

With 6 residual blocks, the ResNet can capture significantly more complex non-linear interactions between features than the MLP while remaining trainable. It excels when mean-reversal signals arise from subtle interactions between RSI, VWAP deviation, and Bollinger distance that a shallow MLP might miss.

**Hyperparameters:** `hidden_size=256, depth=6, dropout=0.1`  
**Current registry status:** `RESUME_TRAINING_REQUIRED` (Sharpe=0.33, acc=51.0%)

---

### Model MR-3: `GRN_MR_v1`

**Architecture class:** `MeanReversionGRN`  
**File:** `quant_core/mean_reversion_models.py`  
**Checkpoint:** `models/checkpoints/mean_reversion/GRN_MR_v1/model_best.pt`

#### Architecture

```
Input [B, 5]
    │
    ▼  Linear(5→128) + ReLU
    │
    ▼
GRNBlock × 4:
    main = GELU( W_main · x )
    gate = sigmoid( W_gate · x )
    output = LayerNorm( x + Dropout(main ⊙ gate) )
    │
    ▼
LayerNorm(128) → Linear(128→1)
```

#### Under the Hood

The **Gated Residual Network*** (GRN) was introduced in the Temporal Fusion Transformer paper. The **gating mechanism** `sigmoid(W_gate · x)` acts as a learned soft switch: it multiplies the block output element-wise, allowing the network to suppress features that are not relevant to the current example.

This is analogous to **feature importance** in gradient boosting trees — the model learns to "turn off" noisy or irrelevant features adaptively per sample. For mean reversion, this is powerful because not all 5 features are equally informative in all market conditions. During high-volatility regimes, the ATR-based features may matter more; during low-volatility regimes, the RSI features may dominate.

The smaller hidden size (128 vs 256) reflects the GRN's efficiency — the gating removes the need for width to compensate for noise.

**Hyperparameters:** `hidden_size=128, depth=4, dropout=0.1`  
**Current registry status:** `RESUME_TRAINING_REQUIRED` (Sharpe=−0.44 — requires significant additional training)

---

## 9. Archetype III — Scalping / Microstructure (3 Models)

### Design Philosophy

Scalping models answer: **In the next few bars, is aggressive buying or selling pressure dominant?**

This is the highest-frequency archetype and the one most sensitive to transaction costs and execution latency. Scalping is profitable only when the model's edge (directional accuracy × average win size) exceeds the cost of trading (spread + fees + slippage*).

The models process **sequential feature windows** of 32 bars, learning to read order-flow patterns that precede short-horizon moves. The target is a **3-class problem**: `0=down, 1=flat, 2=up` over the next 5 bars, with a flat threshold of ±0.03% (below this return the bar is labeled flat, and the model is not expected to bet).

**Optimal market conditions:** High-volume, high-activity sessions (Asian open, US pre-market, macro events). Assets with tight bid-ask spreads and high-frequency institutional participation. Scalping is essentially useless in illiquid or thinly-traded assets.

**Config:** `configs/scalper_phase4.yaml`  
**Checkpoint root:** `models/checkpoints/scalper/`  
**Input shape:** `[Batch, seq_len=32, features=13]`  
**Target:** 3-class logits `[down, flat, up]`  
**Loss function:** `nn.CrossEntropyLoss()`  
**Critical success criteria:** Inference time < 10ms, Directional Accuracy > 55%

---

### Model SC-1: `CNN_Scalper_v1`

**Architecture class:** `ScalperCNN`  
**File:** `quant_core/scalper_models.py`  
**Checkpoint:** `models/checkpoints/scalper/CNN_Scalper_v1/model_best.pt`

#### Architecture

```
Input [B, 32, 13]
    │  transpose → [B, 13, 32]
    ▼
Conv1d(13→64, k=1) [stem — channel projection]
    │
    ▼
ConvBlock(d=1): Conv1d(64→64, k=3, dilation=1) → BN → LeakyReLU → Drop
ConvBlock(d=2): Conv1d(64→64, k=3, dilation=2) → BN → LeakyReLU → Drop
ConvBlock(d=4): Conv1d(64→128, k=3, dilation=4) → BN → LeakyReLU → Drop
    │
    ▼
AdaptiveAvgPool1d(1) → Flatten
Linear(128→64) → LeakyReLU → Dropout → Linear(64→3)
    │
    ▼
3 logits [down, flat, up]
```

#### Under the Hood

Dilated 1D convolutions* scan the time series at increasing receptive scales. For microstructure, the most important patterns are **local** — what happened in the last 3 bars, or the last 7 bars. The dilation stack with d=1,2,4 creates a receptive field of 15 bars without adding parameters. Each convolution filter acts as a learned pattern detector for local order-flow configurations.

`LeakyReLU` is used instead of ReLU because in microstructure modeling, the model must distinguish between weak negative and strong negative signals — a dead neuron (ReLU output permanently zero) is more harmful here than in trend modeling.

**Current registry status:** `RESUME_TRAINING_REQUIRED` (Sharpe=−0.07, acc=15.1% — the 3-class structure makes 33% the chance baseline, not 50%)

---

### Model SC-2: `LinearAttn_Scalper_v1`

**Architecture class:** `ScalperLinearAttn`  
**File:** `quant_core/scalper_models.py`  
**Checkpoint:** `models/checkpoints/scalper/LinearAttn_Scalper_v1/model_best.pt`

#### Architecture

```
Input [B, 32, 13]
    │
    ▼  Linear(13→64) projection
    │
    ▼
LinearAttnBlock × 2:
    Linear Attention O(T·d²) instead of O(T²·d)
    + Feed-Forward(64→128→64) + GELU
    (Pre-LayerNorm on both sub-layers)
    │
    ▼
Mean pool over time → LayerNorm(64) → Linear(64→3)
    │
    ▼
3 logits
```

#### Under the Hood

Standard Transformer attention has **O(T²)** complexity — for sequence length T=32, this is manageable, but for longer sequences it becomes prohibitive. **Linear Attention*** (Katharopoulos et al., 2020) replaces the softmax attention with a kernel approximation:

```
Standard: softmax(Q·Kᵀ)·V  — O(T²·d)
Linear:   Q·(Kᵀ·V)         — O(T·d²)
```

The kernel feature map used here is `φ(x) = relu(x) + 1` — a simple, DirectML-safe linearization. This produces an approximation of the full attention with no quadratic memory or compute cost.

For the 32-bar scalper input, the computational advantage over standard attention is minimal, but Linear Attention provides a different **inductive bias**: it aggregates global context without the sharp winner-take-all behavior of softmax. This may be preferable for order-flow data where multiple past bars contribute equally to the current signal.

**Current registry status:** `RESUME_TRAINING_REQUIRED` (Sharpe=−0.68)

---

### Model SC-3: `GRU_Scalper_v1`

**Architecture class:** `ScalperGRU`  
**File:** `quant_core/scalper_models.py`  
**Checkpoint:** `models/checkpoints/scalper/GRU_Scalper_v1/model_best.pt`

#### Architecture

```
Input [B, 32, 13]
    │
    ▼
BiGRU Layer 1 (forward + backward ManualGRUCell):
    forward:  h_0 → h_1 → ... → h_31
    backward: h_31 → h_30 → ... → h_0
    output:   concat([h_fwd, h_bwd]) per timestep → [B, 32, 128]
    │
BiGRU Layer 2 (same structure, input=128)
    │  (last time step)
    ▼
LayerNorm(128) → Linear(128→64) → LeakyReLU → Linear(64→3)
    │
    ▼
3 logits
```

#### Under the Hood

The **Bidirectional GRU*** runs two separate GRU cells simultaneously — one processing the sequence forward (past→present) and one backward (present→past). The backward pass may seem counter-intuitive for a causal trading model, but remember: **this is a supervised model trained on historical data**. At training time, the model sees the full 32-bar window and can benefit from knowing "what happened next" in the window to better learn what the beginning of the window means. At inference time, the full 32 bars are available (they are all historical), so no lookahead bias is introduced.

A **GRU*** (Gated Recurrent Unit) is a simplified LSTM with only two gates (reset and update) instead of four. For short sequences (32 bars) with high-frequency features, GRU is often preferred over LSTM because it has fewer parameters and trains faster.

**Current registry status:** `RESUME_TRAINING_REQUIRED`

---

## 10. Archetype IV — Statistical Arbitrage (3 Models)

### Design Philosophy

Statistical Arbitrage models answer: **Across correlated assets, has the spread deviated abnormally, and is it likely to converge?**

Unlike the previous archetypes that analyze a single asset, Stat Arb is fundamentally **multi-asset**. The input is a synchronized matrix of fractionally-differenced price series across up to 34 assets simultaneously. The model learns the stable co-movement relationships between assets and signals when those relationships are temporarily disrupted — which represents a tradable dislocation.

The target is the **mean spread Z-score** across assets in the next `horizon` bars — a regression target. A prediction near zero means assets are expected to remain in their normal co-integrated* relationship. A prediction far from zero suggests persistent dislocation.

**Optimal market conditions:** Periods of low correlation regime-shift, post-macro events where highly correlated assets have moved by different amounts, or during mean-reverting corrections after broad market dislocations. Requires at least 2 assets with historically stable correlation.

**Config:** `configs/stat_arb_phase4.yaml`  
**Checkpoint root:** `models/checkpoints/stat_arb/`  
**Input shape:** `[Batch, seq_len=64, num_assets]` (num_assets ≤ 34)  
**Target:** Scalar mean spread Z-score (regression)  
**Loss function:** `nn.MSELoss()`  
**Success threshold:** Reconstruction Error < 0.05, Sharpe > 1.2

---

### Model SA-1: `StatArb_Autoencoder_v1`

**Architecture class:** `StatArbAutoencoder`  
**File:** `quant_core/stat_arb_models.py`  
**Checkpoint:** `models/checkpoints/stat_arb/StatArb_Autoencoder_v1/model_best.pt`

#### Architecture

```
Input [B, 64, A]   (A = num_assets)
    │
    ▼  GRU Encoder (3-layer stack, hidden=32):
       Processes sequence left-to-right
       Output: latent vector z [B, 32]
    │
    ├─────────────────────────────────────────────┐
    │                                             │
    ▼ Regression head                             ▼ Decoder (for training only)
Linear(32→1) → spread prediction             GRU Decoder (latent → [B, 64, A])
                                             Reconstruction loss + regression loss
```

#### Under the Hood

The **Temporal Autoencoder*** is a powerful anomaly detection architecture. It is trained on two objectives simultaneously:

1. **Regression task:** Predict the next spread Z-score from the latent vector
2. **Reconstruction task:** Decode the latent vector back to the original input sequence (weighted by `recon_weight=0.5` in config)

The key insight is that the latent vector `z` is a bottleneck — it must compress the full 64-bar multi-asset sequence into a 32-dimensional representation. The model learns to encode the **normal co-movement structure** of the assets. When it encounters an abnormal regime at inference time, the reconstruction error spikes because the latent space does not have a good representation for the anomaly.

This means the model has two inference modes:
- **Predictive:** Use `regression_head(z)` to predict next spread Z-score
- **Anomaly detection:** Use `1 / (1 + reconstruction_loss)` as a confidence score — low confidence = current market structure is anomalous = potential arbitrage opportunity

**Hyperparameters:** `latent_dim=32, seq_len=64, dropout=0.1`

---

### Model SA-2: `StatArb_GAT_v1`

**Architecture class:** `StatArbGAT`  
**File:** `quant_core/stat_arb_models.py`  
**Checkpoint:** `models/checkpoints/stat_arb/StatArb_GAT_v1/model_best.pt`

#### Architecture

```
Input [B, 64, A]
    │  Per-asset temporal mean pooling → [B, A, 1]
    │  Linear(1→32) projection → [B, A, 32]
    │
    ▼
GATLayer × 2:
    W · h → (B, A, 32)
    attention scores: softmax( a_src + a_dst^T ) → (B, A, A)
    aggregate: attention × Wh → (B, A, 32)
    + residual skip + LayerNorm
    │
    ▼
Mean pool over assets → Linear(32→1)
    │
    ▼
Spread prediction scalar
```

#### Under the Hood

The **Graph Attention Network*** (GAT) treats each asset as a **node** in a fully-connected graph. Unlike static correlation matrices (which are fixed), the GAT learns **dynamic attention weights** that determine how much influence each asset has on the spread prediction.

The attention mechanism between asset `i` and asset `j`:
```
α_ij = softmax( LeakyReLU( a_src · Wh_i + a_dst · Wh_j ) )
```

This learns which asset pairs drive the spread. For example, if BTC and ETH are highly correlated, the GAT will learn to assign high mutual attention. If an altcoin suddenly decorrelates (during a specific event), the GAT can detect this as a change in the learned attention pattern.

The dense graph (all pairs) is valid for N ≤ 20 assets. For larger asset universes, sparse attention masks would be needed.

**Hyperparameters:** `d_model=32, num_layers=2, dropout=0.1`

---

### Model SA-3: `StatArb_LSTM_v1`

**Architecture class:** `StatArbLSTM`  
**File:** `quant_core/stat_arb_models.py`  
**Checkpoint:** `models/checkpoints/stat_arb/StatArb_LSTM_v1/model_best.pt`

#### Architecture

```
Input [B, 64, A]
    │
    ▼
2-layer ManualLSTMCell stack:
    Layer 1: LSTM(A → 64)
    Layer 2: LSTM(64 → 64)
    (Dropout 0.1 between layers)
    │  (final hidden state)
    ▼
LayerNorm(64) → Linear(64→32) → GELU → Linear(32→1)
    │
    ▼
Spread prediction scalar
```

The LSTM serves as a simpler baseline for Stat Arb — treating the multi-asset sequence as a single high-dimensional input. It is less capable of modeling inter-asset relationships than the GAT but is more robust and less prone to instability during training. **Hyperparameters:** `hidden_size=64, num_layers=2, dropout=0.1`

---

## 11. Archetype V — Discretionary / Multimodal (3 Models)

### Design Philosophy

The Discretionary archetype answers: **What does this chart pattern look like, and what historically follows it?**

This archetype attempts to replicate the pattern-recognition capability of an experienced human trader — someone who looks at a chart and thinks "this is a head-and-shoulders*, this is a double bottom*, this is a bull flag*." Human traders rely on visual gestalt recognition of chart shapes, not just numerical indicators.

To enable this, OHLCV data is **rasterized*** into 4-channel 32×32 pixel images (one column per bar, 32 bars total ≈ 160 minutes on 5-minute data). Convolutional networks and Vision Transformers then operate on these images exactly as they would on photographs.

The target is the same 3-class label as scalping (`down, flat, up`) but with a much longer horizon (20 bars). The flat threshold is higher (0.3%) reflecting the longer horizon.

**Optimal market conditions:** When price action is forming recognizable technical structures. Works best after the trend-following models have confirmed a regime, providing context for which patterns are more likely to complete. Useful for timing entries and exits on longer-term positions.

**Config:** `configs/discretionary_phase4.yaml`  
**Checkpoint root:** `models/checkpoints/discretionary/`  
**Input shape (image):** `[Batch, channels=4, H=32, W=32]`  
**Input shape (tabular, multimodal only):** `[Batch, tab_features=5]`  
**Target:** 3-class logits  
**Loss function:** `nn.CrossEntropyLoss()`

---

### Model D-1: `Disc_ViT_v1`

**Architecture class:** `DiscretionaryViT`  
**File:** `quant_core/discretionary_models.py`  
**Checkpoint:** `models/checkpoints/discretionary/Disc_ViT_v1/model_best.pt`

#### Architecture

```
Image [B, 4, 32, 32]
    │
    ▼  PatchEmbed: Conv2d(4→64, k=4, stride=4) → [B, 64, 64]
       (64 patches of size 4×4)
    │
    ▼  Prepend [CLS] token → [B, 65, 64]
       Add learnable positional embeddings [1, 65, 64]
    │
    ▼
ViTBlock × 4:
    Pre-LayerNorm → _ManualMHA(heads=4, d=64) → residual add
    Pre-LayerNorm → FFN(64→256→64) → residual add
    │
    ▼
LayerNorm → CLS token → Linear(64→3)
    │
    ▼
3 logits
```

#### Under the Hood

The **Vision Transformer*** (ViT) was introduced by Dosovitskiy et al. (2020). It divides the image into non-overlapping **patches** (here: 4×4 pixel patches), linearly embeds each patch, and treats the sequence of patch embeddings exactly like a text sequence in BERT.

The **[CLS] token*** is a learned vector prepended to the sequence that aggregates global information about the entire image through the attention mechanism. At the output, only the CLS token is used for classification — it represents the model's holistic understanding of the chart.

For a 32×32 image with 4×4 patches: there are (32/4)² = 64 patches + 1 CLS = 65 tokens. Each attention head can attend to all 65 tokens simultaneously, allowing the model to relate features in different regions of the chart without the spatial locality constraint of convolutions.

**Hyperparameters:** `img_size=32, patch_size=4, embed_dim=64, num_layers=4, nhead=4, dropout=0.1`

---

### Model D-2: `Disc_Multimodal_v1`

**Architecture class:** `DiscretionaryMultimodal`  
**File:** `quant_core/discretionary_models.py`  
**Checkpoint:** `models/checkpoints/discretionary/Disc_Multimodal_v1/model_best.pt`

#### Architecture

```
Image [B, 4, 32, 32]          Tabular [B, 5]
    │                               │
    ▼ _MiniCNN:                     ▼ Tabular MLP:
Conv2d(4→16)→BN→GELU               Linear(5→64)→GELU→Drop
Conv2d(16→32)→BN→GELU              Linear(64→64)→GELU
AdaptiveAvgPool2d(4)→Flatten
Linear(512→64)
    │                               │
    └─────────── Concat ────────────┘
                    │ [B, 128]
                    ▼
              LayerNorm(128)
              Linear(128→128) → GELU → Dropout
              Linear(128→3)
                    │
                    ▼
              3 logits
```

#### Under the Hood

The Multimodal model fuses two complementary information streams:

1. **Visual stream:** A lightweight CNN extracts local texture patterns from the chart image (candlestick body sizes, shadow lengths, cluster density).
2. **Quantitative stream:** A 2-layer MLP processes momentum and statistical features that encode the numerical context.

Concatenation fusion is the simplest multimodal approach — the two embeddings are concatenated and passed through a shared MLP. More sophisticated fusion strategies (cross-attention between modalities) could improve performance but at significantly higher complexity cost.

The rationale: visual patterns are ambiguous without context. A "bullish engulfing" pattern in an oversold market (low RSI, negative EMA spread) is a much stronger buy signal than the same pattern at a neutral market state. The tabular features provide that numerical context.

---

### Model D-3: `Disc_CNNChart_v1`

**Architecture class:** `DiscretionaryCNNChart`  
**File:** `quant_core/discretionary_models.py`  
**Checkpoint:** `models/checkpoints/discretionary/Disc_CNNChart_v1/model_best.pt`

#### Architecture

```
Image [B, 4, 32, 32]
    │
    ▼  Conv2d(4→32, k=3, pad=1) → BN → GELU
    │
    ▼
ChartResBlock × 3:
    Conv2d(32→32, 3, pad=1) → BN → GELU
    Conv2d(32→32, 3, pad=1) → BN
    + residual skip + GELU + Dropout
    │
    ▼  AdaptiveAvgPool2d(4) → Flatten
Linear(512→3)
    │
    ▼
3 logits
```

A pure ResNet-style CNN* on the chart image. Simpler than the ViT, with no attention mechanism. Convolutional operations detect local spatial patterns (edge detectors in early layers, complex shapes in later layers). The spatial locality of convolutions may be advantageous for chart patterns that have well-defined local structure (pin bars*, doji candles*, etc.).

---

## 12. Archetype VI — Market Making / RL (3 Models)

### Design Philosophy

Market Making models answer: **Given my current inventory, where should I quote bid and ask prices to maximize profit while controlling inventory risk?**

This archetype is fundamentally different from the previous five. It is a **Reinforcement Learning*** (RL) problem, not supervised learning. There is no "correct answer" to train against — only a reward signal that evaluates sequences of decisions over time.

The **`MarketMakingEnv`** is a simulation environment that replays historical OHLCV price data. At each step, the agent quotes bid and ask offsets around the mid-price. Fills are simulated probabilistically based on spread width and volatility. The reward function is:

```
Reward = ΔPnL + spread_capture − λ × |inventory| − transaction_costs − market_impact
```

Where:
- `ΔPnL`: Mark-to-market change in portfolio value
- `spread_capture`: Revenue from the bid-ask spread on filled orders  
- `λ × |inventory|`: Penalty for holding large positions (inventory risk*)
- `transaction_costs`: 0.05% per fill
- `market_impact`: Execution price degradation based on order size

**Asymmetric reward shaping:** Positive PnL is raised to power 0.75 (diminishing returns from gains), while negative PnL is raised to power 1.35 (increasing penalty for losses). This prevents the agent from taking large risks for marginal upside.

**Curriculum learning*:** Training progresses through three difficulty phases — EASY (trending, low inventory risk), MEDIUM (moderate), HARD (sideways, high volatility) — preventing the agent from over-specializing on one regime.

**State vector (10 dimensions):**
```
[inventory_normalized, mid_price_change, spread_pct, volatility,
 ofi_proxy, time_fraction, pnl_normalized, inventory_skew,
 funding_rate, open_interest_normalized]
```

**Config:** `configs/mm_phase4.yaml`  
**Checkpoint root:** `models/checkpoints/market_maker/`  
**State dim:** 10  
**Continuous action dim:** 2 `[bid_offset, ask_offset]` — normalized to [0,1]  
**Discrete actions:** 3 `[tight, medium, wide]`

---

### Model MM-1: `MM_PPO_v1`

**Architecture class:** `PPOActorCritic`  
**File:** `quant_core/market_maker_models.py`  
**Checkpoint:** `models/checkpoints/market_maker/PPO/model_best.pt`  
**Also:** `models/phase4_pruned/PPO/`

#### Architecture

```
State [B, 10]
    │
    ▼  Shared trunk:
Linear(10→256) → GELU → Dropout(0.05)
Linear(256→256) → GELU
    │
    ├───────────────────────────────────────────┐
    │                                           │
    ▼ Actor head                                ▼ Critic head
Linear(256→2) → sigmoid → mean                Linear(256→1)
log_std (learned parameter)                   V(s) scalar
    │
    ▼
action = Normal(mean, exp(log_std)).sample()
```

#### Under the Hood

**PPO*** (Proximal Policy Optimization) is an on-policy actor-critic algorithm. It alternates between:

1. **Rollout phase:** Run current policy for `rollout_len=512` steps, collecting (state, action, reward, next_state, done) tuples
2. **Optimization phase:** Update both actor and critic for `ppo_epochs=4` passes over the collected rollout

The key PPO innovation is the **clipped objective**:
```
L_clip = min( ratio × A, clip(ratio, 1−ε, 1+ε) × A )
```
Where `ratio = π_new(a|s) / π_old(a|s)` and `A` is the **advantage** estimate. The clipping prevents too-large policy updates, which cause instability in RL training.

The actor outputs a Gaussian distribution over bid/ask offsets (mean + learned log_std). The **GAE*** (Generalized Advantage Estimation) with `gae_lambda=0.95` provides low-variance advantage estimates.

**Training:** `max_steps=100,000` steps over historical price replays  
**Hyperparameters:** `hidden=256, clip_eps=0.2, gamma=0.99, gae_lambda=0.95`

---

### Model MM-2: `MM_SAC_v1`

**Architecture class:** `SACAgentNetworks`  
**File:** `quant_core/market_maker_models.py`  
**Checkpoint:** `models/checkpoints/market_maker/SAC/model_best.pt`  
**Also:** `models/phase4_pruned/SAC/`

#### Architecture

```
Actor:  State → MLP(10→256→256→4) → split → (mean[2], log_std[2])
Q1:     (State, Action) concat → MLP(12→256→256→1)
Q2:     (State, Action) concat → MLP(12→256→256→1)   ← Twin critics
Q1_target, Q2_target:  Soft-copies of Q1, Q2 (τ=0.005 EMA update)
log_alpha: Learnable entropy temperature
```

#### Under the Hood

**SAC*** (Soft Actor-Critic) is an off-policy maximum-entropy RL algorithm. Unlike PPO (on-policy), SAC uses a **replay buffer** (`replay_buffer_size=50,000`) and can reuse past experiences many times, making it sample-efficient.

The **maximum entropy framework** adds an entropy bonus to the reward:
```
J(π) = E[ Σ_t γ^t ( R_t + α × H(π(·|s_t)) ) ]
```
Where `H(π)` is the policy entropy and `α` is the temperature. Higher α = more exploration (more random quotes), lower α = more exploitation (tighter optimal quotes). SAC **automatically tunes** `α` via gradient descent on `log_alpha` to meet an entropy target.

The **twin critics** (Q1, Q2) address overestimation bias: SAC always uses `min(Q1, Q2)` for bootstrapping, preventing the agent from learning to exploit fictitiously high Q-values.

**Key hyperparameters:** `tau=0.005` (soft target update), `batch_size=2048`, `lr=0.0003`

---

### Model MM-3: `MM_DQN_v1`

**Architecture class:** `DQNNetwork`  
**File:** `quant_core/market_maker_models.py`  
**Checkpoint:** `models/checkpoints/market_maker/DQN/model_best.pt`  
**Also:** `models/phase4_pruned/DQN/`

#### Architecture

```
State [B, 10]
    │
    ▼  Shared trunk:
Linear(10→256) → GELU → Dropout(0.05)
Linear(256→256) → GELU
    │
    ├─────────────────────────────┐
    │                             │
    ▼ Value stream               ▼ Advantage stream
Linear(256→1)                   Linear(256→3)
V(s)                             A(s,a) for each of 3 actions
    │                             │
    └──────── Q = V + (A − mean(A)) ──────────┘
                    │
                    ▼
          Q-values for 3 discrete actions
          [tight_spread, medium_spread, wide_spread]
```

#### Under the Hood

**DQN*** (Deep Q-Network) discretizes the action space into 3 levels:
- `0 = Tight:` Narrow bid-ask spread — high fill probability, low per-unit profit
- `1 = Medium:` Balanced spread — moderate fill probability and profit
- `2 = Wide:` Wide bid-ask spread — low fill probability, high per-unit profit

The **Dueling architecture*** separates the state value `V(s)` (how good is this state overall) from the advantage `A(s,a)` (how much better is this action vs. average). This decomposition enables more efficient learning because `V(s)` can be updated without needing to observe all actions.

**ε-greedy exploration*:** Training starts with full randomness (`eps_start=1.0`), decaying exponentially to `eps_end=0.05` with `eps_decay=0.9995`. This ensures extensive exploration early in training before exploiting learned knowledge.

**Key hyperparameters:** `target_update_freq=500`, `replay_buffer_size=50,000`, `batch_size=2048`

---

## 13. Strategic Model Selection Guide

### 13.1 Market Regime Decision Tree

```
START: What is the current market doing?

├── Strong trend (EMA spread positive/negative, slope consistent)
│   └── PRIMARY: TrendTransformer or TrendLSTM
│       ├── Trend confirmed upward?
│       │   ├── Check Scalper (SC-3 GRU): Is order flow supporting?
│       │   └── Check Stat Arb (SA-2 GAT): Are correlated assets following?
│       └── Trend confirmed downward? → same checks, reverse direction
│
├── Ranging / sideways (Z-score oscillating, low slope)
│   └── PRIMARY: MeanReversionMLP or MeanReversionResNet
│       ├── RSI > 70 or Bollinger > +1.0? → SHORT signal
│       └── RSI < 30 or Bollinger < -1.0? → LONG signal
│
├── High-volatility spike (ATR spike, vol_regime_code=2)
│   └── PRIMARY: ScalperCNN (short-term momentum) + StatArbAutoencoder (anomaly detection)
│       └── If SA reconstruction error high: potential dislocation — watch for reversal
│
├── Post-pattern recognition needed (key support/resistance, chart structures)
│   └── PRIMARY: Disc_ViT or Disc_Multimodal
│       └── Use after Trend context is known (D models work best with direction bias)
│
└── Active position management (inventory, continuous quoting)
    └── PRIMARY: MM_SAC (most sophisticated, best for continuous action)
        └── Secondary: MM_PPO (smoother updates) or MM_DQN (simple discrete cases)
```

### 13.2 Archetype Strengths and Weakness Matrix

| Archetype | Best In | Worst In | Time Horizon | Complexity |
|---|---|---|---|---|
| Trend Following | Strong trends, breakouts | Ranging, choppy | 96 bars ahead | Medium |
| Mean Reversion | Range-bound, post-exhaustion | Strong trends | 20 bars ahead | Low |
| Scalping | High volume, tight spreads | Low liquidity, wide spreads | 5 bars ahead | High (latency) |
| Stat Arb | Correlated assets, macro dislocations | Decorrelated markets | 10 bars ahead | High (multi-asset) |
| Discretionary | Pattern-rich environments | Featureless chop | 20 bars ahead | Medium |
| Market Making | Continuous quoting, high frequency | One-directional markets | Continuous | Very High (RL) |

### 13.3 Multi-Model Ensemble Strategy

For maximum robustness, combine model predictions using a weighted vote:

```python
# Example ensemble for directional bias
trend_signal = LSTM_Trend(x)          # continuous return prediction
mr_signal = MLP_MR(x)                 # binary reversal probability
scalper_signal = GRU_Scalper(x)       # 3-class order flow direction

# Normalize and combine
consensus = 0.5 × trend_signal + 0.3 × (mr_signal - 0.5) + 0.2 × scalper_signal

# Trade only when consensus exceeds threshold
if abs(consensus) > 0.3:
    direction = sign(consensus)
```

A key principle: **models that agree on direction but arrive at it through different paths (technical, statistical, microstructure) provide much stronger combined signal than multiple models trained on the same features.**

---

## 14. Model Registry Reference

All 18 trained models are tracked in `model_registry.json`. Current status (as of 2026-04-28):

| Model ID | Architecture Name | Archetype | Sharpe | Directional Acc | MaxDD | Status |
|---|---|---|---|---|---|---|
| T-1 | `LSTM_Trend_v1` | Trend | 0.51 | 51.6% | -36.8% | RESUME_TRAINING |
| T-2 | `Transformer_Trend_v1` | Trend | 0.80 | 52.5% | -22.2% | RESUME_TRAINING |
| T-3 | `TCN_Trend_v1` | Trend | 0.61 | 51.9% | -38.5% | RESUME_TRAINING |
| MR-1 | `MLP_MR_v1` | Mean Reversion | 0.54 | 51.7% | -38.5% | RESUME_TRAINING |
| MR-2 | `ResNet_MR_v1` | Mean Reversion | 0.33 | 51.0% | -24.8% | RESUME_TRAINING |
| MR-3 | `GRN_MR_v1` | Mean Reversion | -0.44 | 48.6% | -59.8% | RESUME_TRAINING |
| SC-1 | `CNN_Scalper_v1` | Scalping | -0.07 | 15.1%* | -13.6% | RESUME_TRAINING |
| SC-2 | `LinearAttn_Scalper_v1` | Scalping | -0.68 | 40.4% | -54.7% | RESUME_TRAINING |
| SC-3 | `GRU_Scalper_v1` | Scalping | — | — | — | RESUME_TRAINING |
| SA-1 | `StatArb_Autoencoder_v1` | Stat Arb | — | — | — | RESUME_TRAINING |
| SA-2 | `StatArb_GAT_v1` | Stat Arb | — | — | — | RESUME_TRAINING |
| SA-3 | `StatArb_LSTM_v1` | Stat Arb | — | — | — | RESUME_TRAINING |
| D-1 | `Disc_ViT_v1` | Discretionary | — | — | — | RESUME_TRAINING |
| D-2 | `Disc_Multimodal_v1` | Discretionary | — | — | — | RESUME_TRAINING |
| D-3 | `Disc_CNNChart_v1` | Discretionary | — | — | — | RESUME_TRAINING |
| MM-1 | `MM_PPO_v1` | Market Maker | — | — | — | RESUME_TRAINING |
| MM-2 | `MM_SAC_v1` | Market Maker | — | — | — | RESUME_TRAINING |
| MM-3 | `MM_DQN_v1` | Market Maker | — | — | — | RESUME_TRAINING |

*SC-1 directional accuracy appears low (15%) due to 3-class target where ~33% is the chance baseline. The raw number needs to be interpreted differently — a 3-class model accuracy above 40% is meaningful.

**Action required:** All models require additional training epochs to reach the Sharpe > 1.2 validity threshold. The `Transformer_Trend_v1` (T-2) with Sharpe=0.80 is closest to passing.

---

## 15. Global Glossary of Technical Terms

**ATR (Average True Range):** A measure of market volatility that captures the average range between bar high and low, adjusted for overnight gaps. Higher ATR = more volatile market. Used as a normalizing denominator in several features.

**Attention mechanism:** In Transformer models, a learned function that determines how much each position in a sequence should "attend to" (weight) every other position. The model learns which past time steps are most relevant to predicting the current output.

**Autoencoder:** A neural network trained to compress input data into a smaller latent representation and then reconstruct the original input. The reconstruction error measures how "normal" the input is — high error implies anomalous data.

**Backpropagation:** The algorithm used to train neural networks. It computes the gradient of the loss with respect to each weight by applying the chain rule of calculus backward through the network layers.

**BatchNorm1d (Batch Normalization):** A technique that normalizes activations within a mini-batch* during training, stabilizing and accelerating convergence. Also acts as a mild regularizer. Works best with tabular data in fully-connected layers.

**BCEWithLogitsLoss (Binary Cross-Entropy):** Loss function for binary classification. Combines a sigmoid activation and cross-entropy in a numerically stable formula. Measures how far the model's predicted probability is from the true binary label.

**Bidirectional GRU:** A GRU that processes the sequence both forward (past→future) and backward (future→past) simultaneously. Useful for classification when the full sequence is available at inference time.

**Bull flag / Bear flag:** A chart continuation pattern where price consolidates in a narrow channel after a strong move, before resuming the original direction.

**CLS token (Classification Token):** In Vision Transformers and BERT, a special learnable vector prepended to the sequence that aggregates global information via attention. The final CLS embedding is used for classification output.

**Co-integration:** A statistical property where two non-stationary time series move together in the long run such that their linear combination is stationary. Co-integrated pairs are ideal for statistical arbitrage.

**CrossEntropyLoss:** Loss function for multi-class classification. Measures the negative log-probability assigned to the correct class. Used in Scalper (3-class) and Discretionary (3-class) archetypes.

**Curriculum learning:** A training strategy where the difficulty of training examples gradually increases. Used in Market Making to train first on trending, easy environments before exposing the agent to chaotic conditions.

**Dilated convolution:** A convolution where the kernel samples input at regular intervals (the dilation factor), creating a large receptive field without increasing the number of parameters. Dilation=2 means the filter skips every other input value.

**DirectML:** Microsoft's machine learning API that enables GPU acceleration on non-NVIDIA hardware (AMD, Intel) through DirectX 12. Used as the GPU backend on the AMD Radeon RX 6750 XT in this project.

**DQN (Deep Q-Network):** A reinforcement learning algorithm that learns a Q-function (expected cumulative reward for each state-action pair) using a neural network. Actions are selected greedily (highest Q-value).

**Double bottom:** A bullish reversal chart pattern where price makes two similar lows separated by a rally, forming a "W" shape.

**Doji candle:** A candlestick where open and close prices are almost identical, indicating market indecision.

**Dueling architecture:** A DQN variant that separately estimates state value V(s) and action advantage A(s,a), then combines them as Q(s,a) = V(s) + A(s,a) − mean(A). Improves learning efficiency.

**Early stopping:** A regularization technique that halts training when validation loss stops improving for `patience` epochs, preventing overfitting to training data.

**EMA (Exponential Moving Average):** A weighted moving average that gives exponentially more weight to recent values. EMA(12) reacts faster than EMA(26), creating the spread used as a momentum feature.

**Ensemble:** Combining predictions from multiple models. Typically improves accuracy and robustness compared to any single model.

**ε-greedy exploration:** In DQN, with probability ε take a random action (explore), otherwise take the best-known action (exploit). ε decays during training to shift from exploration to exploitation.

**Equity curve:** A time-series plot of the cumulative profit or loss from trading. A healthy equity curve trends upward with manageable drawdowns.

**Expected Shortfall (ES) / CVaR:** The average loss in the worst `α%` of scenarios (e.g., average of the 5% worst daily returns). A more complete measure of tail risk than Value-at-Risk alone.

**Fractional differentiation:** A technique that applies a fractional-order difference operator to a time series, removing just enough non-stationarity to be ML-safe while preserving long-range memory. `d=0` = raw price, `d=1` = first difference (returns), `d=0.4` = an intermediate value.

**GAE (Generalized Advantage Estimation):** A method to estimate the advantage function A(s,a) in RL by computing a weighted sum of TD errors at multiple future time steps. Controls bias-variance tradeoff via λ parameter.

**Gaussian:** A normal distribution, the bell curve. Many financial models assume returns are approximately Gaussian, though in practice they have fat tails.

**GELU (Gaussian Error Linear Unit):** A smooth activation function `x × Φ(x)` where Φ is the Gaussian CDF. Behaves similarly to ReLU but is smooth (differentiable everywhere) and has small non-zero gradients for negative inputs.

**GNN (Graph Neural Network):** A neural network that operates on graph-structured data where nodes have features and edges encode relationships. Used in Stat Arb to model asset correlations as a graph.

**Gradient vanishing:** A problem in deep networks where gradients become extremely small during backpropagation, preventing early layers from learning. Residual connections and LSTM gates are designed to mitigate this.

**GRN (Gated Residual Network):** An architecture block from the Temporal Fusion Transformer paper. Combines a learned gating mechanism (sigmoid) with residual skip connections, enabling the network to selectively suppress irrelevant features.

**GRU (Gated Recurrent Unit):** A recurrent architecture with two gates (reset and update) instead of LSTM's four. Faster and simpler than LSTM; often equally effective for shorter sequences.

**Head-and-shoulders:** A bearish reversal chart pattern consisting of three peaks — a higher central peak (the "head") flanked by two lower peaks (the "shoulders").

**Huber loss (SmoothL1Loss):** A loss function that behaves like L1 (Mean Absolute Error) for large residuals and L2 (Mean Squared Error) for small residuals. Robust to outlier returns while still being smooth for gradient computation.

**Inductive bias:** The set of assumptions a learning algorithm makes about the structure of the data. LSTM assumes sequential dependencies; CNN assumes local spatial patterns; Transformer assumes global pairwise relationships.

**Inventory risk:** In market making, the risk of holding a large directional position. If you bought heavily and the market drops, the inventory loss can exceed spread revenue.

**Iron Wall Splitter:** The proprietary data splitting module that enforces strict chronological train/val/test splits with purge gaps, preventing any form of temporal data leakage.

**LSTM (Long Short-Term Memory):** A recurrent neural network cell with four gates (input, forget, cell update, output) that enables learning long-range dependencies in sequences while preventing gradient vanishing.

**Long-range memory:** The statistical property of a time series where values many time steps apart are still correlated. Financial prices have long memory; pure returns do not. Fractional differentiation preserves a controlled amount of this memory.

**Market impact:** The adverse price movement caused by executing a large order. If you buy aggressively, the price rises, and your average execution price is worse than the mid-price.

**MaxDD (Maximum Drawdown):** The largest peak-to-trough decline in equity. If your portfolio peaked at $100,000 and fell to $80,000 before recovering, MaxDD = 20%.

**Mean Absolute Error (MAE):** The average of |predicted − actual| values. A regression metric in the same units as the prediction target.

**Min-batch:** A small subset of the training data used for one gradient update step. Typical sizes: 256, 1024, 2048 samples.

**MSELoss (Mean Squared Error):** Loss function for regression. Penalizes large errors quadratically. Sensitive to outliers but smooth for gradient computation.

**Multi-head attention:** Attention computed by multiple parallel "heads", each learning to attend to different aspects of the input. Final output is the concatenation of all heads projected to output dimension.

**Non-stationarity:** A time series is non-stationary if its statistical properties (mean, variance) change over time. Raw price series are non-stationary; returns are approximately stationary.

**OHLCV:** Open, High, Low, Close, Volume — the standard representation of a financial price bar for a given time period.

**Overfitting:** When a model memorizes training data patterns that do not generalize to new data. Manifests as high training performance but low test performance.

**Parquet:** A columnar storage format for large datasets. Highly compressed and fast for analytical queries. Used to store all historical bar data.

**Pin bar:** A candlestick with a very long shadow (wick) relative to its body, indicating strong rejection of a price level. Typically a reversal signal.

**Positional embedding:** In Transformer models, a learned or fixed vector added to each input to encode its position in the sequence. Without it, the attention mechanism is permutation-invariant (order-agnostic).

**PPO (Proximal Policy Optimization):** An on-policy RL algorithm that uses a clipped objective to prevent destructively large policy updates. More sample-efficient than REINFORCE and more stable than TRPO.

**Profit Factor:** Gross wins divided by the absolute value of gross losses. A profit factor of 2.0 means for every $1 lost, $2 is earned on average.

**Rasterize:** To convert vector or numerical data into a pixel-based image representation. Here, OHLCV bars are rasterized into chart images for visual pattern recognition.

**Reinforcement Learning (RL):** A machine learning paradigm where an agent learns to take actions in an environment to maximize cumulative reward. No labeled training data is needed — the agent learns from experience.

**ResNet (Residual Network):** A neural network architecture where each block adds its output to the block's input (skip connection): `output = F(x) + x`. This enables training of very deep networks without gradient vanishing.

**SAC (Soft Actor-Critic):** An off-policy RL algorithm that maximizes a combination of reward and policy entropy. The entropy term encourages exploration and prevents premature convergence to suboptimal deterministic policies.

**Sharpe Ratio:** Risk-adjusted return = mean(PnL) / std(PnL) × √(annualization_factor). A Sharpe of 1.0 means one unit of return per unit of risk, annualized. Generally: below 0.5 = poor, 1.0–2.0 = good, above 3.0 = suspicious.

**Slippage:** The difference between the expected execution price and the actual execution price, caused by market movement during order processing and market impact.

**Standard deviation (σ):** A measure of how spread out values are from their mean. In finance, used as a measure of volatility*.

**Stochastic:** Involving randomness or probabilistic processes. In optimization, "stochastic gradient descent" updates weights based on random mini-batches rather than the full dataset.

**TensorBoard:** Google's visualization toolkit for machine learning, allowing real-time monitoring of training metrics (loss, accuracy, learning rate) via a web interface.

**Temporal autoencoder:** An autoencoder applied to sequential data. The encoder uses recurrent layers to compress a time series into a latent vector; the decoder reconstructs the original sequence from the latent vector.

**Transformer:** A neural architecture based entirely on attention mechanisms (no convolutions or recurrence). Introduced in "Attention Is All You Need" (Vaswani et al., 2017). Can capture global dependencies in sequences with O(T²) complexity.

**Value-at-Risk (VaR):** The maximum expected loss at a given confidence level over a specific time horizon. For example, 1-day 95% VaR = the loss level exceeded only 5% of days.

**Vision Transformer (ViT):** A Transformer architecture applied to images by dividing them into patches and treating the patch sequence like a text sequence. Effective for global pattern recognition.

**VWAP (Volume-Weighted Average Price):** The average price weighted by volume, calculated as `sum(price × volume) / sum(volume)` over a period. Used as a benchmark by institutional traders; deviations from VWAP signal temporary mispricing.

**Volatility:** The degree of variation in a price series over time. Measured by the standard deviation of returns. High volatility = large price swings; low volatility = stable, range-bound price.

**Walk-forward validation:** A validation methodology where the model is trained on a growing window of past data and tested on the immediately following period, repeated across the full timeline. More realistic than a single train/test split.

**Z-score normalization:** Standardizing a variable by subtracting its mean and dividing by its standard deviation. Output has mean=0 and std=1. A Z-score of +2.0 means the value is 2 standard deviations above average — typically considered statistically extreme.

---

*End of ChatTrader.KPai Models Handbook v1.0 — Phase 4*  
*Generated: 2026-04-28 | Author: ChatTrader.KPai Analytics Engine*
