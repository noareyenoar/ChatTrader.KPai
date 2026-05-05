🚨 GLOBAL PYTORCH & GPU OPTIMIZATION STANDARDS
Before implementing any module, the Agent MUST adhere to these performance standards:

VRAM Lifecycle Management:

Use torch.utils.data.DataLoader with pin_memory=True and num_workers > 0.

Explicitly call del on large tensors and execute torch.cuda.empty_cache() at the end of every training/backtest epoch to prevent OOM errors.

Precision & Acceleration:

Use torch.amp.autocast('cuda') and GradScaler for Mixed Precision (FP16) training to maximize throughput.

Apply torch.compile(model) on finalized architectures for kernel fusion.

Data Handling:

Preprocessing must be vectorized using torch or numpy. Never iterate through rows in Python.

Large datasets (Scalping/LOB) must be memory-mapped (numpy.memmap) or pre-loaded into GPU memory if the capacity allows.

1. TREND FOLLOWING
Goal: Detect trend regimes and persistent momentum.
Tensor Shape: [Batch, Seq_Len (50-200), Num_Features]

Data Pipeline:

Input: OHLCV (Multi-TF: 15m, 1h, 4h).

Transform: Log Returns torch.log(P_t / P_{t-1}) + Rolling Z-Score.

Synthetic Features: EMA Crossovers, ATR (Volatility), Price Slope.

Architectures:

LSTM: nn.LSTM(hidden_size=128, num_layers=3, batch_first=True).

Transformer: nn.TransformerEncoder with learnable positional encoding.

TCN: nn.Conv1d with dilation stacks for wide receptive fields.

Guideline: Loss = nn.SmoothL1Loss(). Metric = Sharpe Ratio > 1.2.

2. MEAN REVERSION
Goal: Identify over-extended price deviations.
Tensor Shape: [Batch, Num_Features]

Data Pipeline:

Input: Price, VWAP.

Transform: Stationarity-focused normalization (Z-score).

Synthetic Features: Bollinger Band distance (%), RSI divergence.

Architectures:

Deep MLP: nn.Linear layers with nn.Mish() activation + nn.BatchNorm1d.

Tabular ResNet: Residual blocks to prevent vanishing gradients.

Hybrid: LightGBM feature selection followed by PyTorch NN refinement.

Guideline: Loss = nn.BCEWithLogitsLoss(). Focus on Precision (don't bet on false reversals).

3. SCALPING / MICROSTRUCTURE
Goal: Exploit sub-second order flow inefficiency.
Tensor Shape: [Batch, Bid_Ask_Depth (Levels), Features]

Data Pipeline:

Input: LOB (Level 2).

Transform: torch.log1p(volume) to handle extreme outliers.

Synthetic Features: Order Flow Imbalance (OFI), Microprice, Spread.

Architectures:

CNN (LOB Image): nn.Conv2d to capture order book spatial density.

Linear Attention Transformer: Optimized for low-latency inference.

Guideline: Loss = nn.CrossEntropyLoss(). Inference must be optimized for < 5ms latency.

4. STATISTICAL ARBITRAGE
Goal: Predict spread convergence between correlated assets.
Tensor Shape: [Batch, Num_Assets, Seq_Len]

Data Pipeline:

Transform: Fractional Differentiation to preserve memory.

Synthetic Features: Spread Z-score, PCA components.

Architectures:

Temporal Autoencoder: Learn latent mispricing via reconstruction error.

GNN (via PyTorch Geometric): GCNConv to model asset relationship graphs.

Guideline: Loss = nn.MSELoss(). Metric = Tracking Error.

5. DISCRETIONARY (MULTIMODAL)
Goal: Mimic human chart/sentiment analysis.

Data Pipeline: Image (Charts) + Text (Sentiment).

Architectures:

ViT (Vision Transformer): Pretrained timm ViT backbone.

Multimodal Fusion: Concatenate embedding vectors from ViT and BERT; pass to nn.Linear head.

Guideline: Loss = nn.CrossEntropyLoss() (Imitation Learning).

6. MARKET MAKING
Goal: Quote management and inventory control.
Tensor Shape: [Batch, State_Vector (Inventory, Spread, Vol)]

Data Pipeline: State encoding (Inventory must be constrained to [-1, 1]).

Architectures:

PPO Policy Network: Actor-Critic architecture using nn.Sequential.

Guideline: Custom Reward Function = ΔPnL - (λ * |Inventory_Risk|). Requires simulation environment integration.

🛠 TESTING & VALIDATION STANDARDS
Walk-Forward Validation: Mandatory to prevent look-ahead bias.

Backtest Engine: Must simulate slippage, fees, and latency (especially for Scalping/MM).

Performance Report: Agent must generate performance_summary.md after training each archetype, including:

Sharpe Ratio, Max Drawdown, Profit Factor, and Training Time (s).

Instructions for the Coder Agent:
Phase 1-4: Build the pipeline based on these architectures.

Verification: For every model, run a "Sanity Check" using random noise to ensure the architecture accepts the input shape and propagates gradients correctly before feeding actual data.

Checkpointing: Save model_best.pt and optimizer_state.pt only if the validation Sharpe Ratio exceeds the benchmark.

Logging: Use TensorBoard to track Loss/Validation metrics for all 18 models.