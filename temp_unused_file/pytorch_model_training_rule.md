🎯 PURPOSE
This document serves as the Supreme Training Protocol for ChatTrader.KPai. It mandates a zero-tolerance policy for data leakage and defines strict hardware and statistical standards for all 18 models across 6 archetypes.

1. THE "IRON WALL" DATA SPLITTING (NO LEAKAGE POLICY)
To ensure 100% foolproof training, the Agent must strictly follow these temporal constraints. Random sampling across the entire timeline is FORBIDDEN.

1.1 Chronological Split (70% / 15% / 15%)
Training (70%): Oldest data. Used for gradient descent.

Validation (15%): Middle data. Used for hyperparameter tuning and Early Stopping.

Testing (15%): Most recent data (Out-of-Sample). Locked until the model is finalized.

1.2 Leakage Prevention Checklist
No Lookahead: Features at time t must only contain information from t−1 or earlier.

Gap Buffer: Implement a "Purge Gap" between Train/Val and Val/Test sets equal to the maximum prediction horizon (e.g., if predicting t+20, drop 20 bars between sets) to prevent information bleeding.

Scaling Leakage: Fit scalers (StandardScaler/MinMax) ONLY on the Training set. Use the fitted parameters to Transform Val and Test sets.

2. GPU EXECUTION & RESOURCE MANAGEMENT
Every model MUST be optimized for CUDA execution.

Device Handling: Explicitly use device = torch.device("cuda" if torch.cuda.is_available() else "cpu").

Memory Efficiency:

Use torch.cuda.amp.autocast() for mixed-precision training.

Cleanup: After training each of the 18 models, the agent must execute:

Python
model.cpu()
del model, optimizer
torch.cuda.empty_cache()
Data Loading: Use DataLoader(pin_memory=True, num_workers=4).

3. ARCHETYPE-SPECIFIC VALIDATION STANDARDS
A model is only VALID if it passes the specific "Success Metric" for its architecture.

Archetype	Architecture	Validation Focus	Success Threshold
Trend Follower	LSTM / Transformer	Directional Consistency	Directional Accuracy > 55%
Mean Reversion	MLP / ResNet	Catching Extremes	Precision on Reversal > 60%
Scalper	CNN / Transformer	Latency & Tick Flow	Inference Time < 10ms
Stat Arb	Autoencoder / GNN	Spread Stability	Reconstruction Error < 0.05
Discretionary	ViT / Multimodal	Pattern Alignment	F1-Score > 0.65
Market Maker	RL (PPO/SAC)	Inventory Risk	Reward Stability (StdDev < 0.2)
4. REAL-WORLD VALIDITY LOGIC (OOS COMPARISON)
After testing on the 15% Out-of-Sample (OOS) data, the Agent must run this diagnostic:

❌ Model is INVALID if:
Performance Decay: Test PnL or Sharpe is > 50% lower than Validation.

Overfitting Sign: Training Loss is near 0 while Validation Loss remains high.

Regime Failure: Model performs well in "Trending" but loses > 40% in "Sideways" (unless specialized).

✅ Model is VALID if:
Consistency: Sharpe Ratio > 1.2 across both Val and Test sets.

Profit Factor: > 1.5 in Test (OOS) sample.

Maximum Drawdown: < 20% in Test sample.

5. MODEL CATALOG & DESIGN PREMISE
The Agent must output a model_registry.json containing:

Architecture Name: (e.g., TCN_Trend_v1)

Weights Path: /models/checkpoints/...

Design Premise: Why this NN was chosen for this archetype.

Validation Audit: A boolean flag is_valid based on Section 4 criteria.

6. REPRODUCIBILITY & LOGGING
Global Seed: 42 (Must be set for torch, numpy, and random).

Logging: Every epoch must log Train_Loss, Val_Loss, and Current_LR to TensorBoard.

Checkpointing: Save only when Val_Loss improves (Early Stopping patience: 10 epochs).

CODER AGENT INSTRUCTION:
Implement the training script for Phase 4 using this protocol. If you detect any potential data leakage (e.g., shuffling time-series), STOP and alert the user immediately. Do not proceed with an invalid model.

use GPU, force DirectML if possible, ALWAYS