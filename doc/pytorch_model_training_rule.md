# 🎯 Supreme Training Protocol for ChatTrader.KPai (Revised)

This document mandates a zero-tolerance policy for data leakage and defines strict hardware, statistical, and backtesting standards for all 18 models. A model is only production-ready if it survives both the "Iron Wall" statistical test and the "Integrated Backtest Engine" simulation.

---

## 1. THE "IRON WALL" DATA SPLITTING (NO LEAKAGE POLICY)
To ensure 100% foolproof training, the Agent must strictly follow these temporal constraints. Random sampling across the entire timeline is FORBIDDEN.

### 1.1 Chronological Split (70% / 15% / 15%)
* **Training (70%):** Oldest data. Used for gradient descent.
* **Validation (15%):** Middle data. Used for hyperparameter tuning and Early Stopping.
* **Testing (15%):** Most recent data (Out-of-Sample). Locked until the model is finalized.

### 1.2 Leakage Prevention Checklist
* **No Lookahead:** Features at time $t$ must only contain information from $t-1$ or earlier.
* **Gap Buffer:** Implement a "Purge Gap" between Train/Val and Val/Test sets equal to the maximum prediction horizon (e.g., if predicting $t+20$, drop 20 bars between sets) to prevent information bleeding.
* **Scaling Leakage:** Fit scalers (StandardScaler/MinMax) ONLY on the Training set. Use the fitted parameters to Transform Val and Test sets.

---

## 2. GPU EXECUTION & RESOURCE MANAGEMENT
Every model MUST be optimized for CUDA/DirectML execution.

* **Device Handling:** Explicitly use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Force `torch_directml` if on AMD hardware.
* **Memory Efficiency:** Use `torch.cuda.amp.autocast()` for mixed-precision training.
* **Cleanup:** After training each model, execute:
    ```python
    model.cpu()
    del model, optimizer
    torch.cuda.empty_cache()
    ```

---

## 3. ARCHETYPE-SPECIFIC VALIDATION STANDARDS
| Archetype | Architecture | Validation Focus | Success Threshold |
| :--- | :--- | :--- | :--- |
| **Trend Follower** | LSTM / Transformer | Directional Consistency | Directional Accuracy > 55% |
| **Mean Reversion** | MLP / ResNet | Catching Extremes | Precision on Reversal > 60% |
| **Scalper** | CNN / Transformer | Latency & Tick Flow | Inference Time < 10ms |
| **Stat Arb** | Autoencoder / GNN | Spread Stability | Reconstruction Error < 0.05 |
| **Discretionary** | ViT / Multimodal | Pattern Alignment | F1-Score > 0.65 |
| **Market Maker** | RL (PPO/SAC) | Inventory Risk | Reward Stability (StdDev < 0.2) |

---

## 4. REAL-WORLD VALIDITY LOGIC (NET PERFORMANCE GATES)
A model is **INVALID** if it fails any of these gates on the 15% Test (OOS) data **after accounting for transaction costs**:

### 4.1 Transaction Cost Simulation (CRITICAL)
All performance metrics must be calculated using a realistic trading environment:
* **Commission:** Deduct **0.04%** per trade (Standard Binance Futures rate).
* **Slippage:** Deduct a minimum of **1-2 ticks** per trade to account for liquidity friction.

### 4.2 Production Readiness Criteria (The "Four Gates")
1.  **Net Sharpe Ratio:** > 1.2 (Calculated after Commission & Slippage).
2.  **Profit Factor:** > 1.5 in Test (OOS) sample.
3.  **Maximum Drawdown:** < 20% in Test sample.
4.  **Consistency:** Performance decay (Test vs Val Sharpe) must be < 50%.

---

## 5. INTEGRATED BACKTEST ENGINE
Immediately after passing the Testing Split, the Coder Agent must execute a **Vectorized Backtest**:

* **Logic:** Implement a custom backtester using NumPy/Pandas to simulate every trade signaled by the model in the Test Set.
* **Metric:** Calculate **Net PnL** = Gross PnL - Fees - Slippage.
* **Alignment Check:** The Equity Curve must be consistent with the Neural Network metrics. If Directional Accuracy is high but Net PnL is negative, the Coder Agent must trigger a **Data Leakage Audit** in the Feature Engineering module.

---

## 6. STABILITY & ROBUSTNESS TESTING

### 6.1 Walk-Forward Analysis (WFA)
To prevent **Regime Failure**, the agent must not rely on a single 70/15/15 split.
* **Execution:** Implement a rolling Walk-Forward validation to ensure the model remains stable across different market regimes (Bull, Bear, Sideways).
* **Requirement:** The model must maintain a positive Net PnL in at least 80% of the walk-forward windows.

### 6.2 Stress Test (Monte Carlo Simulation)
* **Execution:** Shuffle the sequence of trades from the backtest results 1,000 times.
* **Requirement:** In the 95th percentile worst-case sequence, the **Maximum Drawdown must remain < 20%**. If a specific sequence of trades leads to a total blowout, the model is rejected.

---

## 7. MODEL CATALOG & REPRODUCIBILITY
* **Registry:** Output `model_registry.json` containing Architecture Name, Weights Path, Design Premise, and `is_valid: true/false` based on Section 4 & 6.
* **Global Seed:** 42 (Must be set for torch, numpy, and random).
* **Logging:** Log Train_Loss, Val_Loss, Net_Sharpe, and MDD to TensorBoard.

---
**CODER AGENT INSTRUCTION:**
Implement the Phase 4 training and validation pipeline following this protocol. Do not proceed to deployment if a model fails the Monte Carlo Stress Test or the Net Sharpe gate.