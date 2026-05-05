# Validation Report & Correction Plan (v26-4)

## 1. Executive Summary
- **Overall Status:** Partial Success.
- **Strong Performers:** `statistical_arbitrage` (Sharpe > 40), `mean_reversion` (Sharpe > 20).
- **Critical Failures:** `market_making_rl` (Negative PnL, Total Collapse), `scalping_microstructure` (CNN/GRU Accuracy ~48%).
- **Primary Bottleneck:** Signal-to-Noise ratio in low-timeframe features and potential Reward Mis-specification in RL environments.

## 2. Technical Audit & Gap Analysis

### 2.1 The "Market Making" RL Disaster
- **Observation:** PPO/SAC/DQN show negative PnL (up to -129.32).
- **Hypothesis:** 1. **State Space:** Using 1m bars for Market Making is insufficient. The agent lacks visibility into the Order Book (Bid/Ask depth).
    2. **Reward Function:** The penalty for inventory risk might be too low, or the reward for capturing the spread is being drowned out by slippage/volatility.
- **Correction:** - Implement **Asymmetric Power Law** for rewards.
    - Integrate **Order Flow Imbalance (OFI)** as a state feature.
    - Review `preferred_backend: directml` stability for RL kernels.

### 2.2 Scalping Model Stagnation (CNN/GRU)
- **Observation:** Accuracy < 50% (worse than random).
- **Hypothesis:** 1. **Vanishing Gradients:** Deep CNN/GRU layers on DirectML might be hitting numerical instability.
    2. **Stationarity:** Log-returns alone might not be enough; we need **Fractional Differentiation (d=0.4)** to preserve memory.
- **Correction:** - Transition to **Leaky ReLU** activations and **Gradient Clipping (max_norm=1.0)**.
    - Add **Volatility-Adjusted Z-Score** features.

## 3. Data & Feature Factory Upgrade (Immediate Requirements)
To pass the standards, the Coder Agent must implement:
1. **Fractional Differentiation:** Apply to `close` and `volume` to maximize stationarity without losing memory.
2. **Microstructure Features:**
    - `buy_sell_pressure`: Taker buy base / Taker buy quote ratio.
    - `price_velocity`: Rate of change over short windows (5, 10, 15 bars).
3. **Volatility Regimes:** Categorize market states (Quiet, Normal, Chaotic) and pass as an Embedding to the models.

## 4. Correction Roadmap

### Step 1: Feature Injection (Data Layer)
- Update `Feature Factory` to include the features mentioned in Section 3.
- Re-run Data Integrity Audit to ensure no NaNs or Inf values.

### Step 2: RL Environment Refactoring
- Adjust RL environment to include a "Warm-up" phase.
- Simplify Reward: Focus only on `PnL + Realized Spread - (0.1 * Inventory_Skew)`.

### Step 3: Training Sweep v2
- Re-train all `FAILED` models with `max_epochs: 50` and `patience: 10`.
- Implement **Cyclic Learning Rate (CLR)** to escape local minima.

## 5. Success Criteria
A model is promoted to "Validated" only if:
1. **Trend/Scalp:** Directional Accuracy > 52% and Test Sharpe > 1.0.
2. **Mean Reversion:** Test Sharpe > 1.5.
3. **Market Making:** Positive PnL in Test Set and Max Drawdown < 15%.

---
**Note:** This document serves as the master prompt for the Coder Agent. Do not deviate from these mathematical constraints.
