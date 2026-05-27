# 🎯 Supreme Training Protocol for ChatTrader.KPai v2.0 (Regime-Resistant Edition)

This document overrides previous training rules. The objective is to eliminate "Val->Test Distribution Shift" and extract true Alpha in highly correlated, noisy Crypto markets. Models must survive multi-regime walk-forward testing, not just a single static split.

---

## 1. CRYPTO-SPECIFIC ALPHA EXTRACTION (BETA NEUTRALIZATION)
Crypto assets are heavily heavily correlated with Bitcoin (BTC). A model learning general market direction is not finding Alpha; it is finding Beta.

* **BTC Volatility & Return Neutralization:** For any Altcoin dataset, the Feature Factory MUST compute the rolling Beta ($\beta$) relative to BTC.
* **Residual Features:** Feed the model "Residual Returns" ($Return_{asset} - \beta \cdot Return_{BTC}$) and "Relative Volatility" instead of raw returns. The model must learn what makes the asset move *independently* of the Bitcoin cycle.

---

## 2. THE LUPI FRAMEWORK (ORACLE TEACHER)
Regular supervised learning fails due to the extreme Noise-to-Signal ratio in Crypto. We will implement Learning Using Privileged Information (LUPI) for complex archetypes (e.g., APV-PLN, Trend).

* **The Oracle (Teacher):** During training, this model receives past features $X_{t-k:t}$ AND future structural data $Y_{t:t+h}$ (e.g., actual future wave trajectories, smoothed future trends). It outputs a "Soft Target" probability distribution.
* **The Student (Production Model):** Receives ONLY past features $X_{t-k:t}$. 
* **Knowledge Distillation Loss:** The total loss must combine standard Cross-Entropy (against Ground Truth) and Kullback-Leibler Divergence ($Loss_{KL}$) to match the Oracle's distribution.
* **IRON WALL RULE:** The Oracle Teacher MUST BE COMPLETELY DISABLED during Validation, Testing, and Live Inference.

---

## 3. REGIME-AWARE DATA SPLITTING & WALK-FORWARD
The static 70/15/15 chronological split caused catastrophic failures because the Validation regime (e.g., Bull) differed from the Test regime (e.g., Chop).

* **Multi-Regime Validation:** The Validation set must not be a single continuous block. It must be sampled across stratified market regimes (Trending Up, Trending Down, High Volatility Sideways, Low Volatility Sideways) using a Regime Detector (e.g., HMM).
* **Purged Walk-Forward Testing (WFA):** For final Out-Of-Sample (OOS) testing, the agent must implement a Rolling Walk-Forward evaluation with a strict "Purge Gap" equal to the prediction horizon ($h$) between training windows and testing windows.

---

## 4. PROBABILISTIC TARGETS & TRIPLE BARRIER
Stop predicting point-estimates (e.g., exactly what the price will be). 

* **Span of Probability:** Output layers should predict the probability across discrete Bins (e.g., 51 bins ranging from strong negative to strong positive).
* **Triple Barrier Method:** For Scalpers and Mean Reversion, label data based on which barrier is hit first: Take Profit (Top), Stop Loss (Bottom), or Time Expiration (Vertical). This teaches the model *timing* and *risk-reward*, not just direction.

---

## 5. HARDWARE & EFFICIENCY PROTOCOLS
* **Architecture Priority:** Favor Temporal Convolutional Networks (TCN) with dilated convolutions or Transformers over LSTMs. RNNs (LSTM/GRU) cause severe sequential bottlenecks on DirectML.
* **VRAM Management:** Enforce `torch.cuda.empty_cache()` and `gc.collect()` at the end of every epoch.
* **Optimization Hygiene:** Use `LeakyReLU` or `GELU` (avoid standard ReLU dead neurons). Enforce Gradient Clipping (`max_norm=1.0`). Use `CosineAnnealingLR`.

---

## 6. PRODUCTION READINESS GATES (OOS ONLY)
A model is only valid if it passes these metrics on the **Walk-Forward Test Set**, calculated after 0.04% commission and 1-tick slippage:
1. **Net Sharpe Ratio:** > 1.0 (Positive risk-adjusted returns).
2. **Profit Factor:** > 1.3
3. **Maximum Drawdown:** < 20%
4. **Divergence Limit:** The gap between Validation Sharpe and Test Sharpe must not exceed 2.0. If it does, the model is overfitting and automatically **FAILED**.

**CODER AGENT INSTRUCTION:** Read this protocol. Update `trend_training.py`, `features.py`, and `evaluate_all_checkpoints.py` to enforce Beta Neutralization, LUPI, and Walk-Forward Validation. Do not resume training until the pipeline logic reflects v2.0.