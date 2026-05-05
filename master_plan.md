# ChatTrader.KPai Master Plan (Phase 2)

Date: 2026-04-25  
Scope: Mathematical and systems roadmap for Phases 3-6 with strict anti-leakage and CUDA-first execution.

## 1. Architectural Intent

ChatTrader.KPai is designed as a deterministic pipeline:

1. Data Ingestion (historical Binance parquet)
2. Feature Factory (archetype-specific tensors)
3. Model Training (18 models across 6 archetypes)
4. Multi-Agent Debate (6 traders + 1 orchestrator)
5. Execution/Backtest and Risk Evaluation

Core constraints:
- No lookahead leakage in feature construction.
- Strict chronological split with purge gap.
- CUDA-first compute path for model training and high-volume transforms.
- Blunt rejection of low-quality symbols and unstable model outputs.

## 2. Global Mathematical Definitions

Let raw OHLCV at bar index t be:

\[
X_t = (O_t, H_t, L_t, C_t, V_t, Q_t, N_t, TB_t, TQ_t, OI_t, FR_t)
\]

where:
- \(Q_t\): quote volume
- \(N_t\): trade count
- \(TB_t\): taker buy base
- \(TQ_t\): taker buy quote
- \(OI_t\): open interest
- \(FR_t\): funding rate

Return and scaling primitives:

\[
r_t = \log\left(\frac{C_t}{C_{t-1}}\right), \quad
z_t^{(w)} = \frac{x_t - \mu_t^{(w)}}{\sigma_t^{(w)} + \epsilon}
\]

with rolling window \(w\), numerical stabilizer \(\epsilon=10^{-8}\), and \((\mu_t^{(w)},\sigma_t^{(w)})\) estimated only from historical bars up to \(t\).

## 3. Archetype Feature Foundations

### 3.1 Trend Follower (LSTM/Transformer/TCN)
Input tensor shape:
\[
\mathbf{T} \in \mathbb{R}^{B \times L \times F}
\]
where \(L\in[50,200]\).

Features:
- Log return: \(r_t\)
- EMA spread: \(\Delta\text{EMA}_t = \text{EMA}_{\alpha_1}(C_t)-\text{EMA}_{\alpha_2}(C_t)\)
- ATR proxy:
\[
\text{TR}_t = \max(H_t-L_t, |H_t-C_{t-1}|, |L_t-C_{t-1}|),\;
\text{ATR}_t = \text{EMA}(\text{TR}_t)
\]
- Price slope over \(k\): \(s_t^{(k)}=(C_t-C_{t-k})/k\)

Target example: directional class \(y_t=\mathbb{1}[C_{t+h}>C_t]\).

### 3.2 Mean Reversion (MLP/ResNet)
Input tensor shape:
\[
\mathbf{M}\in\mathbb{R}^{B\times F}
\]

Features:
- Price-to-VWAP deviation: \(d_t=(C_t-\text{VWAP}_t)/\text{VWAP}_t\)
- Bollinger distance:
\[
\text{BBDist}_t = \frac{C_t-\mu_t^{(w)}}{k\sigma_t^{(w)}+\epsilon}
\]
- RSI divergence: \(\text{RSI}_t - \text{RSI}_{t-k}\)
- Reversion z-score \(z_t^{(w)}\)

Binary target: reversal in horizon \(h\).

### 3.3 Scalper / Microstructure (CNN/Linear-Attention)
LOB tensor shape:
\[
\mathbf{S}\in\mathbb{R}^{B\times D\times F}\;\text{or}\;\mathbb{R}^{B\times C\times H\times W}
\]

Core microstructure features:
- Log-volume compression: \(\log(1+v)\)
- Spread: \(\text{ask}_1-\text{bid}_1\)
- Order-flow imbalance:
\[
\text{OFI}_t = \sum_{i=1}^{n}(\Delta q_{i,t}^{bid}-\Delta q_{i,t}^{ask})
\]
- Microprice:
\[
\text{MP}_t = \frac{a_t q_t^{bid} + b_t q_t^{ask}}{q_t^{bid}+q_t^{ask}+\epsilon}
\]

Objective: low-latency directional or fill-probability classification.

### 3.4 Statistical Arbitrage (Autoencoder/GNN)
Input tensor:
\[
\mathbf{A}\in\mathbb{R}^{B\times N_{assets}\times L}
\]

Fractional differentiation (order \(d\in(0,1)\)):
\[
(1-B)^d x_t = \sum_{k=0}^{\infty} w_k x_{t-k},\quad
w_0=1,\; w_k = -w_{k-1}\frac{d-k+1}{k}
\]
Truncate where \(|w_k|<\tau\).

Spread and latent reconstruction:
- Pair spread \(s_t=x_t^{(i)}-\beta x_t^{(j)}\)
- Autoencoder loss: \(\mathcal{L}_{AE}=\|x_t-\hat{x}_t\|_2^2\)
- GNN graph over assets with adjacency \(\mathbf{W}\) from rolling correlation/cointegration statistics.

### 3.5 Discretionary (ViT + Text Fusion)
Inputs:
- Chart image embedding \(e_t^{img}\in\mathbb{R}^{d_1}\)
- Text/sentiment embedding \(e_t^{txt}\in\mathbb{R}^{d_2}\)

Fusion:
\[
e_t^{fusion} = [e_t^{img};e_t^{txt}]\in\mathbb{R}^{d_1+d_2}
\]
Classifier:
\[
\hat{y}_t=\text{softmax}(W e_t^{fusion}+b)
\]

### 3.6 Market Maker (RL: PPO/SAC)
State space:
\[
\mathcal{S} = \{s_t=(I_t,\;\Delta_t,\;\sigma_t,\;\lambda_t,\;q_t^{bid},\;q_t^{ask})\}
\]
with inventory \(I_t\in[-1,1]\), spread \(\Delta_t\), volatility \(\sigma_t\), arrival intensity \(\lambda_t\).

Action space example:
\[
a_t=(\delta_t^{bid},\delta_t^{ask},\nu_t)\]
where \(\delta\) are quote offsets and \(\nu\) is size scale.

Reward:
\[
r_t = \Delta \text{PnL}_t - \lambda_I |I_t| - \lambda_{dd}\,\max(0, \text{DD}_t-\text{DD}_{max})
\]
Policy objective (PPO form):
\[
\max_\theta\;\mathbb{E}_t\left[\min\left(\rho_t(\theta)\hat{A}_t,\;\text{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
\]

## 4. Data Leakage and Split Protocol (Iron Wall)

Chronological split with purge gap g bars:
- Train: oldest 70%
- Validation: next 15%
- Test: latest 15%
- Purge: remove g bars between Train/Val and Val/Test

Formal index partition for T samples:
- \(n_{tr}=\lfloor0.70T\rfloor\)
- \(n_{va}=\lfloor0.15T\rfloor\)
- \(n_{te}=T-n_{tr}-n_{va}\)

Then apply purge shifts so that validation starts after \(n_{tr}+g\), and test starts after validation end + \(g\).

Critical scaler rule:
- Fit scaler on train only: \(\theta_{scaler}=\text{fit}(X_{train})\)
- Transform val/test only with \(\theta_{scaler}\)

## 5. Quality Gates and Rejection Logic

A symbol is excluded if any condition holds:
1. Manifest status is FAIL.
2. Missing-bar ratio > 5%.
3. History length below model-specific minimum bars.

Missing-bar ratio:
\[
\text{MBR} = 1 - \frac{N_{observed}}{N_{expected}}
\]

## 6. CUDA-First and Memory Lifecycle

Training and heavy batch transforms default to CUDA when available:
- `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Mixed precision: `torch.amp.autocast("cuda")`

VRAM cleanup protocol after model/batch lifecycle:
1. Move tensors/models to CPU when done.
2. `del` large references.
3. `torch.cuda.empty_cache()`.

## 7. Phase 3-6 TODO Roadmap

### Phase 3: Data Engineering Pipeline
- [ ] Build manifest-aware quality gate.
- [ ] Enforce missing-bar threshold rejection (>5%).
- [ ] Implement Iron Wall splitter with purge gap.
- [ ] Implement vectorized feature transforms for all archetypes.
- [ ] Create feature registry and data integrity report generation.
- [ ] Add distribution visualization exports for key features.
- [ ] Add unit tests for leakage and split correctness.

### Phase 4: Quant Core (18 Models)
- [ ] Implement 3 model classes per archetype (18 total).
- [ ] Add YAML-driven model/training configs.
- [ ] Integrate early stopping (patience=10) and checkpointing.
- [ ] Log train/val/lr to TensorBoard each epoch.
- [ ] Add random-noise sanity propagation test per architecture.
- [ ] Produce model_registry.json with validation audit flags.

### Phase 5: Multi-Agent Debate + Ollama
- [ ] Implement TraderAgent with model inference hooks.
- [ ] Implement evidence packet (confidence, alignment, historical score).
- [ ] Implement orchestrator debate loop with rebuttal and final sizing.
- [ ] Add retry/self-correct logic for malformed Ollama outputs.
- [ ] Enable parallel LLM calls to control latency.

### Phase 6: End-to-End Simulation
- [ ] Build simulation runner: data -> features -> inference -> debate -> trade.
- [ ] Integrate slippage, fees, and latency in backtest loop.
- [ ] Compute Sharpe, profit factor, max drawdown, regime breakdown.
- [ ] Validate OOS consistency rules and invalidate failing models.
- [ ] Produce full run report and reproducibility manifest.

## 8. Acceptance Criteria for Current Milestone

This milestone (Phase 2) is accepted when:
- Mathematical definitions for all six archetypes are documented.
- RL state and reward constraints are explicit.
- Fractional differentiation is explicitly defined.
- Phase 3-6 tasks are decomposed into actionable checklists.
