# Full Validation Report and Correction Plan (v26-4 Full)

Date: 2026-04-26
Scope: End-to-end recovery plan for all Phase 4 models that have not passed standards.
Objective: Reach full PASS status before Phase 5 in master_plan.md.

## 1. Current Validation State (Ground Truth)

Registry snapshot indicates 18 total models, with these failed models:
- trend_follower: TCN_Trend_v1 (FAILED)
- discretionary_multimodal: Multimodal_Disc_v1, CNNChart_Disc_v1 (FAILED)
- scalping_microstructure: CNN_Scalper_v1, LinearAttn_Scalper_v1, GRU_Scalper_v1 (FAILED)
- market_making_rl: PPO_MM_v1, SAC_MM_v1, DQN_MM_v1 (FAILED)
- mean_reversion: GRN_MR_v1 (FAILED)

Models that are already PASSED and should be frozen as baseline:
- trend_follower: LSTM_Trend_v1, Transformer_Trend_v1
- discretionary_multimodal: ViT_Disc_v1
- mean_reversion: MLP_MR_v1, ResNet_MR_v1
- statistical_arbitrage: Autoencoder_StatArb_v1, GAT_StatArb_v1, LSTM_StatArb_v1

Important note on smoke logs:
- Current scalper smoke run used max_symbols=1 and max_epochs=1, so smoke output is health-check only, not KPI evidence.
- MM smoke run uses short episodes and very limited step budget, also not KPI evidence.

## 2. Data and ETL Findings (Root Causes)

## 2.1 Coverage and quality profile
- DataQualityGate with min_history_bars=50000 yields 34 accepted symbols and 24 rejected.
- Accepted set includes very high-liquidity assets and also low-liquidity/young assets with much shorter history.
- Median open_interest zero ratio across accepted symbols is about 0.99996, meaning open_interest is near-non-informative in current corpus.
- funding_rate has partial coverage and inconsistent informativeness across symbols.

## 2.2 Label and class balance findings (scalper)
- Under current rule (horizon=5, flat_threshold=0.0003), global label mix on sampled accepted symbols is approximately:
  - FLAT: 9.33%
  - LONG: 45.71%
  - SHORT: 44.96%
- Severe under-representation of FLAT causes unstable 3-class calibration and can collapse some architectures (especially CNN) toward directional-only behavior.

## 2.3 ETL bottlenecks blocking model quality
- Training universes are selected by row-count quality only, not by liquidity regime quality.
- No per-symbol liquidity gate in training inputs (trades/quote-volume thresholds), causing noisy low-liquidity symbols to dilute signal.
- Microstructure models are trained from OHLCV proxies only; there is no true L2 order-book depth or trade-tick event stream.
- Discretionary image pipeline rasterizes single-pixel OHLC traces and likely discards much of geometric structure.

## 3. Model-level Bottleneck Analysis

## 3.1 Trend: TCN_Trend_v1
Observed:
- Directional accuracy near random and Sharpe near zero.
Likely bottlenecks:
- Regression-on-return objective with tiny per-bar target magnitude is weak for TCN directional extraction.
- Feature space is narrow (5 features) and may underfit regime transitions for convolutional temporal filters.
- Current TCN config (channels=128, dropout=0.1) may be under-regularized/under-expressive for multi-symbol mixed regimes.

Corrective direction:
- Add classification head variant for direction (up/down) in TCN path, not return regression only.
- Introduce richer trend features: volatility regime, fracdiff_close, volume pressure, and momentum curvature terms.
- Tune dilation stack and receptive field sweep (dilations up to 8/16) with residual scaling.

## 3.2 Discretionary: Multimodal_Disc_v1, CNNChart_Disc_v1
Observed:
- Both fail with low test accuracy and strongly negative test Sharpe.
Likely bottlenecks:
- Rasterization is too sparse (binary pixel marks only), losing candle body/wick geometry and local texture.
- Tabular branch has very small feature set and no explicit sentiment/news or regime conditioning.
- Training objective does not account for class imbalance or confidence calibration.

Corrective direction:
- Replace sparse pixel plotting with dense OHLC image encoding (body fill, wick channel, volume stripe channel, volatility heat channel).
- Add focal loss or class-balanced CE for 3-class directional targets.
- Add confidence calibration and abstain rule for low-conviction outputs.
- Extend multimodal tabular branch with regime and cross-asset context features.

## 3.3 Scalper: CNN_Scalper_v1, LinearAttn_Scalper_v1, GRU_Scalper_v1
Observed:
- Smoke metrics show all invalid; LinearAttn best but still below KPI accuracy threshold.
- CNN severely collapses in smoke under current class distribution.
Likely bottlenecks:
- FLAT class under-representation creates unstable tri-class boundaries.
- Feature set upgraded but still OHLCV-proxy microstructure, no true order-flow event data.
- Sequence length fixed at 32 may be suboptimal for varying volatility regimes.
- GRU latency is high (inference_ms much larger than other scalper models).

Corrective direction:
- Use quantile-adaptive flat band per symbol-regime instead of fixed flat_threshold.
- Add label smoothing and per-class focal weighting sweep.
- Perform sequence-length sweep (32/48/64) and per-architecture tuning.
- Prioritize low-latency architectures for deployment (LinearAttn/CNN once stable), keep GRU as teacher candidate.

## 3.4 Market Making RL: PPO_MM_v1, SAC_MM_v1, DQN_MM_v1
Observed:
- Negative eval_mean_reward and near-max eval drawdown, all invalid.
Likely bottlenecks:
- Environment remains OHLCV-driven with simplified fill simulation; lacks true queue dynamics and depth response.
- Reward shaping now improved but still constrained by weak state realism.
- Price series stitching across symbols may create unrealistic transitions for RL state dynamics.
- Evaluation sample is still too shallow for robust policy ranking.

Corrective direction:
- Split training by symbol sessions (no stitched cross-symbol replay in a single episode).
- Add realistic fill model conditioned on volatility, spread proxy, and trade intensity.
- Introduce action constraints and inventory risk budget schedule.
- Expand eval protocol to multi-seed, multi-regime holdout with confidence intervals.

## 3.5 Mean Reversion: GRN_MR_v1
Observed:
- GRN_MR_v1 underperforms while MLP/ResNet pass.
Likely bottlenecks:
- GRN hidden size/depth may be too small for current feature complexity.
- Stem uses ReLU while peer winners use smoother nonlinear stack patterns.
- Feature set may not expose enough interaction terms for gate utility.

Corrective direction:
- Increase GRN capacity (hidden 256, depth 6), switch stem activation to GELU/SiLU.
- Add interaction features (vwap_dev x rsi_div, bb_distance x volatility).
- Apply stronger regularization search and learning-rate schedule sweep.

## 4. What Needs More Training vs More Data

## 4.1 Needs more training/tuning (no new external data strictly required)
- TCN_Trend_v1
- GRN_MR_v1
- Multimodal_Disc_v1 and CNNChart_Disc_v1 (can improve with better internal image encoding and loss strategy)
- Scalper trio can improve materially with better labeling/regime balancing and full 50-epoch sweeps.

## 4.2 Needs more data (critical for durable PASS)
Scalper and MM need microstructure-grade data beyond OHLCV to consistently pass production standards:
- Required new dataset types:
  - Tick trade stream (price, size, side/aggressor, timestamp at sub-second granularity)
  - L2 order book snapshots/deltas (top 10-20 levels)
  - Best bid/ask event history (for spread and queue dynamics)
- Minimum historical depth:
  - 90 days minimum for initial model stabilization
  - 180-365 days preferred for robust regime coverage
- Asset priority for collection (Tier-1 liquidity first):
  - BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, DOGEUSDT
- Tier-2 expansion after stable pass on Tier-1:
  - ADAUSDT, TRXUSDT, RUNEUSDT, AAVEUSDT, PAXGUSDT, ENJUSDT

## 5. Optimization and Benchmarking Protocol (Mandatory)

For each failed model family, enforce this loop:
1. Profile
- Measure p50/p95 latency, throughput, peak memory, and epoch time per model.
- Record backend behavior separately for directml and cpu fallback.

2. Baseline
- Freeze current baseline metrics from registry before each change wave.

3. Optimize incrementally
- Apply one major change group at a time (data labels, architecture, scheduler, loss, reward model).
- Track delta in accuracy/sharpe/profit_factor/max_drawdown and latency.

4. Validate on production-representative holdout
- Multi-symbol and multi-regime holdout.
- Multi-seed (>=3 seeds) to reject unstable improvements.

5. Gate
- Promote only if pass criteria and stability criteria are both satisfied.

## 6. Recovery Roadmap (Execution Plan)

## Phase A: ETL and Dataset Hardening (Blocker for robust PASS)
A1. Liquidity-aware symbol gating
- Add gates using median trades and quote_volume to create:
  - Core training universe (high liquidity)
  - Secondary universe (optional augmentation)

A2. Regime-balanced sampling
- Build per-symbol regime buckets (quiet/normal/chaotic by realized volatility).
- Enforce balanced sampling across regimes in training loaders.

A3. Label redesign for scalper
- Replace fixed flat_threshold with adaptive threshold by symbol volatility quantiles.
- Recompute class distribution report each run; enforce flat class minimum target range 20-30%.

A4. New market data ingestion (parallel workstream)
- Implement tick and L2 ingestion pipeline with idempotent partitioned storage.
- Build feature factory for OFI, queue imbalance, microprice, spread dynamics, and short-horizon toxicity features.

## Phase B: Model Recovery by Family
B1. Trend (TCN)
- Add directional classification objective and compare against regression objective.
- Sweep receptive field and channel settings.
- Target: test directional accuracy > 0.52 and test sharpe > 1.0.

B2. Discretionary (Multimodal, CNNChart)
- Upgrade chart encoding to dense candle geometry representation.
- Add class-balanced focal loss and confidence calibration.
- Add simple text/sentiment proxy stream if available.
- Target: accuracy > 0.52, sharpe > 1.0, drawdown control.

B3. Scalper (CNN, LinearAttn, GRU)
- Run full sweep with adaptive labels, class-aware loss, sequence-length search, and CLR.
- Keep architecture-specific latency targets:
  - CNN/LinearAttn inference p95 < 5 ms
  - GRU used only if materially superior in alpha and can be optimized.
- Target: directional accuracy > 0.52 and test sharpe > 1.0.

B4. Market Making (PPO, SAC, DQN)
- Refactor environment to symbol-session episodes and improved fill modeling.
- Keep asymmetric reward but calibrate inventory penalty by volatility regime.
- Evaluate with multi-seed robust statistics and confidence intervals.
- Target: positive eval PnL (or eval_mean_reward > 0) and eval_max_drawdown < 0.15.

B5. Mean Reversion (GRN)
- Increase GRN capacity and feature interactions.
- Tune optimizer/scheduler and regularization against MLP/ResNet baseline.
- Target: match or exceed MLP baseline sharpe with drawdown under threshold.

## Phase C: Production Optimization
C1. Inference optimization path
- Profile and optimize each deployable model:
  - torch.compile where stable
  - dynamic quantization for CPU paths where latency-critical
  - mixed precision where backend supports deterministic behavior
- Keep an accuracy degradation budget (max tolerated drop <= 1% absolute on core KPI).

C2. Deployment candidate selection
- For each archetype, keep only best Pareto candidates (accuracy/sharpe vs latency/cost).
- Archive non-Pareto models as research artifacts.

## 7. Governance, Stop Rules, and Promotion Criteria

Hard stop rules:
- Do not move to Phase 5 while any required model family remains below pass gate.
- Do not accept a model based on smoke run metrics.
- Do not promote gains from one seed only.

Promotion checklist per model:
- KPI pass on holdout
- Stability pass across >=3 seeds
- No leakage violations
- Latency and resource envelope documented
- Registry and summary updated

## 8. Immediate Next Sprint (Concrete Task List)

Sprint 1 (highest ROI):
1. Implement liquidity-aware symbol universe and adaptive flat-labeling for scalper.
2. Run full scalper sweep (50 epochs, patience 10) on core liquidity universe.
3. Refactor MM episode generation to avoid stitched cross-symbol transitions.
4. Run full MM sweep with expanded eval episodes and multi-seed checks.
5. Upgrade discretionary rasterization and retrain failing discretionary models.
6. Increase GRN_MR_v1 capacity and retrain.
7. Retrain TCN with directional classification head.

Sprint 2 (data expansion):
1. Build tick + L2 ingestion for Tier-1 assets.
2. Integrate true OFI/queue/spread features into scalper and MM pipelines.
3. Re-benchmark all failed families on new data.

## 9. Deliverables Required Before Phase 5

Must be delivered and reviewed:
- Updated model_registry.json with all required models passing gates.
- Updated model_performance_summary.md with full-sweep results only.
- Data integrity and class-balance report for each retrain wave.
- Latency/memory benchmark report with p50/p95/p99 and throughput.
- Final sign-off report confirming all pass criteria met.

---
This file is the main recovery prompt and execution guide. Keep iterating on this plan until every required model is PASS with stable out-of-sample behavior.
