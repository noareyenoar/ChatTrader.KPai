# Data Acquisition and Transformation Requirements

This document defines the minimum data and feature standards required for Phase 4 model correction and validation.

## 1) Required Raw Fields per Symbol

- timestamp
- open
- high
- low
- close
- volume
- quote_volume
- taker_buy_base
- taker_buy_quote

## 2) Dataset Integrity Rules

- Data must be strictly time-ordered and deduplicated by timestamp.
- No forward filling across missing bars; missing segments must remain explicit gaps.
- Minimum history per accepted symbol must satisfy `min_history_bars` in config.
- All train/validation/test splits must use chronological order with purge gaps.

## 3) Required Derived Features (Scalper)

- ofi_proxy
- microprice_dev
- spread_pct
- log_return
- vol_imbalance
- fracdiff_close_d04
- fracdiff_volume_d04
- buy_sell_pressure
- price_velocity_5
- price_velocity_10
- price_velocity_15
- volatility_z_32
- vol_regime_code

## 4) Transformation Definitions

- `fracdiff_close_d04`, `fracdiff_volume_d04`: fractional differencing with `d=0.4`.
- `buy_sell_pressure`: `taker_buy_base / (taker_buy_quote + 1e-8)`.
- `price_velocity_k`: `(close_t - close_{t-k}) / k` for k in {5, 10, 15}.
- `volatility_z_32`: z-score of 32-bar rolling realized volatility.
- `vol_regime_code`: 3-level regime from realized-volatility terciles.

## 5) Validation and KPI Readiness

- Features must be fit-scaled using train split statistics only.
- Labels must be generated after all features and purge-gap split are established.
- Registry KPI checks for failed-model reruns should use holdout/eval metrics, not only train metrics.

## 6) Operational Guidance

- Run smoke configs first to validate pipeline and registry schema updates.
- Promote to full sweep only after smoke run completes without integrity or schema errors.
- Do not move to Phase 5 until failed model KPIs in `model_registry.json` are met.
