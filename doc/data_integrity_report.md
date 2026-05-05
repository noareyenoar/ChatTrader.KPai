# Data Integrity Report (Phase 3 Feature Factory)

Date: 2026-04-25

## 1) Quality Gate Summary

- Total symbols evaluated: 58
- Accepted symbols: 34
- Rejected symbols: 24
- Missing bar threshold: 5.00%
- Minimum history bars: 50000

### Accepted Symbols

| symbol     |   rows |   expected_rows |   missing_ratio | decision   |
|:-----------|-------:|----------------:|----------------:|:-----------|
| AAVEUSDT   | 580956 |          581243 |        0.000494 | ACCEPT     |
| ADAUSDT    | 842711 |          843887 |        0.001394 | ACCEPT     |
| APEUSDT    | 431905 |          431921 |        3.7e-05  | ACCEPT     |
| API3USDT   | 447793 |          447809 |        3.6e-05  | ACCEPT     |
| AXSUSDT    | 575076 |          575363 |        0.000499 | ACCEPT     |
| BARDUSDT   |  63035 |           63035 |        0        | ACCEPT     |
| BIOUSDT    | 137351 |          137351 |        0        | ACCEPT     |
| BNBUSDT    | 888913 |          890545 |        0.001833 | ACCEPT     |
| DOGEUSDT   | 715305 |          715919 |        0.000858 | ACCEPT     |
| DUSDT      | 135647 |          135647 |        0        | ACCEPT     |
| DYDXUSDT   | 486463 |          486503 |        8.2e-05  | ACCEPT     |
| ENJUSDT    | 737733 |          738479 |        0.00101  | ACCEPT     |
| ETHUSDT    | 912156 |          913871 |        0.001877 | ACCEPT     |
| GIGGLEUSDT |  52439 |           52439 |        0        | ACCEPT     |
| NEIROUSDT  | 168743 |          168743 |        0        | ACCEPT     |
| ORDIUSDT   | 259151 |          259151 |        0        | ACCEPT     |
| PAXGUSDT   | 594672 |          594959 |        0.000482 | ACCEPT     |
| PENGUUSDT  | 142199 |          142199 |        0        | ACCEPT     |
| PEPEUSDT   | 312647 |          312647 |        0        | ACCEPT     |
| RUNEUSDT   | 592718 |          593005 |        0.000484 | ACCEPT     |

### Rejected Symbols

| symbol       | manifest_status   |   rows |   expected_rows |   missing_ratio | decision   | reason               |
|:-------------|:------------------|-------:|----------------:|----------------:|:-----------|:---------------------|
| 1000SATSUSDT | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| ARKMUSDT     | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| ARUSDT       | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| AXLUSDT      | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| BLURUSDT     | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| BONKUSDT     | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| BTCUSD       | PASS              |  40895 |           40895 |               0 | REJECT     | INSUFFICIENT_HISTORY |
| CHIPUSDT     | PASS              |   1085 |            1085 |               0 | REJECT     | INSUFFICIENT_HISTORY |
| CHZUSDT      | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| CTSIUSDT     | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| ENAUSDT      | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| FILUSDT      | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| INJUSDT      | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| LTCUSDT      | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| MBOXUSDT     | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| NEARUSDT     | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| RLUSDUSDT    | PASS              |  26783 |           26783 |               0 | REJECT     | INSUFFICIENT_HISTORY |
| UNIUSDT      | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| UUSDT        | PASS              |  29375 |           29375 |               0 | REJECT     | INSUFFICIENT_HISTORY |
| WIFUSDT      | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| WLDUSDT      | FAIL              |  17280 |           17280 |               0 | REJECT     | MANIFEST_FAIL        |
| XAUTUSDT     | PASS              |   8567 |            8567 |               0 | REJECT     | INSUFFICIENT_HISTORY |
| ZAMAUSDT     | PASS              |  23555 |           23555 |               0 | REJECT     | INSUFFICIENT_HISTORY |
| 币安人生USDT | PASS              |  31031 |           31031 |               0 | REJECT     | INSUFFICIENT_HISTORY |

## 2) Iron Wall Split Validation

Rules enforced:
- Chronological ordering only (no shuffle)
- 70/15/15 split
- Purge gap = 20 bars between train/val and val/test

| symbol   |   train_rows |   val_rows |   test_rows |   purge_gap_bars | train_end                 | val_start                 | val_end                   | test_start                |
|:---------|-------------:|-----------:|------------:|-----------------:|:--------------------------|:--------------------------|:--------------------------|:--------------------------|
| AAVEUSDT |       406669 |      87123 |       87124 |               20 | 2024-08-28 03:55:00+00:00 | 2024-08-28 05:40:00+00:00 | 2025-06-26 17:50:00+00:00 | 2025-06-26 19:35:00+00:00 |
| ADAUSDT  |       589897 |     126386 |      126388 |               20 | 2023-11-29 12:00:00+00:00 | 2023-11-29 13:45:00+00:00 | 2025-02-10 09:50:00+00:00 | 2025-02-10 11:35:00+00:00 |
| APEUSDT  |       302333 |      64765 |       64767 |               20 | 2025-01-30 10:10:00+00:00 | 2025-01-30 11:55:00+00:00 | 2025-09-12 08:55:00+00:00 | 2025-09-12 10:40:00+00:00 |
| API3USDT |       313455 |      67148 |       67150 |               20 | 2025-01-13 21:00:00+00:00 | 2025-01-13 22:45:00+00:00 | 2025-09-04 02:20:00+00:00 | 2025-09-04 04:05:00+00:00 |
| AXSUSDT  |       402553 |      86241 |       86242 |               20 | 2024-09-03 06:55:00+00:00 | 2024-09-03 08:40:00+00:00 | 2025-06-29 19:20:00+00:00 | 2025-06-29 21:05:00+00:00 |
| BARDUSDT |        44124 |       9435 |        9436 |               20 | 2026-02-18 15:55:00+00:00 | 2026-02-18 17:40:00+00:00 | 2026-03-23 11:50:00+00:00 | 2026-03-23 13:35:00+00:00 |
| BIOUSDT  |        96145 |      20582 |       20584 |               20 | 2025-12-03 06:00:00+00:00 | 2025-12-03 07:45:00+00:00 | 2026-02-12 18:50:00+00:00 | 2026-02-12 20:35:00+00:00 |
| BNBUSDT  |       622239 |     133316 |      133318 |               20 | 2023-10-12 09:00:00+00:00 | 2023-10-12 10:45:00+00:00 | 2025-01-17 08:20:00+00:00 | 2025-01-17 10:05:00+00:00 |

## 3) Feature Registry

Generated vectorized features:
- log_return
- zscore_close_64
- ema_spread
- atr_14
- price_slope_20
- vwap_dev
- bb_distance
- zscore_close_20
- rsi_14
- rsi_div_5
- fracdiff_close_d04
- spread_z_64

### Distribution Statistics (Train Partition, Scaler Fit on Train Only)

| feature            |         mean |   std |     skewness |   non_null |
|:-------------------|-------------:|------:|-------------:|-----------:|
| log_return         |  3.37695e-19 |     1 | -1.3021      |    2777407 |
| zscore_close_64    |  4.9128e-19  |     1 | -0.000604136 |    2776911 |
| ema_spread         |  1.6373e-19  |     1 | -0.0953601   |    2777415 |
| atr_14             | -3.40559e-17 |     1 | 27.515       |    2777415 |
| price_slope_20     | -1.39179e-18 |     1 |  0.0242677   |    2777255 |
| vwap_dev           |  1.64209e-19 |     1 |  4.98306     |    2769316 |
| bb_distance        |  6.38584e-18 |     1 | -0.00877504  |    2777263 |
| zscore_close_20    | -7.73669e-18 |     1 | -0.00877519  |    2777263 |
| rsi_14             |  1.51907e-16 |     1 |  0.0226752   |    2777311 |
| rsi_div_5          | -6.4437e-18  |     1 | -0.00881392  |    2777271 |
| fracdiff_close_d04 | -4.71926e-17 |     1 |  1.83259     |    2775167 |
| spread_z_64        |  4.9128e-19  |     1 | -0.000604136 |    2776911 |

## 4) Distribution Visualizations

- ![](data_pipeline/reports/log_return_hist.png)
- ![](data_pipeline/reports/zscore_close_64_hist.png)
- ![](data_pipeline/reports/ema_spread_hist.png)
- ![](data_pipeline/reports/atr_14_hist.png)
- ![](data_pipeline/reports/price_slope_20_hist.png)
- ![](data_pipeline/reports/vwap_dev_hist.png)
- ![](data_pipeline/reports/bb_distance_hist.png)
- ![](data_pipeline/reports/zscore_close_20_hist.png)
- ![](data_pipeline/reports/rsi_14_hist.png)
- ![](data_pipeline/reports/rsi_div_5_hist.png)
- ![](data_pipeline/reports/fracdiff_close_d04_hist.png)
- ![](data_pipeline/reports/spread_z_64_hist.png)

## 5) Leakage Controls

- Scaler fit operation executed only on train partition.
- Validation and test partitions transformed using train-fitted scaler stats.
- Purge gap applied to prevent horizon bleeding across partitions.
