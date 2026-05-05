# Backtest V1 Summary Report

## Run Metadata
- Run started: 2026-05-01T19:08:11
- Run finished: 2026-05-01T19:08:14
- Symbol: BTCUSDT
- Timeframe: 5m
- Model: qwen3.5:4b
- Dry run mode: True
- Available 2025 data files processed: 6

### Data Coverage (2025 files found in workspace)
- Dataset/binance_vision_real/BTCUSDT/aggTrades/2025-10/20251010.parquet
- Dataset/binance_vision_real/BTCUSDT/aggTrades/2025-12/20251227.parquet
- Dataset/binance_vision_real/BTCUSDT/aggTrades/2025-12/20251228.parquet
- Dataset/binance_vision_real/BTCUSDT/aggTrades/2025-12/20251229.parquet
- Dataset/binance_vision_real/BTCUSDT/aggTrades/2025-12/20251230.parquet
- Dataset/binance_vision_real/BTCUSDT/aggTrades/2025-12/20251231.parquet

## Aggregate Performance
- Total bars processed: 1338
- Debates run: 31
- Trades opened: 9
- Trades closed: 9
- Win/Loss: 0/9 (win rate: 0.00%)
- Net PnL (sum sized): -0.001967 (-0.1967%)
- Flat decisions: 22
- Runtime errors caught: 0
- Credibility updates: 54

## Per-Day Backtest Stats
| Date File | Bars | Debates | Opened | Closed | Wins | Losses | Net PnL | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20251010.parquet | 223 | 25 | 4 | 4 | 0 | 4 | -0.001967 | 0 |
| 20251227.parquet | 223 | 1 | 1 | 1 | 0 | 1 | 0.000000 | 0 |
| 20251228.parquet | 223 | 1 | 1 | 1 | 0 | 1 | 0.000000 | 0 |
| 20251229.parquet | 223 | 1 | 1 | 1 | 0 | 1 | 0.000000 | 0 |
| 20251230.parquet | 223 | 2 | 1 | 1 | 0 | 1 | 0.000000 | 0 |
| 20251231.parquet | 223 | 1 | 1 | 1 | 0 | 1 | 0.000000 | 0 |

## Glass Brain / Omni Debate Log Summary
- New Omni events captured in this run: 661
- trade_open events: 9
- trade_close events: 9
- Closed-trade PnL sum from Omni log: -0.001967
- Stage distribution:
  - THINKING: 248
  - INPUT: 217
  - OUTPUT: 80
  - FEELING: 62
  - EVOLVING: 54
- Event kind distribution (top):
  - analyst_input: 186
  - analyst_output: 186
  - credibility_update: 54
  - debate_start: 31
  - shadow_input: 31
  - shadow_output: 31
  - orchestrator_input: 31
  - pm_input: 31
  - pm_output: 31
  - orchestrator_output: 31
  - trade_open: 9
  - trade_close: 9

## Journal Summary
- New journal sessions in this run: 31
- Regime distribution:
  - RANGING: 24
  - HIGH_VOLATILITY: 4
  - LOW_VOLATILITY: 2
  - REVERTING: 1
- Path distribution:
  - SLOW: 31
- Error decomposition averages (sessions with closed outcomes only):
  - avg signal_error: 0.3888
  - avg decision_error: 0.0224
  - avg execution_error: 0.0069
  - avg total_loss_score: 0.4181

## Agent Team Themes (from debate packets)
- DiscretionaryAnalyst
  - dominant direction: FLAT (28 packets)
  - direction mix: LONG=2, SHORT=1, FLAT=28
  - avg confidence: 0.5041
  - recurring thesis terms: unavailable, falling, back
- MarketMakerAnalyst
  - dominant direction: LONG (30 packets)
  - direction mix: LONG=30, SHORT=1, FLAT=0
  - avg confidence: 0.8373
  - recurring thesis terms: unavailable, falling, back
- MeanReversionAnalyst
  - dominant direction: SHORT (17 packets)
  - direction mix: LONG=11, SHORT=17, FLAT=3
  - avg confidence: 0.6726
  - recurring thesis terms: unavailable, falling, back
- ScalperAnalyst
  - dominant direction: SHORT (19 packets)
  - direction mix: LONG=11, SHORT=19, FLAT=1
  - avg confidence: 0.5879
  - recurring thesis terms: unavailable, falling, back
- StatArbAnalyst
  - dominant direction: LONG (13 packets)
  - direction mix: LONG=13, SHORT=12, FLAT=6
  - avg confidence: 0.5370
  - recurring thesis terms: unavailable, falling, back
- TrendAnalyst
  - dominant direction: LONG (17 packets)
  - direction mix: LONG=17, SHORT=7, FLAT=7
  - avg confidence: 0.6543
  - recurring thesis terms: unavailable, falling, back

## Agent Growth (Credibility Evolution)
- New growth events: 54
| Agent | Updates | Net Delta | Avg Delta | Profitable-linked Updates |
|---|---:|---:|---:|---:|
| DiscretionaryAnalyst | 9 | +0.2410 | +0.0268 | 0 |
| MarketMakerAnalyst | 9 | -0.3220 | -0.0358 | 0 |
| MeanReversionAnalyst | 9 | -0.2220 | -0.0247 | 0 |
| ScalperAnalyst | 9 | -0.1319 | -0.0147 | 0 |
| StatArbAnalyst | 9 | -0.2409 | -0.0268 | 0 |
| TrendAnalyst | 9 | +0.2491 | +0.0277 | 0 |

## Notes
- This report summarizes all *available* 2025 BTCUSDT parquet files currently present in the workspace.
- If additional 2025 daily files are added later, re-run this script to expand coverage.
