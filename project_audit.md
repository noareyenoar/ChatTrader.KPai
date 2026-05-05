# Project Audit - Phase 1

Date: 2026-04-25
Workspace Root: `d:\kp_ai_agent\ChatTrader.KPai`
Scope: Initial audit for ChatTrader.KPai build plan kickoff from `master_prompt.md`.

## 1) Structure Audit

### 1.1 Top-Level Layout
Observed primary modules:
- `ChatDev_forked/`: Forked DevAll/ChatDev runtime (backend, frontend, workflow engine).
- `Dataset/binance_historical/`: Market data store.
- `coder_agent_system_prompts/`: Prompt library grouped by discipline (ai-ml, data-engineering, performance, etc.).
- Strategy and protocol docs:
  - `master_prompt.md`
  - `ChatTraderKPai-Fork-Guidelines.md`
  - `pytorch_model_training_rule.md`
  - `trader_archetype_model_list.md`

### 1.2 Fork Core Runtime Shape (`ChatDev_forked/`)
Key areas present and consistent with a multi-agent orchestration base:
- Backend/API: `server/`, `server_main.py`
- Runtime orchestration: `runtime/`, `workflow/`, `entity/`, `schema_registry/`
- Tools/utilities: `tools/`, `utils/`, `check/`
- Frontend (Vue + Vite): `frontend/`
- Tests: `tests/`
- Config/workflows: `yaml_instance/`, `yaml_template/`

### 1.3 Immediate Structural Gaps for ChatTrader Target
Missing dedicated directories/modules for target trading system (to be created in later phases):
- `data_pipeline/` (feature factory by archetype)
- `models/` checkpoint hierarchy by archetype
- `agents/` debate/trader/orchestrator implementation
- `backtesting/` or simulation package for walk-forward and slippage/latency

## 2) Dataset & Schema Audit

### 2.1 File Format Verification
Path audited: `Dataset/binance_historical/`
- Total files: 60
- Parquet files: 58
- CSV files: 0
- Metadata files: `manifest.json`, `schema.json`
- Total size: 690,455,966 bytes (~658.5 MiB)

Result: Dataset is parquet-native and aligned with expected time-series storage.

### 2.2 Declared Schema (`schema.json`)
Contract metadata:
- `contract_version`: `phase7.v1`
- `dataset`: `raw_data_set3`
- `timeframe`: `5m`

Columns:
- `timestamp`
- `symbol`
- `open`, `high`, `low`, `close`
- `volume`, `quote_volume`, `trades`
- `taker_buy_base`, `taker_buy_quote`
- `open_interest`, `funding_rate`

Schema note:
- `open_interest` / `funding_rate` may be zero-filled where unavailable.

### 2.3 Ingestion Quality Signals (`manifest.json`)
Reported by manifest:
- Requested symbols: 59
- Successful symbols: 41
- Failed symbols: 18
- Timeframe: `5m`
- History mode: `full`

Observed anomalies/risk flags:
- At least one major symbol (`BTCUSDT`) is marked `FAIL` in manifest due to timeout.
- Very small parquet files exist (e.g., `CHIPUSDT.parquet` at 94,648 bytes), likely short history/newly listed assets.
- No zero-byte parquet files detected.

Audit conclusion for Phase 1:
- Format and schema are usable.
- Data completeness is mixed and needs symbol-level filtering and quality gates in Phase 3 before model training.

## 3) Dependency Audit

### 3.1 Python Runtime Requirements vs Local Environment
From `ChatDev_forked/pyproject.toml`:
- Required: Python `>=3.12,<3.13`

Observed on machine:
- `python --version` -> 3.11.9
- `uv --version` -> not found

Impact:
- Current local Python does not satisfy project constraint.
- `Makefile` and README workflows assume `uv` (e.g., `uv sync`, `uv run ...`), currently unavailable.

### 3.2 Backend Dependency Snapshot
From `requirements.txt` / `pyproject.toml`:
- API/runtime stack present: FastAPI, Uvicorn, Pydantic, WebSockets.
- Data stack present: pandas, numpy, matplotlib, seaborn.
- Orchestration stack present: mcp/fastmcp, openai, tenacity.
- Memory tooling present: `mem0ai` (pyproject only).

Notable gap for planned Phase 4 quant core:
- No explicit `torch` dependency found in `requirements.txt` or `pyproject.toml`.
- No explicit `tensorboard` dependency found.
- No explicit `pyarrow`/`fastparquet` pin found (required for robust parquet IO with pandas in many environments).

### 3.3 Frontend Dependency Snapshot
From `ChatDev_forked/frontend/package.json`:
- Vue 3 + Vite 7 stack configured.
- Flow graph ecosystem installed (`@vue-flow/*`).
- ESLint tooling present.

## 4) Hardware & Execution Constraints

### 4.1 GPU / CUDA Availability
Observed:
- `nvidia-smi` command not available in current shell environment.

Interpretation:
- No accessible NVIDIA driver/GPU from current runtime context, or PATH/driver not configured.
- GPU-first training mandate from project protocol cannot be confirmed on this machine at this time.

### 4.2 Risk to Master Prompt Requirements
Master prompt Phase 4 requires GPU-oriented PyTorch training for 18 models.
Current constraints indicate likely blockers unless environment is updated:
- Python version mismatch (3.11 vs required 3.12)
- `uv` missing
- CUDA visibility unavailable
- PyTorch not yet declared in dependencies

## 5) Readiness Assessment

### Ready now
- Repository baseline and fork structure are intact.
- Dataset storage format (parquet) and schema metadata are present.
- Manifest gives enough quality metadata to design filtering rules.

### Not ready yet (blocking for later phases)
- Reproducible Python environment bootstrap (`uv`, Python 3.12)
- Verified CUDA-enabled runtime
- Quant training dependencies (`torch`, tensorboard, parquet engine guarantees)
- Data quality gate logic for failed/low-history symbols

## 6) Recommended Next Actions (before Phase 3/4 execution)

1. Align environment to project spec:
   - Install Python 3.12.x
   - Install `uv`
   - Recreate env using `uv sync`
2. Verify CUDA visibility:
   - Confirm NVIDIA driver + CUDA toolkit availability from this workspace runtime
   - Validate `torch.cuda.is_available()` after torch install
3. Add/confirm ML dependencies:
   - `torch` (CUDA-compatible build)
   - `tensorboard`
   - parquet engine compatibility (`pyarrow` preferred)
4. Define data-quality gate in pipeline:
   - Exclude manifest-`FAIL` symbols by default
   - Enforce minimum history length threshold per archetype/model class
   - Track missing-bar ratio per symbol

## 7) Phase 1 Exit Status

Phase 1 deliverable `project_audit.md` completed.
Stopped at checkpoint and waiting for user review/approval before moving to Phase 2.
