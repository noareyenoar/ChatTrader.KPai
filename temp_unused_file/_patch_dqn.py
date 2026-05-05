"""Patch Transformer result + evaluate DQN with correct checkpoint dims."""
import sys, json, time, torch, numpy as np
from pathlib import Path

ROOT = Path("d:/kp_ai_agent/ChatTrader.KPai")
sys.path.insert(0, str(ROOT))
CHECKPOINT_ROOT = ROOT / "models" / "checkpoints"
REG_PATH = ROOT / "model_registry.json"

# Known results from previous run
TRANSFORMER_RESULT = {
    "directional_accuracy": 0.5120,
    "sharpe_ratio": 0.3795,
    "profit_factor": 1.0490,
    "max_drawdown": -0.9924,
    "kpi_gate_status": "RESUME_TRAINING_REQUIRED",
    "n_samples": 20000,
}

# ─── DQN with actual checkpoint dims: state_dim=8, hidden=128 ─────────────
print("=== DQN_MM_v1 (state_dim=8, hidden=128) ===")
from quant_core.market_maker_models import DQNNetwork
import pandas as pd

frames = [pd.read_parquet(f) for f in sorted((ROOT / "Dataset/binance_historical").glob("*.parquet"))[:3]]

all_states, all_labels_mm = [], []
for df in frames:
    close = df["close"].to_numpy(np.float32)
    vol = df["volume"].to_numpy(np.float32)
    hi = df["high"].to_numpy(np.float32)
    lo = df["low"].to_numpy(np.float32)
    n = len(close)
    ts = int(n * 0.85)
    for i in range(ts + 7, min(ts + 3000 + 7, n - 1)):
        lr = np.diff(np.log(close[i-7:i+1] + 1e-10))   # 7 log-returns
        v_window = vol[i-4:i+1]
        v_norm = float((v_window[-1] - v_window.mean()) / (v_window.std() + 1e-10))
        state_vec = np.zeros(8, dtype=np.float32)
        state_vec[:7] = lr
        state_vec[7] = v_norm
        all_states.append(state_vec)
        all_labels_mm.append(int(close[i + 1] > close[i]))

Xs = torch.tensor(np.array(all_states[:8592]), dtype=torch.float32)
ys = np.array(all_labels_mm[:8592])
print(f"  dataset: {len(Xs)} samples, state_dim={Xs.shape[1]}")

dqn = DQNNetwork(state_dim=8, num_actions=3, hidden=128, dropout=0.0)
state_d = torch.load(str(CHECKPOINT_ROOT / "market_maker/DQN_MM_v1/DQN_MM_v1_best.pt"), map_location="cpu")
if isinstance(state_d, dict):
    state_d = state_d.get("model_state_dict", state_d.get("state_dict", state_d))
dqn.load_state_dict(state_d, strict=False)
dqn.eval()
t0 = time.time()
with torch.no_grad():
    q_vals = dqn(Xs).numpy()
elapsed = time.time() - t0

# Map argmax to buy/hold (actions 0,1) vs sell (action 2)
pred_binary = (q_vals.argmax(axis=1) <= 1).astype(float)
acc = float((pred_binary == ys).mean())
# Return series: long if pred=1, short if pred=0
rets = np.where(pred_binary == 1, ys * 2.0 - 1, -(ys * 2.0 - 1))
sharpe = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(252))
wins = rets[rets > 0].sum(); losses = abs(rets[rets < 0].sum())
pf = float(wins / (losses + 1e-10))
cum = np.cumprod(1 + np.clip(rets * 0.01, -0.5, 0.5))
mdd = float((cum / np.maximum.accumulate(cum) - 1).min())
gate = "PASSED" if sharpe > 1.0 and acc > 0.52 else "RESUME_TRAINING_REQUIRED"
print(f"  Acc={acc:.4f}  Sharpe={sharpe:.4f}  PF={pf:.4f}  MDD={mdd:.4f}  [{gate}]  ({elapsed:.1f}s)")

DQN_RESULT = {
    "directional_accuracy": round(acc, 4),
    "sharpe_ratio": round(sharpe, 4),
    "profit_factor": round(pf, 4),
    "max_drawdown": round(mdd, 4),
    "kpi_gate_status": gate,
    "n_samples": len(Xs),
}

# ─── Patch registry ────────────────────────────────────────────────────────
with open(REG_PATH) as f:
    reg = json.load(f)

patched = {"Transformer_Trend_v1": TRANSFORMER_RESULT, "DQN_MM_v1": DQN_RESULT}

for entry in reg:
    mid = entry.get("architecture_name", "")
    if mid in patched:
        v = patched[mid]
        entry["validation"].update({
            "status": v["kpi_gate_status"],
            "directional_accuracy": v["directional_accuracy"],
            "sharpe": v["sharpe_ratio"],
            "profit_factor": v["profit_factor"],
            "max_drawdown": v["max_drawdown"],
            "n_samples": v.get("n_samples", 20000),
        })
        if "reason" in entry["validation"]:
            del entry["validation"]["reason"]
        print(f"  patched {mid}")

with open(REG_PATH, "w") as f:
    json.dump(reg, f, indent=2)
print("Registry saved.")

# Print final summary table
print("\n" + "=" * 90)
print(f"{'Model':<38} {'Acc':>6} {'Sharpe':>8} {'PF':>7} {'MDD':>8}  Status")
print("-" * 90)
for entry in sorted(reg, key=lambda e: e["validation"].get("sharpe") or -99, reverse=True):
    v = entry["validation"]
    mid = entry["architecture_name"]
    acc_v = v.get("directional_accuracy")
    sh = v.get("sharpe")
    pf_v = v.get("profit_factor")
    mdd_v = v.get("max_drawdown")
    st = v.get("status", "?")
    if acc_v is not None:
        print(f"{mid:<38} {acc_v:>6.4f} {sh:>8.4f} {pf_v:>7.4f} {mdd_v:>8.4f}  {st}")
    else:
        print(f"{mid:<38} {'N/A':>6} {'N/A':>8} {'N/A':>7} {'N/A':>8}  {st}")
