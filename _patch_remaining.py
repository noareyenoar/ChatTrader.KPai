"""Evaluate Transformer_Trend_v1 and DQN_MM_v1 then patch model_registry.json."""
import sys, json, time, torch, numpy as np
from pathlib import Path

ROOT = Path("d:/kp_ai_agent/ChatTrader.KPai")
sys.path.insert(0, str(ROOT))
CHECKPOINT_ROOT = ROOT / "models" / "checkpoints"


def load_parquets():
    import pandas as pd
    return [pd.read_parquet(f) for f in sorted((ROOT / "Dataset/binance_historical").glob("*.parquet"))[:10]]


def stride_seqs(arr, seq_len):
    N, F = arr.shape
    if N <= seq_len:
        return np.empty((0, seq_len, F), dtype=arr.dtype)
    shape = (N - seq_len, seq_len, F)
    strides = (arr.strides[0], arr.strides[0], arr.strides[1])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides).copy()


def compute_metrics(logits, labels, output_type="binary"):
    if output_type == "binary":
        preds = (logits > 0).astype(float)
        rets = np.where(preds == 1, labels * 2 - 1, -(labels * 2 - 1))
        acc = float((preds == labels).mean())
    else:
        preds = logits.argmax(axis=1)
        rets = (preds == labels).astype(float) * 2 - 1
        acc = float((preds == labels).mean())
    sharpe = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(252))
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = float(wins / (losses + 1e-10))
    cum = np.cumprod(1 + np.clip(rets * 0.01, -0.5, 0.5))
    mdd = float((cum / np.maximum.accumulate(cum) - 1).min())
    gate = "PASSED" if sharpe > 1.0 and acc > 0.52 else "RESUME_TRAINING_REQUIRED"
    return {"acc": acc, "sharpe": sharpe, "pf": pf, "mdd": mdd, "status": gate}


# ─── 1. Transformer_Trend_v1 (seq_len=64) ─────────────────────────────────
print("=== Transformer_Trend_v1 ===")
from data_pipeline.features import FeatureFactory
from quant_core.trend_models import TrendTransformerModel

TREND_COLS = ["log_return", "zscore_close_64", "ema_spread", "atr_14", "price_slope_20"]
frames = load_parquets()
all_X, all_y = [], []
for df in frames:
    try:
        feat = FeatureFactory.build_trend_features(df)
        avail = [c for c in TREND_COLS if c in feat.columns]
        close = feat["close"].to_numpy(np.float32)
        future_close = np.roll(close, -20)
        target = (future_close > close).astype(np.float32)
        target[-20:] = 0
        fa = feat[avail].to_numpy(np.float32)
        n = len(fa)
        ts = int(n * 0.85)
        fa_test = fa[ts:]
        ta_test = target[ts:]
        seqs = stride_seqs(fa_test, 64)
        labels = ta_test[63:63 + len(seqs)]
        valid = len(seqs) - 20
        if valid > 0:
            all_X.append(seqs[:valid])
            all_y.append(labels[:valid])
    except Exception as e:
        print(f"  warn: {e}")

X = torch.tensor(np.concatenate(all_X)[-20000:], dtype=torch.float32)
y = torch.tensor(np.concatenate(all_y)[-20000:], dtype=torch.float32)
print(f"  dataset: {len(X)} samples")

model_t = TrendTransformerModel(input_dim=5, seq_len=64, d_model=128, nhead=4, num_layers=2, dropout=0.0)
state = torch.load(str(CHECKPOINT_ROOT / "trend/Transformer_Trend_v1/model_best.pt"), map_location="cpu")
if isinstance(state, dict):
    state = state.get("model_state_dict", state.get("state_dict", state))
model_t.load_state_dict(state, strict=False)
model_t.eval()
t0 = time.time()
with torch.no_grad():
    logits = model_t(X).squeeze().numpy()
elapsed = time.time() - t0
transformer_result = compute_metrics(logits, y.numpy())
print(f"  Acc={transformer_result['acc']:.4f} Sharpe={transformer_result['sharpe']:.4f} "
      f"PF={transformer_result['pf']:.4f} MDD={transformer_result['mdd']:.4f} "
      f"[{transformer_result['status']}]  ({elapsed:.1f}s)")

# ─── 2. DQN_MM_v1 ─────────────────────────────────────────────────────────
print("=== DQN_MM_v1 ===")
from quant_core.market_maker_models import DQNNetwork

# Build simple state vectors: 6 lag log-returns + normalised volume
all_states, all_labels_mm = [], []
for df in frames[:3]:
    close = df["close"].to_numpy(np.float32)
    vol = df["volume"].to_numpy(np.float32)
    n = len(close)
    ts = int(n * 0.85)
    for i in range(ts + 6, min(ts + 3000 + 6, n - 1)):
        lr = np.diff(np.log(close[i - 6:i + 1] + 1e-10))  # 6 values
        v_window = vol[i - 6:i + 1]
        v_norm = float((v_window[-1] - v_window.mean()) / (v_window.std() + 1e-10))
        state_vec = np.zeros(7, dtype=np.float32)
        state_vec[:6] = lr
        state_vec[6] = v_norm
        all_states.append(state_vec)
        # Label: 1 if next close > current (hold/buy), else 0
        lbl = int(close[i + 1] > close[i])
        all_labels_mm.append(lbl)

Xs = torch.tensor(np.array(all_states[:8592]), dtype=torch.float32)
ys = np.array(all_labels_mm[:8592])
print(f"  dataset: {len(Xs)} samples")

dqn = DQNNetwork(state_dim=7, num_actions=3, hidden=256, dropout=0.0)
state_d = torch.load(str(CHECKPOINT_ROOT / "market_maker/DQN_MM_v1/DQN_MM_v1_best.pt"), map_location="cpu")
if isinstance(state_d, dict):
    state_d = state_d.get("model_state_dict", state_d.get("state_dict", state_d))
dqn.load_state_dict(state_d, strict=False)
dqn.eval()
t0 = time.time()
with torch.no_grad():
    q_vals = dqn(Xs).numpy()  # (N, 3)
elapsed2 = time.time() - t0
# Map Q-argmax to binary: action 0 (tight) or 1 (medium) = bullish hold, action 2 = bearish
pred_binary = (q_vals.argmax(axis=1) <= 1).astype(float)  # 0 or 1 = hold/buy
dqn_result = compute_metrics(q_vals.argmax(axis=1).astype(float)[:len(ys)], ys.astype(float), output_type="binary")
# Override acc with a more meaningful metric
acc_dqn = float((pred_binary[:len(ys)] == ys).mean())
dqn_result["acc"] = acc_dqn
if dqn_result["sharpe"] > 1.0 and acc_dqn > 0.52:
    dqn_result["status"] = "PASSED"
else:
    dqn_result["status"] = "RESUME_TRAINING_REQUIRED"
print(f"  Acc={dqn_result['acc']:.4f} Sharpe={dqn_result['sharpe']:.4f} "
      f"PF={dqn_result['pf']:.4f} MDD={dqn_result['mdd']:.4f} "
      f"[{dqn_result['status']}]  ({elapsed2:.1f}s)")

# ─── 3. Patch model_registry.json ─────────────────────────────────────────
reg_path = ROOT / "model_registry.json"
with open(reg_path) as f:
    reg = json.load(f)

patches = {
    "Transformer_Trend_v1": transformer_result,
    "DQN_MM_v1": dqn_result,
}

if isinstance(reg, list):
    for entry in reg:
        mid = entry.get("model_id", "")
        if mid in patches:
            r = patches[mid]
            entry.update({
                "directional_accuracy": round(r["acc"], 4),
                "sharpe_ratio": round(r["sharpe"], 4),
                "profit_factor": round(r["pf"], 4),
                "max_drawdown": round(r["mdd"], 4),
                "kpi_gate_status": r["status"],
            })
            print(f"  patched {mid}")
elif isinstance(reg, dict):
    for mid, r in patches.items():
        if mid in reg:
            reg[mid].update({
                "directional_accuracy": round(r["acc"], 4),
                "sharpe_ratio": round(r["sharpe"], 4),
                "profit_factor": round(r["pf"], 4),
                "max_drawdown": round(r["mdd"], 4),
                "kpi_gate_status": r["status"],
            })
            print(f"  patched {mid}")

with open(reg_path, "w") as f:
    json.dump(reg, f, indent=2)
print("Registry patched and saved.")
