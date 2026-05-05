from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "model_registry.json"
SUMMARY_PATH = ROOT / "model_performance_summary.md"


@dataclass
class PerfRow:
    model: str
    archetype: str
    train_accuracy: str
    val_accuracy: str
    train_loss: str
    val_loss: str
    test_sharpe: str
    test_maxdd: str
    test_pf: str
    status: str


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def _pick(v: dict[str, Any], keys: list[str]) -> float | None:
    for k in keys:
        if k in v:
            return _safe_float(v[k])
    return None


def _tb_dir_for_entry(entry: dict[str, Any]) -> Path | None:
    weights_path = entry.get("weights_path")
    if isinstance(weights_path, str) and weights_path:
        weights = Path(weights_path)
        parts = list(weights.parts)
        if len(parts) >= 4 and parts[0] == "models" and parts[1] == "checkpoints":
            return ROOT.joinpath("models", "tensorboard", *parts[2:-1])

    model_name = entry.get("architecture_name")
    if not isinstance(model_name, str):
        return None

    mapping = {
        "LSTM_Trend_v1": ROOT / "models/tensorboard/trend/LSTM_Trend_v1",
        "Transformer_Trend_v1": ROOT / "models/tensorboard/trend/Transformer_Trend_v1",
        "TCN_Trend_v1": ROOT / "models/tensorboard/trend/TCN_Trend_v1",
        "MLP_MR_v1": ROOT / "models/tensorboard/mean_reversion/MLP_MR_v1",
        "ResNet_MR_v1": ROOT / "models/tensorboard/mean_reversion/ResNet_MR_v1",
        "GRN_MR_v1": ROOT / "models/tensorboard/mean_reversion/GRN_MR_v1",
        "CNN_Scalper_v1": ROOT / "models/tensorboard/scalper/CNN_Scalper_v1",
        "LinearAttn_Scalper_v1": ROOT / "models/tensorboard/scalper/LinearAttn_Scalper_v1",
        "GRU_Scalper_v1": ROOT / "models/tensorboard/scalper/GRU_Scalper_v1",
        "Autoencoder_StatArb_v1": ROOT / "models/tensorboard/stat_arb/Autoencoder_StatArb_v1",
        "GAT_StatArb_v1": ROOT / "models/tensorboard/stat_arb/GAT_StatArb_v1",
        "LSTM_StatArb_v1": ROOT / "models/tensorboard/stat_arb/LSTM_StatArb_v1",
        "ViT_Disc_v1": ROOT / "models/tensorboard/discretionary/ViT_Disc_v1",
        "Multimodal_Disc_v1": ROOT / "models/tensorboard/discretionary/Multimodal_Disc_v1",
        "CNNChart_Disc_v1": ROOT / "models/tensorboard/discretionary/CNNChart_Disc_v1",
        "PPO_MM_v1": ROOT / "models/tensorboard/market_maker/PPO_MM_v1",
        "SAC_MM_v1": ROOT / "models/tensorboard/market_maker/SAC_MM_v1",
        "DQN_MM_v1": ROOT / "models/tensorboard/market_maker/DQN_MM_v1",
    }
    return mapping.get(model_name)


def _read_tb_last_scalar(tb_dir: Path | None, tag: str) -> float | None:
    if tb_dir is None or not tb_dir.exists():
        return None
    try:
        ea = event_accumulator.EventAccumulator(str(tb_dir))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if tag not in tags:
            return None
        events = ea.Scalars(tag)
        if not events:
            return None
        return float(events[-1].value)
    except Exception:
        return None


def _read_tb_first_available(tb_dir: Path | None, tags: list[str]) -> float | None:
    for tag in tags:
        value = _read_tb_last_scalar(tb_dir, tag)
        if value is not None:
            return value
    return None


def _finalize_entry(entry: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    v = entry.setdefault("validation", entry.get("validation", {}))

    val_sharpe = _pick(v, ["val_sharpe"])
    test_sharpe = _pick(v, ["test_sharpe", "sharpe_rewards"])
    val_acc = _pick(v, ["val_accuracy", "val_directional_accuracy"])
    test_acc = _pick(v, ["test_accuracy", "test_directional_accuracy"])

    # OOS consistency rule: test Sharpe must not decay > 50% vs validation.
    oos_consistency = True
    if val_sharpe is not None and test_sharpe is not None and abs(val_sharpe) > 1e-12:
        oos_consistency = test_sharpe >= (0.5 * val_sharpe)

    blunt_fail = False
    if test_sharpe is not None and test_sharpe < 0.0:
        blunt_fail = True
    if test_acc is not None and test_acc < 0.50:
        blunt_fail = True

    sanity_passed = bool(v.get("sanity_passed", True))
    final_valid = bool(sanity_passed and oos_consistency and not blunt_fail)

    v["oos_consistency_passed"] = bool(oos_consistency)
    v["is_valid"] = bool(final_valid)
    v["status"] = "FAILED" if blunt_fail else ("PASSED" if final_valid else "REVIEW")

    entry["validation"] = v
    return entry, blunt_fail


def _build_summary(entries: list[dict[str, Any]]) -> str:
    rows: list[PerfRow] = []

    for e in entries:
        name = e.get("architecture_name", "unknown")
        archetype = e.get("archetype", "unknown")
        v = e.get("validation", {})
        tb_dir = _tb_dir_for_entry(e)

        train_loss = _read_tb_last_scalar(tb_dir, "train/loss")
        val_loss_tb = _read_tb_last_scalar(tb_dir, "val/loss")

        val_acc = _pick(v, ["val_accuracy", "val_directional_accuracy"])
        train_acc_tb = _read_tb_first_available(tb_dir, ["train/accuracy", "train/directional_acc", "train/f1"])

        val_loss_metric = _pick(v, ["val_loss", "val_mae", "val_tracking_error"])
        test_sharpe = _pick(v, ["test_sharpe", "sharpe_rewards"])
        test_mdd = _pick(v, ["test_max_drawdown", "max_drawdown_rewards"])
        test_pf = _pick(v, ["test_profit_factor"])

        rows.append(
            PerfRow(
                model=name,
                archetype=archetype,
                train_accuracy=_fmt(train_acc_tb),
                val_accuracy=_fmt(val_acc),
                train_loss=_fmt(train_loss),
                val_loss=_fmt(val_loss_tb if val_loss_tb is not None else val_loss_metric),
                test_sharpe=_fmt(test_sharpe),
                test_maxdd=_fmt(test_mdd),
                test_pf=_fmt(test_pf),
                status=str(v.get("status", "REVIEW")),
            )
        )

    def _sort_key(r: PerfRow) -> tuple[float, float]:
        try:
            s = float(r.test_sharpe)
        except Exception:
            s = -1e9
        try:
            a = float(r.val_accuracy)
        except Exception:
            a = -1e9
        return (s, a)

    leaderboard = sorted(rows, key=_sort_key, reverse=True)

    lines: list[str] = []
    lines.append("# Model Performance Summary (Full Training Sweep)")
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| Rank | Model | Archetype | Test Sharpe | Val Accuracy | Status |")
    lines.append("|---:|---|---|---:|---:|---|")
    for i, r in enumerate(leaderboard, start=1):
        lines.append(f"| {i} | {r.model} | {r.archetype} | {r.test_sharpe} | {r.val_accuracy} | {r.status} |")

    lines.append("")
    lines.append("## Per-Model Metrics")
    lines.append("")
    lines.append("| Model | Archetype | Train Accuracy | Val Accuracy | Train Loss | Val Loss | Test Sharpe | Test Max Drawdown | Test Profit Factor | Status |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r.model} | {r.archetype} | {r.train_accuracy} | {r.val_accuracy} | {r.train_loss} | {r.val_loss} | {r.test_sharpe} | {r.test_maxdd} | {r.test_pf} | {r.status} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Train Accuracy is shown as n/a when not logged by that model family.")
    lines.append("- Val Loss is sourced from TensorBoard val/loss when available; otherwise from registry validation fields.")
    lines.append("- FAILED status is applied when Test Sharpe < 0 or Test Accuracy < 50%.")

    return "\n".join(lines) + "\n"


def main() -> int:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    entries = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    finalized: list[dict[str, Any]] = []
    archetype_failed: dict[str, bool] = {}
    for e in entries:
        upd, failed = _finalize_entry(e)
        arch = str(upd.get("archetype", "unknown"))
        archetype_failed[arch] = archetype_failed.get(arch, False) or failed
        finalized.append(upd)

    for e in finalized:
        arch = str(e.get("archetype", "unknown"))
        e["archetype_status"] = "FAILED" if archetype_failed.get(arch, False) else "PASSED"

    REGISTRY_PATH.write_text(json.dumps(finalized, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(_build_summary(finalized), encoding="utf-8")
    print(f"updated_registry={REGISTRY_PATH}")
    print(f"written_summary={SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
