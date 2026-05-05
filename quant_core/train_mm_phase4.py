from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from data_pipeline.config import PipelineConfig
from data_pipeline.quality_gate import DataQualityGate
from .shared_training import aggressive_cleanup, resolve_device
from .market_maker_training import (
    MMResult,
    train_dqn,
    train_ppo,
    train_sac,
    write_mm_registry,
)


def _load_price_series(cfg: dict) -> np.ndarray:
    pipe_cfg = PipelineConfig(
        dataset_dir=Path(cfg["dataset_dir"]),
        manifest_path=Path(cfg["manifest_path"]),
        min_history_bars=int(cfg["min_history_bars"]),
        purge_gap_bars=int(cfg["purge_gap_bars"]),
    )
    accepted = [r.symbol for r in DataQualityGate(pipe_cfg).evaluate() if r.decision == "ACCEPT"]
    max_symbols = int(cfg.get("max_symbols", 0))
    symbols = accepted[:max_symbols] if max_symbols > 0 else accepted
    if not symbols:
        raise RuntimeError("No accepted symbols for Market Making training")

    import pandas as pd
    closes_list: list[np.ndarray] = []
    for symbol in symbols:
        df = pd.read_parquet(pipe_cfg.dataset_dir / f"{symbol}.parquet", columns=["close"])
        close = df["close"].dropna().to_numpy(np.float32)
        if len(close) < 2:
            continue
        closes_list.append(close)

    if not closes_list:
        raise RuntimeError("No valid close series found for Market Making training")

    # Stitch symbols by returns to use the full accepted universe without
    # introducing discontinuities from raw price scale jumps.
    stitched = [closes_list[0]]
    for close in closes_list[1:]:
        ret = close[1:] / (close[:-1] + 1e-8)
        start = stitched[-1][-1]
        rebuilt = np.empty_like(close)
        rebuilt[0] = start
        rebuilt[1:] = start * np.cumprod(ret)
        stitched.append(rebuilt)

    closes = np.concatenate(stitched, axis=0)
    cap = int(cfg.get("max_rows", 0))
    return closes[:cap] if cap > 0 else closes


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Market Making RL models (Phase 4)")
    parser.add_argument("--config", type=str, default="configs/mm_phase4.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    print(f"[mm-run] config={args.config} backend={cfg['training'].get('preferred_backend', 'auto')} max_episodes={cfg['training'].get('max_episodes')} max_steps={cfg['training'].get('max_steps')}", flush=True)
    prices = _load_price_series(cfg["data"])
    print(f"[mm-run] prices ready points={len(prices)}", flush=True)

    device, backend = resolve_device(str(cfg["training"].get("preferred_backend", "auto")))
    ckpt_root = Path(cfg["training"]["checkpoint_root"])
    tb_root = Path(cfg["training"]["tensorboard_root"])

    common = cfg["training"]
    common["seed"] = int(cfg["training"].get("seed", 42))
    common["purge_gap_bars"] = int(cfg["data"].get("purge_gap_bars", 0))

    started = time.time()
    results: list[MMResult] = []

    for algo in ("ppo", "sac", "dqn"):
        algo_cfg = {**common, **cfg["models"].get(algo, {})}
        print(f"[mm-run] launching algo={algo.upper()} backend={backend} device={device}", flush=True)
        if algo == "ppo":
            r = train_ppo(prices, algo_cfg, device, backend, ckpt_root / "PPO_MM_v1", tb_root)
        elif algo == "sac":
            r = train_sac(prices, algo_cfg, device, backend, ckpt_root / "SAC_MM_v1", tb_root)
        else:
            r = train_dqn(prices, algo_cfg, device, backend, ckpt_root / "DQN_MM_v1", tb_root)
        print(
            f"trained={r.model_name} mean_rew={r.mean_episode_reward:.4f} "
            f"std={r.std_episode_reward:.4f} is_valid={r.is_valid}"
        )
        results.append(r)
        aggressive_cleanup(r)

    registry_path = Path(cfg.get("registry_path", "model_registry.json"))
    write_mm_registry(results, registry_path)
    print(f"registry={registry_path} elapsed_s={time.time() - started:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
