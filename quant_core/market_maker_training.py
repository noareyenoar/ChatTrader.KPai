"""Market Making RL training loops — Phase 4.

Three separate training routines:
  train_ppo()   — PPO with GAE advantage estimation
  train_sac()   — SAC with twin critics + temperature learning
  train_dqn()   — DQN with experience replay + target network

All use the MarketMakingEnv replay over historical price data.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data_pipeline.gpu_utils import cleanup_cuda
from .market_maker_env import MarketMakingEnv, SyncVectorMarketMakingEnv
from .market_maker_models import DQNNetwork, PPOActorCritic, SACAgentNetworks
from .shared_training import (
    append_working_log,
    append_registry,
    compute_max_drawdown,
    compute_sharpe,
    make_optimizer,
    resolve_device,
    set_global_seed,
)


@dataclass
class MMResult:
    model_name: str
    checkpoint_dir: str
    mean_episode_reward: float
    std_episode_reward: float
    sharpe_rewards: float
    max_drawdown_rewards: float
    eval_mean_reward: float
    eval_sharpe_rewards: float
    eval_max_drawdown_rewards: float
    is_valid: bool
    backend: str
    num_episodes: int


def _split_prices(
    prices: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    purge_gap: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(prices)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    gap = max(0, int(purge_gap))
    train = prices[: max(0, i1 - gap)]
    val = prices[min(n, i1 + gap): max(i1 + gap, i2 - gap)]
    test = prices[min(n, i2 + gap):]
    return train, val, test


def _check_result(mean_r: float, std_r: float, sharpe_r: float, max_dd: float) -> bool:
    return mean_r > 0.0 and sharpe_r > 0.0 and max_dd < 0.15 and std_r < 1.0


def _make_env(prices: np.ndarray, cfg: dict[str, Any], seed: int) -> MarketMakingEnv:
    return MarketMakingEnv(
        prices,
        episode_length=cfg["episode_length"],
        inventory_lambda=float(cfg.get("inventory_lambda", 0.02)),
        warmup_steps=int(cfg.get("warmup_steps", 20)),
        alpha_pos=float(cfg.get("reward_alpha_pos", 0.75)),
        alpha_neg=float(cfg.get("reward_alpha_neg", 1.35)),
        reward_scale=float(cfg.get("reward_scale", 1.0)),
        survival_bonus=float(cfg.get("survival_bonus", 0.0005)),
        inventory_penalty_power=float(cfg.get("inventory_penalty_power", 2.0)),
        max_drawdown_terminate=float(cfg.get("max_drawdown_terminate", 0.85)),
        seed=seed,
    )


def _make_vec_env(prices: np.ndarray, cfg: dict[str, Any], base_seed: int, num_envs: int) -> SyncVectorMarketMakingEnv:
    env_fns = [
        (lambda s=base_seed + i: _make_env(prices, cfg, s))
        for i in range(num_envs)
    ]
    return SyncVectorMarketMakingEnv(env_fns)


def _rule_based_continuous_action(state: np.ndarray) -> np.ndarray:
    inventory = float(state[0])
    spread = float(state[2])
    vol = float(state[3])
    base = np.clip(0.15 + 0.35 * vol + 0.15 * spread, 0.05, 0.9)
    inv_bias = np.clip(inventory * 0.35, -0.2, 0.2)
    bid = float(np.clip(base + inv_bias, 0.05, 0.95))
    ask = float(np.clip(base - inv_bias, 0.05, 0.95))
    return np.array([bid, ask], dtype=np.float32)


def _rule_based_discrete_action(state: np.ndarray) -> int:
    inventory = abs(float(state[0]))
    vol = float(state[3])
    if inventory > 0.5 or vol > 0.6:
        return 2  # wide
    if vol > 0.25:
        return 1  # medium
    return 0      # tight


def _log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# PPO Training
# ---------------------------------------------------------------------------

def train_ppo(
    price_series: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    backend: str,
    ckpt_dir: Path,
    tb_dir: Path,
) -> MMResult:
    set_global_seed(cfg["seed"])
    train_prices, val_prices, test_prices = _split_prices(
        price_series,
        purge_gap=int(cfg.get("purge_gap_bars", 0)),
    )
    num_envs = max(1, int(cfg.get("num_envs", 4)))
    vec_env = _make_vec_env(train_prices, cfg, int(cfg["seed"]), num_envs)

    model = PPOActorCritic(
        state_dim=MarketMakingEnv.STATE_DIM,
        action_dim=MarketMakingEnv.CONTINUOUS_ACTION_DIM,
        hidden=cfg["hidden"],
    ).to(device)

    opt = make_optimizer(model, backend, float(cfg["lr"]), float(cfg["weight_decay"]))
    writer = SummaryWriter(log_dir=str(tb_dir / "PPO_MM_v1"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[mm:ppo] start backend={backend} device={device} train_points={len(train_prices)} val_points={len(val_prices)}")

    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    clip_eps = float(cfg.get("clip_eps", 0.2))
    ppo_epochs = int(cfg.get("ppo_epochs", 4))
    rollout_len = int(cfg.get("rollout_len", 512))

    all_ep_rewards: list[float] = []
    best_val_rew = -float("inf")

    for episode in range(int(cfg["max_episodes"])):
        states, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []
        states_vec = vec_env.reset()

        for _ in range(rollout_len):
            state_t = torch.tensor(states_vec, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, lp = model.get_action(state_t)
                _, _, val = model.forward(state_t)
            action_np = action.detach().cpu().numpy()
            next_states_vec, rew_vec, done_vec, _ = vec_env.step(action_np)

            states.extend(states_vec.tolist())
            actions.extend(action_np.tolist())
            log_probs_old.extend(lp.detach().cpu().numpy().tolist())
            rewards.extend(rew_vec.tolist())
            dones.extend(done_vec.astype(np.float32).tolist())
            values.extend(val.detach().cpu().numpy().reshape(-1).tolist())
            states_vec = next_states_vec

        # GAE
        advantages, returns = [], []
        gae = 0.0
        next_val = 0.0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_val = values[i]

        s_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        a_t = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        lp_old_t = torch.tensor(log_probs_old, dtype=torch.float32, device=device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_mean = adv_t.mean()
        adv_var = ((adv_t - adv_mean) ** 2).mean()
        adv_t = (adv_t - adv_mean) / (adv_var.sqrt() + 1e-8)

        for _ in range(ppo_epochs):
            mean, log_std, val_pred = model.forward(s_t)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            lp_new = dist.log_prob(a_t).sum(-1)
            ratio = (lp_new - lp_old_t).exp()
            p1 = ratio * adv_t
            p2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            policy_loss = -torch.min(p1, p2).mean()
            value_loss = 0.5 * (val_pred - ret_t).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        ep_reward = float(np.sum(rewards))
        all_ep_rewards.append(ep_reward)
        writer.add_scalar("train/episode_reward", ep_reward, episode)
        _log(f"[mm:ppo] episode {episode + 1}/{int(cfg['max_episodes'])} reward={ep_reward:.4f} envs={num_envs}")
        append_working_log(
            "PPO_MM_v1",
            "EPISODE",
            {
                "train_episode_reward": ep_reward,
                "val_status": "pending",
                "test_status": "pending",
            },
            epoch=episode + 1,
            total_epochs=int(cfg["max_episodes"]),
        )

        # Validation every 10 episodes
        if (episode + 1) % 10 == 0:
            val_rew = _rollout_eval(model, val_prices, cfg, device, discrete=False, is_sac=False)
            writer.add_scalar("val/mean_reward", val_rew, episode)
            _log(f"[mm:ppo] validation episode={episode + 1} val_mean_reward={val_rew:.4f}")
            append_working_log(
                "PPO_MM_v1",
                "VALIDATION",
                {
                    "train_episode_reward": ep_reward,
                    "val_mean_reward": val_rew,
                    "test_status": "pending",
                },
                epoch=episode + 1,
                total_epochs=int(cfg["max_episodes"]),
            )
            if val_rew > best_val_rew:
                best_val_rew = val_rew
                torch.save(model.state_dict(), ckpt_dir / "PPO_MM_v1_best.pt")
                _log(f"[mm:ppo] checkpoint saved val_mean_reward={val_rew:.4f}")

    writer.close()
    rewards_arr = np.array(all_ep_rewards)
    eval_rewards = _rollout_eval_rewards(model, test_prices, cfg, device, discrete=False, is_sac=False)
    eval_mean = float(np.mean(eval_rewards))
    eval_sharpe = compute_sharpe(np.diff(eval_rewards) if len(eval_rewards) > 1 else eval_rewards)
    eval_max_dd = compute_max_drawdown(eval_rewards)
    append_working_log(
        "PPO_MM_v1",
        "FINAL",
        {
            "train_mean_reward": float(np.mean(rewards_arr)),
            "val_best_mean_reward": float(best_val_rew),
            "test_mean_reward": eval_mean,
            "test_sharpe": eval_sharpe,
            "test_max_drawdown": eval_max_dd,
        },
    )
    return MMResult(
        model_name="PPO_MM_v1",
        checkpoint_dir=str(ckpt_dir),
        mean_episode_reward=float(np.mean(rewards_arr)),
        std_episode_reward=float(np.std(rewards_arr)),
        sharpe_rewards=compute_sharpe(np.diff(rewards_arr) if len(rewards_arr) > 1 else rewards_arr),
        max_drawdown_rewards=compute_max_drawdown(rewards_arr),
        eval_mean_reward=eval_mean,
        eval_sharpe_rewards=eval_sharpe,
        eval_max_drawdown_rewards=eval_max_dd,
        is_valid=_check_result(eval_mean, float(np.std(rewards_arr)), eval_sharpe, eval_max_dd),
        backend=backend,
        num_episodes=len(all_ep_rewards),
    )


def _rollout_eval(
    model: nn.Module,
    prices: np.ndarray,
    cfg: dict,
    device: torch.device,
    discrete: bool,
    is_sac: bool,
    num_episodes: int = 5,
) -> float:
    rewards = _rollout_eval_rewards(
        model,
        prices,
        cfg,
        device,
        discrete=discrete,
        is_sac=is_sac,
        num_episodes=num_episodes,
    )
    return float(np.mean(rewards))


def _rollout_eval_rewards(
    model: nn.Module,
    prices: np.ndarray,
    cfg: dict,
    device: torch.device,
    discrete: bool,
    is_sac: bool,
    num_episodes: int = 5,
) -> np.ndarray:
    env = _make_env(prices, cfg, seed=0)
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        ep_r = 0.0
        for _ in range(cfg["episode_length"]):
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if discrete:
                    q = model(s_t)
                    action = int(q.argmax(dim=-1).item())
                elif is_sac:
                    action, _ = model.actor.sample(s_t)
                    action = action.squeeze().cpu().numpy()
                else:
                    action, _ = model.get_action(s_t)
                    action = action.squeeze().cpu().numpy()
            state, rew, done, _ = env.step(action, discrete=discrete)
            ep_r += rew
            if done:
                break
        total_rewards.append(ep_r)
    return np.array(total_rewards, dtype=float)


# ---------------------------------------------------------------------------
# SAC Training
# ---------------------------------------------------------------------------

def train_sac(
    price_series: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    backend: str,
    ckpt_dir: Path,
    tb_dir: Path,
) -> MMResult:
    set_global_seed(cfg["seed"])
    train_prices, val_prices, test_prices = _split_prices(
        price_series,
        purge_gap=int(cfg.get("purge_gap_bars", 0)),
    )
    num_envs = max(1, int(cfg.get("num_envs", 4)))
    vec_env = _make_vec_env(train_prices, cfg, int(cfg["seed"]), num_envs)

    nets = SACAgentNetworks(
        state_dim=MarketMakingEnv.STATE_DIM,
        action_dim=MarketMakingEnv.CONTINUOUS_ACTION_DIM,
        hidden=cfg["hidden"],
    ).to(device)

    act_opt = make_optimizer(nets.actor, backend, float(cfg["lr"]), float(cfg["weight_decay"]))
    crit_opt = make_optimizer(nets.critic, backend, float(cfg["lr"]), float(cfg["weight_decay"]))
    if backend == "directml":
        alpha_opt = torch.optim.SGD([nets.log_alpha], lr=3e-4, momentum=0.9, nesterov=True)
    else:
        alpha_opt = torch.optim.Adam([nets.log_alpha], lr=3e-4)
    target_entropy = -float(MarketMakingEnv.CONTINUOUS_ACTION_DIM)

    writer = SummaryWriter(log_dir=str(tb_dir / "SAC_MM_v1"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[mm:sac] start backend={backend} device={device} train_points={len(train_prices)} val_points={len(val_prices)}")

    buffer = deque(maxlen=int(cfg.get("replay_buffer_size", 50000)))
    batch_size = int(cfg.get("batch_size", 256))
    gamma = float(cfg.get("gamma", 0.99))
    tau = float(cfg.get("tau", 0.005))

    all_ep_rewards: list[float] = []
    best_val_rew = -float("inf")
    warmup_episodes = int(cfg.get("behavior_warmup_episodes", 8))

    states = vec_env.reset()
    ep_rewards = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)

    for step in range(int(cfg["max_steps"])):
        use_behavior = len(all_ep_rewards) < warmup_episodes
        if use_behavior:
            action_np = np.stack([_rule_based_continuous_action(s) for s in states], axis=0)
        else:
            s_t = torch.tensor(states, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _ = nets.actor.sample(s_t)
                action_np = action.detach().cpu().numpy()

        next_states, rew_vec, done_vec, _ = vec_env.step(action_np)
        for i in range(num_envs):
            buffer.append((states[i], action_np[i], float(rew_vec[i]), next_states[i], float(done_vec[i])))
            ep_rewards[i] += float(rew_vec[i])
            ep_lengths[i] += 1
            if done_vec[i]:
                finished_reward = float(ep_rewards[i])
                finished_steps = int(ep_lengths[i])
                all_ep_rewards.append(finished_reward)
                writer.add_scalar("train/episode_reward", finished_reward, len(all_ep_rewards))
                writer.add_scalar("train/episode_steps", finished_steps, len(all_ep_rewards))
                _log(
                    f"[mm:sac] episode {len(all_ep_rewards)} reward={finished_reward:.4f} steps={finished_steps} "
                    f"buffer={len(buffer)} behavior_warmup={use_behavior}"
                )
                append_working_log(
                    "SAC_MM_v1",
                    "EPISODE",
                    {
                        "train_episode_reward": finished_reward,
                        "episode_steps": finished_steps,
                        "val_status": "pending",
                        "test_status": "pending",
                        "behavior_warmup": use_behavior,
                    },
                    epoch=len(all_ep_rewards),
                )
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0
        states = next_states

        if len(buffer) < batch_size:
            continue

        batch = random.sample(buffer, batch_size)
        s_b, a_b, r_b, ns_b, d_b = map(np.array, zip(*batch))
        s_b = torch.tensor(s_b, dtype=torch.float32, device=device)
        a_b = torch.tensor(a_b, dtype=torch.float32, device=device)
        r_b = torch.tensor(r_b, dtype=torch.float32, device=device)
        ns_b = torch.tensor(ns_b, dtype=torch.float32, device=device)
        d_b = torch.tensor(d_b, dtype=torch.float32, device=device)

        # Critic update
        with torch.no_grad():
            next_a, next_lp = nets.actor.sample(ns_b)
            q1_t, q2_t = nets.critic_target(ns_b, next_a)
            q_target = r_b + gamma * (1 - d_b) * (torch.min(q1_t, q2_t) - nets.alpha * next_lp)

        q1, q2 = nets.critic(s_b, a_b)
        crit_loss = 0.5 * ((q1 - q_target).pow(2).mean() + (q2 - q_target).pow(2).mean())
        crit_opt.zero_grad(set_to_none=True)
        crit_loss.backward()
        nn.utils.clip_grad_norm_(nets.critic.parameters(), 1.0)
        crit_opt.step()

        # Actor update
        new_a, new_lp = nets.actor.sample(s_b)
        q1_a, q2_a = nets.critic(s_b, new_a)
        actor_loss = (nets.alpha * new_lp - torch.min(q1_a, q2_a)).mean()
        act_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(nets.actor.parameters(), 1.0)
        act_opt.step()

        # Alpha update
        alpha_loss = -(nets.log_alpha * (new_lp + target_entropy).detach()).mean()
        alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        nn.utils.clip_grad_norm_([nets.log_alpha], 1.0)
        alpha_opt.step()

        # Soft target update
        with torch.no_grad():
            for p, tp in zip(nets.critic.parameters(), nets.critic_target.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

        if (step + 1) % 1000 == 0 and len(all_ep_rewards) > 0:
            val_rew = _rollout_eval(nets, val_prices, cfg, device, discrete=False, is_sac=True)
            writer.add_scalar("val/mean_reward", val_rew, step)
            _log(f"[mm:sac] step {step + 1}/{int(cfg['max_steps'])} val_mean_reward={val_rew:.4f} alpha={nets.alpha.item():.6f}")
            append_working_log(
                "SAC_MM_v1",
                "VALIDATION",
                {
                    "step": step + 1,
                    "val_mean_reward": val_rew,
                    "alpha": float(nets.alpha.item()),
                    "test_status": "pending",
                },
            )
            if val_rew > best_val_rew:
                best_val_rew = val_rew
                torch.save(nets.state_dict(), ckpt_dir / "SAC_MM_v1_best.pt")
                _log(f"[mm:sac] checkpoint saved val_mean_reward={val_rew:.4f}")

    writer.close()
    rewards_arr = np.array(all_ep_rewards) if all_ep_rewards else np.zeros(1)
    eval_rewards = _rollout_eval_rewards(nets, test_prices, cfg, device, discrete=False, is_sac=True)
    eval_mean = float(np.mean(eval_rewards))
    eval_sharpe = compute_sharpe(np.diff(eval_rewards) if len(eval_rewards) > 1 else eval_rewards)
    eval_max_dd = compute_max_drawdown(eval_rewards)
    append_working_log(
        "SAC_MM_v1",
        "FINAL",
        {
            "train_mean_reward": float(np.mean(rewards_arr)),
            "val_best_mean_reward": float(best_val_rew),
            "test_mean_reward": eval_mean,
            "test_sharpe": eval_sharpe,
            "test_max_drawdown": eval_max_dd,
        },
    )
    return MMResult(
        model_name="SAC_MM_v1",
        checkpoint_dir=str(ckpt_dir),
        mean_episode_reward=float(np.mean(rewards_arr)),
        std_episode_reward=float(np.std(rewards_arr)),
        sharpe_rewards=compute_sharpe(np.diff(rewards_arr) if len(rewards_arr) > 1 else rewards_arr),
        max_drawdown_rewards=compute_max_drawdown(rewards_arr),
        eval_mean_reward=eval_mean,
        eval_sharpe_rewards=eval_sharpe,
        eval_max_drawdown_rewards=eval_max_dd,
        is_valid=_check_result(eval_mean, float(np.std(rewards_arr)), eval_sharpe, eval_max_dd),
        backend=backend,
        num_episodes=len(all_ep_rewards),
    )


# ---------------------------------------------------------------------------
# DQN Training
# ---------------------------------------------------------------------------

def train_dqn(
    price_series: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    backend: str,
    ckpt_dir: Path,
    tb_dir: Path,
) -> MMResult:
    set_global_seed(cfg["seed"])
    train_prices, val_prices, test_prices = _split_prices(
        price_series,
        purge_gap=int(cfg.get("purge_gap_bars", 0)),
    )
    num_envs = max(1, int(cfg.get("num_envs", 4)))
    vec_env = _make_vec_env(train_prices, cfg, int(cfg["seed"]), num_envs)

    online = DQNNetwork(
        state_dim=MarketMakingEnv.STATE_DIM,
        num_actions=MarketMakingEnv.DISCRETE_ACTIONS,
        hidden=cfg["hidden"],
    ).to(device)
    target = DQNNetwork(
        state_dim=MarketMakingEnv.STATE_DIM,
        num_actions=MarketMakingEnv.DISCRETE_ACTIONS,
        hidden=cfg["hidden"],
    ).to(device)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)

    opt = make_optimizer(online, backend, float(cfg["lr"]), float(cfg["weight_decay"]))
    writer = SummaryWriter(log_dir=str(tb_dir / "DQN_MM_v1"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[mm:dqn] start backend={backend} device={device} train_points={len(train_prices)} val_points={len(val_prices)}")

    buffer = deque(maxlen=int(cfg.get("replay_buffer_size", 50000)))
    batch_size = int(cfg.get("batch_size", 256))
    gamma = float(cfg.get("gamma", 0.99))
    target_update_freq = int(cfg.get("target_update_freq", 500))
    eps_start = float(cfg.get("eps_start", 1.0))
    eps_end = float(cfg.get("eps_end", 0.05))
    eps_decay = float(cfg.get("eps_decay", 0.995))

    all_ep_rewards: list[float] = []
    best_val_rew = -float("inf")
    eps = eps_start
    states = vec_env.reset()
    ep_rewards = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    warmup_episodes = int(cfg.get("behavior_warmup_episodes", 8))

    for step in range(int(cfg["max_steps"])):
        use_behavior = len(all_ep_rewards) < warmup_episodes
        actions: list[int] = []
        if use_behavior:
            actions = [_rule_based_discrete_action(s) for s in states]
        else:
            s_t = torch.tensor(states, dtype=torch.float32, device=device)
            if random.random() < eps:
                actions = [random.randint(0, MarketMakingEnv.DISCRETE_ACTIONS - 1) for _ in range(num_envs)]
            else:
                with torch.no_grad():
                    actions = online(s_t).argmax(dim=-1).detach().cpu().numpy().astype(int).tolist()

        action_arr = np.asarray(actions, dtype=np.int64)
        next_states, rew_vec, done_vec, _ = vec_env.step(action_arr, discrete=True)
        for i in range(num_envs):
            buffer.append((states[i], int(action_arr[i]), float(rew_vec[i]), next_states[i], float(done_vec[i])))
            ep_rewards[i] += float(rew_vec[i])
            ep_lengths[i] += 1
            if done_vec[i]:
                finished_reward = float(ep_rewards[i])
                finished_steps = int(ep_lengths[i])
                all_ep_rewards.append(finished_reward)
                writer.add_scalar("train/episode_reward", finished_reward, len(all_ep_rewards))
                writer.add_scalar("train/episode_steps", finished_steps, len(all_ep_rewards))
                _log(
                    f"[mm:dqn] episode {len(all_ep_rewards)} reward={finished_reward:.4f} steps={finished_steps} "
                    f"epsilon={eps:.4f} buffer={len(buffer)} behavior_warmup={use_behavior}"
                )
                append_working_log(
                    "DQN_MM_v1",
                    "EPISODE",
                    {
                        "train_episode_reward": finished_reward,
                        "episode_steps": finished_steps,
                        "epsilon": eps,
                        "val_status": "pending",
                        "test_status": "pending",
                        "behavior_warmup": use_behavior,
                    },
                    epoch=len(all_ep_rewards),
                )
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0
        states = next_states
        eps = max(eps_end, eps * eps_decay)

        if len(buffer) < batch_size:
            continue

        batch = random.sample(buffer, batch_size)
        s_b, a_b, r_b, ns_b, d_b = zip(*batch)
        s_b = torch.tensor(np.array(s_b), dtype=torch.float32, device=device)
        a_b = torch.tensor(np.array(a_b), dtype=torch.long, device=device)
        r_b = torch.tensor(np.array(r_b), dtype=torch.float32, device=device)
        ns_b = torch.tensor(np.array(ns_b), dtype=torch.float32, device=device)
        d_b = torch.tensor(np.array(d_b), dtype=torch.float32, device=device)

        q_vals = online(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = target(ns_b).max(dim=-1).values
        q_target = r_b + gamma * (1 - d_b) * next_q
        loss = 0.5 * (q_vals - q_target).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), 1.0)
        opt.step()

        if (step + 1) % target_update_freq == 0:
            target.load_state_dict(online.state_dict())

        if (step + 1) % 1000 == 0 and len(all_ep_rewards) > 0:
            val_rew = _rollout_eval(online, val_prices, cfg, device, discrete=True, is_sac=False)
            writer.add_scalar("val/mean_reward", val_rew, step)
            _log(f"[mm:dqn] step {step + 1}/{int(cfg['max_steps'])} val_mean_reward={val_rew:.4f} epsilon={eps:.4f}")
            append_working_log(
                "DQN_MM_v1",
                "VALIDATION",
                {
                    "step": step + 1,
                    "val_mean_reward": val_rew,
                    "epsilon": eps,
                    "test_status": "pending",
                },
            )
            if val_rew > best_val_rew:
                best_val_rew = val_rew
                torch.save(online.state_dict(), ckpt_dir / "DQN_MM_v1_best.pt")
                _log(f"[mm:dqn] checkpoint saved val_mean_reward={val_rew:.4f}")

    writer.close()
    rewards_arr = np.array(all_ep_rewards) if all_ep_rewards else np.zeros(1)
    eval_rewards = _rollout_eval_rewards(online, test_prices, cfg, device, discrete=True, is_sac=False)
    eval_mean = float(np.mean(eval_rewards))
    eval_sharpe = compute_sharpe(np.diff(eval_rewards) if len(eval_rewards) > 1 else eval_rewards)
    eval_max_dd = compute_max_drawdown(eval_rewards)
    append_working_log(
        "DQN_MM_v1",
        "FINAL",
        {
            "train_mean_reward": float(np.mean(rewards_arr)),
            "val_best_mean_reward": float(best_val_rew),
            "test_mean_reward": eval_mean,
            "test_sharpe": eval_sharpe,
            "test_max_drawdown": eval_max_dd,
        },
    )
    return MMResult(
        model_name="DQN_MM_v1",
        checkpoint_dir=str(ckpt_dir),
        mean_episode_reward=float(np.mean(rewards_arr)),
        std_episode_reward=float(np.std(rewards_arr)),
        sharpe_rewards=compute_sharpe(np.diff(rewards_arr) if len(rewards_arr) > 1 else rewards_arr),
        max_drawdown_rewards=compute_max_drawdown(rewards_arr),
        eval_mean_reward=eval_mean,
        eval_sharpe_rewards=eval_sharpe,
        eval_max_drawdown_rewards=eval_max_dd,
        is_valid=_check_result(eval_mean, float(np.std(rewards_arr)), eval_sharpe, eval_max_dd),
        backend=backend,
        num_episodes=len(all_ep_rewards),
    )


def write_mm_registry(results: list[MMResult], registry_path: Path) -> None:
    entries = [
        {
            "architecture_name": r.model_name,
            "archetype": "market_making_rl",
            "weights_path": f"{r.checkpoint_dir}/{r.model_name}_best.pt",
            "design_premise": "Reinforcement learning for inventory-aware market making.",
            "standard_interface": {"outputs": ["action_or_q_values"]},
            "validation": {
                "mean_episode_reward": r.mean_episode_reward,
                "std_episode_reward": r.std_episode_reward,
                "sharpe_rewards": r.sharpe_rewards,
                "max_drawdown_rewards": r.max_drawdown_rewards,
                "eval_mean_reward": r.eval_mean_reward,
                "eval_sharpe_rewards": r.eval_sharpe_rewards,
                "eval_max_drawdown_rewards": r.eval_max_drawdown_rewards,
                "backend": r.backend,
                "num_episodes": r.num_episodes,
                "is_valid": r.is_valid,
            },
        }
        for r in results
    ]
    append_registry(entries, registry_path)
