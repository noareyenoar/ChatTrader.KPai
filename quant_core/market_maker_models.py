"""Market Making RL model architectures — Phase 4.

Three RL policy architectures for inventory-aware market making:

  MM_PPO_v1  — PPO Actor-Critic with continuous action space.
  MM_SAC_v1  — Soft Actor-Critic with entropy regularization.
  MM_DQN_v1  — Deep Q-Network with discrete spread-level actions.

State dim:   7  (from MarketMakingEnv.STATE_DIM)
Cont. actions: 2  (bid_offset, ask_offset)
Disc. actions: 3  (tight/medium/wide)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


# ---------------------------------------------------------------------------
# Model 1: PPO Actor-Critic (continuous action)
# ---------------------------------------------------------------------------

class PPOActorCritic(nn.Module):
    """Proximal Policy Optimization Actor-Critic network.

    Shared trunk → split into actor (policy) and critic (value) heads.
    Actor outputs mean + log_std of a diagonal Gaussian for continuous
    bid/ask offset actions.
    Critic outputs scalar state value V(s) for advantage estimation.

    Architecture:
        State (B,7) → Shared MLP(→256) → Actor head (→2×2 for mean+logstd)
                                        → Critic head (→1)

    Best suited for: smooth policy gradient updates; handles continuous
    quote placement without discretization artifacts.
    """
    def __init__(self, state_dim: int = 7, action_dim: int = 2, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        mean = torch.sigmoid(self.actor_mean(h))   # (0,1) — normalized offset
        mean = torch.nan_to_num(mean, nan=0.5, posinf=0.999, neginf=0.001)
        log_std = self.actor_log_std.clamp(-4, 0)
        log_std = torch.nan_to_num(log_std, nan=-2.0, posinf=0.0, neginf=-4.0)
        value = self.critic(h).squeeze(-1)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return mean, log_std, value

    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std, _ = self.forward(state)
        std = log_std.exp().clamp(min=1e-3, max=1.0)
        dist = torch.distributions.Normal(mean, std, validate_args=False)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        action = torch.nan_to_num(action, nan=0.5, posinf=1.0, neginf=0.0)
        log_prob = torch.nan_to_num(log_prob, nan=-10.0, posinf=10.0, neginf=-10.0)
        return action.clamp(0.0, 1.0), log_prob


# ---------------------------------------------------------------------------
# Model 2: Soft Actor-Critic (entropy-regularized, continuous)
# ---------------------------------------------------------------------------

class SACActorNetwork(nn.Module):
    """SAC stochastic actor: outputs Gaussian mean + log_std."""
    def __init__(self, state_dim: int = 7, action_dim: int = 2, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.net = _mlp(state_dim, hidden, action_dim * 2, dropout)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(state)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-4, 2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = log_std.exp().clamp(min=1e-3, max=2.5)
        dist = torch.distributions.Normal(mean, std, validate_args=False)
        z = dist.rsample()
        action = torch.sigmoid(z)             # squash to (0,1)
        log_prob = dist.log_prob(z) - torch.log(action * (1 - action) + 1e-6)
        action = torch.nan_to_num(action, nan=0.5, posinf=1.0, neginf=0.0)
        return action, torch.nan_to_num(log_prob.sum(-1), nan=-10.0, posinf=10.0, neginf=-10.0)


class SACCriticNetwork(nn.Module):
    """Twin Q-networks for SAC (reduces overestimation bias)."""
    def __init__(self, state_dim: int = 7, action_dim: int = 2, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.q1 = _mlp(state_dim + action_dim, hidden, 1, dropout)
        self.q2 = _mlp(state_dim + action_dim, hidden, 1, dropout)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


class SACAgentNetworks(nn.Module):
    """Container for all SAC networks (actor + twin critics + targets).

    Wraps actor and critics in a single module for checkpointing.
    The temperature α (entropy coefficient) is a learnable parameter.
    """
    def __init__(self, state_dim: int = 7, action_dim: int = 2, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.actor = SACActorNetwork(state_dim, action_dim, hidden, dropout)
        self.critic = SACCriticNetwork(state_dim, action_dim, hidden, dropout)
        self.critic_target = SACCriticNetwork(state_dim, action_dim, hidden, dropout)
        # Copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()


# ---------------------------------------------------------------------------
# Model 3: Deep Q-Network (discrete spread levels)
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    """Dueling DQN for discrete market-making actions (tight/medium/wide).

    Dueling architecture separates state value V(s) from advantage A(s,a),
    improving Q-value estimation for actions with similar outcomes.

    Architecture:
        State (B,7) → Shared MLP(→256)
                     → Value stream  → V(s) scalar
                     → Advantage stream → A(s,a) per action
                     → Q(s,a) = V(s) + A(s,a) − mean(A)

    Best suited for: discrete action spaces where tight/medium/wide spread
    levels are sufficient granularity.
    """
    def __init__(self, state_dim: int = 7, num_actions: int = 3, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.advantage_head = nn.Linear(hidden, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.trunk(state)
        V = self.value_head(h)
        A = self.advantage_head(h)
        return V + (A - A.mean(dim=-1, keepdim=True))   # Q-values
