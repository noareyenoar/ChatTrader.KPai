"""Mean Reversion model architectures — Phase 4.

Three architectures targeting tabular input [Batch, Num_Features]:

  MR_MLP_v1        — Deep MLP with Mish activations + BatchNorm.
  MR_ResNet_v1     — Tabular ResNet with residual skip connections.
  MR_GRN_v1        — Gated Residual Network: feature-gated skip-adds
                     that mimic LightGBM-style feature selection in NN form.

All share `TrendModelInterface` for predict_with_confidence().
Target: binary logit (1 = price will move up / revert up, 0 = down).
Loss:   BCEWithLogitsLoss.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .interfaces import TrendModelInterface


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _MishBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.fc(x))))


class _ResBlock(nn.Module):
    """Tabular residual block: two linear layers + skip connection."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act1(self.fc1(h))
        h = self.drop(h)
        h = self.act2(self.fc2(h))
        return h + x


class _GRNBlock(nn.Module):
    """Gated Residual Network block (Lim et al., TFT paper).

    Applies a gating mechanism that learns which features to pass
    through vs. suppress — analogous to feature importance in GBDT.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc_main = nn.Linear(dim, dim)
        self.fc_gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc_main(x))
        gate = torch.sigmoid(self.fc_gate(x))
        return self.norm(x + self.drop(h * gate))


# ---------------------------------------------------------------------------
# Model 1: Deep MLP
# ---------------------------------------------------------------------------

class MeanReversionMLP(TrendModelInterface):
    """Deep MLP with Mish + BatchNorm for tabular mean-reversion features.

    Architecture:
        Input → Linear(input_dim→256) → [MishBlock×4] → Linear(→1)

    Best suited for: fast-reverting assets with clear statistical overextension.
    """
    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = [_MishBlock(input_dim, hidden_size, dropout)]
        for _ in range(num_layers - 1):
            layers.append(_MishBlock(hidden_size, hidden_size, dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Model 2: Tabular ResNet
# ---------------------------------------------------------------------------

class MeanReversionResNet(TrendModelInterface):
    """Tabular ResNet — residual skip connections prevent vanishing gradients
    in deep tabular models.

    Architecture:
        Input → Linear(→256) → [ResBlock×depth] → LayerNorm → Linear(→1)

    Best suited for: complex non-linear reversal patterns across many features.
    """
    def __init__(self, input_dim: int, hidden_size: int = 256, depth: int = 6, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[_ResBlock(hidden_size, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.blocks(self.stem(x))))


# ---------------------------------------------------------------------------
# Model 3: Gated Residual Network (Hybrid)
# ---------------------------------------------------------------------------

class MeanReversionGRN(TrendModelInterface):
    """Gated Residual Network — learns feature importance gates, approximating
    the selective feature behavior of gradient boosting (LightGBM/XGBoost)
    while remaining fully differentiable end-to-end.

    Architecture:
        Input → Linear(→128) → [GRNBlock×depth] → LayerNorm → Linear(→1)

    Best suited for: datasets with many weakly predictive features where
    automatic feature selection improves signal-to-noise ratio.
    """
    def __init__(self, input_dim: int, hidden_size: int = 128, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_size)
        self.stem_norm = nn.LayerNorm(hidden_size)
        self.stem_act = nn.GELU()
        self.blocks = nn.Sequential(*[_GRNBlock(hidden_size, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem_act(self.stem_norm(self.stem(x)))
        return self.head(self.norm(self.blocks(h)))
