"""TG-MNN: Temporal-Gradient Markov Neural Network

A production-grade multi-task deep learning model that predicts:
1. Current price wave state (Steady, Up, Down) via classification
2. Magnitude to next peak/trough via regression
3. Duration until next peak/trough via regression

Architecture:
- 1D CNN backbone with dilated convolutions for long-range temporal dependencies
- Shared hidden representation after backbone
- Classifier head for state prediction
- Regressor head for magnitude/duration prediction

Design intent:
- Capture temporal correlations and gradient structure efficiently
- Prevent lookahead bias by operating on historical data only
- Multi-task learning to improve feature generalization
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass

from .interfaces import TrendModelInterface


@dataclass
class TGMNNOutput:
    """Multi-task output from TG-MNN."""
    state_logits: torch.Tensor      # [B] or [B, seq_len, 3]
    magnitude_pred: torch.Tensor    # [B] or [B, seq_len]
    duration_pred: torch.Tensor     # [B] or [B, seq_len]
    confidence: torch.Tensor        # [B]


class DilatedConvBlock(nn.Module):
    """Dilated 1D convolution block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, kernel_size: int = 3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            dilation=dilation, padding=padding, bias=True
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.1)

        # Residual projection if dimensions change
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: [B, C, T]"""
        residual = self.residual(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class TGMNNBackbone(nn.Module):
    """1D CNN backbone with dilated convolutions for temporal feature extraction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.blocks.append(
                DilatedConvBlock(hidden_dim, hidden_dim, dilation=dilation, kernel_size=3)
            )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, F] - batch, sequence, features

        Returns:
            pooled: [B, hidden_dim] - global average pooled
            seq_repr: [B, hidden_dim, T] - sequence representation
        """
        B, T, F = x.shape

        # Project features to hidden dimension: [B, T, F] -> [B, T, hidden_dim]
        h = self.input_proj(x)  # [B, T, hidden_dim]

        # Transpose for Conv1d: [B, hidden_dim, T]
        h = h.transpose(1, 2)

        # Apply dilated conv blocks
        for block in self.blocks:
            h = block(h)

        # Global average pooling: [B, hidden_dim, T] -> [B, hidden_dim]
        pooled = self.global_pool(h).squeeze(-1)

        return pooled, h


class StateClassifier(nn.Module):
    """Multi-layer perceptron for classifying wave state (Steady, Up, Down)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3),  # 3 classes: Steady, Up, Down
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, hidden_dim] -> logits: [B, 3]"""
        return self.net(x)


class MagnitudeDurationRegressor(nn.Module):
    """Multi-output regressor for magnitude and duration to next extremum."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
        )
        # Magnitude and duration outputs
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),  # Ensure positive magnitude
        )
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),  # Ensure positive duration
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, hidden_dim]

        Returns:
            magnitude: [B, 1]
            duration: [B, 1]
        """
        h = self.net(x)
        magnitude = self.magnitude_head(h)
        duration = self.duration_head(h)
        return magnitude, duration


class TGMNNModel(TrendModelInterface):
    """Temporal-Gradient Markov Neural Network.

    Multi-task architecture for predicting wave properties:
    1. Classification: Current wave state
    2. Regression: Magnitude and duration to next extremum
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_backbone_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Backbone: Dilated CNN
        self.backbone = TGMNNBackbone(input_dim, hidden_dim, num_backbone_layers)

        # Task-specific heads
        self.state_classifier = StateClassifier(hidden_dim)
        self.mag_dur_regressor = MagnitudeDurationRegressor(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference (trend-like interface compliance).

        Args:
            x: [B, T, F] - batch, sequence, features

        Returns:
            state_logits: [B, 3] - logits for state classification
        """
        pooled, _ = self.backbone(x)
        state_logits = self.state_classifier(pooled)
        return state_logits.max(dim=1).values.unsqueeze(-1)  # Return confidence-like signal

    def forward_multitask(self, x: torch.Tensor) -> TGMNNOutput:
        """
        Multi-task forward pass for training and detailed inference.

        Args:
            x: [B, T, F] - batch, sequence features

        Returns:
            TGMNNOutput with all predictions
        """
        pooled, _ = self.backbone(x)

        # State classification
        state_logits = self.state_classifier(pooled)  # [B, 3]

        # Magnitude and duration regression
        magnitude, duration = self.mag_dur_regressor(pooled)  # [B, 1] each

        # Confidence: softmax probability of predicted state
        state_probs = torch.softmax(state_logits, dim=1)
        max_prob = state_probs.max(dim=1).values

        return TGMNNOutput(
            state_logits=state_logits,
            magnitude_pred=magnitude.squeeze(-1),
            duration_pred=duration.squeeze(-1),
            confidence=max_prob,
        )

    def predict_with_confidence(self, x: torch.Tensor):
        """Comply with TrendModelInterface for ensemble compatibility."""
        output = self.forward_multitask(x)
        from .interfaces import ModelOutput
        return ModelOutput(
            prediction=torch.tanh(output.state_logits.max(dim=1).values),
            confidence=output.confidence,
        )
