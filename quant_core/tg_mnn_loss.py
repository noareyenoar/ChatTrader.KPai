"""Custom loss functions for TG-MNN multi-task training.

Implements:
1. Categorical Cross-Entropy for state classification
2. Huber Loss for magnitude/duration regression (robust to outliers)
3. Joint MultiTaskLoss with task weighting
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Combines classification and regression losses with learnable task weighting.

    The model predicts:
    1. State: Discrete classification (3 classes: Steady, Up, Down)
    2. Magnitude: Continuous regression (distance to next peak/trough)
    3. Duration: Continuous regression (bars until next peak/trough)

    Loss structure:
    L_total = α * L_class + β * L_magnitude + γ * L_duration

    where α, β, γ are learnable or fixed weights to balance task contributions.
    """

    def __init__(
        self,
        state_weight: float = 1.0,
        magnitude_weight: float = 0.5,
        duration_weight: float = 0.5,
        regression_loss: str = "huber",
        huber_delta: float = 1.0,
        learnable_weights: bool = False,
    ):
        """
        Args:
            state_weight: Initial weight for classification loss
            magnitude_weight: Initial weight for magnitude loss
            duration_weight: Initial weight for duration loss
            regression_loss: "huber" or "mse"
            huber_delta: Delta parameter for Huber loss (robustness threshold)
            learnable_weights: If True, task weights become learnable parameters
        """
        super().__init__()
        self.state_weight = state_weight
        self.magnitude_weight = magnitude_weight
        self.duration_weight = duration_weight
        self.regression_loss_type = regression_loss
        self.huber_delta = huber_delta

        if learnable_weights:
            # Learnable task weights (soft attention over tasks)
            self.log_weight_state = nn.Parameter(torch.tensor(0.0))      # log(state_weight)
            self.log_weight_mag = nn.Parameter(torch.tensor(-0.69))      # log(magnitude_weight)
            self.log_weight_dur = nn.Parameter(torch.tensor(-0.69))      # log(duration_weight)
        else:
            self.log_weight_state = None

        # Classification loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        # Regression losses
        # NOTE: nn.HuberLoss uses aten::huber_loss which is NOT supported on DirectML
        # and falls back to CPU, causing ~5x slowdown. Use SmoothL1Loss instead:
        # SmoothL1Loss(beta=delta) is numerically equivalent to HuberLoss(delta=delta/beta)
        # and aten::smooth_l1_loss IS DML-native.
        if regression_loss == "huber":
            self.regression_loss_fn = nn.SmoothL1Loss(beta=huber_delta, reduction='mean')
        elif regression_loss == "mse":
            self.regression_loss_fn = nn.MSELoss(reduction='mean')
        else:
            raise ValueError(f"Unknown regression loss: {regression_loss}")

    def forward(
        self,
        state_logits: torch.Tensor,
        magnitude_pred: torch.Tensor,
        duration_pred: torch.Tensor,
        target_state: torch.Tensor,
        target_magnitude: torch.Tensor,
        target_duration: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            state_logits: [B, 3] - logits for 3 states
            magnitude_pred: [B] - predicted magnitude
            duration_pred: [B] - predicted duration
            target_state: [B] - target state labels {0, 1, 2}
            target_magnitude: [B] - target magnitude
            target_duration: [B] - target duration

        Returns:
            total_loss: Scalar loss
            metrics: Dict with individual loss components
        """
        # Classification loss
        l_state = self.ce_loss(state_logits, target_state)

        # Regression losses
        l_magnitude = self.regression_loss_fn(magnitude_pred, target_magnitude)
        l_duration = self.regression_loss_fn(duration_pred, target_duration)

        # Get task weights
        if self.log_weight_state is not None:
            # Learnable weights via softplus of log-parameters
            w_state = torch.exp(self.log_weight_state)
            w_mag = torch.exp(self.log_weight_mag)
            w_dur = torch.exp(self.log_weight_dur)
        else:
            # Fixed weights
            w_state = torch.tensor(self.state_weight, device=state_logits.device)
            w_mag = torch.tensor(self.magnitude_weight, device=state_logits.device)
            w_dur = torch.tensor(self.duration_weight, device=state_logits.device)

        # Total loss
        l_total = w_state * l_state + w_mag * l_magnitude + w_dur * l_duration

        metrics = {
            'loss_state': l_state.item(),
            'loss_magnitude': l_magnitude.item(),
            'loss_duration': l_duration.item(),
            'loss_total': l_total.item(),
            'weight_state': w_state.item() if isinstance(w_state, torch.Tensor) else w_state,
            'weight_magnitude': w_mag.item() if isinstance(w_mag, torch.Tensor) else w_mag,
            'weight_duration': w_dur.item() if isinstance(w_dur, torch.Tensor) else w_dur,
        }

        return l_total, metrics


class RobustStateAndRegression(nn.Module):
    """Alternative loss combining focal loss for imbalanced classification with quantile loss."""

    def __init__(
        self,
        state_weight: float = 1.0,
        magnitude_weight: float = 0.5,
        duration_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        """
        Args:
            focal_gamma: Focusing parameter for focal loss (higher = focus on hard examples)
            focal_alpha: Weighting factor for rare classes
        """
        super().__init__()
        self.state_weight = state_weight
        self.magnitude_weight = magnitude_weight
        self.duration_weight = duration_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for classification (handles class imbalance)."""
        p = torch.softmax(logits, dim=1)
        ce = F.cross_entropy(logits, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.focal_gamma
        return (self.focal_alpha * focal_weight * ce).mean()

    def quantile_loss(
        self, pred: torch.Tensor, target: torch.Tensor, q: float = 0.5
    ) -> torch.Tensor:
        """Quantile loss for robust regression."""
        error = target - pred
        return torch.mean(torch.max((q - 1) * error, q * error))

    def forward(
        self,
        state_logits: torch.Tensor,
        magnitude_pred: torch.Tensor,
        duration_pred: torch.Tensor,
        target_state: torch.Tensor,
        target_magnitude: torch.Tensor,
        target_duration: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute robust multi-task loss."""
        l_state = self.focal_loss(state_logits, target_state)
        l_magnitude = self.quantile_loss(magnitude_pred, target_magnitude, q=0.5)
        l_duration = self.quantile_loss(duration_pred, target_duration, q=0.5)

        l_total = (
            self.state_weight * l_state +
            self.magnitude_weight * l_magnitude +
            self.duration_weight * l_duration
        )

        metrics = {
            'loss_state': l_state.item(),
            'loss_magnitude': l_magnitude.item(),
            'loss_duration': l_duration.item(),
            'loss_total': l_total.item(),
        }

        return l_total, metrics
