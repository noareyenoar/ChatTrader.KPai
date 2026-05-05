from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    prediction: torch.Tensor
    confidence: torch.Tensor


class TrendModelInterface(nn.Module):
    """Standardized interface for prediction + confidence outputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict_with_confidence(self, x: torch.Tensor) -> ModelOutput:
        raw = self.forward(x).squeeze(-1)
        pred = torch.tanh(raw)
        conf = torch.sigmoid(raw.abs())
        return ModelOutput(prediction=pred, confidence=conf)
