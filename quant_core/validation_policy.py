from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProductionGates:
    """Single source of truth for production validation gates."""

    sharpe_min: float = 1.2
    directional_accuracy_min: float = 0.55
    profit_factor_min: float = 1.5
    max_drawdown_max: float = 0.20


PRODUCTION_GATES = ProductionGates()


def passes_production_gates(metrics: dict[str, Any], *, require_directional_accuracy: bool = True) -> bool:
    """Validate a metrics dict against strict production gates.

    Expected keys:
    - sharpe
    - profit_factor
    - max_drawdown (positive fraction, e.g. 0.12 for 12%)
    - directional_accuracy (required only when require_directional_accuracy=True)
    """
    sharpe = metrics.get("sharpe")
    pf = metrics.get("profit_factor")
    mdd = metrics.get("max_drawdown")
    acc = metrics.get("directional_accuracy")

    if sharpe is None or pf is None or mdd is None:
        return False

    if require_directional_accuracy and acc is None:
        return False

    if sharpe < PRODUCTION_GATES.sharpe_min:
        return False
    if pf < PRODUCTION_GATES.profit_factor_min:
        return False
    if mdd > PRODUCTION_GATES.max_drawdown_max:
        return False
    if require_directional_accuracy and acc < PRODUCTION_GATES.directional_accuracy_min:
        return False

    return True
