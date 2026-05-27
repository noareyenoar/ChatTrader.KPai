from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProductionGates:
    """Single source of truth for V2.0 production validation gates.

    Updated per pytorch_model_training_ruleV2.md §2.6 (Walk-Forward OOS only,
    after 0.04% commission + 1-tick slippage):
      - sharpe_min          : 1.0   (was 1.2)
      - profit_factor_min   : 1.3   (was 1.5)
      - max_drawdown_max    : 0.20  (unchanged)
      - directional_accuracy_min : 0.55  (unchanged)
      - sharpe_divergence_max_abs: 2.0   (NEW — absolute Val/Test Sharpe gap)
    """

    sharpe_min: float = 1.0
    directional_accuracy_min: float = 0.55
    profit_factor_min: float = 1.3
    max_drawdown_max: float = 0.20
    # V2.0: If |val_sharpe - test_sharpe| > this threshold, model is overfitting.
    sharpe_divergence_max_abs: float = 2.0


PRODUCTION_GATES = ProductionGates()


def passes_production_gates(
    metrics: dict[str, Any],
    *,
    require_directional_accuracy: bool = True,
    val_sharpe: float | None = None,
) -> bool:
    """Validate a metrics dict against V2.0 production gates.

    Expected keys in ``metrics``:
      - sharpe
      - profit_factor
      - max_drawdown (positive fraction, e.g. 0.12 for 12%)
      - directional_accuracy (required only when require_directional_accuracy=True)

    Optional:
      - ``val_sharpe``: when provided, the absolute divergence gate is enforced.
        If |val_sharpe - metrics['sharpe']| > sharpe_divergence_max_abs the
        model is classified as overfitting and the call returns False.
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

    # V2.0 divergence gate
    if val_sharpe is not None:
        abs_gap = abs(float(val_sharpe) - float(sharpe))
        if abs_gap > PRODUCTION_GATES.sharpe_divergence_max_abs:
            return False

    return True
