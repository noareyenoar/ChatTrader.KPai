"""Phase 6 robustness validation gates.

Walk-Forward Validator and Monte Carlo Stress Tester, applied to a model's
OOS trade-return series produced by evaluate_all_checkpoints.py.

Gates (from master_plan.md Iron Rules):
  - Walk-forward: positive net PnL in >= 80% of sequential windows.
  - Monte Carlo:  1,000 trade-sequence shuffles;
                  95th-percentile worst-case MDD < 20%.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    passed: bool
    pct_positive_windows: float          # fraction of windows with net_pnl > 0
    n_windows: int
    window_pnls: list[float] = field(default_factory=list)
    window_sharpes: list[float] = field(default_factory=list)


@dataclass
class MonteCarloResult:
    passed: bool
    p95_mdd: float
    p99_mdd: float
    median_mdd: float
    mean_mdd: float
    n_shuffles: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray, ann: float = math.sqrt(252 * 24 * 12)) -> float:
    """Annualised Sharpe ratio (5-min bars default annualisation)."""
    std = float(np.std(returns))
    if std < 1e-10 or len(returns) < 2:
        return 0.0
    return float(np.mean(returns) / std * ann)


def _cumulative_mdd(returns: np.ndarray) -> float:
    """Maximum drawdown on compounded equity curve (same formula as evaluator).

    Uses cumprod(1 + r) so equity is bounded in [0, inf) and MDD is a proper
    percentage drawdown in [0, 1].  Clips returns to [-0.99, 1.0] to guard
    against zero or negative equity from extreme individual bars.
    """
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + np.clip(returns, -0.99, 1.0))
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-10)
    return float(np.max(dd))


# ---------------------------------------------------------------------------
# Walk-Forward Validator
# ---------------------------------------------------------------------------

def walk_forward_validate(
    returns: np.ndarray,
    n_windows: int = 7,
    min_window_bars: int = 50,
    gate_pct: float = 0.80,
) -> WalkForwardResult:
    """Split OOS returns into n_windows sequential chunks; test each for positive PnL.

    Args:
        returns:        1-D array of per-bar trade returns from the OOS test period.
        n_windows:      Number of equal-length sequential windows.
        min_window_bars: Minimum bars per window; if too few bars, fall back to
                         fewer windows until the constraint is met.
        gate_pct:       Fraction of windows that must have positive net PnL to pass.

    Returns:
        WalkForwardResult with pass/fail flag and per-window metrics.
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    n = len(returns)

    # Reduce window count if not enough data
    while n_windows > 1 and n // n_windows < min_window_bars:
        n_windows -= 1

    if n_windows < 2 or n < min_window_bars:
        # Insufficient data — single window pass/fail on overall PnL
        net_pnl = float(np.sum(returns))
        passed = net_pnl > 0.0
        return WalkForwardResult(
            passed=passed,
            pct_positive_windows=1.0 if passed else 0.0,
            n_windows=1,
            window_pnls=[net_pnl],
            window_sharpes=[_sharpe(returns)],
        )

    # Compute split indices (equal-length; last window absorbs the remainder)
    base = n // n_windows
    window_pnls: list[float] = []
    window_sharpes: list[float] = []

    for i in range(n_windows):
        start = i * base
        end = (i + 1) * base if i < n_windows - 1 else n
        w = returns[start:end]
        window_pnls.append(float(np.sum(w)))
        window_sharpes.append(_sharpe(w))

    pct_pos = float(np.mean([p > 0.0 for p in window_pnls]))
    passed = pct_pos >= gate_pct

    return WalkForwardResult(
        passed=passed,
        pct_positive_windows=round(pct_pos, 4),
        n_windows=n_windows,
        window_pnls=[round(v, 6) for v in window_pnls],
        window_sharpes=[round(v, 4) for v in window_sharpes],
    )


# ---------------------------------------------------------------------------
# Monte Carlo Stress Tester
# ---------------------------------------------------------------------------

def monte_carlo_stress_test(
    returns: np.ndarray,
    n_shuffles: int = 1000,
    p95_mdd_gate: float = 0.20,
    seed: int = 42,
) -> MonteCarloResult:
    """Shuffle the trade-return sequence 1000 times and measure worst-case MDD.

    Rationale: shuffling breaks temporal order, probing for fragility that
    arises from specific luck-dependent return sequences rather than a robust
    edge.  A strategy is robust if even the worst orderings keep MDD < 20%.

    Args:
        returns:        1-D array of per-bar trade returns.
        n_shuffles:     Number of random permutations to evaluate.
        p95_mdd_gate:   95th-percentile MDD must be below this threshold to pass.
        seed:           RNG seed for reproducibility.

    Returns:
        MonteCarloResult with pass/fail and MDD distribution summary.
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()

    if len(returns) < 10:
        # Insufficient data — cannot stress test
        return MonteCarloResult(
            passed=False,
            p95_mdd=1.0,
            p99_mdd=1.0,
            median_mdd=1.0,
            mean_mdd=1.0,
            n_shuffles=0,
        )

    rng = np.random.default_rng(seed)
    mdd_samples: list[float] = []
    work = returns.copy()

    for _ in range(n_shuffles):
        rng.shuffle(work)
        mdd_samples.append(_cumulative_mdd(work))

    mdds = np.array(mdd_samples, dtype=np.float64)
    p95 = float(np.percentile(mdds, 95))
    p99 = float(np.percentile(mdds, 99))
    median = float(np.median(mdds))
    mean = float(np.mean(mdds))

    return MonteCarloResult(
        passed=p95 < p95_mdd_gate,
        p95_mdd=round(p95, 4),
        p99_mdd=round(p99, 4),
        median_mdd=round(median, 4),
        mean_mdd=round(mean, 4),
        n_shuffles=n_shuffles,
    )


# ---------------------------------------------------------------------------
# Convenience combined runner
# ---------------------------------------------------------------------------

def run_robustness_suite(
    returns: np.ndarray,
    wf_windows: int = 7,
    mc_shuffles: int = 1000,
    wf_gate_pct: float = 0.80,
    mc_p95_mdd_gate: float = 0.20,
    seed: int = 42,
) -> dict:
    """Run both walk-forward and Monte Carlo gates and return a summary dict.

    Suitable for direct serialisation into the backtest report / registry.
    """
    wf = walk_forward_validate(returns, n_windows=wf_windows, gate_pct=wf_gate_pct)
    mc = monte_carlo_stress_test(returns, n_shuffles=mc_shuffles,
                                  p95_mdd_gate=mc_p95_mdd_gate, seed=seed)

    robustness_pass = wf.passed and mc.passed

    return {
        "robustness_pass": robustness_pass,
        "walk_forward": {
            "passed": wf.passed,
            "pct_positive_windows": wf.pct_positive_windows,
            "n_windows": wf.n_windows,
            "gate": wf_gate_pct,
            "window_pnls": wf.window_pnls,
            "window_sharpes": wf.window_sharpes,
        },
        "monte_carlo": {
            "passed": mc.passed,
            "p95_mdd": mc.p95_mdd,
            "p99_mdd": mc.p99_mdd,
            "median_mdd": mc.median_mdd,
            "mean_mdd": mc.mean_mdd,
            "n_shuffles": mc.n_shuffles,
            "gate_p95_mdd": mc_p95_mdd_gate,
        },
    }
