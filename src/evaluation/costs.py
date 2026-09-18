"""Misclassification costs and cost-optimal decision thresholds.

Convention: prediction 1 = expected default = deny the credit; a false
positive denies a good client and a false negative grants credit to a
client who defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils import config


@dataclass(frozen=True)
class CostScenario:
    """A pair of misclassification costs defining a business scenario.

    Attributes:
        key: Short identifier (``scenario_1`` / ``scenario_2``).
        label: Human-readable name.
        cost_fp: Cost of denying a client who would have paid.
        cost_fn: Cost of granting credit to a client who defaults.
    """

    key: str
    label: str
    cost_fp: float
    cost_fn: float

    def __post_init__(self) -> None:
        """Validate the costs.

        Raises:
            ValueError: If a cost is negative or both are zero.
        """
        if self.cost_fp < 0 or self.cost_fn < 0:
            raise ValueError("Costs must be non-negative")
        if self.cost_fp + self.cost_fn == 0:
            raise ValueError("At least one cost must be positive")

    @property
    def theoretical_threshold(self) -> float:
        """Bayes-optimal threshold for a calibrated model.

        Denying is cheaper than granting when
        ``p * cost_fn > (1 - p) * cost_fp``, i.e. when
        ``p > cost_fp / (cost_fp + cost_fn)``.
        """
        return self.cost_fp / (self.cost_fp + self.cost_fn)


@dataclass(frozen=True)
class ThresholdResult:
    """Outcome of a threshold optimisation.

    Attributes:
        threshold: Chosen threshold (deny when ``p >= threshold``).
        cost: Mean cost per client at that threshold.
        denial_rate: Share of clients denied at that threshold.
    """

    threshold: float
    cost: float
    denial_rate: float


def build_scenarios() -> dict[str, CostScenario]:
    """Create the two cost scenarios of the assignment brief.

    Returns:
        Mapping from scenario key to :class:`CostScenario`.
    """
    return {
        key: CostScenario(key, config.SCENARIO_LABELS[key], cfp, cfn)
        for key, (cfp, cfn) in config.SCENARIO_COSTS.items()
    }


def _validate(y_true: np.ndarray, values: np.ndarray) -> None:
    """Check that labels and predictions are aligned and binary."""
    if y_true.shape != values.shape:
        raise ValueError(
            f"Shape mismatch: {y_true.shape} vs {values.shape}"
        )
    if y_true.size == 0:
        raise ValueError("Empty input")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0/1 labels")


def decisions_from_proba(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Turn default probabilities into deny (1) / approve (0) decisions.

    Args:
        proba: Predicted default probabilities.
        threshold: Deny when ``proba >= threshold``.

    Returns:
        Integer array of decisions.
    """
    return (np.asarray(proba) >= threshold).astype(int)


def confusion_counts(
    y_true: np.ndarray, decisions: np.ndarray
) -> dict[str, int]:
    """Count TP, FP, TN and FN for binary decisions.

    Args:
        y_true: True labels (1 = default).
        decisions: Decisions (1 = deny).

    Returns:
        Dictionary with keys ``tp``, ``fp``, ``tn`` and ``fn``.

    Raises:
        ValueError: If inputs are misaligned or not binary.
    """
    y_true, decisions = np.asarray(y_true), np.asarray(decisions)
    _validate(y_true, decisions)
    return {
        "tp": int(((decisions == 1) & (y_true == 1)).sum()),
        "fp": int(((decisions == 1) & (y_true == 0)).sum()),
        "tn": int(((decisions == 0) & (y_true == 0)).sum()),
        "fn": int(((decisions == 0) & (y_true == 1)).sum()),
    }


def expected_cost(
    y_true: np.ndarray, decisions: np.ndarray, scenario: CostScenario
) -> float:
    """Mean misclassification cost per client (the graded metric).

    Args:
        y_true: True labels (1 = default).
        decisions: Decisions (1 = deny).
        scenario: Cost scenario.

    Returns:
        ``(cost_fp * FP + cost_fn * FN) / n``.
    """
    counts = confusion_counts(y_true, decisions)
    total = scenario.cost_fp * counts["fp"] + scenario.cost_fn * counts["fn"]
    return float(total / len(np.asarray(y_true)))


def _sorted_cumulative(
    y_true: np.ndarray, proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores sorted descending with cumulative positives / negatives."""
    order = np.argsort(-proba, kind="mergesort")
    sorted_proba = proba[order]
    cum_pos = np.concatenate(([0], np.cumsum(y_true[order])))
    cum_neg = np.arange(len(y_true) + 1) - cum_pos
    return sorted_proba, cum_pos, cum_neg


def cost_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    scenario: CostScenario,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Mean cost for each threshold of a grid (vectorised).

    Args:
        y_true: True labels.
        proba: Predicted default probabilities.
        scenario: Cost scenario.
        thresholds: Grid of thresholds.

    Returns:
        Array of mean costs, aligned with ``thresholds``.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    _validate(y_true, proba)
    ascending = np.sort(proba)
    n_denied = len(proba) - np.searchsorted(ascending, thresholds, "left")
    _, cum_pos, cum_neg = _sorted_cumulative(y_true, proba)
    fp = cum_neg[n_denied]
    fn = cum_pos[-1] - cum_pos[n_denied]
    return (scenario.cost_fp * fp + scenario.cost_fn * fn) / len(y_true)


def optimal_threshold(
    y_true: np.ndarray, proba: np.ndarray, scenario: CostScenario
) -> ThresholdResult:
    """Exact cost-minimising threshold over every distinct score.

    This generalises the professor's sweep over ``np.unique(proba)``: all
    cut points are evaluated in ``O(n log n)``.  When the optimum lies
    between two consecutive distinct scores the midpoint is returned, which
    is the most robust choice for unseen data.

    Args:
        y_true: True labels.
        proba: Predicted default probabilities.
        scenario: Cost scenario.

    Returns:
        The optimal :class:`ThresholdResult`.

    Raises:
        ValueError: If inputs are misaligned or not binary.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    _validate(y_true, proba)
    n = len(y_true)
    sorted_proba, cum_pos, cum_neg = _sorted_cumulative(y_true, proba)
    fp = cum_neg
    fn = cum_pos[-1] - cum_pos
    costs = (scenario.cost_fp * fp + scenario.cost_fn * fn) / n

    # A cut after k denied clients is only valid between distinct scores.
    k = np.arange(n + 1)
    valid = np.ones(n + 1, dtype=bool)
    valid[1:n] = sorted_proba[:-1] > sorted_proba[1:]
    best_k = int(k[valid][np.argmin(costs[valid])])

    if best_k == 0:
        threshold = float(np.nextafter(sorted_proba[0], np.inf))
    elif best_k == n:
        threshold = float(sorted_proba[-1])
    else:
        threshold = float(
            (sorted_proba[best_k - 1] + sorted_proba[best_k]) / 2.0
        )
    return ThresholdResult(threshold, float(costs[best_k]), best_k / n)


def trivial_policy_costs(
    y_true: np.ndarray, scenario: CostScenario
) -> dict[str, float]:
    """Cost of the two trivial policies (approve all / deny all).

    Args:
        y_true: True labels.
        scenario: Cost scenario.

    Returns:
        Dictionary with ``approve_all`` and ``deny_all`` costs.
    """
    default_rate = float(np.mean(y_true))
    return {
        "approve_all": scenario.cost_fn * default_rate,
        "deny_all": scenario.cost_fp * (1.0 - default_rate),
    }


def policy_summary(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    scenario: CostScenario,
) -> dict[str, float]:
    """Business summary of a thresholded policy.

    Args:
        y_true: True labels.
        proba: Predicted default probabilities.
        threshold: Decision threshold.
        scenario: Cost scenario.

    Returns:
        Dictionary with cost, denial rate, recall of defaults, precision
        when denying, accuracy and confusion counts.
    """
    decisions = decisions_from_proba(proba, threshold)
    counts = confusion_counts(y_true, decisions)
    positives = counts["tp"] + counts["fn"]
    denied = counts["tp"] + counts["fp"]
    return {
        "threshold": float(threshold),
        "cost": expected_cost(y_true, decisions, scenario),
        "denial_rate": float(decisions.mean()),
        "default_recall": counts["tp"] / positives if positives else np.nan,
        "precision_when_denying": counts["tp"] / denied if denied else np.nan,
        "accuracy": (counts["tp"] + counts["tn"]) / len(decisions),
        **{key: float(value) for key, value in counts.items()},
    }
