"""Discrimination, calibration and statistical-uncertainty metrics."""
from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.evaluation.costs import CostScenario, cost_curve
from src.utils import config


def majority_class_accuracy(y_true: np.ndarray) -> float:
    """Accuracy of always predicting the majority class.

    The professor insisted on computing this first: any accuracy below it
    is worthless.  With ~7% defaults it is ~93%.

    Args:
        y_true: Binary labels.

    Returns:
        Share of the most frequent class.
    """
    rate = float(np.mean(y_true))
    return max(rate, 1.0 - rate)


def expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = config.CALIBRATION_BINS,
) -> float:
    """Expected Calibration Error with equal-width bins.

    Args:
        y_true: Binary labels.
        proba: Predicted probabilities.
        n_bins: Number of bins in ``[0, 1]``.

    Returns:
        Occupancy-weighted mean of ``|observed rate - mean prediction|``.

    Raises:
        ValueError: If ``n_bins`` < 1.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    y_true, proba = np.asarray(y_true, float), np.asarray(proba, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = idx == b
        if in_bin.any():
            gap = abs(y_true[in_bin].mean() - proba[in_bin].mean())
            ece += in_bin.mean() * gap
    return float(ece)


def probability_metrics(
    y_true: np.ndarray, proba: np.ndarray
) -> dict[str, float]:
    """AUC, average precision, Brier score and ECE of a score.

    Args:
        y_true: Binary labels.
        proba: Predicted default probabilities.

    Returns:
        Dictionary with ``auc``, ``average_precision``, ``brier``, ``ece``.
    """
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "ece": expected_calibration_error(y_true, proba),
    }


def _midrank(values: np.ndarray) -> np.ndarray:
    """Mid-ranks (ties share their average rank), 1-based."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = len(values)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n)
    out[order] = ranks + 1
    return out


@dataclass(frozen=True)
class DeLongResult:
    """Result of DeLong's test for two correlated ROC AUCs.

    Attributes:
        auc_a: AUC of the first score.
        auc_b: AUC of the second score.
        difference: ``auc_a - auc_b``.
        z_score: Test statistic.
        p_value: Two-sided p-value.
        ci_low: Lower bound of the confidence interval of the difference.
        ci_high: Upper bound of the confidence interval of the difference.
    """

    auc_a: float
    auc_b: float
    difference: float
    z_score: float
    p_value: float
    ci_low: float
    ci_high: float


def delong_test(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    confidence: float = config.CONFIDENCE_LEVEL,
) -> DeLongResult:
    """DeLong's test comparing two AUCs on the same sample (fast version).

    Implements Sun & Xu (2014).  The two scores are evaluated on the same
    clients, so their AUCs are correlated and a naive comparison would be
    wrong.

    Args:
        y_true: Binary labels.
        proba_a: Scores of model A.
        proba_b: Scores of model B.
        confidence: Confidence level of the interval.

    Returns:
        A :class:`DeLongResult`.

    Raises:
        ValueError: If there are no positives or no negatives.
    """
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("DeLong's test needs both classes")
    order = np.argsort(-y_true, kind="mergesort")  # positives first
    scores = np.vstack([np.asarray(proba_a), np.asarray(proba_b)])[:, order]

    pos, neg = scores[:, :n_pos], scores[:, n_pos:]
    tx = np.array([_midrank(row) for row in pos])
    ty = np.array([_midrank(row) for row in neg])
    tz = np.array([_midrank(row) for row in scores])
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg
    aucs -= (n_pos + 1.0) / 2.0 / n_neg
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    cov = np.cov(v01) / n_pos + np.cov(v10) / n_neg

    diff = float(aucs[0] - aucs[1])
    variance = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    std = sqrt(variance) if variance > 0 else 0.0
    z_score = diff / std if std > 0 else 0.0
    p_value = erfc(abs(z_score) / sqrt(2.0))
    z_crit = float(norm.ppf(0.5 + confidence / 2.0))
    return DeLongResult(
        float(aucs[0]), float(aucs[1]), diff, z_score, p_value,
        diff - z_crit * std, diff + z_crit * std,
    )


def bootstrap_cost(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    scenario: CostScenario,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap distribution of the mean cost at a fixed threshold.

    Args:
        y_true: Binary labels.
        proba: Predicted default probabilities.
        threshold: Decision threshold.
        scenario: Cost scenario.
        n_samples: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Array of ``n_samples`` resampled mean costs.

    Raises:
        ValueError: If ``n_samples`` < 1.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    y_true = np.asarray(y_true)
    decisions = np.asarray(proba) >= threshold
    per_client = np.where(
        decisions & (y_true == 0), scenario.cost_fp, 0.0
    ) + np.where(~decisions & (y_true == 1), scenario.cost_fn, 0.0)
    rng = np.random.default_rng(seed)
    n = len(per_client)
    return np.array([
        per_client[rng.integers(0, n, n)].mean() for _ in range(n_samples)
    ])


def confidence_interval(
    samples: np.ndarray, confidence: float = config.CONFIDENCE_LEVEL
) -> tuple[float, float]:
    """Percentile confidence interval of a bootstrap distribution.

    Args:
        samples: Bootstrap statistics.
        confidence: Confidence level.

    Returns:
        Tuple ``(low, high)``.
    """
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)


def lift_table(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = config.LIFT_BINS
) -> pd.DataFrame:
    """Lift and cumulative capture by score decile (riskiest first).

    Args:
        y_true: Binary labels.
        proba: Predicted default probabilities.
        n_bins: Number of quantile bins.

    Returns:
        DataFrame with default rate, lift and cumulative capture per bin.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "score": proba})
    frame["bin"] = pd.qcut(
        frame["score"].rank(method="first"), n_bins, labels=False
    )
    table = (
        frame.groupby("bin")["y"].agg(["mean", "sum", "count"])
        .iloc[::-1]
        .reset_index(drop=True)
    )
    table.index = table.index + 1
    table.index.name = "decile"
    table["lift"] = table["mean"] / frame["y"].mean()
    table["cumulative_capture"] = table["sum"].cumsum() / table["sum"].sum()
    return table.rename(columns={"mean": "default_rate", "sum": "defaults"})


def cost_curve_frame(
    y_true: np.ndarray,
    proba: np.ndarray,
    scenarios: dict[str, CostScenario],
    grid: np.ndarray,
) -> pd.DataFrame:
    """Cost curves of several scenarios on a common threshold grid.

    Args:
        y_true: Binary labels.
        proba: Predicted default probabilities.
        scenarios: Scenarios keyed by name.
        grid: Threshold grid.

    Returns:
        DataFrame indexed by threshold, one column per scenario, plus the
        denial rate.
    """
    data = {
        key: cost_curve(y_true, proba, scenario, grid)
        for key, scenario in scenarios.items()
    }
    data["denial_rate"] = (np.asarray(proba)[:, None] >= grid).mean(axis=0)
    return pd.DataFrame(data, index=pd.Index(grid, name="threshold"))
