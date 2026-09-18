"""Feature-effect and importance techniques (PDP, ALE, permutation)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.models.scorer import CreditScorer

ProbaFn = Callable[[pd.DataFrame], np.ndarray]


def _quantile_grid(values: pd.Series, n_points: int) -> np.ndarray:
    """Distinct quantiles of a variable (robust grid for effect curves)."""
    return np.unique(np.nanquantile(values, np.linspace(0, 1, n_points)))


def partial_dependence_curve(
    proba_fn: ProbaFn, matrix: pd.DataFrame, feature: str, n_points: int
) -> pd.Series:
    """Partial dependence: mean prediction with ``feature`` forced to x.

    Args:
        proba_fn: Maps a feature matrix to default probabilities.
        matrix: Rows over which the prediction is averaged.
        feature: Feature to vary.
        n_points: Maximum number of grid points (quantiles).

    Returns:
        Series of average probability indexed by grid value.
    """
    grid = _quantile_grid(matrix[feature], n_points)
    values = []
    for point in grid:
        modified = matrix.copy()
        modified[feature] = point
        values.append(float(np.mean(proba_fn(modified))))
    return pd.Series(values, index=grid, name="partial_dependence")


def accumulated_local_effects(
    proba_fn: ProbaFn, matrix: pd.DataFrame, feature: str, n_bins: int
) -> pd.Series:
    """First-order ALE (Apley & Zhu), centred to mean zero.

    Unlike the PDP, ALE only evaluates the model on clients that actually
    fall in each interval, so it is not fooled by correlated features
    (e.g. the three delinquency counters).

    Args:
        proba_fn: Maps a feature matrix to default probabilities.
        matrix: Data used to estimate the effects.
        feature: Feature analysed.
        n_bins: Number of quantile intervals.

    Returns:
        Series of centred accumulated effect indexed by interval centre.
    """
    x = matrix[feature].to_numpy()
    edges = _quantile_grid(matrix[feature], n_bins + 1)
    centres, effects, weights = [], [], []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        in_bin = (x >= low) & ((x <= high) if last else (x < high))
        if not in_bin.any():
            continue
        lower, upper = matrix[in_bin].copy(), matrix[in_bin].copy()
        lower[feature], upper[feature] = low, high
        effects.append(float(np.mean(proba_fn(upper) - proba_fn(lower))))
        centres.append((low + high) / 2.0)
        weights.append(int(in_bin.sum()))
    if not effects:
        return pd.Series(dtype=float, name="ale")
    ale = np.cumsum(effects)
    ale -= np.average(ale, weights=weights)
    return pd.Series(ale, index=centres, name="ale")


def permutation_auc_importance(
    scorer: CreditScorer,
    matrix: pd.DataFrame,
    target: np.ndarray,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Drop in AUC when each feature is shuffled (on unseen data).

    Args:
        scorer: Fitted scorer (its estimator is used directly).
        matrix: Validation feature matrix.
        target: Validation labels.
        n_repeats: Shuffles per feature.
        seed: Random seed.

    Returns:
        DataFrame indexed by feature with ``mean`` and ``std`` of the AUC
        drop, sorted by importance, plus the raw repeats.
    """
    result = permutation_importance(
        scorer.estimator, matrix, target, scoring="roc_auc",
        n_repeats=n_repeats, random_state=seed, n_jobs=-1,
    )
    table = pd.DataFrame(
        result.importances, index=matrix.columns,
        columns=[f"repeat_{i}" for i in range(n_repeats)],
    )
    table.insert(0, "std", result.importances_std)
    table.insert(0, "mean", result.importances_mean)
    return table.sort_values("mean", ascending=False)
