"""SHAP attributions of the deployed model (global and local)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap

from src.models.scorer import CreditScorer
from src.utils import config

logger = logging.getLogger(__name__)

BACKGROUND_SIZE = 200


def compute_shap_values(
    scorer: CreditScorer, matrix: pd.DataFrame, seed: int
) -> shap.Explanation:
    """SHAP values of the scorer for a feature matrix.

    Tree ensembles use the exact ``TreeExplainer`` (log-odds scale); any
    other model falls back to the model-agnostic permutation explainer on
    the probability scale.

    Args:
        scorer: Deployed scorer.
        matrix: Rows to explain (model feature space).
        seed: Seed of the background sample for the fallback explainer.

    Returns:
        A ``shap.Explanation`` with one row per explained client.
    """
    try:
        explanation = shap.TreeExplainer(scorer.estimator)(matrix)
        if np.ndim(explanation.values) == 3:  # per-class output (RF)
            explanation = explanation[:, :, 1]
        return explanation
    except Exception as exc:  # shap raises many exception types
        logger.info("TreeExplainer unavailable (%s); using the "
                    "permutation explainer", exc)
    background = matrix.sample(
        min(BACKGROUND_SIZE, len(matrix)), random_state=seed
    )
    explainer = shap.Explainer(
        scorer.proba_from_matrix, shap.maskers.Independent(background),
        seed=seed,
    )
    return explainer(matrix)


def shap_importance(explanation: shap.Explanation) -> pd.Series:
    """Global importance as mean absolute SHAP value.

    Args:
        explanation: SHAP values.

    Returns:
        Series sorted in descending order, indexed by feature.
    """
    values = np.abs(np.asarray(explanation.values)).mean(axis=0)
    return pd.Series(values, index=explanation.feature_names).sort_values(
        ascending=False
    )


def select_local_cases(proba: np.ndarray, threshold: float) -> dict[str, int]:
    """Pick three representative clients to explain locally.

    * ``clear_denial``: highest default probability.
    * ``borderline_denial``: denied, closest to the threshold (the case
      that is hardest to justify).
    * ``clear_approval``: lowest default probability.

    Args:
        proba: Default probabilities of the explained sample.
        threshold: Threshold of the audited policy.

    Returns:
        Mapping from case name to row position.

    Raises:
        ValueError: If ``proba`` is empty.
    """
    proba = np.asarray(proba)
    if proba.size == 0:
        raise ValueError("proba is empty")
    denied = np.flatnonzero(proba >= threshold)
    if denied.size:
        borderline = int(denied[np.argmin(proba[denied] - threshold)])
    else:
        borderline = int(np.argmin(np.abs(proba - threshold)))
    return {
        "clear_denial": int(np.argmax(proba)),
        "borderline_denial": borderline,
        "clear_approval": int(np.argmin(proba)),
    }


def top_risk_drivers(
    explanation: shap.Explanation,
    position: int,
    n_drivers: int = config.N_TOP_FEATURES,
) -> pd.Series:
    """Features pushing one client's score towards default the most.

    Args:
        explanation: SHAP values.
        position: Row of the client in the explanation.
        n_drivers: Maximum number of drivers returned.

    Returns:
        Positive SHAP contributions, largest first.
    """
    row = pd.Series(np.asarray(explanation.values)[position],
                    index=explanation.feature_names)
    return row[row > 0].sort_values(ascending=False).head(n_drivers)
