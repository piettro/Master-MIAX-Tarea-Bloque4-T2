"""LIME local explanation, used to cross-check SHAP on one client."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

CLASS_NAMES = ["pays", "defaults"]


def explain_with_lime(
    proba_fn: Callable[[pd.DataFrame], np.ndarray],
    background: pd.DataFrame,
    instance: pd.Series,
    num_features: int,
    seed: int,
) -> pd.DataFrame:
    """Local linear explanation of one client's default probability.

    LIME cannot perturb missing values, so NaNs are filled with the
    background medians; the model itself still receives valid inputs.

    Args:
        proba_fn: Maps a feature matrix to default probabilities.
        background: Training feature matrix (defines perturbations).
        instance: Client to explain (same columns as ``background``).
        num_features: Number of conditions reported.
        seed: Random seed of the perturbations.

    Returns:
        DataFrame with the LIME ``condition`` and its ``weight`` (positive
        pushes towards default).

    Raises:
        ValueError: If ``instance`` and ``background`` columns differ.
    """
    columns = list(background.columns)
    if list(instance.index) != columns:
        raise ValueError("instance and background must share columns")
    medians = background.median()
    filled = background.fillna(medians)

    def predict(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=columns)
        positive = proba_fn(frame)
        return np.column_stack([1.0 - positive, positive])

    explainer = LimeTabularExplainer(
        filled.to_numpy(),
        feature_names=columns,
        class_names=CLASS_NAMES,
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )
    explanation = explainer.explain_instance(
        instance.fillna(medians).to_numpy(), predict,
        num_features=num_features,
    )
    return pd.DataFrame(explanation.as_list(),
                        columns=["condition", "weight"])
