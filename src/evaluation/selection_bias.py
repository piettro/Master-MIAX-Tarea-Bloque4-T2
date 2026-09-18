"""Simulation of sample-selection bias and reject inference (parceling).

The historical file only contains clients who were granted credit in the
past.  This module makes that bias measurable with a controlled
simulation: an earlier model "rejects" the riskiest clients, a model is
trained on accepted clients only, and parceling reject inference is used
to try to recover the lost information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.features import build_feature_matrix
from src.models.supervised import ModelSpec, fit_estimator


def simulate_reject_inference(
    spec: ModelSpec,
    train_clean: pd.DataFrame,
    y_train: np.ndarray,
    valid_clean: pd.DataFrame,
    y_valid: np.ndarray,
    acceptance_share: float,
    seed: int,
) -> pd.Series:
    """Measure the AUC lost by selection bias and regained by parceling.

    Args:
        spec: Model configuration used for all three models.
        train_clean: Cleaned training split.
        y_train: Training labels.
        valid_clean: Cleaned validation split.
        y_valid: Validation labels.
        acceptance_share: Share of lowest-risk clients deemed "accepted".
        seed: Random seed of the inferred labels.

    Returns:
        Series with the validation AUC of the biased model, the model with
        reject inference and the oracle trained on everyone.

    Raises:
        ValueError: If ``acceptance_share`` is not in ``(0, 1)``.
    """
    if not 0.0 < acceptance_share < 1.0:
        raise ValueError("acceptance_share must be in (0, 1)")
    y_train, y_valid = np.asarray(y_train), np.asarray(y_valid)
    x_train = build_feature_matrix(train_clean, spec.features)
    x_valid = build_feature_matrix(valid_clean, spec.features)

    oracle = fit_estimator(spec, x_train, y_train)
    previous_score = oracle.predict_proba(x_train)[:, 1]
    accepted = previous_score < np.quantile(previous_score, acceptance_share)

    biased = fit_estimator(spec, x_train[accepted], y_train[accepted])
    rejected_proba = biased.predict_proba(x_train[~accepted])[:, 1]
    rng = np.random.default_rng(seed)
    inferred = (rng.random(len(rejected_proba)) < rejected_proba).astype(int)

    augmented_x = pd.concat([x_train[accepted], x_train[~accepted]])
    augmented_y = np.concatenate([y_train[accepted], inferred])
    with_inference = fit_estimator(spec, augmented_x, augmented_y)

    def auc(model: object) -> float:
        return float(roc_auc_score(y_valid,
                                   model.predict_proba(x_valid)[:, 1]))

    return pd.Series({
        "biased_accepted_only": auc(biased),
        "reject_inference_parceling": auc(with_inference),
        "oracle_full_population": auc(oracle),
    }, name="validation_auc")
