"""Catalogue of supervised candidate models.

Every candidate is described by a :class:`ModelSpec` (a builder plus the
feature list it consumes) so that model selection, final training and the
XAI audit all use exactly the same configuration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.utils import config
from src.utils.config import RunSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Recipe of a candidate model.

    Attributes:
        name: Unique identifier.
        description: One-line explanation for reports.
        builder: Zero-argument callable returning an unfitted estimator.
        features: Ordered model inputs (cleaned or derived columns).
        class_weighted: Whether to train with balanced sample weights.
    """

    name: str
    description: str
    builder: Callable[[], BaseEstimator]
    features: tuple[str, ...]
    class_weighted: bool = False

    def without_feature(self, feature: str) -> ModelSpec:
        """Return a copy of the spec that does not use ``feature``.

        Monotonic constraints (if any) are rebuilt for the new inputs.

        Args:
            feature: Feature to remove.

        Returns:
            A new :class:`ModelSpec`.

        Raises:
            ValueError: If ``feature`` is not used by the spec.
        """
        if feature not in self.features:
            raise ValueError(f"{feature!r} is not used by {self.name}")
        features = tuple(f for f in self.features if f != feature)
        base_builder = self.builder

        def builder() -> BaseEstimator:
            estimator = base_builder()
            if getattr(estimator, "monotonic_cst", None) is not None:
                estimator.set_params(monotonic_cst=monotonic_constraints(
                    features))
            return estimator

        return ModelSpec(
            f"{self.name}_without_{feature}",
            f"{self.description} (without {feature})",
            builder,
            features,
            self.class_weighted,
        )


def monotonic_constraints(features: tuple[str, ...]) -> list[int]:
    """Monotonic constraint vector for HistGradientBoosting.

    Args:
        features: Ordered model inputs.

    Returns:
        One of +1 / -1 / 0 per feature (see ``config.MONOTONIC_SIGNS``).
    """
    return [config.MONOTONIC_SIGNS.get(f, 0) for f in features]


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Weights that give both classes the same total weight.

    Args:
        y: Binary labels.

    Returns:
        Weight per sample (``n_neg / n_pos`` for positives, 1 otherwise).

    Raises:
        ValueError: If one of the classes is absent.
    """
    y = np.asarray(y)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        raise ValueError("Both classes are needed to balance weights")
    return np.where(y == 1, (len(y) - n_pos) / n_pos, 1.0)


def fit_estimator(
    spec: ModelSpec, features: pd.DataFrame, target: np.ndarray
) -> BaseEstimator:
    """Build and fit the estimator of a spec.

    Args:
        spec: Candidate specification.
        features: Model input matrix (columns = ``spec.features``).
        target: Binary labels.

    Returns:
        The fitted estimator.

    Raises:
        ValueError: If the columns do not match ``spec.features``.
    """
    if tuple(features.columns) != spec.features:
        raise ValueError(f"Feature mismatch for {spec.name}")
    estimator = spec.builder()
    if spec.class_weighted:
        estimator.fit(features, target,
                      sample_weight=balanced_sample_weights(target))
    else:
        estimator.fit(features, target)
    return estimator


def _hist_gb_builder(
    params: dict[str, Any], seed: int, monotonic: list[int] | None
) -> Callable[[], BaseEstimator]:
    """Return a builder for a configured HistGradientBoostingClassifier."""
    def builder() -> BaseEstimator:
        return HistGradientBoostingClassifier(
            **params, random_state=seed, monotonic_cst=monotonic
        )
    return builder


def _xgboost_builder(seed: int) -> Callable[[], BaseEstimator] | None:
    """Return an XGBoost builder, or ``None`` if XGBoost is unavailable."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed: candidate skipped")
        return None

    def builder() -> BaseEstimator:
        return XGBClassifier(**config.XGBOOST_PARAMS, random_state=seed)
    return builder


def build_candidate_specs(
    settings: RunSettings, hist_gb_params: dict[str, Any]
) -> list[ModelSpec]:
    """Candidate models compared during model selection.

    Following the professor's advice the set always contains a linear
    baseline, a random forest and a single decision tree, next to the
    gradient-boosting variants of the original submission.

    Args:
        settings: Runtime settings (seed, forest size, XGBoost switch).
        hist_gb_params: Hyper-parameters of the HistGB models.

    Returns:
        List of :class:`ModelSpec`.
    """
    seed = settings.seed
    clean, extended = config.CLEAN_FEATURES, config.EXTENDED_FEATURES

    def logistic() -> BaseEstimator:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**config.LOGISTIC_PARAMS,
                                         random_state=seed)),
        ])

    def random_forest() -> BaseEstimator:
        # NOTE: added — the professor's first model of choice (trees need
        # no scaling); missing in the original submission.
        params = {**config.RANDOM_FOREST_PARAMS,
                  "n_estimators": settings.random_forest_trees}
        return RandomForestClassifier(**params, random_state=seed)

    def decision_tree() -> BaseEstimator:
        # NOTE: added — single white-box tree recommended in class as a
        # sanity check of which variables the data really supports.
        return DecisionTreeClassifier(**config.DECISION_TREE_PARAMS,
                                      random_state=seed)

    specs = [
        ModelSpec("logistic", "Logistic regression (linear baseline)",
                  logistic, clean),
        ModelSpec("decision_tree", "Single decision tree (white box)",
                  decision_tree, clean),
        ModelSpec("random_forest", "Random forest", random_forest, clean),
        ModelSpec("hist_gb", "HistGradientBoosting",
                  _hist_gb_builder(hist_gb_params, seed, None), clean),
        ModelSpec("hist_gb_derived", "HistGB + derived features",
                  _hist_gb_builder(hist_gb_params, seed, None), extended),
        ModelSpec("hist_gb_monotonic",
                  "HistGB + derived features + monotonic constraints",
                  _hist_gb_builder(hist_gb_params, seed,
                                   monotonic_constraints(extended)),
                  extended),
        ModelSpec("hist_gb_weighted",
                  "HistGB + derived features + balanced sample weights",
                  _hist_gb_builder(hist_gb_params, seed, None), extended,
                  class_weighted=True),
    ]
    xgb_builder = _xgboost_builder(seed) if settings.include_xgboost else None
    if xgb_builder is not None:
        specs.append(ModelSpec("xgboost_derived",
                               "XGBoost + derived features", xgb_builder,
                               extended))
    return specs


def tune_hist_gb(
    features: pd.DataFrame, target: np.ndarray, settings: RunSettings
) -> dict[str, Any]:
    """Randomised search of the HistGB hyper-parameters (optional step).

    Args:
        features: Training matrix.
        target: Binary labels.
        settings: Runtime settings (seed and number of folds).

    Returns:
        The default parameters updated with the best ones found.
    """
    base = HistGradientBoostingClassifier(
        **config.HIST_GB_PARAMS, random_state=settings.seed
    )
    search = RandomizedSearchCV(
        base,
        config.HIST_GB_SEARCH_SPACE,
        n_iter=config.HIST_GB_SEARCH_ITERATIONS,
        scoring="roc_auc",
        cv=StratifiedKFold(settings.cv_folds, shuffle=True,
                           random_state=settings.seed),
        random_state=settings.seed,
        n_jobs=-1,
    )
    search.fit(features, target)
    logger.info("Tuned HistGB: %s (CV AUC %.4f)", search.best_params_,
                search.best_score_)
    return {**config.HIST_GB_PARAMS, **search.best_params_}
