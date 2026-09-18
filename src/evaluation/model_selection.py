"""Cost-driven model selection with out-of-fold thresholds.

NOTE: improved over the original submission — there, the configuration
was chosen on the same validation split used to report its cost, and the
threshold tuned for one model (HistGB) was applied to another (XGBoost).
Here each candidate gets its own thresholds from out-of-fold predictions
on the training split, the winner is chosen on those OOF costs, and the
validation split is used only once, for an unbiased estimate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.features import build_feature_matrix
from src.evaluation.costs import (
    CostScenario,
    ThresholdResult,
    expected_cost,
    decisions_from_proba,
    optimal_threshold,
    trivial_policy_costs,
)
from src.evaluation.metrics import probability_metrics
from src.models.scorer import CreditScorer
from src.models.supervised import ModelSpec, fit_estimator

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """Everything learnt about one candidate during selection.

    Attributes:
        spec: Candidate specification.
        oof_proba: Out-of-fold probabilities on the training split.
        holdout_proba: Probabilities on the validation split (model fitted
            on the whole training split).
        thresholds: OOF-optimal threshold per scenario key.
        scorer: Candidate fitted on the whole training split.
    """

    spec: ModelSpec
    oof_proba: np.ndarray
    holdout_proba: np.ndarray
    thresholds: dict[str, ThresholdResult]
    scorer: CreditScorer


def out_of_fold_proba(
    spec: ModelSpec,
    clean: pd.DataFrame,
    target: np.ndarray,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold default probabilities of a candidate.

    Args:
        spec: Candidate specification.
        clean: Cleaned training data.
        target: Binary labels aligned with ``clean``.
        n_folds: Number of stratified folds.
        seed: Seed of the fold split (shared by all candidates so that the
            comparison is paired).

    Returns:
        Probability for every row, predicted by a model that never saw it.

    Raises:
        ValueError: If ``n_folds`` < 2.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    target = np.asarray(target)
    matrix = build_feature_matrix(clean, spec.features)
    oof = np.empty(len(target))
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(matrix, target):
        estimator = fit_estimator(
            spec, matrix.iloc[train_idx], target[train_idx]
        )
        oof[test_idx] = estimator.predict_proba(matrix.iloc[test_idx])[:, 1]
    return oof


def fit_scorer(
    spec: ModelSpec,
    clean: pd.DataFrame,
    target: np.ndarray,
    scenarios: dict[str, CostScenario],
    n_folds: int,
    seed: int,
) -> tuple[CreditScorer, np.ndarray, dict[str, ThresholdResult]]:
    """Fit a candidate and attach OOF-optimal thresholds.

    Args:
        spec: Candidate specification.
        clean: Cleaned training data.
        target: Binary labels.
        scenarios: Cost scenarios keyed by name.
        n_folds: Folds used for the OOF thresholds.
        seed: Random seed.

    Returns:
        Tuple ``(scorer, oof_proba, threshold_results)``.
    """
    target = np.asarray(target)
    oof = out_of_fold_proba(spec, clean, target, n_folds, seed)
    results = {
        key: optimal_threshold(target, oof, scenario)
        for key, scenario in scenarios.items()
    }
    estimator = fit_estimator(
        spec, build_feature_matrix(clean, spec.features), target
    )
    thresholds = {key: result.threshold for key, result in results.items()}
    return CreditScorer(spec, estimator, thresholds), oof, results


def evaluate_candidates(
    specs: list[ModelSpec],
    train_clean: pd.DataFrame,
    y_train: np.ndarray,
    valid_clean: pd.DataFrame,
    y_valid: np.ndarray,
    scenarios: dict[str, CostScenario],
    n_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, CandidateResult]]:
    """Compare candidates on OOF cost and on the untouched validation set.

    Args:
        specs: Candidates to compare.
        train_clean: Cleaned training split.
        y_train: Training labels.
        valid_clean: Cleaned validation split.
        y_valid: Validation labels.
        scenarios: Cost scenarios keyed by name.
        n_folds: Folds of the OOF estimate.
        seed: Random seed.

    Returns:
        Tuple ``(table, results)``: one row per candidate with OOF and
        hold-out metrics plus a ``selection_score`` (mean over scenarios
        of the OOF cost relative to the best trivial policy), and the
        detailed :class:`CandidateResult` per name.

    Raises:
        ValueError: If ``specs`` is empty.
    """
    if not specs:
        raise ValueError("At least one candidate is required")
    y_train, y_valid = np.asarray(y_train), np.asarray(y_valid)
    rows, results = [], {}
    for spec in specs:
        logger.info("Evaluating candidate %s", spec.name)
        scorer, oof, thresholds = fit_scorer(
            spec, train_clean, y_train, scenarios, n_folds, seed
        )
        holdout = scorer.predict_proba(valid_clean)
        row: dict[str, object] = {
            "model": spec.name,
            "description": spec.description,
            "oof_auc": probability_metrics(y_train, oof)["auc"],
        }
        row.update({
            f"holdout_{k}": v
            for k, v in probability_metrics(y_valid, holdout).items()
        })
        relative = []
        for key, scenario in scenarios.items():
            result = thresholds[key]
            trivial = min(trivial_policy_costs(y_train, scenario).values())
            relative.append(result.cost / trivial)
            row[f"{key}_threshold"] = result.threshold
            row[f"{key}_oof_cost"] = result.cost
            row[f"{key}_holdout_cost"] = expected_cost(
                y_valid, decisions_from_proba(holdout, result.threshold),
                scenario,
            )
        row["selection_score"] = float(np.mean(relative))
        rows.append(row)
        results[spec.name] = CandidateResult(
            spec, oof, holdout, thresholds, scorer
        )
    table = pd.DataFrame(rows).set_index("model")
    return table.sort_values("selection_score"), results


def select_best(table: pd.DataFrame) -> str:
    """Name of the candidate with the lowest selection score.

    Ties are broken by the higher OOF AUC.

    Args:
        table: Output table of :func:`evaluate_candidates`.

    Returns:
        The winning model name.
    """
    ranked = table.sort_values(
        ["selection_score", "oof_auc"], ascending=[True, False]
    )
    return str(ranked.index[0])
