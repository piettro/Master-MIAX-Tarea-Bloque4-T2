"""Tests of costs, thresholds, metrics, fairness and model selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.data.cleaning import CreditDataCleaner, drop_invalid_training_rows
from src.evaluation import fairness
from src.evaluation.costs import (
    CostScenario,
    build_scenarios,
    confusion_counts,
    cost_curve,
    decisions_from_proba,
    expected_cost,
    optimal_threshold,
    trivial_policy_costs,
)
from src.evaluation.metrics import (
    bootstrap_cost,
    confidence_interval,
    delong_test,
    expected_calibration_error,
    lift_table,
    majority_class_accuracy,
)
from src.evaluation.model_selection import (
    evaluate_candidates,
    out_of_fold_proba,
    select_best,
)
from src.models.supervised import build_candidate_specs
from src.utils import config
from src.utils.config import RunSettings

SCENARIO_2 = CostScenario("s2", "FN costs 10", 1.0, 10.0)


@pytest.fixture
def scores() -> tuple[np.ndarray, np.ndarray]:
    """Calibrated synthetic scores and labels."""
    rng = np.random.default_rng(3)
    proba = rng.beta(1, 8, 3000)
    labels = (rng.random(3000) < proba).astype(int)
    return labels, proba


def test_theoretical_thresholds() -> None:
    scenarios = build_scenarios()
    assert scenarios[config.SCENARIO_1_KEY].theoretical_threshold == 0.5
    assert scenarios[config.SCENARIO_2_KEY].theoretical_threshold == (
        pytest.approx(1 / 11))
    with pytest.raises(ValueError):
        CostScenario("bad", "bad", -1.0, 1.0)


def test_expected_cost_and_counts() -> None:
    y = np.array([0, 0, 1, 1])
    decisions = np.array([1, 0, 0, 1])  # one FP, one FN
    assert confusion_counts(y, decisions) == {"tp": 1, "fp": 1, "tn": 1,
                                              "fn": 1}
    assert expected_cost(y, decisions, SCENARIO_2) == pytest.approx(11 / 4)
    with pytest.raises(ValueError):
        expected_cost(y, decisions[:3], SCENARIO_2)


def test_optimal_threshold_matches_brute_force(scores) -> None:
    labels, proba = scores
    result = optimal_threshold(labels, proba, SCENARIO_2)
    brute = min(
        expected_cost(labels, decisions_from_proba(proba, t), SCENARIO_2)
        for t in np.append(np.unique(proba), np.inf)
    )
    assert result.cost == pytest.approx(brute)
    realised = expected_cost(
        labels, decisions_from_proba(proba, result.threshold), SCENARIO_2)
    assert realised == pytest.approx(result.cost)


def test_optimal_threshold_near_bayes_for_calibrated_scores(scores) -> None:
    labels, proba = scores
    result = optimal_threshold(labels, proba, SCENARIO_2)
    assert abs(result.threshold - SCENARIO_2.theoretical_threshold) < 0.05


def test_cost_curve_matches_pointwise(scores) -> None:
    labels, proba = scores
    grid = np.linspace(0, 1, 11)
    curve = cost_curve(labels, proba, SCENARIO_2, grid)
    pointwise = [expected_cost(labels, decisions_from_proba(proba, t),
                               SCENARIO_2) for t in grid]
    np.testing.assert_allclose(curve, pointwise)


def test_trivial_policies() -> None:
    costs = trivial_policy_costs(np.array([0, 0, 0, 1]), SCENARIO_2)
    assert costs == {"approve_all": 2.5, "deny_all": 0.75}


def test_calibration_and_accuracy_baseline(scores) -> None:
    labels, proba = scores
    assert expected_calibration_error(labels, proba) < 0.03
    assert majority_class_accuracy(np.array([0, 0, 0, 1])) == 0.75


def test_delong(scores) -> None:
    labels, proba = scores
    same = delong_test(labels, proba, proba)
    assert same.difference == 0.0 and same.p_value == 1.0
    noisy = proba + np.random.default_rng(1).normal(0, 0.2, len(proba))
    result = delong_test(labels, proba, noisy)
    assert result.auc_a == pytest.approx(roc_auc_score(labels, proba))
    assert result.difference > 0 and result.p_value < 0.05
    assert result.ci_low < result.difference < result.ci_high


def test_bootstrap_interval_contains_point_estimate(scores) -> None:
    labels, proba = scores
    samples = bootstrap_cost(labels, proba, 0.1, SCENARIO_2, 300, seed=0)
    low, high = confidence_interval(samples)
    point = expected_cost(labels, decisions_from_proba(proba, 0.1),
                          SCENARIO_2)
    assert len(samples) == 300 and low <= point <= high


def test_lift_table(scores) -> None:
    labels, proba = scores
    table = lift_table(labels, proba)
    assert table["cumulative_capture"].iloc[-1] == pytest.approx(1.0)
    assert table["lift"].iloc[0] > 1.0


def test_fairness_metrics() -> None:
    groups = pd.Series(["young", "young", "old", "old"])
    table = fairness.group_metrics(np.array([1, 0, 1, 0]),
                                   np.array([1, 1, 1, 0]), groups)
    summary = fairness.fairness_summary(table)
    assert table.loc["young", "denial_rate"] == 1.0
    assert summary["disparate_impact"] == 0.0
    assert summary["fpr_gap"] == 1.0


def test_model_selection_on_synthetic_data(credit_frame) -> None:
    valid_rows = drop_invalid_training_rows(credit_frame)
    clean = CreditDataCleaner().fit_transform(valid_rows)
    y = clean.pop(config.TARGET).to_numpy()
    train, valid = clean.iloc[:3000], clean.iloc[3000:]
    settings = RunSettings(random_forest_trees=20, include_xgboost=False)
    specs = [s for s in build_candidate_specs(
        settings, {**config.HIST_GB_PARAMS, "max_iter": 50})
        if s.name in ("logistic", "hist_gb")]
    oof = out_of_fold_proba(specs[0], train, y[:3000], 3, seed=0)
    assert oof.shape == (3000,) and np.all((oof >= 0) & (oof <= 1))

    table, results = evaluate_candidates(
        specs, train, y[:3000], valid, y[3000:], build_scenarios(), 3, 0)
    assert set(table.index) == {"logistic", "hist_gb"}
    assert table["holdout_auc"].min() > 0.65
    winner = select_best(table)
    assert set(results[winner].scorer.thresholds) == set(
        config.SCENARIO_COSTS)
