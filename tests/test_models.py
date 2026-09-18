"""Tests of the supervised catalogue, the scorer and the bandit."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.cleaning import CreditDataCleaner, drop_invalid_training_rows
from src.data.features import build_feature_matrix
from src.evaluation.costs import CostScenario, expected_cost
from src.models.bandit import (
    ContextEncoder,
    CreditEnvironment,
    LinearThompsonSampling,
    RandomPolicy,
    reward,
    run_interactions,
)
from src.models.scorer import CreditScorer
from src.models.supervised import (
    balanced_sample_weights,
    build_candidate_specs,
    fit_estimator,
    monotonic_constraints,
)
from src.utils import config
from src.utils.config import RunSettings

SCENARIO = CostScenario("s2", "FN costs 10", 1.0, 10.0)


def test_reward_matches_costs() -> None:
    assert reward(config.ACTION_DENY, 0, SCENARIO) == -1.0
    assert reward(config.ACTION_APPROVE, 1, SCENARIO) == -10.0
    assert reward(config.ACTION_APPROVE, 0, SCENARIO) == 0.0
    assert reward(config.ACTION_DENY, 1, SCENARIO) == 0.0
    with pytest.raises(ValueError):
        reward(2, 0, SCENARIO)


def test_environment_hides_label_and_requires_client() -> None:
    env = CreditEnvironment(np.eye(3), np.array([0, 1, 0]), SCENARIO,
                            np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        env.client_context()
    env.new_client()
    assert env.client_context().shape == (3,)


def test_sherman_morrison_update_matches_inverse() -> None:
    rng = np.random.default_rng(0)
    agent = LinearThompsonSampling(4, rng)
    contexts = rng.normal(size=(50, 4))
    for x in contexts:
        agent.update(x, 1, 1.0)
    expected = np.linalg.inv(np.eye(4) + contexts.T @ contexts)
    np.testing.assert_allclose(agent.covariance[1], expected, atol=1e-10)


def test_bandit_beats_random_policy() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    features = rng.normal(size=(n, 2))
    labels = (features[:, 0] + 0.3 * rng.normal(size=n) > 1.0).astype(int)
    contexts = np.hstack([features, np.ones((n, 1))])
    env = CreditEnvironment(contexts, labels, SCENARIO, rng)
    agent = LinearThompsonSampling(3, rng)
    run_interactions(env, agent, 3000)
    bandit_cost = expected_cost(labels, agent.greedy_actions(contexts),
                                SCENARIO)
    random_rewards = run_interactions(env, RandomPolicy(rng), 3000,
                                      learn=False)
    assert bandit_cost < -random_rewards.mean()


def test_context_encoder_adds_bias(credit_frame) -> None:
    clean = CreditDataCleaner().fit_transform(credit_frame)
    encoder = ContextEncoder(config.CLEAN_FEATURES).fit(clean)
    contexts = encoder.transform(clean)
    assert contexts.shape == (len(clean), len(config.CLEAN_FEATURES) + 1)
    assert np.all(contexts[:, -1] == 1.0)
    assert not np.isnan(contexts).any()


def test_balanced_weights() -> None:
    weights = balanced_sample_weights(np.array([0, 0, 0, 1]))
    assert weights.tolist() == [1.0, 1.0, 1.0, 3.0]
    with pytest.raises(ValueError):
        balanced_sample_weights(np.zeros(3))


def test_monotonic_constraints_signs() -> None:
    signs = monotonic_constraints(config.EXTENDED_FEATURES)
    lookup = dict(zip(config.EXTENDED_FEATURES, signs))
    assert lookup[config.REVOLVING_UTILIZATION] == 1
    assert lookup[config.AGE] == -1
    assert lookup[config.DEPENDENTS] == 0


def test_monotonic_model_is_monotonic(credit_frame) -> None:
    clean = CreditDataCleaner().fit_transform(
        drop_invalid_training_rows(credit_frame))
    y = clean.pop(config.TARGET).to_numpy()
    specs = {s.name: s for s in build_candidate_specs(
        RunSettings(include_xgboost=False),
        {**config.HIST_GB_PARAMS, "max_iter": 60})}
    spec = specs["hist_gb_monotonic"]
    matrix = build_feature_matrix(clean, spec.features)
    scorer = CreditScorer(spec, fit_estimator(spec, matrix, y))
    grid = matrix.median().to_frame().T.loc[[0] * 30].reset_index(drop=True)
    grid[config.REVOLVING_UTILIZATION] = np.linspace(0, 1.2, 30)
    assert np.all(np.diff(scorer.proba_from_matrix(grid)) >= -1e-12)


def test_spec_without_feature(credit_frame) -> None:
    specs = {s.name: s for s in build_candidate_specs(
        RunSettings(include_xgboost=False), config.HIST_GB_PARAMS)}
    reduced = specs["hist_gb_monotonic"].without_feature(config.AGE)
    assert config.AGE not in reduced.features
    estimator = reduced.builder()
    assert len(estimator.monotonic_cst) == len(reduced.features)
    with pytest.raises(ValueError):
        reduced.without_feature(config.AGE)


def test_scorer_decide_requires_threshold(credit_frame) -> None:
    clean = CreditDataCleaner().fit_transform(
        drop_invalid_training_rows(credit_frame))
    y = clean.pop(config.TARGET).to_numpy()
    spec = build_candidate_specs(RunSettings(include_xgboost=False),
                                 config.HIST_GB_PARAMS)[0]
    scorer = CreditScorer(
        spec, fit_estimator(spec, build_feature_matrix(clean, spec.features),
                            y), {"s": 0.5})
    decisions = scorer.decide(clean, "s")
    assert set(np.unique(decisions)) <= {0, 1}
    with pytest.raises(KeyError):
        scorer.decide(clean, "missing")
