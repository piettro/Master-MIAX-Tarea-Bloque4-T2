"""Reinforcement-learning framing of the credit problem.

The table of historical clients is turned into a *game* exactly as done in
class: clients arrive at random, the agent approves or denies, and it
receives a reward equal to minus the cost of its mistake.  The agent is a
linear contextual bandit trained with Thompson sampling.

NOTE: improved over professor's solution — the class notebook imports
``space_bandits.LinearBandits`` from a zip downloaded from Google Drive,
which is not installable from PyPI and pins obsolete dependencies.  The
same algorithm (Bayesian linear regression per action + Thompson sampling)
is implemented here in NumPy so that the project installs cleanly.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.evaluation.costs import CostScenario
from src.utils import config

logger = logging.getLogger(__name__)

N_ACTIONS = 2


def reward(action: int, label: int, scenario: CostScenario) -> float:
    """Reward of an action: 0 when right, minus the error cost otherwise.

    Args:
        action: 0 = approve, 1 = deny.
        label: True outcome (1 = default).
        scenario: Cost scenario.

    Returns:
        ``0``, ``-cost_fp`` (good client denied) or ``-cost_fn``
        (defaulter approved).

    Raises:
        ValueError: If ``action`` or ``label`` is not 0/1.
    """
    if action not in (0, 1) or label not in (0, 1):
        raise ValueError("action and label must be 0 or 1")
    if action == config.ACTION_DENY and label == 0:
        return -float(scenario.cost_fp)
    if action == config.ACTION_APPROVE and label == 1:
        return -float(scenario.cost_fn)
    return 0.0


class CreditEnvironment:
    """Simulated stream of credit applicants (the RL *environment*).

    Mirrors the professor's ``entorno`` class: ``new_client`` draws a
    random applicant, ``client_context`` reveals everything but the label
    and ``act`` returns the reward of a decision.
    """

    def __init__(
        self,
        contexts: np.ndarray,
        labels: np.ndarray,
        scenario: CostScenario,
        rng: np.random.Generator,
    ) -> None:
        """Create the environment.

        Args:
            contexts: Encoded client features, one row per client.
            labels: True outcomes aligned with ``contexts``.
            scenario: Cost scenario defining the rewards.
            rng: Random generator used to draw clients.

        Raises:
            ValueError: If shapes do not match or data is empty.
        """
        if len(contexts) != len(labels) or len(labels) == 0:
            raise ValueError("contexts and labels must be non-empty and "
                             "aligned")
        self.contexts = contexts
        self.labels = np.asarray(labels).astype(int)
        self.scenario = scenario
        self.rng = rng
        self.current: int | None = None

    def new_client(self) -> None:
        """Draw a new applicant at random."""
        self.current = int(self.rng.integers(len(self.labels)))

    def client_context(self) -> np.ndarray:
        """Features of the current applicant (the label stays hidden).

        Returns:
            Context vector.

        Raises:
            RuntimeError: If no client has been drawn yet.
        """
        if self.current is None:
            raise RuntimeError("Call new_client() first")
        return self.contexts[self.current]

    def act(self, action: int) -> float:
        """Apply a decision to the current applicant.

        Args:
            action: 0 = approve, 1 = deny.

        Returns:
            The reward (minus the cost of the decision).

        Raises:
            RuntimeError: If no client has been drawn yet.
        """
        if self.current is None:
            raise RuntimeError("Call new_client() first")
        return reward(action, int(self.labels[self.current]), self.scenario)


class ContextEncoder:
    """Median imputation + standardisation + bias term for linear agents."""

    def __init__(self, features: tuple[str, ...]) -> None:
        """Create an unfitted encoder.

        Args:
            features: Columns used as context.
        """
        self.features = features
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()

    def fit(self, frame: pd.DataFrame) -> ContextEncoder:
        """Learn imputation and scaling parameters on training data.

        Args:
            frame: Training data with the context columns.

        Returns:
            The fitted encoder.
        """
        values = self._imputer.fit_transform(frame[list(self.features)])
        self._scaler.fit(values)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Encode data as context vectors (with a trailing bias of 1).

        Args:
            frame: Data with the context columns.

        Returns:
            Array of shape ``(n, len(features) + 1)``.
        """
        values = self._imputer.transform(frame[list(self.features)])
        scaled = self._scaler.transform(values)
        return np.hstack([scaled, np.ones((len(scaled), 1))])


class RandomPolicy:
    """Benchmark agent choosing actions uniformly at random."""

    def __init__(self, rng: np.random.Generator) -> None:
        """Create the policy.

        Args:
            rng: Random generator.
        """
        self.rng = rng

    def select(self, context: np.ndarray) -> int:
        """Pick a random action (the context is ignored).

        Args:
            context: Context vector (unused).

        Returns:
            0 or 1.
        """
        return int(self.rng.integers(N_ACTIONS))

    def update(self, context: np.ndarray, action: int, value: float) -> None:
        """No learning for the random benchmark.

        Args:
            context: Context vector (unused).
            action: Action taken (unused).
            value: Reward received (unused).
        """


class LinearThompsonSampling:
    """Linear contextual bandit with Thompson sampling.

    For every action the expected reward is modelled as ``theta_a . x``
    with a Gaussian prior; the posterior is updated online with the
    Sherman-Morrison formula.  Sampling ``theta_a`` from the posterior
    balances exploration and exploitation.
    """

    def __init__(
        self,
        n_features: int,
        rng: np.random.Generator,
        prior_precision: float = config.BANDIT_PRIOR_PRECISION,
        noise_std: float = config.BANDIT_NOISE_STD,
    ) -> None:
        """Create the agent with an isotropic Gaussian prior.

        Args:
            n_features: Length of the context vectors.
            rng: Random generator used for posterior sampling.
            prior_precision: Precision of the prior on ``theta``.
            noise_std: Assumed standard deviation of the reward noise.

        Raises:
            ValueError: If a parameter is not positive.
        """
        if n_features < 1 or prior_precision <= 0 or noise_std <= 0:
            raise ValueError("n_features, prior_precision and noise_std "
                             "must be positive")
        self.rng = rng
        self.noise_std = noise_std
        self.covariance = [
            np.eye(n_features) / prior_precision for _ in range(N_ACTIONS)
        ]
        self.b = [np.zeros(n_features) for _ in range(N_ACTIONS)]

    def posterior_mean(self, action: int) -> np.ndarray:
        """Posterior mean of ``theta`` for an action.

        Args:
            action: Action index.

        Returns:
            Coefficient vector.
        """
        return self.covariance[action] @ self.b[action]

    def select(self, context: np.ndarray) -> int:
        """Thompson sampling: act greedily on a posterior sample.

        Args:
            context: Context vector.

        Returns:
            Selected action.
        """
        values = []
        for action in range(N_ACTIONS):
            cov = self.covariance[action]
            chol = np.linalg.cholesky((cov + cov.T) / 2.0)
            noise = chol @ self.rng.standard_normal(len(context))
            theta = self.posterior_mean(action) + self.noise_std * noise
            values.append(theta @ context)
        return int(np.argmax(values))

    def greedy_actions(self, contexts: np.ndarray) -> np.ndarray:
        """Exploitation-only decisions (posterior mean), vectorised.

        Args:
            contexts: Matrix of context vectors.

        Returns:
            Chosen action per row.
        """
        expected = np.column_stack(
            [contexts @ self.posterior_mean(a) for a in range(N_ACTIONS)]
        )
        return expected.argmax(axis=1)

    def update(self, context: np.ndarray, action: int, value: float) -> None:
        """Bayesian update with the observed reward.

        Args:
            context: Context vector.
            action: Action taken.
            value: Reward received.
        """
        cov = self.covariance[action]
        cov_x = cov @ context
        self.covariance[action] = cov - np.outer(cov_x, cov_x) / (
            1.0 + context @ cov_x
        )
        self.b[action] = self.b[action] + value * context


def run_interactions(
    environment: CreditEnvironment,
    policy: RandomPolicy | LinearThompsonSampling,
    n_steps: int,
    learn: bool = True,
    show_progress: bool = False,
    description: str = "bandit",
) -> np.ndarray:
    """Play the credit game for ``n_steps`` applicants.

    Args:
        environment: Environment providing applicants and rewards.
        policy: Agent choosing the actions.
        n_steps: Number of interactions.
        learn: Whether the agent updates after every reward.
        show_progress: Whether to show a progress bar (``tqdm``).
        description: Label of the progress bar.

    Returns:
        Array with the reward of every interaction.

    Raises:
        ValueError: If ``n_steps`` < 1.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    rewards = np.empty(n_steps)
    for step in tqdm(range(n_steps), desc=description,
                     disable=not show_progress):
        environment.new_client()
        context = environment.client_context()
        action = policy.select(context)
        rewards[step] = environment.act(action)
        if learn:
            policy.update(context, action, rewards[step])
    return rewards


class BanditDecisionModel:
    """Wraps an encoder + agent so it can be audited like any classifier.

    The surrogate model needs ``predict`` on *raw* (cleaned) features, so
    that the extracted rules are expressed in business units instead of
    standardised values.
    """

    def __init__(
        self, encoder: ContextEncoder, agent: LinearThompsonSampling
    ) -> None:
        """Create the wrapper.

        Args:
            encoder: Fitted context encoder.
            agent: Trained bandit.
        """
        self.encoder = encoder
        self.agent = agent

    @property
    def features(self) -> tuple[str, ...]:
        """Context features used by the agent."""
        return self.encoder.features

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Greedy (exploitation) decisions for cleaned client data.

        Args:
            frame: Cleaned client data.

        Returns:
            Decision per client (1 = deny).
        """
        return self.agent.greedy_actions(self.encoder.transform(frame))
