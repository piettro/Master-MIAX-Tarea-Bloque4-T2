"""The deployed credit scorer: features + fitted model + cost thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.data.features import build_feature_matrix
from src.models.supervised import ModelSpec


@dataclass
class CreditScorer:
    """Bundle of everything needed to take a credit decision.

    Keeping the specification, the fitted estimator and the thresholds
    together guarantees that the model that is audited is exactly the
    model that produced the delivered decisions.

    Attributes:
        spec: Configuration of the model.
        estimator: Fitted estimator.
        thresholds: Decision threshold per cost-scenario key.
    """

    spec: ModelSpec
    estimator: BaseEstimator
    thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def features(self) -> tuple[str, ...]:
        """Ordered model inputs."""
        return self.spec.features

    def feature_matrix(self, clean: pd.DataFrame) -> pd.DataFrame:
        """Build the model input matrix from cleaned data.

        Args:
            clean: Output of ``CreditDataCleaner.transform``.

        Returns:
            Matrix with the model's feature columns.
        """
        return build_feature_matrix(clean, self.features)

    def proba_from_matrix(
        self, matrix: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """Default probability for an already-built feature matrix.

        Args:
            matrix: Model inputs (DataFrame or array in feature order).

        Returns:
            Probability of default per row.
        """
        if not isinstance(matrix, pd.DataFrame):
            matrix = pd.DataFrame(matrix, columns=list(self.features))
        return self.estimator.predict_proba(matrix)[:, 1]

    def predict_proba(self, clean: pd.DataFrame) -> np.ndarray:
        """Default probability for cleaned client data.

        Args:
            clean: Cleaned client data.

        Returns:
            Probability of default per client.
        """
        return self.proba_from_matrix(self.feature_matrix(clean))

    def decide(self, clean: pd.DataFrame, scenario_key: str) -> np.ndarray:
        """Deny (1) / approve (0) decisions for a cost scenario.

        Args:
            clean: Cleaned client data.
            scenario_key: Key of the cost scenario.

        Returns:
            Integer decisions.

        Raises:
            KeyError: If no threshold is stored for ``scenario_key``.
        """
        if scenario_key not in self.thresholds:
            raise KeyError(f"No threshold for scenario {scenario_key!r}")
        proba = self.predict_proba(clean)
        return (proba >= self.thresholds[scenario_key]).astype(int)
