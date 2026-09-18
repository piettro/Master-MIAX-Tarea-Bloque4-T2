"""Actionable counterfactual explanations for denied clients.

Answers the brief's question "if a client asks why the credit was denied,
what do we tell them?" with the smallest plausible change of *actionable*
variables that would turn the denial into an approval.

NOTE: improved over the original submission — DiCE varied ``MonthlyIncome``
and ``DebtRatio`` independently, although ``DebtRatio`` is debt divided
by income, and it ignored derived features.  This search operates on the
cleaned client record, keeps the debt amount consistent when income
changes, rebuilds every derived feature through the deployed scorer and
is deterministic for a given seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.utils import config

PROBA_COLUMN = "default_proba"
CHANGES_COLUMN = "n_changes"
DISTANCE_COLUMN = "distance"
CHANGED_COLUMN = "changed"
RATIO_DECIMALS = 3
INCOME_DECIMALS = 0
MIN_SCALE = 1e-6
SPARSIFY_CANDIDATES = 25


@dataclass(frozen=True)
class ActionableFeature:
    """A variable the client can change, and in which direction.

    Attributes:
        name: Column name.
        direction: ``"decrease"`` (down to zero) or ``"increase"``.
        max_relative_increase: Cap of an increase, as a fraction.
        integer: Whether the variable only takes integer values.
    """

    name: str
    direction: str
    max_relative_increase: float | None = None
    integer: bool = False

    def __post_init__(self) -> None:
        """Validate the direction.

        Raises:
            ValueError: If ``direction`` is unknown.
        """
        if self.direction not in ("decrease", "increase"):
            raise ValueError(f"Unknown direction {self.direction!r}")

    def bounds(self, value: float) -> tuple[float, float] | None:
        """Plausible range of new values for the current ``value``.

        Args:
            value: Current value of the variable.

        Returns:
            ``(low, high)`` or ``None`` when no plausible change exists
            (unknown value, or nothing left to decrease / increase).
        """
        if pd.isna(value) or value <= 0:
            return None
        if self.direction == "decrease":
            return 0.0, float(value)
        increase = self.max_relative_increase or 0.0
        if increase <= 0:
            return None
        return float(value), float(value) * (1.0 + increase)


def default_actionable_features() -> list[ActionableFeature]:
    """Actionable variables configured for the credit problem.

    Returns:
        List of :class:`ActionableFeature`.
    """
    return [
        ActionableFeature(
            name,
            direction,
            config.MAX_INCOME_INCREASE if direction == "increase" else None,
            name in config.INTEGER_FEATURES,
        )
        for name, direction in config.ACTIONABLE_DIRECTIONS.items()
    ]


def robust_scales(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    """Median absolute deviation per feature, used to normalise distances.

    Args:
        frame: Reference data (training split).
        names: Features to scale.

    Returns:
        Positive scale per feature.
    """
    scales = {}
    for name in names:
        values = frame[name].dropna()
        mad = float((values - values.median()).abs().median())
        scales[name] = max(mad if mad > 0 else float(values.std()),
                           MIN_SCALE)
    return pd.Series(scales)


@dataclass
class CounterfactualResult:
    """Counterfactuals found for one client.

    Attributes:
        original_proba: Default probability of the real client.
        table: Current situation followed by the counterfactuals, with the
            actionable variables, probability, number of changes and
            normalised distance.
        found: Whether at least one valid counterfactual exists.
    """

    original_proba: float
    table: pd.DataFrame
    found: bool


class CounterfactualSearch:
    """Random search + greedy sparsification of actionable changes."""

    def __init__(
        self,
        score_fn: Callable[[pd.DataFrame], np.ndarray],
        actionable: list[ActionableFeature],
        scales: pd.Series,
        threshold: float,
        n_candidates: int,
        seed: int,
    ) -> None:
        """Create the search.

        Args:
            score_fn: Maps cleaned client records to default probability
                (e.g. ``CreditScorer.predict_proba``).
            actionable: Variables allowed to change.
            scales: Normalisation scale per actionable variable.
            threshold: Policy threshold; valid counterfactuals must fall
                strictly below it (not below a generic 0.5).
            n_candidates: Random candidates sampled per client.
            seed: Random seed.

        Raises:
            ValueError: If ``actionable`` is empty or ``n_candidates`` < 1.
        """
        if not actionable:
            raise ValueError("At least one actionable feature is needed")
        if n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        self.score_fn = score_fn
        self.actionable = {f.name: f for f in actionable}
        self.scales = scales
        self.threshold = threshold
        self.n_candidates = n_candidates
        self.seed = seed

    @staticmethod
    def _keep_debt_consistent(
        records: pd.DataFrame, client: pd.Series
    ) -> pd.DataFrame:
        """Keep the debt amount coherent with a changed income.

        ``DebtRatio`` values proposed by the search are read as "ratio at
        the current income"; when the income changes the ratio is rescaled
        so that the implied monthly debt stays the same.
        """
        income, ratio = config.MONTHLY_INCOME, config.DEBT_RATIO
        if ratio in records:
            records[ratio] = records[ratio] * (
                client[income] / records[income])
        return records

    def _score(self, values: pd.DataFrame, client: pd.Series) -> np.ndarray:
        """Default probability of candidate actionable values."""
        records = pd.DataFrame(
            np.repeat(client.to_frame().T.values, len(values), axis=0),
            columns=client.index,
        ).astype(float)
        for name in values.columns:
            records[name] = values[name].to_numpy()
        if config.MONTHLY_INCOME in values.columns:
            records = self._keep_debt_consistent(records, client)
        return self.score_fn(records)

    def _distance(self, values: pd.DataFrame, client: pd.Series) -> pd.Series:
        """Normalised L1 distance to the current situation."""
        deltas = (values - client[values.columns].astype(float)).abs()
        return (deltas / self.scales[values.columns]).sum(axis=1)

    def generate(
        self, client: pd.Series, n_counterfactuals: int
    ) -> CounterfactualResult:
        """Counterfactuals for one cleaned client record.

        Args:
            client: Cleaned record of the client (all model inputs).
            n_counterfactuals: Maximum number of counterfactuals returned.

        Returns:
            A :class:`CounterfactualResult`.
        """
        client = client.astype(float)
        original_proba = float(self.score_fn(client.to_frame().T)[0])
        ranges = {
            name: feature.bounds(client[name])
            for name, feature in self.actionable.items()
        }
        ranges = {k: v for k, v in ranges.items() if v is not None}
        current = client[list(self.actionable)].to_frame().T
        current.index = ["current"]
        current[PROBA_COLUMN] = original_proba
        if not ranges:
            return CounterfactualResult(original_proba, current, False)

        names = list(ranges)
        rng = np.random.default_rng(self.seed)
        low = np.array([ranges[n][0] for n in names])
        high = np.array([ranges[n][1] for n in names])
        mask = rng.random((self.n_candidates, len(names))) < 0.5
        empty = ~mask.any(axis=1)
        mask[empty, rng.integers(len(names), size=int(empty.sum()))] = True
        draws = low + rng.random(mask.shape) * (high - low)
        values = np.where(mask, draws, client[names].to_numpy(float))
        candidates = pd.DataFrame(values, columns=names)
        for name in names:
            if self.actionable[name].integer:
                candidates[name] = np.floor(candidates[name])

        proba = self._score(candidates, client)
        valid = candidates[proba < self.threshold]
        if valid.empty:
            return CounterfactualResult(original_proba, current, False)

        shortlist = valid.loc[self._distance(valid, client).nsmallest(
            SPARSIFY_CANDIDATES).index]
        sparse = [self._sparsify(row, client) for _, row in
                  shortlist.iterrows()]
        found = self._rounded_valid(pd.DataFrame(sparse), client)
        proba = self._score(found, client)
        changed = found != client[names]
        changes = changed.sum(axis=1)
        changed_names = changed.apply(
            lambda row: ",".join(row.index[row]), axis=1)
        distance = self._distance(found, client)
        # Report the effective record, with every actionable variable.
        found = found.reindex(columns=list(self.actionable))
        for name in self.actionable:
            if name not in names:
                found[name] = client[name]
        if config.MONTHLY_INCOME in names:
            found = self._keep_debt_consistent(found, client)
        found[PROBA_COLUMN] = proba
        found[CHANGES_COLUMN] = changes
        found[DISTANCE_COLUMN] = distance
        found[CHANGED_COLUMN] = changed_names
        found = found.sort_values([CHANGES_COLUMN, DISTANCE_COLUMN]).head(
            n_counterfactuals)
        current[CHANGES_COLUMN] = 0
        current[DISTANCE_COLUMN] = 0.0
        current[CHANGED_COLUMN] = ""
        table = pd.concat([current, found], ignore_index=True)
        table.index = ["current"] + [
            f"counterfactual_{i}" for i in range(1, len(found) + 1)
        ]
        return CounterfactualResult(original_proba, table, True)

    def _rounded_valid(
        self, candidates: pd.DataFrame, client: pd.Series
    ) -> pd.DataFrame:
        """Round to communicable precision, keep valid, drop duplicates.

        Rounding can push a borderline candidate back above the threshold,
        so candidates are re-scored; the unrounded ones are kept if no
        rounded candidate remains valid.
        """
        decimals = {
            name: (INCOME_DECIMALS if name == config.MONTHLY_INCOME
                   else RATIO_DECIMALS)
            for name in candidates.columns
        }
        rounded = candidates.round(decimals)
        for name in rounded.columns:
            low, high = self.actionable[name].bounds(client[name])
            rounded[name] = rounded[name].clip(low, high)
        rounded = rounded[self._score(rounded, client) < self.threshold]
        chosen = rounded if not rounded.empty else candidates
        return chosen.drop_duplicates().reset_index(drop=True)

    def _sparsify(self, row: pd.Series, client: pd.Series) -> pd.Series:
        """Revert changes one at a time while the approval still holds."""
        row = row.copy()
        changed = [n for n in row.index if row[n] != client[n]]
        contribution = {
            n: abs(row[n] - client[n]) / self.scales[n] for n in changed
        }
        for name in sorted(changed, key=contribution.get):
            trial = row.copy()
            trial[name] = client[name]
            if self._score(trial.to_frame().T, client)[0] < self.threshold:
                row = trial
        return row
