"""Group-fairness metrics used to audit the use of age in the model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import config


def age_groups(age: pd.Series) -> pd.Series:
    """Bucket ages into the configured bands.

    Args:
        age: Client ages.

    Returns:
        Categorical series of age bands.
    """
    return pd.cut(age, bins=list(config.AGE_BINS))


def group_metrics(
    y_true: np.ndarray, decisions: np.ndarray, groups: pd.Series
) -> pd.DataFrame:
    """Decision and error rates per group.

    Args:
        y_true: Binary labels (1 = default).
        decisions: Decisions (1 = deny).
        groups: Group label of every client (aligned positionally).

    Returns:
        DataFrame indexed by group with size, default rate, denial rate,
        default recall (TPR) and false-positive rate of good clients.

    Raises:
        ValueError: If the inputs have different lengths.
    """
    y_true, decisions = np.asarray(y_true), np.asarray(decisions)
    if not len(y_true) == len(decisions) == len(groups):
        raise ValueError("Inputs must have the same length")
    frame = pd.DataFrame(
        {"group": groups.to_numpy(), "y": y_true, "d": decisions}
    )
    rows = {}
    for name, part in frame.groupby("group", observed=True):
        positives, negatives = part["y"] == 1, part["y"] == 0
        rows[name] = {
            "n": len(part),
            "default_rate": part["y"].mean(),
            "denial_rate": part["d"].mean(),
            "default_recall": (
                part.loc[positives, "d"].mean() if positives.any()
                else np.nan
            ),
            "good_client_denial_rate": (
                part.loc[negatives, "d"].mean() if negatives.any()
                else np.nan
            ),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def fairness_summary(table: pd.DataFrame) -> pd.Series:
    """Formal fairness indicators computed from :func:`group_metrics`.

    * Disparate impact: lowest / highest approval rate (80% rule).
    * Demographic parity difference: range of denial rates.
    * Equal opportunity difference: range of default recall.
    * FPR gap: range of the denial rate among good clients.

    Args:
        table: Output of :func:`group_metrics`.

    Returns:
        Series with the four indicators.
    """
    approval = 1.0 - table["denial_rate"]
    return pd.Series({
        "disparate_impact": approval.min() / approval.max(),
        "demographic_parity_diff": np.ptp(table["denial_rate"]),
        "equal_opportunity_diff": np.ptp(table["default_recall"].dropna()),
        "fpr_gap": np.ptp(table["good_client_denial_rate"].dropna()),
    })
