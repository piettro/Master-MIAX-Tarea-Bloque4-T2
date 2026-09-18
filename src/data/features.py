"""Feature engineering shared by training, production and counterfactuals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import config


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven derived features to a cleaned DataFrame.

    * ``TotalPastDue``: sum of the three delinquency counters.
    * ``IncomePerDependent``: income per person supported by the client.
    * ``MonthlyDebtAmount``: absolute monthly debt (ratio x income).
    * ``RealEstateLineShare``: share of real-estate lines among open ones.

    NaNs propagate where the income is unknown, consistent with the native
    missing-value handling of the tree models.

    Args:
        frame: Cleaned DataFrame (output of ``CreditDataCleaner``).

    Returns:
        A copy with the derived columns appended.
    """
    out = frame.copy()
    out[config.TOTAL_PAST_DUE] = out[list(config.DELINQUENCY_COLUMNS)].sum(
        axis=1
    )
    out[config.INCOME_PER_DEPENDENT] = out[config.MONTHLY_INCOME] / (
        out[config.DEPENDENTS].fillna(0) + 1
    )
    out[config.MONTHLY_DEBT_AMOUNT] = (
        out[config.DEBT_RATIO] * out[config.MONTHLY_INCOME]
    )
    open_lines = out[config.OPEN_CREDIT_LINES].replace(0, np.nan)
    out[config.REAL_ESTATE_SHARE] = (
        out[config.REAL_ESTATE_LINES] / open_lines
    ).fillna(0.0)
    return out


def build_feature_matrix(
    frame: pd.DataFrame, features: tuple[str, ...] | list[str]
) -> pd.DataFrame:
    """Return the model input matrix for a cleaned DataFrame.

    Derived features are computed on demand, so callers can always pass
    the cleaned (non-derived) data.

    Args:
        frame: Cleaned DataFrame.
        features: Ordered feature names expected by the model.

    Returns:
        DataFrame with exactly ``features`` as columns, as floats.

    Raises:
        KeyError: If a requested feature cannot be produced.
    """
    needs_derived = any(f in config.DERIVED_FEATURES for f in features)
    source = add_derived_features(frame) if needs_derived else frame
    missing = [f for f in features if f not in source.columns]
    if missing:
        raise KeyError(f"Cannot build features: {missing}")
    return source.loc[:, list(features)].astype(float)
