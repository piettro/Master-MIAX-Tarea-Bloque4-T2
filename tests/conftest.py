"""Shared fixtures: synthetic data with the real schema and quirks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import config


def make_credit_frame(n_rows: int, seed: int,
                      labelled: bool = True) -> pd.DataFrame:
    """Synthetic clients with a learnable default signal.

    Args:
        n_rows: Number of clients.
        seed: Random seed.
        labelled: Whether to add the target column.

    Returns:
        DataFrame with the base features (and target if requested).
    """
    rng = np.random.default_rng(seed)
    utilization = rng.beta(0.8, 2.0, n_rows) * 1.2
    age = rng.integers(21, 90, n_rows).astype(float)
    late_90 = rng.poisson(0.15, n_rows).astype(float)
    late_30 = rng.poisson(0.3, n_rows).astype(float)
    late_60 = rng.poisson(0.1, n_rows).astype(float)
    income = rng.lognormal(8.5, 0.6, n_rows)
    income[rng.random(n_rows) < 0.2] = np.nan
    dependents = rng.integers(0, 4, n_rows).astype(float)
    dependents[rng.random(n_rows) < 0.03] = np.nan
    frame = pd.DataFrame({
        config.REVOLVING_UTILIZATION: utilization,
        config.AGE: age,
        config.PAST_DUE_30_59: late_30,
        config.DEBT_RATIO: rng.gamma(1.5, 0.3, n_rows),
        config.MONTHLY_INCOME: income,
        config.OPEN_CREDIT_LINES: rng.integers(0, 20, n_rows).astype(float),
        config.PAST_DUE_90: late_90,
        config.REAL_ESTATE_LINES: rng.integers(0, 4, n_rows).astype(float),
        config.PAST_DUE_60_89: late_60,
        config.DEPENDENTS: dependents,
    })
    special = rng.random(n_rows) < 0.005
    frame.loc[special, list(config.DELINQUENCY_COLUMNS)] = 98.0
    if labelled:
        logit = (-3.2 + 2.5 * utilization + 1.2 * np.minimum(late_90, 3)
                 + 0.5 * np.minimum(late_30, 3) - 0.02 * (age - 50))
        logit[special] += 3.0
        proba = 1.0 / (1.0 + np.exp(-logit))
        frame[config.TARGET] = (rng.random(n_rows) < proba).astype(int)
        frame.loc[0, config.AGE] = 0.0  # a recording error, as in the data
    return frame


@pytest.fixture
def credit_frame() -> pd.DataFrame:
    """Labelled synthetic construction data."""
    return make_credit_frame(4000, seed=7)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Folder with synthetic construction, production and dictionary."""
    construction = make_credit_frame(3000, seed=11)
    production = make_credit_frame(1200, seed=12, labelled=False)
    production.insert(0, config.TARGET, np.nan)
    construction.to_csv(tmp_path / config.CONSTRUCTION_FILENAME,
                        index=False)
    production.to_csv(tmp_path / config.PRODUCTION_FILENAME, index=False)
    dictionary = pd.DataFrame({
        "Variable Name": [config.TARGET, *config.BASE_FEATURES],
        "Description": "synthetic",
        "Type": "real",
    })
    dictionary.to_csv(tmp_path / config.DICTIONARY_FILENAME,
                      sep=config.DICTIONARY_SEPARATOR)
    return tmp_path
