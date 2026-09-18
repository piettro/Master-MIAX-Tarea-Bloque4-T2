"""Tests of loading, cleaning, feature engineering and EDA helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import eda
from src.data.cleaning import CreditDataCleaner, drop_invalid_training_rows
from src.data.features import add_derived_features, build_feature_matrix
from src.data.loader import (
    load_construction,
    load_data_dictionary,
    load_production,
    validate_columns,
)
from src.utils import config


def test_loaders_read_synthetic_files(data_dir: Path) -> None:
    construction = load_construction(data_dir)
    production = load_production(data_dir)
    dictionary = load_data_dictionary(data_dir)
    assert config.TARGET in construction
    assert config.TARGET not in production
    assert len(production) == 1200
    assert "Unnamed: 0" not in dictionary.columns
    assert config.AGE in dictionary.index


def test_missing_file_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Place the workshop"):
        load_construction(tmp_path)


def test_validate_columns_reports_missing() -> None:
    with pytest.raises(ValueError, match="age"):
        validate_columns(pd.DataFrame({"x": [1]}), ["x", "age"])


def test_drop_invalid_training_rows(credit_frame: pd.DataFrame) -> None:
    cleaned = drop_invalid_training_rows(credit_frame)
    assert len(cleaned) == len(credit_frame) - 1
    assert (cleaned[config.AGE] >= config.MIN_VALID_AGE).all()


def test_cleaner_keeps_rows_and_flags(credit_frame: pd.DataFrame) -> None:
    cleaner = CreditDataCleaner().fit(credit_frame)
    cleaned = cleaner.transform(credit_frame)
    assert len(cleaned) == len(credit_frame)
    assert (cleaned[config.MISSING_INCOME_FLAG]
            == credit_frame[config.MONTHLY_INCOME].isna()).all()
    special = (credit_frame[config.PAST_DUE_90]
               >= config.SPECIAL_CODE_MIN_VALUE)
    assert cleaned.loc[special, config.SPECIAL_CODE_FLAG].eq(1).all()
    for col in config.DELINQUENCY_COLUMNS:
        assert cleaned[col].max() <= config.SPECIAL_CODE_REPLACEMENT
    # NaNs are preserved for the native handling of tree models.
    assert cleaned[config.MONTHLY_INCOME].isna().sum() == (
        credit_frame[config.MONTHLY_INCOME].isna().sum())


def test_cleaner_uses_construction_caps(credit_frame: pd.DataFrame) -> None:
    cleaner = CreditDataCleaner().fit(credit_frame)
    shifted = credit_frame.copy()
    shifted[config.DEBT_RATIO] *= 100
    cleaned = cleaner.transform(shifted)
    assert cleaned[config.DEBT_RATIO].max() <= (
        cleaner.caps[config.DEBT_RATIO] + 1e-12)


def test_cleaner_requires_fit(credit_frame: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError):
        CreditDataCleaner().transform(credit_frame)
    with pytest.raises(ValueError):
        CreditDataCleaner(cap_quantile=0.0)


def test_derived_features_values() -> None:
    frame = pd.DataFrame({
        config.PAST_DUE_30_59: [1.0], config.PAST_DUE_60_89: [2.0],
        config.PAST_DUE_90: [3.0], config.MONTHLY_INCOME: [3000.0],
        config.DEPENDENTS: [2.0], config.DEBT_RATIO: [0.5],
        config.OPEN_CREDIT_LINES: [4.0], config.REAL_ESTATE_LINES: [1.0],
    })
    out = add_derived_features(frame).iloc[0]
    assert out[config.TOTAL_PAST_DUE] == 6.0
    assert out[config.INCOME_PER_DEPENDENT] == 1000.0
    assert out[config.MONTHLY_DEBT_AMOUNT] == 1500.0
    assert out[config.REAL_ESTATE_SHARE] == 0.25


def test_build_feature_matrix(credit_frame: pd.DataFrame) -> None:
    clean = CreditDataCleaner().fit_transform(credit_frame)
    matrix = build_feature_matrix(clean, config.EXTENDED_FEATURES)
    assert list(matrix.columns) == list(config.EXTENDED_FEATURES)
    with pytest.raises(KeyError):
        build_feature_matrix(clean, ("does_not_exist",))


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(size=5000))
    same = pd.Series(rng.normal(size=5000))
    shifted = pd.Series(rng.normal(1.0, size=5000))
    assert eda.population_stability_index(reference, same) < 0.02
    assert eda.population_stability_index(reference, shifted) > 0.25
    constant = pd.Series(np.ones(100))
    assert eda.population_stability_index(constant, constant) == 0.0


def test_default_rate_by_missingness(credit_frame: pd.DataFrame) -> None:
    rates = eda.default_rate_by_missingness(credit_frame,
                                            config.MONTHLY_INCOME)
    assert set(rates.index) == {"missing", "present"}
    assert rates.between(0, 1).all()
    with pytest.raises(KeyError):
        eda.default_rate_by_missingness(credit_frame, "nope")


def test_anomaly_report(credit_frame: pd.DataFrame) -> None:
    report = eda.anomaly_report(credit_frame)
    assert report["rows_with_age_zero"] == 1
    assert report["rows_with_special_code_any"] == (
        report["rows_with_special_code_all_three"])
