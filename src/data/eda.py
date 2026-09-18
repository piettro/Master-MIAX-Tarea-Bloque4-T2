"""Exploratory data analysis and population-stability checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.utils import config

SUMMARY_STATISTICS = ("count", "min", "max", "mean", "std")
MAX_SEGMENT_COUNT = 5
N_INCOME_DECILES = 10


def describe_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Transposed summary statistics, as shown by the professor in class.

    Args:
        frame: DataFrame to describe.

    Returns:
        One row per column with count, min, max, mean and std.
    """
    return frame.describe().T.loc[:, list(SUMMARY_STATISTICS)]


def missing_value_report(frame: pd.DataFrame) -> pd.Series:
    """Percentage of missing values per column (only columns with NaNs).

    Args:
        frame: DataFrame to inspect.

    Returns:
        Series of percentages sorted in descending order.
    """
    pct = 100.0 * frame.isna().mean()
    return pct[pct > 0].sort_values(ascending=False)


def default_rate_by_missingness(
    frame: pd.DataFrame, column: str, target: str = config.TARGET
) -> pd.Series:
    """Compare the default rate of rows with and without ``column``.

    The professor stressed checking this before dropping rows: if the
    rates differ, missingness is informative.

    Args:
        frame: Labelled DataFrame.
        column: Column whose missingness is analysed.
        target: Binary target column.

    Returns:
        Series with the default rate for ``missing`` and ``present`` rows.

    Raises:
        KeyError: If ``column`` or ``target`` is not in ``frame``.
    """
    if column not in frame or target not in frame:
        raise KeyError(f"{column!r} or {target!r} not in DataFrame")
    is_missing = frame[column].isna()
    return pd.Series(
        {
            "missing": float(frame.loc[is_missing, target].mean()),
            "present": float(frame.loc[~is_missing, target].mean()),
        },
        name="default_rate",
    )


def anomaly_report(frame: pd.DataFrame) -> dict[str, float]:
    """Quantify the data-quality anomalies found during the EDA.

    Args:
        frame: Raw construction DataFrame.

    Returns:
        Dictionary of anomaly name to count or rate.
    """
    delinquency = frame[list(config.DELINQUENCY_COLUMNS)]
    special_any = (delinquency >= config.SPECIAL_CODE_MIN_VALUE).any(axis=1)
    special_all = (delinquency >= config.SPECIAL_CODE_MIN_VALUE).all(axis=1)
    income_known = frame[config.MONTHLY_INCOME].notna()
    report = {
        "rows_with_age_zero": int((frame[config.AGE] <= 0).sum()),
        "rows_with_special_code_any": int(special_any.sum()),
        "rows_with_special_code_all_three": int(special_all.sum()),
        "revolving_utilization_above_1": int(
            (frame[config.REVOLVING_UTILIZATION] > 1).sum()
        ),
        "revolving_utilization_above_10": int(
            (frame[config.REVOLVING_UTILIZATION] > 10).sum()
        ),
        "median_debt_ratio_income_known": float(
            frame.loc[income_known, config.DEBT_RATIO].median()
        ),
        "median_debt_ratio_income_missing": float(
            frame.loc[~income_known, config.DEBT_RATIO].median()
        ),
        "clients_aged_80_or_more": int((frame[config.AGE] >= 80).sum()),
    }
    if config.TARGET in frame:
        report["default_rate_special_code"] = float(
            frame.loc[special_any, config.TARGET].mean()
        )
        report["default_rate_overall"] = float(frame[config.TARGET].mean())
    return report


def default_rate_by_segments(
    frame: pd.DataFrame, target: str = config.TARGET
) -> dict[str, pd.Series]:
    """Default rate along the three classic credit-risk axes.

    Args:
        frame: Labelled DataFrame (raw or cleaned).
        target: Binary target column.

    Returns:
        Dictionary with the default rate by age band, by income decile and
        by number of 90+ days delinquencies (capped at 5).
    """
    age_bands = pd.cut(frame[config.AGE], bins=list(config.AGE_BINS))
    by_age = frame.groupby(age_bands, observed=True)[target].mean()

    with_income = frame.dropna(subset=[config.MONTHLY_INCOME])
    deciles = pd.qcut(
        with_income[config.MONTHLY_INCOME].rank(method="first"),
        N_INCOME_DECILES,
        labels=range(1, N_INCOME_DECILES + 1),
    )
    by_income = with_income.groupby(deciles, observed=True)[target].mean()

    late = frame[config.PAST_DUE_90].where(
        frame[config.PAST_DUE_90] < config.SPECIAL_CODE_MIN_VALUE
    ).clip(upper=MAX_SEGMENT_COUNT)
    by_late = frame.groupby(late)[target].mean()
    return {"age": by_age, "income_decile": by_income, "late_90": by_late}


def spearman_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation matrix, robust to the heavy tails of the data.

    Args:
        frame: Numeric DataFrame.

    Returns:
        Spearman correlation matrix.
    """
    return frame.corr(method="spearman")


def pca_projection(
    frame: pd.DataFrame, features: tuple[str, ...], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two-component PCA of the median-imputed, standardised features.

    Args:
        frame: DataFrame with the feature columns.
        features: Columns to project.
        seed: Random seed of the PCA solver.

    Returns:
        Tuple ``(projection, explained_variance_ratio)``.
    """
    values = SimpleImputer(strategy="median").fit_transform(
        frame[list(features)]
    )
    values = StandardScaler().fit_transform(values)
    pca = PCA(n_components=2, random_state=seed)
    projection = pca.fit_transform(values)
    return projection, pca.explained_variance_ratio_


def population_stability_index(
    expected: pd.Series, observed: pd.Series, bins: int = config.PSI_BINS
) -> float:
    """Population Stability Index between two samples of one variable.

    Bins are quantiles of ``expected``; empty bins are floored at a tiny
    probability to keep the logarithm finite.

    Args:
        expected: Reference sample (construction data).
        observed: New sample (production data).
        bins: Number of quantile bins.

    Returns:
        The PSI value (0 means identical distributions).

    Raises:
        ValueError: If ``bins`` < 2 or a sample has no valid values.
    """
    if bins < 2:
        raise ValueError("bins must be >= 2")
    ref, new = expected.dropna(), observed.dropna()
    if ref.empty or new.empty:
        raise ValueError("Both samples need at least one non-missing value")
    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    floor = 1e-6
    ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
    new_share = np.histogram(new, bins=edges)[0] / len(new)
    ref_share = np.clip(ref_share, floor, None)
    new_share = np.clip(new_share, floor, None)
    return float(np.sum((new_share - ref_share)
                        * np.log(new_share / ref_share)))


def psi_table(
    reference: pd.DataFrame, new: pd.DataFrame, columns: tuple[str, ...]
) -> pd.DataFrame:
    """PSI per column with the conventional stability verdict.

    Args:
        reference: Construction data (cleaned).
        new: Production data (cleaned the same way).
        columns: Columns to compare.

    Returns:
        DataFrame indexed by column with ``psi`` and ``verdict``.
    """
    values = {
        col: population_stability_index(reference[col], new[col])
        for col in columns
    }
    table = pd.DataFrame({"psi": pd.Series(values)})

    def verdict(value: float) -> str:
        if value < config.PSI_STABLE_LIMIT:
            return "stable"
        if value < config.PSI_SEVERE_LIMIT:
            return "moderate shift"
        return "severe shift"

    table["verdict"] = table["psi"].map(verdict)
    return table.sort_values("psi", ascending=False)
