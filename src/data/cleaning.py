"""Cleaning applied identically to construction and production data."""
from __future__ import annotations

import logging

import pandas as pd

from src.utils import config

logger = logging.getLogger(__name__)


def drop_invalid_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are pure recording errors from the training data.

    Only applied to the construction file: every production client must
    receive a decision, so no production row is ever removed.

    Args:
        frame: Construction DataFrame.

    Returns:
        A copy without rows whose age is below ``config.MIN_VALID_AGE``.
    """
    valid = frame[config.AGE] >= config.MIN_VALID_AGE
    n_removed = int((~valid).sum())
    if n_removed:
        logger.info("Removed %d training rows with invalid age", n_removed)
    return frame.loc[valid].reset_index(drop=True)


class CreditDataCleaner:
    """Row-preserving cleaner fitted on construction data only.

    Design decisions (see EDA):

    * Informative missingness is kept as an explicit flag instead of
      dropping ~20% of the rows (the professor dropped them in class, but
      production clients with missing income still need a decision).
    * The 96/98 administrative codes of the delinquency counters are
      flagged and replaced by a value above any genuine count.
    * Heavy tails are capped at a quantile learnt on construction data
      only, which avoids leakage and guarantees identical treatment.
    * Remaining NaNs are left untouched: tree models learn a branch for
      them natively.

    Attributes:
        caps: Upper caps per heavy-tailed column, learnt by :meth:`fit`.
    """

    def __init__(self, cap_quantile: float = config.CAP_QUANTILE) -> None:
        """Create an unfitted cleaner.

        Args:
            cap_quantile: Quantile used as winsorisation cap.

        Raises:
            ValueError: If ``cap_quantile`` is not in ``(0, 1]``.
        """
        if not 0.0 < cap_quantile <= 1.0:
            raise ValueError("cap_quantile must be in (0, 1]")
        self.cap_quantile = cap_quantile
        self.caps: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> CreditDataCleaner:
        """Learn the winsorisation caps from the construction data.

        Args:
            frame: Construction DataFrame.

        Returns:
            The fitted cleaner (for chaining).
        """
        self.caps = {
            col: float(frame[col].quantile(self.cap_quantile))
            for col in config.HEAVY_TAIL_COLUMNS
        }
        logger.info("Cleaner caps: %s", {k: round(v, 2)
                                         for k, v in self.caps.items()})
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the cleaning without removing any row.

        Args:
            frame: DataFrame with at least the base feature columns.

        Returns:
            A cleaned copy with the flag columns added.

        Raises:
            RuntimeError: If the cleaner has not been fitted.
        """
        if not self.caps:
            raise RuntimeError("CreditDataCleaner must be fitted first")
        out = frame.copy()
        delinquency = list(config.DELINQUENCY_COLUMNS)

        # NOTE: improved over professor's solution — in class the rows with
        # missing income and with 96/98 codes were dropped; here they are
        # flagged and kept, because production clients with the same
        # profile must still be scored (dropping them is impossible there).
        # Flags are computed before any value is modified.
        out[config.MISSING_INCOME_FLAG] = (
            out[config.MONTHLY_INCOME].isna().astype(int)
        )
        out[config.SPECIAL_CODE_FLAG] = (
            (out[delinquency] >= config.SPECIAL_CODE_MIN_VALUE)
            .any(axis=1)
            .astype(int)
        )
        for col in delinquency:
            out[col] = out[col].where(
                out[col] < config.SPECIAL_CODE_MIN_VALUE,
                config.SPECIAL_CODE_REPLACEMENT,
            )
        for col, cap in self.caps.items():
            out[col] = out[col].clip(upper=cap)
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``frame`` and return its cleaned version.

        Args:
            frame: Construction DataFrame.

        Returns:
            The cleaned DataFrame.
        """
        return self.fit(frame).transform(frame)
