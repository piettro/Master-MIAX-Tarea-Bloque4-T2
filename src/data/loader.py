"""Loading of the construction, production and dictionary files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils import config

logger = logging.getLogger(__name__)


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read a CSV file with an actionable error message when it is missing.

    Args:
        path: File to read.
        **kwargs: Extra arguments forwarded to :func:`pandas.read_csv`.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file exists but cannot be parsed.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Required data file not found: {path}. Place the workshop "
            f"files ({config.CONSTRUCTION_FILENAME}, "
            f"{config.PRODUCTION_FILENAME}, {config.DICTIONARY_FILENAME}) "
            f"in the '{path.parent}' folder (they are distributed with the "
            "assignment materials)."
        )
    try:
        return pd.read_csv(path, **kwargs)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc


def validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    """Check that a DataFrame contains every required column.

    Args:
        frame: Table to check.
        required: Column names that must be present.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_construction(data_dir: Path) -> pd.DataFrame:
    """Load the labelled historical data set used to build the model.

    Args:
        data_dir: Folder containing the construction file.

    Returns:
        The construction DataFrame (features + target).

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If columns are missing or the target is not binary.
    """
    frame = _read_csv(data_dir / config.CONSTRUCTION_FILENAME)
    validate_columns(frame, (*config.BASE_FEATURES, config.TARGET))
    if not set(frame[config.TARGET].dropna().unique()) <= {0, 1}:
        raise ValueError(f"Target {config.TARGET!r} must be binary (0/1)")
    logger.info("Construction data: %d rows x %d cols", *frame.shape)
    return frame


def load_production(data_dir: Path) -> pd.DataFrame:
    """Load the unlabelled production clients that need a decision.

    The production file ships an empty target column, which is dropped.

    Args:
        data_dir: Folder containing the production file.

    Returns:
        The production DataFrame with feature columns only.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If feature columns are missing.
    """
    frame = _read_csv(data_dir / config.PRODUCTION_FILENAME)
    validate_columns(frame, config.BASE_FEATURES)
    if config.TARGET in frame.columns:
        n_labelled = int(frame[config.TARGET].notna().sum())
        if n_labelled:
            logger.warning(
                "Production file has %d non-empty target values; they are "
                "ignored.",
                n_labelled,
            )
        frame = frame.drop(columns=[config.TARGET])
    logger.info("Production data: %d rows x %d cols", *frame.shape)
    return frame


def load_data_dictionary(data_dir: Path) -> pd.DataFrame:
    """Load the variable dictionary (semicolon separated).

    Args:
        data_dir: Folder containing the dictionary file.

    Returns:
        DataFrame indexed by variable name with description and type.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the expected ``Variable Name`` column is absent.
    """
    frame = _read_csv(
        data_dir / config.DICTIONARY_FILENAME,
        sep=config.DICTIONARY_SEPARATOR,
    )
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    validate_columns(frame, ["Variable Name"])
    return frame.set_index("Variable Name")
