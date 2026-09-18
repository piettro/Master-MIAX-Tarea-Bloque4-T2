"""Writers for the machine-readable and Markdown reports."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_serialisable(value: Any) -> Any:
    """Convert NumPy / pandas objects into JSON-friendly Python types."""
    if isinstance(value, dict):
        return {str(k): _to_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _to_serialisable(value.reset_index().to_dict("records"))
    if isinstance(value, pd.Series):
        return _to_serialisable(value.to_dict())
    if isinstance(value, np.generic):
        return _to_serialisable(value.item())
    if isinstance(value, np.ndarray):
        return _to_serialisable(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)  # Paths, pandas Intervals, ...


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write a dictionary as indented JSON.

    Args:
        data: Content to write; NumPy / pandas values are converted.
        path: Destination file.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(_to_serialisable(data), indent=2), encoding="utf-8"
        )
    except OSError as exc:
        raise OSError(f"Could not write JSON report to {path}") from exc
    logger.info("Wrote %s", path)


def dataframe_to_markdown(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Args:
        frame: Table to render; the index is included as first column.
        float_digits: Decimals used for floating-point values.

    Returns:
        The Markdown table as a string.
    """
    table = frame.reset_index()

    def fmt(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.{float_digits}f}"
        return str(value)

    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    separator = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def write_text(text: str, path: Path) -> None:
    """Write a UTF-8 text file, creating parent folders when needed.

    Args:
        text: Content to write.
        path: Destination file.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not write {path}") from exc
    logger.info("Wrote %s", path)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV.

    Args:
        frame: Table to write (index included).
        path: Destination CSV file.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(path)
    except OSError as exc:
        raise OSError(f"Could not write table to {path}") from exc
    logger.debug("Wrote %s", path)
