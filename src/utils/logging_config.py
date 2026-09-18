"""Root-logger configuration, called once from the entry point."""
from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"
NOISY_LOGGERS = ("matplotlib", "PIL", "numba", "shap")


def configure_logging(
    level: str = "INFO", log_file: Path | None = None
) -> None:
    """Configure the root logger with a console and optional file handler.

    Args:
        level: Logging level name (``DEBUG``, ``INFO``, ...).
        log_file: Optional file where the log is also written.

    Raises:
        ValueError: If ``level`` is not a valid logging level name.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level!r}")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
