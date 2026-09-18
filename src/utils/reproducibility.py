"""Seeding helpers so that every run produces the same results."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and the hash seed, and return a generator.

    Args:
        seed: Non-negative integer seed.

    Returns:
        A NumPy ``Generator`` seeded with ``seed``.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
