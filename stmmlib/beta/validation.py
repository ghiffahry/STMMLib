"""Input validation and range checks for Beta models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_y(y: np.ndarray) -> np.ndarray:
    """Validate response is in (0,1) for Beta family.

    Raises ValueError if any values outside open interval.
    """
    y = np.asarray(y, dtype=np.float64)
    if np.any(y <= 0) or np.any(y >= 1):
        raise ValueError("Beta response y must be in (0,1). Found values outside this range.")
    return y


def validate_model(model: str) -> str:
    """Validate model string is known."""
    from stmmlib.beta.structure import BETA_MODELS
    if model not in BETA_MODELS:
        raise KeyError(f"Unknown model: {model!r}. Available: {list(BETA_MODELS.keys())}")
    return model


