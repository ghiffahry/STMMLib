"""Out-of-sample evaluation: MAPE, MAD, MSE, REL, coverage."""

from __future__ import annotations

import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def mad(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Deviation."""
    return float(np.median(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical coverage: proportion of y_true in [lower, upper]."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average interval width."""
    return float(np.mean(upper - lower))
