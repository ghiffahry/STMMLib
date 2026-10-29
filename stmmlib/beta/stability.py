"""Model stability diagnostics for Beta models: VIF, confidence intervals, robust SE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stmmlib._backend import Backend


def variance_inflation_factor(X: np.ndarray, idx: int) -> float:
    """VIF for the idx-th column of X."""
    n, p = X.shape
    y = X[:, idx]
    X_rest = np.delete(X, idx, axis=1)
    X_rest_aug = np.column_stack([np.ones(n), X_rest])
    hat = X_rest_aug @ np.linalg.pinv(X_rest_aug)
    y_hat = hat @ y
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return 1 / (1 - r_sq) if r_sq < 1 else np.inf


