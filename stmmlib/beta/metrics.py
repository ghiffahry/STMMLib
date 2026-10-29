"""Scoring metrics for Beta regression: MSE, RMSE, MAE, logLik, AIC, BIC."""

from __future__ import annotations

import numpy as np
from scipy.special import betaln


def betaml_metrics(y: np.ndarray, y_hat: np.ndarray, phi: float, k: int = 2) -> dict:
    """Compute Beta log-likelihood, MSE, RMSE, MAE, AIC, BIC.

    Args:
        y: True values in (0,1).
        y_hat: Predicted mean (mu).
        phi: Precision parameter.
        k: Number of parameters.

    Returns:
        dict with logLik, MSE, RMSE, MAE, AIC, BIC.
    """
    mu = np.clip(y_hat, 1e-15, 1.0 - 1e-15)
    a = mu * phi
    b = (1 - mu) * phi

    loglik = float(np.sum(-betaln(a, b) + (a - 1) * np.log(y) + (b - 1) * np.log(1 - y)))

    resid = y - mu
    n = len(y)
    mse = float(np.mean(resid ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(resid)))

    aic = float(-2 * loglik + 2 * k)
    bic = float(-2 * loglik + k * np.log(n))

    return {
        "logLik": loglik,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "AIC": aic,
        "BIC": bic,
        "n": n,
        "k": k,
    }


