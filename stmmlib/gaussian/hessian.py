"""Finite-difference Hessian and delta-method standard errors.

Computes Hessian of the log-likelihood with respect to raw parameters,
then derives standard errors via delta method for the transformed parameters.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from stmmlib.gaussian.transforms import (
    CORR_BOUND_DEFAULT,
    get_theta_layout,
    raw_to_theta,
    theta_to_raw,
)

if TYPE_CHECKING:
    from stmmlib._backend import Backend


# ---------------------------------------------------------------------------
# Finite-difference Hessian
# ---------------------------------------------------------------------------


def hessian_fd(
    theta_hat: dict, y, X, data: dict, model: dict,
    method: str, backend: Backend, h: float = 1e-5,
    loglik_func=None,
):
    """Compute Hessian of log-likelihood on raw parameter scale via central differences.

    Args:
        theta_hat: Estimated variance parameters.
        y: Response vector.
        X: Fixed-effects design.
        data: Prepared data.
        model: Model specification.
        method: "ML" or "REML".
        backend: Array backend.
        h: Finite-difference step size.
        loglik_func: Custom loglik function.

    Returns:
        H_raw: p x p Hessian matrix on raw scale.
    """
    if loglik_func is None:
        from stmmlib.gaussian.gls import logLik_REML, logLik_ML
        loglik_func = logLik_REML if method == "REML" else logLik_ML

    raw0 = theta_to_raw(theta_hat, model)
    p = len(raw0)

    H = np.zeros((p, p), dtype=np.float64)

    def _eval(raw_vec):
        theta = raw_to_theta(raw_vec, model)
        return loglik_func(theta, y, X, data, model, backend)

    f0 = _eval(raw0)

    # Central differences
    for i in range(p):
        ei = np.zeros(p, dtype=np.float64)
        ei[i] = h
        f_plus = _eval(raw0 + ei)
        f_minus = _eval(raw0 - ei)
        H[i, i] = (f_plus - 2 * f0 + f_minus) / (h * h)
        for j in range(i + 1, p):
            ej = np.zeros(p, dtype=np.float64)
            ej[j] = h
            f_pp = _eval(raw0 + ei + ej)
            f_pm = _eval(raw0 + ei - ej)
            f_mp = _eval(raw0 - ei + ej)
            f_mm = _eval(raw0 - ei - ej)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
            H[j, i] = H[i, j]

    return H


# ---------------------------------------------------------------------------
# Delta-method SE
# ---------------------------------------------------------------------------


def delta_se(theta_hat: dict, H_inv: np.ndarray, model: dict, corr_bound: float = CORR_BOUND_DEFAULT):
    """Compute standard errors via delta method for transformed parameters.

    For theta_raw -> theta_constrained transform:
        SE_theta = |dtheta/draw| * SE_raw

    Args:
        theta_hat: Estimated parameters.
        H_inv: Inverse Hessian on raw scale.
        model: Model specification.
        corr_bound: Correlation bound.

    Returns:
        dict with se_theta (SE for each parameter on constrained scale).
    """
    layout = get_theta_layout(model)
    raw0 = theta_to_raw(theta_hat, model)
    p = len(layout)
    se_raw = np.sqrt(np.clip(np.diag(H_inv), 0, None))

    se_theta = {}
    for i, param in enumerate(layout):
        val = raw0[i]
        if param.startswith("sigma2_"):
            # dtheta/draw = exp(raw) = theta
            deriv = np.exp(val)
        elif param in ("rho", "phi", "rho_gamma", "phi_gamma"):
            # dtheta/draw = corr_bound * sech^2(raw) = corr_bound * (1 - tanh^2(raw))
            deriv = corr_bound * (1.0 - np.tanh(val) ** 2)
        else:
            deriv = 1.0
        se_theta[param] = float(abs(deriv) * se_raw[i])

    return se_theta


# ---------------------------------------------------------------------------
# Full Hessian pipeline
# ---------------------------------------------------------------------------


def compute_hessian_se(
    theta_hat: dict, y, X, data: dict, model: dict,
    method: str, backend: Backend, h: float = 1e-5,
    verbose: bool = False, loglik_func=None,
) -> dict:
    """Full Hessian computation: FD, PD check, SE, CI.

    Args:
        theta_hat: Estimated parameters.
        y: Response.
        X: Fixed-effects design.
        data: Prepared data.
        model: Model specification.
        method: "ML" or "REML".
        backend: Array backend.
        h: Finite-difference step.
        verbose: Print diagnostics.

    Returns:
        dict with H_raw, H_inv, se_raw, se_theta, ci_95.
    """
    H_raw = hessian_fd(theta_hat, y, X, data, model, method, backend, h, loglik_func)

    # Symmetrize
    H_raw = 0.5 * (H_raw + H_raw.T)

    # Check PD
    from stmmlib._utils import is_pd, near_pd

    if is_pd(H_raw, backend=None):
        H_inv = np.linalg.inv(H_raw)
    else:
        warnings.warn("Hessian not PD, using near_pd projection.")
        H_pd = near_pd(H_raw, backend=None)
        H_inv = np.linalg.inv(H_pd)

    se_raw = np.sqrt(np.clip(np.diag(H_inv), 0, None))
    se_theta = delta_se(theta_hat, H_inv, model)

    # 95% CI on constrained scale
    z = 1.96
    ci_95 = {}
    layout = get_theta_layout(model)
    for param in layout:
        val = theta_hat[param]
        se = se_theta[param]
        ci_95[param] = (val - z * se, val + z * se)

    if verbose:
        print("Hessian SEs:")
        for p in layout:
            print(f"  {p}: {theta_hat[p]:.4f} (SE={se_theta[p]:.4f})")

    return {
        "H_raw": H_raw,
        "H_inv": H_inv,
        "se_raw": se_raw,
        "se_theta": se_theta,
        "ci_95": ci_95,
    }


# ---------------------------------------------------------------------------
# Model-specific Hessian variants
# ---------------------------------------------------------------------------

