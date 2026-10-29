"""GLS solver, likelihood evaluation (ML/REML), AIC/BIC computation.

Core Gaussian fitting: given y, X, V -> solve for beta_hat, residuals,
log-likelihood, variance decomposition.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from stmmlib._utils import logdet_matrix, safe_cholesky, safe_solve
from stmmlib.gaussian.covariance import build_V

if TYPE_CHECKING:
    from stmmlib._backend import Backend


# ---------------------------------------------------------------------------
# GLS solver
# ---------------------------------------------------------------------------


def gls(y, X, V, backend: Backend) -> dict:
    """Generalized Least Squares solver.

    beta_hat = (X' V^{-1} X)^{-1} X' V^{-1} y

    Args:
        y: Response vector (n,).
        X: Fixed-effects design matrix (n, p).
        V: Marginal covariance (n, n).
        backend: Array backend.

    Returns:
        dict with beta_hat, se_beta, vcov_beta, residuals, fitted,
        log_lik_components (chol, quad_form, logdet_V).
    """
    xp = backend.xp
    y = xp.asarray(y, dtype=xp.float64).ravel()
    X = xp.asarray(X, dtype=xp.float64)
    n, p = X.shape

    # Cholesky of V for efficient solve
    L = safe_cholesky(V, backend)
    # _solve_chol returns CPU numpy (scipy); push back to the active device
    Vinv_X = xp.asarray(_solve_chol(L, X, backend))
    Vinv_y = xp.asarray(_solve_chol(L, y, backend))

    XtVinvX = X.T @ Vinv_X
    XtVinvy = X.T @ Vinv_y

    # Solve for beta
    beta_hat = safe_solve(XtVinvX, XtVinvy, backend)

    # Fitted values and residuals
    fitted = X @ beta_hat
    residuals = y - fitted

    # Variance-covariance of beta
    vcov_beta = safe_solve(XtVinvX, xp.eye(p, dtype=xp.float64), backend)
    se_beta = xp.sqrt(xp.clip(xp.diag(vcov_beta), 0, None))

    # Log-likelihood components
    quad_form = float(y @ Vinv_y - 2.0 * beta_hat @ XtVinvy + beta_hat @ XtVinvX @ beta_hat)
    logdet_V = logdet_matrix(V, backend)

    return {
        "beta_hat": beta_hat,
        "se_beta": se_beta,
        "vcov_beta": vcov_beta,
        "residuals": residuals,
        "fitted": fitted,
        "quad_form": quad_form,
        "logdet_V": logdet_V,
        "L": L,
    }


def _to_np(backend: Backend, a):
    """Convert any backend array (numpy/cupy/sparse) to a dense CPU numpy array.

    Needed because scipy (solve_triangular, kron, ...) cannot consume cupy
    arrays; cupy arrays are pulled to host with .get().
    """
    import numpy as np
    if backend.is_sparse(a):
        a = a.toarray()
    if hasattr(a, "get"):  # cupy -> numpy
        a = a.get()
    return np.asarray(a, dtype=np.float64)


def _solve_chol(L, b, backend: Backend):
    """Solve L L' x = b via forward/back substitution.

    Uses scipy.linalg.solve_triangular on dense CPU arrays so it works
    regardless of whether backend.xp is numpy or cupy (both lack
    xp.linalg.solve_triangular). GPU arrays are pulled to host.
    """
    import numpy as np
    from scipy.linalg import solve_triangular
    L = _to_np(backend, L)
    b = _to_np(backend, b)
    # Forward: L z = b
    z = solve_triangular(L, b, lower=True)
    # Backward: L' x = z
    x = solve_triangular(L.T, z, lower=False)
    return x


# ---------------------------------------------------------------------------
# Common components
# ---------------------------------------------------------------------------


def common_components(theta, y, X, data, model, backend: Backend) -> dict:
    """Shared computation: build V, GLS, residuals, quad form, log-det.

    Args:
        theta: Variance parameter dict.
        y: Response.
        X: Fixed-effects design.
        data: Prepared panel data with design matrices.
        model: Model specification dict.
        backend: Array backend.

    Returns:
        dict with all GLS components plus V.
    """
    V = build_V(data, model, theta, backend)
    result = gls(y, X, V, backend)
    result["V"] = V
    result["theta"] = theta
    return result


# ---------------------------------------------------------------------------
# Likelihood evaluation
# ---------------------------------------------------------------------------


def logLik_ML(theta, y, X, data, model, backend: Backend, return_components: bool = False):
    """Gaussian log-likelihood under ML (not REML).

    log L = -n/2 log(2pi) - 1/2 logdet(V) - 1/2 (y-Xb)' V^{-1} (y-Xb)
    """
    from stmmlib.gaussian.transforms import validate_theta

    xp = backend.xp
    validate_theta(theta, model, context="logLik_ML")
    n = len(y)

    comps = common_components(theta, y, X, data, model, backend)
    log_lik = -0.5 * n * np.log(2 * np.pi) - 0.5 * comps["logdet_V"] - 0.5 * comps["quad_form"]

    if return_components:
        comps["log_lik"] = float(log_lik)
        return comps
    return float(log_lik)


def logLik_REML(theta, y, X, data, model, backend: Backend, return_components: bool = False):
    """Restricted (REML) log-likelihood.

    log L_R = -1/2[(n-p)log(2pi) + logdet(V) + logdet(X'V^{-1}X) + (y-Xb)'V^{-1}(y-Xb)]
    """
    from stmmlib.gaussian.transforms import validate_theta

    xp = backend.xp
    validate_theta(theta, model, context="logLik_REML")
    n = len(y)
    p = X.shape[1]

    comps = common_components(theta, y, X, data, model, backend)

    # logdet(X'V^{-1}X)
    L = comps["L"]
    Vinv_X = _solve_chol(L, X, backend)
    XtVinvX = X.T @ Vinv_X
    logdet_XtVinvX = logdet_matrix(XtVinvX, backend)

    log_lik = (
        -0.5 * (n - p) * np.log(2 * np.pi)
        - 0.5 * comps["logdet_V"]
        - 0.5 * logdet_XtVinvX
        - 0.5 * comps["quad_form"]
    )

    if return_components:
        comps["log_lik"] = float(log_lik)
        return comps
    return float(log_lik)


# ---------------------------------------------------------------------------
# Model-specific likelihoods
# ---------------------------------------------------------------------------


def logLik_seblup5(theta, y, X, prepared, interaction, area_type, method, backend, return_components=False):
    """Log-likelihood for SEBLUP5 model."""
    from stmmlib.gaussian.covariance import build_V_seblup5

    V = build_V_seblup5(prepared, theta, interaction, area_type, backend)
    result = gls(y, X, V, backend)
    n = len(y)
    p = X.shape[1]

    logdet_V = result["logdet_V"]
    quad_form = result["quad_form"]

    if method == "REML":
        L = result["L"]
        Vinv_X = _solve_chol(L, X, backend)
        XtVinvX = X.T @ Vinv_X
        logdet_X = logdet_matrix(XtVinvX, backend)
        log_lik = -0.5 * (n - p) * np.log(2 * np.pi) - 0.5 * logdet_V - 0.5 * logdet_X - 0.5 * quad_form
    else:
        log_lik = -0.5 * n * np.log(2 * np.pi) - 0.5 * logdet_V - 0.5 * quad_form

    if return_components:
        result["log_lik"] = float(log_lik)
        return result
    return float(log_lik)


def logLik_nested(theta, y, X, prepared, interaction, method, backend, return_components=False):
    """Log-likelihood for nested IID+IID model."""
    from stmmlib.gaussian.covariance import build_V_nested

    V = build_V_nested(prepared, theta, interaction, backend)
    result = gls(y, X, V, backend)
    n = len(y)
    p = X.shape[1]

    if method == "REML":
        L = result["L"]
        Vinv_X = _solve_chol(L, X, backend)
        XtVinvX = X.T @ Vinv_X
        logdet_X = logdet_matrix(XtVinvX, backend)
        log_lik = -0.5 * (n - p) * np.log(2 * np.pi) - 0.5 * result["logdet_V"] - 0.5 * logdet_X - 0.5 * result["quad_form"]
    else:
        log_lik = -0.5 * n * np.log(2 * np.pi) - 0.5 * result["logdet_V"] - 0.5 * result["quad_form"]

    if return_components:
        result["log_lik"] = float(log_lik)
        return result
    return float(log_lik)


def logLik_nested_ar1(theta, y, X, prepared, interaction, method, backend, return_components=False):
    """Log-likelihood for nested IID+AR1 model."""
    from stmmlib.gaussian.covariance import build_V_nested_ar1

    V = build_V_nested_ar1(prepared, theta, interaction, backend)
    result = gls(y, X, V, backend)
    n = len(y)
    p = X.shape[1]

    if method == "REML":
        L = result["L"]
        Vinv_X = _solve_chol(L, X, backend)
        XtVinvX = X.T @ Vinv_X
        logdet_X = logdet_matrix(XtVinvX, backend)
        log_lik = -0.5 * (n - p) * np.log(2 * np.pi) - 0.5 * result["logdet_V"] - 0.5 * logdet_X - 0.5 * result["quad_form"]
    else:
        log_lik = -0.5 * n * np.log(2 * np.pi) - 0.5 * result["logdet_V"] - 0.5 * result["quad_form"]

    if return_components:
        result["log_lik"] = float(log_lik)
        return result
    return float(log_lik)


# ---------------------------------------------------------------------------
# AIC / BIC
# ---------------------------------------------------------------------------


def compute_AIC(log_lik: float, k: int) -> float:
    """Akaike Information Criterion: AIC = -2 log L + 2k."""
    return -2.0 * log_lik + 2.0 * k
