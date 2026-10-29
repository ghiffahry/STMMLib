"""Multi-start optimization for variance parameters.

Outer L-BFGS-B optimization over raw (unconstrained) parameters.
Includes auto-starting values and model-specific fit loops.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from stmmlib.gaussian.transforms import (
    CORR_BOUND_DEFAULT,
    get_param_bounds,
    get_theta_layout,
    raw_to_theta,
    theta_to_raw,
    _with_preserved_seed,
)

if TYPE_CHECKING:
    from stmmlib._backend import Backend


# ---------------------------------------------------------------------------
# Auto starting values
# ---------------------------------------------------------------------------


def auto_start(y, X, model: dict, backend: Backend) -> dict:
    """Generate starting theta from OLS residuals.

    Estimates residual variance from OLS, sets area/time variances to
    reasonable fractions.
    """
    xp = backend.xp
    y_arr = xp.asarray(y, dtype=xp.float64).ravel()
    X_arr = xp.asarray(X, dtype=xp.float64)

    # OLS residuals
    beta_ols = xp.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
    resid = y_arr - X_arr @ beta_ols
    sigma2_e = float(xp.mean(resid**2))

    layout = get_theta_layout(model)
    theta = {}
    for p in layout:
        if p == "sigma2_e":
            theta[p] = max(sigma2_e, 1e-6)
        elif p == "sigma2_v":
            theta[p] = max(sigma2_e * 0.5, 1e-6)
        elif p == "sigma2_u":
            theta[p] = max(sigma2_e * 0.3, 1e-6)
        elif p == "sigma2_eta":
            theta[p] = max(sigma2_e * 0.3, 1e-6)
        elif p in ("rho", "phi"):
            theta[p] = 0.1
        elif p.startswith("sigma2_"):
            theta[p] = max(sigma2_e * 0.2, 1e-6)
        elif p in ("rho_gamma", "phi_gamma"):
            theta[p] = 0.0
        else:
            theta[p] = 1.0
    return theta


# ---------------------------------------------------------------------------
# Core optimization loop
# ---------------------------------------------------------------------------


def run_optimization(
    y,
    X,
    data: dict,
    model: dict,
    backend: Backend,
    start: dict = None,
    method: str = "REML",
    n_starts: int = 3,
    max_iter: int = 200,
    corr_bound: float = CORR_BOUND_DEFAULT,
    verbose: bool = False,
    loglik_func=None,
) -> dict:
    """Multi-start L-BFGS-B optimization of variance parameters.

    Args:
        y: Response vector.
        X: Fixed-effects design.
        data: Prepared data dict.
        model: Model specification.
        backend: Array backend.
        start: Starting theta (dict). If None, uses auto_start.
        method: "ML" or "REML".
        n_starts: Number of random restarts.
        max_iter: Max iterations per start.
        corr_bound: Correlation parameter bound.
        verbose: Print progress.
        loglik_func: Custom loglik function. Default: logLik_REML.

    Returns:
        dict with theta_hat, log_lik, converged, raw, n_starts_used.
    """
    if loglik_func is None:
        from stmmlib.gaussian.gls import logLik_REML, logLik_ML
        loglik_func = logLik_REML if method == "REML" else logLik_ML

    bounds_dict = get_param_bounds(model, corr_bound)
    layout = get_theta_layout(model)

    def neg_loglik(raw):
        theta = raw_to_theta(raw, model, corr_bound)
        try:
            ll = loglik_func(theta, y, X, data, model, backend)
            return -ll
        except Exception:
            return 1e20

    # Get bounds in raw space
    raw_bounds = []
    for p in layout:
        lo, hi = bounds_dict.get(p, (-20, 20))
        # Transform bounds to raw space
        if p.startswith("sigma2_"):
            raw_bounds.append((np.log(max(lo, 1e-15)), np.log(hi) if hi is not None else None))
        elif p in ("rho", "phi", "rho_gamma", "phi_gamma"):
            raw_bounds.append((-np.arctanh(min(abs(lo), 0.999)), np.arctanh(min(abs(hi), 0.999)) if hi is not None else np.arctanh(0.999)))
        else:
            raw_bounds.append((lo, hi))

    best_result = None
    best_val = np.inf

    # Starting point 0: provided or auto
    starts = []
    if start is not None:
        starts.append(start)
    else:
        starts.append(auto_start(y, X, model, backend))

    # Additional random starts
    for i in range(n_starts - 1):
        perturbed = starts[0].copy()
        for p in layout:
            if p.startswith("sigma2_"):
                perturbed[p] *= np.exp(np.random.normal(0, 0.15))
            elif p in ("rho", "phi", "rho_gamma", "phi_gamma"):
                perturbed[p] += np.random.normal(0, 0.1)
                perturbed[p] = np.clip(perturbed[p], -corr_bound * 0.9, corr_bound * 0.9)
        starts.append(perturbed)

    for i, start_theta in enumerate(starts):
        try:
            raw0 = theta_to_raw(start_theta, model, corr_bound)
            res = minimize(
                neg_loglik,
                raw0,
                method="L-BFGS-B",
                bounds=raw_bounds,
                options={"maxiter": max_iter, "ftol": 1e-10},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res
                if verbose:
                    print(f"Start {i+1}: loglik={-res.fun:.4f}, converged={res.success}")
        except Exception as e:
            if verbose:
                print(f"Start {i+1} failed: {e}")
            continue

    if best_result is None:
        raise RuntimeError("All optimization starts failed.")

    theta_hat = raw_to_theta(best_result.x, model, corr_bound)

    return {
        "theta_hat": theta_hat,
        "log_lik": -best_result.fun,
        "converged": best_result.success,
        "raw": best_result.x,
        "n_starts_used": len(starts),
        "optim_result": best_result,
    }


# ---------------------------------------------------------------------------
# SEBLUP5 fit loop
# ---------------------------------------------------------------------------


def fit_seblup5(
    y, X, prepared, interaction, area_type, backend: Backend,
    method="REML", start=None, n_starts=3, max_iter=200,
    verbose=False, corr_bound=CORR_BOUND_DEFAULT,
):
    """Full optimization loop for SEBLUP5 model."""
    from stmmlib.gaussian.gls import logLik_seblup5
    from stmmlib.gaussian.transforms import seblup5_layout, seblup5_theta_to_raw, seblup5_raw_to_theta

    layout = seblup5_layout(interaction, area_type)

    def neg_ll(raw):
        theta = seblup5_raw_to_theta(raw, interaction, area_type, corr_bound)
        try:
            ll = logLik_seblup5(theta, y, X, prepared, interaction, area_type, method, backend)
            return -ll
        except Exception:
            return 1e20

    raw_bounds = []
    for p in layout:
        if p.startswith("sigma2_"):
            raw_bounds.append((np.log(1e-10), None))
        elif p in ("rho", "phi", "rho_gamma", "phi_gamma"):
            raw_bounds.append((-np.arctanh(0.99), np.arctanh(0.99)))
        else:
            raw_bounds.append((-20, 20))

    best_val = np.inf
    best_res = None

    # Auto start
    from stmmlib.gaussian.gls import gls
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = np.asarray(y).ravel() - np.asarray(X) @ beta_ols
    sigma2_e = float(np.mean(resid**2))

    start_theta = {p: 0.1 if p in ("rho", "phi") else max(sigma2_e * 0.3, 1e-4) for p in layout}
    start_theta["sigma2_e"] = max(sigma2_e, 1e-6)
    if start is not None:
        start_theta.update(start)

    starts = [start_theta]
    for _ in range(n_starts - 1):
        pert = start_theta.copy()
        for p in layout:
            if p.startswith("sigma2_"):
                pert[p] *= np.exp(np.random.normal(0, 0.15))
        starts.append(pert)

    for i, st in enumerate(starts):
        try:
            raw0 = seblup5_theta_to_raw(st, interaction, area_type, corr_bound)
            res = minimize(neg_ll, raw0, method="L-BFGS-B", bounds=raw_bounds,
                          options={"maxiter": max_iter, "ftol": 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
        except Exception:
            continue

    if best_res is None:
        raise RuntimeError("SEBLUP5 optimization failed.")

    theta_hat = seblup5_raw_to_theta(best_res.x, interaction, area_type, corr_bound)
    return {"theta_hat": theta_hat, "log_lik": -best_res.fun, "converged": best_res.success}


# ---------------------------------------------------------------------------
# Nested fit loops
# ---------------------------------------------------------------------------


def nested_fit(
    y, X, prepared, interaction, backend: Backend,
    method="REML", start=None, n_starts=3, max_iter=200, verbose=False,
):
    """Full optimization loop for nested IID+IID model."""
    from stmmlib.gaussian.gls import logLik_nested
    from stmmlib.gaussian.transforms import nested_layout, nested_theta_to_raw, nested_raw_to_theta

    layout = nested_layout(interaction)

    def neg_ll(raw):
        theta = nested_raw_to_theta(raw, interaction)
        try:
            return -logLik_nested(theta, y, X, prepared, interaction, method, backend)
        except Exception:
            return 1e20

    raw_bounds = [(np.log(1e-10), None)] * len(layout)

    best_val = np.inf
    best_res = None
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = np.asarray(y).ravel() - np.asarray(X) @ beta_ols
    sigma2_e = float(np.mean(resid**2))

    start_theta = {p: max(sigma2_e * 0.3, 1e-4) for p in layout}
    start_theta["sigma2_e"] = max(sigma2_e, 1e-6)
    if start:
        start_theta.update(start)

    for _ in range(n_starts):
        try:
            raw0 = nested_theta_to_raw(start_theta, interaction)
            res = minimize(neg_ll, raw0, method="L-BFGS-B", bounds=raw_bounds,
                          options={"maxiter": max_iter, "ftol": 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
        except Exception:
            break

    if best_res is None:
        raise RuntimeError("Nested optimization failed.")
    theta_hat = nested_raw_to_theta(best_res.x, interaction)
    return {"theta_hat": theta_hat, "log_lik": -best_res.fun, "converged": best_res.success}


def nested_ar1_fit(
    y, X, prepared, interaction, backend: Backend,
    method="REML", start=None, n_starts=3, max_iter=200, verbose=False,
):
    """Full optimization loop for nested IID+AR1 model."""
    from stmmlib.gaussian.gls import logLik_nested_ar1
    from stmmlib.gaussian.transforms import nested_ar1_layout, nested_ar1_theta_to_raw, nested_ar1_raw_to_theta

    layout = nested_ar1_layout(interaction)

    def neg_ll(raw):
        theta = nested_ar1_raw_to_theta(raw, interaction)
        try:
            return -logLik_nested_ar1(theta, y, X, prepared, interaction, method, backend)
        except Exception:
            return 1e20

    raw_bounds = [(np.log(1e-10), None)] * len(layout)

    best_val = np.inf
    best_res = None
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = np.asarray(y).ravel() - np.asarray(X) @ beta_ols
    sigma2_e = float(np.mean(resid**2))

    start_theta = {p: max(sigma2_e * 0.3, 1e-4) for p in layout}
    start_theta["sigma2_e"] = max(sigma2_e, 1e-6)
    start_theta["phi"] = 0.1
    if start:
        start_theta.update(start)

    for _ in range(n_starts):
        try:
            raw0 = nested_ar1_theta_to_raw(start_theta, interaction)
            res = minimize(neg_ll, raw0, method="L-BFGS-B", bounds=raw_bounds,
                          options={"maxiter": max_iter, "ftol": 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
        except Exception:
            break

    if best_res is None:
        raise RuntimeError("Nested AR1 optimization failed.")
    theta_hat = nested_ar1_raw_to_theta(best_res.x, interaction)
    return {"theta_hat": theta_hat, "log_lik": -best_res.fun, "converged": best_res.success}
