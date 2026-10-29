"""BLUP (Best Linear Unbiased Prediction) computation.

Computes conditional modes v_hat, u_hat, gamma_hat with optional
standard errors via the Henderson equations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stmmlib._utils import interaction_active, safe_cholesky
from stmmlib.gaussian.covariance import (
    build_G_area, build_G_time, build_G_interaction, expand_it, G_ar1_core,
)
from stmmlib.gaussian.gls import _solve_chol, safe_cholesky
from stmmlib.gaussian.transforms import get_theta_layout

if TYPE_CHECKING:
    from stmmlib._backend import Backend


def compute_blup(
    y, X, data: dict, model: dict, theta: dict, beta_hat,
    backend: Backend, compute_se: bool = True,
) -> dict:
    """Compute BLUPs: v_hat (area), u_hat (time), gamma_hat (interaction).

    Uses Henderson mixed model equations:
        [X'X  X'Z    ] [beta]   [X'y]
        [ZX   Z'Z+G^-1] [u  ] = [Z'y]

    Solved via the Cholesky of V.

    Args:
        y: Response vector.
        X: Fixed-effects design.
        data: Prepared data with design matrices.
        model: Model specification.
        theta: Variance parameters.
        beta_hat: Estimated fixed effects.
        backend: Array backend.

    Returns:
        dict with v_hat, u_hat, gamma_hat, se components.
    """
    xp = backend.xp
    n = len(y)
    y_arr = xp.asarray(y, dtype=xp.float64).ravel()
    X_arr = xp.asarray(X, dtype=xp.float64)

    Z_area = data["Z_area"]
    m = data["m"]
    T = data["T"]

    area_type = model["area_type"]
    time_type = model["time_type"]
    interaction = model.get("interaction", "none")

    sigma2_v = theta.get("sigma2_v", 1.0)
    sigma2_u = theta.get("sigma2_u", 1.0)
    sigma2_eta = theta.get("sigma2_eta", 1.0)
    phi = theta.get("phi", 0.0)
    sigma2_e = theta.get("sigma2_e", 1.0)
    rho = theta.get("rho", 0.0)
    W = data.get("W")

    # G_area inverse
    from stmmlib.gaussian.covariance import _to_np_backend
    G_a = _to_np_backend(backend, build_G_area(area_type, sigma2_v, m, W, rho, backend))
    G_a_inv = np.linalg.inv(G_a)

    # Residuals from fixed effects
    e = _to_np_backend(backend, y_arr) - _to_np_backend(backend, X_arr) @ _to_np_backend(backend, beta_hat)

    # v_hat = G_a Z_area' V^{-1} e  (area BLUP)
    # Simplified: v_hat = G_a Z_area' e / sigma2_e for IID
    if area_type == "iid":
        v_hat = sigma2_v * (_to_np_backend(backend, Z_area).T @ e) / sigma2_e
    else:
        # For SAR: use GLS-based BLUP
        from stmmlib.gaussian.covariance import build_V
        V = build_V(data, model, theta, backend)
        L = safe_cholesky(V, backend)
        Vinv_e = _solve_chol(L, e, backend)
        v_hat = G_a @ _to_np_backend(backend, Z_area).T @ Vinv_e

    # Time BLUP
    u_hat = xp.zeros(T, dtype=xp.float64)
    if time_type == "iid":
        u_hat = sigma2_u * (_to_np_backend(backend, data["Z_time"]).T @ e) / sigma2_e if "Z_time" in data else u_hat
    elif time_type == "ar1":
        G_t = _to_np_backend(backend, build_G_time(time_type, T, sigma2_u, sigma2_eta, phi, backend))
        if "Z_time" in data:
            from stmmlib.gaussian.covariance import build_V
            V = build_V(data, model, theta, backend)
            L = safe_cholesky(V, backend)
            Vinv_e = _solve_chol(L, e, backend)
            u_hat = G_t @ _to_np_backend(backend, data["Z_time"]).T @ Vinv_e

    # Interaction BLUP
    gamma_hat = xp.zeros(0, dtype=xp.float64)
    if interaction_active(interaction):
        Z_gamma = data.get("Z_gamma")
        if Z_gamma is not None:
            sigma2_g = theta.get("sigma2_gamma", 0.0)
            if sigma2_g > 0:
                from stmmlib.gaussian.design import build_interaction_contrast
                contrasts = build_interaction_contrast(m, T, backend)
                G_g = build_G_interaction(
                    interaction, sigma2_g, m, T, W,
                    theta.get("rho_gamma", 0.0),
                    theta.get("phi_gamma", 0.0),
                    contrasts, backend,
                )
                if G_g is not None:
                    from stmmlib.gaussian.covariance import build_V
                    V = build_V(data, model, theta, backend)
                    L = safe_cholesky(V, backend)
                    Vinv_e = _solve_chol(L, e, backend)
                    Zg = _to_np_backend(backend, Z_gamma)
                    Gg_np = _to_np_backend(backend, G_g)
                    gamma_hat = Gg_np @ Zg.T @ Vinv_e

    result = {
        "v_hat": v_hat,
        "u_hat": u_hat,
        "gamma_hat": gamma_hat,
        "beta_hat": beta_hat,
    }

    if compute_se:
        # Approximate SE from diagonal of G
        result["se_v"] = xp.sqrt(xp.diag(G_a))
        result["se_gamma"] = (
            xp.sqrt(sigma2_g) * xp.ones_like(gamma_hat) if len(gamma_hat) > 0
            else xp.zeros(0, dtype=xp.float64)
        )

    return result


def prediction_intervals(blup: dict, level: float = 0.95, backend: Backend = None) -> dict:
    """Wald confidence intervals for BLUP estimates.

    Args:
        blup: dict from compute_blup.
        level: Confidence level (default 0.95).
        backend: Array backend.

    Returns:
        dict with ci_v, ci_u, ci_gamma.
    """
    from scipy.stats import norm

    xp = backend.xp
    z = norm.ppf(1 - (1 - level) / 2)

    result = {}
    for key in ("v_hat", "u_hat", "gamma_hat"):
        est = blup.get(key, xp.zeros(0))
        se = blup.get(f"se_{key[0:1]}{key[2:]}", xp.ones_like(est))
        result[f"ci_{key}"] = (est - z * se, est + z * se)

    return result


# ---------------------------------------------------------------------------
# Nested BLUP
# ---------------------------------------------------------------------------


def _nested_henderson_blup(theta, y, X, prepared, interaction, beta_hat,
                           backend: Backend, ar1: bool):
    """Exact Henderson BLUP for nested models via V^{-1} (y - X beta).

    All four blocks share one V^{-1}e pass:
        b = G Z' V^{-1} e   for b in {v_prov, v_kab, u_it, gamma}.
    """
    from stmmlib.gaussian.covariance import build_V_nested, build_V_nested_ar1
    xp = backend.xp
    n_kab = prepared["n_kab"]
    T = prepared["T"]

    V = (build_V_nested_ar1 if ar1 else build_V_nested)(
        prepared, theta, interaction, backend)
    e = xp.asarray(y, dtype=xp.float64).ravel() \
        - xp.asarray(X, dtype=xp.float64) @ xp.asarray(beta_hat, dtype=xp.float64).ravel()
    L = safe_cholesky(V, backend)
    w = _solve_chol(L, e, backend).ravel()

    from stmmlib.gaussian.covariance import _to_np_backend
    Z_prov = _to_np_backend(backend, prepared["Z_prov"])
    Z_kab = _to_np_backend(backend, prepared["Z_kab"])
    Z_cell = prepared.get("Z_cell")
    Z_cell = None if Z_cell is None else _to_np_backend(backend, Z_cell)
    n_prov = prepared["n_prov"]
    sigma2_prov = theta.get("sigma2_prov", theta.get("sigma2_v", 1.0))
    sigma2_kab = theta.get("sigma2_kab", theta.get("sigma2_u", 1.0))

    v_prov = sigma2_prov * (Z_prov.T @ w)
    v_kab = sigma2_kab * (Z_kab.T @ w)

    # Cell-level u_it: G_cell = block-diag of G_t repeated n_kab times
    if ar1:
        G_t = _to_np_backend(backend, G_ar1_core(
            T, theta.get("sigma2_eta", 0.0), theta.get("phi", 0.0), backend))
    else:
        G_t = sigma2_kab * np.eye(T, dtype=np.float64)
    if Z_cell is not None:
        G_cell_full = np.kron(np.eye(n_kab, dtype=np.float64), G_t)
        u_it = G_cell_full @ (Z_cell.T @ w)
    else:
        u_it = np.zeros(n_kab * T, dtype=np.float64)

    # Interaction gamma
    gamma = np.zeros(0, dtype=np.float64)
    if interaction_active(interaction):
        Z_gamma = prepared.get("Z_gamma")
        sigma2_g = theta.get("sigma2_gamma", 0.0)
        if Z_gamma is not None and sigma2_g > 0:
            from stmmlib.gaussian.design import build_interaction_contrast
            contrasts = build_interaction_contrast(prepared["m"], T, backend)
            G_g = build_G_interaction(
                interaction, sigma2_g, prepared["m"], T, prepared.get("W"),
                theta.get("rho_gamma", 0.0), theta.get("phi_gamma", 0.0),
                contrasts, backend,
            )
            Gg_np = _to_np_backend(backend, G_g)
            Zg = _to_np_backend(backend, Z_gamma)
            gamma = Gg_np @ (Zg.T @ w)

    return {
        "v_prov": v_prov,
        "v_kab": v_kab,
        "u_it": u_it,
        "gamma": gamma,
        "beta_hat": beta_hat,
    }


def nested_blup(theta, y, X, prepared, interaction, beta_hat, backend: Backend):
    """BLUP for nested IID+IID: v_prov, v_kab, u_it, gamma."""
    return _nested_henderson_blup(theta, y, X, prepared, interaction,
                                  beta_hat, backend, ar1=False)


def nested_ar1_blup(theta, y, X, prepared, interaction, beta_hat, backend: Backend):
    """BLUP for nested IID+AR1: v_prov, v_kab, u_it (AR1), gamma."""
    return _nested_henderson_blup(theta, y, X, prepared, interaction,
                                  beta_hat, backend, ar1=True)
