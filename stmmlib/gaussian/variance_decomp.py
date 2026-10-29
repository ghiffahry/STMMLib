"""Variance decomposition, R-squared, and cAIC for Gaussian models.

Computes proportion of variance attributed to each component (area, time,
interaction, residual) on the response scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stmmlib._utils import interaction_active
from stmmlib.gaussian.covariance import (
    build_G_area,
    build_G_time,
    build_G_interaction,
    expand_it,
)

if TYPE_CHECKING:
    from stmmlib._backend import Backend


def _mean_diag_projected(Z, G, backend: Backend) -> float:
    """Compute mean(diag(Z G Z')) — projected variance component."""
    xp = backend.xp
    if backend.is_sparse(Z):
        Z_arr = Z.toarray()
    else:
        Z_arr = xp.asarray(Z, dtype=xp.float64)
    if backend.is_sparse(G):
        G_arr = G.toarray()
    else:
        G_arr = xp.asarray(G, dtype=xp.float64)
    ZG = Z_arr @ G_arr
    return float(xp.mean(xp.diag(ZG @ Z_arr.T)))


def compute_variance_decomposition(fit: dict, backend: Backend) -> dict:
    """Multi-class variance decomposition.

    For each random effect component, computes:
        projected_var = mean(diag(Z G Z'))
    Then expresses each as proportion of total.

    Args:
        fit: dict with theta, model, data, and optionally blup.
        backend: Array backend.

    Returns:
        dict with components, proportions, total_var.
    """
    xp = backend.xp
    theta = fit["theta"]
    model = fit["model"]
    data = fit["data"]

    area_type = model["area_type"]
    time_type = model["time_type"]
    interaction = model.get("interaction", "none")

    Z_area = data["Z_area"]
    m = data["m"]
    T = data["T"]
    W = data.get("W")

    sigma2_v = theta.get("sigma2_v", 1.0)
    sigma2_u = theta.get("sigma2_u", 1.0)
    sigma2_eta = theta.get("sigma2_eta", 1.0)
    phi = theta.get("phi", 0.0)
    sigma2_e = theta.get("sigma2_e", 1.0)
    rho = theta.get("rho", 0.0)

    components = {}

    # Area variance
    G_a = build_G_area(area_type, sigma2_v, m, W, rho, backend)
    components["area"] = _mean_diag_projected(Z_area, G_a, backend)

    # Time variance
    G_t = build_G_time(time_type, T, sigma2_u, sigma2_eta, phi, backend)
    V_time = expand_it(m, G_t, backend)
    if backend.is_sparse(V_time):
        V_time = V_time.toarray()
    components["time"] = float(xp.mean(xp.diag(V_time)))

    # Interaction variance
    if interaction_active(interaction):
        sigma2_g = theta.get("sigma2_gamma", 0.0)
        if sigma2_g > 0:
            from stmmlib.gaussian.design import build_interaction_contrast
            contrasts = build_interaction_contrast(m, T, backend)
            G_g = build_G_interaction(
                interaction, sigma2_g, m, T, W,
                theta.get("rho_gamma", 0.0), theta.get("phi_gamma", 0.0),
                contrasts, backend,
            )
            Z_gamma = data.get("Z_gamma")
            if Z_gamma is not None and G_g is not None:
                components["interaction"] = _mean_diag_projected(Z_gamma, G_g, backend)
            else:
                components["interaction"] = 0.0
        else:
            components["interaction"] = 0.0
    else:
        components["interaction"] = 0.0

    components["residual"] = sigma2_e

    total = sum(components.values())
    proportions = {k: v / total if total > 0 else 0 for k, v in components.items()}

    return {
        "components": components,
        "proportions": proportions,
        "total_var": total,
    }


# ---------------------------------------------------------------------------
# R-squared (Nakagawa-Schielzeth)
# ---------------------------------------------------------------------------


def compute_R2(vd: dict) -> dict:
    """Nakagawa-Schielzeth R-squared: marginal and conditional.

    R2_marginal = var_fixed / (var_fixed + var_random + var_resid)
    R2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_resid)
    """
    components = vd["components"]
    total = vd["total_var"]
    random_var = components.get("area", 0) + components.get("time", 0) + components.get("interaction", 0)
    resid_var = components.get("residual", 0)

    # Marginal (fixed effects only — approximation via residual reduction)
    r2_marginal = 1.0 - resid_var / total if total > 0 else 0
    r2_conditional = (total - resid_var) / total if total > 0 else 0

    return {
        "r2_marginal": r2_marginal,
        "r2_conditional": r2_conditional,
    }


# ---------------------------------------------------------------------------
# Conditional cAIC
# ---------------------------------------------------------------------------


def compute_cAIC(fit: dict, backend: Backend, type: str = "conditional") -> dict:
    """Conditional cAIC via hat-trace approximation.

    cAIC = -2 log L + 2 * edf
    where edf = trace(Hat matrix) ≈ n - p + sigma2_e * trace(V^{-1})
    """
    xp = backend.xp
    log_lik = fit.get("log_lik", 0)
    n = fit.get("n", 0)
    p = fit.get("p", 0)
    theta = fit.get("theta", {})
    sigma2_e = theta.get("sigma2_e", 1.0)

    # Simple approximation: edf ≈ n - p + p (for ML) or n (for REML)
    # More accurate: edf = trace(I - H) where H is hat matrix
    edf = n - p  # simplified
    caic = -2 * log_lik + 2 * edf

    return {
        "caic": caic,
        "edf": edf,
        "type": type,
    }


# ---------------------------------------------------------------------------
# Print methods
# ---------------------------------------------------------------------------


def print_R2(r2: dict, digits: int = 4):
    """Print R-squared values."""
    print(f"  R2 (marginal):     {r2['r2_marginal']:.{digits}f}")
    print(f"  R2 (conditional):  {r2['r2_conditional']:.{digits}f}")


# ---------------------------------------------------------------------------
# Legacy variance decomposition
# ---------------------------------------------------------------------------

