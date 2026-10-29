"""Beta model diagnostics: residuals, ACF, permutation tests, visual diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from stmmlib._backend import Backend


def compute_beta_marginal_resid(result) -> dict:
    """Compute marginal residuals: y_hat - mu_hat from X @ beta."""
    beta = result.beta
    design = result.design
    if not design or "Z" not in design:
        raise ValueError("Result missing design info.")
    from stmmlib.beta.design import stmm_beta_make_design
    # shortcut: design tidak pernah menyimpan X; matrix design ada di _opt
    X = design.get("X") or result._opt.get("X")
    if X is None:
        n = len(result._opt.get("y", []))
        X = np.ones((n, 1), dtype=np.float64)
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
    y = result._opt.get("y", None)
    if y is None:
        raise ValueError("Response not found in result.")
    resid = y - mu
    return {"mu": mu, "resid_marginal": resid}


def compute_beta_cond_resid(result) -> dict:
    """Compute conditional residuals using fitted mu."""
    from scipy.special import digamma
    y = result._opt.get("y")
    if y is None:
        raise ValueError("Response not found.")
    mu = result._opt.get("mu_hat", None)
    if mu is None:
        from scipy.special import digamma
        beta = result.beta
        X = result._opt.get("X")
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
    phi = result.phi
    a = mu * phi
    b = (1.0 - mu) * phi
    denom = digamma(a) + digamma(b)
    raw_resid = (y - mu) / np.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))
    return {"mu": mu, "resid_cond": raw_resid}


def compute_beta_random_resid(result) -> dict:
    """Randomized quantile residuals (Dunn-Smyth 1996).

    For a CONTINUOUS response (beta) the randomization interval collapses:
    r_i = Phi^-1(F_beta(y_i; mu_i, phi)). No jitter is needed (the earlier
    implementation mistakenly used the beta DENSITY in place of the CDF,
    which produced spuriously extreme tails). Probabilities are clipped to
    [1e-12, 1-1e-12] to keep ndtri finite when y sits at a fitted boundary.
    """
    y = result._opt.get("y")
    if y is None:
        raise ValueError("Response not found.")
    mu = result._opt.get("mu_hat", None)
    if mu is None:
        beta = result.beta
        X = result._opt.get("X")
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
    phi = result.phi
    from scipy.stats import beta as _beta_dist
    u = _beta_dist.cdf(y, mu * phi, (1.0 - mu) * phi)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    resid = norm.ppf(u)
    return {"mu": mu, "resid_random": resid}


def acf(x, nlags=10):
    """Simple autocorrelation function."""
    n = len(x)
    x = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.zeros(nlags + 1)
    return np.array([np.mean(x[:n - k] * x[k:]) / var for k in range(nlags + 1)])


