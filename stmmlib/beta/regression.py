"""Fixed-effects Beta regression (no random effects)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def beta_regression_base(formula, data, y, method="ML", model="beta_regression"):
    """Fixed-effects Beta ML (beta_regression, no random effects): L-BFGS-B
    optim + Hessian SEs.

    X is built from ALL non-grouping columns of ``data`` (intercept first),
    so multi-predictor beta regression works; a data frame with only the
    response yields the legacy intercept-only fit. Standardization is NOT
    applied here (plain ML regression path).
    """
    n = len(y)
    _skip = {".stmm_area", ".stmm_time", ".stmm_prov", ".stmm_kab", "y"}
    x_cols = [c for c in getattr(data, "columns", []) if c not in _skip]
    if x_cols:
        X = np.column_stack(
            [np.ones(n)] + [np.asarray(data[c], dtype=np.float64) for c in x_cols])
    else:
        X = np.ones((n, 1), dtype=np.float64)
        x_cols = []
    p = X.shape[1]

    def _neg_ll(params):
        beta = params[:p]
        logphi = params[p]
        phi = np.exp(logphi)
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
        a = mu * phi
        b = (1.0 - mu) * phi
        from scipy.special import betaln
        return -np.sum(-betaln(a, b) + (a - 1) * np.log(np.clip(y, 1e-15, 1.0)) +
                        (b - 1) * np.log(np.clip(1 - y, 1e-15, 1.0)))

    yc = np.clip(np.asarray(y, dtype=np.float64), 1e-6, 1 - 1e-6)
    beta0 = np.linalg.lstsq(X, np.log(yc / (1 - yc)), rcond=None)[0]
    param0 = np.concatenate([beta0, [np.log(20.0)]])

    res = minimize(_neg_ll, param0, method="L-BFGS-B",
                   bounds=[(-20, 20)] * p + [(-10, 10)],
                   # Standalone beta regression (no STMM/preset concept).
                   options={"maxiter": 500, "ftol": 1e-10})

    beta_hat = res.x[:p]
    phi_hat = np.exp(res.x[p])
    log_lik = -res.fun
    k = len(res.x)
    aic = -2 * log_lik + 2 * k
    bic = -2 * log_lik + np.log(n) * k

    # Numerical Hessian -> SEs (params: beta..., logphi)
    eps = 1e-5
    H = np.zeros((k, k))
    f0 = _neg_ll(res.x)
    for i in range(k):
        pp, pm = res.x.copy(), res.x.copy()
        pp[i] += eps; pm[i] -= eps
        H[i, i] = (_neg_ll(pp) - 2 * f0 + _neg_ll(pm)) / eps**2
        for j in range(i + 1, k):
            ppp, ppm, pmp, pmm = (res.x.copy() for _ in range(4))
            ppp[i] += eps; ppp[j] += eps
            ppm[i] += eps; ppm[j] -= eps
            pmp[i] -= eps; pmp[j] += eps
            pmm[i] -= eps; pmm[j] -= eps
            H[i, j] = H[j, i] = (_neg_ll(ppp) - _neg_ll(ppm)
                                 - _neg_ll(pmp) + _neg_ll(pmm)) / (4 * eps**2)
    try:
        vcov = np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(vcov), 0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
        vcov = None

    labels = ["(Intercept)"] + x_cols + ["log(phi)"]
    z = beta_hat.tolist() + [res.x[p]]
    from scipy.stats import norm
    pvals = [2 * float(norm.sf(abs(zv / s))) if (np.isfinite(s) and s > 0) else np.nan
             for zv, s in zip(z, se)]

    return {
        "beta_hat": beta_hat,
        "phi": phi_hat,
        "log_lik": log_lik,
        "AIC": aic,
        "BIC": bic,
        "k": k,
        "n": n,
        "p": p,
        "x_cols": x_cols,
        "param_labels": labels,
        "se": se,
        "vcov": vcov,
        "zvalue": z,
        "pvalue": pvals,
    }
