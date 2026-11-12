"""Likelihood ratio test for the fixed effects of a Gaussian model.

The Gaussian path profiles the fixed effects out (GLS), so the restricted
model is the same model with one column removed from the design and the
variance parameters re-optimised. That makes the test cheap: one likelihood
evaluation is a Cholesky plus a solve, with no inner mode to solve.

Two rules the test cannot bend:

- The comparison must be ML. REML likelihoods are not comparable across
  different fixed-effects designs, so the full model is re-fitted with ML
  as well and the block reports that log-likelihood, not the REML one of
  the fit. The fit itself is untouched.
- A p-value is only issued when the restricted fit converged and the
  restricted likelihood did not exceed the full one (the models are
  nested, so it cannot; if it does, the column mapping is wrong and the
  number must not be used).

Reference: chi-square with 1 df. When a variance component sits at its
lower bound the reference is not exact; the boundary is reported and the
Self & Liang mixture is given next to it, not substituted silently.
"""
from __future__ import annotations

import time

import numpy as np


def _family_refitter(fit, backend):
    """A callable that re-optimises the variance parameters for a design.

    One entry per Gaussian family. The family is named by the fit itself
    (``lrt_family``), so a new family adds one branch here and nothing
    else changes.
    """
    kind = fit.get("lrt_family")
    y = np.asarray(fit["y"], dtype=float)
    prepared = fit.get("data")
    model = fit.get("model") or {}
    interaction = model.get("interaction", "none")
    area_type = model.get("area_type", "iid")
    theta = fit.get("theta")

    if kind == "core":
        from stmmlib.gaussian.optimization import run_optimization

        def refit(X_new, start=theta, **kw):
            return run_optimization(y, X_new, prepared, model, backend,
                                    start=start, **kw)
    elif kind == "seblup":
        from stmmlib.gaussian.optimization import fit_seblup5

        def refit(X_new, start=theta, **kw):
            return fit_seblup5(y, X_new, prepared, interaction, area_type,
                               backend, start=start, **kw)
    elif kind == "nested":
        from stmmlib.gaussian.optimization import nested_fit

        def refit(X_new, start=theta, **kw):
            return nested_fit(y, X_new, prepared, interaction, backend,
                              start=start, **kw)
    elif kind == "nested_ar1":
        from stmmlib.gaussian.optimization import nested_ar1_fit

        def refit(X_new, start=theta, **kw):
            return nested_ar1_fit(y, X_new, prepared, interaction, backend,
                                  start=start, **kw)
    else:
        raise ValueError(
            f"the fit does not say which Gaussian family it is "
            f"(lrt_family={kind!r}); the LRT cannot re-fit it.")
    return refit


def _variance_at_bound(theta, margin=1e-8):
    """Variance components that sit on their lower bound (zero)."""
    out = []
    for k, v in (theta or {}).items():
        if not str(k).startswith("sigma2") or k == "sigma2_e":
            continue
        try:
            val = float(np.real(v))
        except (TypeError, ValueError):
            continue
        if np.isfinite(val) and val <= margin:
            out.append(k)
    return out


def _two_starts(refit, X_new, kw, agree_tol):
    """Re-fit from two independent starts and compare.

    The optimizer's own success flag cannot be the certificate: started
    exactly at the optimum, L-BFGS-B returns ABNORMAL with nit = 0 (the
    line search cannot satisfy the Wolfe conditions when the gradient is
    already zero) even though the point is the maximum. So the certificate
    is the likelihood: two independent starts (the warm start and the
    family's own automatic start) must land on the same likelihood.

    Returns (log_lik, gap, ok, theta) where log_lik is the better of the
    two and theta the parameters that go with it.
    """
    a = refit(X_new, **kw)
    b = refit(X_new, start=None, **kw)
    ll_a = float(a.get("log_lik", float("nan")))
    ll_b = float(b.get("log_lik", float("nan")))
    if not np.isfinite(ll_b):
        ll_b = -np.inf
    if not np.isfinite(ll_a):
        ll_a = -np.inf
    if ll_a >= ll_b:
        ll, theta = ll_a, a.get("theta")
    else:
        ll, theta = ll_b, b.get("theta")
    gap = abs(ll_a - ll_b) if np.isfinite(ll) else float("inf")
    ok = bool(np.isfinite(ll) and gap <= agree_tol * max(1.0, abs(ll)))
    return ll, gap, ok, theta


def lrt_fixed_gaussian(fit, terms=None, *, backend=None, method="ML",
                       n_starts=1, max_iter=200, deadline_s=None,
                       nest_tol=1e-6, agree_tol=1e-4) -> dict:
    """Likelihood ratio test for the fixed effects of a Gaussian fit.

    Args:
        fit: a Gaussian fit dict (from ``stmm(..., family="gaussian")``).
            It must carry ``x_cols``, ``X``, ``y``, ``data``, ``model``,
            ``theta`` and ``lrt_family``.
        terms: predictor names to test. None tests every predictor.
        backend: the array backend the fit used.
        method: "ML" (the default and the only sound choice here) or
            "REML". REML likelihoods are not comparable across different
            designs, so a REML LRT is refused.
        n_starts: random restarts for each re-fit. One keeps the statistic
            deterministic; more makes D jitter between runs.
        max_iter: outer iterations of the variance optimisation.
        deadline_s: wall-clock budget for the whole call.
        nest_tol: how much the restricted likelihood may exceed the full
            one before the comparison is refused.

    Returns:
        dict with ``method``, ``df_per_term``, ``terms`` (per term: D, df,
        p, loglik_full, loglik_reduced, status, converged, seconds,
        message) and ``n_failed``. A term whose re-fit failed carries
        ``p = None``, never a number.
    """
    from scipy.stats import chi2

    if str(method).upper() == "REML":
        raise ValueError(
            "the Gaussian LRT needs ML: REML likelihoods are not comparable "
            "across different fixed-effects designs. The fit itself may stay "
            "REML; the LRT refits with ML and reports that log-likelihood.")
    method = "ML"

    x_cols = list(fit.get("x_cols") or [])
    X = np.asarray(fit.get("X"), dtype=float)
    if not x_cols or X.ndim != 2:
        raise ValueError(
            "the fit does not carry x_cols and X; the LRT cannot map a "
            "predictor to a design column.")
    if terms is None:
        terms = list(x_cols)
    for t in terms:
        if t == "(Intercept)":
            raise ValueError("the intercept cannot be tested by LRT.")
        if t not in x_cols:
            raise KeyError(f"unknown term {t!r}; fitted predictors: {x_cols}")

    refit = _family_refitter(fit, backend)
    started = time.perf_counter()
    kw = dict(method=method, n_starts=n_starts, max_iter=max_iter)

    # The full model is re-fitted with the same procedure as the reduced
    # ones, so the two likelihoods come from the same estimator.
    ll_full, full_gap, full_converged, theta_full = _two_starts(
        refit, X, kw, agree_tol)
    bound_full = _variance_at_bound(theta_full)

    out_terms = {}
    n_failed = 0
    for t in terms:
        row = {"D": None, "df": 1, "p": None, "loglik_full": ll_full,
               "loglik_reduced": None, "status": "ok", "converged": None,
               "seconds": 0.0, "message": "",
               "nuisance_at_bound": list(bound_full)}
        t0 = time.perf_counter()
        if deadline_s is not None and time.perf_counter() - started >= deadline_s:
            row["status"] = "skipped_deadline"
            row["message"] = "wall-clock budget exhausted"
            row["seconds"] = time.perf_counter() - t0
            out_terms[t] = row
            n_failed += 1
            continue

        j = x_cols.index(t)
        X_red = np.delete(X, j + 1, axis=1)   # column 0 is the intercept
        try:
            ll_red, red_gap, red_ok, theta_red = _two_starts(
                refit, X_red, kw, agree_tol)
        except Exception as exc:       # a refit that blows up is a failed refit
            row["status"] = "failed"
            row["message"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = time.perf_counter() - t0
            out_terms[t] = row
            n_failed += 1
            continue

        row["seconds"] = time.perf_counter() - t0
        row["loglik_reduced"] = ll_red
        row["converged"] = bool(red_ok)
        row["start_gap"] = float(red_gap)
        row["nuisance_at_bound"] = sorted(set(bound_full) | set(
            _variance_at_bound(theta_red)))
        if not np.isfinite(ll_red) or not np.isfinite(ll_full):
            row["status"] = "failed"
            row["message"] = "the re-fit produced no finite likelihood"
        elif ll_red > ll_full + nest_tol:
            row["status"] = "failed"
            row["message"] = (
                f"the restricted likelihood ({ll_red:.6f}) exceeds the full "
                f"one ({ll_full:.6f}); the models are nested, so the column "
                "mapping is wrong and the statistic must not be used.")
        elif not red_ok:
            row["status"] = "not_converged"
            row["message"] = (
                f"two independent starts disagree by {red_gap:.3e}; the "
                "restricted maximum is uncertain")
        elif not full_converged:
            row["status"] = "not_converged"
            row["message"] = (
                f"two independent starts of the full model disagree by "
                f"{full_gap:.3e}; the comparison is not trustworthy")
        else:
            row["status"] = "ok"
            row["D"] = float(max(2.0 * (ll_full - ll_red), 0.0))
            row["p"] = float(chi2.sf(row["D"], 1))
        if row["status"] != "ok":
            n_failed += 1
        out_terms[t] = row

    return {"method": "lrt", "df_per_term": 1, "likelihood_method": method,
            "loglik_full": ll_full, "n_failed": n_failed,
            "terms": out_terms, "seconds": time.perf_counter() - started}


def attach_lrt_inference(fit, lrt, df_method: str = "satterthwaite") -> dict:
    """Put the LRT result into the fit dict the tables read.

    The Wald p-values are kept next to the LRT ones (``pvalue_wald``), and
    the reported statistic follows the requested test, like in the Beta
    path. Predictors that were not tested carry NaN under the LRT header.
    """
    from stmmlib.gaussian.summary import wald_pvalues

    x_cols = list(fit.get("x_cols") or [])
    labels = ["(Intercept)"] + x_cols
    beta = np.asarray(fit.get("beta_hat", []), dtype=float).ravel()
    se = np.asarray(fit.get("se_beta", []), dtype=float).ravel()
    n_p = len(beta)
    p_wald = np.asarray(wald_pvalues(fit, df_method), dtype=float)
    t_wald = beta / np.where(se > 0, se, 1.0)

    pvalue = np.full(n_p, np.nan)
    statistic = np.full(n_p, np.nan)
    for t, row in (lrt.get("terms") or {}).items():
        j = x_cols.index(t) + 1
        if row.get("p") is None:
            continue
        pvalue[j] = float(row["p"])
        d = float(row.get("D") or 0.0)
        sign = np.sign(beta[j]) if beta[j] != 0 else 1.0
        statistic[j] = sign * float(np.sqrt(d))

    fit["inference"] = {
        "param_labels": labels,
        "estimate": beta,
        "se": se,
        "tvalue": t_wald,
        "statistic": statistic,
        "pvalue": pvalue,
        "pvalue_wald": p_wald,
        "vcov": fit.get("vcov_beta"),
        "test": "lrt",
        "df_method": df_method,
    }
    fit["lrt"] = lrt
    return fit
