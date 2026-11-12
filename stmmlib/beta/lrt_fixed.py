"""Restricted-refit likelihood ratio test for Beta fixed effects.

The fixed-effect Wald test in `beta/inference.py` uses Hessian standard
errors. This module provides the alternative: drop one fixed-effect column,
re-optimise the same model, and compare marginal likelihoods.

    D = 2 * (logLik_full - logLik_reduced),   df = 1,   p = chi2.sf(D, 1)

Only single-column restrictions are supported. Multi-column (joint) tests
are out of scope.

Parameter vector layout, as the solver packs it
(`beta/solvers.py`: `eval_param[:p]` is beta, `eval_param[p]` is logphi,
`eval_param[p+1:p+1+ksd]` are the log sigmas, the rest is raw_dep):

    theta = [beta, logphi, log_sigma..., raw_dep...]
"""
from __future__ import annotations

import numpy as np


def pack_theta(opt: dict) -> np.ndarray:
    """Pack a fitted `_opt` into the solver's parameter vector."""
    beta = np.asarray(opt["beta_hat"], dtype=float).ravel()
    logphi = float(np.asarray(opt["logphi"], dtype=float).ravel()[0])
    logs = np.asarray(opt["logs"], dtype=float).ravel()
    raw = np.asarray(opt["raw_dep"], dtype=float).ravel()
    return np.concatenate([beta, [logphi], logs, raw])


def unpack_theta(theta, p: int, n_struct: int) -> dict:
    """Inverse of `pack_theta`.

    `p` is the number of fixed-effect coefficients, `n_struct` the number of
    log sigmas. raw_dep takes whatever is left.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    beta = theta[:p]
    logphi = float(theta[p])
    logs = theta[p + 1:p + 1 + n_struct]
    raw = theta[p + 1 + n_struct:]
    return {"beta_hat": beta, "logphi": logphi, "logs": logs, "raw_dep": raw}


def _refit_reduced(d, y, model, W, *, start_b=None, n_starts=2, config=None,
                    max_iter=1000, inner_max_iter=50, ftol=1e-12, gtol=1e-6):
    """Re-fit the model with one fixed-effect column already removed.

    The solver is called directly, not through `stmm_beta`, so the
    objective is identical to the full fit: `stmm_beta` re-sorts the areas
    and may run an extra weak-prior refit, and either would make the two
    likelihoods incomparable.

    Two warm starts, and the difference matters:

    `start_b` is the mode of the full fit. Without it the inner solve
    starts from b = 0, which diverges on large panels and makes the whole
    start non-finite.

    The outer start is NOT the full optimum. There the reduced model's
    gradient can be large, and L-BFGS-B scales its first step by g/max|g|,
    so the step is O(1) on coefficients of order 0.01: the objective
    explodes and the fit ends invalid. The solver's own start is used
    instead, so the refit is a
    fresh optimisation under the same procedure as the full fit.
    """
    from stmmlib.beta.solvers import beta_laplace_structural
    return beta_laplace_structural(
        None, d, y, "Laplace", model, W,
        n_starts=n_starts, max_iter=max_iter, inner_max_iter=inner_max_iter,
        start_b=start_b, compute_hessian=False,
        ftol=ftol, gtol=gtol, config=config)


def _is_penalized(opt: dict, config=None) -> bool:
    """True when the stored logLik comes from a penalised (MAP) point.

    A penalised fit is not a plain likelihood, so 2*diff(logLik) is not
    chi2_1. Two cases count: the automatic weak-prior refit produced the
    point that was kept, and the caller supplied a prior in the config.
    """
    optim = opt.get("optim") or {}
    if optim.get("penalty_auto") and not optim.get("penalty_refit_failed"):
        return True
    if config is not None:
        if getattr(config, "phi_prior_sd", None) is not None:
            return True
        if getattr(config, "dep_prior_sd", None) is not None:
            return True
    return False


def _phi_at_bound(opt: dict, config=None) -> bool:
    """True when the Beta precision sits on its box bound."""
    from stmmlib.beta.config import BetaFitConfig
    cfg = config if config is not None else BetaFitConfig()
    logphi = float(np.asarray(opt["logphi"], dtype=float).ravel()[0])
    hi = float(cfg.logphi_bounds[1])
    return abs(logphi) >= hi - 1e-6


def _hessian(opt: dict):
    """The Hessian the solver stored, or None."""
    H = (opt.get("optim") or {}).get("hessian")
    if H is None:
        return None
    H = np.asarray(H, dtype=float)
    return H if H.ndim == 2 else None


def _bounds_arrays(opt: dict, config=None):
    """The solver's box bounds as two arrays over the parameter vector.

    Layout: [beta (p), logphi, logs (ksd), raw_dep (q)]. Without these the
    refinement can walk out of the box (it did: logphi 10.002963 against a
    bound of 10), and a point outside the box is not a point of the model.
    """
    from stmmlib.beta.config import BetaFitConfig
    from stmmlib.beta.solvers import beta_parameter_bounds
    cfg = config if config is not None else BetaFitConfig()
    p = int(opt["p"])
    ksd = len(np.asarray(opt["logs"], dtype=float).ravel())
    q = len(np.asarray(opt["raw_dep"], dtype=float).ravel())
    ref = beta_parameter_bounds(cfg, p, ksd, q)
    lo = np.array([b[0] for b in ref], dtype=float)
    hi = np.array([b[1] for b in ref], dtype=float)
    return lo, hi


def _refine_restricted(theta, d, y, model, W, b, H_red=None, *,
                       ll_full=None, max_iter=40, tol=1e-9, step0=1.0,
                       bounds=None, config=None, grad_tol=None,
                       recompute_curvature=True, curvature_every=3,
                       deadline_at=None):
    """Restricted maximum by a projected Newton step, Armijo backtracking.

    The solver's own L-BFGS-B cannot be used for this: starting from the full
    optimum the reduced model's gradient is large, and L-BFGS-B scales its
    first step by g/max|g|, so the step is O(1) on coefficients of order
    0.01. The objective then overflows and the fit ends invalid.

    Here the objective is evaluated directly (cheap with a warm mode) and the
    step is a Newton step for the log-likelihood, with a backtracking line
    search, so the objective can only rise.

    The step is projected onto the KKT active set, and that is what makes the
    search work. The Newton step of the reduced model can go almost entirely
    into coordinates that sit on a bound (logphi and the dependence
    parameter): their curvature is small (around 40) while the coefficients
    have a curvature of 1e5, and both want to leave the box. Clipping them
    afterwards leaves an almost empty step, while the Armijo test still
    compares against the slope of the clipped direction, so the line search
    can stall with a large projected gradient after one step.

    With the projection the step is zero in the coordinates whose gradient
    pushes outwards at a bound (the KKT condition is satisfied there, for the
    full model and for the reduced one alike), the curvature comes from the
    free block only, and the Armijo test uses the displacement that is really
    taken, so the slope it compares against is the slope that is really used.

    Two details decide whether this finds the maximum:

    - the step is NOT normalised. A unit-length step with a fixed trial
      length travels at most a few times the trial length in total, which
      stops the search far from the optimum and inflates D.
    - `g` is the gradient of the NEGATIVE log-likelihood, so the ascent
      slope of the log-likelihood along a displacement d is `-g @ d`. The
      Armijo test uses that sign; with the opposite sign it accepts downhill
      steps.

    `config` must be the config the fit used: it decides the rho and phi
    bounds and the penalty, so a different config means a different model
    and the two likelihoods are not comparable.

    Returns dict with `log_lik`, `converged`, `iterations`, `grad_max`,
    `grad_proj`.
    """
    import time

    from stmmlib.beta.solvers import beta_laplace_structural
    from stmmlib.beta.solvers import _num_hessian_grad
    from stmmlib.beta.config import BetaFitConfig

    if grad_tol is None:
        grad_tol = float((config if config is not None
                          else BetaFitConfig()).grad_tol)

    n = len(theta)

    def _obj(th):
        ev = beta_laplace_structural(None, d, y, "Laplace", model, W,
                                     eval_param=th, eval_b0=b,
                                     compute_hessian=False, config=config)
        return float(ev.get("log_lik", float("nan")))

    def _grad(th):
        gr = beta_laplace_structural(None, d, y, "Laplace", model, W,
                                     eval_grad=th, eval_b0=b, config=config)
        return np.asarray(gr.get("grad"), dtype=float)

    def _active_set(g, th):
        """Coordinates on a bound whose gradient pushes outwards (KKT met).

        At the lower bound the objective cannot fall any further when the
        gradient is positive, at the upper bound when it is negative. Those
        coordinates must not move, and their gradient is not a reason to
        move either.
        """
        if bounds is None:
            return []
        lo, hi = bounds
        act = []
        for i in range(n):
            x, gi = float(th[i]), float(g[i])
            at_lo = abs(x - lo[i]) <= 1e-9
            at_hi = abs(x - hi[i]) <= 1e-9
            if (at_lo and gi > 0.0) or (at_hi and gi < 0.0):
                act.append(i)
        return act

    def _proj_norm(g, th):
        """Gradient norm with the bound-active components removed (KKT)."""
        if bounds is None:
            return float(np.linalg.norm(g))
        lo, hi = bounds
        pg = np.array(g, dtype=float, copy=True)
        for i in range(n):
            x, gi = float(th[i]), float(pg[i])
            at_lo = abs(x - lo[i]) <= 1e-9
            at_hi = abs(x - hi[i]) <= 1e-9
            if (at_lo and gi > 0.0) or (at_hi and gi < 0.0):
                pg[i] = 0.0
        return float(np.linalg.norm(pg))

    def _curvature(th, free):
        """Curvature of the objective on the free coordinates."""
        if H_red is not None and not recompute_curvature:
            return H_red[np.ix_(free, free)]
        try:
            H = _num_hessian_grad(_grad, th, eps=1e-4)
        except Exception:
            return None
        return H[np.ix_(free, free)]

    ll = _obj(theta)
    if not np.isfinite(ll):
        return {"log_lik": ll, "converged": False, "iterations": 0,
                "grad_max": float("nan"), "grad_proj": float("nan"),
                "theta": theta}
    prev = ll
    it = 0
    H_cur = None
    if deadline_at is not None and time.perf_counter() > deadline_at:
        g0 = _grad(theta)
        return {"log_lik": ll, "converged": False, "iterations": 0,
                "grad_max": float(np.max(np.abs(g0))),
                "grad_proj": _proj_norm(g0, theta), "theta": theta}
    for it in range(1, max_iter + 1):
        if deadline_at is not None and time.perf_counter() > deadline_at:
            break
        g = _grad(theta)
        if g.shape != theta.shape or not np.all(np.isfinite(g)):
            break
        active = set(_active_set(g, theta))
        free = [i for i in range(n) if i not in active]
        if not free:
            break
        if it == 1 and H_red is not None and recompute_curvature:
            # Start from the stored curvature of the full model: it costs
            # nothing and it is the right order of magnitude. The
            # finite-difference curvature replaces it further down.
            H_cur = H_red[np.ix_(free, free)]
        elif H_cur is None or (recompute_curvature
                               and (it - 1) % max(1, curvature_every) == 0):
            H_cur = _curvature(theta, free)
        if H_cur is not None:
            try:
                step_free = -np.linalg.solve(H_cur, g[free])
            except (np.linalg.LinAlgError, ValueError):
                step_free = -g[free]
        else:
            step_free = -g[free]
        if (not np.all(np.isfinite(step_free))
                or float(np.linalg.norm(step_free)) == 0.0):
            break
        step = np.zeros(n, dtype=float)
        step[free] = step_free
        slope = -float(g @ step)          # ascent slope of the log-likelihood
        if not np.isfinite(slope) or slope <= 0.0:
            # the curvature gave a descent direction: fall back to steepest
            # ascent on the free block
            step_free = -g[free]
            step = np.zeros(n, dtype=float)
            step[free] = step_free
            slope = -float(g @ step)
            if slope <= 0.0:
                break
        t = step0
        accepted = False
        for _ls in range(60):
            cand = theta + t * step
            if bounds is not None:
                cand = np.clip(cand, bounds[0], bounds[1])
            ll_c = _obj(cand)
            # Armijo against the displacement that is really taken: after the
            # clip it can be shorter than t*step.
            rise = -float(g @ (cand - theta))
            if np.isfinite(ll_c) and ll_c >= ll + 1e-4 * rise:
                theta, ll = cand, ll_c
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break
        if ll_full is not None and ll > ll_full + 1e-6:
            # A restriction cannot improve the maximum: something is wrong
            # with the column mapping, so the result must not be used.
            return {"log_lik": ll, "converged": False, "iterations": it,
                    "grad_max": float("nan"), "grad_proj": float("nan"),
                    "theta": theta}
        gain = ll - prev
        prev = ll
        if abs(gain) < tol:
            break
    g = _grad(theta)
    gmax = (float(np.max(np.abs(g))) if g.shape == theta.shape
            else float("nan"))
    gproj = _proj_norm(g, theta) if g.shape == theta.shape else float("nan")
    # Converged means the projected gradient satisfies the same KKT rule the
    # solver's own certificate uses (bound-active components do not count).
    # "Finite logLik" is not convergence: a refinement that made no progress
    # at all would otherwise be reported as a converged maximum.
    converged = bool(np.isfinite(ll) and np.isfinite(gproj)
                     and gproj <= grad_tol)
    return {"log_lik": ll, "converged": converged,
            "iterations": it, "grad_max": gmax, "grad_proj": gproj,
            "theta": theta}


def lrt_fixed_terms(opt: dict, d, y, model, W, terms=None, *,
                    config=None, max_iter=1000, inner_max_iter=50,
                    n_starts=2, ftol=1e-12, gtol=1e-6, deadline_s=None,
                    method="refine", refine_max_iter=40) -> dict:
    """Likelihood ratio test for single fixed-effect terms.

    Args:
        opt: `_opt` of a fitted model (the full model).
        d: the internal data frame of that fit (`.stmm_area`, `.stmm_time`,
            `y`, and the predictor columns).
        y: response vector of that fit.
        model: model id of that fit.
        W: spatial weight matrix of that fit.
        terms: predictor names to test. None tests every predictor.
        config: the same BetaFitConfig the full fit used.
        max_iter, inner_max_iter, n_starts, ftol, gtol: used by the "plain"
            path, which re-runs the solver's own optimiser under the same
            procedure as the full fit. The default "refine" path does not
            run that optimiser: it evaluates the objective directly and
            steps with a Newton direction, so only `config` (the model) and
            the bounds matter there.
        deadline_s: wall-clock budget for the whole call. Terms left over
            when it runs out are reported as `skipped_deadline`.

    Returns:
        dict with `method`, `df_per_term`, `penalized`, `phi_at_bound`,
        `n_failed`, and `terms` (per term: D, df, p, loglik_full,
        loglik_reduced, status, converged, seconds, message). A term whose
        refit failed carries `p = None`, never a number.
    """
    import time
    from scipy.stats import chi2

    x_cols = list(opt["des"]["x_cols"])
    if terms is None:
        terms = list(x_cols)
    for t in terms:
        if t in ("(Intercept)", "Intercept"):
            raise ValueError("the intercept cannot be tested by LRT.")
        if t not in x_cols:
            raise KeyError(f"unknown term {t!r}; fitted predictors: {x_cols}")

    ll_full_stored = float(opt["log_lik"])
    theta_full = pack_theta(opt)
    _b_hat = opt.get("b_hat")
    _start_b = (np.asarray(_b_hat, dtype=float).ravel()
                if _b_hat is not None else None)
    _H = _hessian(opt)
    _bounds = _bounds_arrays(opt, config)
    started = time.perf_counter()

    # The stored logLik is not necessarily the maximum: the certificate of
    # the stored optimum may not be the maximum, and the refinement below
    # finds a point with a slightly higher likelihood. An LRT compares two
    # maxima, so the full model is refined with the same procedure before
    # the restricted models are refined.
    _deadline_at = (started + deadline_s) if deadline_s else None
    if method == "refine" and _start_b is not None:
        _full_ref = _refine_restricted(theta_full, d, y, model, W, _start_b,
                                       _H, ll_full=None,
                                       max_iter=refine_max_iter,
                                       bounds=_bounds, config=config,
                                       deadline_at=_deadline_at)
        ll_full = float(_full_ref["log_lik"])
    else:
        _full_ref = {"log_lik": ll_full_stored, "iterations": 0,
                     "grad_max": float("nan")}
        ll_full = ll_full_stored
    if not np.isfinite(ll_full):
        ll_full = ll_full_stored

    out_terms = {}
    n_failed = 0
    for t in terms:
        row = {"D": None, "df": 1, "p": None, "loglik_full": ll_full,
               "loglik_full_stored": ll_full_stored,
               "loglik_reduced": None, "status": "ok", "converged": None,
               "seconds": 0.0, "message": ""}
        if deadline_s is not None and time.perf_counter() - started >= deadline_s:
            row["status"] = "skipped_deadline"
            row["message"] = "wall-clock budget exhausted"
            out_terms[t] = row
            n_failed += 1
            continue

        j = x_cols.index(t)
        d_red = d.drop(columns=[t])
        theta_red = np.delete(theta_full, 1 + j)
        t0 = time.perf_counter()
        try:
            if method == "plain":
                res = _refit_reduced(d_red, y, model, W,
                                     start_b=_start_b, n_starts=n_starts,
                                     config=config, max_iter=max_iter,
                                     inner_max_iter=inner_max_iter,
                                     ftol=ftol, gtol=gtol)
                ll_red = float(res.get("log_lik", float("nan")))
                converged = bool(res.get("converged", True))
                message = str(res.get("status", ""))
                grad_max = float("nan")
            else:
                _keep = [i for i in range(len(theta_full)) if i != 1 + j]
                _H = _hessian(opt)
                _H_red = (_H[np.ix_(_keep, _keep)] if _H is not None else None)
                _bnd = (_bounds[0][_keep], _bounds[1][_keep]) \
                    if _bounds is not None else None
                ref = _refine_restricted(theta_red, d_red, y, model, W,
                                          _start_b, _H_red, ll_full=ll_full,
                                          max_iter=refine_max_iter,
                                          bounds=_bnd, config=config,
                                          deadline_at=_deadline_at)
                ll_red = float(ref["log_lik"])
                converged = bool(ref["converged"])
                message = (f"refine {ref['iterations']} it, "
                           f"max|g|={ref['grad_max']:.3e}, "
                           f"|g_proj|={ref['grad_proj']:.3e}")
                grad_max = float(ref["grad_max"])
        except Exception as exc:  # a refit that blows up is a failed refit
            row["status"] = "failed"
            row["message"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = time.perf_counter() - t0
            out_terms[t] = row
            n_failed += 1
            continue

        row["seconds"] = time.perf_counter() - t0
        row["loglik_reduced"] = ll_red
        row["converged"] = converged
        row["message"] = message
        row["method"] = "plain" if method == "plain" else "refine"
        row["grad_max"] = grad_max
        if not np.isfinite(ll_red):
            row["status"] = "failed"
        elif not converged:
            row["status"] = "not_converged"
        else:
            row["status"] = "ok"
            row["D"] = float(max(2.0 * (ll_full - ll_red), 0.0))
            row["p"] = float(chi2.sf(row["D"], 1))
        if row["status"] != "ok":
            n_failed += 1
        out_terms[t] = row

    return {"method": "lrt", "df_per_term": 1,
            "penalized": _is_penalized(opt, config),
            "phi_at_bound": _phi_at_bound(opt, config),
            "n_failed": n_failed, "terms": out_terms,
            "seconds": time.perf_counter() - started}


# Columns the solver never puts into the design matrix.
_SKIP_COLS = {".stmm_area", ".stmm_time", ".stmm_prov", ".stmm_kab", "y"}


def _check_labels(fit, area_labels, time_labels):
    """The data must describe the same panel as the fit."""
    for name, new_lab, old_lab in (
            ("area", area_labels, getattr(fit, "_area_labels", None)),
            ("time", time_labels, getattr(fit, "_time_labels", None))):
        if not old_lab:
            continue
        _new = [new_lab[k] for k in sorted(new_lab)]
        _old = [old_lab[k] for k in sorted(old_lab)]
        if _new != _old:
            raise ValueError(
                f"label {name} pada data berbeda dari data pada fit; "
                "LRT hanya sah untuk panel yang sama.")


def lrt_fixed_effects(fit, data, terms=None, *, response_col="y",
                      area_col="area", time_col="time", prov_col=None,
                      kab_col=None, **kw) -> dict:
    """Likelihood ratio test for the fixed effects of an existing fit.

    The universal entry point: hand it the fitted object and the data the
    fit was made from, and it tests whatever predictors that model has.
    New predictors and new datasets need no code change.

    The data must be the data behind the fit. A mismatch in the response or
    in the predictor set raises ValueError, because comparing two
    likelihoods from different datasets is meaningless.
    """
    from stmmlib.beta.fit import _sanitize_data
    opt = fit._opt
    d, area_labels, time_labels = _sanitize_data(data, response_col=response_col,
                              area_col=area_col, time_col=time_col,
                              prov_col=prov_col, kab_col=kab_col)
    _check_labels(fit, area_labels, time_labels)
    x_fit = list(opt["des"]["x_cols"])
    x_new = [c for c in d.columns if c not in _SKIP_COLS]
    if set(x_new) != set(x_fit):
        missing = sorted(set(x_fit) - set(x_new))
        extra = sorted(set(x_new) - set(x_fit))
        raise ValueError(
            f"the data does not match the fitted model: missing predictors "
            f"{missing}, unexpected predictors {extra}.")

    # Every beta index is tied to the column order of the fit, so restore it.
    head = [c for c in d.columns if c in _SKIP_COLS]
    d = d[head + x_fit]

    y_fit = np.asarray(opt["y"], dtype=float)
    y_new = d["y"].values.astype(np.float64)
    if y_new.shape != y_fit.shape or not np.allclose(y_new, y_fit,
                                                     rtol=1e-12, atol=1e-12):
        raise ValueError(
            "the data passed does not match the fitted model: the response "
            "differs from the one the model was estimated on.")

    return lrt_fixed_terms(opt, d, y_new, opt["model"],
                            getattr(fit, "_W", None), terms, **kw)
