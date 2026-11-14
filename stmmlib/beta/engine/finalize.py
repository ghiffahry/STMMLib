"""Final estimates, Hessian, convergence certificate, payload."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from stmmlib.beta.precision import PrecisionShapeError
from stmmlib.beta.beta_nll import (
    beta_nll_curvature,
    _beta_nll_score,
    _beta_grad_pieces,
)
from stmmlib.beta.schur import _blocked_hinv_terms, _blocked_pen_terms
from stmmlib.beta.mode import (
    beta_mode_newton_blocked,
    beta_solve_mode,
    beta_solve_mode_blocked,
)
from stmmlib.beta.numerics import (
    _num_hessian,
    _num_hessian_grad,
    _projected_grad_norm,
    _os_env_debug,
)


def build_payload(ctx, best_param, best_val, best_meta, starts, failed_starts, _interrupted, _from_abort):
    W = ctx["W"]
    X = ctx["X"]
    Z = ctx["Z"]
    _build_Q = ctx["_build_Q"]
    _eval_count = ctx["_eval_count"]
    _jac_ok = ctx["_jac_ok"]
    _neg_marginal_l = ctx["_neg_marginal_l"]
    _laplace_grad = ctx["_laplace_grad"]
    _state = ctx["_state"]
    bounds = ctx["bounds"]
    cfg = ctx["cfg"]
    compute_hessian = ctx["compute_hessian"]
    data = ctx["data"]
    des = ctx["des"]
    ftol = ctx["ftol"]
    gtol = ctx["gtol"]
    hess_eps = ctx["hess_eps"]
    is_nested = ctx["is_nested"]
    ksd = ctx["ksd"]
    max_iter = ctx["max_iter"]
    model = ctx["model"]
    n = ctx["n"]
    n_gamma = ctx["n_gamma"]
    p = ctx["p"]
    param_dim = ctx["param_dim"]
    q_dep = ctx["q_dep"]
    x_cols = ctx["x_cols"]
    y = ctx["y"]
    # Extract results (copy: putus view ke _boot_theta agar refit tak sentuh utama)
    beta_hat = np.array(best_param[:p], dtype=np.float64, copy=True)
    logphi_hat = best_param[p]
    logs_hat = np.array(best_param[p + 1: p + 1 + ksd], dtype=np.float64, copy=True)
    raw_dep_hat = np.array(best_param[p + 1 + ksd:], dtype=np.float64, copy=True)

    phi_hat = np.exp(logphi_hat)
    sigma_area = np.exp(logs_hat[0]) if ksd > 0 else 1.0
    sigma_time = np.exp(logs_hat[2 if is_nested else 1]) if ksd > (2 if is_nested else 1) else 1.0
    sigma_gamma = np.exp(logs_hat[ksd - 1]) if n_gamma > 0 else None

    log_lik = -best_val
    k_total = param_dim
    aic = -2 * log_lik + 2 * k_total
    bic = -2 * log_lik + np.log(n) * k_total

    # Recompute the inner mode at best param so `_state["b"]` is the conditional
    # mode for the final estimates (used for predictions: eta = X beta + Z b_hat).
    try:
        _neg_marginal_l(best_param)
    except KeyboardInterrupt:
        print("interrupted: mode recompute dibatalkan user, pakai b terakhir")
        _interrupted = True
    b_hat = _state["b"].copy()

    # Laplace posterior of the random effects at the final fit, needed by the
    # EBP draws: p(b|y) ~ N(b_hat, H_b^-1), H_b = Z'diag(w)Z + Q(theta_hat).
    # Read-only post-convergence computation; opt-in because H_b is dense q x q.
    H_b_chol = None
    if getattr(cfg, "compute_posterior", False):
        try:
            _Qf = _build_Q(logs_hat, raw_dep_hat)
            # Q and Z must agree; a silent pad/truncate hides structural bugs.
            _qz = Z.shape[1]
            if _Qf.shape[0] != _qz:
                raise PrecisionShapeError(
                    f"Precision/design size mismatch: Q is {_Qf.shape[0]}x"
                    f"{_Qf.shape[1]} but Z has {_qz} columns for {model!r}."
                )
            _w = beta_nll_curvature(X @ beta_hat + Z @ b_hat, y, phi_hat)
            _Zw = Z * _w[:, None]
            H_b_chol = np.linalg.cholesky(_Zw.T @ Z + _Qf)
        except np.linalg.LinAlgError:
            H_b_chol = None  # not PD: EBP unavailable, fit itself untouched

    # Degenerate guard: if the optimizer slid to the variance upper bound, the
    # model sits in the sigma-collapse region; flag it so it is not trusted.
    # Also flag a "degenerate well" -- a per-observation log-lik far more
    # negative than any reasonable beta-density fit (inner mode wandered into a
    # huge negative-well, e.g. spatial models). Scale by n so the rule is
    # universal, not data-specific. Healthy beta fits sit near ~0..~5/n.
    slope = 0.999
    degenerate_well = bool((log_lik / n) < -20.0)
    degenerate = bool(
        (ksd > 0 and sigma_area >= slope * np.exp(3.0))
        or (ksd > 1 and sigma_time >= slope * np.exp(3.0))
        or degenerate_well
    )

    # Single convergence definition (sole source of truth for tables/summaries):
    # `converged` == the honest 3-state label `status` == "konvergen". A fit is
    # konvergen only if the gradient norm is below GRAD_TOL, OR the optimizer
    # reported success with a gradient within 2 orders of GRAD_TOL (trust-but-
    # verify). scipy's success=True alone is NOT enough: a failed line search
    # at the start reports "CONVERGENCE: RELATIVE REDUCTION OF F" with
    # success=True and grad can still be large (measured: it=1).
    # Otherwise it is honestly `max_iter` (never forced to "converged").
    # `degenerate` is a separate, non-equivalent signal -- `not degenerate`
    # is NOT convergence.
    nit = best_meta.get("nit", -1)
    grad_norm = best_meta.get("grad_norm", np.inf)
    _grad_finite = bool(np.isfinite(grad_norm))
    gconv = (_grad_finite and grad_norm < cfg.grad_tol) or (
        bool(best_meta.get("success", False)) and _grad_finite
        and grad_norm <= cfg.grad_slack * cfg.grad_tol)
    if degenerate or _from_abort:
        status = "invalid"
    elif not gconv:
        status = "max_iter"
    else:
        status = "konvergen"

    # Boundary flag: any parameter sitting at its box bound at the solution.
    # A boundary hit (e.g. logphi at +10, log_sigma at cap) signals possible weak
    # identification -- surfaced in the convergence certificate, not hidden.
    _cert_names = (["(Intercept)"] + list(x_cols)
                   + ["log_phi"]
                   + [f"log_sigma_{i}" for i in range(ksd)]
                   + [f"raw_dep_{i}" for i in range(q_dep)])
    _bound_names = [_cert_names[i] for i, (_lo, _hi) in enumerate(bounds)
                    if abs(best_param[i] - _lo) <= 1e-6
                    or abs(best_param[i] - _hi) <= 1e-6]
    at_bound = bool(_bound_names)

    # Hessian of the marginal negative log-likelihood w.r.t. theta at the
    # converged point (finite differences through the inner solve). Reported as
    # lambda_min / pd so the certificate reflects curvature (identification),
    # not just the optimizer's convergence flag. A non-PD Hessian at convergence
    # means the point is not a valid local maximum (weak identification).
    # Single Hessian source for SE display: FD eps=hess_eps at best_param via
    # _neg_marginal_l (warm-chain inner plus polish). Jacobian-ready models
    # (analytic jac, _jac_ok) take the cheap path: central FD of the analytic
    # objective gradient (2k evals); the others keep objective FD (~2k^2).
    # Inference reuses this H and must not compute a second Hessian with other
    # eps or tol.
    from stmmlib.beta.engine.hessian import final_hessian
    hessian, lambda_min, hessian_pd, _hess_timeout, _hi = final_hessian(
        cfg, compute_hessian, hess_eps, best_param, b_hat,
        _jac_ok, _state, _neg_marginal_l, _laplace_grad)
    _interrupted = bool(_interrupted or _hi)

    # - Convergence certificate, two levels -
    # valid = inference-grade: legacy rule unchanged (raw grad <= grad_tol AND
    # not at_bound AND Hessian not non-PD). Ranking CertValid reads this key,
    # so its behavior is unchanged. Point plus Wald inference allowed.
    # optimum_valid = constrained-optimum KKT: projected grad <= grad_tol with
    # pg_i = 0 when (x_i == lo and g_i > 0) or (x_i == hi and g_i < 0), else
    # pg_i = g_i (Byrd et al. 1995; Zhu, Byrd and Nocedal 1997). Point estimate
    # and prediction allowed; Wald SE/CI/p not allowed when valid is False
    # because boundary inference needs a mixture LRT (Self and Liang 1987).
    # ll_per_obs negatif adalah RED FLAG (warning), bukan hukum universal
    # "invalid": secara matematis log-lik per observasi GLMM tidak wajib positif.
    ll_per_obs = log_lik / n
    _warnings = []
    _grad_finite = bool(np.isfinite(grad_norm))
    _gvec = best_meta.get("grad_vec", None)
    _proj = _projected_grad_norm(best_param, _gvec, bounds)
    _proj_finite = bool(_proj is not None and np.isfinite(_proj))
    _optimum_valid = bool(_proj_finite and _proj <= cfg.grad_tol)
    if not _grad_finite or grad_norm > cfg.grad_tol:
        _warnings.append(
            f"gradient tidak mendekati nol (|grad|={grad_norm:.3e} > {cfg.grad_tol:g}); "
            "titik ini mungkin bukan maksimum lokal")
    if at_bound:
        _warnings.append(
            "parameter menempel boundary: " + ", ".join(_bound_names))
    if hessian_pd is False:
        _warnings.append("Hessian tidak positive definite (identifiability lemah)")
    elif hessian_pd is None:
        if _hess_timeout:
            _warnings.append("Hessian dibatalkan (budget waktu); SE tidak tersedia")
        elif _interrupted:
            _warnings.append("Hessian dibatalkan (interrupted); SE tidak tersedia")
        else:
            _warnings.append("Hessian tidak dihitung (compute_hessian=False)")
    _ll_flag = bool(np.isfinite(ll_per_obs) and ll_per_obs < 0.0)
    if _ll_flag:
        _warnings.append(
            f"log-lik per observasi negatif ({ll_per_obs:.3f}); indikasi misfit "
            "berat -- evaluaasi diagnostik wajib sebelum dipakai")
    if degenerate:
        _warnings.append("variance collapse / degenerate well terdeteksi")
    if _from_abort:
        _warnings.append(
            "semua start abort iterasi 0 (ABNORMAL/grad raksasa); titik abort "
            "terbaik dipakai dgn status invalid, bukan max_iter")
    certificate = {
        "valid": bool(
            bool(best_meta.get("success", False))
            and np.isfinite(log_lik)
            and _grad_finite and grad_norm <= cfg.grad_tol
            and not at_bound
            and (hessian_pd is not False)
        ),
        "optimum_valid": _optimum_valid,
        "grad_norm": float(grad_norm) if _grad_finite else None,
        "projected_grad": float(_proj) if _proj_finite else None,
        "at_bound": bool(at_bound),
        "bound_params": _bound_names,
        "hessian_pd": hessian_pd,
        "lambda_min": lambda_min,
        "ll_per_obs": float(ll_per_obs) if np.isfinite(ll_per_obs) else None,
        "warnings": _warnings,
    }

    return {
        "status": status,
        "at_bound": bool(at_bound),
        "certificate": certificate,
        "beta_hat": beta_hat,
        "logphi": logphi_hat,
        "phi": phi_hat,
        "logs": logs_hat,
        "raw_dep": raw_dep_hat,
        "b_hat": b_hat,
        "H_b_chol": H_b_chol,
        "sigma_area": sigma_area,
        "sigma_time": sigma_time,
        "sigma_gamma": sigma_gamma,
        "ksd": ksd,
        "q_dep": q_dep,
        "log_lik": log_lik,
        "AIC": aic,
        "BIC": bic,
        "k": k_total,
        "n": n,
        "p": p,
        "converged": status == "konvergen",
        "degenerate": degenerate,
        "optim": {
            "success": best_meta.get("success", False),
            "nit": best_meta.get("nit", -1),
            "iters": best_meta.get("nit", -1),
            "n_evals": best_meta.get("n_evals", -1),
            "max_iter": int(max_iter),
            "fun": best_meta.get("fun", best_val),
            "start": best_meta.get("start", -1),
            "n_starts": len(starts),
            "n_failed_starts": len(failed_starts),
            "failed_starts": list(failed_starts),
            "n_evals_total": _eval_count["n"],
            "n_dense_fallbacks": _eval_count["fb"],
            "jac": "analytic" if _jac_ok else "finite-diff",
            "message": best_meta.get("message", ""),
            "grad_norm": best_meta.get("grad_norm", np.nan),
            "ftol": best_meta.get("ftol", ftol),
            "gtol": best_meta.get("gtol", gtol),
            "hessian": hessian,
            "lambda_min": lambda_min,
            "hessian_pd": hessian_pd,
            "hessian_timeout": bool(_hess_timeout),
            "interrupted": bool(_interrupted),
            "all_starts_abort": bool(_from_abort),
        },
        "W": W,
        "X": X,
        "Z": Z,
        "des": des,
        "model": model,
        "y": y,
        "_boot_frame": data,
        "_boot_theta": np.array(best_param, dtype=np.float64, copy=True),
    }
