"""Negative Laplace marginal log-likelihood."""

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


def neg_marginal_l(ctx, param_vec):
    """Negative Laplace marginal log-likelihood."""
    T = ctx["T"]
    X = ctx["X"]
    Z = ctx["Z"]
    _build_Q = ctx["_build_Q"]
    _eval_count = ctx["_eval_count"]
    _jac_ok = ctx["_jac_ok"]
    _logdet_fn = ctx["_logdet_fn"]
    _mode_blocked = ctx["_mode_blocked"]
    _schur_inner = ctx["_schur_inner"]
    _state = ctx["_state"]
    _use_blocked = ctx["_use_blocked"]
    cfg = ctx["cfg"]
    inner_max_iter = ctx["inner_max_iter"]
    inner_tol = ctx["inner_tol"]
    ksd = ctx["ksd"]
    m = ctx["m"]
    max_iter = ctx["max_iter"]
    model = ctx["model"]
    p = ctx["p"]
    verbose = ctx["verbose"]
    y = ctx["y"]
    _eval_count["n"] += 1
    beta = param_vec[:p]
    logphi = param_vec[p]
    logs = param_vec[p + 1: p + 1 + ksd]
    raw_dep = param_vec[p + 1 + ksd:]

    phi = np.exp(logphi)

    try:
        Q_full = _build_Q(logs, raw_dep)
    except Exception as _e:
        if _os_env_debug():
            import traceback as _tb
            _tb.print_exc()
        return 1e20

    # Q and Z must agree; a silent pad/truncate hides structural bugs.
    q = Z.shape[1]
    if Q_full.shape[0] != q:
        raise PrecisionShapeError(
            f"Precision/design size mismatch: Q is {Q_full.shape[0]}x"
            f"{Q_full.shape[1]} but Z has {q} columns for {model!r}."
        )

    # Inner solve: find modes b (warm-start from previous outer iterate)
    if _use_blocked:
        b, conv, _ = _mode_blocked(X, Z, Q_full, m, T, beta, logphi, y,
                                   inner_max_iter, start_b=_state["b"],
                                   inner_tol=inner_tol)
        # If the blocked Newton diverges (rare, during extreme outer probes),
        # fall back to the dense Newton + BFGS path so b stays well-defined.
        if not conv:
            _eval_count["fb"] += 1
            b, conv, _ = beta_solve_mode(X, Z, Q_full, beta, logphi, y,
                                         inner_max_iter, max_iter, start_b=_state["b"],
                                         inner_tol=inner_tol)
    else:
        b, conv, _ = beta_solve_mode(X, Z, Q_full, beta, logphi, y,
                                     inner_max_iter, max_iter, start_b=_state["b"],
                                     inner_tol=inner_tol)
    # Stationarity polish (jac models only): the analytic gradient needs
    # the mode to satisfy Z's_nll + Qb = 0 tightly (envelope theorem +
    # implicit db/dtheta). Fallback/retry modes can stop with a loose
    # residual (~1e-1), which poisons every gradient entry. Warm extra
    # blocked Newton iterations cost ~2-3 cheap solves when needed.
    if _jac_ok:
        import os as _os
        # Probe Hessian (FD eps 1e-4 sudah aproksimasi): inner super-presisi
        # terbuang, pakai toleransi reguler saat _state["hessian_probe"] set.
        _polish_iter, _polish_tol = ((inner_max_iter, inner_tol)
                                     if _state.get("hessian_probe") else (100, 1e-9))
        for _ in range(3):
            g_stat = Z.T @ _beta_nll_score(eta_hat_pol := X @ beta + Z @ b, y, phi) \
                + Q_full @ b
            if _os.environ.get("STMM_DEBUG"):
                print(f"    [polish] resid={np.max(np.abs(g_stat)):.3e} "
                      f"max|b|={np.max(np.abs(b)):.4f}", flush=True)
            if np.max(np.abs(g_stat)) < 1e-8 * (1.0 + np.max(np.abs(b))):
                break
            b, _, _, _ = beta_mode_newton_blocked(
                X, Z, Q_full, m, T, beta, logphi, y, _polish_iter,
                start_b=b, inner_tol=_polish_tol)
    _state["b"] = b.copy()

    # Laplace approximation
    eta_hat = X @ beta + Z @ b
    mu_hat = 1.0 / (1.0 + np.exp(-eta_hat))
    mu_hat = np.clip(mu_hat, 1e-15, 1.0 - 1e-15)

    from scipy.special import betaln
    a_hat = mu_hat * phi
    b_hat = (1.0 - mu_hat) * phi
    nll = -np.sum(-betaln(a_hat, b_hat) + (a_hat - 1) * np.log(np.clip(y, 1e-15, 1.0)) +
                   (b_hat - 1) * np.log(np.clip(1.0 - y, 1e-15, 1.0)))

    pen = 0.5 * b @ Q_full @ b

    # Log-determinants
    try:
        if _schur_inner:
            logdet_H, logdet_Q = _logdet_fn(
                Z, beta_nll_curvature(eta_hat, y, phi), Q_full, m, T
            )
        else:
            logdet_Q = np.linalg.slogdet(Q_full)[1]
            H = Z.T @ (Z * beta_nll_curvature(eta_hat, y, phi)[:, None]) + Q_full
            logdet_H = np.linalg.slogdet(H)[1]
    except Exception:
        # A failed logdet is not a zero logdet: replacing it with 0
        # silently corrupts the objective. Treat it like the Q-assembly
        # guard above (reject the probe point).
        if _os_env_debug():
            import traceback as _tb
            _tb.print_exc()
        return 1e20

    # Laplace approx: l = log p(y|b_hat) + log N(b_hat;0,Q^-1) + 0.5 log det(2pi H^-1)
    #   log N(b;0,Q^-1) = -0.5 b'Q b + 0.5 log|Q|                 (drop 0.5 q log 2pi)
    #   0.5 log det(2pi H^-1) = -0.5 log|H| (+ const)             (drop 0.5 q log 2pi)
    loglik = -nll - pen + 0.5 * logdet_Q - 0.5 * logdet_H

    # Weak-prior MAP penalty (opt-in; default None = off, identical to ML).
    pen_prior = 0.0
    if cfg.phi_prior_sd is not None:
        pen_prior += 0.5 * ((logphi - cfg.phi_prior_mean) / cfg.phi_prior_sd) ** 2
    if cfg.dep_prior_sd is not None and raw_dep.size > 0:
        pen_prior += 0.5 * float(np.sum(raw_dep ** 2)) / cfg.dep_prior_sd ** 2
    obj = -loglik + pen_prior

    if not np.isfinite(obj):
        return 1e20
    _state["cache"] = {"x": param_vec.copy(), "b": b.copy()}
    if verbose and _eval_count["n"] % max(int(verbose), 1) == 0:
        print(f"    [eval {_eval_count['n']:>6}] -loglik={obj:.4f}", flush=True)
    return obj
