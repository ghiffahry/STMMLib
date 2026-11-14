"""Analytic gradient of the negative Laplace marginal log-likelihood."""

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


def laplace_grad(ctx, param_vec):
    """Analytic gradient of the negative Laplace marginal log-likelihood."""
    T = ctx["T"]
    W = ctx["W"]
    X = ctx["X"]
    Z = ctx["Z"]
    _build_Q = ctx["_build_Q"]
    _has_ar1 = ctx["_has_ar1"]
    _has_sar = ctx["_has_sar"]
    _state = ctx["_state"]
    cfg = ctx["cfg"]
    inner_max_iter = ctx["inner_max_iter"]
    inner_tol = ctx["inner_tol"]
    ksd = ctx["ksd"]
    m = ctx["m"]
    n_area = ctx["n_area"]
    n_time = ctx["n_time"]
    p = ctx["p"]
    param_dim = ctx["param_dim"]
    q_dep = ctx["q_dep"]
    rb = ctx["rb"]
    y = ctx["y"]
    beta = param_vec[:p]
    logphi = param_vec[p]
    logs = param_vec[p + 1: p + 1 + ksd]
    raw_dep = param_vec[p + 1 + ksd:]
    phi = np.exp(logphi)

    cache = _state.get("cache")
    if cache is not None and cache["x"].shape == param_vec.shape \
            and np.array_equal(cache["x"], param_vec):
        b = cache["b"]
    else:
        # Mode solve at this theta. MUST be tightly converged: the
        # envelope theorem (total = partial derivative) only holds at the
        # exact inner mode; a sloppy b poisons every entry of the grad.
        blocks_q0 = _build_Q(logs, raw_dep)
        Q_q0 = blocks_q0
        b, conv, _, _ = beta_mode_newton_blocked(
            X, Z, Q_q0, m, T, beta, logphi, y, max(200, inner_max_iter * 4),
            start_b=_state["b"], inner_tol=min(inner_tol, 1e-8))

    eta = X @ beta + Z @ b
    w = beta_nll_curvature(eta, y, phi)
    s_pos = -_beta_nll_score(eta, y, phi)          # d l / d eta
    pieces = _beta_grad_pieces(eta, y, phi)

    Q_full = _build_Q(logs, raw_dep)
    # Objective Hessian uses the FLOORED curvature (matches the logdet in
    # the marginal); the stationarity Jacobian d(Z's_nll + Qb)/db uses the
    # UNFLOORED w_raw (the score is never floored). Two different matrices
    # whenever any curvature entry hits the floor.
    terms = _blocked_hinv_terms(Z, w, Q_full, m, T)
    terms_s = _blocked_hinv_terms(Z, pieces["w_raw"], Q_full, m, T)
    pen_area, pen_cell = _blocked_pen_terms(b, Q_full, m, T)

    grad_ll = np.zeros(param_dim, dtype=np.float64)
    # Base partial derivatives (b frozen) ...
    grad_ll[:p] = X.T @ s_pos - 0.5 * (X.T @ (pieces["w_dot"] * terms["D"]))
    grad_ll[p] = phi * (float(np.sum(pieces["l_phi"]))
                        - 0.5 * float(np.sum(pieces["w_dphi"] * terms["D"])))
    grad_ll[p + 1] = pen_area - n_area + terms["tr_area"]
    grad_ll[p + 2] = pen_cell - n_time + terms["tr_cell"]

    # ... PLUS the mode-shift correction. The envelope theorem does NOT
    # apply naively: logdet(H(b)) depends on b, so at the joint mode
    # dL/db = -0.5 tr(H^-1 dH/db) = -0.5 u != 0 with u = Z'(w_dot * D).
    # Total derivative = partial + (-0.5 u') db_hat/dtheta, with
    # db_hat/dtheta from implicit differentiation of the stationarity
    # Z's_nll + Qb = 0  =>  db/dtheta = -Hinv_stat dg/dtheta.
    wdotD = pieces["w_dot"] * terms["D"]
    u = np.zeros(Z.shape[1], dtype=np.float64)
    u[:m] = np.bincount(terms["area_idx"], weights=wdotD, minlength=m)
    np.add.at(u, m + terms["cell_idx"], wdotD)
    hvec = terms_s["Hinv"] @ u
    v = hvec[terms["area_idx"]] + hvec[m + terms["cell_idx"]]
    # beta: dg/dbeta_k = Z'(w_raw * x_k)  =>  db = -Hinv Z'(w_raw x_k)
    grad_ll[:p] += 0.5 * (X.T @ (pieces["w_raw"] * v))
    # logphi: dg/dlogphi = -phi * Z'(t g + phi t Gphi) = -phi Z' s_dphi
    grad_ll[p] += -0.5 * phi * float(v @ pieces["s_dphi"])
    # log_sigma: stationarity g = Z's_nll + Qb = 0; dg/dlog_sigma =
    # (dQ/dlog_sigma) b = -2 Q_blk b  =>  db = -Hinv dg = +2 Hinv Q_blk b
    grad_ll[p + 1] += -float(hvec[:m] @ (Q_full[:m, :m] @ b[:m]))
    bc2 = b[m:].reshape(m, T)
    Qc_stack = np.stack([Q_full[m + a * T: m + (a + 1) * T,
                                m + a * T: m + (a + 1) * T] for a in range(m)])
    grad_ll[p + 2] += -float(np.einsum("at,atu,au->",
                                       hvec[m:].reshape(m, T), Qc_stack, bc2))

    # raw_dep: SAR rho and/or AR1 phi (blocked path).
    # dQ/draw = e^{-2s} * (ddep/draw) * structural derivative.
    # base: -0.5 b'dQ b + 0.5 tr(Qinv dQ) - 0.5 tr(Hinv dQ)
    # corr: dg/draw = (dQ/draw) b  =>  corr = +0.5 hvec' (dQ_emb b)
    if q_dep > 0:
        _rid = 0
        if _has_sar:
            rr = float(raw_dep[0])
            _th = np.tanh(rr)
            rho = rb * _th
            drho = rb * (1.0 - _th ** 2)
            W_arr = W.toarray() if hasattr(W, "toarray") else np.asarray(W, dtype=np.float64)
            W_arr = np.asarray(W_arr, dtype=np.float64)[:m, :m]
            A_s = np.eye(m, dtype=np.float64) - rho * W_arr
            # d(A'A)/d rho = -(A'W + W'A);  d rho/d raw = drho
            dQ_a = np.exp(-2.0 * logs[0]) * drho * (-(A_s.T @ W_arr + W_arr.T @ A_s))
            tr_Qinv_dQ = -drho * float(np.trace(
                np.linalg.solve(A_s.T @ A_s, A_s.T @ W_arr + W_arr.T @ A_s)))
            pen_dQ = float(b[:m] @ dQ_a @ b[:m])
            tr_H_dQ = float(np.sum(terms["Hinv"][:m, :m] * dQ_a))
            corr_dQ = float(hvec[:m] @ (dQ_a @ b[:m]))
            grad_ll[p + 1 + ksd + 0] = (
                -0.5 * pen_dQ + 0.5 * tr_Qinv_dQ - 0.5 * tr_H_dQ + 0.5 * corr_dQ)
            _rid = 1
        if _has_ar1:
            rr = float(raw_dep[_rid])
            # Frozen double squash via cfg.phi_bound (baseline M2/M5 depend
            # on it); gradient differentiates the SAME mapping.
            _pb = cfg.phi_bound
            u_sq = _pb * np.tanh(rr)
            phi_t = _pb * np.tanh(u_sq)
            dphi_t = (_pb * (1.0 - np.tanh(u_sq) ** 2)
                      * _pb * (1.0 - np.tanh(rr) ** 2))
            # A(phi): diag 1 (edges), 1+phi^2 (interior), off-diag -phi
            Af = np.eye(T, dtype=np.float64)
            if T > 2:
                Af[np.arange(1, T - 1), np.arange(1, T - 1)] += phi_t ** 2
            if T > 1:
                Af += np.diag(np.full(T - 1, -phi_t), 1) \
                    + np.diag(np.full(T - 1, -phi_t), -1)
                dAf = (np.diag(np.full(T - 1, -1.0), 1)
                       + np.diag(np.full(T - 1, -1.0), -1))
                if T > 2:
                    dAf[np.arange(1, T - 1), np.arange(1, T - 1)] += 2.0 * phi_t
            else:
                dAf = np.zeros((1, 1), dtype=np.float64)
            dQc = np.exp(-2.0 * logs[1]) * dphi_t * dAf
            tr_Qinv_dQ = m * dphi_t * float(np.trace(np.linalg.solve(Af, dAf)))
            pen_dQ = float(np.einsum("at,atu,au->", bc2,
                                     np.broadcast_to(dQc, (m, T, T)), bc2))
            Hinv_cell = np.stack([terms["Hinv"][m + a * T: m + (a + 1) * T,
                                                m + a * T: m + (a + 1) * T]
                                  for a in range(m)])
            tr_H_dQ = float(np.einsum("atu,tu->", Hinv_cell, dQc))
            corr_dQ = float(np.einsum("at,atu,au->",
                                      hvec[m:].reshape(m, T),
                                      np.broadcast_to(dQc, (m, T, T)), bc2))
            grad_ll[p + 1 + ksd + _rid] = (
                -0.5 * pen_dQ + 0.5 * tr_Qinv_dQ - 0.5 * tr_H_dQ + 0.5 * corr_dQ)

    if not np.all(np.isfinite(grad_ll)):
        # Extreme outer probe (eta overflow etc.): abort this start loudly;
        # the starts loop try/except moves to the next start. Mirrors the
        # 1e20 objective guard in _neg_marginal_l.
        raise FloatingPointError("non-finite analytic gradient at probe point")
    neg_grad = -grad_ll
    # Weak-prior MAP penalty gradient (opt-in; default off).
    if cfg.phi_prior_sd is not None:
        neg_grad[p] += (logphi - cfg.phi_prior_mean) / cfg.phi_prior_sd ** 2
    if cfg.dep_prior_sd is not None and q_dep > 0:
        neg_grad[p + 1 + ksd:] += raw_dep / cfg.dep_prior_sd ** 2
    return neg_grad
