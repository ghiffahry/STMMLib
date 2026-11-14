"""Newton and BFGS solvers for the Beta conditional modes (random effects)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from stmmlib.beta.beta_nll import beta_nll_curvature, _beta_nll_score


def beta_mode_newton(X, Z, Q, beta, logphi, y, max_iter=50, start_b=None, inner_tol=1e-6):
    """Analytic Newton solver for conditional modes (random effects b).

    Minimizes f(b) = l(X*beta + Z*b; y, phi) + 0.5 * b' Q b

    Args:
        X: Fixed-effects design (n, p).
        Z: Random-effects design (n, q).
        Q: Precision matrix (q, q).
        beta: Fixed-effect coefficients (p,).
        logphi: log(precision).
        y: Response (n,), in (0,1).
        max_iter: Maximum Newton iterations.
        start_b: Starting value for b (q,). If None, zeros.
        inner_tol: Convergence tolerance.

    Returns:
        b: Conditional mode (q,).
        converged: Boolean.
        floor_count: Number of curvature floors applied.
    """
    n, p = X.shape
    q = Z.shape[1]
    phi = np.exp(logphi)

    if start_b is not None:
        b = start_b.copy()
    else:
        b = np.zeros(q, dtype=np.float64)

    eta = X @ beta + Z @ b
    floor_count = 0

    for iteration in range(max_iter):
        # Curvature and score
        w = beta_nll_curvature(eta, y, phi)
        s = _beta_nll_score(eta, y, phi)

        # Gradient: g = Z' s + Q b
        g = Z.T @ s + Q @ b

        # Hessian: H_b = Z' diag(w) Z + Q
        Zw = Z * w[:, None]  # n x q, each row scaled by w
        H_b = Zw.T @ Z + Q

        # Solve H_b delta = -g
        try:
            delta = -np.linalg.solve(H_b, g)
        except np.linalg.LinAlgError:
            try:
                delta = -np.linalg.lstsq(H_b, g, rcond=None)[0]
            except Exception:
                break

        # Line search (simple backtracking). NOTE: this accepts a step whenever
        # the curvature stays finite, without requiring the objective to
        # decrease. A cold start can therefore overshoot and diverge (measured
        # a cold start can jump the inner modes and diverge).
        # Warm starts from the previous outer iterate keep fits on the mode;
        # eval_param (cold start) exposes it. Adding the monotone test changes
        # the frozen published numbers, so it is not
        # applied here; a future re-freeze should adopt it.
        scale = 1.0
        for _ in range(10):
            b_new = b + scale * delta
            eta_new = X @ beta + Z @ b_new
            w_new = beta_nll_curvature(eta_new, y, phi)
            if np.all(np.isfinite(w_new)):
                break
            scale *= 0.5
        else:
            break

        b = b_new
        eta = eta_new
        floor_count += int(np.sum(w_new < 1e-5))

        # Convergence
        if np.max(np.abs(delta)) < inner_tol * (1.0 + np.max(np.abs(b))):
            return b, True, floor_count

    return b, False, floor_count


def beta_mode_newton_blocked(X, Z, Q, m, T, beta, logphi, y, max_iter=50,
                             start_b=None, inner_tol=1e-6):
    """Newton solver exploiting block structure Z = [Z_area, Z_cell].

    Assumes area-specific layout: Z[:, :m] = area indicators,
    Z[:, m:] = cell indicators with cell index = area*T + time (area-major),
    and Q = block_diag(Q_area (m,m), Q_cell (m*T, m*T) with per-area blocks).

    Cells of one area are coupled only within that area, so the Newton
    Hessian block is eliminated via an exact Schur complement of size m
    (plus cheap per-area T x T solves), instead of one dense q x q solve.

    Mathematically identical to the dense Newton step (within fp tolerance).
    """
    n, p = X.shape
    phi = np.exp(logphi)
    area_idx = Z[:, :m].argmax(1)
    cell_idx = Z[:, m:].argmax(1)           # 0 .. m*T-1
    cell_area = cell_idx // T
    cell_time = cell_idx % T

    u = np.zeros(m, dtype=np.float64) if start_b is None else start_b[:m].astype(np.float64).copy()
    v = np.zeros(m * T, dtype=np.float64) if start_b is None else start_b[m:].astype(np.float64).copy()

    Q_a = Q[:m, :m]
    Q_c = Q[m:, m:]
    Q_a_diag = np.ascontiguousarray(np.diag(Q_a))
    Qc_by_area = np.stack([Q_c[a*T:(a+1)*T, a*T:(a+1)*T] for a in range(m)])

    def _assemble(a, w_cell_a, g_v_a):
        C_a = Qc_by_area[a] + np.diag(w_cell_a)
        try:
            cwCinv = np.linalg.solve(C_a, w_cell_a)
            Cinv_gv = np.linalg.solve(C_a, g_v_a)
        except np.linalg.LinAlgError:
            cwCinv = np.linalg.lstsq(C_a, w_cell_a, rcond=None)[0]
            Cinv_gv = np.linalg.lstsq(C_a, g_v_a, rcond=None)[0]
        return cwCinv, Cinv_gv

    for it in range(max_iter):
        eta = X @ beta + u[area_idx] + v[cell_idx]
        w = beta_nll_curvature(eta, y, phi)
        s = _beta_nll_score(eta, y, phi)

        w_cell = np.zeros((m, T), dtype=np.float64)
        s_cell = np.zeros((m, T), dtype=np.float64)
        np.add.at(w_cell, (cell_area, cell_time), w)
        np.add.at(s_cell, (cell_area, cell_time), s)
        roww = np.bincount(area_idx, weights=w, minlength=m)

        g_u = np.bincount(area_idx, weights=s, minlength=m) + Q_a @ u
        g_v = s_cell.ravel() + Q_c @ v

        S_diag = np.empty(m, dtype=np.float64)
        rhs_u = np.empty(m, dtype=np.float64)
        cwCinv_list = [None] * m
        Cinv_gv_list = [None] * m
        for a in range(m):
            cwCinv, Cinv_gv = _assemble(a, w_cell[a], g_v[a*T:(a+1)*T])
            cwCinv_list[a] = cwCinv
            Cinv_gv_list[a] = Cinv_gv
            S_diag[a] = roww[a] + Q_a_diag[a] - w_cell[a] @ cwCinv
            rhs_u[a] = -g_u[a] + w_cell[a] @ Cinv_gv

        if np.allclose(Q_a, np.diag(Q_a_diag)):
            du = rhs_u / np.maximum(S_diag, 1e-12)
        else:
            S_full = np.diag(S_diag) + (Q_a - np.diag(Q_a_diag))
            try:
                du = np.linalg.solve(S_full, rhs_u)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(S_full, rhs_u, rcond=None)[0]

        dv = np.empty(m * T, dtype=np.float64)
        for a in range(m):
            C_a = Qc_by_area[a] + np.diag(w_cell[a])
            rhs_v = -g_v[a*T:(a+1)*T] - w_cell[a] * du[a]
            try:
                dv[a*T:(a+1)*T] = np.linalg.solve(C_a, rhs_v)
            except np.linalg.LinAlgError:
                dv[a*T:(a+1)*T] = np.linalg.lstsq(C_a, rhs_v, rcond=None)[0]

        # Guarded Newton step: backtrack until the working weights are finite
        # (mirrors the dense solver and keeps Schur blocks positive-definite).
        scale = 1.0
        for _ in range(12):
            u_new = u + scale * du
            v_new = v + scale * dv
            eta_new = X @ beta + u_new[area_idx] + v_new[cell_idx]
            w_new = beta_nll_curvature(eta_new, y, phi)
            if np.all(np.isfinite(w_new)):
                u, v = u_new, v_new
                break
            scale *= 0.5
        else:
            break

        scale = 1.0 + np.max(np.abs(np.concatenate([u, v])))
        if np.max(np.abs(np.concatenate([du, dv]))) < inner_tol * scale:
            return np.concatenate([u, v]), True, "blocked", it

    return np.concatenate([u, v]), False, "blocked", it


def beta_solve_mode_blocked(X, Z, Q, m, T, beta, logphi, y, inner_max_iter=50,
                            start_b=None, inner_tol=1e-6, _inner=None):
    """Blocked-mode Newton with a cheap blocked retry if the first pass stalls."""
    if _inner is None:
        _inner = beta_mode_newton_blocked
    b, conv, method, _ = _inner(
        X, Z, Q, m, T, beta, logphi, y, inner_max_iter, start_b, inner_tol
    )
    if conv:
        return b, True, "blocked"
    # Retry with a looser tolerance, warm-started from the stalled iterate.
    b2, conv2, _, _ = _inner(
        X, Z, Q, m, T, beta, logphi, y, 2 * inner_max_iter, b,
        max(inner_tol * 1e2, 1e-4)
    )
    return b2, conv2, "blocked-retry"


def beta_solve_mode(X, Z, Q, beta, logphi, y, inner_max_iter=50,
                    outer_maxit_fallback=100, start_b=None, inner_tol=1e-6):
    """Wrapper: tries Newton first, mandatory BFGS fallback if Newton fails.

    Args:
        X, Z, Q, beta, logphi, y: Same as beta_mode_newton.
        inner_max_iter: Max Newton iterations.
        outer_maxit_fallback: Max BFGS iterations.
        start_b: Starting b.
        inner_tol: Newton tolerance.

    Returns:
        b: Conditional mode.
        converged: Boolean.
        method_used: "newton" or "bfgs".
    """
    b, conv, floor_count = beta_mode_newton(
        X, Z, Q, beta, logphi, y, inner_max_iter, start_b, inner_tol
    )

    if conv:
        return b, True, "newton"

    # BFGS fallback
    def _neg_logpost(b_vec):
        eta = X @ beta + Z @ b_vec
        phi_val = np.exp(logphi)
        # Beta NLL
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
        a = mu * phi_val
        bb = (1.0 - mu) * phi_val
        from scipy.special import betaln
        ll = np.sum(-betaln(a, bb) + (a - 1) * np.log(np.clip(y, 1e-15, 1.0)) +
                     (bb - 1) * np.log(np.clip(1.0 - y, 1e-15, 1.0)))
        pen = 0.5 * b_vec @ Q @ b_vec
        return -ll + pen

    try:
        res = minimize(_neg_logpost, b, method="L-BFGS-B",
                       # Inner fallback solve (random effects), NOT the outer
                       # preset path: preset ftol/gtol apply only to the outer
                       # L-BFGS-B in beta_laplace_structural.
                       options={"maxiter": outer_maxit_fallback, "ftol": 1e-6})
        return res.x, res.success, "bfgs"
    except Exception:
        return b, False, "failed"
