"""Newton solver, BFGS fallback, and Laplace marginal likelihood engine.

Core computational engine for Beta mixed model fitting:
- Inner loop: Newton solver for conditional modes (random effects b)
- Outer loop: L-BFGS-B optimization of Laplace marginal likelihood
- Analytical curvature (digamma/trigamma) for Beta NLL
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, expit, polygamma

if TYPE_CHECKING:
    from stmmlib._backend import Backend


def _os_env_debug() -> bool:
    import os
    return os.environ.get("STMM_DEBUG", "") != ""


# Gradient-norm tolerance for a "genuine" convergence verdict. L-BFGS-B fills
# `res.jac` (finite-difference gradient) at the final point; a small norm there
# means the first-order condition holds. This is a second signal alongside
# `res.success` because scipy sometimes flags success=False on bounded problems
# even at a valid minimum. Tunable: pass GRAD_TOL=... to override the default.
GRAD_TOL = 1.0


def _projected_grad_norm(x, grad, bounds, rel_tol=1e-12):
    """Projected-gradient norm for box-constrained optimum (L-BFGS-B KKT).

    For each parameter i with bound (lo, hi):
        pg_i = 0 if (x_i == lo and g_i > 0) or (x_i == hi and g_i < 0),
        else pg_i = g_i.
    Components pointing outside the box admit no feasible descent move, so
    they collapse to zero; the norm is ||pg|| over the feasible components.

    Bound equality uses a relative tolerance: |x_i - bound| <= rel_tol *
    max(1, |lo|, |hi|, |x_i|), with rel_tol = 1e-12 by default. Infinite
    bounds never count as active.

    Two-level certificate semantics:
        valid = inference-grade (raw grad <= grad_tol AND not at_bound AND
            Hessian PD). Point and Wald inference allowed.
        optimum_valid = constrained-optimum KKT (projected grad <= grad_tol).
            Point estimate and prediction allowed; Wald SE/CI/p not allowed
            when valid is False (boundary inference needs a mixture LRT).

    Refs: Byrd et al. 1995 (L-BFGS-B projected-gradient stopping); Zhu, Byrd
    and Nocedal 1997 (L-BFGS-B projected gradient); Self and Liang 1987
    (boundary inference needs mixture LRT, reason for two levels).
    """
    try:
        xv = np.asarray(x, dtype=np.float64).ravel()
        gv = np.asarray(grad, dtype=np.float64).ravel()
    except Exception:
        return None
    if xv.shape != gv.shape or len(bounds) != len(xv):
        return None
    try:
        pg = gv.copy()
        for i, (_lo, _hi) in enumerate(bounds):
            xi = float(xv[i])
            gi = float(gv[i])
            if not np.isfinite(xi) or not np.isfinite(gi):
                continue
            _scale = max(1.0, abs(float(_lo)) if np.isfinite(_lo) else 0.0,
                         abs(float(_hi)) if np.isfinite(_hi) else 0.0,
                         abs(xi))
            _tol = float(rel_tol) * _scale
            _at_lo = bool(np.isfinite(_lo) and abs(xi - float(_lo)) <= _tol)
            _at_hi = bool(np.isfinite(_hi) and abs(xi - float(_hi)) <= _tol)
            if (_at_lo and gi > 0.0) or (_at_hi and gi < 0.0):
                pg[i] = 0.0
        _n = float(np.linalg.norm(pg))
        return _n if np.isfinite(_n) else None
    except Exception:
        return None


def _num_hessian(f, x, eps=1e-4):
    """Symmetric finite-difference Hessian of scalar f at x (central, O(n^2))."""
    x = np.asarray(x, dtype=np.float64)
    n_ = len(x)
    H = np.zeros((n_, n_))
    f0 = f(x)
    for i in range(n_):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        H[i, i] = (f(xp) - 2 * f0 + f(xm)) / eps ** 2
        for j in range(i + 1, n_):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps ** 2)
    return H


def _num_hessian_grad(grad_fun, x, eps=1e-4):
    """Symmetric finite-difference Hessian from an analytic gradient.

    Central difference per column: H[:, j] = (g(x + h e_j) - g(x - h e_j)) / (2h)
    with h = eps * max(1, |x_j|), then symmetrized. Costs 2n gradient
    evaluations vs ~2n^2 objective evaluations for _num_hessian, the cheap
    Hessian path once the analytic gradient of the same objective exists.
    """
    x = np.asarray(x, dtype=np.float64)
    n_ = len(x)
    H = np.zeros((n_, n_))
    for j in range(n_):
        h = eps * max(1.0, abs(float(x[j])))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        H[:, j] = (grad_fun(xp) - grad_fun(xm)) / (2.0 * h)
    return 0.5 * (H + H.T)


# ---------------------------------------------------------------------------
# Beta NLL curvature (analytical)
# ---------------------------------------------------------------------------


def _block_diag_dense(blocks):
    """Dense block-diagonal assembly: one allocation, no sparse roundtrip.

    Replaces sp_block_diag(...).toarray() which builds a sparse matrix from
    dense blocks only to densify it again (dominant per-eval cost at q ~ 1e3).
    """
    blocks = [b.toarray() if hasattr(b, "toarray")
              else np.asarray(b, dtype=np.float64) for b in blocks]
    sizes = [b.shape[0] for b in blocks]
    out = np.zeros((sum(sizes), sum(sizes)), dtype=np.float64)
    off = 0
    for blk, sz in zip(blocks, sizes):
        out[off:off + sz, off:off + sz] = blk
        off += sz
    return out


def beta_nll_curvature(eta, y, phi):
    """Analytical curvature (working weight) for Beta NLL.

    w_i = -d^2 l_i / d eta_i^2

    With mu = logit^{-1}(eta), m1=mu, m2=1-mu:
        h_i = -phi^2 [psi_1(mu*phi) + psi_1((1-mu)*phi)] (m1*m2)^2
              + phi [psi_0((1-mu)*phi) - psi_0(mu*phi) + logit(y)] (m1*m2)(1-2*mu)

    w_i = max(-h_i, floor)

    Args:
        eta: Linear predictor (n,).
        y: Response (n,), in (0,1).
        phi: Precision parameter.

    Returns:
        w: Working weights (n,), floored for PD guarantee.
    """
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
    m1 = mu
    m2 = 1.0 - mu

    a = mu * phi
    b = (1.0 - mu) * phi

    psi1_a = polygamma(1, a)
    psi1_b = polygamma(1, b)
    psi0_a = digamma(a)
    psi0_b = digamma(b)

    logit_y = np.log(np.clip(y, 1e-15, 1.0 - 1e-15) / np.clip(1.0 - y, 1e-15, 1.0))

    h = (
        -phi**2 * (psi1_a + psi1_b) * (m1 * m2) ** 2
        + phi * (psi0_b - psi0_a + logit_y) * (m1 * m2) * (1.0 - 2.0 * mu)
    )

    w = np.maximum(-h, 1e-6)
    return w


def _beta_nll_score(eta, y, phi):
    """Score (gradient) of Beta NLL per observation.

    s_i = -phi * mu * (1-mu) [psi_0((1-mu)*phi) - psi_0(mu*phi) + log(y) - log(1-y)]
    """
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-15, 1.0 - 1e-15)

    a = mu * phi
    b = (1.0 - mu) * phi

    psi0_a = digamma(a)
    psi0_b = digamma(b)

    log_y = np.log(np.clip(y, 1e-15, 1.0))
    log_1my = np.log(np.clip(1.0 - y, 1e-15, 1.0))

    s = -phi * mu * (1.0 - mu) * (psi0_b - psi0_a + log_y - log_1my)
    return s


def _beta_grad_pieces(eta, y, phi):
    """Analytic derivative pieces of the Beta log-lik for the Laplace gradient.

    Returns dict with (all elementwise over obs):
      l_phi : d l_i / d phi                        (for the -nll term)
      w_dot : d w_i / d eta   (= -d^2 s / d eta^2) (beta log-det term)
      w_dphi: d w_i / d phi at fixed eta           (log(phi) log-det term)
    Both w-derivatives are zeroed where the curvature floor (1e-6) binds,
    matching beta_nll_curvature's max(-h, floor).

    Derivation (envelope theorem: at the inner mode b_hat the total
    derivative of the Laplace marginal equals the partial derivative with
    b frozen -- TMB paper, Kristensen et al. 2016, JSS 70):
      l_i   = -betaln(a,b) + (a-1)ln y + (b-1)ln(1-y),  a=mu*phi, b=(1-mu)*phi
      h     = d^2 l/d eta^2 = phi*t*(1-2mu)*g - phi^2*t^2*A
      w     = -h,  t = mu(1-mu),  g = logit(y) - psi(a) + psi(b)
      A     = psi1(a) + psi1(b),  D2 = psi2(a) - psi2(b)
      w_dot = -dh/d eta = phi*t*g*[(1-2mu)^2 - 2t] ... (see code)
    """
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
    t = mu * (1.0 - mu)
    a = mu * phi
    b = (1.0 - mu) * phi

    psi0_a, psi0_b = digamma(a), digamma(b)
    psi1_a, psi1_b = polygamma(1, a), polygamma(1, b)
    psi2_a, psi2_b = polygamma(2, a), polygamma(2, b)

    log_y = np.log(np.clip(y, 1e-15, 1.0))
    log_1my = np.log(np.clip(1.0 - y, 1e-15, 1.0))
    logit_y = log_y - log_1my

    g = logit_y - psi0_a + psi0_b           # d l / d mu = phi * t * g
    A = psi1_a + psi1_b
    D2 = psi2_a - psi2_b
    om = 1.0 - 2.0 * mu                      # (1 - 2 mu)

    # l_phi = d l / d phi  (= -[psi(a) mu + psi(b)(1-mu) - psi(a+b)] + mu ln y
    #        + (1-mu) ln(1-y);  a+b = phi so lnGamma(a+b) contributes +psi(phi))
    l_phi = (-(psi0_a * mu + psi0_b * (1.0 - mu)) + digamma(phi)
             + mu * log_y + (1.0 - mu) * log_1my)

    # h = phi*om*t*g - phi^2*t^2*A ; w = -h
    h = phi * om * t * g - phi**2 * t**2 * A
    floor_mask = h >= -1e-6                  # w floored at 1e-6 here

    # dh/d eta = phi*t*g*(om^2 - 2t) - 3 phi^2 om t^2 A - phi^3 t^3 D2
    dh_deta = (phi * t * g * (om**2 - 2.0 * t)
               - 3.0 * phi**2 * om * t**2 * A
               - phi**3 * t**3 * D2)
    w_dot = -dh_deta

    # dh/d phi at fixed eta:
    #   d g/d phi = -mu*psi1(a) + (1-mu)*psi1(b)   (mixed signs!)
    #   d A/d phi =  mu*psi2(a) + (1-mu)*psi2(b)   = F
    Gphi = -mu * psi1_a + (1.0 - mu) * psi1_b
    F = mu * psi2_a + (1.0 - mu) * psi2_b
    dh_dphi = (om * t * g + phi * om * t * Gphi
               - 2.0 * phi * t**2 * A - phi**2 * t**2 * F)
    w_dphi = -dh_dphi

    w_dot = np.where(floor_mask, 0.0, w_dot)
    w_dphi = np.where(floor_mask, 0.0, w_dphi)
    return {"l_phi": l_phi, "w_dot": w_dot, "w_dphi": w_dphi,
            "w_raw": -h, "s_dphi": t * g + phi * t * Gphi}


def _blocked_hinv_terms(Z, w, Q, m, T):
    """Exact H^{-1} and Schur traces for blocked (area-specific) models.

    H = Z'diag(w)Z + Q with Z = [Z_area | Z_cell] one-hot layout.
    Schur: C_a = Qc_a + diag(w_cell[a]); S = diag(roww) + Q_a - diag(delta).
    Returns dict:
      Hinv    : dense (q, q) inverse  (q = m + m*T)
      D       : diag(Z Hinv Z') (n,) via one-hot indexing -- O(n)
      tr_area : tr(Hinv[:m,:m] Q_a)
      tr_cell : sum_a tr(Hinv[blk_a, blk_a] Qc_a)
    Mathematically identical to dense inv(H) (within fp tolerance).
    """
    n = Z.shape[0]
    area_idx = Z[:, :m].argmax(1)
    cell_idx = Z[:, m:].argmax(1)
    cell_area = cell_idx // T
    cell_time = cell_idx % T
    Q_a = Q[:m, :m]
    Qc_stack = np.stack([Q[m + a * T: m + (a + 1) * T, m + a * T: m + (a + 1) * T]
                         for a in range(m)])

    w_cell = np.zeros((m, T), dtype=np.float64)
    np.add.at(w_cell, (cell_area, cell_time), w)
    roww = np.bincount(area_idx, weights=w, minlength=m)

    Cinv = np.empty((m, T, T), dtype=np.float64)
    Cinvw = np.empty((m, T), dtype=np.float64)
    cwCinv = np.empty((m, T), dtype=np.float64)
    delta = np.empty(m, dtype=np.float64)
    for a in range(m):
        C_a = Qc_stack[a] + np.diag(w_cell[a])
        try:
            Cinv[a] = np.linalg.inv(C_a)
            Cinvw[a] = Cinv[a] @ w_cell[a]
            cwCinv[a] = Cinv[a].T @ w_cell[a]
        except np.linalg.LinAlgError:
            Cinv[a] = np.linalg.pinv(C_a)
            Cinvw[a] = Cinv[a] @ w_cell[a]
            cwCinv[a] = Cinv[a].T @ w_cell[a]
        delta[a] = w_cell[a] @ cwCinv[a]

    S = np.diag(roww) + Q_a - np.diag(delta)
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)

    q = m + m * T
    Hinv = np.empty((q, q), dtype=np.float64)
    Hinv[:m, :m] = Sinv
    # Off-diagonal: Hinv[:m, blk_a] = -Sinv[:, a] cwCinv[a]'
    Hinv[:m, m:] = -(Sinv[:, :, None] * cwCinv[None, :, :]).reshape(m, m * T)
    Hinv[m:, :m] = Hinv[:m, m:].T
    # Cell blocks: Cinv_a (diag blocks) + Sinv[a,b] outer(Cinvw_a, cwCinv_b)
    cross = np.einsum("at,ab,bu->atbu", Cinvw, Sinv, cwCinv).reshape(m * T, m * T)
    Hinv[m:, m:] = cross
    for a in range(m):
        Hinv[m + a * T: m + (a + 1) * T, m + a * T: m + (a + 1) * T] += Cinv[a]

    D = (Hinv[area_idx, area_idx]
         + 2.0 * Hinv[area_idx, m + cell_idx]
         + Hinv[m + cell_idx, m + cell_idx])

    tr_area = float(np.sum(Hinv[:m, :m] * Q_a))
    tr_cell = 0.0
    for a in range(m):
        tr_cell += float(np.sum(
            Hinv[m + a * T: m + (a + 1) * T, m + a * T: m + (a + 1) * T] * Qc_stack[a]))

    return {"Hinv": Hinv, "D": D, "tr_area": tr_area, "tr_cell": tr_cell,
            "area_idx": area_idx, "cell_idx": cell_idx}


def _blocked_pen_terms(b, Q, m, T):
    """b'Qb split by block: (area part, cell part) for the penalty gradient."""
    b_a = b[:m]
    b_c = b[m:].reshape(m, T)
    Q_a = Q[:m, :m]
    pen_area = float(b_a @ Q_a @ b_a)
    Qc_stack = np.stack([Q[m + a * T: m + (a + 1) * T, m + a * T: m + (a + 1) * T]
                         for a in range(m)])
    pen_cell = float(np.einsum("at,atu,au->", b_c, Qc_stack, b_c))
    return pen_area, pen_cell


# ---------------------------------------------------------------------------
# Newton solver for conditional modes
# ---------------------------------------------------------------------------


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

        # Line search (simple backtracking)
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


# ---------------------------------------------------------------------------
# Laplace marginal likelihood engine
# ---------------------------------------------------------------------------


def laplace_block_logdets(Z, w, Q, m, T):
    """logdet(H) and logdet(Q) via exact Schur complement (area-specific, iid cell).

    H = Z' diag(w) Z + Q, Q = block_diag(Q_area, Q_cell with per-area blocks).
    Per-area cells are coupled only within their own area, so:
        logdet(H) = logdet(C) + logdet(S),  S = A - B C^{-1} B'
    where the C (per-area) and S (m x m) blocks are small.
    Mathematically identical to slogdet of the dense matrix (within fp tolerance).
    """
    n = Z.shape[0]
    area_idx = Z[:, :m].argmax(1)
    cell_idx = Z[:, m:].argmax(1)
    cell_area = cell_idx // T
    cell_time = cell_idx % T
    Q_a = Q[:m, :m]
    Q_c = Q[m:, m:]

    roww = np.bincount(area_idx, weights=w, minlength=m)
    w_cell = np.zeros((m, T), dtype=np.float64)
    np.add.at(w_cell, (cell_area, cell_time), w)

    Q_a_diag = np.ascontiguousarray(np.diag(Q_a))

    logdet_C = 0.0
    S_diag = np.empty(m, dtype=np.float64)
    for a in range(m):
        C_a = Q_c[a*T:(a+1)*T, a*T:(a+1)*T] + np.diag(w_cell[a])
        logdet_C += np.linalg.slogdet(C_a)[1]
        try:
            cwCinv = np.linalg.solve(C_a, w_cell[a])
        except np.linalg.LinAlgError:
            cwCinv = np.linalg.lstsq(C_a, w_cell[a], rcond=None)[0]
        S_diag[a] = roww[a] + Q_a_diag[a] - w_cell[a] @ cwCinv

    if np.allclose(Q_a, np.diag(Q_a_diag)):
        logdet_S = np.sum(np.log(np.maximum(S_diag, 1e-12)))
    else:
        S_full = np.diag(S_diag) + (Q_a - np.diag(Q_a_diag))
        logdet_S = np.linalg.slogdet(S_full)[1]

    logdet_H_block = logdet_C + logdet_S

    logdet_Q = np.linalg.slogdet(Q_a)[1]
    for a in range(m):
        logdet_Q += np.linalg.slogdet(Q_c[a*T:(a+1)*T, a*T:(a+1)*T])[1]

    return logdet_H_block, logdet_Q


def beta_laplace_structural(
    formula, data, y, method, model, W,
    n_starts=3, max_iter=200, inner_max_iter=50,
    inner_tol=1e-6, backend: Backend = None, eval_param=None, use_gpu=False,
    start_theta=None, seed=None, compute_hessian=True, hess_eps=1e-4,
    eval_many=None, eval_b0=None, verbose=0, eval_grad=None,
    use_analytic_jac=True, ftol=1e-12, gtol=1e-6, config=None,
    compute_posterior=False,
):
    """Full Laplace marginal likelihood engine.

    Outer: L-BFGS-B optimization over (beta, logphi, log_sigma, raw_rho/phi).
    Inner: Newton/BFGS mode solver for random effects b.

    log L(theta) approx -l(eta_hat) - 0.5 b_hat' Q b_hat
                          - 0.5 logdet(Q) - 0.5 logdet(H)

    If eval_param (full parameter vector) is provided, skip optimization and
    return a dict with objective/log_lik at that parameter (debugging).

    Args:
        formula: Formula (unused; X from data columns).
        data: pandas DataFrame with internal columns.
        y: Response vector (n,).
        method: "ML" or "Laplace" (always Laplace here).
        model: Model ID string.
        W: Spatial weight matrix.
        n_starts: Number of random restarts.
        max_iter: Max outer iterations.
        inner_max_iter: Max inner Newton iterations.
        inner_tol: Inner convergence tolerance.
        backend: Array backend.
        eval_param: Optional fixed parameter vector (evaluate only, no fit).

    Returns:
        dict with estimates, SEs, logLik, AIC, BIC, convergence info.
    """
    from stmmlib.beta.config import BetaFitConfig
    cfg = config if config is not None else BetaFitConfig()
    if compute_posterior and not cfg.compute_posterior:
        from dataclasses import replace as _replace
        cfg = _replace(cfg, compute_posterior=True)
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu="auto")

    # The Laplace numerical pipeline is numpy/scipy-based (solves, block_diag,
    # slogdet). The current cupy backend only wraps a few helpers and does not
    # move the heavy math to GPU; running it here mixes cupy arrays into scipy
    # and breaks. Force CPU numeric backend so fitting is stable and fast; real
    # GPU acceleration is a separate effort (torch-based inner solves).
    if backend.device == "gpu":
        from stmmlib._backend import Backend as _B
        backend = _B(gpu=False)

    # Optional torch-GPU acceleration of the blocked inner hot loops. When
    # requested and CUDA is available AND a torch has been built, dispatch to
    # the batched solvers below; otherwise stay on the numpy path.
    use_gpu = bool(use_gpu)
    if use_gpu:
        try:
            from stmmlib.beta.gpu_solvers import gpu_available
            if not gpu_available():
                use_gpu = False
        except Exception:
            use_gpu = False

    xp = backend.xp

    # Build design
    from stmmlib.beta.design import stmm_beta_make_design
    des = stmm_beta_make_design(data, model, W, backend, phi_bound=cfg.phi_bound)
    Z = des["Z"]
    blocks = des["blocks"]
    n = des["n"]
    m = des["m"]
    T = des["T"]

    # Fixed-effects design X
    # Exclude grouping cols (area/time/prov/kab + internal .stmm_*) and y.
    skip_x = {".stmm_area", ".stmm_time", ".stmm_prov", ".stmm_kab",
              "y", "area", "time", "prov", "kab"}
    x_cols = [c for c in data.columns if c not in skip_x]
    if x_cols:
        X = np.column_stack([np.ones(n)] + [data[c].values.astype(np.float64) for c in x_cols])
    else:
        X = np.ones((n, 1), dtype=np.float64)
    p = X.shape[1]
    des["x_cols"] = x_cols

    # Standardize continuous predictors (lme4/glmmTMB convergence practice):
    # column 0 is the intercept (left at 1); each predictor (j>=1) is centered
    # and scaled to unit sd so the optimizer sees O(1) covariates. The fitted
    # beta_hat is on the STANDARDIZED scale; original-scale coefficients are
    # recovered later (report/result) using des["x_scale"]. Zero-variance
    # columns are left unscaled (scale=1) to avoid division by zero.
    x_scale = {}
    if x_cols:
        for j, c in enumerate(x_cols):
            col = X[:, j + 1]
            sd = float(col.std())
            if sd > 1e-12:
                mn = float(col.mean())
                X[:, j + 1] = (col - mn) / sd
                x_scale[c] = (mn, sd)
            else:
                x_scale[c] = (0.0, 1.0)
    des["x_scale"] = x_scale

    # Structure
    from stmmlib.beta.structure import stmm_beta_structure, stmm_beta_nraw_interaction
    s = stmm_beta_structure(model)
    nr = stmm_beta_nraw_interaction(s["interaction"])

    # Block sizes
    is_nested = "prov" in blocks
    area_block = blocks.get("area", [])
    time_block = blocks.get("area_time", blocks.get("time", []))
    gamma_block = blocks.get("interaction", [])

    n_area = len(area_block)
    n_time = len(time_block)
    n_gamma = len(gamma_block)
    n_prov = len(blocks.get("prov", []))
    n_kab = len(blocks.get("kab", []))
    n_cell = len(blocks.get("cell", []))

    from stmmlib.beta.precision import rho_bound as _rho_bound, ar1_precision
    rb = _rho_bound(W, backend, margin=cfg.phi_bound) if W is not None else cfg.phi_bound
    des["rho_bound"] = rb  # actual SAR spectral bound, for display (VarCorr)
    des["phi_bound"] = cfg.phi_bound

    # Position of sar rho (index 0 when present) and ar1 phi (after rho)
    # seblup_rao_yu = SAR area (rho) + per-area AR(1) time; treat as "sar" for rho.
    # nested_sar is NOT spatial in the beta path: hierarchy is IID (prov/kab/cell).
    _has_sar = s["area"] in ("sar", "seblup_rao_yu")
    _has_ar1 = s["time"] == "ar1"
    _rho_i = 0 if _has_sar else None
    _ph_i = (1 if _has_sar else 0) if _has_ar1 else None

    # Variance components: flat = [area, time]; nested = [prov, kab] (+eta if ar1)
    ksd = 2 + (1 if (is_nested and _has_ar1) else 0)
    # Interaction adds one variance component (log_sigma_gamma)
    if n_gamma > 0:
        ksd += 1

    def _build_Q(logs, raw_dep):
        """Build full precision matrix from parameters."""
        blocks_Q = []

        if is_nested:
            # Nested hierarchy: [prov, kab, cell]; cell time shares kab variance
            # for iid time (B9) and uses its own eta variance for AR(1) time.
            blocks_Q.append(np.exp(-2.0 * logs[0]) * np.eye(n_prov, dtype=np.float64))
            blocks_Q.append(np.exp(-2.0 * logs[1]) * np.eye(n_kab, dtype=np.float64))
            if _has_ar1:
                phi_t = cfg.phi_bound * np.tanh(raw_dep[_ph_i]) if q_dep > _ph_i else 0.0
                from scipy.sparse import block_diag as _sp_blkdiag
                TQ = ar1_precision(T, logs[2], phi_t, backend, phi_bound=cfg.phi_bound)
                if backend.is_sparse(TQ):
                    TQ = TQ.toarray()
                blocks_Q.append(_block_diag_dense([TQ] * n_kab))
            else:
                blocks_Q.append(np.exp(-2.0 * logs[1]) * np.eye(n_cell, dtype=np.float64))
        else:
            # Area
            if s["area"] in ("iid",):
                Q_a = np.exp(-2.0 * logs[0]) * np.eye(n_area, dtype=np.float64)
            elif _has_sar:
                rho = rb * np.tanh(raw_dep[_rho_i]) if q_dep > _rho_i else 0.0
                W_arr = W.toarray() if hasattr(W, 'toarray') else np.asarray(W, dtype=np.float64)
                A = np.eye(n_area, dtype=np.float64) - rho * W_arr
                Q_a = np.exp(-2.0 * logs[0]) * (A.T @ A)
            else:
                raise ValueError(f"Unknown area type: {s['area']!r}")
            blocks_Q.append(Q_a)

            # Time
            if _has_ar1:
                phi_t = cfg.phi_bound * np.tanh(raw_dep[_ph_i]) if q_dep > _ph_i else 0.0
                ls_t = logs[1] if len(logs) > 1 else 0.0
                if _use_blocked:
                    # area_specific ar1: block-diag per area (m blok T x T), no cross-area coupling
                    TQ = ar1_precision(T, ls_t, phi_t, backend, phi_bound=cfg.phi_bound)
                    if backend.is_sparse(TQ):
                        TQ = TQ.toarray()
                    Q_t = _block_diag_dense([TQ] * n_area)
                else:
                    Q_t = ar1_precision(n_time, ls_t, phi_t, backend, phi_bound=cfg.phi_bound)
                    if backend.is_sparse(Q_t):
                        Q_t = Q_t.toarray()
            else:
                Q_t = np.exp(-2.0 * logs[1]) * np.eye(n_time, dtype=np.float64) if len(logs) > 1 else np.eye(n_time, dtype=np.float64)
            blocks_Q.append(Q_t)

        # Interaction precision block (shared helper; matches Gaussian V_gamma)
        Q_g = beta_Q_gamma(s, des, m, T, n_gamma, logs, raw_dep, ksd, q_dep,
                           _has_sar, _has_ar1, backend) if n_gamma > 0 else None

        return blocks_Q, Q_g

    # Isolated per-call state: fresh dict each beta_laplace_structural call,
    # no module-level mutable shared across calls (static check: only GRAD_TOL
    # lives at module level and is read-only). Refit (penalized) gets its own
    # _state, so it cannot mutate the main fit b/mu.
    _state = {"b": np.zeros(Z.shape[1], dtype=np.float64)}
    _eval_count = {"n": 0, "fb": 0}
    # Reset entry path normal ke nol deterministik. Eval hooks
    # (eval_param/eval_many/eval_grad) atur _state sendiri di bawah.
    if eval_param is None and eval_many is None and eval_grad is None:
        _state["b"] = np.zeros(Z.shape[1], dtype=np.float64)
        _state.pop("cache", None)

    # Resolve GPU-accelerated solver aliases once (torch), else numpy reference.
    if use_gpu:
        from stmmlib.beta.gpu_solvers import (
            beta_mode_newton_blocked_gpu as _nb_gpu,
            laplace_block_logdets_gpu as _logdets_gpu,
        )
        _mode_blocked = lambda *a, **k: beta_solve_mode_blocked(*a, **k, _inner=_nb_gpu)
        _logdet_fn = lambda *a, **k: _logdets_gpu(*a, **k)
    else:
        _mode_blocked = beta_solve_mode_blocked
        _logdet_fn = laplace_block_logdets

    # Blocked (Schur) inner solve only for area-specific non-interaction models,
    # where Z = [Z_area, Z_cell] and Q = block_diag(Q_area, Q_cell per area).
    _use_blocked = (not is_nested and s["interaction"] == "none"
                    and s["time_scope"] == "area_specific")
    # Q_c is block-per-area for both iid (diagonal) and ar1 (per-area T x T) time,
    # so the exact per-area Schur logdet is always valid for blocked models.
    _schur_inner = _use_blocked

    # Analytic gradient: blocked area-specific non-interaction models
    # (m1-m3, m5; nested/interactions keep finite-diff jac for now). Covers
    # [beta, logphi, logs, raw_dep(sar rho, ar1 phi)] via the envelope
    # theorem + mode-shift correction at the inner mode.
    _jac_ok = bool(use_analytic_jac and (not use_gpu) and (not is_nested)
                   and _use_blocked)

    # STABILITY GATE (measured): on real-scale panels the analytic-jac path
    # MUST use a truncated inner solve (<=15 iters). A full inner_max_iter=50
    # solve jumps inner modes at/near the outer start (it=1 false convergence,
    # F garbage), while inner=15 stays on a smooth warm-started landscape and
    # climbs (M1 real: nit=3, ll 3741 and rising vs it=1, ll=-29). Truncation
    # is a stabilizer, not a shortcut: the gradient's own deep solve
    # (max(200, inner*4), tol 1e-8 below) is untouched. FD-jac/GPU/nested
    # paths keep their inner budget (baseline GPU runs healthy with 50).
    if _jac_ok:
        inner_max_iter = min(inner_max_iter, cfg.inner_probe_cap)

    def _neg_marginal_l(param_vec):
        """Negative Laplace marginal log-likelihood."""
        _eval_count["n"] += 1
        beta = param_vec[:p]
        logphi = param_vec[p]
        logs = param_vec[p + 1: p + 1 + ksd]
        raw_dep = param_vec[p + 1 + ksd:]

        phi = np.exp(logphi)

        try:
            blocks_Q, Q_g = _build_Q(logs, raw_dep)
        except Exception as _e:
            if _os_env_debug():
                import traceback as _tb
                _tb.print_exc()
            return 1e20

        # Assemble full Q (single dense allocation, no sparse roundtrip)
        Q_base = _block_diag_dense(blocks_Q)
        if Q_g is not None:
            Q_full = _block_diag_dense([Q_base, Q_g]) if Q_g.size > 0 else Q_base
        else:
            Q_full = Q_base

        # Ensure Q_full matches Z columns
        q = Z.shape[1]
        if Q_full.shape[0] < q:
            pad = q - Q_full.shape[0]
            Q_full = np.block([
                [Q_full, np.zeros((Q_full.shape[0], pad))],
                [np.zeros((pad, Q_full.shape[0])), np.eye(pad, dtype=np.float64) * 1e-10]
            ])
        elif Q_full.shape[0] > q:
            Q_full = Q_full[:q, :q]

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
            logdet_Q = 0.0
            logdet_H = 0.0

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

    # Parameter vector layout: [beta, logphi, logs_ksd, raw_dep]
    # raw_dep = [sar_rho (if area sar), ar1_phi (if time ar1),
    #            rho_gamma (if inter spatial/st), phi_gamma (if inter temporal/st)]
    q_dep = (1 if _has_sar else 0) \
        + (1 if _has_ar1 else 0) \
        + nr
    param_dim = p + 1 + ksd + q_dep

    # Bounds
    bounds = [tuple(cfg.beta_bounds)] * p  # beta
    bounds.append(tuple(cfg.logphi_bounds))  # logphi
    # log_sigma upper cap (3 -> sigma < e^3 ~ 20, logit scale). Large sigma
    # collapses Q toward 0 (weak penalty), letting the inner mode wander and
    # the Laplace marginal fall into a degenerate well (huge negative logLik +
    # slow inner solve). Capping keeps the objective stable across models.
    bounds.extend([tuple(cfg.logs_bounds)] * ksd)  # logs
    bounds.extend([(-np.arctanh(cfg.dep_margin), np.arctanh(cfg.dep_margin))] * q_dep)  # raw dep

    # Starting values (match R: logit-scale OLS for beta, log sds on logit scale)
    def _get_start():
        yc = np.clip(y, 1e-6, 1 - 1e-6)
        ylog = np.log(yc / (1 - yc))
        beta_ols = np.linalg.lstsq(X, ylog, rcond=None)[0]
        logphi0 = np.log(20.0)
        _sd = float(np.std(ylog))
        if n_gamma > 0:
            # Model interaksi q~=n: sigma penuh menaruh Laplace di wilayah kasar
            # (grad FD 1e5, abort it=0). Bagi budget varians per komponen.
            logs0 = np.full(ksd, np.log(max(_sd / max(ksd, 1), 0.05)),
                            dtype=np.float64)
        else:
            logs0 = np.full(ksd, np.log(max(_sd, 0.2)), dtype=np.float64)
        raw0 = np.zeros(q_dep, dtype=np.float64)
        return np.concatenate([beta_ols, [logphi0], logs0, raw0])

    def _laplace_grad(param_vec):
        """Analytic gradient of the negative Laplace marginal log-likelihood."""
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
            blocks_q0, Q_g0 = _build_Q(logs, raw_dep)
            Q_q0 = _block_diag_dense(blocks_q0)
            if Q_g0 is not None and Q_g0.size > 0:
                Q_q0 = _block_diag_dense([Q_q0, Q_g0])
            b, conv, _, _ = beta_mode_newton_blocked(
                X, Z, Q_q0, m, T, beta, logphi, y, max(200, inner_max_iter * 4),
                start_b=_state["b"], inner_tol=min(inner_tol, 1e-8))

        eta = X @ beta + Z @ b
        w = beta_nll_curvature(eta, y, phi)
        s_pos = -_beta_nll_score(eta, y, phi)          # d l / d eta
        pieces = _beta_grad_pieces(eta, y, phi)

        blocks_Q, Q_g = _build_Q(logs, raw_dep)
        Q_full = _block_diag_dense(blocks_Q)
        if Q_g is not None and Q_g.size > 0:
            Q_full = _block_diag_dense([Q_full, Q_g])
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

    # Debug hooks: evaluate objective at fixed parameter vector(s) WITHOUT
    # optimizing. Must run before the starts loop (eval_param/eval_many are
    # eval-only contracts; running the optimizer first wastes a full fit).
    if eval_grad is not None:
        if eval_b0 is not None:
            _state["b"] = np.asarray(eval_b0, dtype=np.float64).copy()
        return {"grad": _laplace_grad(np.asarray(eval_grad, dtype=np.float64))}

    if eval_many is not None:
        eval_many = np.atleast_2d(np.asarray(eval_many, dtype=np.float64))
        _b0 = (np.zeros(Z.shape[1], dtype=np.float64) if eval_b0 is None
               else np.asarray(eval_b0, dtype=np.float64))
        vals = []
        for row in eval_many:
            _state["b"] = _b0.copy()
            vals.append(float(_neg_marginal_l(np.asarray(row, float))))
        return {"eval_many": np.asarray(vals)}

    if eval_param is not None:
        eval_param = np.asarray(eval_param, dtype=np.float64)
        _state["b"] = (np.zeros(Z.shape[1], dtype=np.float64) if eval_b0 is None
                       else np.asarray(eval_b0, dtype=np.float64))
        val = _neg_marginal_l(eval_param)
        b_debug = _state["b"].copy()
        import os as _os
        if _os.environ.get("STMM_DEBUG"):
            _gs = Z.T @ _beta_nll_score(X @ eval_param[:p] + Z @ b_debug,
                                        y, float(np.exp(eval_param[p]))) \
                + _block_diag_dense(_build_Q(eval_param[p + 1: p + 1 + ksd],
                                             eval_param[p + 1 + ksd:])[0]) @ b_debug
            print(f"    [hook] b_debug resid={np.max(np.abs(_gs)):.3e}", flush=True)
        hessian = None
        lambda_min = None
        hessian_pd = None
        if compute_hessian:
            _state["hessian_probe"] = True
            try:
                hessian = _num_hessian(_neg_marginal_l, eval_param, eps=hess_eps)
                lam = np.linalg.eigvalsh(hessian)
                lambda_min = float(lam.min())
                hessian_pd = bool(lambda_min > 1e-6)
            except KeyboardInterrupt:
                print("interrupted: Hessian dibatalkan user; SE tidak tersedia")
                hessian, lambda_min, hessian_pd = None, None, None
            except Exception:
                hessian, lambda_min, hessian_pd = None, None, None
            finally:
                _state.pop("hessian_probe", None)
        return {"eval_param": np.array(eval_param, dtype=np.float64, copy=True),
                "neg_log_lik": val,
                "log_lik": -val, "b_hat": b_debug, "X": X, "Z": Z,
                "des": des, "model": model, "y": y, "p": p, "n": n, "W": W,
                "hessian": hessian, "lambda_min": lambda_min, "hessian_pd": hessian_pd}

    best_val = np.inf
    best_param = None
    best_meta = {}

    # Warm start (F2): caller supplies a full parameter vector (from a previous
    # cheaper/warm-start model map), bypassing OLS + random perturbed starts.
    if start_theta is not None:
        starts = [np.asarray(start_theta, dtype=np.float64)]
    else:
        # Perturb random restarts; seeded for reproducibility (F3).
        rng = np.random.default_rng(seed) if seed is not None else np.random
        starts = [_get_start()]
        for _ in range(max(n_starts - 1, 0)):
            s0 = starts[0].copy()
            s0[p] += rng.normal(0, 0.5)  # perturb logphi
            s0[p + 1: p + 1 + ksd] += rng.normal(0, 0.15, ksd)
            s0[:p] += rng.normal(0, 0.05, p)  # perturb beta (explore basins)
            if q_dep > 0:
                s0[p + 1 + ksd:] += rng.normal(0, 0.15, q_dep)  # perturb raw dep
            starts.append(s0)

    failed_starts = []
    _interrupted = False
    _abort_val = np.inf
    _abort_param = None
    _abort_meta = None
    try:
        for i, param0 in enumerate(starts):
            param0 = np.asarray(param0, dtype=np.float64)
            if param0.shape != (param_dim,) or not np.all(np.isfinite(param0)):
                failed_starts.append(f"start {i} ditolak: non-finite")
                continue
            _state["b"] = np.zeros(Z.shape[1], dtype=np.float64)
            _state.pop("cache", None)
            try:
                v0 = _neg_marginal_l(param0)
            except Exception as e:  # noqa: BLE001
                failed_starts.append(f"start {i} ditolak: {e}")
                continue
            if not np.isfinite(v0) or v0 >= 1e19:
                failed_starts.append(f"start {i} ditolak: objective non-finite ({v0})")
                continue
            # Pre-check gradien di titik start: tolak start kasar sebelum optimasi.
            try:
                if _jac_ok:
                    _gn_pre = float(np.linalg.norm(_laplace_grad(param0)))
                else:
                    _eps_pre = 1e-6
                    _gp = []
                    for _k in range(param_dim):
                        _dp = param0.copy(); _dm = param0.copy()
                        _dp[_k] += _eps_pre; _dm[_k] -= _eps_pre
                        _vp = _neg_marginal_l(_dp)
                        _vm = _neg_marginal_l(_dm)
                        if (not np.isfinite(_vp) or not np.isfinite(_vm)
                                or _vp >= 1e19 or _vm >= 1e19):
                            raise FloatingPointError("FD start non-finite")
                        _gp.append((_vp - _vm) / (2.0 * _eps_pre))
                    _gn_pre = float(np.linalg.norm(np.asarray(_gp, dtype=np.float64)))
            except Exception:
                _gn_pre = float("inf")
            if not np.isfinite(_gn_pre) or _gn_pre > 1e6:
                failed_starts.append(
                    f"start {i} ditolak: grad start non-finite/ledak ({_gn_pre:.3e})")
                continue
            _state["b"] = np.zeros(Z.shape[1], dtype=np.float64)
            _state.pop("cache", None)
            try:
                # maxfun guards total F,G evals (default 15000). Stiff log-scale
                # params eat many evals per line search; cap it generously so the
                # run is limited by max_iter, not an early eval cutoff.
                # NOTE: gradient stop is `gtol`, not `pgtol` -- pgtol is not a valid
                # L-BFGS-B option for scipy.minimize and was silently ignored before.
                res = minimize(_neg_marginal_l, param0, method="L-BFGS-B",
                               jac=_laplace_grad if _jac_ok else None,
                               bounds=bounds,
                               options={"maxiter": max_iter,
                                        "maxfun": max(200_000, max_iter * 8),
                                        "ftol": ftol, "gtol": gtol})
                if not np.isfinite(res.fun):
                    failed_starts.append(f"start {i} ditolak: objective non-finite")
                    continue
                _jac = getattr(res, "jac", None)
                try:
                    _gn0 = float(np.linalg.norm(_jac)) if _jac is not None else np.inf
                except Exception:
                    _gn0 = np.inf
                try:
                    _gvec0 = (np.asarray(_jac, dtype=np.float64).ravel().copy()
                              if _jac is not None else None)
                    if _gvec0 is not None and not np.all(np.isfinite(_gvec0)):
                        pass
                except Exception:
                    _gvec0 = None
                _nit0 = int(getattr(res, "nit", -1))
                _msg0 = str(getattr(res, "message", ""))
                if _nit0 == 0 and ((not np.isfinite(_gn0)) or _gn0 > 1e6
                                   or "abnormal" in _msg0.lower()):
                    failed_starts.append(
                        f"start {i} ditolak: abort iterasi 0 "
                        f"({_msg0.strip()} grad {_gn0:.3e})")
                    if float(res.fun) < _abort_val:
                        _abort_val = float(res.fun)
                        _abort_param = res.x.copy()
                        _abort_meta = {
                            "success": False,
                            "nit": _nit0,
                            "n_evals": int(getattr(res, "nfev", -1)),
                            "fun": float(res.fun),
                            "start": int(i),
                            "message": _msg0.strip(),
                            "grad_norm": _gn0,
                            "grad_vec": _gvec0,
                            "ftol": float(ftol),
                            "gtol": float(gtol),
                        }
                    continue
                if res.fun < best_val:
                    best_val = float(res.fun)
                    best_param = res.x
                    best_meta = {
                        "success": bool(getattr(res, "success", False)),
                        "nit": _nit0,
                        "n_evals": int(getattr(res, "nfev", -1)),
                        "fun": float(res.fun),
                        "start": int(i),
                        "message": _msg0.strip(),
                        "grad_norm": _gn0,
                        "grad_vec": _gvec0,
                        "ftol": float(ftol),
                        "gtol": float(gtol),
                    }
            except KeyboardInterrupt:
                print("interrupted: start dihentikan user, pakai titik terbaik sejauh ini")
                _interrupted = True
                break
            except Exception as e:  # noqa: BLE001
                failed_starts.append(f"start {i} gagal: {e}")
                continue
    except KeyboardInterrupt:
        print("interrupted: start dihentikan user, pakai titik terbaik sejauh ini")
        _interrupted = True

    # Semua start abort iterasi 0 dgn fun finite: pakai titik abort terbaik sbg
    # hasil INVALID (angka sama dgn perilaku lama, status jujur bukan max_iter).
    _from_abort = False
    if best_param is None and _abort_param is not None:
        best_param, best_val, best_meta = _abort_param, _abort_val, _abort_meta
        _from_abort = True

    if best_param is None:
        msg = ("semua start gagal/non-finite objective; tidak ada hasil optimasi. "
               + (" | ".join(failed_starts) if failed_starts else "tidak ada start valid."))
        nan_param = np.full(param_dim, np.nan, dtype=np.float64)
        nan_hess = np.full((param_dim, param_dim), np.nan, dtype=np.float64)
        return {
            "status": "invalid",
            "at_bound": False,
            "certificate": {
                "valid": False,
                "optimum_valid": False,
                "grad_norm": None,
                "projected_grad": None,
                "at_bound": False,
                "bound_params": [],
                "hessian_pd": None,
                "lambda_min": None,
                "ll_per_obs": None,
                "warnings": [msg],
            },
            "beta_hat": np.array(nan_param[:p], dtype=np.float64, copy=True),
            "logphi": float("nan"),
            "phi": float("nan"),
            "logs": np.array(nan_param[p + 1: p + 1 + ksd], dtype=np.float64, copy=True),
            "raw_dep": np.array(nan_param[p + 1 + ksd:], dtype=np.float64, copy=True),
            "b_hat": np.zeros(Z.shape[1], dtype=np.float64),
            "H_b_chol": None,
            "sigma_area": float("nan"),
            "sigma_time": float("nan"),
            "sigma_gamma": None,
            "ksd": ksd,
            "q_dep": q_dep,
            "log_lik": float("nan"),
            "AIC": float("nan"),
            "BIC": float("nan"),
            "k": param_dim,
            "n": n,
            "p": p,
            "converged": False,
            "degenerate": False,
            "optim": {
                "success": False,
                "nit": 0,
                "iters": 0,
                "n_evals": 0,
                "max_iter": int(max_iter),
                "fun": float("nan"),
                "start": -1,
                "n_starts": len(starts),
                "n_failed_starts": len(failed_starts),
                "failed_starts": list(failed_starts),
                "n_evals_total": _eval_count["n"],
                "n_dense_fallbacks": _eval_count["fb"],
                "jac": "analytic" if _jac_ok else "finite-diff",
                "message": msg,
                "grad_norm": float("nan"),
                "ftol": float(ftol),
                "gtol": float(gtol),
                "hessian": nan_hess,
                "lambda_min": float("nan"),
                "hessian_pd": None,
                "hessian_timeout": False,
                "interrupted": bool(_interrupted),
                "all_starts_abort": False,
            },
            "W": W,
            "X": X,
            "Z": Z,
            "des": des,
            "model": model,
            "y": y,
            "_boot_frame": data,
            "_boot_theta": np.array(nan_param, dtype=np.float64, copy=True),
        }

    # Extract results (copy: putus view ke _boot_theta agar refit tak sentuh utama)
    beta_hat = np.array(best_param[:p], dtype=np.float64, copy=True)
    logphi_hat = best_param[p]
    logs_hat = np.array(best_param[p + 1: p + 1 + ksd], dtype=np.float64, copy=True)
    raw_dep_hat = np.array(best_param[p + 1 + ksd:], dtype=np.float64, copy=True)

    phi_hat = np.exp(logphi_hat)
    sigma_area = np.exp(logs_hat[0]) if ksd > 0 else 1.0
    sigma_time = np.exp(logs_hat[1]) if ksd > 1 else 1.0
    sigma_gamma = np.exp(logs_hat[2]) if (n_gamma > 0 and ksd > 2) else None

    log_lik = -best_val
    k_total = param_dim
    aic = -2 * log_lik + 2 * k_total
    bic = -2 * log_lik + np.log(n) * k_total

    # Recompute inner mode at best param so `_state["b"]` is the conditional mode
    # for the final estimates (used for predictions: eta = X beta + Z b_hat).
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
            _blocks_Q, _Q_g = _build_Q(logs_hat, raw_dep_hat)
            _Qf = _block_diag_dense(_blocks_Q)
            if _Q_g is not None and _Q_g.size > 0:
                _Qf = _block_diag_dense([_Qf, _Q_g])
            # Same column padding as the objective's Q_full assembly.
            _qz = Z.shape[1]
            if _Qf.shape[0] < _qz:
                _pad = _qz - _Qf.shape[0]
                _Qf = np.block([
                    [_Qf, np.zeros((_Qf.shape[0], _pad))],
                    [np.zeros((_pad, _Qf.shape[0])),
                     np.eye(_pad, dtype=np.float64) * 1e-10],
                ])
            elif _Qf.shape[0] > _qz:
                _Qf = _Qf[:_qz, :_qz]
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
    # success=True and grad ~1e3-1e5 (measured M1/M2: it=1, grad 3745/2.1e5).
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
    hessian = None
    lambda_min = None
    hessian_pd = None
    _hess_timeout = False
    if compute_hessian:
        import time as _time
        _hess_t0 = _time.monotonic()
        _budget = cfg.hessian_time_budget
        _orig_neg = _neg_marginal_l

        def _hess_obj(v):
            if _budget is not None and _time.monotonic() - _hess_t0 > _budget:
                raise TimeoutError("hessian budget")
            return _orig_neg(v)

        def _hess_grad(v):
            # Cheap Hessian path (analytic-jac models): refresh the inner mode
            # and cache at the perturbed point through the same mechanism as
            # _neg_marginal_l, then evaluate the existing analytic gradient of
            # the negative objective. Same time budget as the objective path.
            va = np.asarray(v, dtype=np.float64)
            if _budget is not None and _time.monotonic() - _hess_t0 > _budget:
                raise TimeoutError("hessian budget")
            _orig_neg(va)
            return _laplace_grad(va)

        _state["hessian_probe"] = True
        try:
            if _jac_ok:
                hessian = _num_hessian_grad(_hess_grad, best_param, eps=hess_eps)
            else:
                hessian = _num_hessian(_hess_obj, best_param, eps=hess_eps)
            lam = np.linalg.eigvalsh(hessian)
            lambda_min = float(lam.min())
            hessian_pd = bool(lambda_min > 1e-6)
        except KeyboardInterrupt:
            print("interrupted: Hessian dibatalkan user; SE tidak tersedia")
            hessian, lambda_min, hessian_pd = None, None, None
            _interrupted = True
        except TimeoutError:
            hessian, lambda_min, hessian_pd = None, None, None
            _hess_timeout = True
        except Exception:  # noqa: BLE001
            hessian, lambda_min, hessian_pd = None, None, None
        finally:
            _state.pop("hessian_probe", None)
            # Restore final mode: FD Hessian probes mutate _state b and cache
            # across perturbed points; put back the best-param mode so the
            # returned b_hat always comes from this call own state.
            _state["b"] = b_hat.copy()
            _state["cache"] = {"x": best_param.copy(), "b": b_hat.copy()}

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


def beta_Q_gamma(s, des, m, T, n_gamma, logs, raw_dep, ksd, q_dep,
                 has_sar, has_ar1, backend):
    """Interaction precision block Q_g = G_g^{-1} (shared solver + inference).

    Matches the Gaussian marginal covariance V_gamma = Z_gamma G_gamma Z_gamma':
        iid:      Q = s2p I
        spatial:  Q = s2p kron(I_T, qa^{-1}),  qa = B_area'A'A B_area
        temporal: Q = s2p kron(I_m, qt^{-1}),  qt = B_time' Ar B_time
        st:       Q = s2p kron(qa^{-1}, qt^{-1})
    """
    from scipy.sparse import kron as _sp_kron, identity as _sp_ident
    # logs has exactly ksd entries (area, time, [gamma]); gamma log-sd at ksd-1
    if n_gamma <= 0 or len(logs) < ksd:
        return None
    s2p = np.exp(-2.0 * logs[ksd - 1])
    itype = s["interaction"]
    B_area = des["B_area"]
    B_time = des["B_time"]
    rb_g = des.get("rho_bound_g", 0.95)
    _rg_i = (1 if has_sar else 0) + (1 if has_ar1 else 0)
    # raw_dep order: [rho(sar), phi(ar1), rho_gamma(spatial|st), phi_gamma(temporal|st)]
    _phg_i = _rg_i + (1 if itype in ("spatial", "st") else 0)
    qa_inv = qt_inv = None
    if itype in ("spatial", "st"):
        rho_g = rb_g * np.tanh(raw_dep[_rg_i]) if q_dep > _rg_i else 0.0
        Wg = des.get("W_gamma")
        if Wg is None:
            Wg = np.eye(m, dtype=np.float64)
        elif hasattr(Wg, "toarray"):
            Wg = Wg.toarray()
        Ag = np.eye(m, dtype=np.float64) - rho_g * np.asarray(Wg, dtype=np.float64)
        qa = B_area.T @ (Ag.T @ Ag) @ B_area
        qa_inv = np.linalg.inv(qa)
    if itype in ("temporal", "st"):
        from stmmlib.beta.precision import ar1_precision as _ar1_prec
        _pb = des.get("phi_bound", 0.95)
        phi_g = _pb * np.tanh(raw_dep[_phg_i]) if q_dep > _phg_i else 0.0
        Ap = _ar1_prec(T, 0.0, np.arctanh(np.clip(phi_g / _pb, -0.999, 0.999)), backend,
                       phi_bound=_pb)
        if hasattr(Ap, "toarray"):
            Ap = Ap.toarray()
        Ar = np.linalg.inv(np.asarray(Ap, dtype=np.float64))
        qt = B_time.T @ Ar @ B_time
        qt_inv = np.linalg.inv(qt)
    if itype == "iid":
        return s2p * np.eye(n_gamma, dtype=np.float64)
    if itype == "spatial":
        return np.asarray((s2p * _sp_kron(_sp_ident(T), qa_inv, format="csr")).toarray())
    if itype == "temporal":
        return np.asarray((s2p * _sp_kron(_sp_ident(m), qt_inv, format="csr")).toarray())
    if itype == "st":
        return np.asarray((s2p * _sp_kron(qa_inv, qt_inv, format="csr")).toarray())
    return None


def beta_regression_base(formula, data, y, method="ML", model="beta_regression"):
    """Fixed-effects Beta ML (beta_regression, no random effects): L-BFGS-B
    optim + Hessian SEs.

    X is built from ALL non-grouping columns of ``data`` (intercept first),
    so multi-predictor beta regression works; a data frame with only the
    response yields the legacy intercept-only fit. Standardization is NOT
    applied here (plain ML regression path).
    """
    n = len(y)
    _skip = {".stmm_area", ".stmm_time", ".stmm_prov", ".stmm_kab",
             "y", "area", "time", "prov", "kab"}
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
