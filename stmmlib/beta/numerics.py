"""Numerical helpers shared by the Beta fit engine.

Finite-difference Hessians, projected-gradient norm for box constraints,
debug flag, and dense block-diagonal assembly.
"""

from __future__ import annotations

import numpy as np


def _os_env_debug() -> bool:
    import os
    return os.environ.get("STMM_DEBUG", "") != ""


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
