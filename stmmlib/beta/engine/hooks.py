"""Evaluation hooks (fixed-parameter objective, gradient, Hessian)."""

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


def run_hooks(ctx, eval_param=None, eval_many=None, eval_grad=None, eval_b0=None, compute_hessian=True, hess_eps=1e-4):
    W = ctx["W"]
    X = ctx["X"]
    Z = ctx["Z"]
    _build_Q = ctx["_build_Q"]
    _jac_ok = ctx["_jac_ok"]
    _laplace_grad = ctx["_laplace_grad"]
    _neg_marginal_l = ctx["_neg_marginal_l"]
    _state = ctx["_state"]
    des = ctx["des"]
    ksd = ctx["ksd"]
    model = ctx["model"]
    n = ctx["n"]
    p = ctx["p"]
    y = ctx["y"]
    # Debug hooks: evaluate objective at fixed parameter vector(s) WITHOUT
    # optimizing. Must run before the starts loop (eval_param/eval_many are
    # eval-only contracts; running the optimizer first wastes a full fit).
    if eval_grad is not None:
        pv = np.asarray(eval_grad, dtype=np.float64)
        if _jac_ok:
            if eval_b0 is not None:
                _state["b"] = np.asarray(eval_b0, dtype=np.float64).copy()
            return {"grad": _laplace_grad(pv), "grad_kind": "analytic"}
        # Non-blocked / nested / interaction models have no analytic gradient.
        # Fall back to central finite differences of the same objective instead
        # of calling the blocked-only path (it used to crash on M4+).
        _b0 = (np.zeros(Z.shape[1], dtype=np.float64) if eval_b0 is None
               else np.asarray(eval_b0, dtype=np.float64))
        eps = 1e-6
        fd = np.empty_like(pv)
        for i in range(len(pv)):
            vals = []
            for sign in (1.0, -1.0):
                row = pv.copy()
                row[i] += sign * eps
                _state["b"] = _b0.copy()
                _state.pop("cache", None)
                vals.append(float(_neg_marginal_l(row)))
            fd[i] = (vals[0] - vals[1]) / (2.0 * eps)
        return {"grad": fd, "grad_kind": "finite-diff"}

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
        # Stationarity of the returned mode. Non-zero means the inner solve did
        # not reach a stationary point, so the Laplace value describes a point
        # that is not the conditional mode (a cold start can diverge and
        # leave a large residual). Exposed instead of silently returned.
        _Qdbg = _build_Q(eval_param[p + 1: p + 1 + ksd],
                         eval_param[p + 1 + ksd:])
        _eta_dbg = X @ eval_param[:p] + Z @ b_debug
        _gs = Z.T @ _beta_nll_score(_eta_dbg, y,
                                    float(np.exp(eval_param[p]))) \
            + _Qdbg @ b_debug
        _mode_resid = float(np.max(np.abs(_gs)))
        if _os_env_debug():
            # f = NLL + 0.5 b'Qb: what the inner solver minimizes. The mode is
            # argmin f; a stationary point with larger f is not the mode.
            from scipy.special import betaln as _betaln
            _mu = np.clip(1.0 / (1.0 + np.exp(-_eta_dbg)), 1e-15, 1.0 - 1e-15)
            _ph = float(np.exp(eval_param[p]))
            _av, _bv = _mu * _ph, (1.0 - _mu) * _ph
            _f = -np.sum(-_betaln(_av, _bv)
                         + (_av - 1.0) * np.log(np.clip(y, 1e-15, 1.0))
                         + (_bv - 1.0) * np.log(np.clip(1.0 - y, 1e-15, 1.0))) \
                + 0.5 * b_debug @ _Qdbg @ b_debug
            print(f"    [hook] b_debug resid={_mode_resid:.3e} "
                  f"max|b|={np.max(np.abs(b_debug)):.4f} f={_f:.6f}",
                  flush=True)
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
                "neg_log_lik": val, "mode_residual": _mode_resid,
                "log_lik": -val, "b_hat": b_debug, "X": X, "Z": Z,
                "des": des, "model": model, "y": y, "p": p, "n": n, "W": W,
                "hessian": hessian, "lambda_min": lambda_min, "hessian_pd": hessian_pd}
