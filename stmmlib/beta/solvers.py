"""Laplace marginal likelihood engine for Beta spatio-temporal mixed models.

The numerical helpers live in sibling modules (numerics, beta_nll, schur,
mode, laplace); this module keeps the engine and re-exports the historical
names so existing imports from `stmmlib.beta.solvers` keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from stmmlib.beta.precision import PrecisionShapeError
from stmmlib.beta.numerics import (
    _os_env_debug,
    _projected_grad_norm,
    _num_hessian,
    _num_hessian_grad,
    _block_diag_dense,
)
from stmmlib.beta.beta_nll import (
    beta_nll_curvature,
    _beta_nll_score,
    _beta_grad_pieces,
)
from stmmlib.beta.schur import _blocked_hinv_terms, _blocked_pen_terms
from stmmlib.beta.mode import (
    beta_mode_newton,
    beta_mode_newton_blocked,
    beta_solve_mode_blocked,
    beta_solve_mode,
)
from stmmlib.beta.laplace import laplace_block_logdets, beta_parameter_bounds
from stmmlib.beta.prior import beta_Q_gamma
from stmmlib.beta.regression import beta_regression_base

if TYPE_CHECKING:
    from stmmlib._backend import Backend

# Gradient-norm tolerance for a "genuine" convergence verdict. L-BFGS-B fills
# `res.jac` (finite-difference gradient) at the final point; a small norm there
# means the first-order condition holds. This is a second signal alongside
# `res.success` because scipy sometimes flags success=False on bounded problems
# even at a valid minimum. Tunable: pass GRAD_TOL=... to override the default.
GRAD_TOL = 1.0


def beta_laplace_structural(
    formula, data, y, method, model, W,
    n_starts=3, max_iter=200, inner_max_iter=50,
    inner_tol=1e-6, backend: Backend = None, eval_param=None, use_gpu=False,
    start_theta=None, start_b=None, seed=None, compute_hessian=True, hess_eps=1e-4,
    eval_many=None, eval_b0=None, verbose=0, eval_grad=None,
    use_analytic_jac=True, ftol=1e-12, gtol=1e-6, config=None,
    compute_posterior=False,
):
    """Full Laplace marginal likelihood engine.

    Outer: L-BFGS-B optimization over (beta, logphi, log_sigma, raw_rho/phi).
    Inner: Newton/BFGS mode solver for random effects b.

    log L(theta) approx -l(eta_hat) - 0.5 b_hat' Q b_hat
                          + 0.5 logdet(Q) - 0.5 logdet(H)

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
        start_b: Optional warm start for the inner mode solve. Without it the
            inner solve always starts from b = 0, which diverges on large
            panels; a restricted refit reuses the mode of the full fit.

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
    des = stmm_beta_make_design(data, model, W, backend, rho_bound=cfg.rho_bound)
    Z = des["Z"]
    blocks = des["blocks"]
    n = des["n"]
    m = des["m"]
    T = des["T"]

    # Fixed-effects design X
    # Exclude grouping cols (area/time/prov/kab + internal .stmm_*) and y.
    skip_x = {".stmm_area", ".stmm_time", ".stmm_prov", ".stmm_kab", "y"}
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
    rb = _rho_bound(W, backend, margin=cfg.rho_bound) if W is not None else cfg.rho_bound
    des["rho_bound"] = rb  # actual SAR spectral bound, for display (VarCorr)
    des["phi_bound"] = cfg.phi_bound  # AR(1) coefficient bound

    # Position of sar rho (index 0 when present) and ar1 phi (after rho)
    # seblup_rao_yu = SAR area (rho) + per-area AR(1) time; treat as "sar" for rho.
    # nested_sar is NOT spatial in the beta path: hierarchy is IID (prov/kab/cell).
    _has_sar = s["area"] in ("sar", "seblup_rao_yu")
    _has_ar1 = s["time"] == "ar1"
    _rho_i = 0 if _has_sar else None
    _ph_i = (1 if _has_sar else 0) if _has_ar1 else None

    # Variance components: flat = [area, time]; nested = [prov, kab, time]
    ksd = 3 if is_nested else 2
    # Interaction adds one variance component (log_sigma_gamma)
    if n_gamma > 0:
        ksd += 1

    from stmmlib.beta.prior import build_beta_Q

    def _build_Q(logs, raw_dep):
        """Build the full dense precision matrix from parameters."""
        return build_beta_Q(
            s, des, W, logs, raw_dep, ksd, q_dep,
            rho_bound=rb, phi_bound=cfg.phi_bound, backend=backend,
            use_blocked=_use_blocked)

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

    # STABILITY GATE: on real-scale panels the analytic-jac path
    # MUST use a truncated inner solve (<=15 iters). A full inner_max_iter=50
    # solve jumps inner modes at/near the outer start (it=1 false convergence,
    # F garbage), while inner=15 stays on a smooth warm-started landscape and
    # climbs (few iterations, ll rising) instead of it=1 with a huge negative
    # ll. Truncation
    # is a stabilizer, not a shortcut: the gradient's own deep solve
    # (max(200, inner*4), tol 1e-8 below) is untouched. FD-jac/GPU/nested
    # paths keep their inner budget (baseline GPU runs healthy with 50).
    if _jac_ok:
        inner_max_iter = min(inner_max_iter, cfg.inner_probe_cap)

    # Parameter vector layout: [beta, logphi, logs_ksd, raw_dep]
    # raw_dep = [sar_rho (if area sar), ar1_phi (if time ar1),
    #            rho_gamma (if inter spatial/st), phi_gamma (if inter temporal/st)]
    q_dep = (1 if _has_sar else 0) \
        + (1 if _has_ar1 else 0) \
        + nr
    param_dim = p + 1 + ksd + q_dep

    # Bounds
    # log_sigma upper cap (3 -> sigma < e^3 ~ 20, logit scale). Large sigma
    # collapses Q toward 0 (weak penalty), letting the inner mode wander and
    # the Laplace marginal fall into a degenerate well (huge negative logLik +
    # slow inner solve). Capping keeps the objective stable across models.
    bounds = beta_parameter_bounds(cfg, p, ksd, q_dep)

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

    from stmmlib.beta.engine import (
        build_payload as _build_payload,
        laplace_grad as _engine_grad,
        neg_marginal_l as _engine_neg,
        optimize_starts as _optimize_starts,
        run_hooks as _run_hooks,
    )

    # Every fit-scoped value the Laplace engine reads lives in one context
    # dict; the engine modules are the verbatim numerical bodies, so the
    # estimate path is unchanged by the split.
    ctx = {
        "T": T,
        "W": W,
        "X": X,
        "Z": Z,
        "_eval_count": _eval_count,
        "_get_start": _get_start,
        "_has_ar1": _has_ar1,
        "_has_sar": _has_sar,
        "_jac_ok": _jac_ok,
        "_logdet_fn": _logdet_fn,
        "_mode_blocked": _mode_blocked,
        "_schur_inner": _schur_inner,
        "_state": _state,
        "_use_blocked": _use_blocked,
        "bounds": bounds,
        "cfg": cfg,
        "compute_hessian": compute_hessian,
        "data": data,
        "des": des,
        "ftol": ftol,
        "gtol": gtol,
        "hess_eps": hess_eps,
        "inner_max_iter": inner_max_iter,
        "inner_tol": inner_tol,
        "is_nested": is_nested,
        "ksd": ksd,
        "m": m,
        "max_iter": max_iter,
        "model": model,
        "n": n,
        "n_area": n_area,
        "n_gamma": n_gamma,
        "n_starts": n_starts,
        "n_time": n_time,
        "p": p,
        "param_dim": param_dim,
        "q_dep": q_dep,
        "rb": rb,
        "seed": seed,
        "start_b": start_b,
        "start_theta": start_theta,
        "verbose": verbose,
        "x_cols": x_cols,
        "y": y,
    }
    ctx["_build_Q"] = _build_Q
    ctx["_neg_marginal_l"] = lambda v: _engine_neg(ctx, v)
    ctx["_laplace_grad"] = lambda v: _engine_grad(ctx, v)

    hooked = _run_hooks(
        ctx, eval_param=eval_param, eval_many=eval_many, eval_grad=eval_grad,
        eval_b0=eval_b0, compute_hessian=compute_hessian, hess_eps=hess_eps)
    if hooked is not None:
        return hooked

    outcome = _optimize_starts(ctx)
    if isinstance(outcome, dict):
        return outcome["abort_payload"]

    (best_param, best_val, best_meta, starts, failed_starts,
     _interrupted, _from_abort) = outcome
    return _build_payload(
        ctx, best_param, best_val, best_meta, starts, failed_starts,
        _interrupted, _from_abort)
