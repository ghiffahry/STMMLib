"""Top-level fitting functions: stmm_beta(), betaml().

Orchestrates data prep, internal name mapping, optimization,
and returns appropriate result object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from stmmlib._backend import Backend


def _sanitize_data(data: pd.DataFrame,
                   response_col: str = "y",
                   area_col: str = "area",
                   time_col: str = "time",
                   prov_col: str | None = "prov",
                   kab_col: str | None = "kab",
                   ) -> tuple[pd.DataFrame, dict, dict]:
    """Rename user columns to internal names and copy data.

    Arbitrary source column names are mapped to the internal convention:
    response→``y``, area→``.stmm_area``, time→``.stmm_time``,
    prov→``.stmm_prov``, kab→``.stmm_kab``. Any dataset works as long as the
    caller supplies the column names (or uses the conventional defaults).
    """
    col_map = {}
    for internal, original in ((".stmm_area", area_col), (".stmm_time", time_col),
                               (".stmm_prov", prov_col), (".stmm_kab", kab_col)):
        if original and original in data.columns and original != internal:
            col_map[original] = internal

    # Labels must match the codes assigned below. pd.Categorical sorts the
    # categories, so code k = k-th name in sorted order (not file order).
    area_labels = {}
    if area_col and area_col in data.columns:
        area_labels = dict(enumerate(pd.Categorical(data[area_col]).categories.tolist()))
    time_labels = {}
    if time_col and time_col in data.columns:
        time_labels = dict(enumerate(sorted(data[time_col].unique().tolist())))

    if response_col and response_col in data.columns:
        col_map[response_col] = "y"

    d = data.copy()
    if col_map:
        d = d.rename(columns=col_map)

    # Create categorical indices for area/time/prov/kab
    for c in (".stmm_area", ".stmm_time", ".stmm_prov", ".stmm_kab"):
        if c in d.columns:
            d[c] = pd.Categorical(d[c]).codes.astype(np.int32)

    return d, area_labels, time_labels


def _validate_contract(d: pd.DataFrame, W=None) -> None:
    """Validate input contract before optimization.

    W ordering contract: row/col i of W corresponds to the i-th area in
    SORTED order (pd.Categorical sorts categories before assigning codes).
    Keep W rows/cols aligned to that order; reordering areas without
    reordering W silently misassigns spatial neighbours.

    Checks:
        - W (when given) is square, finite, and matches n unique areas.
        - time codes are contiguous 0..T-1.
        - warns (UserWarning) when n < 10*p (p = # X cols incl. intercept).
    """
    import warnings
    if ".stmm_area" in d.columns:
        n_area = int(d[".stmm_area"].nunique())
        if W is not None:
            W_arr = W.toarray() if hasattr(W, "toarray") else np.asarray(W, dtype=np.float64)
            W_arr = np.asarray(W_arr, dtype=float)
            if W_arr.ndim != 2 or W_arr.shape[0] != W_arr.shape[1]:
                raise ValueError(f"W must be square, got shape {W_arr.shape}.")
            if not np.all(np.isfinite(W_arr)):
                raise ValueError("W must be finite (no NaN/inf).")
            if W_arr.shape[0] != n_area:
                raise ValueError(
                    f"W shape {W_arr.shape} mismatches n areas {n_area}.")
    if ".stmm_time" in d.columns:
        t = np.asarray(d[".stmm_time"].values)
        if len(t):
            if int(t.max()) - int(t.min()) + 1 != int(np.unique(t).size):
                raise ValueError("time codes must be contiguous 0..T-1.")
    skip_x = {".stmm_area", ".stmm_time", ".stmm_prov", ".stmm_kab",
              "y", "area", "time", "prov", "kab"}
    x_cols = [c for c in d.columns if c not in skip_x]
    p = len(x_cols) + 1
    if len(d) < 10 * p:
        warnings.warn(f"n={len(d)} < 10*p={10 * p}: small-sample fit.",
                      UserWarning, stacklevel=3)


def stmm_beta(
    formula: str = None,
    data: pd.DataFrame = None,
    model: str = "m1_iid_iid",
    W=None,
    method: str = "Laplace",
    gpu: str = "auto",
    n_starts: int = 3,
    max_iter: int = 200,
    inner_max_iter: int = 50,
    start_theta: "np.ndarray | None" = None,
    seed: int | None = None,
    compute_hessian: bool = True,
    preset: str | None = None,
    response_col: str = "y",
    area_col: str = "area",
    time_col: str = "time",
    prov_col: str | None = "prov",
    kab_col: str | None = "kab",
    config=None,
    compute_posterior: bool = False,
    **kwargs,
) -> "BetastmmResult":
    """Fit a spatio-temporal Beta mixed model.

    Args:
        formula: Formula string (currently unused; X from data columns
                 minus the grouping/response columns).
        data: pandas DataFrame. Response must be a proportion in (0,1).
        model: Model ID (e.g. "m1_iid_iid", "m3_sar_iid", "m5_seblup_rao_yu_it").
        W: Spatial weight matrix (scipy.sparse or numpy array).
        method: "Laplace" (default) or "ML".
        gpu: "auto", "yes", or "no".
        n_starts: Number of random restarts.
        max_iter: Outer optimization iterations.
        inner_max_iter: Inner Newton iterations.
        compute_hessian: finite-difference Hessian at the optimum (for the
            certificate + SEs). Set False for a fast fit (skips ~2k+1 solves).
        response_col: name of the (0,1)-bounded response column.
        area_col: name of the area/grouping column.
        time_col: name of the time/period column.
        prov_col: optional name of the province column (nested models).
        kab_col: optional name of the district column (nested models).
        preset: "fast" (low: ftol=1e-5, gtol=1e-5, no Hessian, max_iter
            cap 500), "balanced" (medium tol: ftol=1e-8, gtol=1e-6, Hessian,
            max_iter cap 500; cheap outer + honest SEs), "medium" (standard:
            ftol=1e-8, gtol=1e-6, no Hessian, no cap) or "full"
            (high/scientific: ftol=1e-12, gtol=1e-6, Hessian; default).
            See `stmm()` docstring for journal anchors.
        config: BetaFitConfig. Opt-in weak-prior MAP penalty (default off):
            phi_prior_mean/sd on logphi, dep_prior_sd on raw_dep. MAP with
            weak Gaussian prior; precedent Chung et al. 2013 blme and brglm2
            MPL Jeffreys. Always report penalized alongside unpenalized.
            Auto MAP refit on boundary: when config is None (no explicit
            config sent) and the unpenalized fit is invalid with log_phi or
            raw_dep parameters at their box bounds, stmm_beta refits ONCE
            with an automatic weak prior (phi_prior_mean=6.9, sd=2.0;
            dep off; blme-style, Chung et al. 2013). The
            unpenalized result is kept in optim["unpenalized"]; disable the
            auto by passing an explicit config (even BetaFitConfig()).
        compute_posterior: store the Laplace posterior curvature of the random
            effects (H_b = Z'diag(w)Z + Q) as a Cholesky factor, needed by
            BetastmmResult.ebp(). Orthogonal to `config` (no effect on the
            auto MAP refit); adds q^2 memory per fit, no extra refit.
        **kwargs: Additional arguments.

    Returns:
        BetastmmResult with coefficients, inference, logLik, AIC, BIC.
    """
    # Accuracy presets (L-BFGS-B tolerances; journal anchors in docstring of
    # `stmm()`): fast/low (ftol=1e-5, gtol=1e-5, no Hessian, cap 500),
    # balanced/medium-tol (ftol=1e-8, gtol=1e-6, Hessian, cap 500),
    # medium/standard (ftol=1e-8, gtol=1e-6, no Hessian, no cap),
    # full/high (ftol=1e-12, gtol=1e-6, Hessian, no cap -- original behavior).
    # Only the Hessian flag changes cost structure: the Hessian is used for
    # SEs/lambda_min, never for the optimization path (benchmark: identical
    # logLik/theta/cAIC, delta <= 1e-12). No Hessian -> no SEs, hessian_pd
    # None in the certificate, and predict_interval loses var_fixed.
    from stmmlib.beta.config import resolve_preset
    _ftol, _gtol, _hess, _cap = resolve_preset(preset)
    compute_hessian = compute_hessian and _hess
    if _cap is not None:
        max_iter = min(max_iter, _cap)
    from stmmlib._backend import Backend
    backend = Backend(gpu=gpu)

    if data is None:
        raise ValueError("data must be provided.")
    if response_col not in data.columns and "y" not in data.columns:
        raise ValueError(
            f"response column {response_col!r} not found; pass response_col=<name>.")

    d, area_labels, time_labels = _sanitize_data(
        data, response_col=response_col, area_col=area_col, time_col=time_col,
        prov_col=prov_col, kab_col=kab_col)
    if "y" not in d.columns:
        raise ValueError(f"response column {response_col!r} not found in data.")
    _validate_contract(d, W)

    # T-aware warning for AR(1) temporal structures (Nickell 1981: AR(1)
    # persistence is weakly identified on short panels; IID time is the
    # safer comparator). Fires at most once per fit.
    import warnings
    from stmmlib.beta.structure import stmm_beta_structure
    try:
        _time_struct = stmm_beta_structure(model)["time"]
    except KeyError:
        _time_struct = None
    if _time_struct == "ar1" and ".stmm_time" in d.columns:
        _T = int(d[".stmm_time"].nunique())
        if _T < 15:
            warnings.warn(
                "AR(1) temporal pada panel T pendek (<15): persistensi lemah "
                "teridentifikasi (Nickell 1981, Econometrica 49(6)); "
                "pertimbangkan pembanding varian waktu IID",
                UserWarning, stacklevel=2)
    y = d["y"].values.astype(np.float64)

    # Drop y and grouping columns from X columns
    drop_cols = {"y", "area", "time", "prov", "kab",
                 ".stmm_area", ".stmm_time", ".stmm_prov", ".stmm_kab"}

    if model == "beta_regression":
        from stmmlib.beta.solvers import beta_regression_base
        opt = beta_regression_base(formula, d, y, method)
        from stmmlib.beta.result import BetaResult
        return BetaResult(opt)

    from stmmlib.beta.solvers import beta_laplace_structural
    opt = beta_laplace_structural(
        formula, d, y, method, model, W,
        n_starts=n_starts, max_iter=max_iter,
        inner_max_iter=inner_max_iter, backend=backend,
        use_gpu=(str(gpu).lower() == "yes"),
        start_theta=start_theta, seed=seed,
        compute_hessian=compute_hessian,
        ftol=_ftol, gtol=_gtol, config=config,
        compute_posterior=compute_posterior,
    )

    # Auto weak-prior MAP refit (blme-style, Chung et al. 2013): when the
    # unpenalized fit hits the log_phi/raw_dep box boundary and the
    # certificate goes invalid, refit ONCE with a weak Gaussian prior.
    # Trigger requires config=None (an explicit config, even
    # BetaFitConfig(), disables the auto). The unpenalized snapshot is
    # kept in optim["unpenalized"]; no recursion (the refit passes a
    # config, so this block cannot fire again).
    # Isolasi hasil utama vs refit: copy final SEBELUM refit kedua jalan.
    # _opt_unpen = opt adalah alias; copy putus rantai view/reference.
    import copy as _copy

    def _snap(a):
        return np.array(a, dtype=np.float64, copy=True) if a is not None else None
    _main_X = _snap(opt.get("X"))
    _main_Z = _snap(opt.get("Z"))
    _main_beta = _snap(opt.get("beta_hat"))
    _main_b = _snap(opt.get("b_hat"))
    _main_logs = _snap(opt.get("logs"))
    _main_raw = _snap(opt.get("raw_dep"))
    _main_boot = _snap(opt.get("_boot_theta"))
    _main_hb = _snap(opt.get("H_b_chol"))
    _main_des = _copy.deepcopy(opt.get("des")) if opt.get("des") is not None else None

    def _restore_main(o):
        if _main_X is not None:
            o["X"] = _main_X
        if _main_Z is not None:
            o["Z"] = _main_Z
        if _main_beta is not None:
            o["beta_hat"] = _main_beta
        if _main_b is not None:
            o["b_hat"] = _main_b
        if _main_logs is not None:
            o["logs"] = _main_logs
        if _main_raw is not None:
            o["raw_dep"] = _main_raw
        if _main_boot is not None:
            o["_boot_theta"] = _main_boot
        if _main_hb is not None:
            o["H_b_chol"] = _main_hb
        if _main_des is not None:
            o["des"] = _main_des
            try:
                o["des"]["Z"] = o["Z"]
            except Exception:
                pass
        return o
    from stmmlib.beta.config import (BetaFitConfig, DEP_PRIOR_AUTO_SD,
                                     PHI_PRIOR_AUTO_MEAN, PHI_PRIOR_AUTO_SD)
    _cert = opt.get("certificate") or {}
    if (config is None and _cert.get("valid") is False
            and _cert.get("at_bound") is True
            and any(b == "log_phi" or str(b).startswith("raw_dep")
                    for b in (_cert.get("bound_params") or []))
            and "penalty_auto" not in opt):
        _raw = opt.get("raw_dep")
        _unpen = {
            "log_lik": opt.get("log_lik"),
            "logphi": opt.get("logphi"),
            "raw_dep": list(np.asarray(_raw).ravel()) if _raw is not None else [],
            "status": opt.get("status"),
            "cert_valid": _cert.get("valid"),
        }
        _auto_cfg = BetaFitConfig(phi_prior_mean=PHI_PRIOR_AUTO_MEAN,
                                   phi_prior_sd=PHI_PRIOR_AUTO_SD,
                                   dep_prior_sd=DEP_PRIOR_AUTO_SD,
                                   compute_posterior=compute_posterior)
        _opt_unpen = opt
        _opt_pen = beta_laplace_structural(
            formula, d, y, method, model, W,
            n_starts=n_starts, max_iter=max_iter,
            inner_max_iter=inner_max_iter, backend=backend,
            use_gpu=(str(gpu).lower() == "yes"),
            start_theta=start_theta, seed=seed,
            compute_hessian=compute_hessian,
            ftol=_ftol, gtol=_gtol, config=_auto_cfg,
        )
        # Seleksi validity-first (prioritas):
        # 1. VALID-penalized menang atas unpenalized INVALID apapun ll-nya.
        # 2. Sesama VALID menang ll UNPENALIZED tertinggi (eval tanpa penalti).
        # 3. Penalized INVALID/abort/it=0/degenerate -> pakai unpenalized awal.
        # Satu sumber angka: X = ll unpenalized di titik awal,
        # Y = ll unpenalized di titik penalized (eval tanpa penalti).
        # Bila Y fallback ke objective penalized, label eksplisit obj penalized.
        _ll_unpen = float(_opt_unpen.get("log_lik", float("nan")))
        _pen_theta = _opt_pen.get("_boot_theta")
        _ll_pen_unpen = float("nan")
        _y_from_obj = False
        if _pen_theta is not None:
            try:
                _ev = beta_laplace_structural(
                    formula, d, y, method, model, W, config=BetaFitConfig(),
                    compute_hessian=False, eval_param=np.asarray(_pen_theta, float))
                _ll_pen_unpen = float(_ev.get("log_lik", float("nan")))
            except Exception:
                _ll_pen_unpen = float("nan")
        if not np.isfinite(_ll_pen_unpen):
            try:
                _ll_pen_unpen = float(_opt_pen.get("log_lik", float("nan")))
                _y_from_obj = bool(np.isfinite(_ll_pen_unpen))
            except Exception:
                _ll_pen_unpen = float("nan")
        def _fll(v):
            return f"{v:.4f}" if np.isfinite(v) else "NA"
        _y_lab = "obj penalized" if _y_from_obj else "ll unpenalized"
        _pen_cert = _opt_pen.get("certificate") or {}
        _pen_optim = _opt_pen.get("optim") or {}
        try:
            _pen_nit = int(_pen_optim.get("nit", 1) or 0)
        except (TypeError, ValueError):
            _pen_nit = 1
        _pen_valid = (
            _pen_cert.get("valid") is True
            and not _opt_pen.get("degenerate", False)
            and not _pen_optim.get("all_starts_abort", False)
            and _opt_pen.get("status") != "invalid"
            and _pen_nit != 0
            and np.isfinite(float(_opt_pen.get("log_lik", float("nan"))))
        )
        _unpen_valid = (_cert.get("valid") is True)
        if _pen_valid and not _unpen_valid:
            opt = _opt_pen
            _cert = opt.get("certificate") or {}
            _optim = opt.setdefault("optim", {})
            _optim["penalty_auto"] = True
            _optim["unpenalized"] = _unpen
            _msg = (f"auto-prior: ll unpenalized {_fll(_ll_unpen)} -> "
                    f"{_y_lab} {_fll(_ll_pen_unpen)} "
                    f"(titik penalized dipakai; unpenalized INVALID)")
            _optim["penalty_refit_msg"] = _msg
            _cert.setdefault("warnings", []).append(_msg)
        elif _pen_valid and _unpen_valid:
            _pen_wins = (np.isfinite(_ll_pen_unpen) and np.isfinite(_ll_unpen)
                         and _ll_pen_unpen >= _ll_unpen)
            if _pen_wins:
                opt = _opt_pen
                _cert = opt.get("certificate") or {}
                _optim = opt.setdefault("optim", {})
                _optim["penalty_auto"] = True
                _optim["unpenalized"] = _unpen
                _msg = (f"auto-prior: ll unpenalized {_fll(_ll_unpen)} -> "
                        f"{_y_lab} {_fll(_ll_pen_unpen)} "
                        f"(titik penalized dipakai)")
                _optim["penalty_refit_msg"] = _msg
                _cert.setdefault("warnings", []).append(_msg)
            else:
                opt = _restore_main(_opt_unpen)
                _cert = opt.get("certificate") or {}
                _optim = opt.setdefault("optim", {})
                _optim["penalty_auto"] = True
                _optim["unpenalized"] = _unpen
                _msg = (f"auto-prior: ll unpenalized {_fll(_ll_unpen)} -> "
                        f"{_y_lab} {_fll(_ll_pen_unpen)}; "
                        f"hasil unpenalized dipakai")
                _optim["penalty_refit_msg"] = _msg
                _cert.setdefault("warnings", []).append(_msg)
        else:
            opt = _restore_main(_opt_unpen)
            _cert = opt.get("certificate") or {}
            _optim = opt.setdefault("optim", {})
            _optim["penalty_auto"] = True
            _optim["unpenalized"] = _unpen
            _optim["penalty_refit_failed"] = True
            _msg = (f"refit penalized ditolak: penalized INVALID "
                    f"(ll unpenalized {_fll(_ll_unpen)} -> "
                    f"{_y_lab} {_fll(_ll_pen_unpen)}); "
                    f"hasil unpenalized dipakai")
            _optim["penalty_refit_msg"] = _msg
            _cert.setdefault("warnings", []).append(_msg)

    from stmmlib.beta.inference import compute_stbl_inference
    inf = compute_stbl_inference(opt, skip_hessian=not _hess)

    # Conditional mean (fixed + random effects) for the fitted units.
    X = opt.get("X")
    Z = opt.get("Z")
    b = opt.get("b_hat")
    beta = opt.get("beta_hat")
    if X is not None and beta is not None:
        eta = X @ beta
        if Z is not None and b is not None:
            eta = eta + Z @ b
        opt["mu_hat"] = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-15, 1.0 - 1e-15)

    from stmmlib.beta.result import BetastmmResult
    res = BetastmmResult(opt, inf)
    res._area_labels = area_labels
    res._time_labels = time_labels
    res._W = W
    return res


def betaml(formula=None, data=None, method="ML", gpu="auto"):
    """Fit a fixed-effects Beta regression via ML (L-BFGS-B).

    Args:
        formula: Formula (unused; intercept-only for now).
        data: pandas DataFrame with 'y' column in (0,1).
        method: "ML".
        gpu: "auto", "yes", or "no".

    Returns:
        BetaResult.
    """
    if data is None or "y" not in data.columns:
        raise ValueError("data must contain a 'y' column.")

    y = data["y"].values.astype(np.float64)
    from stmmlib.beta.solvers import beta_regression_base
    opt = beta_regression_base(formula, data, y, method, "beta_regression")
    from stmmlib.beta.result import BetaResult
    return BetaResult(opt)
