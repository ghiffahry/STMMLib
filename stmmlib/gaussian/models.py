"""Per-model wrappers: lmm_base, rao_yu, sar_only, sar_ar1, seblup_rao_yu_it,
nested_iid_iid, nested_iid_ar1.

Each wraps the core Gaussian fitting engine for a specific model structure.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stmmlib.gaussian.transforms import (
    make_model,
    prepare_panel_data,
)
from stmmlib.gaussian.design import (
    build_design_matrices,
    build_interaction_matrices,
)
from stmmlib.gaussian.optimization import (
    auto_start,
    run_optimization,
    fit_seblup5,
    nested_fit,
    nested_ar1_fit,
)
from stmmlib.gaussian.gls import (
    logLik_ML,
    logLik_REML,
    compute_AIC,
)
from stmmlib.gaussian.hessian import compute_hessian_se
from stmmlib.gaussian.blup import compute_blup, nested_blup, nested_ar1_blup
from stmmlib.gaussian.variance_decomp import (
    compute_variance_decomposition,
    compute_R2,
)

if TYPE_CHECKING:
    from stmmlib._backend import Backend


# ---------------------------------------------------------------------------
# Core fitting engine (M1-M4)
# ---------------------------------------------------------------------------


def stmm_normal_core(
    formula: str,
    data: pd.DataFrame,
    area: str,
    time: str,
    spatial: str = "iid",
    temporal: str = "iid",
    interaction: str = "none",
    W=None,
    method: str = "REML",
    start: dict = None,
    n_starts: int = 3,
    max_iter: int = 200,
    compute_blup_flag: bool = True,
    compute_se_theta: bool = True,
    verbose: bool = False,
    prov_col: str | None = None,
    kab_col: str | None = None,
    response_col: str | None = None,
    backend: Backend = None,
) -> dict:
    """Main core fitting engine for stLMM class (M1-M4).

    Args:
        formula: Formula string (currently unused; X constructed from data cols).
        data: pandas DataFrame.
        area: Area column name.
        time: Time column name.
        spatial: Area type ("iid" or "sar").
        temporal: Time type ("iid" or "ar1").
        interaction: Interaction type.
        W: Spatial weight matrix.
        method: "ML" or "REML".
        start: Starting theta.
        n_starts: Number of optimization starts.
        max_iter: Max iterations.
        compute_blup_flag: Compute BLUPs.
        compute_se_theta: Compute Hessian SEs.
        verbose: Print progress.
        backend: Array backend.

    Returns:
        dict with all fit results (stMM-compatible).
    """
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu="auto")

    # Prepare panel data
    prepared = prepare_panel_data(data, area, time, verbose=verbose)
    prepared["W"] = W

    # Build model
    time_scope = "global" if (spatial == "sar" and temporal == "ar1") else "area_specific"
    model = make_model(spatial, temporal, interaction, time_scope)

    # Design matrices
    des = build_design_matrices(prepared, backend)
    prepared.update(des)
    Z_area = prepared["Z_area"]
    Z_time = prepared["Z_time"]

    n = prepared["n"]
    m = prepared["m"]
    T = prepared["T"]

    # Fixed-effects design X (grouping columns prov/kab must not leak in)
    y_col = response_col or "y"
    _skip = {area, time, y_col, "area", "time", "y",
             prov_col or "prov", kab_col or "kab"}
    x_cols = [c for c in data.columns if c not in _skip]
    if y_col in data.columns:
        y = data[y_col].values.astype(np.float64)
    else:
        raise ValueError(f"Data must contain response column {y_col!r}.")
    X = np.column_stack([np.ones(n)] + [data[c].values.astype(np.float64) for c in x_cols]) if x_cols else np.ones((n, 1), dtype=np.float64)

    # Build Z for interaction
    interaction_data = build_interaction_matrices(
        prepared, interaction, prepared["area_index"], m, T, backend
    )
    if interaction_data["Z_gamma"] is not None:
        # Augment design: [Z_area, Z_time, Z_gamma]
        from stmmlib._backend import Backend as _
        Z_base = backend.sparse_kron(Z_area, backend.sparse_eye(T, dtype=backend.xp.float64), format="csr") if backend.is_sparse(Z_area) else Z_area
        # For simplicity, use dense for now
        prepared["Z_gamma"] = interaction_data["Z_gamma"]
    else:
        prepared["Z_gamma"] = None

    # Optimization
    opt = run_optimization(
        y, X, prepared, model, backend,
        start=start, method=method, n_starts=n_starts, max_iter=max_iter,
        verbose=verbose,
    )
    theta_hat = opt["theta_hat"]
    log_lik = opt["log_lik"]

    # AIC/BIC
    k = len(opt["theta_hat"]) + X.shape[1]
    aic = compute_AIC(log_lik, k)
    bic = -2 * log_lik + np.log(n) * k

    # GLS at optimal theta
    from stmmlib.gaussian.gls import gls as _gls, common_components
    from stmmlib.gaussian.covariance import build_V
    V = build_V(prepared, model, theta_hat, backend)
    gls_result = _gls(y, X, V, backend)
    beta_hat = gls_result["beta_hat"]
    se_beta = gls_result["se_beta"]

    # BLUPs
    blup = None
    if compute_blup_flag:
        blup = compute_blup(y, X, prepared, model, theta_hat, beta_hat, backend)

    # Hessian SEs
    hessian = None
    if compute_se_theta:
        try:
            hessian = compute_hessian_se(theta_hat, y, X, prepared, model, method, backend)
        except Exception as e:
            if verbose:
                warnings.warn(f"Hessian computation failed: {e}")

    # Variance decomposition
    fit_info = {"theta": theta_hat, "model": model, "data": prepared}
    vd = compute_variance_decomposition(fit_info, backend)
    r2 = compute_R2(vd)

    # Assemble result
    coef_names = [f"beta_{i}" for i in range(X.shape[1])]
    result = {
        "model_name": model.get("area_type", "iid") + "+" + model.get("time_type", "iid"),
        "family": "normal",
        "link": "identity",
        "method": method,
        "n": n,
        "m": m,
        "T": T,
        "p": X.shape[1],
        "theta": theta_hat,
        "beta_hat": beta_hat,
        "se_beta": se_beta,
        "vcov_beta": gls_result["vcov_beta"],
        "residuals": gls_result["residuals"],
        "fitted": gls_result["fitted"],
        "log_lik": log_lik,
        "AIC": aic,
        "BIC": bic,
        "k": k,
        "blup": blup,
        "hessian": hessian,
        "variance_decomp": vd,
        "R2": r2,
        "X": X,
        "V": V,
        "coef_names": coef_names,
        "model": model,
        "data": prepared,
        "W": W,
        "converged": opt["converged"],
    }
    return result


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


def lmm_base(formula, data, area, time, interaction="none", method="REML",
             start=None, n_starts=3, max_iter=200, compute_blup=True,
             compute_se=True, verbose=False, backend=None):
    """IID + IID base model (M1)."""
    return stmm_normal_core(
        formula, data, area, time, spatial="iid", temporal="iid",
        interaction=interaction, method=method, start=start, n_starts=n_starts,
        max_iter=max_iter, compute_blup_flag=compute_blup, compute_se_theta=compute_se,
        verbose=verbose, backend=backend,
    )


def rao_yu(formula, data, area, time, interaction="none", method="REML",
           start=None, n_starts=3, max_iter=200, compute_blup=True,
           compute_se=True, verbose=False, backend=None):
    """IID + AR(1) (Rao-Yu) model (M2)."""
    return stmm_normal_core(
        formula, data, area, time, spatial="iid", temporal="ar1",
        interaction=interaction, method=method, start=start, n_starts=n_starts,
        max_iter=max_iter, compute_blup_flag=compute_blup, compute_se_theta=compute_se,
        verbose=verbose, backend=backend,
    )


def sar_only(formula, data, area, time, W, interaction="none", method="REML",
             start=None, n_starts=3, max_iter=200, compute_blup=True,
             compute_se=True, verbose=False, backend=None):
    """SAR + IID model (M3)."""
    return stmm_normal_core(
        formula, data, area, time, spatial="sar", temporal="iid",
        interaction=interaction, W=W, method=method, start=start, n_starts=n_starts,
        max_iter=max_iter, compute_blup_flag=compute_blup, compute_se_theta=compute_se,
        verbose=verbose, backend=backend,
    )


def sar_ar1(formula, data, area, time, W, interaction="none", method="REML",
            start=None, n_starts=3, max_iter=200, compute_blup=True,
            compute_se=True, verbose=False, backend=None):
    """SAR + AR(1) (full ST) model (M4)."""
    return stmm_normal_core(
        formula, data, area, time, spatial="sar", temporal="ar1",
        interaction=interaction, W=W, method=method, start=start, n_starts=n_starts,
        max_iter=max_iter, compute_blup_flag=compute_blup, compute_se_theta=compute_se,
        verbose=verbose, backend=backend,
    )


def seblup_rao_yu_it(
    formula, data, area, time, W, interaction="none", area_type="sar",
    method="REML", start=None, n_starts=3, max_iter=200,
    compute_se=True, verbose=False, response_col="y", backend=None,
):
    """SEBLUP Rao-Yu IT model (M5/M12)."""
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu="auto")

    prepared = prepare_panel_data(data, area, time, verbose=verbose)
    prepared["W"] = W

    model = make_model(area_type, "ar1", interaction, "area_specific")
    des = build_design_matrices(prepared, backend)
    prepared.update(des)
    # Interaction design (Z_gamma) so logLik_seblup5/build_V_seblup5 include V_gamma
    interaction_data = build_interaction_matrices(
        prepared, interaction, prepared["area_index"], prepared["m"], prepared["T"], backend
    )
    prepared["Z_gamma"] = interaction_data["Z_gamma"]
    n = prepared["n"]

    y = data[response_col].values.astype(np.float64)
    x_cols = [c for c in data.columns if c not in (area, time, "y", "area", "time", "prov", "kab")]
    X = np.column_stack([np.ones(n)] + [data[c].values.astype(np.float64) for c in x_cols]) if x_cols else np.ones((n, 1), dtype=np.float64)

    opt = fit_seblup5(y, X, prepared, interaction, area_type, backend,
                      method=method, start=start, n_starts=n_starts, verbose=verbose)

    theta_hat = opt["theta_hat"]
    from stmmlib.gaussian.covariance import build_V_seblup5
    V = build_V_seblup5(prepared, theta_hat, interaction, area_type, backend)
    from stmmlib.gaussian.gls import gls as _gls
    gls_res = _gls(y, X, V, backend)
    k = len(theta_hat) + X.shape[1]
    log_lik = opt["log_lik"]

    # EBLUP at theta_hat (Henderson): v_hat = G_a Z_a' V^-1 e,
    # u_hat = (I_m (x) G_u) Z_c' V^-1 e. V is the one already built above, so
    # this adds no refit and changes no fit quantity.
    from stmmlib.gaussian.covariance import (
        _to_np_backend, build_G_area, G_ar1_core,
    )
    from stmmlib.gaussian.gls import _to_np, _solve_chol, safe_cholesky
    _e = (_to_np(backend, y)
          - _to_np(backend, X) @ _to_np(backend, gls_res["beta_hat"]))
    _L = gls_res.get("L")
    if _L is None:
        _L = safe_cholesky(V, backend)
    _Vinv_e = _solve_chol(_L, _e, backend)
    _Ga = _to_np_backend(
        backend,
        build_G_area(area_type, theta_hat.get("sigma2_v", 1.0),
                     prepared["m"], W, theta_hat.get("rho", 0.0), backend))
    _Za = _to_np_backend(backend, prepared["Z_area"])
    v_hat = _Ga @ _Za.T @ _Vinv_e
    _Gu = _to_np_backend(
        backend,
        G_ar1_core(prepared["T"], theta_hat.get("sigma2_eta", 1.0),
                   theta_hat.get("phi", 0.0), backend))
    _cell_idx = np.asarray(prepared["area_index"], dtype=np.int64) * prepared["T"] \
        + np.asarray(prepared["time_index"], dtype=np.int64)
    _Zc = np.zeros((n, prepared["m"] * prepared["T"]), dtype=np.float64)
    _Zc[np.arange(n), _cell_idx] = 1.0
    u_hat = np.kron(np.eye(prepared["m"], dtype=np.float64), _Gu) @ (_Zc.T @ _Vinv_e)

    return {
        "model_name": f"seblup_{area_type}_rao_yu_it",
        "family": "normal", "method": method,
        "n": n, "m": prepared["m"], "T": prepared["T"], "p": X.shape[1],
        "theta": theta_hat,
        "beta_hat": gls_res["beta_hat"], "se_beta": gls_res["se_beta"],
        "vcov_beta": gls_res["vcov_beta"],
        "residuals": gls_res["residuals"], "fitted": gls_res["fitted"],
        "blup": {"v_hat": v_hat, "u_hat": u_hat,
                 "beta_hat": gls_res["beta_hat"]},
        "data": prepared,
        "log_lik": log_lik, "AIC": compute_AIC(log_lik, k),
        "BIC": -2 * log_lik + np.log(n) * k, "k": k,
        "converged": opt["converged"],
    }


def nested_iid_iid(
    formula, data, area, time, prov, kab, interaction="none", W=None,
    method="REML", start=None, n_starts=3, max_iter=200,
    compute_blup=True, compute_se=True, verbose=False, response_col="y", backend=None,
):
    """Nested IID + IID model (M6/M13)."""
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu="auto")

    prepared = prepare_panel_data(data, area, time, prov_col=prov, kab_col=kab, verbose=verbose)
    prepared["W"] = W

    model = make_model("nested_iid", "iid", interaction)
    n = prepared["n"]

    y = data[response_col].values.astype(np.float64)
    x_cols = [c for c in data.columns if c not in (area, time, prov, kab, "y")]
    X = np.column_stack([np.ones(n)] + [data[c].values.astype(np.float64) for c in x_cols]) if x_cols else np.ones((n, 1), dtype=np.float64)

    # Build nested Z matrices
    from stmmlib.gaussian.design import build_Z_nested, build_interaction_matrices
    nested_des = build_Z_nested(prepared, backend)
    prepared.update(nested_des)
    interaction_data = build_interaction_matrices(
        prepared, interaction, prepared["area_index"], prepared["m"], prepared["T"], backend
    )
    prepared["Z_gamma"] = interaction_data["Z_gamma"]

    opt = nested_fit(y, X, prepared, interaction, backend, method=method,
                     start=start, n_starts=n_starts, verbose=verbose)

    theta_hat = opt["theta_hat"]
    k = len(theta_hat) + X.shape[1]
    log_lik = opt["log_lik"]

    # GLS at optimal theta + Henderson BLUP
    from stmmlib.gaussian.covariance import build_V_nested
    from stmmlib.gaussian.gls import gls as _gls
    from stmmlib.gaussian.blup import nested_blup
    V = build_V_nested(prepared, theta_hat, interaction, backend)
    gls_res = _gls(y, X, V, backend)
    beta_hat = gls_res["beta_hat"]
    blup = nested_blup(theta_hat, y, X, prepared, interaction, beta_hat, backend) \
        if compute_blup else None

    return {
        "model_name": "nested_iid_iid",
        "family": "normal", "method": method,
        "n": n, "m": prepared["m"], "T": prepared["T"], "p": X.shape[1],
        "theta": theta_hat,
        "beta_hat": beta_hat, "se_beta": gls_res["se_beta"],
        "vcov_beta": gls_res["vcov_beta"],
        "residuals": gls_res["residuals"], "fitted": gls_res["fitted"],
        "blup": blup,
        "log_lik": log_lik,
        "AIC": compute_AIC(log_lik, k), "BIC": -2 * log_lik + np.log(n) * k, "k": k,
        "converged": opt["converged"],
    }


def nested_iid_ar1(
    formula, data, area, time, prov, kab, interaction="none", W=None,
    method="REML", start=None, n_starts=3, max_iter=200,
    compute_blup=True, compute_se=True, verbose=False, response_col="y", backend=None,
):
    """Nested IID + AR1 model (M7/M14)."""
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu="auto")

    prepared = prepare_panel_data(data, area, time, prov_col=prov, kab_col=kab, verbose=verbose)
    prepared["W"] = W

    model = make_model("nested_sar", "ar1", interaction)
    n = prepared["n"]

    y = data[response_col].values.astype(np.float64)
    x_cols = [c for c in data.columns if c not in (area, time, prov, kab, "y")]
    X = np.column_stack([np.ones(n)] + [data[c].values.astype(np.float64) for c in x_cols]) if x_cols else np.ones((n, 1), dtype=np.float64)

    from stmmlib.gaussian.design import build_Z_nested, build_interaction_matrices
    nested_des = build_Z_nested(prepared, backend)
    prepared.update(nested_des)
    interaction_data = build_interaction_matrices(
        prepared, interaction, prepared["area_index"], prepared["m"], prepared["T"], backend
    )
    prepared["Z_gamma"] = interaction_data["Z_gamma"]

    opt = nested_ar1_fit(y, X, prepared, interaction, backend, method=method,
                         start=start, n_starts=n_starts, verbose=verbose)

    theta_hat = opt["theta_hat"]
    k = len(theta_hat) + X.shape[1]
    log_lik = opt["log_lik"]

    # GLS at optimal theta + Henderson BLUP
    from stmmlib.gaussian.covariance import build_V_nested_ar1
    from stmmlib.gaussian.gls import gls as _gls
    from stmmlib.gaussian.blup import nested_ar1_blup
    V = build_V_nested_ar1(prepared, theta_hat, interaction, backend)
    gls_res = _gls(y, X, V, backend)
    beta_hat = gls_res["beta_hat"]
    blup = nested_ar1_blup(theta_hat, y, X, prepared, interaction, beta_hat, backend) \
        if compute_blup else None

    return {
        "model_name": "nested_iid_ar1",
        "family": "normal", "method": method,
        "n": n, "m": prepared["m"], "T": prepared["T"], "p": X.shape[1],
        "theta": theta_hat,
        "beta_hat": beta_hat, "se_beta": gls_res["se_beta"],
        "vcov_beta": gls_res["vcov_beta"],
        "residuals": gls_res["residuals"], "fitted": gls_res["fitted"],
        "blup": blup,
        "log_lik": log_lik,
        "AIC": compute_AIC(log_lik, k), "BIC": -2 * log_lik + np.log(n) * k, "k": k,
        "converged": opt["converged"],
    }
