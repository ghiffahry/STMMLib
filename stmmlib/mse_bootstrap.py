"""Parametric bootstrap MSE of the area-level predictor.

One layer only: ``beta`` and ``theta`` are held at the fitted values while
generating, and the refit re-estimates them. The bootstrap target is the
realized area mean of the drawn random effects (the identity link for
Gaussian, inverse-logit for Beta). No second-order bias correction, and no
double bootstrap.

Gaussian: ``mse_eblup(fit)`` where ``fit`` is the dict returned by ``stmm``.
Beta:     ``mse_ebp(fit)`` where ``fit`` is a ``BetastmmResult``.

v1 supports flat models only (no nested M6/M7/M13/M14, no interaction
M8-M12). Nested or interaction fits raise ``NotImplementedError``.

A replication whose refit raises, or whose predictor is not finite, counts as
a failed replication. If fewer than ``max(10, B // 4)`` replications succeed,
``RuntimeError`` is raised instead of returning a result built on a handful
of draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stmmlib._mse_common import _frame, _invlogit, _onehot, _row_means
from stmmlib._mse_beta import (
    beta_area_labels,
    beta_fit_predictor,
    beta_laplace_structural,
    predict_beta_area_ebp,
    simulate_beta,
)
from stmmlib._mse_gaussian import (
    gaussian_model_id,
    predict_gaussian_rows,
    refit_gaussian,
    simulate_gaussian,
)


def mse_ebp(fit, B: int = 100, seed: int = 0) -> pd.DataFrame:
    """Bootstrap MSE of the EBP of each area-level proportion.

    Returns a DataFrame with columns ``area, ebp, mse, rmse, rel_rmse,
    n_boot_ok``. ``rel_rmse`` is ``rmse / max(abs(ebp), eps)``, a flag rather
    than a standardized measure.
    """
    opt = getattr(fit, "_opt", None) or {}
    des = opt.get("des") or {}
    blocks = des.get("blocks") or {}
    s = fit._structure() if callable(getattr(fit, "_structure", None)) else {}
    interaction = (s or {}).get("interaction", "none")
    if "prov" in blocks or interaction != "none":
        raise NotImplementedError(
            "MSE bootstrap: nested and interaction models are not supported yet. "
            "Use a flat model (M1-M5).")

    X = np.asarray(opt["X"], dtype=np.float64)
    Z = np.asarray(opt["Z"], dtype=np.float64)
    beta0 = np.asarray(opt["beta_hat"], dtype=np.float64).ravel()
    logs = np.asarray(opt.get("logs", []), dtype=np.float64).ravel()
    raw_dep = np.asarray(opt.get("raw_dep", []), dtype=np.float64).ravel()
    logphi = float(opt["logphi"]) if "logphi" in opt else float(np.log(float(opt["phi"])))
    area_idx = np.asarray(des["area_idx"], dtype=np.int64)
    T = int(des["T"])
    m = int(des["m"])

    fit_W = getattr(fit, "_W", None)
    W = fit_W.toarray() if hasattr(fit_W, "toarray") else fit_W
    rho_bound = float(des.get("rho_bound", 0.95))
    phi_bound = float(des.get("phi_bound", 0.95))
    area_type = (s or {}).get("area", "iid")
    time_type = (s or {}).get("time", "iid")
    time_scope = (s or {}).get("time_scope", "area_specific")

    labels = beta_area_labels(fit, m)
    rng = np.random.default_rng(seed)
    fit_pred = beta_fit_predictor(fit, rng, area_idx, m)

    preds, targets, n_ok = [], [], 0
    for _ in range(B):
        try:
            eta, y_star = simulate_beta(rng, X, Z, beta0, logs, raw_dep, logphi,
                                        area_type, time_type, time_scope,
                                        m, T, W, rho_bound, phi_bound)
            refit = beta_laplace_structural(
                None, opt["_boot_frame"], y_star, "Laplace", opt["model"],
                W=fit_W, n_starts=1, max_iter=600, inner_max_iter=30,
                start_theta=opt.get("_boot_theta"),
                compute_posterior=True, compute_hessian=False)
            ebps = predict_beta_area_ebp(refit, rng, area_idx, m)
            target = _row_means(_invlogit(eta), area_idx, m)
            if not np.all(np.isfinite(ebps)) or not np.all(np.isfinite(target)):
                continue
            preds.append(ebps)
            targets.append(target)
            n_ok += 1
        except Exception:
            continue
    return _frame(labels, fit_pred, preds, targets, n_ok, B, "ebp")


def mse_eblup(fit, B: int = 100, seed: int = 0) -> pd.DataFrame:
    """Bootstrap MSE of the EBLUP of each area-level mean.

    Same contract as :func:`mse_ebp`; the predictor is the Gaussian BLUP of the
    conditional mean at the refit. ``fit`` is the dict returned by ``stmm``.
    """
    model = fit.get("model") or {}
    area_type = model.get("area_type")
    interaction = model.get("interaction", "none")
    if area_type in ("nested_iid", "nested_sar") or interaction != "none":
        raise NotImplementedError(
            "MSE bootstrap: nested and interaction models are not supported yet. "
            "Use a flat model (M1-M5).")

    prep = fit["data"]
    X = np.asarray(fit["X"], dtype=np.float64)
    beta0 = np.asarray(fit["beta_hat"], dtype=np.float64).ravel()
    theta = fit["theta"]
    area_idx = np.asarray(prep["area_index"], dtype=np.int64)
    time_idx = np.asarray(prep["time_index"], dtype=np.int64)
    m = int(prep["m"])
    T = int(prep["T"])
    labels = list(prep.get("area_labels") or list(range(m)))
    W = fit.get("W", prep.get("W"))
    method = fit.get("method", "REML")
    x_cols = list(fit.get("x_cols", []))
    dfr = prep["data"]

    Z_area = _onehot(area_idx, m)
    Z_cell = _onehot(area_idx * T + time_idx, m * T)
    model_id = gaussian_model_id(model)
    rng = np.random.default_rng(seed)
    fit_pred = _row_means(predict_gaussian_rows(fit, Z_area, Z_cell), area_idx, m)

    preds, targets, n_ok = [], [], 0
    for _ in range(B):
        try:
            eta, y_star = simulate_gaussian(rng, theta, model, W, m, T, X, beta0,
                                            area_idx, time_idx)
            df = pd.DataFrame({"y": y_star})
            for c in x_cols:
                df[c] = dfr[c].to_numpy()
            df["area"] = area_idx
            df["time"] = time_idx
            refit = refit_gaussian(fit, df, model_id, W, method)
            pred = _row_means(predict_gaussian_rows(refit, Z_area, Z_cell),
                              area_idx, m)
            target = _row_means(eta, area_idx, m)
            if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(target)):
                continue
            preds.append(pred)
            targets.append(target)
            n_ok += 1
        except Exception:
            continue
    return _frame(labels, fit_pred, preds, targets, n_ok, B, "eblup")
