"""Gaussian family internals for the parametric bootstrap MSE."""

from __future__ import annotations

import numpy as np

from stmmlib._backend import Backend
from stmmlib._mse_common import _draw_mvn, _onehot, _to_np


def gaussian_model_id(model):
    from stmmlib.registry import MODELS
    for model_id, spec in MODELS.items():
        if (spec.area == model.get("area_type")
                and spec.time == model.get("time_type")
                and spec.interaction == model.get("interaction", "none")
                and spec.time_scope == model.get("time_scope", "area_specific")):
            return model_id
    raise ValueError(f"MSE bootstrap: cannot map the fitted model back to a registry id: {model}")


def _gaussian_G_parts(theta, model, m, T, W, backend):
    from stmmlib.gaussian.covariance import build_G_area, build_G_time
    area_type = model["area_type"]
    if area_type in ("sar", "seblup_rao_yu"):
        Ga = build_G_area(area_type, theta["sigma2_v"], m, W, theta.get("rho", 0.0), backend)
    else:
        Ga = build_G_area("iid", theta["sigma2_v"], m, backend=backend)
    if model["time_type"] == "ar1":
        Gt = build_G_time("ar1", T, theta.get("sigma2_u", 1.0),
                          theta.get("sigma2_eta", 1.0), theta.get("phi", 0.0), backend)
    else:
        Gt = build_G_time("iid", T, theta.get("sigma2_u", 1.0), backend=backend)
    return _to_np(Ga), _to_np(Gt)


def refit_gaussian(fit, df, model_id, W, method):
    from stmmlib.stmm import stmm
    return stmm(data=df, model=model_id, family="gaussian", W=W, method=method,
                gpu="no", n_starts=1)


def simulate_gaussian(rng, theta, model, W, m, T, X, beta0, area_idx, time_idx):
    backend = Backend(gpu=False)
    Ga, Gt = _gaussian_G_parts(theta, model, m, T, W, backend)
    b_area = _draw_mvn(rng, Ga)
    G_cell = np.kron(np.eye(m, dtype=np.float64), Gt)
    b_cell = _draw_mvn(rng, G_cell)
    Za = _onehot(area_idx, m)
    Zc = _onehot(area_idx * T + time_idx, m * T)
    eta = X @ beta0 + Za @ b_area + Zc @ b_cell
    sigma2_e = float(theta.get("sigma2_e", 1.0))
    y_star = eta + rng.standard_normal(eta.size) * np.sqrt(sigma2_e)
    return eta, y_star


def predict_gaussian_rows(refit, Za, Zc):
    prep = refit["data"]
    backend = Backend(gpu=False)
    Ga, Gt = _gaussian_G_parts(refit["theta"], refit["model"], prep["m"], prep["T"],
                               refit["W"], backend)
    G_cell = np.kron(np.eye(prep["m"], dtype=np.float64), Gt)
    X2 = refit["X"]
    beta = np.asarray(refit["beta_hat"], dtype=np.float64).ravel()
    e = np.asarray(refit["y"], dtype=np.float64).ravel() - X2 @ beta
    V = np.asarray(refit["V"], dtype=np.float64)
    try:
        Vinv_e = np.linalg.solve(V, e)
    except np.linalg.LinAlgError:
        Vinv_e = np.linalg.solve(V + 1e-10 * np.eye(V.shape[0]), e)
    b_area = Ga @ Za.T @ Vinv_e
    b_cell = G_cell @ Zc.T @ Vinv_e
    return X2 @ beta + Za @ b_area + Zc @ b_cell
