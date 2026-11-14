"""Regression tests for the 0.2.4 must-fix audit findings (1-8).

Each test fails on 0.2.3 and pins the corrected behaviour: sorted-panel
invariance, global time covariance, finite Hessian SEs, area-major spatial
interaction, Beta criteria slot/squash, honest prediction intervals,
intercept-SE delta method, and the V^{-1} BLUP.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import fit_beta, fit_gauss
from stmmlib._backend import Backend
from stmmlib.beta.precision import ar1_precision
from stmmlib.beta.solvers import beta_Q_gamma
from stmmlib.beta.structure import stmm_beta_structure
from stmmlib.gaussian.covariance import (
    build_G_area, build_G_interaction, build_V, G_ar1_core,
)
from stmmlib.gaussian.design import build_design_matrices, build_interaction_contrast
from stmmlib.gaussian.transforms import make_model, prepare_panel_data

BACKEND = Backend(gpu=False)


def _panel(m=5, T=5, seed=3):
    rng = np.random.default_rng(seed)
    area = np.repeat(np.arange(m), T)
    return pd.DataFrame({
        "area": area, "time": np.tile(np.arange(T), m),
        "x1": rng.normal(size=m * T),
    })


def _prepared(df):
    p = prepare_panel_data(df, "area", "time")
    p.update(build_design_matrices(p, BACKEND))
    p["W"] = None
    return p


def _dense(a):
    return np.asarray(a.toarray() if hasattr(a, "toarray") else a, dtype=float)


def test_gaussian_fit_invariant_to_row_order(df_gauss):
    shuffled = df_gauss.sample(frac=1.0, random_state=11).reset_index(drop=True)
    a = fit_gauss(df_gauss, "m1_iid_iid")
    b = fit_gauss(shuffled, "m1_iid_iid")
    np.testing.assert_allclose(a["beta_hat"], b["beta_hat"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(a["log_lik"], b["log_lik"], rtol=1e-8, atol=1e-8)


def test_global_time_scope_uses_time_design():
    p = _prepared(_panel())
    theta = {"sigma2_v": 0.5, "sigma2_eta": 0.4, "sigma2_e": 0.3, "phi": 0.6}
    model = make_model("iid", "ar1", "none", "global")
    V = build_V(p, model, theta, BACKEND)

    Za = _dense(p["Z_area"])
    Zt = _dense(p["Z_time"])
    Ga = _dense(build_G_area("iid", theta["sigma2_v"], p["m"], None, 0.0, BACKEND))
    Gt = _dense(G_ar1_core(p["T"], theta["sigma2_eta"], theta["phi"], BACKEND))
    expected = Za @ Ga @ Za.T + Zt @ Gt @ Zt.T + theta["sigma2_e"] * np.eye(p["n"])
    np.testing.assert_allclose(V, expected, rtol=1e-10, atol=1e-10)

    per_area = build_V(
        p, make_model("iid", "ar1", "none", "area_specific"), theta, BACKEND)
    assert not np.allclose(V, per_area)


def test_gaussian_hessian_se_available(df_gauss):
    r = fit_gauss(df_gauss, "m1_iid_iid")
    h = r["hessian"]
    assert h is not None
    se = h["se_theta"]
    assert np.isfinite(list(se.values())).all()
    assert se["sigma2_e"] > 0.0


def _spatial_qa(m, T, W, rho, backend=BACKEND):
    B = _dense(build_interaction_contrast(m, T, backend)["B_area"])
    A = np.eye(m) - rho * np.asarray(W, dtype=float)
    return B.T @ (A.T @ A) @ B


def test_gaussian_spatial_interaction_is_area_major():
    m, T = 5, 3
    rng = np.random.default_rng(4)
    W = rng.random((m, m))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(1, keepdims=True)
    sigma2, rho = 0.7, 0.3
    B = build_interaction_contrast(m, T, BACKEND)
    G = _dense(build_G_interaction(
        "spatial", sigma2, m, T, W, rho, 0.0, B, BACKEND))
    qa = _spatial_qa(m, T, W, rho)
    np.testing.assert_allclose(G, sigma2 * np.kron(qa, np.eye(T)),
                               rtol=1e-10, atol=1e-12)


def test_beta_spatial_interaction_is_area_major():
    m, T = 5, 3
    rng = np.random.default_rng(4)
    W = rng.random((m, m))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(1, keepdims=True)
    B = build_interaction_contrast(m, T, BACKEND)
    des = {"B_area": B["B_area"], "B_time": B["B_time"],
           "rho_bound_g": 0.9, "W_gamma": W, "phi_bound": 0.95}
    logs = np.array([0.0, 0.0, np.log(0.8)])
    raw = np.array([0.0, 1.0])
    s = {"interaction": "spatial"}
    Q = beta_Q_gamma(s, des, m, T, B["B_area"].shape[1] * T,
                     logs, raw, 3, 2, True, False, BACKEND)
    rho = 0.9 * np.tanh(1.0)
    qa = _spatial_qa(m, T, W, rho)
    expected = np.exp(-2.0 * logs[2]) * np.kron(np.linalg.inv(qa), np.eye(T))
    np.testing.assert_allclose(Q, expected, rtol=1e-10, atol=1e-12)


def test_beta_simulation_spatial_is_area_major():
    from stmmlib.beta.simulation import _gamma_re
    m, T = 4, 3
    rng = np.random.default_rng(5)
    W = rng.random((m, m))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(1, keepdims=True)
    B = _dense(build_interaction_contrast(m, T, BACKEND)["B_area"])
    rho, sigma = 0.25, 0.9
    e = rng.normal(size=(m - 1) * T)
    gamma, _ = _gamma_re("spatial", m, T, sigma, rho, 0.0, W,
                         type("R", (), {"normal": lambda self, a, b, size: e})())
    qa = B.T @ ((np.eye(m) - rho * W).T @ (np.eye(m) - rho * W)) @ B
    L = np.linalg.cholesky(qa + 1e-10 * np.eye(m - 1))
    np.testing.assert_allclose(gamma, sigma * np.kron(L, np.eye(T)) @ e,
                               rtol=1e-10, atol=1e-12)


def _criteria_opt(model):
    m, T = 5, 5
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "y": rng.uniform(0.05, 0.95, m * T),
        "x1": rng.normal(size=m * T),
        ".stmm_area": np.repeat(np.arange(m), T),
        ".stmm_time": np.tile(np.arange(T), m),
    })
    area_idx = df[".stmm_area"].to_numpy()
    time_idx = df[".stmm_time"].to_numpy()
    W = rng.random((m, m))
    np.fill_diagonal(W, 0.0)
    W /= W.sum(1, keepdims=True)
    from stmmlib.beta.design import stmm_beta_make_design
    des = stmm_beta_make_design(df, model, W, BACKEND)
    des["rho_bound"] = 0.9
    des["phi_bound"] = 0.95
    p = 2
    return {
        "X": np.column_stack([np.ones(m * T), df["x1"]]),
        "Z": des["Z"], "y": df["y"].to_numpy(), "beta_hat": np.array([0.2, -0.3]),
        "b_hat": np.zeros(des["Z"].shape[1]), "phi": 15.0,
        "logs": np.array([np.log(0.6), np.log(0.5)]),
        "raw_dep": np.array([0.4] if model.startswith("m2") else [0.3, 0.7]),
        "log_lik": -200.0, "model": model, "des": des, "W": W,
        "area_idx": area_idx, "time_idx": time_idx, "p": p,
    }


def _manual_criteria(opt):
    from scipy.special import betaln
    model, des, logs = opt["model"], opt["des"], opt["logs"]
    raw, X, Z = opt["raw_dep"], opt["X"], opt["Z"]
    s = stmm_beta_structure(model)
    m, T = int(des["m"]), int(des["T"])
    n_area, n_time = m, T if s["time_scope"] == "global" else m * T
    pb, rb = 0.95, 0.9
    if s["area"] == "sar":
        rho = rb * np.tanh(raw[0])
        W = np.asarray(opt["W"], float)
        A = np.eye(m) - rho * W
        Q_a = np.exp(-2.0 * logs[0]) * (A.T @ A)
        ph_i = 1
    else:
        Q_a = np.exp(-2.0 * logs[0]) * np.eye(n_area)
        ph_i = 0
    raw_phi = pb * np.tanh(raw[ph_i])
    TQ = ar1_precision(T, logs[1], raw_phi, BACKEND, phi_bound=pb)
    TQ = _dense(TQ)
    if s["time_scope"] == "global":
        Q_t = TQ
    else:
        Q_t = np.kron(np.eye(n_area), TQ)
    Q = np.block([[Q_a, np.zeros((n_area, n_time))],
                  [np.zeros((n_time, n_area)), Q_t]])
    mu = 1.0 / (1.0 + np.exp(-(X @ opt["beta_hat"] + Z @ opt["b_hat"])))
    w = opt["phi"] * mu * (1.0 - mu)
    A11 = X.T @ (X * w[:, None])
    A12 = X.T @ (Z * w[:, None])
    M22 = Z.T @ (Z * w[:, None])
    Hm = np.block([[A11, A12], [A12.T, M22 + Q]])
    Mm = np.block([[A11, A12], [A12.T, M22]])
    df_hat = float(np.trace(np.linalg.solve(Hm, Mm)))
    a = mu * opt["phi"]
    b = (1.0 - mu) * opt["phi"]
    y = opt["y"]
    ll = float(np.sum((a - 1.0) * np.log(y) + (b - 1.0) * np.log(1.0 - y)
                      - betaln(a, b)))
    return -2.0 * ll + 2.0 * df_hat


@pytest.mark.parametrize("model", ["m2_iid_ar1", "m4_sar_ar1"])
def test_beta_criteria_matches_engine_prior(model):
    from stmmlib.beta.criteria import beta_criteria
    opt = _criteria_opt(model)
    got = beta_criteria(opt)
    assert got is not None and np.isfinite(got["cAIC"])
    np.testing.assert_allclose(got["cAIC"], _manual_criteria(opt), rtol=1e-8)


def test_beta_predict_interval_refuses_missing_vcov(df_flat):
    fit = fit_beta(df_flat, "m1_iid_iid")
    fit._inf["vcov"] = None
    with pytest.raises(ValueError, match="vcov"):
        fit.predict_interval()


def test_beta_intercept_se_uses_full_delta(df_flat):
    rng = np.random.default_rng(21)
    n = len(df_flat)
    eta = -0.4 + 0.5 * (df_flat["x1"].to_numpy() * 0.3 + 3.0)
    df = df_flat.assign(
        x1=3.0 + 0.3 * rng.normal(size=n),
        y=rng.beta(20 * 1 / (1 + np.exp(-eta)), 20))
    fit = fit_beta(df, "m1_iid_iid")
    p = fit._opt["p"]
    V = np.asarray(fit._inf["vcov"], float)[:p, :p]
    scale = fit._opt["des"]["x_scale"]
    D = np.eye(p)
    for j, c in enumerate(fit._opt["des"]["x_cols"]):
        mn, sd = scale[c]
        D[0, j + 1] = -mn / sd
        D[j + 1, j + 1] = 1.0 / sd
    expected = np.sqrt(np.clip(np.diag(D @ V @ D.T), 0.0, None))
    np.testing.assert_allclose(fit.se_original, expected, rtol=1e-10, atol=1e-12)


def test_iid_blup_uses_generalized_least_squares(df_gauss):
    r = fit_gauss(df_gauss, "m1_iid_iid")
    theta, data = r["theta"], r["data"]
    e = r["y"] - r["X"] @ r["beta_hat"]
    Vinv_e = np.linalg.solve(r["V"], e)
    Za = _dense(data["Z_area"])
    Ga = _dense(build_G_area("iid", theta["sigma2_v"], r["m"], None, 0.0, BACKEND))
    np.testing.assert_allclose(r["blup"]["v_hat"], Ga @ Za.T @ Vinv_e,
                               rtol=1e-8, atol=1e-10)
    Zt = _dense(data["Z_time"])
    np.testing.assert_allclose(r["blup"]["u_hat"],
                               theta["sigma2_u"] * Zt.T @ Vinv_e,
                               rtol=1e-8, atol=1e-10)
