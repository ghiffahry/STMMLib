"""Numeric golden tests for the fitting engines.

Values were captured from stmmlib 0.2.2 on fixed, deterministic panels.
They are the safety net for refactoring: any change to the covariance
assembly or the Laplace pipeline must keep these numbers, unless the
change is intentional and the constants are updated in the same commit.
"""

import numpy as np
import pandas as pd
import pytest

from stmmlib.stmm import stmm


def _close(actual, expected, rel=1e-5, abs_=1e-7):
    assert actual == pytest.approx(expected, rel=rel, abs=abs_)


def _beta_panel():
    m, T = 10, 6
    n = m * T
    area = np.repeat(np.arange(m), T)
    time = np.tile(np.arange(T), m)
    rng = np.random.default_rng(11)
    a = rng.normal(0, 0.5, m)
    t = np.zeros((m, T))
    t[:, 0] = rng.normal(0, 0.4 / np.sqrt(1 - 0.6 ** 2), m)
    for j in range(1, T):
        t[:, j] = 0.6 * t[:, j - 1] + rng.normal(0, 0.4, m)
    x = rng.normal(size=n)
    eta = -1.0 + 0.6 * x + a[area] + t.ravel()
    mu = 1 / (1 + np.exp(-eta))
    y = rng.beta(20 * mu, 20 * (1 - mu))
    return pd.DataFrame({"y": y, "x1": x, "area": area, "time": time})


def _ring_w(m):
    W = np.zeros((m, m))
    for i in range(m):
        W[i, (i - 1) % m] = 1
        W[i, (i + 1) % m] = 1
    return W / W.sum(1, keepdims=True)


def _gauss_sar_panel():
    m, T = 10, 4
    n = m * T
    area = np.repeat(np.arange(m), T)
    time = np.tile(np.arange(T), m)
    rng = np.random.default_rng(20261113)
    W = _ring_w(m)
    u = np.linalg.solve(np.eye(m) - 0.4 * W, rng.normal(0, 0.5, m))
    g_time = rng.normal(0, 0.3, (m, T)).ravel()
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + u[area] + g_time + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x1": x, "area": area, "time": time}), W


def _gauss_ar1_panel():
    m, T = 10, 8
    n = m * T
    area = np.repeat(np.arange(m), T)
    time = np.tile(np.arange(T), m)
    rng = np.random.default_rng(20261113)
    a = rng.normal(0, 0.5, m)[area]
    e = rng.normal(0, 0.2, (m, T))
    t = np.zeros((m, T))
    for j in range(1, T):
        t[:, j] = 0.4 * t[:, j - 1] + e[:, j]
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + a + t.ravel() + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x1": x, "area": area, "time": time})


def test_golden_beta_m2():
    fit = stmm(data=_beta_panel(), model="m2_iid_ar1", family="beta",
               gpu="no", n_starts=1, max_iter=400, inner_max_iter=30)
    fe = fit.fixef()
    _close(float(fe["(Intercept)"]), -1.0472551336106792)
    _close(float(fe["x1"]), 0.6700976792968045)
    _close(float(fit.se_original[0]), 0.1771815732449099, rel=1e-4)
    _close(float(fit.se_original[1]), 0.09223363718582205, rel=1e-4)
    _close(float(fit.phi), 12.613661676181923)
    _close(float(fit.log_lik), 40.11460955518177, rel=1e-6)
    _close(float(fit.AIC), -68.22921911036354, rel=1e-5)
    vc = fit.VarCorr()
    _close(vc["area"]["sigma_v^2"], 0.2421733203072392)
    _close(vc["time"]["phi"], -0.47250337710015433)
    _close(vc["time"]["sigma_eta^2 (innovation)"], 0.012386449030715499)


def test_golden_gaussian_m3_sar():
    data, W = _gauss_sar_panel()
    fit = stmm(data=data, model="m3_sar_iid", family="gaussian", W=W,
               gpu="no", method="REML", n_starts=1)
    _close(float(fit["beta_hat"][0]), 0.987700456769413)
    _close(float(fit["beta_hat"][1]), 0.3709344553341112)
    _close(float(fit["se_beta"][0]), 0.40545234936106855, rel=1e-4)
    _close(float(fit["se_beta"][1]), 0.09650019948213204, rel=1e-4)
    _close(float(fit["theta"]["sigma2_e"]), 0.1646046090515253)
    _close(float(fit["theta"]["sigma2_v"]), 0.4876931940743246)
    _close(float(fit["theta"]["rho"]), 0.4427244464280934)
    _close(float(fit["theta"]["sigma2_u"]), 0.10994593753678435)
    _close(float(fit["log_lik"]), -42.34832984326952, rel=1e-6)
    _close(float(fit["AIC"]), 96.69665968653904, rel=1e-5)


def test_golden_gaussian_m2_ar1():
    fit = stmm(data=_gauss_ar1_panel(), model="m2_iid_ar1",
               family="gaussian", gpu="no", method="REML", n_starts=1)
    _close(float(fit["beta_hat"][0]), 0.8875764532257525)
    _close(float(fit["beta_hat"][1]), 0.6163935273277077)
    _close(float(fit["se_beta"][0]), 0.20406223848931443, rel=1e-4)
    _close(float(fit["se_beta"][1]), 0.07151709420370161, rel=1e-4)
    _close(float(fit["theta"]["sigma2_e"]), 0.23754013874702584)
    _close(float(fit["theta"]["sigma2_v"]), 0.35912490987966317)
    _close(float(fit["theta"]["sigma2_eta"]), 0.060820018666988725)
    _close(float(fit["theta"]["phi"]), 0.5126376980599631)
    _close(float(fit["log_lik"]), -78.65262064945551, rel=1e-6)
    _close(float(fit["BIC"]), 183.59740110695432, rel=1e-5)
