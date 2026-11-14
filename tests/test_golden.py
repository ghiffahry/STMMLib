"""Numeric golden tests for the fitting engines.

Values were captured from stmmlib 0.2.2 on fixed panels. RandomState is
used on purpose: its stream is frozen across NumPy versions, unlike the
newer Generator. These numbers are the safety net for refactoring: any
change to covariance assembly or the Laplace pipeline must keep them,
unless the change is intentional and the constants move in the same
commit.
"""

import numpy as np
import pandas as pd
import pytest

from stmmlib.stmm import stmm


def _close(actual, expected, rel=1e-5, abs_=1e-7):
    assert actual == pytest.approx(expected, rel=rel, abs=abs_)


def _close_var(actual, expected):
    """Variance components sit on a flat optimum, so different BLAS and
    LAPACK builds move the fifth significant digit. Beta, ll, and the
    information criteria stay tight; the scale parameters do not."""
    _close(actual, expected, rel=1e-3)


def _beta_panel():
    m, T = 10, 6
    n = m * T
    area = np.repeat(np.arange(m), T)
    time = np.tile(np.arange(T), m)
    rng = np.random.RandomState(4)
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
    rng = np.random.RandomState(1)
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
    rng = np.random.RandomState(10)
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
    _close(float(fe["(Intercept)"]), -1.0026732286577897)
    _close(float(fe["x1"]), 0.535643298237001)
    _close(float(fit.se_original[0]), 0.17891076274311982, rel=1e-4)
    _close(float(fit.se_original[1]), 0.0937823810468137, rel=1e-4)
    _close(float(fit.phi), 22.638198402404793, rel=1e-3)
    _close(float(fit.log_lik), 41.189261875923435, rel=1e-6)
    _close(float(fit.AIC), -70.37852375184687, rel=1e-5)
    vc = fit.VarCorr()
    _close_var(vc["area"]["sigma_v^2"], 0.12743356815937823)
    _close_var(vc["time"]["phi"], 0.6770953504137515)
    _close_var(vc["time"]["sigma_eta^2 (innovation)"], 0.14514398535764536)


def test_golden_gaussian_m3_sar():
    data, W = _gauss_sar_panel()
    fit = stmm(data=data, model="m3_sar_iid", family="gaussian", W=W,
               gpu="no", method="REML", n_starts=1)
    _close(float(fit["beta_hat"][0]), 0.9285991893236056)
    _close(float(fit["beta_hat"][1]), 0.6044106560724577)
    _close(float(fit["se_beta"][0]), 0.2642089961501095, rel=1e-4)
    _close(float(fit["se_beta"][1]), 0.1298346798792091, rel=1e-4)
    _close_var(float(fit["theta"]["sigma2_e"]), 0.25588580589227)
    _close_var(float(fit["theta"]["sigma2_v"]), 0.220575319009873)
    _close_var(float(fit["theta"]["rho"]), 0.39025481256130345)
    _close_var(float(fit["theta"]["sigma2_u"]), 0.14441227236486576)
    _close(float(fit["log_lik"]), -45.257483655355536, rel=1e-6)
    _close(float(fit["AIC"]), 102.51496731071107, rel=1e-5)


def test_golden_gaussian_m2_ar1():
    fit = stmm(data=_gauss_ar1_panel(), model="m2_iid_ar1",
               family="gaussian", gpu="no", method="REML", n_starts=1)
    _close(float(fit["beta_hat"][0]), 1.0306963259970152)
    _close(float(fit["beta_hat"][1]), 0.48629637898178313)
    _close(float(fit["se_beta"][0]), 0.13607228981793287, rel=1e-4)
    _close(float(fit["se_beta"][1]), 0.06448473997635551, rel=1e-4)
    _close_var(float(fit["theta"]["sigma2_e"]), 0.16800678571324348)
    _close_var(float(fit["theta"]["sigma2_v"]), 0.12379089820299206)
    _close_var(float(fit["theta"]["sigma2_eta"]), 0.1223495742764354)
    _close_var(float(fit["theta"]["phi"]), 0.42931590778846695)
    _close(float(fit["log_lik"]), -73.57848839767219, rel=1e-6)
    _close(float(fit["BIC"]), 173.44913660338767, rel=1e-5)
