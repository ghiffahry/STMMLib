"""P7 beta_regression (fixed-effects, multi-predictor) tests."""
import numpy as np
import pandas as pd

from stmmlib.stmm import stmm


def test_beta_regression_multi_predictor():
    rng = np.random.default_rng(5)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = -1.0 + 0.6 * x1 - 0.4 * x2
    mu = np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98)
    phi = 20.0
    y = rng.beta(mu * phi, (1 - mu) * phi)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    r = stmm(data=df, model="beta_regression", family="beta", gpu="no")
    o = r._opt
    assert o["p"] == 3  # intercept + x1 + x2
    assert o["x_cols"] == ["x1", "x2"]
    labs = o["param_labels"]
    assert labs == ["(Intercept)", "x1", "x2", "log(phi)"]
    # recovery: coefficients near truth (n=300, phi=20 -> tight)
    assert abs(o["beta_hat"][1] - 0.6) < 0.15, o["beta_hat"]
    assert abs(o["beta_hat"][2] + 0.4) < 0.15, o["beta_hat"]
    assert np.all(np.isfinite(o["se"]))
    assert 0 <= o["pvalue"][1] <= 1


def test_beta_regression_intercept_only_fallback():
    rng = np.random.default_rng(6)
    n = 200
    y = rng.beta(20 * 0.3, 20 * 0.7, size=n)
    df = pd.DataFrame({"y": y})
    r = stmm(data=df, model="beta_regression", family="beta", gpu="no")
    o = r._opt
    assert o["p"] == 1 and o["x_cols"] == []
    assert np.isfinite(o["log_lik"])
