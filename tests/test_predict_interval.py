"""P5 prediction interval tests (beta + gaussian)."""
import numpy as np
import pandas as pd

from conftest import fit_beta, fit_gauss
from stmmlib.diagnostics_suite import prediction_intervals_gaussian


def _panel(seed=7):
    m, T = 12, 4
    rng = np.random.default_rng(seed)
    area = np.repeat(np.arange(m), T)
    eta = -1.2 + 0.5 * rng.normal(size=m * T) + 0.4 * rng.normal(0, 1, m)[area]
    y = rng.beta(20 * np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98), 20)
    return pd.DataFrame({"y": y, "x1": rng.normal(size=m * T),
                         "area": area, "time": np.tile(np.arange(T), m)})


def test_beta_pi_structure_and_coverage():
    df = _panel()
    fit = fit_beta(df, "m2_iid_ar1", max_iter=600)
    pi = fit.predict_interval()
    assert pi["lower"].min() >= 0 and pi["upper"].max() <= 1
    assert np.all(pi["lower"] <= pi["mean"]) and np.all(pi["mean"] <= pi["upper"])
    assert pi["width_mean"] > 0
    # obs-noise interval must be wider than the mean-only one
    pi_mean = fit.predict_interval(include_obs=False)
    assert pi["width_mean"] > pi_mean["width_mean"]
    # in-sample coverage with obs noise should be high on a fitting model
    y = np.asarray(df["y"])
    cov = float(np.mean((y >= pi["lower"]) & (y <= pi["upper"])))
    assert cov >= 0.8, cov
    # level 0.90 narrower than 0.95
    pi90 = fit.predict_interval(level=0.90)
    assert pi90["width_mean"] < pi["width_mean"]


def test_gaussian_pi_structure(df_gauss, W_flat):
    r = fit_gauss(df_gauss, "m2_iid_ar1")
    pi = prediction_intervals_gaussian(r)
    assert np.all(pi["lower"] <= pi["mean"]) and np.all(pi["mean"] <= pi["upper"])
    assert pi["width_mean"] > 0
    assert np.isfinite(pi["se_pred"]).all()
    y = np.asarray(df_gauss["y"])
    cov = float(np.mean((y >= pi["lower"]) & (y <= pi["upper"])))
    assert cov >= 0.8, cov
