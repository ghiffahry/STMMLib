"""P9 residual diagnostic suite tests."""
import numpy as np
import pandas as pd

from conftest import fit_beta
from stmmlib.diagnostics_suite import (
    residual_diagnostics_beta, residual_diagnostics_gaussian,
    format_residual_report,
)


def _panel(seed=7, n_extra_shift=0.0):
    m, T = 12, 4
    rng = np.random.default_rng(seed)
    area = np.repeat(np.arange(m), T)
    eta = -1.2 + 0.5 * rng.normal(size=m * T) + 0.4 * rng.normal(0, 1, m)[area]
    eta = eta + n_extra_shift
    y = rng.beta(20 * np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98), 20)
    return pd.DataFrame({"y": y, "x1": rng.normal(size=m * T),
                         "area": area, "time": np.tile(np.arange(T), m)})


def test_beta_diagnostics_correct_model():
    df = _panel()
    fit = fit_beta(df, "m2_iid_ar1", max_iter=600)
    d = residual_diagnostics_beta(fit)
    assert d["family"] == "beta"
    n = d["normality"]
    assert np.isfinite(n["shapiro_p"]) and np.isfinite(n["ks_p"])
    # correct model -> quantile residuals consistent with N(0,1)
    assert n["shapiro_p"] > 0.001, n
    e = d["envelope"]
    assert e["pct_outside"] < 15.0, e  # ~5% expected, generous bound
    assert "cooks_d_max" in d.get("influence", {})
    rep = format_residual_report(d)
    assert "envelope" in rep and "influence" in rep


def test_beta_diagnostics_detects_misfit():
    """Residuals evaluated on a SHIFTED response with the unchanged fit
    (y_override) must leak the envelope far more than the honest fit."""
    df = _panel()
    fit = fit_beta(df, "m2_iid_ar1", max_iter=600)
    d_ok = residual_diagnostics_beta(fit)
    y_shift = np.clip(np.asarray(df["y"]) + 0.15, 1e-6, 1 - 1e-6)
    d_bad = residual_diagnostics_beta(fit, y_override=y_shift)
    assert d_bad["envelope"]["pct_outside"] > 30.0, d_bad["envelope"]
    assert d_bad["envelope"]["pct_outside"] > d_ok["envelope"]["pct_outside"]


def test_gaussian_diagnostics_basic(df_gauss):
    from conftest import fit_gauss
    r = fit_gauss(df_gauss, "m2_iid_ar1")
    d = residual_diagnostics_gaussian(r)
    assert d["family"] == "gaussian"
    assert np.isfinite(d["normality"]["shapiro_p"])
    assert np.isfinite(d["heteroskedasticity"]["bp_p"])
