"""P10 random-effects diagnostics tests."""
import numpy as np
import pandas as pd

from conftest import fit_beta
from stmmlib.diagnostics_suite import (
    random_effects_diagnostics_beta, format_random_effects_report,
)


def _panel(seed=7, area_sd=0.4):
    m, T = 12, 4
    rng = np.random.default_rng(seed)
    area = np.repeat(np.arange(m), T)
    eta = -1.2 + 0.5 * rng.normal(size=m * T) + area_sd * rng.normal(0, 1, m)[area]
    y = rng.beta(20 * np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98), 20)
    return pd.DataFrame({"y": y, "x1": rng.normal(size=m * T),
                         "area": area, "time": np.tile(np.arange(T), m)})


def test_random_effects_basic_structure():
    df = _panel()
    m = 12
    W = np.zeros((m, m))
    for i in range(m):
        W[i, (i + 1) % m] = 1.0
        W[i, (i - 1) % m] = 1.0
    W = W / W.sum(1, keepdims=True)
    fit = fit_beta(df, "m3_sar_iid", max_iter=600, W=W)
    d = random_effects_diagnostics_beta(fit)
    assert "area" in d["blocks"] and "area_time" in d["blocks"]
    a = d["blocks"]["area"]
    assert np.isfinite(a["shapiro_p"]) and np.isfinite(a["moran_I"])
    assert -1.0 <= a["moran_I"] <= 1.0
    assert "acf1" in d["blocks"]["area_time"]
    assert np.isfinite(d["shrinkage_corr_area"])
    # variance CIs present and contain the point estimate
    for lab, ci in d["variance_cis"].items():
        assert ci["ci"][0] <= ci["sigma"] <= ci["ci"][1], lab
    rep = format_random_effects_report(d)
    assert "Moran" in rep and "shrinkage" in rep


def test_random_effects_moran_detects_spatial_signal():
    """Area effects from a smooth spatial field -> Moran I >> E[I].

    Smoothing via (I+W)/2 iteration: all eigenvalues of (I+W)/2 lie in [0,1],
    so high frequencies die and the surviving field is positively
    autocorrelated (a plain 0.9*W iteration keeps the alternating
    eigen-component and yields NEGATIVE Moran).
    """
    m, T = 12, 4
    rng = np.random.default_rng(3)
    area = np.repeat(np.arange(m), T)
    W = np.zeros((m, m))
    for i in range(m):
        W[i, (i + 1) % m] = 1.0
        W[i, (i - 1) % m] = 1.0
    W = W / W.sum(1, keepdims=True)
    u = rng.normal(0, 1, m)
    for _ in range(6):
        u = (u + W @ u) / 2.0
    eta = -1.2 + 2.0 * u[area]
    y = rng.beta(20 * np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98), 20)
    df = pd.DataFrame({"y": y, "x1": rng.normal(size=m * T),
                       "area": area, "time": np.tile(np.arange(T), m)})
    fit = fit_beta(df, "m1_iid_iid", max_iter=600, W=W)
    d = random_effects_diagnostics_beta(fit)
    a = d["blocks"]["area"]
    assert a["moran_I"] > a["moran_EI"] + 0.2, (a["moran_I"], a["moran_EI"])
