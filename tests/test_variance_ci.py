"""P (profile/bootstrap CI varian) tests."""
import numpy as np
import pandas as pd

from conftest import fit_beta
from stmmlib.diagnostics_suite import (
    variance_ci_beta, bootstrap_variance_ci_beta,
)


def _panel(seed=7):
    m, T = 12, 4
    rng = np.random.default_rng(seed)
    area = np.repeat(np.arange(m), T)
    eta = -1.2 + 0.5 * rng.normal(size=m * T) + 0.5 * rng.normal(0, 1, m)[area]
    y = rng.beta(20 * np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98), 20)
    return pd.DataFrame({"y": y, "x1": rng.normal(size=m * T),
                         "area": area, "time": np.tile(np.arange(T), m)})


def test_profile_ci_variance():
    fit = fit_beta(_panel(), "m1_iid_iid", max_iter=400)
    wald = variance_ci_beta(fit, method="wald")
    prof = variance_ci_beta(fit, method="profile")
    assert "log_sigma_area" in wald and "log_sigma_area" in prof
    for lab, ci in prof.items():
        assert np.isfinite(ci["lo"]) and np.isfinite(ci["hi"]), lab
        assert ci["lo"] <= ci["sigma"] <= ci["hi"], lab
    # profile SE >= wald SE usually for variance components; just sanity width
    assert prof["log_sigma_area"]["hi"] > wald["log_sigma_area"]["lo"]


def test_bootstrap_ci_variance():
    fit = fit_beta(_panel(), "m1_iid_iid", max_iter=300)
    boot = bootstrap_variance_ci_beta(fit, n_boot=10, max_iter=300,
                                      inner_max_iter=10, seed=42)
    assert "log_sigma_area" in boot
    ci = boot["log_sigma_area"]
    assert ci["n_boot_ok"] >= 5
    assert np.isfinite(ci["lo"]) and np.isfinite(ci["hi"])
    assert ci["lo"] <= ci["hi"]
