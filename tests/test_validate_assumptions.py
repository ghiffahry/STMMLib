"""P12 structural assumption validator tests."""
import numpy as np
import pandas as pd
import pytest

from stmmlib.diagnostics_suite import validate_assumptions


def _panel(m=6, T=3, drop_one=False):
    rng = np.random.default_rng(1)
    area = np.repeat(np.arange(m), T)
    time = np.tile(np.arange(T), m)
    y = rng.beta(20 * np.ones(m * T), 20 * (1 - np.ones(m * T) / 2))
    df = pd.DataFrame({"y": y, "x1": rng.normal(size=m * T),
                       "area": area, "time": time})
    if drop_one:
        df = df.drop(index=5).reset_index(drop=True)
    return df


def test_balanced_panel():
    v = validate_assumptions(_panel())
    assert v["panel"]["balanced"] and v["panel"]["obs_per_cell"] == 1.0
    v2 = validate_assumptions(_panel(drop_one=True))
    assert not v2["panel"]["balanced"]
    assert v2["panel"]["balanced_note"].startswith("UNBALANCED")


def test_w_validity():
    m = 6
    W = np.ones((m, m)); np.fill_diagonal(W, 0)
    W = W / W.sum(1, keepdims=True)
    v = validate_assumptions(_panel(m), W=W)
    w = v["W"]
    assert w["shape_ok"] and w["nonneg"] and w["zero_diag"] and w["connected"]
    assert abs(w["row_sum_max"] - 1.0) < 1e-12
    # row-standardized complete graph: eigenvalues 1 and -1/(m-1), so
    # max|eig| = 1.0 < 1/0.95 -> SAR stable with rho bound 0.95
    assert w["max_abs_eig"] == pytest.approx(1.0)
    assert w["sar_stable_0.95"]


def test_fit_flags():
    from conftest import fit_beta
    df = _panel()
    fit = fit_beta(df, "m1_iid_iid", max_iter=300)
    v = validate_assumptions(df, fit=fit)
    assert "fit" in v and v["fit"]["status"] in ("konvergen", "max_iter")
    assert v["fit"]["mu_near_0"] == 0 and v["fit"]["mu_near_1"] == 0
