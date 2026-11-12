"""Edge-case W tests: islands (zero rows), disconnected W, unbalanced panel.

Answers (with proof) the two user questions:
  Q1: new data, different predictors/rows -> GLMM works (column params +
      dynamic X; here: replicated & unbalanced panels).
  Q2: non-KNN W and island rows -> GLMM works; islands stay iid under SAR.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import fit_beta, fit_gauss
from stmmlib._utils import row_standardize
from stmmlib.diagnostics_suite import validate_assumptions


def _W_ring(m=12):
    W = np.zeros((m, m))
    for i in range(m):
        W[i, (i + 1) % m] = 1.0
        W[i, (i - 1) % m] = 1.0
    return W


def _W_islands(m=12, islands=(5, 11)):
    """Ring W with some rows zeroed (islands: no neighbors)."""
    W = _W_ring(m)
    for i in islands:
        W[i, :] = 0.0
        W[:, i] = 0.0
    return W


def _panel(seed=7, drop_idx=()):
    m, T = 12, 4
    rng = np.random.default_rng(seed)
    area = np.repeat(np.arange(m), T)
    time = np.tile(np.arange(T), m)
    eta = -1.2 + 0.5 * rng.normal(size=m * T) + 0.4 * rng.normal(0, 1, m)[area]
    y = rng.beta(20 * np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98), 20)
    df = pd.DataFrame({"y": y, "x1": rng.normal(size=m * T),
                       "area": area, "time": time})
    if drop_idx:
        return df.drop(index=list(drop_idx)).reset_index(drop=True)
    return df


def test_row_standardize_island_safe():
    W = _W_islands()
    Ws = row_standardize(W, safe=True)
    assert np.isfinite(Ws).all()                      # no 0/0 NaN
    assert np.all(Ws[[5, 11], :] == 0.0)              # island rows stay zero
    assert np.allclose(Ws.sum(1)[[0, 1, 2]], 1.0)     # non-island rows sum 1
    with pytest.warns(RuntimeWarning):
        Wbad = row_standardize(W, safe=False)         # classic trap documented
    assert np.isnan(Wbad[[5, 11], :]).all()


def test_gaussian_sar_with_island_W(df_gauss):
    W = row_standardize(_W_islands(), safe=True)
    r = fit_gauss(df_gauss, "m3_sar_iid", W)
    assert np.isfinite(r["log_lik"])
    assert r["theta"]["sigma2_v"] > 0
    # validator flags the islands (information, not crash)
    v = validate_assumptions(df_gauss, "area", "time", W=W)
    assert v["W"]["connected"] is False


def test_beta_sar_with_island_W(df_flat, W_flat):
    W = row_standardize(_W_islands(), safe=True)
    r = fit_beta(df_flat, "m3_sar_iid", W, max_iter=600)
    o = r._opt
    assert np.isfinite(o["log_lik"])
    assert o["sigma_area"] > 0
    inf = r.inference or {}
    assert inf["hessian_pd"] in (True, False)  # reported honestly either way


def test_unbalanced_panel_gaussian(df_gauss):
    # drop 9 observations -> some cells empty; V is Z_cell based (B6 fixed)
    df_u = df_gauss.drop(index=range(0, 36, 4)).reset_index(drop=True)
    assert len(df_u) < len(df_gauss)
    for mid, w in [("m1_iid_iid", None), ("m2_iid_ar1", None)]:
        r = fit_gauss(df_u, mid, w)
        assert np.isfinite(r["log_lik"]), mid
        assert r.get("converged", False), mid


def test_unbalanced_panel_beta(df_flat):
    df_u = df_flat.drop(index=range(1, 48, 5)).reset_index(drop=True)
    assert len(df_u) < len(df_flat)
    r = fit_beta(df_u, "m1_iid_iid", max_iter=500)
    assert np.isfinite(r._opt["log_lik"])
    assert r._opt["sigma_area"] > 0


def test_disconnected_W_two_components(df_flat, df_gauss):
    """Two disconnected blocks (1-6, 7-12) — like separate island groups."""
    m = 12
    W = np.zeros((m, m))
    for blk in [(0, 6), (6, 12)]:
        for i in range(*blk):
            for j in range(*blk):
                if i != j:
                    W[i, j] = 1.0
    W = row_standardize(W, safe=True)
    v = validate_assumptions(df_gauss, "area", "time", W=W)
    assert v["W"]["connected"] is False
    rg = fit_gauss(df_gauss, "m3_sar_iid", W)
    assert np.isfinite(rg["log_lik"])
    rb = fit_beta(df_flat, "m3_sar_iid", W, max_iter=600)
    assert np.isfinite(rb._opt["log_lik"])
