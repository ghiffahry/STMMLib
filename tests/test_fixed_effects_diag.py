"""P11 fixed-effects & model tests: VIF, interior nested LRT, Wald table."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as _sps

from conftest import fit_beta
from stmmlib.diagnostics_suite import (
    vif_fixed_effects, nested_lrt_with_df, wald_table_beta,
)
from stmmlib.lrt import boundary_lrt


def test_vif_flags_collinear_predictor():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=200)
    x2 = 0.98 * x1 + 0.2 * rng.normal(size=200)  # near-duplicate
    x3 = rng.normal(size=200)
    X = np.column_stack([np.ones(200), x1, x2, x3])
    v = vif_fixed_effects(X, ["x1", "x2", "x3"])
    assert v["x2"]["vif"] > 10 and v["x2"]["flag"] == "HIGH"
    assert v["x3"]["vif"] < 2 and v["x3"]["flag"] == "ok"


def test_nested_lrt_interior():
    # interior H0: D=6 with df=2 -> p ~ 0.05
    r = nested_lrt_with_df(np.log(100.0) / 2 + 3.0, np.log(100.0) / 2, df=2)
    assert r["df"] == 2 and r["D"] == pytest.approx(6.0)
    assert r["p_value"] == pytest.approx(_sps.chi2.sf(6.0, 2), rel=1e-9)
    r0 = nested_lrt_with_df(10.0, 10.5, df=1)
    assert r0["D"] == 0.0 and r0["p_value"] == 1.0


def test_wald_table_beta(df_flat, W_flat):
    fit = fit_beta(df_flat, "m2_iid_ar1", max_iter=600)
    rows = wald_table_beta(fit)
    labs = [r["param"] for r in rows]
    assert labs[0] == "(Intercept)" and "log(phi)" in labs
    for r in rows:
        assert np.isfinite(r["estimate"])
        assert r["se"] > 0
        assert 0.0 <= r["p_value"] <= 1.0
    # consistency with boundary LRT plumbing: same fit objects work
    b = boundary_lrt(fit._opt["log_lik"], fit._opt["log_lik"] - 0.0)
    assert b["D_clamped"] >= 0
