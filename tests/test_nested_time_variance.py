"""M6/M13 must give the IID time block its own variance.

The registry declares a time variance for these models. Sharing sigma2_kab
between the kab effect and the time effect silently fits a different model,
so these tests pin the separate parameter in both families.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import fit_beta
from stmmlib._backend import Backend
from stmmlib.gaussian.covariance import build_V_nested
from stmmlib.gaussian.design import build_Z_nested
from stmmlib.gaussian.transforms import nested_layout, prepare_panel_data
from stmmlib.registry import get_model_spec


@pytest.mark.parametrize("model_id", ["m6_nested_iid_iid", "m13_nested_iid_iid_int"])
def test_nested_iid_layout_matches_registry(model_id):
    spec = get_model_spec(model_id)
    layout = nested_layout(spec.interaction)
    assert len(layout) == spec.n_params_normal
    assert "sigma2_u" in layout


def test_gaussian_nested_time_variance_is_free():
    m, T = 6, 3
    area = np.repeat(np.arange(m), T)
    n = m * T
    data = pd.DataFrame({
        "y": np.zeros(n), "x1": np.zeros(n), "area": area,
        "time": np.tile(np.arange(T), m),
        "prov": np.repeat(np.arange(2), 3 * T), "kab": area,
    })
    backend = Backend(gpu=False)
    prepared = prepare_panel_data(data, "area", "time",
                                  prov_col="prov", kab_col="kab")
    prepared.update(build_Z_nested(prepared, backend))
    theta = {"sigma2_e": 0.0, "sigma2_prov": 0.0, "sigma2_kab": 0.0,
             "sigma2_u": 1.0}
    V = build_V_nested(prepared, theta, "none", backend)
    assert np.allclose(V, np.eye(prepared["n"]))


def test_beta_nested_time_variance_is_free(df_nested):
    fit = fit_beta(df_nested, "m6_nested_iid_iid")
    opt = fit._opt
    assert len(opt["logs"]) == 3
    vc = fit.VarCorr()
    assert vc["time"]["sigma^2"] == pytest.approx(float(np.exp(2 * opt["logs"][2])))
    assert not np.isclose(vc["time"]["sigma^2"], vc["kab"]["sigma_kab^2"],
                          rtol=1e-3)
