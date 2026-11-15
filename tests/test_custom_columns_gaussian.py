"""Arbitrary column names for Gaussian models.

The public signature promises configurable response/area/time/prov/kab names.
The flat and SEBLUP paths used to ignore prov_col/kab_col, so a frame carrying
custom-named grouping columns leaked those text columns into X (opaque numpy
dtype error, or a silently wrong design). These tests pin the forwarding and
the named predictor error.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import fit_gauss


def _panel(m=6, T=4, seed=13):
    rng = np.random.default_rng(seed)
    n = m * T
    area = np.repeat(np.arange(m), T)
    time = np.tile(np.arange(T), m)
    eta = 0.3 * rng.normal(size=n)
    return pd.DataFrame({
        "respon": eta + rng.normal(0.0, 0.5, size=n),
        "kec": [f"A{i:02d}" for i in area],
        "thn": 2010 + time,
        "belanja": rng.normal(size=n),
        "ipm": rng.normal(size=n),
        "provinsi": [f"P{i:02d}" for i in area // 2],
        "kabupaten": [f"K{i:03d}" for i in area],
    })


def _cols(**extra):
    kw = dict(response_col="respon", area_col="kec", time_col="thn",
              prov_col="provinsi", kab_col="kabupaten")
    kw.update(extra)
    return kw


def test_flat_gaussian_skips_custom_grouping_columns():
    df = _panel()
    r = fit_gauss(df, "m1_iid_iid", **_cols())
    assert r["x_cols"] == ["belanja", "ipm"]
    assert np.isfinite(r["log_lik"])


def test_seblup_skips_custom_grouping_columns():
    df = _panel()
    r = fit_gauss(df, "m5_seblup_rao_yu_it", W=np.eye(6) * 0.0, **_cols())
    assert r["x_cols"] == ["belanja", "ipm"]
    assert np.isfinite(r["log_lik"])


@pytest.mark.parametrize("model", ["m1_iid_iid", "m5_seblup_rao_yu_it"])
def test_non_numeric_predictor_names_the_column(model):
    df = _panel().assign(kode_area=["k%02d" % i for i in range(len(_panel()))])
    W = np.eye(6) * 0.0 if model.startswith("m5") else None
    with pytest.raises(ValueError, match="kode_area"):
        fit_gauss(df, model, W=W, **_cols())
