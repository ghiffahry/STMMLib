"""Parametric bootstrap MSE for the area-level predictor.

Targets the design doc in docs/superpowers/specs/2026-11-13-mse-bootstrap-design.md.
The predictor is recomputed at each refit; the bootstrap target is the
realized area mean of the drawn random effects.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import fit_beta, fit_gauss
from stmmlib.mse_bootstrap import mse_ebp, mse_eblup


def test_mse_ebp_returns_area_frame(df_flat):
    fit = fit_beta(df_flat, "m2_iid_ar1")
    out = mse_ebp(fit, B=12, seed=0)
    assert list(out.columns) == ["area", "ebp", "mse", "rmse", "rel_rmse", "n_boot_ok"]
    assert len(out) == df_flat["area"].nunique()
    assert int(out["n_boot_ok"].iloc[0]) >= 8
    assert np.all(np.isfinite(out["mse"].to_numpy(dtype=float)))
    assert np.all(out["mse"].to_numpy(dtype=float) >= 0)
    assert np.allclose(out["rmse"].to_numpy(dtype=float),
                       np.sqrt(out["mse"].to_numpy(dtype=float)))


def test_mse_ebp_is_deterministic(df_flat):
    fit = fit_beta(df_flat, "m1_iid_iid")
    a = mse_ebp(fit, B=8, seed=123)
    b = mse_ebp(fit, B=8, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_mse_ebp_rejects_nested():
    class _FakeFit:
        _opt = {"des": {"blocks": {"prov": [0]}, "interaction": "none"}}

        def _structure(self):
            return {"interaction": "none"}

    with pytest.raises(NotImplementedError):
        mse_ebp(_FakeFit(), B=2)


def test_mse_ebp_rejects_interaction():
    class _FakeFit:
        _opt = {"des": {"blocks": {"area": [0]}, "interaction": "iid"}}

        def _structure(self):
            return {"interaction": "iid"}

    with pytest.raises(NotImplementedError):
        mse_ebp(_FakeFit(), B=2)


def test_mse_ebp_low_success_rate_raises(df_flat, monkeypatch):
    import stmmlib.mse.bootstrap as mb

    def _boom(*args, **kwargs):
        raise ValueError("forced failure")

    monkeypatch.setattr(mb, "beta_laplace_structural", _boom)
    fit = fit_beta(df_flat, "m1_iid_iid")
    with pytest.raises(RuntimeError):
        mse_ebp(fit, B=12, seed=0)


def test_mse_eblup_returns_area_frame(df_gauss):
    fit = fit_gauss(df_gauss, "m1_iid_iid")
    out = mse_eblup(fit, B=12, seed=0)
    assert list(out.columns) == ["area", "eblup", "mse", "rmse", "rel_rmse", "n_boot_ok"]
    assert len(out) == df_gauss["area"].nunique()
    assert int(out["n_boot_ok"].iloc[0]) >= 8
    assert np.all(np.isfinite(out["mse"].to_numpy(dtype=float)))
    assert np.all(out["mse"].to_numpy(dtype=float) >= 0)
    assert np.allclose(out["rmse"].to_numpy(dtype=float),
                       np.sqrt(out["mse"].to_numpy(dtype=float)))


def test_mse_eblup_is_deterministic(df_gauss):
    fit = fit_gauss(df_gauss, "m2_iid_ar1")
    a = mse_eblup(fit, B=8, seed=123)
    b = mse_eblup(fit, B=8, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_mse_eblup_rejects_nested():
    fit = {"model": {"area_type": "nested_iid", "interaction": "none"},
           "data": {"prov_index": np.array([0])}}
    with pytest.raises(NotImplementedError):
        mse_eblup(fit, B=2)


def test_mse_eblup_low_success_rate_raises(df_gauss, monkeypatch):
    import stmmlib.mse.bootstrap as mb

    def _boom(*args, **kwargs):
        raise ValueError("forced failure")

    monkeypatch.setattr(mb, "refit_gaussian", _boom)
    fit = fit_gauss(df_gauss, "m1_iid_iid")
    with pytest.raises(RuntimeError):
        mse_eblup(fit, B=12, seed=0)
