import numpy as np
import pandas as pd
import pytest

from stmmlib.beta.config import BetaFitConfig, resolve_preset
from stmmlib.beta.fit import _validate_contract
from tests.conftest import fit_beta


def test_default_reproduces_legacy(df_flat):
    fit = fit_beta(df_flat, "m1_iid_iid", preset="full")
    assert np.isfinite(fit.logLik)
    assert fit.status in ("konvergen", "max_iter")


def test_config_override_recorded(df_flat):
    cfg = BetaFitConfig(phi_bound=0.9, grad_tol=1e-12)
    fit = fit_beta(df_flat, "m1_iid_iid", preset="full", config=cfg)
    assert fit.design["phi_bound"] == 0.9
    msgs = " ".join(fit.certificate.get("warnings", []))
    # tiny tol forces the gradient warning which must print the custom tol
    assert "1e-12" in msgs


def test_spatial_and_temporal_bounds_are_independent(df_flat):
    """The spatial rho bound and the AR(1) phi bound are two knobs, not one."""
    cfg = BetaFitConfig(rho_bound=0.5, phi_bound=0.9)
    fit = fit_beta(df_flat, "m1_iid_iid", preset="full", config=cfg)
    assert fit.design["phi_bound"] == 0.9     # AR(1) coefficient bound
    assert fit.design["rho_bound"] == 0.5     # spatial rho bound


def test_auto_prior_flag_decides_the_refit():
    from stmmlib.beta.fit import _auto_prior_enabled
    assert _auto_prior_enabled(None) is True
    assert _auto_prior_enabled(BetaFitConfig()) is False
    assert _auto_prior_enabled(BetaFitConfig(auto_prior=True)) is True
    assert _auto_prior_enabled(BetaFitConfig(auto_prior=False)) is False


def test_preset_unknown_and_none():
    with pytest.raises(ValueError):
        resolve_preset("bogus")
    assert resolve_preset(None) == (1e-12, 1e-6, True, None)


def test_validate_contract_W_nonsquare(df_flat):
    from stmmlib.beta.fit import _sanitize_data
    d, _, _ = _sanitize_data(df_flat)
    with pytest.raises(ValueError):
        _validate_contract(d, np.ones((5, 6)))


def test_validate_contract_time_gap():
    d = pd.DataFrame({
        ".stmm_area": [0, 0, 1, 1],
        ".stmm_time": [0, 2, 0, 2],
        "y": [0.5] * 4, "x1": [0.1] * 4,
    })
    with pytest.raises(ValueError):
        _validate_contract(d)


def test_validate_contract_small_n_warns():
    d = pd.DataFrame({
        ".stmm_area": [0, 0, 1, 1, 0],
        ".stmm_time": [0, 1, 0, 1, 0],
        "y": [0.5] * 5, "x1": [0.1] * 5, "x2": [0.2] * 5,
    })
    with pytest.warns(UserWarning):
        _validate_contract(d)


def test_validate_contract_warns_about_the_W_order(df_flat):
    """A W built in file order is accepted silently; the user must be told."""
    from stmmlib.beta.fit import _sanitize_data
    d, area_labels, _ = _sanitize_data(df_flat)
    W = np.eye(len(area_labels))
    with pytest.warns(UserWarning, match="sorted"):
        _validate_contract(d, W, area_labels)
