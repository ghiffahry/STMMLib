"""Weak-prior MAP penalty (opt-in) + start sanitization."""
import numpy as np
import pytest

from stmmlib.beta.config import BetaFitConfig
from tests.conftest import fit_beta


def test_default_off_identical(df_flat):
    assert BetaFitConfig().phi_prior_sd is None
    assert BetaFitConfig().dep_prior_sd is None
    a = fit_beta(df_flat, "m1_iid_iid", preset="fast")
    b = fit_beta(df_flat, "m1_iid_iid", preset="fast", config=BetaFitConfig())
    assert np.isfinite(a.logLik) and np.isfinite(b.logLik)
    assert a.logLik == pytest.approx(b.logLik)


def test_prior_on_runs(df_flat):
    cfg = BetaFitConfig(phi_prior_mean=6.9, phi_prior_sd=1.0, dep_prior_sd=1.0)
    fit = fit_beta(df_flat, "m1_iid_iid", preset="fast", config=cfg)
    assert np.isfinite(fit.logLik)
    assert fit.design  # des terekam
    assert fit.status in ("konvergen", "max_iter", "invalid")


def test_prior_validation():
    with pytest.raises(ValueError):
        BetaFitConfig(phi_prior_mean=6.9)
    with pytest.raises(ValueError):
        BetaFitConfig(phi_prior_sd=0.0, phi_prior_mean=6.9)
    with pytest.raises(ValueError):
        BetaFitConfig(dep_prior_sd=-1.0)


def test_nonfinite_start_invalid(df_flat):
    for bad in (np.nan, np.inf):
        fit = fit_beta(df_flat, "m1_iid_iid", preset="fast",
                       start_theta=np.full(5, bad))
        assert fit.status == "invalid"


def test_penalized_deterministic(df_flat):
    cfg = BetaFitConfig(phi_prior_mean=6.9, phi_prior_sd=1.0)
    kw = dict(preset="fast", config=cfg, n_starts=2, seed=42)
    assert fit_beta(df_flat, "m1_iid_iid", **kw).logLik == \
        fit_beta(df_flat, "m1_iid_iid", **kw).logLik
