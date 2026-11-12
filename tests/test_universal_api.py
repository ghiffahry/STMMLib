"""The package must not depend on how a user spells their columns, and the
family a model belongs to must be explicit."""
import numpy as np
import pytest

from conftest import fit_beta, fit_gauss
from stmmlib.stmm import stmm


def test_arbitrary_column_names_behave_identically(df_flat):
    """Respons and area/time in any language: same data, same numbers."""
    from stmmlib.stmm import stmm_beta  # noqa: F401  (public API surface)
    custom = df_flat.rename(columns={"y": "respons_proporsi",
                                     "area": "wilayah",
                                     "time": "periode"})
    default = fit_beta(df_flat, "m1_iid_iid")
    renamed = fit_beta(custom, "m1_iid_iid",
                        response_col="respons_proporsi", area_col="wilayah",
                        time_col="periode")
    assert float(default.log_lik) == pytest.approx(float(renamed.log_lik), rel=1e-9)
    assert np.allclose(default.beta, renamed.beta)
    assert list(renamed.coefficients.keys()) == ["(Intercept)", "x1"]


def test_beta_regression_is_not_available_for_gaussian(df_gauss):
    """beta_regression has no Gaussian structure; silently fitting an
    IID+IID Gaussian model instead would be a wrong answer, not a fit."""
    with pytest.raises(ValueError, match="beta-regression|beta_regression"):
        stmm(data=df_gauss, model="beta_regression", family="gaussian",
             gpu="no", n_starts=1)


def test_beta_regression_fits_as_beta(df_flat):
    fit = fit_beta(df_flat, "beta_regression")
    assert np.isfinite(float(fit.logLik))
