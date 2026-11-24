"""Response-scale marginal effects for Beta fixed effects."""
import numpy as np
import pytest
from scipy.special import expit

from conftest import fit_beta
from stmmlib.beta.marginal import marginal_effects


@pytest.fixture(scope="module")
def fit_m1(df_flat):
    return fit_beta(df_flat, "m1_iid_iid", max_iter=200)


def _manual(fit, at, b=None):
    opt = fit._opt
    b = np.asarray(fit.beta if b is None else b, dtype=float)
    X = np.asarray(opt["X"], dtype=float)
    cols = list(opt["des"]["x_cols"])
    sds = np.array([opt["des"]["x_scale"][c][1] for c in cols], dtype=float)
    if at == "average":
        mu = expit(X @ b)
        return (b[1:] / sds) * np.mean(mu * (1 - mu))
    mu0 = expit(b[0])
    return (b[1:] / sds) * mu0 * (1 - mu0)


@pytest.mark.parametrize("at", ["average", "mean"])
def test_estimate_matches_manual(fit_m1, at):
    me = marginal_effects(fit_m1, at=at)
    assert list(me["term"]) == list(fit_m1._opt["des"]["x_cols"])
    np.testing.assert_allclose(me["estimate"], _manual(fit_m1, at), rtol=1e-10)


@pytest.mark.parametrize("at", ["average", "mean"])
def test_delta_se_matches_numerical_jacobian(fit_m1, at):
    p = int(fit_m1._opt["p"])
    b = np.asarray(fit_m1.beta, dtype=float)
    J = np.zeros((p - 1, p))
    for k in range(p):
        bp, bm = b.copy(), b.copy()
        bp[k] += 1e-6
        bm[k] -= 1e-6
        J[:, k] = (_manual(fit_m1, at, bp) - _manual(fit_m1, at, bm)) / 2e-6
    V = np.asarray(fit_m1._inf["vcov"], dtype=float)[:p, :p]
    expected = np.sqrt(np.diag(J @ V @ J.T))
    me = marginal_effects(fit_m1, at=at)
    np.testing.assert_allclose(me["se"], expected, rtol=1e-4, atol=1e-8)


def test_scale_100_is_percentage_points(fit_m1):
    base = marginal_effects(fit_m1, at="average")
    pp = marginal_effects(fit_m1, at="average", scale=100.0)
    np.testing.assert_allclose(pp["estimate"], 100.0 * base["estimate"], equal_nan=True)
    np.testing.assert_allclose(pp["se"], 100.0 * base["se"], equal_nan=True)
    np.testing.assert_allclose(pp["ci_low"], 100.0 * base["ci_low"], equal_nan=True)
    np.testing.assert_allclose(pp["z"], base["z"], equal_nan=True)
    np.testing.assert_allclose(pp["p"], base["p"], equal_nan=True)


def test_missing_vcov_gives_nan_inference(fit_m1, monkeypatch):
    monkeypatch.setitem(fit_m1._inf, "vcov", None)
    me = marginal_effects(fit_m1, at="average")
    assert np.all(np.isfinite(me["estimate"]))
    assert np.all(np.isnan(me["se"]))
    assert np.all(np.isnan(me["p"]))


def test_bad_at_raises(fit_m1):
    with pytest.raises(ValueError, match="at"):
        marginal_effects(fit_m1, at="median")
