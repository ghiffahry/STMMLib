"""The Gaussian fixed-effect LRT: the statistic, the certificate, the API."""
import numpy as np
import pytest

from conftest import fit_gauss


def test_the_lrt_finds_the_restricted_maximum(df_gauss):
    """The restricted likelihood must be the maximum of the reduced model.

    The LRT re-fits the reduced model from two starts; if it stopped short,
    D would be too large and the p-value too small. The check is an
    independent fit of the same reduced model: the LRT must be at least as
    good, and its D at least as small.
    """
    fit = fit_gauss(df_gauss, "m1_iid_iid", None, inference="lrt",
                    lrt_terms=["x1"])
    row = fit["lrt"]["terms"]["x1"]
    assert row["status"] == "ok", row
    reduced = fit_gauss(df_gauss.drop(columns=["x1"]), "m1_iid_iid", None)
    assert row["loglik_reduced"] >= reduced["log_lik"] - 1e-6
    assert row["D"] <= 2 * (fit["lrt"]["loglik_full"] - reduced["log_lik"]) + 1e-6


def test_the_statistic_is_twice_the_likelihood_difference(df_gauss):
    fit = fit_gauss(df_gauss, "m1_iid_iid", None, inference="lrt",
                    lrt_terms=["x1"])
    row = fit["lrt"]["terms"]["x1"]
    assert row["D"] == pytest.approx(
        2 * (row["loglik_full"] - row["loglik_reduced"]), rel=1e-9)
    assert row["df"] == 1
    assert 0.0 <= row["p"] <= 1.0


def test_the_statistic_equals_the_wald_when_the_variance_is_fixed(df_gauss):
    """With the variance components held fixed the profile is quadratic,
    so D must equal z^2 exactly.

    The re-fitted LRT and the Wald test differ in a mixed model because
    the restricted model re-estimates the variance components. Pinning
    them turns the profile into a parabola and the two must agree, which
    is what proves the machinery.
    """
    from stmmlib._backend import Backend
    from stmmlib.gaussian.design import build_design_matrices
    from stmmlib.gaussian.gls import logLik_ML
    from stmmlib.gaussian.models import make_model
    from stmmlib.gaussian.transforms import prepare_panel_data

    fit = fit_gauss(df_gauss, "m1_iid_iid", None, inference="lrt",
                    lrt_terms=["x1"])
    row = fit["lrt"]["terms"]["x1"]
    assert row["status"] == "ok", row

    backend = Backend(gpu=False)
    prepared = prepare_panel_data(df_gauss, "area", "time")
    prepared.update(build_design_matrices(prepared, backend))
    model = make_model("iid", "iid", "none", "area_specific")
    y = np.asarray(fit["y"], dtype=float)
    X = np.asarray(fit["X"], dtype=float)
    theta = fit["theta"]
    j = list(fit["x_cols"]).index("x1")
    X_red = np.delete(X, j + 1, axis=1)
    ll_full = logLik_ML(theta, y, X, prepared, model, backend)
    ll_red = logLik_ML(theta, y, X_red, prepared, model, backend)
    z = float(np.asarray(fit["beta_hat"])[j + 1]
              / np.asarray(fit["se_beta"])[j + 1])
    assert 2 * (ll_full - ll_red) == pytest.approx(z ** 2, rel=1e-6)


def test_an_untested_term_carries_no_p(df_gauss):
    fit = fit_gauss(df_gauss, "m1_iid_iid", None, inference="lrt",
                    lrt_terms=["x1"])
    inf = fit["inference"]
    j = list(fit["x_cols"]).index("x1") + 1
    assert np.isfinite(inf["pvalue"][j])
    assert np.isnan(inf["pvalue"][0])          # the intercept is never tested
    assert np.all(np.isfinite(inf["pvalue_wald"]))


def test_the_wald_p_value_stays_available(df_gauss):
    from stmmlib.gaussian.summary import wald_pvalues
    plain = fit_gauss(df_gauss, "m1_iid_iid", None)
    lrt = fit_gauss(df_gauss, "m1_iid_iid", None, inference="lrt",
                     lrt_terms=["x1"])
    assert lrt["inference"]["pvalue_wald"] == pytest.approx(
        wald_pvalues(plain), rel=1e-9)


def test_a_failed_refit_gives_no_p(df_gauss, monkeypatch):
    """A refit that blows up must not produce a p-value."""
    import stmmlib.gaussian.optimization as opt_mod

    real = opt_mod.run_optimization

    def boom(y, X, *a, **kw):
        if X.shape[1] == 1:          # the reduced design (intercept only)
            raise RuntimeError("the refit blew up")
        return real(y, X, *a, **kw)

    monkeypatch.setattr(opt_mod, "run_optimization", boom)
    fit = fit_gauss(df_gauss, "m1_iid_iid", None, inference="lrt",
                    lrt_terms=["x1"])
    row = fit["lrt"]["terms"]["x1"]
    assert row["status"] == "failed"
    assert row["p"] is None


def test_reml_is_refused(df_gauss):
    """REML likelihoods are not comparable across designs."""
    from stmmlib.gaussian.lrt_fixed import lrt_fixed_gaussian
    fit = fit_gauss(df_gauss, "m1_iid_iid", None)
    with pytest.raises(ValueError, match="ML"):
        lrt_fixed_gaussian(fit, ["x1"], method="REML")


def test_the_intercept_cannot_be_tested(df_gauss):
    from stmmlib.gaussian.lrt_fixed import lrt_fixed_gaussian
    fit = fit_gauss(df_gauss, "m1_iid_iid", None)
    with pytest.raises(ValueError, match="intercept"):
        lrt_fixed_gaussian(fit, ["(Intercept)"])


def test_the_fit_is_identical_with_and_without_the_lrt(df_gauss):
    """The LRT must not touch the fit it is attached to."""
    plain = fit_gauss(df_gauss, "m1_iid_iid", None)
    lrt = fit_gauss(df_gauss, "m1_iid_iid", None, inference="lrt",
                    lrt_terms=["x1"])
    assert lrt["log_lik"] == pytest.approx(plain["log_lik"], rel=1e-12)
    assert lrt["beta_hat"] == pytest.approx(plain["beta_hat"], rel=1e-12)
    assert lrt["se_beta"] == pytest.approx(plain["se_beta"], rel=1e-12)


def test_the_summary_shows_the_variable_names(df_gauss):
    """A table with beta_0 instead of the name is not readable."""
    from stmmlib.output_format_normal import format_normal
    txt = format_normal(fit_gauss(df_gauss, "m1_iid_iid", None))
    assert "x1" in txt
    assert "beta_0" not in txt


def test_the_lrt_runs_for_the_nested_family(df_nested):
    fit = fit_gauss(df_nested, "m6_nested_iid_iid", None, inference="lrt",
                    lrt_terms=["x1"])
    row = fit["lrt"]["terms"]["x1"]
    assert row["status"] == "ok", row
    assert row["p"] is not None


def test_satterthwaite_says_it_uses_residual_df(df_gauss):
    """The satterthwaite branch is residual df; it must say so."""
    from stmmlib.gaussian.summary import fixed_table, wald_pvalues
    fit = fit_gauss(df_gauss, "m1_iid_iid", None)
    with pytest.warns(UserWarning, match="satterthwaite"):
        wald_pvalues(fit, "satterthwaite")
    with pytest.warns(UserWarning, match="satterthwaite"):
        fixed_table(fit, "satterthwaite")
