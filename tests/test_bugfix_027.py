"""Regression tests for the 0.2.7 audit fixes.

Each test pins one bug that changed a number or made an inference invalid.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from conftest import fit_beta, fit_gauss


# --- A1: Gaussian AIC/BIC must use the ML likelihood -----------------------

def test_gaussian_reml_aic_bic_use_ml(df_gauss):
    from stmmlib._backend import Backend
    from stmmlib.gaussian.gls import logLik_ML
    r = fit_gauss(df_gauss, "m1_iid_iid")  # default REML
    assert str(r["method"]).upper() == "REML"
    ll_ml = float(logLik_ML(r["theta"], r["y"], r["X"], r["data"],
                            r["model"], Backend(gpu="no")))
    k, n = r["k"], r["n"]
    assert r["AIC"] == pytest.approx(-2 * ll_ml + 2 * k)
    assert r["BIC"] == pytest.approx(-2 * ll_ml + np.log(n) * k)
    # the old value came from the REML logLik and is a different number
    assert r["AIC"] != pytest.approx(-2 * r["log_lik"] + 2 * k)


def test_gaussian_nested_reml_aic_bic_use_ml(df_nested):
    from stmmlib._backend import Backend
    from stmmlib.gaussian.gls import logLik_nested
    r = fit_gauss(df_nested, "m6_nested_iid_iid")
    ll_ml = float(logLik_nested(r["theta"], r["y"], r["X"], r["data"],
                                "none", "ML", Backend(gpu="no")))
    k, n = r["k"], r["n"]
    assert r["AIC"] == pytest.approx(-2 * ll_ml + 2 * k)
    assert r["BIC"] == pytest.approx(-2 * ll_ml + np.log(n) * k)


def test_gaussian_seblup_reml_aic_bic_use_ml(df_flat, W_flat):
    from stmmlib._backend import Backend
    from stmmlib.gaussian.gls import logLik_seblup5
    r = fit_gauss(df_flat, "m5_seblup_rao_yu_it", W=W_flat)
    ll_ml = float(logLik_seblup5(r["theta"], r["y"], r["X"], r["data"],
                                 "none", "sar", "ML", Backend(gpu="no")))
    k, n = r["k"], r["n"]
    assert r["AIC"] == pytest.approx(-2 * ll_ml + 2 * k)
    assert r["BIC"] == pytest.approx(-2 * ll_ml + np.log(n) * k)


def test_gaussian_nested_ar1_reml_aic_bic_use_ml(df_nested):
    from stmmlib._backend import Backend
    from stmmlib.gaussian.gls import logLik_nested_ar1
    r = fit_gauss(df_nested, "m7_nested_ar1")
    ll_ml = float(logLik_nested_ar1(r["theta"], r["y"], r["X"], r["data"],
                                    "none", "ML", Backend(gpu="no")))
    k, n = r["k"], r["n"]
    assert r["AIC"] == pytest.approx(-2 * ll_ml + 2 * k)
    assert r["BIC"] == pytest.approx(-2 * ll_ml + np.log(n) * k)


# --- A2: p-values must not underflow to zero -------------------------------

def test_wald_p_does_not_underflow_for_large_z():
    from stmmlib.beta.inference import compute_stbl_inference
    n_params = 5  # intercept, x1, logphi, log_sigma_area, log_sigma_time
    opt = {
        "p": 2, "ksd": 2, "k": n_params, "n": 20,
        "beta_hat": np.array([0.0, 12.0]), "logphi": 0.0,
        "logs": np.array([0.0, 0.0]), "raw_dep": np.array([]),
        "model": "m1_iid_iid", "des": {"x_cols": ["x1"]},
        "optim": {"hessian": np.eye(n_params)},
    }
    inf = compute_stbl_inference(opt)
    p = float(inf["pvalue"][1])
    assert p > 0.0
    assert p == pytest.approx(2 * norm.sf(12.0))


def test_boundary_mixture_p_does_not_underflow():
    from stmmlib.lrt import mixture_chi2_pvalue
    p = mixture_chi2_pvalue(80.0)
    assert p > 0.0
    assert p < 1e-15


# --- A5/A6: prediction on the original scale, correct obs variance ----------

def test_predict_accepts_original_scale(df_flat):
    fit = fit_beta(df_flat, "m1_iid_iid")
    opt = fit._opt
    des = opt["des"]
    X_std = np.asarray(opt["X"], float)
    X_raw = X_std.copy()
    for j, c in enumerate(des["x_cols"]):
        mn, sd = des["x_scale"][c]
        X_raw[:, j + 1] = X_std[:, j + 1] * sd + mn
    assert np.allclose(fit.predict(X_raw), fit.mu_hat)
    assert np.allclose(fit.predict(X_std, standardized=True), fit.mu_hat)
    fixed = 1.0 / (1.0 + np.exp(-(X_std @ fit.beta)))
    assert np.allclose(fit.predict(X_raw, include_random=False), fixed)


def test_predict_interval_accepts_original_scale(df_flat):
    fit = fit_beta(df_flat, "m1_iid_iid")
    opt = fit._opt
    des = opt["des"]
    X_std = np.asarray(opt["X"], float)
    X_raw = X_std.copy()
    for j, c in enumerate(des["x_cols"]):
        mn, sd = des["x_scale"][c]
        X_raw[:, j + 1] = X_std[:, j + 1] * sd + mn
    pi_in = fit.predict_interval()
    pi_raw = fit.predict_interval(X_raw)
    pi_std = fit.predict_interval(X_std, standardized=True)
    assert np.allclose(pi_raw["mean"], pi_in["mean"])
    assert np.allclose(pi_raw["se_logit"], pi_in["se_logit"])
    assert np.allclose(pi_std["se_logit"], pi_in["se_logit"])


def test_predict_interval_obs_variance_matches_delta_method(df_flat):
    fit = fit_beta(df_flat, "m1_iid_iid")
    pi_obs = fit.predict_interval()
    pi_mean = fit.predict_interval(include_obs=False)
    mu = pi_obs["mean"]
    extra = pi_obs["se_logit"] ** 2 - pi_mean["se_logit"] ** 2
    expected = 1.0 / ((1.0 + fit.phi) * mu * (1.0 - mu))
    assert np.allclose(extra, expected, rtol=1e-8)


# --- A3: LRT p is not chi2 when the objective is penalized ------------------

def _tiny_panel(seed=11, m=6, T=5):
    rng = np.random.default_rng(seed)
    area = np.repeat(np.arange(m), T)
    x1 = rng.normal(size=m * T)
    x2 = rng.normal(size=m * T)
    mu = 1.0 / (1.0 + np.exp(-(-1.0 + 0.7 * x1 - 0.4 * x2)))
    phi = 25.0
    y = rng.beta(phi * mu, phi * (1 - mu))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2,
                         "area": area, "time": np.tile(np.arange(T), m)})


def test_lrt_penalized_fit_reports_no_chi2_p():
    from stmmlib.beta.config import BetaFitConfig
    from stmmlib.beta.lrt_fixed import lrt_fixed_effects
    df = _tiny_panel()
    cfg = BetaFitConfig(phi_prior_mean=6.9, phi_prior_sd=2.0)
    fit = fit_beta(df, "m1_iid_iid", preset="fast", max_iter=50, config=cfg)
    lrt = lrt_fixed_effects(fit, df, terms=["x1"])
    assert lrt["penalized"] is True
    row = lrt["terms"]["x1"]
    assert row["status"] == "ok"
    assert row["D"] is not None
    assert row["p"] is None
    assert "p_note" in row


def test_is_penalized_reads_auto_prior_and_stored_config():
    from stmmlib.beta.config import BetaFitConfig
    from stmmlib.beta.lrt_fixed import _is_penalized
    auto = {"optim": {"penalty_auto": True, "penalty_refit_failed": False}}
    assert _is_penalized(auto) is True
    failed = {"optim": {"penalty_auto": True, "penalty_refit_failed": True}}
    assert _is_penalized(failed) is False
    stored = {"optim": {},
              "fit_config": BetaFitConfig(phi_prior_mean=0.0, phi_prior_sd=1.0)}
    assert _is_penalized(stored) is True
    assert _is_penalized({"optim": {}, "fit_config": BetaFitConfig()}) is False


# --- A7: SAR without W must fail, not silently become iid -------------------

def test_sar_without_W_raises(df_flat):
    from stmmlib.stmm import stmm
    with pytest.raises(ValueError, match="bobot spasial W|W wajib"):
        stmm(data=df_flat, model="m3_sar_iid", family="beta", gpu="no",
             n_starts=1, max_iter=20, response_col="y",
             area_col="area", time_col="time")


def test_build_beta_Q_requires_W_and_checks_shape():
    from stmmlib._backend import Backend
    from stmmlib.beta.prior import build_beta_Q
    from stmmlib.beta.structure import stmm_beta_structure
    s = stmm_beta_structure("m3_sar_iid")
    des = {"blocks": {"area": [0] * 4, "time": [0] * 2}, "T": 2, "m": 4}
    kw = dict(logs=np.array([0.0, 0.0]), raw_dep=np.array([0.0]),
              ksd=2, q_dep=1, rho_bound=0.95, phi_bound=0.95,
              backend=Backend(gpu="no"), use_blocked=False)
    with pytest.raises(ValueError, match="needs W"):
        build_beta_Q(s, des, None, **kw)
    with pytest.raises(ValueError, match="mismatches n_area"):
        build_beta_Q(s, des, np.ones((3, 3)), **kw)


# --- A8/A10: nested post-hoc helpers use the same columns as the fit -------

def test_nested_helpers_default_columns_and_reason(df_nested):
    from stmmlib.beta.criteria import beta_criteria
    from stmmlib.beta.lrt_fixed import lrt_fixed_effects
    fit = fit_beta(df_nested, "m6_nested_iid_iid", max_iter=300)
    lrt = lrt_fixed_effects(fit, df_nested, terms=["x1"])
    assert lrt["terms"]["x1"]["status"] in ("ok", "not_converged", "failed")
    crit = beta_criteria(fit)
    assert np.isnan(crit["cAIC"])
    assert crit["reason"]


# --- A9: the fixed-effect table exposes the certificate --------------------

def test_wald_table_certificate_and_fixed_only(df_flat):
    from stmmlib.diagnostics_suite import wald_table_beta
    fit = fit_beta(df_flat, "m1_iid_iid")
    rows = wald_table_beta(fit)
    assert "certificate_valid" in rows[0]
    fixed = wald_table_beta(fit, fixed_only=True)
    assert len(fixed) == int(fit._opt["p"])
    assert len(fixed) < len(rows)
