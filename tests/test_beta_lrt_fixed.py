"""Restricted-refit LRT for Beta fixed effects.

Tests the theta packing round trip and the restricted refit itself.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from conftest import fit_beta
from stmmlib.beta.fit import _sanitize_data
from stmmlib.beta.lrt_fixed import (lrt_fixed_terms, pack_theta,
                                    unpack_theta)
from stmmlib.beta.solvers import beta_laplace_structural


def _panel(signal: float, seed: int = 11):
    """Panel with one signal predictor (x2) and one noise predictor (x3)."""
    m, T = 12, 4
    n = m * T
    rng = np.random.default_rng(seed)
    area = np.repeat(np.arange(m), T)
    time = np.tile(np.arange(T), m)
    aef = rng.normal(0, 0.4, m)
    x2 = rng.normal(size=n)
    eta = (-1.2 + signal * x2 + rng.normal(size=n) + aef[area])
    y = rng.beta(20 * np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98), 20)
    return pd.DataFrame({"y": y, "x2": x2, "x3": rng.normal(size=n),
                         "area": area, "time": time})


def test_pack_unpack_roundtrip(df_flat):
    fit = fit_beta(df_flat, "m1_iid_iid")
    opt = fit._opt
    theta = pack_theta(opt)
    back = unpack_theta(theta, opt["p"], len(np.asarray(opt["logs"]).ravel()))
    assert np.allclose(back["beta_hat"], opt["beta_hat"])
    assert back["logphi"] == float(np.asarray(opt["logphi"]).ravel()[0])
    assert np.allclose(back["logs"], opt["logs"])
    assert np.allclose(back["raw_dep"], opt["raw_dep"])
    assert len(theta) == opt["p"] + 1 + len(np.asarray(opt["logs"]).ravel()) \
        + len(np.asarray(opt["raw_dep"]).ravel())


def test_pack_theta_matches_loglik(df_flat):
    """Packed theta must reproduce the stored logLik when re-evaluated."""
    fit = fit_beta(df_flat, "m1_iid_iid")
    d, _, _ = _sanitize_data(df_flat)
    ev = beta_laplace_structural(None, d, d["y"].to_numpy(float),
                                 "Laplace", "m1_iid_iid", None,
                                 eval_param=pack_theta(fit._opt))
    assert abs(float(ev["log_lik"]) - fit.log_lik) < 1e-6


def _lrt(df, terms, **kw):
    fit = fit_beta(df, "m1_iid_iid")
    d, _, _ = _sanitize_data(df)
    return lrt_fixed_terms(fit._opt, d, d["y"].to_numpy(float),
                           "m1_iid_iid", None, terms, **kw)


def test_lrt_rejects_strong_signal():
    """A predictor that drives the response must be rejected by the LRT."""
    out = _lrt(_panel(signal=1.5), ["x2", "x3"])
    assert out["terms"]["x2"]["p"] < 0.05, out["terms"]["x2"]
    assert out["terms"]["x2"]["status"] == "ok"
    assert out["terms"]["x3"]["p"] > 0.05, out["terms"]["x3"]


def test_lrt_reports_df_and_d():
    out = _lrt(_panel(signal=1.5), ["x2"])
    row = out["terms"]["x2"]
    assert row["df"] == 1
    assert row["D"] >= 0.0
    assert row["loglik_full"] > row["loglik_reduced"]
    assert row["p"] == pytest.approx(float(__import__("scipy.stats",
                                                     fromlist=["chi2"]).chi2.sf(
        row["D"], 1)))


def test_unknown_term_raises():
    with pytest.raises(KeyError):
        _lrt(_panel(signal=1.5), ["nope"])


def test_intercept_is_rejected():
    with pytest.raises(ValueError):
        _lrt(_panel(signal=1.5), ["(Intercept)"])


def test_failed_refit_has_no_p(df_flat, monkeypatch):
    """A refit that returns nan must not produce a p-value."""
    monkeypatch.setattr("stmmlib.beta.lrt_fixed._refit_reduced",
                        lambda *a, **k: {"log_lik": float("nan"),
                                         "status": "invalid",
                                         "converged": False})
    out = _lrt(df_flat, ["x1"], method="plain")
    row = out["terms"]["x1"]
    assert row["p"] is None
    assert row["status"] == "failed"
    assert out["n_failed"] == 1


def test_failed_refinement_has_no_p(df_flat, monkeypatch):
    """Same rule for the refinement path: no number without a usable fit."""
    monkeypatch.setattr("stmmlib.beta.lrt_fixed._refine_restricted",
                        lambda *a, **k: {"log_lik": float("nan"),
                                         "converged": False,
                                         "iterations": 0,
                                         "grad_max": float("nan")})
    out = _lrt(df_flat, ["x1"])
    row = out["terms"]["x1"]
    assert row["p"] is None
    assert row["status"] == "failed"
    assert out["n_failed"] == 1


def test_refined_restricted_likelihood_cannot_exceed_the_full_one(df_flat):
    """A restriction cannot beat the full model; if it does, the number is
    refused instead of reported."""
    out = _lrt(df_flat, ["x1"])
    row = out["terms"]["x1"]
    assert row["loglik_reduced"] <= row["loglik_full"] + 1e-6
    assert row["method"] == "refine"


def test_deadline_marks_terms_skipped(df_flat):
    out = _lrt(df_flat, ["x1"], deadline_s=0)
    row = out["terms"]["x1"]
    assert row["status"] == "skipped_deadline"
    assert row["p"] is None


def test_lrt_block_reports_flags(df_flat):
    out = _lrt(df_flat, ["x1"])
    assert out["method"] == "lrt"
    assert out["df_per_term"] == 1
    assert isinstance(out["phi_at_bound"], bool)
    assert isinstance(out["penalized"], bool)


# --- the inference switch on the fitter -----------------------------------

def test_default_is_wald_and_unchanged(df_flat):
    f0 = fit_beta(df_flat, "m1_iid_iid")
    f1 = fit_beta(df_flat, "m1_iid_iid", inference="wald")
    assert f1.lrt is None
    assert f1.inference_method == "wald"
    assert f1.log_lik == pytest.approx(f0.log_lik, rel=0, abs=1e-9)


def test_bad_inference_value_raises(df_flat):
    with pytest.raises(ValueError, match="inference"):
        fit_beta(df_flat, "m1_iid_iid", inference="score")


def test_lrt_block_present_when_requested(df_flat):
    f = fit_beta(df_flat, "m1_iid_iid", inference="lrt", lrt_terms=["x1"])
    assert list(f.lrt["terms"]) == ["x1"]
    assert f.lrt["terms"]["x1"]["df"] == 1
    assert f.inference_method == "lrt"


def test_lrt_pvalue_follows_method_and_keeps_wald(df_flat):
    f = fit_beta(df_flat, "m1_iid_iid", inference="lrt", lrt_terms=["x1"])
    row = f.lrt["terms"]["x1"]
    assert f.inference["method"] == "lrt"
    assert f.inference["test"] == "lrt"
    assert f.inference["pvalue"][1] == pytest.approx(row["p"])
    z = float(f.inference["tvalue"][1])
    assert f.inference["pvalue_wald"][1] == pytest.approx(
        2.0 * (1.0 - norm.cdf(abs(z))))
    beta = f.inference["estimate"][1]
    assert f.inference["statistic"][1] == pytest.approx(
        np.sign(beta) * np.sqrt(row["D"]))
    # the Wald alias is untouched
    assert np.isfinite(f.inference["tvalue"][1])


def test_lrt_terms_subset_is_respected(df_flat):
    f = fit_beta(df_flat, "m1_iid_iid", inference="lrt", lrt_terms=["x1"])
    assert list(f.lrt["terms"]) == ["x1"]


def test_beta_report_labels_the_test(df_flat):
    from stmmlib.output_format_beta import format_beta
    f_wald = fit_beta(df_flat, "m1_iid_iid")
    assert "Uji Wald" in format_beta(f_wald)
    f_lrt = fit_beta(df_flat, "m1_iid_iid", inference="lrt", lrt_terms=["x1"])
    txt = format_beta(f_lrt)
    assert "LRT" in txt
    assert "chi2_1" in txt


# --- universal entry point: raw data in, LRT out ---------------------------

def test_lrt_fixed_effects_accepts_raw_data(df_flat):
    from stmmlib.beta.lrt_fixed import lrt_fixed_effects
    fit = fit_beta(df_flat, "m1_iid_iid")
    out = lrt_fixed_effects(fit, df_flat, terms=["x1"])
    assert out["terms"]["x1"]["status"] == "ok"
    assert out["terms"]["x1"]["p"] is not None


def test_lrt_fixed_effects_covers_a_new_predictor(df_flat):
    """A predictor that did not exist when the module was written."""
    from stmmlib.beta.lrt_fixed import lrt_fixed_effects
    df = df_flat.copy()
    df["x_new"] = np.random.default_rng(3).normal(size=len(df))
    fit = fit_beta(df, "m1_iid_iid")
    out = lrt_fixed_effects(fit, df)
    assert set(out["terms"]) == {"x1", "x_new"}


def test_lrt_fixed_effects_rejects_data_that_is_not_the_fitted_data(df_flat):
    from stmmlib.beta.lrt_fixed import lrt_fixed_effects
    fit = fit_beta(df_flat, "m1_iid_iid")
    other = df_flat.copy()
    other["y"] = other["y"] * 0.5 + 0.1
    with pytest.raises(ValueError, match="data"):
        lrt_fixed_effects(fit, other, terms=["x1"])


def test_lrt_fixed_effects_rejects_missing_predictor(df_flat):
    from stmmlib.beta.lrt_fixed import lrt_fixed_effects
    fit = fit_beta(df_flat, "m1_iid_iid")
    with pytest.raises(ValueError):
        lrt_fixed_effects(fit, df_flat.drop(columns=["x1"]), terms=["x1"])


# --- the fixed-effect table, one shape, two tests -------------------------

def test_table_dispatches_to_lrt():
    from stmmlib.diagnostics_suite import wald_table_beta
    fit = fit_beta(_panel(signal=1.5), "m1_iid_iid", inference="lrt",
                    lrt_terms=["x2"])
    rows = wald_table_beta(fit, method="lrt")
    row = [r for r in rows if r["param"] == "x2"][0]
    assert row["test"] == "lrt"
    assert row["p_value"] == pytest.approx(fit.lrt["terms"]["x2"]["p"])
    assert row["p_value_wald"] == pytest.approx(
        fit.inference["pvalue_wald"][1])
    assert row["se"] is not None
    # an untested fixed effect keeps the Wald p and says so
    other = [r for r in rows if r["param"] == "x3"][0]
    assert other["test"] == "wald_z"
    # the intercept is never LRT
    assert rows[0]["test"] == "wald_z"


def test_table_default_follows_the_fit():
    from stmmlib.diagnostics_suite import wald_table_beta
    fit = fit_beta(_panel(signal=1.5), "m1_iid_iid", inference="lrt",
                    lrt_terms=["x2"])
    rows = wald_table_beta(fit)
    row = [r for r in rows if r["param"] == "x2"][0]
    assert row["test"] == "lrt"
    wald = fit_beta(_panel(signal=1.5), "m1_iid_iid")
    assert [r for r in wald_table_beta(wald) if r["param"] == "x2"][0]["test"] \
        == "wald_z"


def test_table_lrt_mode_needs_an_lrt_fit(df_flat):
    from stmmlib.diagnostics_suite import wald_table_beta
    fit = fit_beta(df_flat, "m1_iid_iid")
    with pytest.raises(ValueError, match="LRT"):
        wald_table_beta(fit, method="lrt")


def test_refit_with_warm_start_b_reproduces_the_full_loglik(df_flat):
    """Re-fitting the same model with the fitted mode as the inner start
    must reproduce the stored logLik. This is the warm start a restricted
    refit needs: the cold start makes the inner solve diverge."""
    fit = fit_beta(df_flat, "m1_iid_iid")
    d, _, _ = _sanitize_data(df_flat)
    res = beta_laplace_structural(None, d, d["y"].to_numpy(float), "Laplace",
                                   "m1_iid_iid", None, n_starts=1,
                                   start_theta=pack_theta(fit._opt),
                                   start_b=np.asarray(fit._opt["b_hat"],
                                                      dtype=float),
                                   compute_hessian=False)
    assert abs(float(res["log_lik"]) - fit.log_lik) < 1e-6


def test_start_b_of_the_wrong_length_is_refused(df_flat):
    """A warm start that does not match the random-effect structure is a
    programming error, not something to ignore silently."""
    fit = fit_beta(df_flat, "m1_iid_iid")
    d, _, _ = _sanitize_data(df_flat)
    with pytest.raises(ValueError, match="start_b"):
        beta_laplace_structural(None, d, d["y"].to_numpy(float), "Laplace",
                                "m1_iid_iid", None, n_starts=1,
                                start_theta=pack_theta(fit._opt),
                                start_b=np.zeros(3), compute_hessian=False)


def test_restricted_refit_does_not_start_from_the_full_optimum(df_flat,
                                                               monkeypatch):
    """The restricted refit must use the solver's own start, not the full
    optimum: at the full optimum the gradient of the reduced model is large
    and L-BFGS-B's first step (g/max|g|) blows the objective up. The mode b
    of the full fit is still passed, so the inner solve does not diverge."""
    import stmmlib.beta.solvers as solvers
    from stmmlib.beta.lrt_fixed import _refit_reduced
    seen = {}

    def fake(*a, **kw):
        seen.update(kw)
        return {"log_lik": 1.0, "status": "konvergen", "converged": True}

    monkeypatch.setattr(solvers, "beta_laplace_structural", fake)
    d, _, _ = _sanitize_data(df_flat)
    _refit_reduced(d, d["y"].to_numpy(float), "m1_iid_iid", None,
                   start_b=np.zeros(12), n_starts=2)
    assert seen.get("start_theta") is None
    assert seen.get("start_b") is not None
    assert seen.get("n_starts") == 2
    assert seen.get("compute_hessian") is False


def test_refine_and_plain_agree_on_the_same_data():
    """The refinement and a plain restricted refit must find the same
    restricted maximum. If they disagree, the refinement is stopping short
    and every D it reports is too large."""
    df = _panel(signal=1.5)
    fit = fit_beta(df, "m1_iid_iid")
    d, _, _ = _sanitize_data(df)
    y = d["y"].to_numpy(float)
    ref = lrt_fixed_terms(fit._opt, d, y, "m1_iid_iid", None, ["x2"],
                          method="refine")
    plain = lrt_fixed_terms(fit._opt, d, y, "m1_iid_iid", None, ["x2"],
                            method="plain")
    d_ref = ref["terms"]["x2"]["D"]
    d_plain = plain["terms"]["x2"]["D"]
    assert d_ref is not None and d_plain is not None
    assert abs(d_ref - d_plain) / d_plain < 0.10, (d_ref, d_plain)


def test_refinement_must_reach_a_small_gradient(df_flat):
    """A refinement that stops with a large projected gradient has not found
    the restricted maximum, so it must not produce a p-value."""
    fit = fit_beta(df_flat, "m1_iid_iid")
    d, _, _ = _sanitize_data(df_flat)
    out = lrt_fixed_terms(fit._opt, d, d["y"].to_numpy(float), "m1_iid_iid",
                          None, ["x1"], method="refine")
    row = out["terms"]["x1"]
    if row["status"] == "ok":
        assert row["grad_max"] <= 1e-2, row
    else:
        assert row["p"] is None


def test_untested_terms_carry_no_p_under_an_lrt_header():
    """A predictor that was not LRT-tested must not show a Wald p-value in a
    table whose header says LRT."""
    df = _panel(signal=1.5)
    fit = fit_beta(df, "m1_iid_iid", inference="lrt", lrt_terms=["x2"])
    inf = fit.inference
    labels = list(inf["param_labels"])
    jx2 = labels.index("x2")
    jx3 = labels.index("x3")
    assert np.isfinite(inf["pvalue"][jx2])          # tested: LRT p
    assert np.isnan(inf["pvalue"][jx3])              # not tested: no p
    assert np.isnan(inf["pvalue"][0])                # intercept: never tested
    assert np.isfinite(inf["pvalue_wald"][jx3])      # Wald p is still there


def test_strict_is_reachable_from_summary(df_flat):
    """fit.summary() must be able to run the report in strict mode."""
    fit = fit_beta(df_flat, "m1_iid_iid")
    fit._opt.setdefault("certificate", {})["valid"] = False
    txt = fit.summary(strict=True)
    assert "SERTIFIKAT" in txt


def test_strict_in_lrt_mode_keeps_the_lrt_p_value(df_flat):
    """In LRT mode the p-value is the likelihood ratio result the caller
    asked for; only the Wald interval is hidden."""
    from stmmlib.output_format_beta import format_beta
    fit = fit_beta(df_flat, "m1_iid_iid", inference="lrt", lrt_terms=["x1"])
    fit._opt.setdefault("certificate", {})["valid"] = False
    p_lrt = float(fit.inference["pvalue"][1])
    assert np.isfinite(p_lrt)
    txt = format_beta(fit, strict=True)
    assert f"{p_lrt:.4f}" in txt


def test_lrt_evaluates_the_same_model_as_the_fit(df_flat, W_flat):
    """With a custom config the objective used by the LRT must be the same
    model the fit was estimated on, or the two likelihoods are not
    comparable. The spatial model is used because only there does the rho
    bound change the model (measured: the two differ by 0.0047 logLik)."""
    from stmmlib.beta.config import BetaFitConfig
    from stmmlib.beta.lrt_fixed import _refine_restricted
    cfg = BetaFitConfig(rho_bound=0.99)
    fit = fit_beta(df_flat, "m3_sar_iid", W_flat, config=cfg)
    d, _, _ = _sanitize_data(df_flat)
    ref = _refine_restricted(pack_theta(fit._opt), d,
                             d["y"].to_numpy(float), "m3_sar_iid", W_flat,
                             np.asarray(fit._opt["b_hat"], dtype=float),
                             None, max_iter=0, config=cfg)
    assert ref["log_lik"] == pytest.approx(fit.log_lik, abs=1e-6)


def test_non_numeric_predictor_names_the_column(df_flat):
    """The guard must say which column is the problem."""
    bad = df_flat.copy()
    bad["x1"] = "abc"
    with pytest.raises(ValueError, match="x1"):
        fit_beta(bad, "m1_iid_iid")


def test_lrt_fixed_effects_rejects_different_areas(df_flat):
    """A different panel behind the same fit must be refused."""
    from stmmlib.beta.lrt_fixed import lrt_fixed_effects
    fit = fit_beta(df_flat, "m1_iid_iid")
    other = df_flat.copy()
    other["area"] = other["area"] + 100
    with pytest.raises(ValueError, match="area"):
        lrt_fixed_effects(fit, other, terms=["x1"])


def test_refinement_stops_at_the_deadline(df_flat):
    """A refinement that runs out of time must report no convergence."""
    import time as _t
    from stmmlib.beta.lrt_fixed import _refine_restricted
    fit = fit_beta(df_flat, "m1_iid_iid")
    d, _, _ = _sanitize_data(df_flat)
    ref = _refine_restricted(pack_theta(fit._opt), d,
                             d["y"].to_numpy(float), "m1_iid_iid", None,
                             np.asarray(fit._opt["b_hat"], dtype=float),
                             None, deadline_at=_t.perf_counter() - 1.0)
    assert ref["iterations"] == 0
    assert ref["converged"] is False


def test_bounds_come_from_the_solver(df_flat):
    """The LRT bounds must be the solver's own bounds, not a copy."""
    from stmmlib.beta.config import BetaFitConfig
    from stmmlib.beta.lrt_fixed import _bounds_arrays
    from stmmlib.beta.solvers import beta_parameter_bounds
    cfg = BetaFitConfig()
    fit = fit_beta(df_flat, "m1_iid_iid", config=cfg)
    opt = fit._opt
    p = int(opt["p"])
    ksd = len(np.asarray(opt["logs"]).ravel())
    q = len(np.asarray(opt["raw_dep"]).ravel())
    lo, hi = _bounds_arrays(opt, cfg)
    ref = beta_parameter_bounds(cfg, p, ksd, q)
    assert lo.tolist() == [b[0] for b in ref]
    assert hi.tolist() == [b[1] for b in ref]


def test_refinement_survives_a_precision_on_its_bound(df_flat):
    """A precision that sits on its bound must not stall the restricted refit.

    The Newton step of the reduced model can go almost entirely into
    coordinates that sit on a bound: the curvature there is small while the
    coefficients have a curvature of 1e5, and both want to leave the box.
    Clipping them afterwards leaves an almost empty step while the Armijo
    test still uses the slope of the clipped direction, so the search can
    stall with a large projected gradient after one step. The step is now
    projected onto the KKT active set.
    """
    from stmmlib.beta.config import BetaFitConfig
    cfg = BetaFitConfig(logphi_bounds=(-10.0, 0.0))
    fit = fit_beta(df_flat, "m1_iid_iid", config=cfg)
    opt = fit._opt
    logphi = float(np.asarray(opt["logphi"], dtype=float).ravel()[0])
    assert logphi >= 0.0 - 1e-9, f"the fit should sit on the bound: {logphi}"
    d, _, _ = _sanitize_data(df_flat)
    out = lrt_fixed_terms(opt, d, d["y"].to_numpy(float), "m1_iid_iid", None,
                          ["x1"], method="refine", config=cfg)
    row = out["terms"]["x1"]
    assert row["status"] == "ok", row
    assert row["p"] is not None


def test_lrt_statistic_agrees_with_the_wald_statistic(df_flat):
    """On a well-conditioned panel the LRT statistic and z^2 must agree.

    They come from different places (z^2 from the Hessian at the optimum, D
    from a re-optimised restricted model), so a large disagreement means one
    of the two is wrong.
    """
    fit = fit_beta(df_flat, "m1_iid_iid")
    inf = fit.inference
    labels = list(inf["param_labels"])
    j = labels.index("x1")
    z = float(inf["tvalue"][j])
    d, _, _ = _sanitize_data(df_flat)
    out = lrt_fixed_terms(fit._opt, d, d["y"].to_numpy(float), "m1_iid_iid",
                          None, ["x1"], method="refine")
    row = out["terms"]["x1"]
    assert row["status"] == "ok", row
    D = float(row["D"])
    assert abs(D - z * z) / max(z * z, 1e-12) < 0.10, (D, z * z)


def test_refinement_projects_the_step_onto_the_active_set(monkeypatch):
    """The Newton step must not go into the coordinates that sit on a bound.

    This reproduces the failure measured on the real panel. A bound
    coordinate with a small curvature (40) and a large gradient (-30) drags
    the whole step into itself, the realised step is empty, and the search
    stalls with a projected gradient of 523. The step is now projected onto
    the KKT active set, so the free coordinate gets a proper Newton step.
    """
    import stmmlib.beta.solvers as solvers

    H = np.array([[1.0e5, 900.0, 0.0],
                  [900.0, 40.0, 0.0],
                  [0.0, 0.0, 40.0]])
    g0 = np.array([-100.0, -30.0, -20.0])
    theta0 = np.array([0.0, 10.0, 1.8317807])
    bounds = (np.array([-20.0, -10.0, -1.8317807]),
              np.array([20.0, 10.0, 1.8317807]))

    def fake(formula, data, y, method, model, W, **kw):
        th = kw.get("eval_param")
        if th is not None:
            th = np.asarray(th, float)
            d = th - theta0
            return {"log_lik": -float(0.5 * d @ H @ d + g0 @ d),
                    "mode_residual": 0.0}
        th = np.asarray(kw["eval_grad"], float)
        return {"grad": H @ (th - theta0) + g0, "grad_kind": "analytic"}

    monkeypatch.setattr(solvers, "beta_laplace_structural", fake)
    from stmmlib.beta.lrt_fixed import _refine_restricted
    ref = _refine_restricted(theta0.copy(), None, None, "m1_iid_iid", None,
                             None, bounds=bounds, max_iter=40)
    assert ref["converged"] is True, ref
    assert ref["grad_proj"] <= 1e-6, ref
    # the bound coordinates must not have moved
    assert abs(ref["theta"][1] - 10.0) < 1e-12, ref["theta"]
    assert abs(ref["theta"][2] - 1.8317807) < 1e-12, ref["theta"]
    # the free coordinate took a proper Newton step
    assert abs(ref["theta"][0] - 0.001) < 1e-9, ref["theta"]
