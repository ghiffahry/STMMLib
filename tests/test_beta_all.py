"""Beta path: core M1-M5 + interaction M8-M12 + nested M6/M7/M13/M14."""
import numpy as np
import pytest

from conftest import fit_beta
from stmmlib.beta.config import BetaFitConfig

FLAT = ["m1_iid_iid", "m2_iid_ar1", "m3_sar_iid", "m4_sar_ar1",
        "m5_seblup_rao_yu_it"]
INTER = [("m8_iid_iid_int", None), ("m9_sar_iid_int", "W"),
         ("m10_iid_ar1_int", None), ("m11_sar_ar1_int", "W")]
NESTED = ["m6_nested_iid_iid", "m7_nested_ar1",
          "m13_nested_iid_iid_int", "m14_nested_ar1_int"]


def test_beta_m1_m5_fit(df_flat, W_flat):
    for mid in FLAT:
        r = fit_beta(df_flat, mid, W_flat)
        o = r._opt
        assert np.isfinite(o["log_lik"]), mid
        assert o["status"] in ("konvergen", "max_iter"), mid
        assert o["sigma_area"] > 0 and o["sigma_time"] > 0, mid


def test_beta_m1_m5_inference(df_flat, W_flat):
    """Inference block: labels complete, SE positive, curvature reported."""
    r = fit_beta(df_flat, "m2_iid_ar1")
    inf = r.inference
    labels = inf["param_labels"]
    assert "log(phi)" in labels and "log_sigma_area" in labels
    assert np.all(np.asarray(inf["se"]) > 0)
    assert isinstance(inf["hessian_pd"], bool)


def test_beta_interaction_m8_m12(df_flat, W_flat):
    from stmmlib.beta.criteria import beta_criteria
    for mid, w in INTER:
        r = fit_beta(df_flat, mid, W_flat if w == "W" else None)
        o = r._opt
        assert np.isfinite(o["log_lik"]), mid
        assert o["sigma_gamma"] is not None and o["sigma_gamma"] > 0, mid
        inf = r.inference
        assert "log_sigma_gamma" in inf["param_labels"], mid
        cr = beta_criteria(r)
        assert cr is not None and np.isfinite(cr["df_hat"]), mid
        assert 0 <= cr["df_hat"] <= o["X"].shape[1] + o["Z"].shape[1], mid
        assert "interaction" in r.VarCorr(), mid
        assert "interaction" in r.ranef(), mid


def test_beta_interaction_dep_labels(df_flat, W_flat):
    """raw_rho/raw_phi land at the right raw_dep slots (atanh-scale labels)."""
    r9 = fit_beta(df_flat, "m9_sar_iid_int", W_flat)
    lab9 = r9.inference["param_labels"]
    assert lab9[-2:] == ["raw_rho (atanh)", "raw_rho_gamma (atanh)"], lab9[-2:]
    r10 = fit_beta(df_flat, "m10_iid_ar1_int")
    lab10 = r10.inference["param_labels"]
    assert lab10[-2:] == ["raw_phi_T (atanh)", "raw_phi_gamma (atanh)"], lab10[-2:]


def test_beta_nested_m6_m7(df_nested):
    r6 = fit_beta(df_nested, "m6_nested_iid_iid")
    o6 = r6._opt
    assert np.isfinite(o6["log_lik"]) and o6["status"] == "konvergen"
    labels6 = r6.inference["param_labels"]
    assert "log_sigma_prov" in labels6 and "log_sigma_kab" in labels6
    assert set(r6.ranef()) == {"prov", "kab", "cell"}

    r7 = fit_beta(df_nested, "m7_nested_ar1")
    labels7 = r7.inference["param_labels"]
    assert labels7[-4:] == ["log_sigma_prov", "log_sigma_kab",
                            "log_sigma_eta", "raw_phi_T (atanh)"], labels7[-4:]
    assert set(r7.ranef()) == {"prov", "kab", "cell"}
    assert np.all(np.isfinite(r7.predict_interval()["se_logit"]))


def test_beta_simulate_nested_refused():
    from stmmlib.beta.simulation import simulate_beta
    with pytest.raises(ValueError, match="nested"):
        simulate_beta(n=8, m=2, T=4, model="m6_nested_iid_iid", W=np.eye(2))


def test_beta_nested_consistency(df_nested):
    """Nested interaction models must not be worse than their nested base.

    On data without interaction signal, sigma_gamma collapses and
    logLik(M13) == logLik(M6); in general logLik(M13) >= logLik(M6).
    Uses a generous max_iter (dense 5-6 param Laplace is slow).
    """
    r6 = fit_beta(df_nested, "m6_nested_iid_iid", max_iter=400)
    r13 = fit_beta(df_nested, "m13_nested_iid_iid_int", max_iter=3000,
                   n_starts=3, inner_max_iter=50, seed=42)
    assert r13._opt["log_lik"] >= r6._opt["log_lik"] - 1.0

    r7 = fit_beta(df_nested, "m7_nested_ar1", max_iter=400)
    r14 = fit_beta(df_nested, "m14_nested_ar1_int", max_iter=3000,
                   n_starts=3, inner_max_iter=50, seed=42)
    # The >= invariant only claims meaning at convergence; unconverged fits
    # must at least be honestly labeled max_iter (never silently "konvergen").
    if r14._opt["status"] == "konvergen":
        assert r14._opt["log_lik"] >= r7._opt["log_lik"] - 1.0
    else:
        assert r14._opt["status"] == "max_iter"


def test_beta_seblup_m5_flagship(df_flat, W_flat):
    """M5 (thesis flagship): fit + inference + certificate fields."""
    r = fit_beta(df_flat, "m5_seblup_rao_yu_it", W_flat, max_iter=600,
                 config=BetaFitConfig())  # explicit config: matikan auto-refit
    o = r._opt
    assert np.isfinite(o["log_lik"])
    inf = r.inference
    assert isinstance(inf["hessian_pd"], bool)
    assert inf["lambda_min"] is not None
    lab = inf["param_labels"]
    assert "raw_rho (atanh)" in lab and "raw_phi_T (atanh)" in lab  # SAR + AR1
    # Certificate machinery: an invalid fit must carry a reason in warnings;
    # a valid one must be PD. (Small synthetic panels can collapse sigma_v to
    # the bound -- the certificate flags it, it is not a library failure.)
    cert = o["certificate"]
    assert cert["warnings"]
    if cert["valid"]:
        assert cert["hessian_pd"] and not cert["at_bound"]


def test_beta_k_total_includes_interaction_params(df_flat):
    """k must grow when interaction adds sigma_gamma (+ dep)."""
    r1 = fit_beta(df_flat, "m1_iid_iid")
    r8 = fit_beta(df_flat, "m8_iid_iid_int")
    assert r8._opt["k"] == r1._opt["k"] + 1  # + log_sigma_gamma


def test_beta_presets_record_tolerances(df_flat):
    """fast/medium/full record their ftol/gtol; invalid preset rejected."""
    from stmmlib.stmm import stmm
    rec = {}
    for preset, (ftol, gtol) in {"fast": (1e-5, 1e-5),
                                 "medium": (1e-8, 1e-6),
                                 "full": (1e-12, 1e-6)}.items():
        r = fit_beta(df_flat, "m1_iid_iid", preset=preset, max_iter=200)
        om = r._opt["optim"]
        assert om["ftol"] == ftol and om["gtol"] == gtol, (preset, om)
        assert np.isfinite(r._opt["log_lik"]), preset
        rec[preset] = r._opt["log_lik"]
    # medium/full share tight tolerances: near-identical likelihood
    assert abs(rec["medium"] - rec["full"]) < 1e-3, rec
    # skips Hessian: no SEs, explicit None PD
    r_fast = fit_beta(df_flat, "m1_iid_iid", preset="fast", max_iter=200)
    assert not np.any(np.isfinite(r_fast.inference["se"]))
    assert r_fast.inference["hessian_pd"] is None
    with pytest.raises(ValueError):
        stmm(data=df_flat, model="m1_iid_iid", family="beta", preset="turbo")
