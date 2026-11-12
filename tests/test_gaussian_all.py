"""Gaussian path: all 14 models fit via public stmm() dispatch."""
import numpy as np

from conftest import fit_gauss

CASES = [
    ("m1_iid_iid", False),
    ("m2_iid_ar1", False),
    ("m3_sar_iid", True),
    ("m4_sar_ar1", True),
    ("m5_seblup_rao_yu_it", True),
    ("m8_iid_iid_int", False),
    ("m9_sar_iid_int", True),
    ("m10_iid_ar1_int", False),
    ("m11_sar_ar1_int", True),
    ("m12_seblup_rao_yu_it_int", True),
]
NESTED = ["m6_nested_iid_iid", "m7_nested_ar1",
          "m13_nested_iid_iid_int", "m14_nested_ar1_int"]


def test_gaussian_flat_10_fit(df_gauss, W_flat):
    for mid, needs_W in CASES:
        r = fit_gauss(df_gauss, mid, W_flat if needs_W else None)
        assert "log_lik" in r, mid
        assert np.isfinite(r["log_lik"]), mid
        assert r.get("converged", False), mid
        assert "theta" in r and "beta_hat" in r, mid


def test_gaussian_nested_4_fit(df_nested):
    for mid in NESTED:
        r = fit_gauss(df_nested, mid)
        assert np.isfinite(r["log_lik"]), mid
        assert r.get("converged", False), mid


def test_gaussian_interaction_enters_likelihood(df_gauss, W_flat):
    """M8-M12 logLik must differ from flat counterparts, unless the fitted
    sigma2_gamma collapses to the lower bound (legitimate on noise data)."""
    pairs = [("m1_iid_iid", "m8_iid_iid_int", None),
             ("m2_iid_ar1", "m10_iid_ar1_int", None),
             ("m3_sar_iid", "m9_sar_iid_int", W_flat)]
    for flat, inter, Wm in pairs:
        rf = fit_gauss(df_gauss, flat, Wm)
        ri = fit_gauss(df_gauss, inter, Wm)
        diff = abs(ri["log_lik"] - rf["log_lik"])
        g = ri["theta"].get("sigma2_gamma", 0.0) or 0.0
        assert diff > 1e-6 or g < 1e-6, (flat, inter, diff, g)


def test_gaussian_sigma_gamma_present(df_gauss):
    r = fit_gauss(df_gauss, "m8_iid_iid_int")
    assert r["theta"].get("sigma2_gamma") is not None
    assert r["theta"]["sigma2_gamma"] >= 0


def test_gaussian_nested_blup_real(df_nested):
    """Nested BLUP must not be stubbed: u_it nonzero, blocks present."""
    r = fit_gauss(df_nested, "m7_nested_ar1")
    b = r["blup"]
    assert len(b["u_it"]) == 48
    assert np.abs(np.asarray(b["u_it"])).max() > 1e-12
    assert "gamma" in b and "v_prov" in b and "v_kab" in b


def test_the_response_is_never_its_own_predictor(df_gauss, df_nested, W_flat):
    """A response that is not named "y" must not enter the design.

    The seblup and nested paths skipped the literal "y", so with any other
    response name the response became a predictor of itself: a silently
    wrong model.
    """
    from stmmlib.stmm import stmm
    cases = [
        (df_gauss.rename(columns={"y": "respons"}), "m1_iid_iid", None),
        (df_gauss.rename(columns={"y": "respons"}), "m5_seblup_rao_yu_it", W_flat),
        (df_nested.rename(columns={"y": "respons"}), "m6_nested_iid_iid", None),
    ]
    for df, mid, Wm in cases:
        r = stmm(data=df, model=mid, family="gaussian", W=Wm, method="ML",
                 n_starts=1, gpu="no", response_col="respons",
                 area_col="area", time_col="time",
                 prov_col="prov", kab_col="kab")
        assert r["p"] == 2, (mid, r["p"])
