"""stmm_model_equation: one equation string per M1-M14, both families."""
import pytest

from stmmlib.registry import MODELS, stmm_model_equation


def test_equation_all_models_both_families():
    for mid in MODELS:
        for fam in ("beta", "gaussian"):
            eq = stmm_model_equation(mid, fam)
            assert mid in eq and MODELS[mid].label in eq


def test_equation_family_heads():
    assert "Beta(mu_it" in stmm_model_equation("m1_iid_iid", "beta")
    assert "logit(mu_it)" in stmm_model_equation("m5_seblup_rao_yu_it", "beta")
    g = stmm_model_equation("m5_seblup_rao_yu_it", "gaussian")
    assert "e_it ~ N(0" in g and "rho_S" in g and "phi_T" in g


def test_equation_interaction_types():
    assert "Type I" in stmm_model_equation("m8_iid_iid_int", "beta")
    assert "Type II" in stmm_model_equation("m9_sar_iid_int", "beta")
    assert "Type III" in stmm_model_equation("m10_iid_ar1_int", "beta")
    assert "Type IV" in stmm_model_equation("m12_seblup_rao_yu_it_int", "beta")
    assert "global" in stmm_model_equation("m4_sar_ar1", "beta")
    assert "Type" not in stmm_model_equation("m1_iid_iid", "beta")


def test_equation_errors():
    with pytest.raises(KeyError):
        stmm_model_equation("m99_nope", "beta")
    with pytest.raises(ValueError):
        stmm_model_equation("m1_iid_iid", "poisson")
