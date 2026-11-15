"""Public entry points reject typos and malformed frames before fitting.

Each case used to be silent: family='poisson' fitted Gaussian, method='reml'
fitted ML, duplicate columns produced inf, and a Beta SAR without W returned
NaN. The guards must name the argument, not fail inside the optimizer.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import fit_beta, fit_gauss
from stmmlib import stmm, stmm_available_models


# NB: " beta " is accepted as an alias (input is stripped and lower-cased).
@pytest.mark.parametrize("family", ["poisson", "betaa", None, 1])
def test_unknown_family_is_rejected(df_gauss, family):
    with pytest.raises(ValueError, match="family"):
        stmm(data=df_gauss, model="m1_iid_iid", family=family, gpu="no",
             n_starts=1)


@pytest.mark.parametrize("family", ["normal", "GAUSSIAN"])
def test_gaussian_family_aliases_are_accepted(df_gauss, family):
    r = stmm(data=df_gauss, model="m1_iid_iid", family=family, gpu="no",
             n_starts=1)
    assert np.isfinite(r["log_lik"])


def test_beta_family_alias_is_accepted(df_flat):
    r = stmm(data=df_flat, model="m1_iid_iid", family="Beta", gpu="no",
             n_starts=1)
    assert np.isfinite(r.log_lik)


def test_gaussian_method_is_case_insensitive(df_gauss):
    a = fit_gauss(df_gauss, "m1_iid_iid", method="REML")["log_lik"]
    b = fit_gauss(df_gauss, "m1_iid_iid", method="reml")["log_lik"]
    assert a == pytest.approx(b, rel=1e-12)


@pytest.mark.parametrize("method", ["Laplace", "poisson", "REMLX"])
def test_gaussian_unknown_method_is_rejected(df_gauss, method):
    with pytest.raises(ValueError, match="method"):
        fit_gauss(df_gauss, "m1_iid_iid", method=method)


def test_beta_rejects_reml_and_accepts_ml(df_flat):
    with pytest.raises(ValueError, match="method"):
        fit_beta(df_flat, "m1_iid_iid", method="REML")
    r = fit_beta(df_flat, "m1_iid_iid", method="ML")
    assert np.isfinite(r.log_lik)


def test_available_models_accepts_gaussian_alias():
    assert stmm_available_models("gaussian") == stmm_available_models("normal")
    assert "m1_iid_iid" in stmm_available_models("Gaussian")


def test_duplicate_columns_are_rejected(df_gauss, df_flat):
    dup = pd.concat([df_gauss, df_gauss[["x1"]]], axis=1)
    with pytest.raises(ValueError, match="duplikat"):
        fit_gauss(dup, "m1_iid_iid")
    dupb = pd.concat([df_flat, df_flat[["x1"]]], axis=1)
    with pytest.raises(ValueError, match="duplikat"):
        fit_beta(dupb, "m1_iid_iid")


@pytest.mark.parametrize("family", ["gaussian", "beta"])
def test_missing_and_empty_data_are_named(family):
    with pytest.raises(ValueError, match="data"):
        stmm(data=None, model="m1_iid_iid", family=family, gpu="no",
             n_starts=1)
    empty = pd.DataFrame(columns=["y", "area", "time", "x1"])
    with pytest.raises(ValueError, match="kosong"):
        stmm(data=empty, model="m1_iid_iid", family=family, gpu="no",
             n_starts=1)


def test_gaussian_nan_columns_are_named(df_gauss):
    bad = df_gauss.copy()
    bad.loc[0, "x1"] = np.nan
    with pytest.raises(ValueError, match="x1"):
        fit_gauss(bad, "m1_iid_iid")
    bad = df_gauss.copy()
    bad.loc[0, "y"] = np.nan
    with pytest.raises(ValueError, match="kolom 'y'"):
        fit_gauss(bad, "m1_iid_iid")


@pytest.mark.parametrize("family", ["gaussian", "beta"])
def test_unknown_model_is_named(family, df_gauss, df_flat):
    data = df_gauss if family == "gaussian" else df_flat
    with pytest.raises(ValueError, match="m99"):
        stmm(data=data, model="m99", family=family, gpu="no", n_starts=1)


def test_beta_sar_requires_W(df_flat):
    with pytest.raises(ValueError, match="W"):
        fit_beta(df_flat, "m3_sar_iid")


def test_interaction_typo_is_rejected(df_gauss):
    with pytest.raises(ValueError, match="interaction"):
        fit_gauss(df_gauss, "m1_iid_iid", spatial="iid", temporal="iid",
                  interaction="tempoarl")


@pytest.mark.parametrize("family", ["gaussian", "beta"])
@pytest.mark.parametrize("n_starts", [0, -2])
def test_n_starts_must_be_positive(family, n_starts, df_gauss, df_flat):
    data = df_gauss if family == "gaussian" else df_flat
    with pytest.raises(ValueError, match="n_starts"):
        stmm(data=data, model="m1_iid_iid", family=family, gpu="no",
             n_starts=n_starts)


def test_beta_flat_model_warns_about_swallowed_prov_predictor(df_flat):
    d = df_flat.assign(prov=np.arange(len(df_flat), dtype=float))
    with pytest.warns(UserWarning, match="prov"):
        r = fit_beta(d, "m1_iid_iid")
    assert "prov" not in set(r._opt["des"]["x_cols"])


def test_beta_unknown_kwarg_is_rejected(df_flat):
    with pytest.raises(ValueError, match="argumen tidak dikenal"):
        fit_beta(df_flat, "m1_iid_iid", n_start=1)


def test_stmm_requires_a_known_model(df_gauss):
    with pytest.raises(ValueError, match="model"):
        stmm(data=df_gauss, model=None, family="gaussian", gpu="no",
             n_starts=1)


def test_wrong_grouping_columns_are_named(df_gauss, df_flat):
    with pytest.raises(ValueError, match="wilayah"):
        fit_gauss(df_gauss, "m1_iid_iid", area_col="wilayah")
    with pytest.raises(ValueError, match="thn"):
        fit_gauss(df_gauss, "m1_iid_iid", time_col="thn")
    with pytest.raises(ValueError, match="wilayah"):
        fit_beta(df_flat, "m1_iid_iid", area_col="wilayah")
    with pytest.raises(ValueError, match="thn"):
        fit_beta(df_flat, "m1_iid_iid", time_col="thn")


def test_explicit_wrong_response_col_is_not_silently_y(df_gauss, df_flat):
    with pytest.raises(ValueError, match="y2"):
        fit_gauss(df_gauss, "m1_iid_iid", response_col="y2")
    with pytest.raises(ValueError, match="y2"):
        fit_beta(df_flat, "m1_iid_iid", response_col="y2")


def test_gaussian_W_shape_mismatch_is_named(df_gauss):
    with pytest.raises(ValueError, match="W"):
        fit_gauss(df_gauss, "m3_sar_iid", W=np.eye(3))


def test_gaussian_nested_requires_grouping_columns(df_gauss):
    with pytest.raises(ValueError, match="prov"):
        fit_gauss(df_gauss, "m6_nested_iid_iid")


@pytest.mark.parametrize("family", ["gaussian", "beta"])
def test_unknown_inference_is_rejected(family, df_gauss, df_flat):
    data = df_gauss if family == "gaussian" else df_flat
    with pytest.raises(ValueError, match="inference"):
        stmm(data=data, model="m1_iid_iid", family=family, gpu="no",
             n_starts=1, inference="foo")
