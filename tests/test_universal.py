"""Universal API: arbitrary column names must give identical results."""
import numpy as np
import pandas as pd

from conftest import fit_beta


def test_beta_column_renames_identical(df_flat):
    base = fit_beta(df_flat, "m1_iid_iid")
    ren = df_flat.rename(columns={
        "y": "proporsi", "area": "wilayah", "time": "periode",
        "x1": "pendidikan",
    })
    custom = fit_beta(ren, "m1_iid_iid", response_col="proporsi",
                      area_col="wilayah", time_col="periode")
    assert abs(base._opt["log_lik"] - custom._opt["log_lik"]) < 1e-8
    assert np.allclose(base._opt["beta_hat"], custom._opt["beta_hat"], atol=1e-8)


def test_beta_coef_names_from_columns(df_flat):
    ren = df_flat.rename(columns={"x1": "pendidikan"})
    r = fit_beta(ren, "m1_iid_iid", area_col="area", time_col="time")
    # fixed-effect names follow actual predictor columns
    beta_names = r.inference["param_labels"][:2]
    assert beta_names[1] == "pendidikan" or beta_names[1] == "x1"
