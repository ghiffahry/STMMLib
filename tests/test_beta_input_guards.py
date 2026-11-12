"""New data must fail loudly at the door, not silently in the optimizer."""
import numpy as np
import pytest

from conftest import fit_beta


def test_response_outside_zero_one_is_rejected(df_flat):
    """A percentage response (0-100) is the most common new-data mistake."""
    bad = df_flat.copy()
    bad["y"] = bad["y"] * 100.0
    with pytest.raises(ValueError, match="0,1"):
        fit_beta(bad, "m1_iid_iid")


def test_nan_response_is_rejected(df_flat):
    bad = df_flat.copy()
    bad.loc[0, "y"] = np.nan
    with pytest.raises(ValueError):
        fit_beta(bad, "m1_iid_iid")


def test_nan_predictor_is_rejected(df_flat):
    bad = df_flat.copy()
    bad.loc[0, "x1"] = np.nan
    with pytest.raises(ValueError):
        fit_beta(bad, "m1_iid_iid")


def test_unbalanced_panel_is_reported_not_blocked(df_flat):
    """The design supports an unbalanced panel: an empty cell carries no
    information about its own effect. The state is reported, not blocked."""
    bad = df_flat.iloc[:-1].copy()
    with pytest.warns(UserWarning, match="seimbang"):
        fit = fit_beta(bad, "m1_iid_iid")
    assert np.isfinite(fit.log_lik)


def test_balanced_panel_still_fits(df_flat):
    fit = fit_beta(df_flat, "m1_iid_iid")
    assert np.isfinite(fit.log_lik)


def test_a_predictor_named_area_is_kept_when_the_grouping_column_is_renamed(df_flat):
    """The skip list must not hold bare "area"/"time"/"prov"/"kab".

    Those literals silently dropped a user predictor that happened to carry
    one of those names, so a different dataset lost a variable without any
    message.
    """
    df = df_flat.rename(columns={"area": "wilayah"})
    df["area"] = np.arange(len(df), dtype=float)
    fit = fit_beta(df, "m1_iid_iid", area_col="wilayah", time_col="time")
    x_cols = list(fit._opt["des"]["x_cols"])
    assert "area" in x_cols
    assert "wilayah" not in x_cols
    assert "x1" in x_cols


def test_a_predictor_named_y_collides_with_the_internal_response(df_flat):
    """The collision must be named, not reported as a broken response."""
    df = df_flat.rename(columns={"y": "miskin"})
    df["y"] = np.linspace(0.01, 0.99, len(df))
    with pytest.raises(ValueError, match="bertabrakan"):
        fit_beta(df, "m1_iid_iid", response_col="miskin")
