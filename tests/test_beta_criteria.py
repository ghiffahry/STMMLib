"""Library cAIC/cR2 (stmmlib.beta.criteria): sane on healthy fits, NaN on garbage."""
import numpy as np

from conftest import fit_beta
from stmmlib.beta.criteria import beta_criteria


def test_criteria_healthy_m1(df_flat):
    r = fit_beta(df_flat, "m1_iid_iid", max_iter=200)
    cr = beta_criteria(r)
    assert cr is not None and np.isfinite(cr["cAIC"])
    p_dof = r._opt["X"].shape[1] + r._opt["Z"].shape[1]
    assert 0 <= cr["df_hat"] <= p_dof
    assert cr["ll_cond"] >= r._opt["log_lik"] - max(50.0, 0.05 * abs(r._opt["log_lik"]))
    assert 0 <= cr["cR2_marginal"] <= cr["cR2_conditional"] <= 1.0


def test_criteria_refuses_stale(df_flat):
    r = fit_beta(df_flat, "m1_iid_iid", max_iter=200)
    p = len(r._opt["beta_hat"])
    shift = np.zeros(p)
    shift[0] = 3.0  # mu -> ~0.95 flat: grossly miscalibrated
    r._opt["b_hat"] = np.zeros_like(r._opt["b_hat"])
    r._opt["beta_hat"] = r._opt["beta_hat"] + shift
    r._inf["estimate"][:p] = r._inf["estimate"][:p] + shift
    cr = beta_criteria(r)
    assert cr is not None and not np.isfinite(cr["cAIC"]), cr


def test_criteria_nested_nan_not_garbage(df_nested):
    r = fit_beta(df_nested, "m6_nested_iid_iid", max_iter=200)
    cr = beta_criteria(r)
    assert cr is not None and not np.isfinite(cr["cAIC"])


def test_criteria_precision_mismatch_raises(df_flat):
    import pytest

    from stmmlib.beta.criteria import PrecisionShapeError

    r = fit_beta(df_flat, "m1_iid_iid", max_iter=200)
    r._opt["des"]["blocks"]["area_time"].append(999)
    with pytest.raises(PrecisionShapeError):
        beta_criteria(r)
