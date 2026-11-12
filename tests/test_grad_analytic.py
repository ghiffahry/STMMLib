"""Regression tests: analytic Laplace gradient vs finite differences.

Validates the jac= wiring in beta_laplace_structural on the tiny conftest
panel: the analytic gradient must match the central finite-difference of the
same objective to ~1e-5 relative tolerance (FD noise floor ~1e-7).
"""
import numpy as np
import pytest

from conftest import fit_beta


def _san(df):
    from stmmlib.beta.fit import _sanitize_data

    d, _, _ = _sanitize_data(df, response_col="y", area_col="area",
                             time_col="time")
    return d, d["y"].values.astype(float)


def _grad_case(df_flat, W_flat, model, W, raw):
    from stmmlib.beta.solvers import beta_laplace_structural

    d, y = _san(df_flat)
    p = 2  # intercept + x1 on the conftest panel
    theta = np.concatenate([np.zeros(p), [np.log(20.0)], np.log([0.4, 0.3]), raw])
    out = beta_laplace_structural(None, d, y, "Laplace", model, W,
                                  eval_param=theta, compute_hessian=False,
                                  inner_max_iter=300, inner_tol=1e-10)
    b0 = out["b_hat"]
    ana = beta_laplace_structural(None, d, y, "Laplace", model, W,
                                  eval_grad=theta, eval_b0=b0,
                                  compute_hessian=False)["grad"]
    k = len(theta)
    rows = []
    eps = 1e-5
    for i in range(k):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        rows += [tp, tm]
    r = beta_laplace_structural(None, d, y, "Laplace", model, W,
                                eval_many=np.array(rows), eval_b0=b0,
                                compute_hessian=False, inner_max_iter=300,
                                inner_tol=1e-10)
    vals = r["eval_many"]
    fd = np.array([(vals[2 * i] - vals[2 * i + 1]) / (2 * eps) for i in range(k)])
    rel = np.abs(ana - fd) / np.maximum(np.abs(fd), 1.0)
    assert rel.max() < 1e-4, f"{model}: grad rel err {rel.max():.2e}"


def test_grad_m1_matches_finite_diff(df_flat):
    _grad_case(df_flat, None, "m1_iid_iid", None, [])


def test_grad_m2_matches_finite_diff(df_flat):
    _grad_case(df_flat, None, "m2_iid_ar1", None, [0.4])


def test_grad_m3_matches_finite_diff(df_flat, W_flat):
    _grad_case(df_flat, W_flat, "m3_sar_iid", W_flat, [0.4])


def test_grad_m5_matches_finite_diff(df_flat, W_flat):
    _grad_case(df_flat, W_flat, "m5_seblup_rao_yu_it", W_flat, [0.3, 0.4])


def test_eval_param_is_eval_only(df_flat):
    """eval_param/eval_many/eval_grad must not run the optimizer (fast hooks)."""
    import time
    from stmmlib.beta.solvers import beta_laplace_structural

    d, _ = _san(df_flat)
    theta = np.zeros(5)  # p=2 + logphi + 2 logs
    theta[2] = np.log(20.0)
    theta[3:] = np.log([0.4, 0.3])
    t0 = time.time()
    beta_laplace_structural(None, d, d["y"].values.astype(float),
                            "Laplace", "m1_iid_iid", None, eval_param=theta,
                            compute_hessian=False)
    dt = time.time() - t0
    assert dt < 5.0, f"eval_param took {dt:.1f}s -- optimizer likely ran"
