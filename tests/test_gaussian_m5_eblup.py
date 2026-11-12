"""Gaussian M5 (SEBLUP Rao-Yu IT) must return an EBLUP, not just GLS.

The wrapper's v_hat/u_hat are checked against a direct Henderson (mixed model
equation) solve at the same theta_hat. Both are the same estimator, so they
must agree to numerical precision.
"""
import numpy as np
import pytest

from stmmlib.gaussian.covariance import (
    G_ar1_core, build_G_area, build_V_seblup5,
)
from tests.conftest import fit_gauss


def test_m5_seblup_returns_eblup(df_gauss, W_flat):
    fit = fit_gauss(df_gauss, "m5_seblup_rao_yu_it", W=W_flat)
    assert "blup" in fit and "v_hat" in fit["blup"]
    th = fit["theta"]
    m, T = fit["m"], fit["T"]
    assert fit["blup"]["v_hat"].shape == (m,)
    assert fit["blup"]["u_hat"].shape == (m * T,)

    # Direct MME at theta_hat
    prep = fit["data"]
    y = df_gauss["y"].to_numpy(float)
    X = np.column_stack([np.ones(len(y)),
                         df_gauss["x1"].to_numpy(float)])
    Za = prep["Z_area"]
    Za = Za.toarray() if hasattr(Za, "toarray") else np.asarray(Za, float)
    cidx = (np.asarray(prep["area_index"], int) * T
            + np.asarray(prep["time_index"], int))
    Zc = np.zeros((len(y), m * T))
    Zc[np.arange(len(y)), cidx] = 1.0
    Z = np.hstack([Za, Zc])

    from stmmlib._backend import Backend
    be = Backend(gpu=False)
    Ga = np.asarray(build_G_area("sar", th["sigma2_v"], m, W_flat, th["rho"], be))
    Gu = np.asarray(G_ar1_core(T, th["sigma2_eta"], th["phi"], be))
    G = np.zeros((m + m * T, m + m * T))
    G[:m, :m] = Ga
    G[m:, m:] = np.kron(np.eye(m), Gu)

    s2e = th["sigma2_e"]
    Ginv = np.linalg.inv(G) * s2e
    A = np.block([[X.T @ X, X.T @ Z],
                  [Z.T @ X, Z.T @ Z + Ginv]])
    rhs = np.concatenate([X.T @ y, Z.T @ y])
    sol = np.linalg.solve(A, rhs)

    assert np.allclose(sol[:X.shape[1]], fit["beta_hat"], atol=1e-7)
    assert np.allclose(sol[X.shape[1]:X.shape[1] + m],
                       fit["blup"]["v_hat"], atol=1e-7)
    assert np.allclose(sol[X.shape[1] + m:],
                       fit["blup"]["u_hat"], atol=1e-6)

    # V used for the BLUP is the same V the fit reports (sanity, no refit).
    V = build_V_seblup5(prep, th, "none", "sar", be)
    assert V.shape == (len(y), len(y))
