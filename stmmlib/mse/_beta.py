"""Beta family internals for the parametric bootstrap MSE."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from stmmlib._backend import Backend
from stmmlib.beta.precision import ar1_precision
from stmmlib.beta.solvers import beta_laplace_structural
from stmmlib.mse._common import (
    _N_DRAWS, _draw_from_precision, _invlogit, _row_means, _to_np,
)


def beta_area_labels(fit, m):
    labels = getattr(fit, "_area_labels", None)
    if isinstance(labels, dict) and labels:
        return [labels.get(i, i) for i in range(m)]
    return list(range(m))


def beta_Q_total(area_type, time_type, time_scope, logs, raw_dep, m, T, n_time,
                 W, rho_bound, phi_bound):
    backend = Backend(gpu=False)
    rh = logs[0] if len(logs) > 0 else 0.0
    if area_type in ("sar", "seblup_rao_yu"):
        rho = rho_bound * np.tanh(raw_dep[0]) if len(raw_dep) > 0 else 0.0
        A = np.eye(m, dtype=np.float64) - rho * np.asarray(W, dtype=np.float64)
        Q_a = np.exp(-2.0 * rh) * (A.T @ A)
    else:
        Q_a = np.exp(-2.0 * rh) * np.eye(m, dtype=np.float64)

    if time_type == "ar1":
        ph_i = 1 if area_type in ("sar", "seblup_rao_yu") else 0
        phi_t = phi_bound * np.tanh(raw_dep[ph_i]) if len(raw_dep) > ph_i else 0.0
        from scipy.sparse import block_diag
        if time_scope == "area_specific":
            per_area = [ar1_precision(T, logs[1], phi_t, backend,
                                       phi_bound=phi_bound) for _ in range(m)]
            Q_t = block_diag(per_area, format="csr").toarray()
        else:
            Q_t = ar1_precision(T, logs[1], phi_t, backend,
                                phi_bound=phi_bound).toarray()
    else:
        lt = logs[1] if len(logs) > 1 else 0.0
        Q_t = np.exp(-2.0 * lt) * np.eye(n_time, dtype=np.float64)
    return np.block([[Q_a, np.zeros((m, n_time))], [np.zeros((n_time, m)), Q_t]])


def simulate_beta(rng, X, Z, beta0, logs, raw_dep, logphi, area_type,
                  time_type, time_scope, m, T, W, rho_bound, phi_bound):
    n_time = Z.shape[1] - m
    Q_total = beta_Q_total(area_type, time_type, time_scope, logs, raw_dep,
                           m, T, n_time, W, rho_bound, phi_bound)
    b_draw = _draw_from_precision(rng, Q_total)
    eta = X @ beta0 + Z @ b_draw
    mu = _invlogit(eta)
    phi = np.exp(logphi)
    y_star = rng.beta(np.clip(mu * phi, 1e-9, None),
                      np.clip((1.0 - mu) * phi, 1e-9, None))
    y_star = np.clip(y_star, 1e-6, 1.0 - 1e-6)
    return eta, y_star


def predict_beta_area_ebp(refit, rng, area_idx, m, n_draws=_N_DRAWS):
    X2 = np.asarray(refit["X"], dtype=np.float64)
    Z2 = np.asarray(refit["Z"], dtype=np.float64)
    beta = np.asarray(refit["beta_hat"], dtype=np.float64).ravel()
    b = np.asarray(refit["b_hat"], dtype=np.float64).ravel()
    L = np.asarray(refit["H_b_chol"], dtype=np.float64)
    if L is None or b.size == 0 or L.shape[0] != b.size:
        raise ValueError("refit missing the Laplace posterior curvature")
    eta_fixed = X2 @ beta
    s = np.zeros(eta_fixed.shape[0], dtype=np.float64)
    for _ in range(n_draws):
        bd = b + solve_triangular(L.T, rng.standard_normal(b.size), lower=False)
        s += _invlogit(eta_fixed + Z2 @ bd)
    return _row_means(s / n_draws, area_idx, m)


def beta_fit_predictor(fit, rng, area_idx, m, n_draws=_N_DRAWS):
    opt = fit._opt
    try:
        return predict_beta_area_ebp(opt, rng, area_idx, m, n_draws)
    except Exception:
        mu_hat = np.asarray(opt.get("mu_hat", []), dtype=np.float64)
        if mu_hat.size:
            return _row_means(mu_hat, area_idx, m)
        raise ValueError(
            "MSE bootstrap: the Beta fit has no usable posterior or mu_hat. "
            "Refit with compute_posterior=True for the EBP predictor.")
