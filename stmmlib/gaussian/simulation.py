"""Simulation: generate synthetic spatio-temporal panel data.

Generates data from the DGP of models M1-M4 and SEBLUP5, with known
parameters for simulation studies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stmmlib.gaussian.covariance import build_G_area, build_G_time, expand_it

if TYPE_CHECKING:
    from stmmlib._backend import Backend


def generate_W(m: int, structure: str = "queen", normalized: bool = True, seed: int = None, backend: Backend = None):
    """Generate spatial adjacency/weight matrix.

    Args:
        m: Number of areas.
        structure: "queen", "rook", or "random".
        normalized: Row-normalize.
        seed: Random seed.
        backend: Array backend.

    Returns:
        W: m x m weight matrix.
    """
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu=False)
    xp = backend.xp

    rng = np.random.default_rng(seed)
    W = np.zeros((m, m), dtype=np.float64)

    if structure == "queen":
        # Each area connected to neighbors (simple ring)
        for i in range(m):
            W[i, (i + 1) % m] = 1.0
            W[i, (i - 1) % m] = 1.0
            if i + m // 2 < m:
                W[i, (i + m // 2) % m] = 0.5
    elif structure == "rook":
        for i in range(m):
            W[i, (i + 1) % m] = 1.0
            W[i, (i - 1) % m] = 1.0
    elif structure == "random":
        W = rng.random((m, m))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        W[W < 0.3] = 0

    if normalized:
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums

    return xp.asarray(W, dtype=xp.float64)


def simulate_stlmm(
    m: int = 50,
    T_periods: int = 10,
    p: int = 2,
    beta=None,
    sigma2_v: float = 1.0,
    sigma2_u: float = 0.5,
    sigma2_eta: float = 0.3,
    sigma2_e: float = 0.5,
    phi: float = 0.5,
    rho: float = 0.3,
    W=None,
    w_structure: str = "queen",
    seed: int = None,
    verbose: bool = False,
    backend: Backend = None,
) -> dict:
    """Simulate balanced panel data for models M1-M4.

    DGP: y_it = X beta + area_effect + time_effect + epsilon

    Args:
        m: Number of areas.
        T_periods: Number of time periods.
        p: Number of fixed effects (including intercept).
        beta: Fixed-effect coefficients (length p).
        sigma2_v: Area variance.
        sigma2_u: Time variance (IID) or not used.
        sigma2_eta: Innovation variance (AR1).
        sigma2_e: Residual variance.
        phi: AR(1) coefficient.
        rho: Spatial autocorrelation.
        W: Spatial weight matrix (generated if None).
        w_structure: W generation structure.
        seed: Random seed.
        verbose: Print DGP info.
        backend: Array backend.

    Returns:
        dict with data, W, true parameters.
    """
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu=False)
    xp = backend.xp

    rng = np.random.default_rng(seed)
    n = m * T_periods

    if beta is None:
        beta = np.array([1.0] + [0.5] * (p - 1), dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    # Fixed effects
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    eta_fixed = X @ beta

    # Area effects
    if rho != 0 and W is not None:
        G_a = build_G_area("sar", sigma2_v, m, W, rho, backend)
    else:
        G_a = build_G_area("iid", sigma2_v, m, backend=backend)
    if backend.is_sparse(G_a):
        G_a = G_a.toarray()
    area_effects = rng.multivariate_normal(np.zeros(m), G_a)
    area_rep = np.repeat(area_effects, T_periods)

    # Time effects (AR1)
    time_effects = np.zeros(n, dtype=np.float64)
    if sigma2_eta > 0 and phi != 0:
        G_t = build_G_time("ar1", T_periods, sigma2_u, sigma2_eta, phi, backend)
        if backend.is_sparse(G_t):
            G_t = G_t.toarray()
        for i in range(m):
            te = rng.multivariate_normal(np.zeros(T_periods), G_t)
            time_effects[i * T_periods:(i + 1) * T_periods] = te
    elif sigma2_u > 0:
        time_effects = np.tile(rng.standard_normal(T_periods) * np.sqrt(sigma2_u), m)

    # Residuals
    epsilon = rng.standard_normal(n) * np.sqrt(sigma2_e)

    # Response
    y = eta_fixed + area_rep + time_effects + epsilon

    # Build DataFrame
    import pandas as pd
    area_labels = [f"area_{i}" for i in range(m)]
    time_labels = list(range(T_periods))
    data = pd.DataFrame({
        "y": y,
        "area": np.repeat(area_labels, T_periods),
        "time": np.tile(time_labels, m),
    })
    for j in range(1, p):
        data[f"x{j}"] = X[:, j]

    if W is None:
        W = generate_W(m, w_structure, seed=seed, backend=backend)

    if verbose:
        print(f"Simulated: n={n}, m={m}, T={T_periods}, p={p}")
        print(f"  beta={beta}, sigma2_v={sigma2_v}, sigma2_u={sigma2_u}")
        print(f"  sigma2_eta={sigma2_eta}, phi={phi}, rho={rho}, sigma2_e={sigma2_e}")

    return {
        "data": data,
        "W": W,
        "X": X,
        "y": y,
        "beta_true": beta,
        "sigma2_v_true": sigma2_v,
        "sigma2_u_true": sigma2_u,
        "sigma2_eta_true": sigma2_eta,
        "sigma2_e_true": sigma2_e,
        "phi_true": phi,
        "rho_true": rho,
        "m": m,
        "T": T_periods,
        "n": n,
        "p": p,
    }
