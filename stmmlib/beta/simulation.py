"""Simulation and prediction for Beta mixed models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import betaln

from stmmlib.beta.structure import stmm_beta_structure

if TYPE_CHECKING:
    from stmmlib._backend import Backend


def _ar1_chain(T: int, phi: float, sigma_eta: float, rng) -> np.ndarray:
    """Draw an AR(1) chain of length T with innovation SD sigma_eta.

    u_t = phi u_{t-1} + eps_t,  eps_t ~ N(0, sigma_eta^2)
    u_0 ~ N(0, sigma_eta^2/(1-phi^2)),  so stationary variance is
    sigma_eta^2/(1-phi^2) == sigma_marginal^2.

    This matches the covariance implied by `ar1_precision` (Q = sigma_eta^-2 A(phi)).
    """
    s = sigma_eta / np.sqrt(max(1.0 - phi * phi, 1e-12))
    u = np.empty(T, dtype=np.float64)
    u[0] = rng.normal(0.0, s)
    for t in range(1, T):
        u[t] = phi * u[t - 1] + rng.normal(0.0, sigma_eta)
    return u


def _area_re(area: str, m: int, sigma_area: float, rho: float, W, rng) -> np.ndarray:
    """Draw area random effects matching Q_a^{-1} for iid vs sar/seblup.

    - iid:          b ~ N(0, sigma_area^2 I)
    - sar/seblup:   Q_a = sigma_area^-2 (I-rhoW)'(I-rhoW), so
                    b = sigma_area (I-rhoW)^{-1} e,  e ~ N(0,I)
    """
    e = rng.normal(0.0, 1.0, m)
    if area in ("sar", "nested_sar", "seblup_rao_yu"):
        W_arr = W.toarray() if hasattr(W, "toarray") else np.asarray(W, dtype=np.float64)
        A = np.eye(m, dtype=np.float64) - rho * W_arr
        return sigma_area * np.linalg.solve(A, e)
    return sigma_area * e


def _gamma_re(interaction: str, m: int, T: int, sigma_gamma: float,
              rho_gamma: float, phi_gamma: float, W, rng) -> tuple:
    """Draw interaction coefficients gamma ~ N(0, G) for Types I-IV.

    G built on the CONTRAST coordinates (same B as design._build_Z_gamma):
      iid:      kron(I_T, I_m1)
      spatial:  kron(I_T, B'A'AB),  A = I - rho_g W
      temporal: kron(I_m, B'ArB),   Ar = AR(1) correlation
      st:       kron(B'A'AB, B'ArB)
    Returns (gamma vector [n_gamma], Z_gamma matrix [n x n_gamma]).
    """
    from stmmlib.beta.design import _contr_sum, _build_Z_gamma
    B_area = _contr_sum(m)
    B_time = _contr_sum(T)
    area_idx = np.repeat(np.arange(m), T)
    time_idx = np.tile(np.arange(T), m)
    Z_gamma = _build_Z_gamma(area_idx, time_idx, m * T, m, T,
                             interaction, B_area, B_time)
    m1, T1 = B_area.shape[1], B_time.shape[1]
    n_gamma = Z_gamma.shape[1]
    if n_gamma == 0:
        return np.zeros(0), Z_gamma

    def _chol(mat):
        mat = np.asarray(mat, float)
        mat = 0.5 * (mat + mat.T)
        try:
            return np.linalg.cholesky(mat + 1e-10 * np.eye(mat.shape[0]))
        except np.linalg.LinAlgError:
            from stmmlib._utils import near_pd
            from stmmlib._backend import Backend
            pd_mat = np.asarray(near_pd(mat, Backend(gpu=False)), float)
            return np.linalg.cholesky(pd_mat + 1e-10 * np.eye(mat.shape[0]))

    if interaction == "iid":
        L = np.eye(m1 * T)
    elif interaction in ("spatial", "st"):
        W_arr = W.toarray() if hasattr(W, "toarray") else np.asarray(W, float)
        A = np.eye(m, dtype=np.float64) - rho_gamma * W_arr
        La = _chol(B_area.T @ (A.T @ A) @ B_area)
        if interaction == "spatial":
            L = np.kron(np.eye(T), La)
        else:
            Ar = _ar1_corr(T, phi_gamma)
            Lt = _chol(B_time.T @ Ar @ B_time)
            L = np.kron(La, Lt)
    else:  # temporal
        Ar = _ar1_corr(T, phi_gamma)
        Lt = _chol(B_time.T @ Ar @ B_time)
        L = np.kron(np.eye(m), Lt)
    gamma = sigma_gamma * (L @ rng.normal(0.0, 1.0, n_gamma))
    return gamma, Z_gamma


def _ar1_corr(T: int, phi: float) -> np.ndarray:
    """AR(1) correlation matrix (marginal, stationarity phi^|i-j|)."""
    i, j = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    return phi ** np.abs(i - j)


def simulate_beta(
    n: int,
    m: int = 1,
    T: int = 1,
    beta=None,
    phi: float = 20.0,
    sigma_area: float = 1.0,
    sigma_time: float = 1.0,
    rho: float = 0.3,
    ar1_phi: float = 0.5,
    sigma_gamma: float = 0.0,
    rho_gamma: float = 0.3,
    phi_gamma: float = 0.5,
    model: str = "m1_iid_iid",
    W=None,
    X=None,
    seed: int = 42,
) -> dict:
    """Generate synthetic Beta response data from the structural M1-M14 models.

    The random effects are drawn from the covariance Q^{-1} implied by the
    fitted structural model (see `solvers.beta_laplace_structural`), so the
    output can be used to check parameter recovery for each model. Interaction
    models (M8-M12, M13/M14 flat/nested) draw gamma ~ N(0, G) on the contrast
    coordinates and add Z_gamma @ gamma to the linear predictor.

    Balances: n = m * T, ordered area-major (per area rows of T times),
    matching `design.stmm_beta_make_design`.

    Args:
        n: Total sample size (must equal m*T for structural models).
        m: Number of areas.
        T: Number of time periods.
        beta: Fixed-effect coefficients (p,).
        phi: Beta precision parameter.
        sigma_area: Area random-effect SD.
        sigma_time: Time random-effect SD; for AR(1) models this is the
            INNOVATION SD sigma_eta (not the marginal SD).
        rho: SAR spatial parameter (M3/M4/M5).
        ar1_phi: AR(1) time parameter (M2/M4/M5).
        sigma_gamma: Interaction random-effect SD (M8-M14; 0 disables).
        rho_gamma: Spatial dependence of the interaction (M9/M11/M12).
        phi_gamma: Temporal dependence of the interaction (M10/M11/M12).
        model: Model ID (M1-M14 supported; nested requires W-like W=None ok).
        W: Spatial weight matrix (m x m, required for SAR/seblup/spatial-int).
        seed: Random seed.

    Returns:
        dict with y, mu, X, Z, Z_area, Z_cell/Z_time, Z_gamma, b_area, b_time,
        gamma and the true parameter values for recovery checks.

    Raises:
        ValueError: if n != m*T, or W is missing for a spatial model.
    """
    s = stmm_beta_structure(model)
    area = s["area"]
    time = s["time"]
    time_scope = s["time_scope"]

    if n != m * T:
        raise ValueError(f"Structural model '{model}' needs balanced data: n must equal m*T ({m}*{T}={m*T}), got n={n}.")
    if area in ("sar", "nested_sar", "seblup_rao_yu") and W is None:
        raise ValueError(f"Model '{model}' needs spatial weight matrix W.")
    if time == "ar1" and W is None and area not in ("sar", "nested_sar", "seblup_rao_yu"):
        raise ValueError(f"Model '{model}' needs AR(1) which requires W (identity fallback), got None.")

    rng = np.random.default_rng(seed)

    if beta is None:
        beta = np.array([0.5])

    # Area-major ordering: each area occupies T rows (times 0..T-1).
    area_idx = np.repeat(np.arange(m), T)
    time_idx = np.tile(np.arange(T), m)
    n_cell = m * T

    # Design matrices (identical to design.stmm_beta_make_design).
    from scipy.sparse import csr_matrix as _csr

    def _make_Z(idx, n_rows, n_cols):
        row = np.arange(n_rows, dtype=np.int32)
        return _csr((np.ones(n_rows, dtype=np.float64), (row, idx.astype(np.int32))), shape=(n_rows, n_cols))

    if X is None:
        X = np.ones((n, 1), dtype=np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)
    Z_area = _make_Z(area_idx, n, m)

    # Random effects
    b_area = _area_re(area, m, sigma_area, rho, W, rng)

    Z_cell = None
    Z_time = None
    if time == "ar1" and time_scope == "global":
        # Shared AR(1) chain over the T time points (M4).
        b_time = _ar1_chain(T, ar1_phi, sigma_time, rng)
        Z_time = _make_Z(time_idx, n, T).toarray()
        Z = np.hstack([Z_area.toarray(), Z_time])
    else:
        # Per-area AR(1) chain (M2/M5) or iid cell level (M1/M3).
        if time == "ar1":
            cells = [_ar1_chain(T, ar1_phi, sigma_time, rng) for _ in range(m)]
            b_time = np.concatenate(cells)
        else:
            b_time = sigma_time * rng.normal(0.0, 1.0, n_cell)
        Z_cell = _make_Z(area_idx * T + time_idx, n, n_cell).toarray()
        Z = np.hstack([Z_area.toarray(), Z_cell])

    # Linear predictor
    z_dep = Z_time if Z_time is not None else Z_cell
    eta = X @ beta + Z_area.toarray() @ b_area + z_dep @ b_time

    # Interaction (M8-M12, M13/M14): Z_gamma @ gamma with gamma ~ N(0, G)
    gamma = np.zeros(0)
    Z_gamma = None
    if s["interaction"] != "none":
        gamma, Z_gamma = _gamma_re(s["interaction"], m, T, sigma_gamma,
                                   rho_gamma, phi_gamma, W, rng)
        if gamma.size:
            eta = eta + np.asarray(Z_gamma, float) @ gamma

    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-15, 1.0 - 1e-15)

    # Beta response
    a = mu * phi
    b_val = (1 - mu) * phi
    y = rng.beta(a, b_val)

    return {
        "y": y,
        "mu": mu,
        "X": X,
        "Z": Z,
        "Z_area": Z_area.toarray(),
        "Z_time": Z_time,
        "Z_cell": Z_cell,
        "b_area": b_area,
        "b_time": b_time,
        "gamma": gamma,
        "Z_gamma": Z_gamma,
        "area_idx": area_idx,
        "time_idx": time_idx,
        "beta": beta,
        "phi": phi,
        "sigma_area": sigma_area,
        "sigma_time": sigma_time,
        "sigma_gamma": sigma_gamma if s["interaction"] != "none" else None,
        "rho": rho,
        "ar1_phi": ar1_phi,
        "rho_gamma": rho_gamma,
        "phi_gamma": phi_gamma,
        "model": model,
        "n": n,
        "m": m,
        "T": T,
    }


def simulate_beta_prediction(result, newdata=None) -> dict:
    """Generate predictions from fitted Beta model.

    Args:
        result: BetastmmResult.
        newdata: Optional DataFrame for prediction.

    Returns:
        dict with y_pred, mu_pred, se_pred.
    """
    beta = result.beta
    design = result.design
    # shortcut: design tidak pernah menyimpan X; matrix design ada di _opt
    X = design.get("X") or result._opt.get("X")
    if X is None:
        n = len(result._opt.get("y", []))
        X = np.ones((n, 1), dtype=np.float64)

    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-15, 1.0 - 1e-15)

    return {
        "mu_pred": mu,
        "se_pred": np.full_like(mu, np.nan),
    }


if __name__ == "__main__":
    # ponytail: on-change self-check. Simulates M1-M5 and verifies the
    # generated design/ordering is consistent (area-major, balanced, and the
    # covariance of the random effects matches Q^{-1} for the SAR case).
    m, T = 30, 8
    rng = np.random.default_rng(1)
    W = np.ones((m, m)); np.fill_diagonal(W, 0)
    W = W / W.sum(axis=1, keepdims=True)
    rho, ar1_phi = 0.5, 0.6

    # SAR area covariance sanity: cov(b_area) == sigma_a^2 (I-rhoW)^{-1}(I-rhoW)^{-T}
    sigma_a = 1.5
    A = np.eye(m) - rho * W
    cov_true = sigma_a**2 * np.linalg.inv(A.T @ A)
    draws = np.stack([_area_re("sar", m, sigma_a, rho, W, rng) for _ in range(2000)])
    cov_emp = np.cov(draws.T)
    assert np.allclose(cov_true[0, :3], cov_emp[0, :3], atol=0.4), "SAR cov mismatch"

    for model in ["m1_iid_iid", "m2_iid_ar1", "m3_sar_iid", "m4_sar_ar1", "m5_seblup_rao_yu_it"]:
        sim = simulate_beta(m * T, m=m, T=T, model=model, W=W, rho=rho, ar1_phi=ar1_phi,
                            sigma_area=sigma_a, sigma_time=0.8, seed=2)
        assert sim["y"].shape == (m * T,)
        assert 0 < sim["y"].min() and sim["y"].max() < 1
        # ordering: area-major, cell_idx = area*T + time
        ci = sim["area_idx"] * T + sim["time_idx"]
        assert np.array_equal(ci, np.arange(m * T)), f"{model} ordering wrong"
        assert sim["Z"].shape[0] == m * T
        print(f"{model}: ok (Z {sim['Z'].shape})")
    print("simulate_beta self-check OK")

