"""Precision (inverse-covariance) matrix construction for Beta models.

Beta models work with precision Q rather than covariance G=Q^{-1},
because AR(1) precision is tridiagonal (sparse, efficient) and SAR
precision factorizes as (I-rhoW)'(I-rhoW).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stmmlib._backend import Backend


def ar1_precision(n: int, log_sd: float, raw_phi: float, backend: Backend,
                  phi_bound: float = 0.95):
    """AR(1) innovation precision matrix (tridiagonal).

    Q = sigma_eta^{-2} * A(phi) where:
        A_tt = 1 (t=0,n-1), 1+phi^2 (interior)
        A_{t,t+1} = A_{t+1,t} = -phi

    Args:
        n: Number of time periods.
        log_sd: log(sigma_eta), innovation SD.
        raw_phi: Unconstrained AR(1) parameter.
        backend: Array backend.
        phi_bound: Stationarity bound for |phi| (default 0.95).

    Returns:
        Sparse n x n precision matrix.
    """
    xp = backend.xp
    phi = phi_bound * np.tanh(raw_phi)  # constrain to (-phi_bound, phi_bound)
    sigma2_inv = np.exp(-2.0 * log_sd)

    diag_main = np.full(n, 1.0 + phi**2, dtype=np.float64)
    diag_main[0] = 1.0
    diag_main[-1] = 1.0

    diag_off = np.full(n - 1, -phi, dtype=np.float64)

    # Build sparse tridiagonal
    from scipy.sparse import diags as sp_diags
    A = sp_diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(n, n), format="csr")
    Q = sigma2_inv * A
    return backend.sparse.csr_matrix(Q) if backend.device == "gpu" else Q


def rho_bound(W, backend: Backend, margin: float = 0.95) -> float:
    """Compute SAR spectral bound: margin / max|eigen(W)|.

    Args:
        W: Spatial weight matrix.
        backend: Array backend.
        margin: Stationarity margin (default 0.95).

    Returns:
        Upper bound for |rho|.
    """
    xp = backend.xp
    if backend.is_sparse(W):
        W_arr = W.toarray()
    else:
        W_arr = xp.asarray(W, dtype=xp.float64)

    # BUG-B1 fix: eigvalsh assumes a SYMMETRIC matrix (reads one triangle
    # only) -- row-standardized KNN W is asymmetric, so the "bound" was a
    # meaningless surrogate (measured: 0.770 vs the true 0.950 on KNN-4).
    # The correct stability bound is the spectral radius: max |eig(W)|.
    if backend.device == "gpu":
        eigenvalues = xp.linalg.eigvals(xp.asarray(W_arr))
        max_abs_eig = float(xp.max(xp.abs(eigenvalues)))
    else:
        max_abs_eig = float(np.max(np.abs(np.linalg.eigvals(np.asarray(W_arr)))))
    if max_abs_eig < 1e-10:
        return margin
    return margin / max_abs_eig


def effective_phi(raw_phi: float, phi_bound: float = 0.95) -> float:
    """EFFECTIVE AR(1) phi of the frozen model mapping.

    BUG-B3 fix, single source of truth: the model passes phi_t = phi_bound *
    tanh(raw) into ar1_precision, which squashes AGAIN (phi_bound * tanh(raw)).
    Display/SE/interval code must use this double squash, not a single one
    (old display showed phi 0.9025 where the model really had 0.733).
    """
    u = phi_bound * np.tanh(raw_phi)
    return phi_bound * np.tanh(u)


def precision(
    s: dict,
    W=None,
    raw: dict = None,
    logs: dict = None,
    n_area: int = 0,
    n_time: int = 0,
    rho_bound_val: float = 0.95,
    backend: Backend = None,
):
    """Assemble full block-diagonal precision matrix.

    Q = blkdiag(Q_area, Q_time) for non-nested.
    Q = blkdiag(Q_prov, Q_kab, Q_time) for nested.

    Args:
        s: Structure dict from stmm_beta_structure.
        W: Spatial weight matrix.
        raw: Raw parameter dict (raw_rho, raw_phi, etc.).
        logs: Log-scale parameters (log_sigma_area, log_sigma_time).
        n_area: Number of areas.
        n_time: Number of time periods.
        rho_bound_val: Spectral bound for rho.
        backend: Array backend.

    Returns:
        Sparse block-diagonal precision matrix.
    """
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu=False)

    xp = backend.xp
    area = s.get("area", "iid")
    time = s.get("time", "iid")

    blocks = []

    # Area precision
    if area in ("iid",):
        log_sigma_area = logs.get("log_sigma_area", 0.0) if logs else 0.0
        Q_area = xp.exp(-2.0 * log_sigma_area) * xp.eye(n_area, dtype=xp.float64)
        blocks.append(Q_area)
    elif area in ("sar",):
        log_sigma_area = logs.get("log_sigma_area", 0.0) if logs else 0.0
        raw_rho = raw.get("raw_rho", 0.0) if raw else 0.0
        rho = rho_bound_val * np.tanh(raw_rho)
        if backend.is_sparse(W):
            W_arr = W.toarray()
        else:
            W_arr = xp.asarray(W, dtype=xp.float64)
        A = xp.eye(n_area, dtype=xp.float64) - rho * W_arr
        AtA = A.T @ A
        Q_area = xp.exp(-2.0 * log_sigma_area) * AtA
        blocks.append(Q_area)

    # Time precision
    if time == "iid":
        log_sigma_time = logs.get("log_sigma_time", 0.0) if logs else 0.0
        Q_time = xp.exp(-2.0 * log_sigma_time) * xp.eye(n_time, dtype=xp.float64)
        blocks.append(Q_time)
    elif time == "ar1":
        log_sigma_time = logs.get("log_sigma_time", 0.0) if logs else 0.0
        raw_phi = raw.get("raw_phi", 0.0) if raw else 0.0
        Q_time = ar1_precision(n_time, log_sigma_time, raw_phi, backend)
        blocks.append(Q_time)

    from scipy.sparse import block_diag as sp_block_diag
    Q_full = sp_block_diag(blocks, format="csr")
    return backend.sparse.csr_matrix(Q_full) if backend.device == "gpu" else Q_full


def interaction_shape(des: dict, s: dict, raw_gamma: float = 0.0, backend: Backend = None):
    """Build unscaled interaction precision shape using Kronecker of contrasts.

    Args:
        des: Design dict with B_area, B_time.
        s: Structure dict.
        raw_gamma: Raw interaction dependence parameter.
        backend: Array backend.

    Returns:
        Sparse precision shape matrix.
    """
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu=False)

    xp = backend.xp
    interaction = s.get("interaction", "none")
    B_area = des.get("B_area")
    B_time = des.get("B_time")

    if interaction == "iid" or interaction == "none":
        # Identity shape
        n_gamma = B_area.shape[1] * B_time.shape[1] if B_area is not None and B_time is not None else 0
        return xp.eye(n_gamma, dtype=xp.float64) if n_gamma > 0 else None

    if interaction == "spatial":
        return xp.eye(B_area.shape[1] * B_time.shape[1], dtype=xp.float64)

    if interaction == "temporal":
        return xp.eye(B_area.shape[1] * B_time.shape[1], dtype=xp.float64)

    if interaction == "st":
        return xp.eye(B_area.shape[1] * B_time.shape[1], dtype=xp.float64)

    return None


