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


class PrecisionShapeError(ValueError):
    """Assembled precision Q does not match the design matrix Z."""


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
    # meaningless surrogate for the spatial dependence.
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
    (an old display could show a phi the model did not have).
    """
    u = phi_bound * np.tanh(raw_phi)
    return phi_bound * np.tanh(u)

