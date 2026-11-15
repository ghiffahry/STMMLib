"""Shared matrix utilities used by both Gaussian and Beta backends.

All functions accept a Backend instance and use backend.xp for array ops.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stmmlib._backend import Backend


# ---------------------------------------------------------------------------
# Positive-definiteness checks
# ---------------------------------------------------------------------------


def is_pd(M, backend: Backend, tol: float = 1e-8) -> bool:
    """Test if matrix M is symmetric positive-definite via Cholesky attempt."""
    xp = backend.xp
    if backend.is_sparse(M):
        M = M.toarray()
    M = xp.asarray(M, dtype=xp.float64)
    # Check approximate symmetry
    if not xp.allclose(M, M.T, atol=tol):
        return False
    try:
        backend.cholesky(M)
        return True
    except Exception:
        return False


def near_pd(M, backend: Backend, eps: float = 1e-6):
    """Project matrix to nearest positive-definite if needed.

    If M is already PD, returns M unchanged. Otherwise adds small ridge
    to the diagonal and retries.
    """
    xp = backend.xp
    if backend.is_sparse(M):
        M = M.toarray()
    M = xp.asarray(M, dtype=xp.float64)

    if is_pd(M, backend, tol=eps):
        return M

    # Symmetrize
    M = 0.5 * (M + M.T)

    # Add increasing ridge until PD
    n = M.shape[0]
    for k in range(50):
        shift = eps * (10.0**k) * xp.eye(n, dtype=xp.float64)
        M_try = M + shift
        if is_pd(M_try, backend, tol=eps):
            return M_try

    warnings.warn(
        f"Matrix not PD after max ridge iterations. Returning with ridge={eps * 10**49}.",
        stacklevel=2,
    )
    return M + eps * 1e49 * xp.eye(n, dtype=xp.float64)


def safe_cholesky(A, backend: Backend):
    """Cholesky decomposition with PD fallback.

    Returns lower-triangular L such that A = L @ L.T.
    """
    xp = backend.xp
    if backend.is_sparse(A):
        A = A.toarray()
    A = xp.asarray(A, dtype=xp.float64)
    try:
        return backend.cholesky(A)
    except Exception:
        A_pd = near_pd(A, backend)
        return backend.cholesky(A_pd)


def safe_solve(A, b, backend: Backend):
    """Solve A @ x = b with fallback to pseudo-inverse."""
    xp = backend.xp
    if backend.is_sparse(A):
        A = A.toarray()
    A = xp.asarray(A, dtype=xp.float64)
    b = xp.asarray(b, dtype=xp.float64)
    try:
        return backend.solve(A, b)
    except Exception:
        return backend.pinv(A) @ b


def logdet_matrix(M, backend: Backend):
    """Log-determinant of a matrix. Uses slogdet for numerical stability."""
    xp = backend.xp
    if backend.is_sparse(M):
        M = M.toarray()
    M = xp.asarray(M, dtype=xp.float64)
    sign, logabsdet = backend.slogdet(M)
    if float(sign) < 0:
        warnings.warn("Negative determinant encountered.", stacklevel=2)
    return float(logabsdet)


# ---------------------------------------------------------------------------
# Interaction helpers
# ---------------------------------------------------------------------------


def interaction_parse(interaction: str) -> str:
    """Normalize interaction type string; reject an unknown type.

    A typo used to fall through to "none", silently dropping the whole
    interaction variance component.
    """
    if interaction is None:
        return "none"
    key = str(interaction).strip().lower()
    mapping = {
        "": "none",
        "none": "none",
        "0": "none",
        "iid": "iid",
        "spatial": "spatial",
        "temporal": "temporal",
        "st": "st",
        "type i": "iid",
        "type ii": "spatial",
        "type iii": "temporal",
        "type iv": "st",
    }
    if key not in mapping:
        raise ValueError(
            f"interaction {interaction!r} tidak dikenal; pilihan: none, iid, "
            "spatial, temporal, st (atau Type I-IV).")
    return mapping[key]


def interaction_active(interaction: str) -> bool:
    """True if interaction is not 'none'."""
    return interaction_parse(interaction) != "none"


def row_standardize(W, safe: bool = True):
    """Row-standardize a spatial weight matrix, island-safe.

    Args:
        W: (m, m) array-like weight matrix (numpy or scipy sparse).
        safe: True (default) -> rows with zero sum (islands, no neighbors)
            are LEFT AS ZERO (no NaN); False -> plain W/row_sum (0/0 -> NaN,
            the classic island trap).

    Returns:
        numpy (m, m) row-standardized matrix. Island rows stay zero so that
        A = I - rho W keeps them iid under the SAR prior (valid: no
        neighbors -> no spatial shrinkage for that area).
    """
    import numpy as np
    from scipy import sparse as _sp
    if _sp.issparse(W):
        W = W.toarray()
    W = np.asarray(W, dtype=np.float64)
    rs = W.sum(axis=1, keepdims=True)
    if safe:
        return W / np.maximum(rs, 1.0)
    return W / rs


def interaction_param_names(interaction: str) -> list[str]:
    """Extra variance parameter names for interaction type."""
    itype = interaction_parse(interaction)
    if itype == "none":
        return []
    if itype == "iid":
        return ["sigma2_gamma"]
    if itype == "spatial":
        return ["sigma2_gamma", "rho_gamma"]
    if itype == "temporal":
        return ["sigma2_gamma", "phi_gamma"]
    if itype == "st":
        return ["sigma2_gamma", "rho_gamma", "phi_gamma"]
    return []


def require_numeric_predictors(data, exempt) -> None:
    """Name the offending column instead of numpy's conversion error.

    Every column that is not the response or a grouping column becomes a
    predictor, so a leftover text column reaches the design matrix and fails
    there with "could not convert string to float", which names nothing.
    A NaN/Inf value fails even later (SVD), so it is rejected here too.
    """
    import numpy as np
    if data is None or not hasattr(data, "columns"):
        return
    skip = {c for c in exempt if c}
    for col in data.columns:
        if col in skip:
            continue
        try:
            values = data[col].to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"kolom {col!r} bukan numerik; kolom selain area, waktu, dan "
                "kolom pengelompokan dipakai sebagai prediktor. Ubah ke "
                "numerik atau keluarkan dari data.") from None
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"kolom {col!r} mengandung NaN/Inf; bersihkan atau hilangkan "
                "kolom bermasalah sebelum fitting.")


