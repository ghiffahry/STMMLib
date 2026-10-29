"""Design matrix construction for Gaussian spatio-temporal mixed models.

Builds sparse incidence matrices Z_area, Z_time, Z_cell (for area-specific
models), and interaction contrast/full matrices for types I-IV.

All functions accept a Backend and return GPU arrays when GPU is active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stmmlib._backend import Backend


# ---------------------------------------------------------------------------
# Core Z builder
# ---------------------------------------------------------------------------


def _build_Z(index, n: int, n_cols: int, label: str, backend: Backend):
    """Build sparse incidence matrix from integer index vector.

    Args:
        index: Integer array of column assignments (0-indexed), length n.
        n: Number of rows.
        n_cols: Number of columns.
        label: Label for error messages.
        backend: Array backend.

    Returns:
        Sparse n x n_cols matrix.
    """
    xp = backend.xp
    index = xp.asarray(index, dtype=xp.int32).ravel()
    if len(index) != n:
        raise ValueError(f"{label}: index length {len(index)} != n={n}")
    row = xp.arange(n, dtype=xp.int32)
    data = xp.ones(n, dtype=xp.float64)
    return backend.sparse.csr_matrix((data, (row, index)), shape=(n, n_cols))


# ---------------------------------------------------------------------------
# Area and time design matrices
# ---------------------------------------------------------------------------


def build_Z_area(prepared: dict, backend: Backend):
    """Build area-level incidence matrix Z_area from prepared panel data.

    Args:
        prepared: dict with keys 'area_index' (int array), 'n' (int),
                  'm' (int, number of areas).
        backend: Array backend.

    Returns:
        Sparse n x m matrix.
    """
    return _build_Z(
        prepared["area_index"],
        prepared["n"],
        prepared["m"],
        "Z_area",
        backend,
    )


def build_Z_time(prepared: dict, backend: Backend):
    """Build time-level incidence matrix Z_time.

    Args:
        prepared: dict with keys 'time_index', 'n', 'T' (number of time periods).
        backend: Array backend.

    Returns:
        Sparse n x T matrix.
    """
    return _build_Z(
        prepared["time_index"],
        prepared["n"],
        prepared["T"],
        "Z_time",
        backend,
    )


def build_design_matrices(
    prepared: dict,
    backend: Backend,
    validate: bool = True,
    verbose: bool = False,
) -> dict:
    """Build Z_area and Z_time, optionally validate rank.

    Returns:
        dict with 'Z_area', 'Z_time'.
    """
    Z_area = build_Z_area(prepared, backend)
    Z_time = build_Z_time(prepared, backend)

    if validate:
        xp = backend.xp
        n = prepared["n"]
        m = prepared["m"]
        T = prepared["T"]
        rank_area = min(n, m)
        rank_time = min(n, T)
        if verbose:
            print(f"Z_area rank: {rank_area}/{m}, Z_time rank: {rank_time}/{T}")

    return {"Z_area": Z_area, "Z_time": Z_time}


# ---------------------------------------------------------------------------
# Nested design matrices (3-level: prov, kab, time)
# ---------------------------------------------------------------------------


def build_Z_nested(prepared: dict, backend: Backend):
    """Build nested design matrices for 3-level models.

    Args:
        prepared: dict with 'prov_index', 'kab_index', 'time_index',
                  'n', 'n_prov', 'n_kab', 'T'.
        backend: Array backend.

    Returns:
        dict with 'Z_prov', 'Z_kab', 'Z_time', 'Z_cell'.
    """
    Z_prov = _build_Z(
        prepared["prov_index"], prepared["n"], prepared["n_prov"], "Z_prov", backend
    )
    Z_kab = _build_Z(
        prepared["kab_index"], prepared["n"], prepared["n_kab"], "Z_kab", backend
    )
    Z_time = _build_Z(
        prepared["time_index"], prepared["n"], prepared["T"], "Z_time", backend
    )
    # Cell = kab x time
    n_cell = prepared["n_kab"] * prepared["T"]
    cell_index = (
        prepared["kab_index"] * prepared["T"] + prepared["time_index"]
    )
    Z_cell = _build_Z(cell_index, prepared["n"], n_cell, "Z_cell", backend)

    return {
        "Z_prov": Z_prov,
        "Z_kab": Z_kab,
        "Z_time": Z_time,
        "Z_cell": Z_cell,
    }


# ---------------------------------------------------------------------------
# Interaction matrices (Types I-IV)
# ---------------------------------------------------------------------------


def _contr_sum(n: int, backend: Backend):
    """Sum-to-zero contrast matrix (like R's contr.sum).

    Returns n x (n-1) matrix where each column has +1 and -1 coding
    ensuring column sums are zero.

    Args:
        n: Number of levels.
        backend: Array backend.

    Returns:
        Dense (n-1) x n contrast matrix (transposed to match R convention).
    """
    xp = backend.xp
    if n <= 1:
        return xp.zeros((1, 1), dtype=xp.float64)
    B = xp.eye(n, dtype=xp.float64)[:, :-1]
    # Last level = -1 for all other columns
    B[-1, :] = -1.0
    return B  # n x (n-1)


def build_interaction_contrast(m_area: int, T: int, backend: Backend) -> dict:
    """Build contrast matrices for interaction.

    Args:
        m_area: Number of area levels.
        T: Number of time periods.
        backend: Array backend.

    Returns:
        dict with 'B_area' (m_area x m_area-1), 'B_time' (T x T-1).
    """
    return {
        "B_area": _contr_sum(m_area, backend),
        "B_time": _contr_sum(T, backend),
    }


def build_Z_interaction_full(
    area_index, time_index, n: int, m: int, T: int, backend: Backend
):
    """Build full one-hot Z for area x time interaction cells.

    Args:
        area_index: Integer array of area assignments (0-indexed).
        time_index: Integer array of time assignments (0-indexed).
        n: Number of observations.
        m: Number of areas.
        T: Number of time periods.
        backend: Array backend.

    Returns:
        Sparse n x (m*T) matrix.
    """
    xp = backend.xp
    area_index = xp.asarray(area_index, dtype=xp.int32).ravel()
    time_index = xp.asarray(time_index, dtype=xp.int32).ravel()
    cell_index = area_index * T + time_index
    return _build_Z(cell_index, n, m * T, "Z_interaction_full", backend)


def build_interaction_matrices(
    prepared: dict,
    interaction: str,
    area_index,
    m_area: int,
    T: int,
    backend: Backend,
    validate: bool = True,
) -> dict:
    """Assemble Z_gamma for interaction from contrast * full Z.

    For interaction type:
        iid:      Z_gamma = Z_full @ kron(B_area, I_T)
        spatial:  Z_gamma = Z_full @ kron(B_area, I_T)  (precision is spatial)
        temporal: Z_gamma = Z_full @ kron(I_area, B_time)
        st:       Z_gamma = Z_full @ kron(B_area, B_time)

    Args:
        prepared: dict with panel data.
        interaction: Interaction type string.
        area_index: Area index array.
        m_area: Number of areas.
        T: Number of time periods.
        backend: Array backend.

    Returns:
        dict with 'Z_gamma', 'B_area', 'B_time', 'n_gamma' (cols).
    """
    from stmmlib._utils import interaction_parse

    itype = interaction_parse(interaction)
    if itype == "none":
        return {"Z_gamma": None, "B_area": None, "B_time": None, "n_gamma": 0}

    contrasts = build_interaction_contrast(m_area, T, backend)
    B_area = contrasts["B_area"]  # m_area x (m_area-1)
    B_time = contrasts["B_time"]  # T x (T-1)

    n = prepared["n"]
    xp = backend.xp
    time_index = prepared["time_index"]

    # Full one-hot for cells
    Z_full = build_Z_interaction_full(
        area_index, time_index, n, m_area, T, backend
    )
    area_index_arr = xp.asarray(area_index, dtype=xp.int32).ravel()
    time_index_arr = xp.asarray(time_index, dtype=xp.int32).ravel()

    m1 = m_area - 1
    T1 = T - 1

    if itype == "iid":
        # Z_gamma = Z_full @ kron(B_area, I_T): m*T x m1*T cols
        I_T = backend.sparse_eye(T, dtype=xp.float64)
        K = backend.sparse_kron(B_area, I_T, format="csr")
        Z_gamma = Z_full @ K
        n_gamma = m1 * T
    elif itype == "spatial":
        # Same structure as iid for Z; precision differs
        I_T = backend.sparse_eye(T, dtype=xp.float64)
        K = backend.sparse_kron(B_area, I_T, format="csr")
        Z_gamma = Z_full @ K
        n_gamma = m1 * T
    elif itype == "temporal":
        # Z_gamma = Z_full @ kron(I_area, B_time): m*T x m*T1 cols
        I_m = backend.sparse_eye(m_area, dtype=xp.float64)
        K = backend.sparse_kron(I_m, B_time, format="csr")
        Z_gamma = Z_full @ K
        n_gamma = m_area * T1
    elif itype == "st":
        # Z_gamma = Z_full @ kron(B_area, B_time): m*T x m1*T1 cols
        K = backend.sparse_kron(B_area, B_time, format="csr")
        Z_gamma = Z_full @ K
        n_gamma = m1 * T1
    else:
        raise ValueError(f"Unknown interaction type: {itype!r}")

    if validate:
        assert Z_gamma.shape == (n, n_gamma), (
            f"Z_gamma shape {Z_gamma.shape} != expected ({n}, {n_gamma})"
        )

    return {
        "Z_gamma": Z_gamma,
        "B_area": B_area,
        "B_time": B_time,
        "n_gamma": n_gamma,
    }
