"""Design matrix construction for Beta mixed models.

Builds Z (area/time/cell/interaction), block indices, W_gamma,
and metadata for the Beta STMM Laplace engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stmmlib.beta.structure import stmm_beta_structure

if TYPE_CHECKING:
    from stmmlib._backend import Backend


def stmm_beta_make_design(d: pd.DataFrame, model: str, W=None, backend: Backend = None,
                          phi_bound: float = 0.95) -> dict:
    """Construct design matrix Z and metadata for a Beta STMM.

    Z has area-major cell order: (area1:time1..timeT, area2:time1..timeT, ...)

    Args:
        d: pandas DataFrame with area, time columns (internal names).
        model: Model ID string.
        W: Spatial weight matrix.
        backend: Array backend.

    Returns:
        dict with Z, blocks, W_gamma, B_area, B_time, metadata.
    """
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu=False)

    xp = backend.xp
    s = stmm_beta_structure(model)
    area = s["area"]
    time = s["time"]
    time_scope = s["time_scope"]
    interaction = s["interaction"]

    # Extract indices
    area_col = ".stmm_area"
    time_col = ".stmm_time"
    area_idx = d[area_col].values.astype(np.int32)
    time_idx = d[time_col].values.astype(np.int32)
    n = len(d)
    m = int(np.max(area_idx)) + 1
    T = int(np.max(time_idx)) + 1
    n_cell = m * T

    # Build Z
    from scipy.sparse import csr_matrix, eye as sp_eye, kron as sp_kron

    def _make_Z(idx, n_rows, n_cols):
        row = np.arange(n_rows, dtype=np.int32)
        return csr_matrix((np.ones(n_rows, dtype=np.float64), (row, idx)), shape=(n_rows, n_cols))

    # Nested hierarchy (M6/M7/M13/M14): Z = [Z_prov, Z_kab, Z_cell]
    if area in ("nested_iid", "nested_sar"):
        if ".stmm_prov" not in d.columns or ".stmm_kab" not in d.columns:
            raise ValueError(
                "Nested models require province and district columns; "
                "pass prov_col=/kab_col= (defaults 'prov'/'kab').")
        prov_idx = d[".stmm_prov"].values.astype(np.int32)
        kab_idx = area_idx  # .stmm_area IS the kab level for nested models
        n_prov = int(np.max(prov_idx)) + 1
        n_kab = m
        Z_prov = _make_Z(prov_idx, n, n_prov)
        Z_kab = _make_Z(kab_idx, n, n_kab)
        Zc = _make_Z(area_idx * T + time_idx, n, n_cell)
        Z = np.hstack([Z_prov.toarray(), Z_kab.toarray(), Zc.toarray()])
        blocks = {
            "prov": list(range(n_prov)),
            "kab": list(range(n_prov, n_prov + n_kab)),
            "cell": list(range(n_prov + n_kab, n_prov + n_kab + n_cell)),
        }
        des = {
            "Z": Z, "blocks": blocks, "W_gamma": None, "rho_bound_g": phi_bound,
            "B_area": None, "B_time": None, "n": n, "m": m, "T": T,
            "n_cell": n_cell, "n_gamma": 0,
            "n_prov": n_prov, "n_kab": n_kab,
            "area_labels": d[area_col].unique().tolist(),
            "time_labels": sorted(d[time_col].unique().tolist()),
            "area_idx": area_idx, "time_idx": time_idx,
        }
        # Interaction at kab x time level (M13 temporal / M14 iid)
        if interaction != "none":
            B_area = _contr_sum(m)
            B_time = _contr_sum(T)
            Z_gamma = _build_Z_gamma(area_idx, time_idx, n, m, T, interaction,
                                     B_area, B_time)
            n_gamma = Z_gamma.shape[1]
            blocks["interaction"] = list(range(Z.shape[1], Z.shape[1] + n_gamma))
            des["Z"] = np.hstack([Z, Z_gamma])
            des["B_area"] = B_area
            des["B_time"] = B_time
            des["n_gamma"] = n_gamma
        return des
    else:
        des = _design_flat(d, area, time, time_scope, interaction, W, backend,
                           area_idx, time_idx, n, m, T, n_cell, _make_Z,
                           phi_bound=phi_bound)
    return des


def _design_flat(d, area, time, time_scope, interaction, W, backend,
                 area_idx, time_idx, n, m, T, n_cell, _make_Z,
                 phi_bound: float = 0.95):
    """Flat (non-nested) design: Z = [Z_area, Z_cell|Z_time] (+ Z_gamma)."""
    # Area dummy
    Z_area = _make_Z(area_idx, n, m)

    # Cell dummy (area x time)
    cell_idx = area_idx * T + time_idx
    Z_cell = _make_Z(cell_idx, n, n_cell)

    if time_scope == "global":
        # Global: Z = [Z_area, Z_time]
        Z_time = _make_Z(time_idx, n, T)
        Z = np.hstack([Z_area.toarray(), Z_time.toarray()])
        blocks = {
            "area": list(range(m)),
            "time": list(range(m, m + T)),
        }
    else:
        # Area-specific: Z = [Z_area, Z_cell]
        Z = np.hstack([Z_area.toarray(), Z_cell.toarray()])
        blocks = {
            "area": list(range(m)),
            "area_time": list(range(m, m + n_cell)),
        }

    # Contrast matrices for interaction
    B_area = None
    B_time = None
    W_gamma = None
    rho_bound_g = phi_bound

    if interaction != "none":
        B_area = _contr_sum(m)
        B_time = _contr_sum(T)

        if interaction in ("spatial", "st") and W is not None:
            if backend.is_sparse(W):
                W_arr = W.toarray()
            else:
                W_arr = np.asarray(W, dtype=np.float64)
            W_gamma = W_arr
            # BUG-B1 fix: eigvals (spectral radius), NOT eigvalsh -- W is
            # asymmetric; eigvalsh silently reads one triangle (measured
            # 0.770 vs true 0.950 on KNN-4 row-standardized W).
            eigenvalues = np.linalg.eigvals(W_arr)
            max_abs_eig = np.max(np.abs(eigenvalues))
            rho_bound_g = phi_bound / max_abs_eig if max_abs_eig > 1e-10 else phi_bound

        # Interaction Z
        Z_gamma = _build_Z_gamma(area_idx, time_idx, n, m, T, interaction, B_area, B_time)
        n_gamma = Z_gamma.shape[1]
        blocks["interaction"] = list(range(Z.shape[1], Z.shape[1] + n_gamma))
        Z = np.hstack([Z, Z_gamma])

    return {
        "Z": Z,
        "blocks": blocks,
        "W_gamma": W_gamma,
        "rho_bound_g": rho_bound_g,
        "B_area": B_area,
        "B_time": B_time,
        "n": n,
        "m": m,
        "T": T,
        "n_cell": n_cell,
        "n_gamma": Z_gamma.shape[1] if interaction != "none" else 0,
        "area_labels": d[".stmm_area"].unique().tolist() if ".stmm_area" in d.columns else [],
        "time_labels": sorted(d[".stmm_time"].unique().tolist()) if ".stmm_time" in d.columns else [],
        "area_idx": area_idx,
        "time_idx": time_idx,
    }


def _contr_sum(n: int):
    """Sum-to-zero contrast matrix (R-style contr.sum). Returns n x (n-1)."""
    if n <= 1:
        return np.ones((1, 1), dtype=np.float64)
    B = np.eye(n, dtype=np.float64)[:, :-1]
    B[-1, :] = -1.0
    return B


def _build_Z_gamma(area_idx, time_idx, n, m, T, interaction, B_area, B_time):
    """Build Z_gamma from contrast matrices and full cell Z."""
    from scipy.sparse import eye as sp_eye, kron as sp_kron, csr_matrix

    # Full cell Z
    cell_idx = area_idx * T + time_idx
    Z_full = csr_matrix(
        (np.ones(n, dtype=np.float64), (np.arange(n, dtype=np.int32), cell_idx)),
        shape=(n, m * T),
    )

    m1 = B_area.shape[1]
    T1 = B_time.shape[1]

    if interaction == "iid":
        K = sp_kron(csr_matrix(B_area), sp_eye(T, dtype=np.float64), format="csr")
    elif interaction == "spatial":
        K = sp_kron(csr_matrix(B_area), sp_eye(T, dtype=np.float64), format="csr")
    elif interaction == "temporal":
        K = sp_kron(sp_eye(m, dtype=np.float64), csr_matrix(B_time), format="csr")
    elif interaction == "st":
        K = sp_kron(csr_matrix(B_area), csr_matrix(B_time), format="csr")
    else:
        return np.zeros((n, 0), dtype=np.float64)

    Z_gamma = (Z_full @ K).toarray()
    return Z_gamma
