"""Covariance matrix construction for Gaussian spatio-temporal models.

Builds area-level G_area, time-level G_time, interaction G_interaction,
and the full marginal covariance V = Z G Z' + sigma2_e I.

All matrix ops routed through Backend for GPU acceleration.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from stmmlib._utils import is_pd, near_pd, logdet_matrix, safe_solve


def _to_np_backend(backend, a):
    """Dense CPU numpy view of any backend array (numpy/cupy/sparse)."""
    if backend.is_sparse(a):
        a = a.toarray()
    if hasattr(a, "get"):  # cupy -> numpy
        a = a.get()
    return np.asarray(a, dtype=np.float64)


if TYPE_CHECKING:
    from stmmlib._backend import Backend


# ---------------------------------------------------------------------------
# AR(1) core
# ---------------------------------------------------------------------------


def G_ar1_core(T: int, sigma2_eta: float, phi: float, backend: Backend):
    """AR(1) covariance matrix: sigma2_eta/(1-phi^2) * R(phi).

    R_t,s = phi^|t-s|.

    Args:
        T: Number of time periods.
        sigma2_eta: Innovation variance.
        phi: AR(1) coefficient, |phi| < 1.
        backend: Array backend.

    Returns:
        Dense T x T covariance matrix.
    """
    xp = backend.xp
    t_idx = xp.arange(T, dtype=xp.float64)
    # |t - s| matrix
    diff = xp.abs(t_idx[:, None] - t_idx[None, :])
    R = phi**diff
    scale = sigma2_eta / (1.0 - phi**2) if abs(phi) < 1.0 else sigma2_eta * 1e6
    return scale * R


# ---------------------------------------------------------------------------
# Area-level G
# ---------------------------------------------------------------------------


def build_G_area(
    area_type: str,
    sigma2_v: float,
    m: int,
    W=None,
    rho: float = 0.0,
    backend: Backend = None,
):
    """Build area-level covariance block G_area.

    Args:
        area_type: "iid" or "sar".
        sigma2_v: Area variance.
        m: Number of areas.
        W: Spatial weight matrix (required for SAR).
        rho: Spatial autocorrelation parameter.
        backend: Array backend.

    Returns:
        Dense m x m covariance matrix.
    """
    xp = backend.xp

    if area_type == "iid":
        return sigma2_v * xp.eye(m, dtype=xp.float64)

    if area_type == "sar":
        if W is None:
            raise ValueError("SAR model requires spatial weight matrix W.")
        if backend.is_sparse(W):
            W = W.toarray()
        W = xp.asarray(W, dtype=xp.float64)
        A = xp.eye(m, dtype=xp.float64) - rho * W
        # G = sigma2_v * (A'A)^{-1}
        AtA = A.T @ A
        G = sigma2_v * safe_solve(AtA, xp.eye(m, dtype=xp.float64), backend)
        if not is_pd(G, backend):
            G = near_pd(G, backend)
        return G

    raise ValueError(f"Unknown area_type: {area_type!r}")


# ---------------------------------------------------------------------------
# Time-level G
# ---------------------------------------------------------------------------


def build_G_time(
    time_type: str,
    T: int,
    sigma2_u: float,
    sigma2_eta: float = 0.0,
    phi: float = 0.0,
    backend: Backend = None,
):
    """Build time-level covariance block G_time.

    Args:
        time_type: "iid" or "ar1".
        T: Number of time periods.
        sigma2_u: Time variance (iid) or not used.
        sigma2_eta: Innovation variance (ar1).
        phi: AR(1) coefficient (ar1).
        backend: Array backend.

    Returns:
        Dense T x T covariance matrix.
    """
    xp = backend.xp

    if time_type == "iid":
        return sigma2_u * xp.eye(T, dtype=xp.float64)

    if time_type == "ar1":
        return G_ar1_core(T, sigma2_eta, phi, backend)

    raise ValueError(f"Unknown time_type: {time_type!r}")


# ---------------------------------------------------------------------------
# Kronecker expansion
# ---------------------------------------------------------------------------


def expand_it(m: int, G, backend: Backend):
    """Kronecker I_m x G to expand per-area temporal covariance.

    Args:
        m: Number of areas.
        G: T x T time covariance block.
        backend: Array backend.

    Returns:
        (m*T) x (m*T) block-diagonal matrix.
    """
    I_m = backend.sparse_eye(m, dtype=backend.xp.float64)
    return backend.sparse_kron(I_m, G, format="csr")


# ---------------------------------------------------------------------------
# Interaction G
# ---------------------------------------------------------------------------


def _build_ar1_corr(T: int, phi: float, backend: Backend):
    """AR(1) correlation matrix (unit variance)."""
    xp = backend.xp
    t_idx = xp.arange(T, dtype=xp.float64)
    diff = xp.abs(t_idx[:, None] - t_idx[None, :])
    return phi**diff


def build_G_interaction(
    interaction: str,
    sigma2_gamma: float,
    m: int,
    T: int,
    W=None,
    rho_gamma: float = 0.0,
    phi_gamma: float = 0.0,
    B_st=None,
    backend: Backend = None,
):
    """Build interaction covariance G_gamma for types I-IV.

    Args:
        interaction: Interaction type ("iid", "spatial", "temporal", "st").
        sigma2_gamma: Interaction variance.
        m: Number of areas.
        T: Number of time periods.
        W: Spatial weight matrix (for spatial/st).
        rho_gamma: Spatial dependence for interaction.
        phi_gamma: Temporal dependence for interaction.
        B_st: Pre-computed contrast matrices (optional).
        backend: Array backend.

    Returns:
        Covariance matrix for interaction random effects.
    """
    from stmmlib._utils import interaction_parse

    itype = interaction_parse(interaction)
    xp = backend.xp

    if itype == "none":
        return None

    if itype == "iid":
        n_gamma = (m - 1) * T
        return sigma2_gamma * xp.eye(n_gamma, dtype=xp.float64)

    if itype == "spatial":
        if W is None:
            raise ValueError("Spatial interaction requires W.")
        if backend.is_sparse(W):
            W_arr = W.toarray()
        else:
            W_arr = xp.asarray(W, dtype=xp.float64)
        # qa = B_area' (I - rho W)'(I - rho W) B_area
        A = xp.eye(m, dtype=xp.float64) - rho_gamma * W_arr
        I_T = backend.sparse_eye(T, dtype=xp.float64)
        AtA = A.T @ A
        if B_st is not None:
            B_area = B_st["B_area"]
        else:
            from stmmlib.gaussian.design import _contr_sum
            B_area = _contr_sum(m, backend)
        qa = B_area.T @ AtA @ B_area
        if not is_pd(qa, backend):
            qa = near_pd(qa, backend)
        n_gamma_a = m - 1
        return sigma2_gamma * backend.sparse_kron(I_T, qa, format="csr")

    if itype == "temporal":
        # qt = B_time' A(phi_gamma) B_time
        Ar = _build_ar1_corr(T, phi_gamma, backend)
        if B_st is not None:
            B_time = B_st["B_time"]
        else:
            from stmmlib.gaussian.design import _contr_sum
            B_time = _contr_sum(T, backend)
        qt = B_time.T @ Ar @ B_time
        # Z_gamma = kron(I_m, B_time): gamma grouped per area, n_gamma = m*(T-1).
        # So G = block-diag over areas of qt -> kron(I_m, qt).
        I_m = backend.sparse_eye(m, dtype=xp.float64)
        return sigma2_gamma * backend.sparse_kron(I_m, qt, format="csr")

    if itype == "st":
        # Q_gamma = kron(qt, qa) where qt=AR1, qa=SAR
        if W is None:
            raise ValueError("ST interaction requires W.")
        if backend.is_sparse(W):
            W_arr = W.toarray()
        else:
            W_arr = xp.asarray(W, dtype=xp.float64)
        A = xp.eye(m, dtype=xp.float64) - rho_gamma * W_arr
        AtA = A.T @ A
        if B_st is not None:
            B_area = B_st["B_area"]
            B_time = B_st["B_time"]
        else:
            from stmmlib.gaussian.design import _contr_sum
            B_area = _contr_sum(m, backend)
            B_time = _contr_sum(T, backend)
        qa = B_area.T @ AtA @ B_area
        Ar = _build_ar1_corr(T, phi_gamma, backend)
        qt = B_time.T @ Ar @ B_time
        if not is_pd(qa, backend):
            qa = near_pd(qa, backend)
        if not is_pd(qt, backend):
            qt = near_pd(qt, backend)
        Q_gamma = backend.sparse_kron(qt, qa, format="csr")
        return sigma2_gamma * Q_gamma

    raise ValueError(f"Unknown interaction type: {itype!r}")


# ---------------------------------------------------------------------------
# Full V builder
# ---------------------------------------------------------------------------


def build_V(
    data: dict,
    model: dict,
    theta: dict,
    backend: Backend,
    return_components: bool = False,
):
    """Build full marginal covariance V = Z G Z' + sigma2_e I.

    V = Z_area G_area Z_area' + expand(m, G_time) + G_gamma + sigma2_e I

    Args:
        data: dict with prepared panel data + design matrices.
        model: dict with area_type, time_type, interaction, etc.
        theta: dict with sigma2_v, sigma2_u, sigma2_eta, phi, sigma2_e,
               sigma2_gamma, rho, rho_gamma, phi_gamma.
        backend: Array backend.
        return_components: If True, also return component matrices.

    Returns:
        V: Dense n x n covariance matrix, or dict if return_components=True.
    """
    xp = backend.xp
    n = data["n"]
    m = data["m"]
    T = data["T"]

    Z_area = data["Z_area"]
    Z_time = data.get("Z_time")

    area_type = model["area_type"]
    time_type = model["time_type"]
    interaction = model.get("interaction", "none")

    sigma2_v = theta.get("sigma2_v", 1.0)
    sigma2_u = theta.get("sigma2_u", 1.0)
    sigma2_eta = theta.get("sigma2_eta", 1.0)
    phi = theta.get("phi", 0.0)
    sigma2_e = theta.get("sigma2_e", 1.0)
    rho = theta.get("rho", 0.0)
    sigma2_gamma = theta.get("sigma2_gamma", 0.0)
    rho_gamma = theta.get("rho_gamma", 0.0)
    phi_gamma = theta.get("phi_gamma", 0.0)
    W = data.get("W")

    # Area block (converted to CPU numpy; the whole V assembly is numpy-based
    # so CPU/GPU backends mix safely — matrices are small n x n dense)
    G_a = build_G_area(area_type, sigma2_v, m, W, rho, backend)
    Za = _to_np_backend(backend, Z_area)
    Ga = _to_np_backend(backend, G_a)
    ZGZ_area = Za @ Ga @ Za.T

    # Time block: Z_cell G_cell Z_cell' with G_cell = kron(I_m, G_t).
    # Z_cell is built from observed (area, time) indices, so this is valid for
    # any n (replicated observations per cell), not just n = m*T (fixes B6).
    G_t = build_G_time(time_type, T, sigma2_u, sigma2_eta, phi, backend)
    G_t = _to_np_backend(backend, G_t)
    _cell_idx = np.asarray(data["area_index"], dtype=np.int64) * T \
        + np.asarray(data["time_index"], dtype=np.int64)
    _Zc = np.zeros((n, m * T), dtype=np.float64)
    _Zc[np.arange(n), _cell_idx] = 1.0
    _G_cell = np.kron(np.eye(m, dtype=np.float64), G_t)
    V_time = _Zc @ _G_cell @ _Zc.T

    # Interaction block
    from stmmlib._utils import interaction_active
    V_gamma = np.zeros((n, n), dtype=np.float64)
    if interaction_active(interaction) and sigma2_gamma > 0:
        from stmmlib.gaussian.design import build_interaction_contrast
        contrasts = build_interaction_contrast(m, T, backend)
        G_g = build_G_interaction(
            interaction, sigma2_gamma, m, T, W, rho_gamma, phi_gamma,
            contrasts, backend,
        )
        Z_gamma = data.get("Z_gamma")
        if Z_gamma is not None and G_g is not None:
            if backend.is_sparse(G_g):
                G_g = G_g.toarray()
            if backend.is_sparse(Z_gamma):
                Z_gamma = Z_gamma.toarray()
            V_gamma = _to_np_backend(backend, Z_gamma) @ _to_np_backend(backend, G_g) @ _to_np_backend(backend, Z_gamma).T

    # Residual block
    V_e = sigma2_e * np.eye(n, dtype=np.float64)

    V = ZGZ_area + V_time + V_gamma + V_e
    V = 0.5 * (V + V.T)  # enforce symmetry

    if return_components:
        return {
            "V": V,
            "G_area": G_a,
            "G_time": G_t,
            "G_interaction": G_g if interaction_active(interaction) else None,
            "Z_area": Z_area,
            "Z_time": Z_time,
            "sigma2_e": sigma2_e,
        }
    return V


# ---------------------------------------------------------------------------
# SEBLUP5 V builder
# ---------------------------------------------------------------------------


def build_V_seblup5(
    prepared: dict,
    theta: dict,
    interaction: str,
    area_type: str,
    backend: Backend,
):
    """Build V for SEBLUP Rao-Yu IT model (M5/M12).

    V = Z_area G_area Z_area' + kron(I, G_u_it) + G_gamma + sigma2_e I
    where G_u_it has per-area AR(1) structure.
    """
    xp = backend.xp
    n = prepared["n"]
    m = prepared["m"]
    T = prepared["T"]
    W = prepared.get("W")

    sigma2_v = theta.get("sigma2_v", 1.0)
    sigma2_eta = theta.get("sigma2_eta", 1.0)
    phi = theta.get("phi", 0.0)
    sigma2_e = theta.get("sigma2_e", 1.0)
    rho = theta.get("rho", 0.0)

    Z_area = prepared["Z_area"]
    G_a = build_G_area(area_type, sigma2_v, m, W, rho, backend)
    V_area = _to_np_backend(backend, Z_area) @ _to_np_backend(backend, G_a) @ _to_np_backend(backend, Z_area).T

    # Per-area u_it AR(1): Z_cell (I_m x G_u) Z_cell' — Z-based, valid for
    # any n (replicated/unbalanced panels), not just n = m*T.
    G_u = G_ar1_core(T, sigma2_eta, phi, backend)
    G_u = _to_np_backend(backend, G_u)
    _cell_idx = np.asarray(prepared["area_index"], dtype=np.int64) * T \
        + np.asarray(prepared["time_index"], dtype=np.int64)
    _Zc = np.zeros((n, m * T), dtype=np.float64)
    _Zc[np.arange(n), _cell_idx] = 1.0
    V_uit = _Zc @ np.kron(np.eye(m, dtype=np.float64), G_u) @ _Zc.T

    # Interaction
    from stmmlib._utils import interaction_active
    V_gamma = np.zeros((n, n), dtype=np.float64)
    if interaction_active(interaction):
        from stmmlib.gaussian.design import build_interaction_contrast
        contrasts = build_interaction_contrast(m, T, backend)
        sigma2_g = theta.get("sigma2_gamma", 0.0)
        rho_g = theta.get("rho_gamma", 0.0)
        phi_g = theta.get("phi_gamma", 0.0)
        G_g = build_G_interaction(
            interaction, sigma2_g, m, T, W, rho_g, phi_g, contrasts, backend
        )
        Z_gamma = prepared.get("Z_gamma")
        if Z_gamma is not None and G_g is not None:
            if backend.is_sparse(G_g):
                G_g = G_g.toarray()
            if backend.is_sparse(Z_gamma):
                Z_gamma = Z_gamma.toarray()
            V_gamma = _to_np_backend(backend, Z_gamma) @ _to_np_backend(backend, G_g) @ _to_np_backend(backend, Z_gamma).T

    V_e = sigma2_e * np.eye(n, dtype=np.float64)
    V = V_area + V_uit + V_gamma + V_e
    V = 0.5 * (V + V.T)
    return V


# ---------------------------------------------------------------------------
# Nested V builders
# ---------------------------------------------------------------------------


def _nested_gamma_block(prepared: dict, theta: dict, interaction: str,
                        n: int, backend: Backend):
    """V_gamma block for nested models (M13/M14): Z_gamma G_gamma Z_gamma'.

    Returns zeros when interaction inactive or Z_gamma absent.
    """
    from stmmlib._utils import interaction_active
    xp = backend.xp
    V_gamma = np.zeros((n, n), dtype=np.float64)
    if not interaction_active(interaction):
        return V_gamma
    m = prepared["m"]
    T = prepared["T"]
    sigma2_g = theta.get("sigma2_gamma", 0.0)
    if sigma2_g <= 0:
        return V_gamma
    from stmmlib.gaussian.design import build_interaction_contrast
    contrasts = build_interaction_contrast(m, T, backend)
    G_g = build_G_interaction(
        interaction, sigma2_g, m, T, prepared.get("W"),
        theta.get("rho_gamma", 0.0), theta.get("phi_gamma", 0.0),
        contrasts, backend,
    )
    Z_gamma = prepared.get("Z_gamma")
    if Z_gamma is not None and G_g is not None:
        if backend.is_sparse(G_g):
            G_g = G_g.toarray()
        if backend.is_sparse(Z_gamma):
            Z_gamma = Z_gamma.toarray()
        V_gamma = _to_np_backend(backend, Z_gamma) @ _to_np_backend(backend, G_g) @ _to_np_backend(backend, Z_gamma).T
    return V_gamma


def _nested_cell_v(prepared: dict, G_t, n_kab: int, T: int, n: int, backend: Backend):
    """V_cell = Z_cell (I_{n_kab} x G_t) Z_cell' — valid for any n (fixes B6)."""
    xp = backend.xp
    Z_cell = prepared.get("Z_cell")
    if Z_cell is None:
        return xp.zeros((n, n), dtype=xp.float64)
    G_t = _to_np_backend(backend, G_t)
    G_cell = np.kron(np.eye(n_kab, dtype=np.float64), G_t)
    Zc = _to_np_backend(backend, Z_cell)
    return Zc @ G_cell @ Zc.T


def build_V_nested(
    prepared: dict,
    theta: dict,
    interaction: str,
    backend: Backend,
):
    """Build V for nested IID+IID model (M6/M13).

    V = Z_prov G_prov Z_prov' + Z_kab G_kab Z_kab' + expand(kab, G_time) + G_gamma + sigma2_e I
    """
    xp = backend.xp
    n = prepared["n"]
    n_prov = prepared["n_prov"]
    n_kab = prepared["n_kab"]
    T = prepared["T"]

    sigma2_prov = theta.get("sigma2_prov", theta.get("sigma2_v", 1.0))
    sigma2_kab = theta.get("sigma2_kab", theta.get("sigma2_u", 1.0))
    sigma2_e = theta.get("sigma2_e", 1.0)

    Z_prov = prepared["Z_prov"]
    Z_kab = prepared["Z_kab"]
    Z_cell = prepared["Z_cell"]

    G_prov = sigma2_prov * xp.eye(n_prov, dtype=xp.float64)
    G_kab = sigma2_kab * xp.eye(n_kab, dtype=xp.float64)

    V_prov = _to_np_backend(backend, Z_prov) @ _to_np_backend(backend, G_prov) @ _to_np_backend(backend, Z_prov).T
    V_kab = _to_np_backend(backend, Z_kab) @ _to_np_backend(backend, G_kab) @ _to_np_backend(backend, Z_kab).T

    # Time is IID for nested_iid_iid
    G_t = sigma2_kab * xp.eye(T, dtype=xp.float64)
    V_cell = _nested_cell_v(prepared, G_t, n_kab, T, n, backend)
    if backend.is_sparse(V_cell):
        V_cell = V_cell.toarray()
    if backend.is_sparse(V_prov):
        V_prov = V_prov.toarray()
    if backend.is_sparse(V_kab):
        V_kab = V_kab.toarray()

    V_e = sigma2_e * np.eye(n, dtype=np.float64)
    V_gamma = _nested_gamma_block(prepared, theta, interaction, n, backend)
    V = V_prov + V_kab + V_cell + V_gamma + V_e
    V = 0.5 * (V + V.T)
    return V


def build_V_nested_ar1(
    prepared: dict,
    theta: dict,
    interaction: str,
    backend: Backend,
):
    """Build V for nested IID+AR1 model (M7/M14)."""
    xp = backend.xp
    n = prepared["n"]
    n_prov = prepared["n_prov"]
    n_kab = prepared["n_kab"]
    T = prepared["T"]

    sigma2_prov = theta.get("sigma2_prov", theta.get("sigma2_v", 1.0))
    sigma2_kab = theta.get("sigma2_kab", theta.get("sigma2_u", 1.0))
    sigma2_eta = theta.get("sigma2_eta", 0.0)
    phi = theta.get("phi", 0.0)
    sigma2_e = theta.get("sigma2_e", 1.0)

    Z_prov = prepared["Z_prov"]
    Z_kab = prepared["Z_kab"]

    G_prov = sigma2_prov * xp.eye(n_prov, dtype=xp.float64)
    G_kab = sigma2_kab * xp.eye(n_kab, dtype=xp.float64)

    V_prov = _to_np_backend(backend, Z_prov) @ _to_np_backend(backend, G_prov) @ _to_np_backend(backend, Z_prov).T
    V_kab = _to_np_backend(backend, Z_kab) @ _to_np_backend(backend, G_kab) @ _to_np_backend(backend, Z_kab).T

    # AR(1) time
    G_t = G_ar1_core(T, sigma2_eta, phi, backend)
    V_cell = _nested_cell_v(prepared, G_t, n_kab, T, n, backend)
    if backend.is_sparse(V_cell):
        V_cell = V_cell.toarray()
    if backend.is_sparse(V_prov):
        V_prov = V_prov.toarray()
    if backend.is_sparse(V_kab):
        V_kab = V_kab.toarray()

    V_e = sigma2_e * np.eye(n, dtype=np.float64)
    V_gamma = _nested_gamma_block(prepared, theta, interaction, n, backend)
    V = V_prov + V_kab + V_cell + V_gamma + V_e
    V = 0.5 * (V + V.T)
    return V


# ---------------------------------------------------------------------------
# V diagnostics
# ---------------------------------------------------------------------------

