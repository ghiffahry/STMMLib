"""Single precision-matrix builder for the Beta STMM.

The fit engine, inference, and cAIC all need the same Q(theta); this module
is the one implementation. It used to be copied three times, which is how
the criteria path drifted (wrong raw_dep slot, single squash for global AR1).

Layout of raw_dep is [rho(sar), phi(ar1), rho_gamma, phi_gamma], so the phi
slot is 1 when the area block is SAR and 0 otherwise.
"""

from __future__ import annotations

import numpy as np

from stmmlib.beta.precision import ar1_precision


def _block_diag(blocks):
    """Dense block-diagonal assembly in one allocation."""
    blocks = [b.toarray() if hasattr(b, "toarray")
              else np.asarray(b, dtype=np.float64) for b in blocks]
    sizes = [b.shape[0] for b in blocks]
    out = np.zeros((sum(sizes), sum(sizes)), dtype=np.float64)
    off = 0
    for blk, sz in zip(blocks, sizes):
        out[off:off + sz, off:off + sz] = blk
        off += sz
    return out


def build_beta_Q(s, des, W, logs, raw_dep, ksd, q_dep, *,
                 rho_bound, phi_bound, backend, use_blocked):
    """Build the full dense precision Q exactly as the Laplace engine does.

    Args:
        s: structure dict from stmm_beta_structure.
        des: design dict (Z, blocks, W_gamma, B_area, B_time).
        W: spatial weight matrix for the area block, or None.
        logs: log SDs, layout [area, time, (gamma)] or nested [prov, kab, time].
        raw_dep: raw dependence params,
            [rho(sar), phi(ar1), rho_gamma, phi_gamma].
        ksd: number of log-SD entries.
        q_dep: number of raw dependence entries.
        rho_bound: SAR spectral bound.
        phi_bound: AR(1) stationarity bound.
        backend: array backend.
        use_blocked: True when Z is [Z_area, Z_cell] area-specific; the AR(1)
            time precision is then block-diagonal per area.

    Returns:
        Dense q x q precision matrix for the whole random-effect vector.
    """
    blocks = des["blocks"]
    is_nested = "prov" in blocks
    n_area = len(blocks.get("area", []))
    n_time = len(blocks.get("area_time", blocks.get("time", [])))
    n_prov = len(blocks.get("prov", []))
    n_kab = len(blocks.get("kab", []))
    n_cell = len(blocks.get("cell", []))
    n_gamma = len(blocks.get("interaction", []))

    has_sar = s["area"] in ("sar", "seblup_rao_yu")
    has_ar1 = s["time"] == "ar1"
    rho_i = 0 if has_sar else None
    ph_i = (1 if has_sar else 0) if has_ar1 else None

    blocks_Q = []
    if is_nested:
        blocks_Q.append(np.exp(-2.0 * logs[0]) * np.eye(n_prov, dtype=np.float64))
        blocks_Q.append(np.exp(-2.0 * logs[1]) * np.eye(n_kab, dtype=np.float64))
        if has_ar1:
            phi_t = phi_bound * np.tanh(raw_dep[ph_i]) if q_dep > ph_i else 0.0
            TQ = ar1_precision(des["T"], logs[2], phi_t, backend,
                               phi_bound=phi_bound)
            if hasattr(TQ, "toarray"):
                TQ = TQ.toarray()
            blocks_Q.append(_block_diag([TQ] * n_kab))
        else:
            blocks_Q.append(np.exp(-2.0 * logs[2]) * np.eye(n_cell, dtype=np.float64))
    else:
        if s["area"] == "iid":
            Q_a = np.exp(-2.0 * logs[0]) * np.eye(n_area, dtype=np.float64)
        elif has_sar:
            rho = rho_bound * np.tanh(raw_dep[rho_i]) if q_dep > rho_i else 0.0
            W_arr = W if W is not None else des.get("W_gamma")
            if W_arr is None:
                W_arr = np.eye(n_area, dtype=np.float64)
            elif hasattr(W_arr, "toarray"):
                W_arr = W_arr.toarray()
            W_arr = np.asarray(W_arr, dtype=np.float64)[:n_area, :n_area]
            A = np.eye(n_area, dtype=np.float64) - rho * W_arr
            Q_a = np.exp(-2.0 * logs[0]) * (A.T @ A)
        else:
            raise ValueError(f"Unknown area type: {s['area']!r}")
        blocks_Q.append(Q_a)

        if has_ar1:
            phi_t = phi_bound * np.tanh(raw_dep[ph_i]) if q_dep > ph_i else 0.0
            ls_t = logs[1] if len(logs) > 1 else 0.0
            if use_blocked:
                TQ = ar1_precision(des["T"], ls_t, phi_t, backend,
                                   phi_bound=phi_bound)
                if hasattr(TQ, "toarray"):
                    TQ = TQ.toarray()
                Q_t = _block_diag([TQ] * n_area)
            else:
                Q_t = ar1_precision(n_time, ls_t, phi_t, backend,
                                    phi_bound=phi_bound)
                if hasattr(Q_t, "toarray"):
                    Q_t = Q_t.toarray()
        else:
            Q_t = (np.exp(-2.0 * logs[1]) * np.eye(n_time, dtype=np.float64)
                   if len(logs) > 1 else np.eye(n_time, dtype=np.float64))
        blocks_Q.append(Q_t)

    Q_base = _block_diag(blocks_Q)
    if n_gamma <= 0:
        return Q_base

    Q_g = beta_Q_gamma(s, des, des.get("m", n_area), des.get("T", n_time),
                       n_gamma, logs, raw_dep, ksd, q_dep,
                       has_sar, has_ar1, backend)
    if Q_g is None or Q_g.size == 0:
        return Q_base
    return _block_diag([Q_base, Q_g])


def beta_Q_gamma(s, des, m, T, n_gamma, logs, raw_dep, ksd, q_dep,
                 has_sar, has_ar1, backend):
    """Interaction precision block Q_g = G_g^{-1} (shared solver + inference).

    Matches the Gaussian marginal covariance V_gamma = Z_gamma G_gamma Z_gamma':
        iid:      Q = s2p I
        spatial:  Q = s2p kron(qa^{-1}, I_T),  qa = B_area'A'A B_area
        temporal: Q = s2p kron(I_m, qt^{-1}),  qt = B_time' Ar B_time
        st:       Q = s2p kron(qa^{-1}, qt^{-1})
    """
    from scipy.sparse import kron as _sp_kron, identity as _sp_ident
    # logs has exactly ksd entries (area, time, [gamma]); gamma log-sd at ksd-1
    if n_gamma <= 0 or len(logs) < ksd:
        return None
    s2p = np.exp(-2.0 * logs[ksd - 1])
    itype = s["interaction"]
    B_area = des["B_area"]
    B_time = des["B_time"]
    rb_g = des.get("rho_bound_g", 0.95)
    _rg_i = (1 if has_sar else 0) + (1 if has_ar1 else 0)
    # raw_dep order: [rho(sar), phi(ar1), rho_gamma(spatial|st), phi_gamma(temporal|st)]
    _phg_i = _rg_i + (1 if itype in ("spatial", "st") else 0)
    qa_inv = qt_inv = None
    if itype in ("spatial", "st"):
        rho_g = rb_g * np.tanh(raw_dep[_rg_i]) if q_dep > _rg_i else 0.0
        Wg = des.get("W_gamma")
        if Wg is None:
            Wg = np.eye(m, dtype=np.float64)
        elif hasattr(Wg, "toarray"):
            Wg = Wg.toarray()
        Ag = np.eye(m, dtype=np.float64) - rho_g * np.asarray(Wg, dtype=np.float64)
        qa = B_area.T @ (Ag.T @ Ag) @ B_area
        qa_inv = np.linalg.inv(qa)
    if itype in ("temporal", "st"):
        from stmmlib.beta.precision import ar1_precision as _ar1_prec
        _pb = des.get("phi_bound", 0.95)
        phi_g = _pb * np.tanh(raw_dep[_phg_i]) if q_dep > _phg_i else 0.0
        Ap = _ar1_prec(T, 0.0, np.arctanh(np.clip(phi_g / _pb, -0.999, 0.999)), backend,
                       phi_bound=_pb)
        if hasattr(Ap, "toarray"):
            Ap = Ap.toarray()
        Ar = np.linalg.inv(np.asarray(Ap, dtype=np.float64))
        qt = B_time.T @ Ar @ B_time
        qt_inv = np.linalg.inv(qt)
    if itype == "iid":
        return s2p * np.eye(n_gamma, dtype=np.float64)
    if itype == "spatial":
        # Z_gamma = Z_full @ kron(B_area, I_T) is area-major.
        return np.asarray((s2p * _sp_kron(qa_inv, _sp_ident(T), format="csr")).toarray())
    if itype == "temporal":
        return np.asarray((s2p * _sp_kron(_sp_ident(m), qt_inv, format="csr")).toarray())
    if itype == "st":
        return np.asarray((s2p * _sp_kron(qa_inv, qt_inv, format="csr")).toarray())
    return None

