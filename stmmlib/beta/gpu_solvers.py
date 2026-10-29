"""GPU-accelerated (torch) variants of the Laplace hot loops.

Only the per-area T x T batched solves/logdets move to GPU; the cheap
curvature/score (numpy digamma/polygamma) and the outer L-BFGS-B stay on CPU.
Numerically identical to the numpy blocked solvers in solvers.py within fp
tolerance (float64 on CUDA).

When CUDA is unavailable these functions fall back to the numpy reference.
"""

from __future__ import annotations

import numpy as np

_TORCH = None


def _get_torch():
    global _TORCH
    if _TORCH is None:
        try:
            import torch
            _TORCH = torch
        except Exception:
            _TORCH = False
    return _TORCH


def gpu_available() -> bool:
    torch = _get_torch()
    return bool(torch) and torch.cuda.is_available() and torch.cuda.device_count() > 0


def _to_torch(torch, arr, device):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float64)).to(device)


def beta_mode_newton_blocked_gpu(X, Z, Q, m, T, beta, logphi, y, max_iter=50,
                                 start_b=None, inner_tol=1e-6,
                                 _numpy=None):
    """Batched-torch version of beta_mode_newton_blocked (see solvers.py)."""
    import torch

    from stmmlib.beta.solvers import beta_nll_curvature, _beta_nll_score

    if _numpy is None:
        from stmmlib.beta.solvers import beta_mode_newton_blocked as _numpy

    n, p = X.shape
    phi = np.exp(logphi)
    area_idx = Z[:, :m].argmax(1)
    cell_idx = Z[:, m:].argmax(1)
    cell_area = cell_idx // T
    cell_time = cell_idx % T

    u = np.zeros(m, dtype=np.float64) if start_b is None else start_b[:m].astype(np.float64).copy()
    v = np.zeros(m * T, dtype=np.float64) if start_b is None else start_b[m:].astype(np.float64).copy()

    Q_a = Q[:m, :m]
    Qc_by_area = np.stack([Q[m:, m:][a*T:(a+1)*T, a*T:(a+1)*T] for a in range(m)])
    Q_a_diag = np.ascontiguousarray(np.diag(Q_a))

    dev = "cuda"
    t_Qc = _to_torch(torch, Qc_by_area, dev)          # (m, T, T)
    t_Qa = _to_torch(torch, Q_a, dev)                 # (m, m)
    t_Qa_diag = _to_torch(torch, Q_a_diag, dev)       # (m,)
    t_area = torch.from_numpy(area_idx).to(dev)
    t_carea = torch.from_numpy(cell_area).to(dev)
    t_ctime = torch.from_numpy(cell_time).to(dev)

    for it in range(max_iter):
        eta = X @ beta + u[area_idx] + v[cell_idx]
        w = beta_nll_curvature(eta, y, phi)
        s = _beta_nll_score(eta, y, phi)

        w_cell = np.zeros((m, T), dtype=np.float64)
        s_cell = np.zeros((m, T), dtype=np.float64)
        np.add.at(w_cell, (cell_area, cell_time), w)
        np.add.at(s_cell, (cell_area, cell_time), s)
        roww = np.bincount(area_idx, weights=w, minlength=m)

        g_u = np.bincount(area_idx, weights=s, minlength=m) + Q_a @ u
        g_v = s_cell.ravel() + (Q[m:, m:] @ v)

        t_wc = _to_torch(torch, w_cell, dev)          # (m, T)
        t_gv = _to_torch(torch, g_v, dev).reshape(m, T, 1)
        t_gv2 = _to_torch(torch, g_v, dev).reshape(m, T)
        C = t_Qc + torch.diag_embed(t_wc)             # (m, T, T)
        cwCinv = torch.linalg.solve(C, t_wc.unsqueeze(2))[:, :, 0]      # (m, T)
        Cinv_gv = torch.linalg.solve(C, t_gv)[:, :, 0]                 # (m, T)

        S_diag = roww + Q_a_diag - (t_wc * cwCinv).sum(1).cpu().numpy()
        rhs_u = -g_u + (t_wc * Cinv_gv).sum(1).cpu().numpy()

        if np.allclose(Q_a, np.diag(Q_a_diag)):
            du = rhs_u / np.maximum(S_diag, 1e-12)
        else:
            S_full = np.diag(S_diag) + (Q_a - np.diag(Q_a_diag))
            try:
                du = np.linalg.solve(S_full, rhs_u)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(S_full, rhs_u, rcond=None)[0]

        t_du = torch.from_numpy(du).to(dev)
        rhs_v = -t_gv2 - t_wc * t_du.unsqueeze(1)      # (m, T)
        dv_b = torch.linalg.solve(C, rhs_v.unsqueeze(2))[:, :, 0]
        dv = dv_b.cpu().numpy().ravel()

        scale = 1.0
        for _ in range(12):
            u_new = u + scale * du
            v_new = v + scale * dv
            eta_new = X @ beta + u_new[area_idx] + v_new[cell_idx]
            w_new = beta_nll_curvature(eta_new, y, phi)
            if np.all(np.isfinite(w_new)):
                u, v = u_new, v_new
                break
            scale *= 0.5
        else:
            break

        scale = 1.0 + np.max(np.abs(np.concatenate([u, v])))
        if np.max(np.abs(np.concatenate([du, dv]))) < inner_tol * scale:
            return np.concatenate([u, v]), True, "blocked-gpu", it

    return np.concatenate([u, v]), False, "blocked-gpu", it


def laplace_block_logdets_gpu(Z, w, Q, m, T, _numpy=None):
    """Batched-torch version of laplace_block_logdets (see solvers.py)."""
    import torch

    from stmmlib.beta.solvers import laplace_block_logdets as _numpy

    area_idx = Z[:, :m].argmax(1)
    cell_idx = Z[:, m:].argmax(1)
    cell_area = cell_idx // T
    cell_time = cell_idx % T
    Q_a = Q[:m, :m]
    Q_c = Q[m:, m:]

    roww = np.bincount(area_idx, weights=w, minlength=m)
    w_cell = np.zeros((m, T), dtype=np.float64)
    np.add.at(w_cell, (cell_area, cell_time), w)
    Q_a_diag = np.ascontiguousarray(np.diag(Q_a))

    dev = "cuda"
    t_Qc = _to_torch(torch, np.stack([Q_c[a*T:(a+1)*T, a*T:(a+1)*T] for a in range(m)]), dev)
    t_Qa = _to_torch(torch, Q_a, dev)
    t_Qa_diag = _to_torch(torch, Q_a_diag, dev)
    t_wc = _to_torch(torch, w_cell, dev)
    C = t_Qc + torch.diag_embed(t_wc)
    _sgn, ldet = torch.linalg.slogdet(C)
    logdet_C = float(ldet.sum().cpu().item())
    cwCinv = torch.linalg.solve(C, t_wc.unsqueeze(2))[:, :, 0]
    S_diag = roww + Q_a_diag - (t_wc * cwCinv).sum(1).cpu().numpy()

    if np.allclose(Q_a, np.diag(Q_a_diag)):
        logdet_S = np.sum(np.log(np.maximum(S_diag, 1e-12)))
    else:
        S_full = np.diag(S_diag) + (Q_a - np.diag(Q_a_diag))
        logdet_S = np.linalg.slogdet(S_full)[1]

    logdet_H = logdet_C + logdet_S
    logdet_Q = float(torch.linalg.slogdet(t_Qa)[1].cpu().item())
    logdet_Qblocks = float(torch.linalg.slogdet(t_Qc)[1].sum().cpu().item())
    logdet_Q = logdet_Q + logdet_Qblocks

    return logdet_H, logdet_Q
