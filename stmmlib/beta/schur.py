"""Exact Schur-complement terms for blocked (area-specific) Beta models."""

from __future__ import annotations

import numpy as np


def _blocked_hinv_terms(Z, w, Q, m, T):
    """Exact H^{-1} and Schur traces for blocked (area-specific) models.

    H = Z'diag(w)Z + Q with Z = [Z_area | Z_cell] one-hot layout.
    Schur: C_a = Qc_a + diag(w_cell[a]); S = diag(roww) + Q_a - diag(delta).
    Returns dict:
      Hinv    : dense (q, q) inverse  (q = m + m*T)
      D       : diag(Z Hinv Z') (n,) via one-hot indexing -- O(n)
      tr_area : tr(Hinv[:m,:m] Q_a)
      tr_cell : sum_a tr(Hinv[blk_a, blk_a] Qc_a)
    Mathematically identical to dense inv(H) (within fp tolerance).
    """
    n = Z.shape[0]
    area_idx = Z[:, :m].argmax(1)
    cell_idx = Z[:, m:].argmax(1)
    cell_area = cell_idx // T
    cell_time = cell_idx % T
    Q_a = Q[:m, :m]
    Qc_stack = np.stack([Q[m + a * T: m + (a + 1) * T, m + a * T: m + (a + 1) * T]
                         for a in range(m)])

    w_cell = np.zeros((m, T), dtype=np.float64)
    np.add.at(w_cell, (cell_area, cell_time), w)
    roww = np.bincount(area_idx, weights=w, minlength=m)

    Cinv = np.empty((m, T, T), dtype=np.float64)
    Cinvw = np.empty((m, T), dtype=np.float64)
    cwCinv = np.empty((m, T), dtype=np.float64)
    delta = np.empty(m, dtype=np.float64)
    for a in range(m):
        C_a = Qc_stack[a] + np.diag(w_cell[a])
        try:
            Cinv[a] = np.linalg.inv(C_a)
            Cinvw[a] = Cinv[a] @ w_cell[a]
            cwCinv[a] = Cinv[a].T @ w_cell[a]
        except np.linalg.LinAlgError:
            Cinv[a] = np.linalg.pinv(C_a)
            Cinvw[a] = Cinv[a] @ w_cell[a]
            cwCinv[a] = Cinv[a].T @ w_cell[a]
        delta[a] = w_cell[a] @ cwCinv[a]

    S = np.diag(roww) + Q_a - np.diag(delta)
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)

    q = m + m * T
    Hinv = np.empty((q, q), dtype=np.float64)
    Hinv[:m, :m] = Sinv
    # Off-diagonal: Hinv[:m, blk_a] = -Sinv[:, a] cwCinv[a]'
    Hinv[:m, m:] = -(Sinv[:, :, None] * cwCinv[None, :, :]).reshape(m, m * T)
    Hinv[m:, :m] = Hinv[:m, m:].T
    # Cell blocks: Cinv_a (diag blocks) + Sinv[a,b] outer(Cinvw_a, cwCinv_b)
    cross = np.einsum("at,ab,bu->atbu", Cinvw, Sinv, cwCinv).reshape(m * T, m * T)
    Hinv[m:, m:] = cross
    for a in range(m):
        Hinv[m + a * T: m + (a + 1) * T, m + a * T: m + (a + 1) * T] += Cinv[a]

    D = (Hinv[area_idx, area_idx]
         + 2.0 * Hinv[area_idx, m + cell_idx]
         + Hinv[m + cell_idx, m + cell_idx])

    tr_area = float(np.sum(Hinv[:m, :m] * Q_a))
    tr_cell = 0.0
    for a in range(m):
        tr_cell += float(np.sum(
            Hinv[m + a * T: m + (a + 1) * T, m + a * T: m + (a + 1) * T] * Qc_stack[a]))

    return {"Hinv": Hinv, "D": D, "tr_area": tr_area, "tr_cell": tr_cell,
            "area_idx": area_idx, "cell_idx": cell_idx}


def _blocked_pen_terms(b, Q, m, T):
    """b'Qb split by block: (area part, cell part) for the penalty gradient."""
    b_a = b[:m]
    b_c = b[m:].reshape(m, T)
    Q_a = Q[:m, :m]
    pen_area = float(b_a @ Q_a @ b_a)
    Qc_stack = np.stack([Q[m + a * T: m + (a + 1) * T, m + a * T: m + (a + 1) * T]
                         for a in range(m)])
    pen_cell = float(np.einsum("at,atu,au->", b_c, Qc_stack, b_c))
    return pen_area, pen_cell
