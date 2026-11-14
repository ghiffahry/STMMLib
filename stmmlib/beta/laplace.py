"""Laplace log-determinants and the parameter box for the Beta engine."""

from __future__ import annotations

import numpy as np


def laplace_block_logdets(Z, w, Q, m, T):
    """logdet(H) and logdet(Q) via exact Schur complement (area-specific, iid cell).

    H = Z' diag(w) Z + Q, Q = block_diag(Q_area, Q_cell with per-area blocks).
    Per-area cells are coupled only within their own area, so:
        logdet(H) = logdet(C) + logdet(S),  S = A - B C^{-1} B'
    where the C (per-area) and S (m x m) blocks are small.
    Mathematically identical to slogdet of the dense matrix (within fp tolerance).
    """
    n = Z.shape[0]
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

    logdet_C = 0.0
    S_diag = np.empty(m, dtype=np.float64)
    for a in range(m):
        C_a = Q_c[a*T:(a+1)*T, a*T:(a+1)*T] + np.diag(w_cell[a])
        logdet_C += np.linalg.slogdet(C_a)[1]
        try:
            cwCinv = np.linalg.solve(C_a, w_cell[a])
        except np.linalg.LinAlgError:
            cwCinv = np.linalg.lstsq(C_a, w_cell[a], rcond=None)[0]
        S_diag[a] = roww[a] + Q_a_diag[a] - w_cell[a] @ cwCinv

    if np.allclose(Q_a, np.diag(Q_a_diag)):
        logdet_S = np.sum(np.log(np.maximum(S_diag, 1e-12)))
    else:
        S_full = np.diag(S_diag) + (Q_a - np.diag(Q_a_diag))
        logdet_S = np.linalg.slogdet(S_full)[1]

    logdet_H_block = logdet_C + logdet_S

    logdet_Q = np.linalg.slogdet(Q_a)[1]
    for a in range(m):
        logdet_Q += np.linalg.slogdet(Q_c[a*T:(a+1)*T, a*T:(a+1)*T])[1]

    return logdet_H_block, logdet_Q


def beta_parameter_bounds(cfg, p: int, ksd: int, q: int):
    """The box bounds of the parameter vector [beta, logphi, logs, raw_dep].

    One source of truth: the solver and the LRT refinement both take their
    bounds from here, so the two cannot drift apart. The LRT refinement has
    to clip to the same box the fit was estimated in, otherwise the
    restricted maximum is taken outside the model.
    """
    bounds = [tuple(cfg.beta_bounds)] * p
    bounds.append(tuple(cfg.logphi_bounds))
    bounds.extend([tuple(cfg.logs_bounds)] * ksd)
    bounds.extend([(-np.arctanh(cfg.dep_margin),
                    np.arctanh(cfg.dep_margin))] * q)
    return bounds
