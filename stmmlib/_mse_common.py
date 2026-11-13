"""Shared helpers for the parametric bootstrap MSE modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

_N_DRAWS = 128


def _onehot(index, n_cols):
    index = np.asarray(index, dtype=np.int64).ravel()
    n = index.shape[0]
    Z = np.zeros((n, n_cols), dtype=np.float64)
    Z[np.arange(n), index] = 1.0
    return Z


def _row_means(values, area_idx, m):
    values = np.asarray(values, dtype=np.float64).ravel()
    out = np.zeros(m, dtype=np.float64)
    counts = np.zeros(m, dtype=np.int64)
    np.add.at(out, area_idx, values)
    np.add.at(counts, area_idx, 1)
    mean = np.full(m, np.nan, dtype=np.float64)
    mask = counts > 0
    mean[mask] = out[mask] / counts[mask]
    return mean


def _invlogit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _to_np(a):
    return np.asarray(a.toarray(), dtype=np.float64) if hasattr(a, "toarray") else np.asarray(a, dtype=np.float64)


def _draw_mvn(rng, cov):
    cov = _to_np(cov)
    scale = float(np.trace(cov)) / max(cov.shape[0], 1)
    cov = cov + 1e-12 * np.eye(cov.shape[0]) * max(scale, 1e-12)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(cov.shape[0]))
    return L @ rng.standard_normal(cov.shape[0])


def _draw_from_precision(rng, Q):
    from scipy.linalg import solve_triangular
    Q = _to_np(Q)
    q = np.linalg.cholesky(Q + 1e-12 * np.eye(Q.shape[0]))
    return solve_triangular(q.T, rng.standard_normal(Q.shape[0]), lower=False)


def _frame(labels, fit_pred, preds, targets, n_ok, B, pred_name):
    min_ok = min(B, max(10, B // 4))
    if n_ok < min_ok:
        raise RuntimeError(
            f"parametric bootstrap MSE: only {n_ok}/{B} replications succeeded "
            f"(need at least {min_ok}). The refit is failing for most draws.")
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    pred = np.asarray(fit_pred, dtype=np.float64)
    mse = np.square(preds - targets).mean(axis=0)
    rmse = np.sqrt(mse)
    rel_rmse = rmse / np.maximum(np.abs(pred), 1e-12)
    return pd.DataFrame({
        "area": list(labels),
        pred_name: pred,
        "mse": mse,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "n_boot_ok": int(n_ok),
    })
