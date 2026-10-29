"""Plot-ready data helpers for Beta mixed models (no matplotlib).

Pure extractors for the thesis figures: each function returns a DataFrame,
ndarray, or dict that a plotting layer consumes directly. All accessors are
defensive: a missing block/field or a boundary/degenerate fit yields
empty/NaN output instead of an exception.

Functions:
    blup_area(fit)          area BLUPs, sorted (DataFrame)
    forest_fe(fit)          fixed effects forest data (DataFrame)
    blup_time(fit)          time BLUPs, mean per period (DataFrame)
    moran_scatter(fit, W)   Moran scatter inputs + permutation test (dict)
    resid_normality(fit)    Dunn-Smyth conditional residuals (ndarray)
    resid_fit(fit)          fitted vs Pearson residuals (DataFrame)
    re_area(fit)            area BLUP values (ndarray)
    re_time(fit)            all time BLUP values (ndarray)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _opt(fit) -> dict:
    return getattr(fit, "_opt", None) or {}


def _blocks(des: dict) -> dict:
    return des.get("blocks", {}) or {}


def _bhat(opt: dict) -> np.ndarray:
    return np.asarray(opt.get("b_hat", []), dtype=float).ravel()


def _phi(opt: dict) -> float:
    phi = opt.get("phi")
    if phi is None:
        logphi = opt.get("logphi")
        phi = np.exp(logphi) if logphi is not None else np.nan
    try:
        return float(phi)
    except (TypeError, ValueError):
        return float("nan")


def _labels(fit, attr: str, fallback) -> list:
    """Human labels from fit._area_labels/_time_labels dict, else metadata."""
    lab = getattr(fit, attr, None)
    if isinstance(lab, dict) and lab:
        n = max(int(k) for k in lab) + 1
        return [lab.get(i, i) for i in range(n)]
    return list(fallback or [])


def _mu_hat(fit, opt: dict, n: int) -> np.ndarray:
    """Conditional mean (clipped); NaN vector when unavailable."""
    try:
        mu = np.asarray(fit.mu_hat, dtype=float).ravel()
        if mu.size == n:
            return np.clip(mu, 1e-15, 1.0 - 1e-15)
    except Exception:
        pass
    mu = np.asarray(opt.get("mu_hat", np.full(n, np.nan)), dtype=float).ravel()
    if mu.size != n:
        return np.full(n, np.nan)
    return np.clip(mu, 1e-15, 1.0 - 1e-15)


def _area_idx(blocks: dict) -> list:
    for key in ("area", "prov"):
        idx = blocks.get(key)
        if idx:
            return [int(i) for i in idx]
    return []


def _time_mean(des: dict, blocks: dict, b: np.ndarray):
    """Mean BLUP per period from the cell/time block; (None, labels) if absent.

    Area-specific models (AR(1) or IID) carry an area_time / cell block ordered
    area-major, averaged over areas per time. Global-time models carry a time
    block directly.
    """
    cell_block = blocks.get("area_time") or blocks.get("cell")
    t_block = blocks.get("time")
    if cell_block:
        idx = [int(i) for i in cell_block if 0 <= int(i) < b.size]
        vals = b[idx]
        m = int(des.get("m", 0) or 0)
        T = int(des.get("T", 0) or 0)
        if m > 0 and T > 0 and vals.size == m * T:
            return vals.reshape(m, T).mean(axis=0)
        return None
    if t_block:
        idx = [int(i) for i in t_block if 0 <= int(i) < b.size]
        return b[idx]
    return None


def blup_area(fit) -> pd.DataFrame:
    """Area BLUPs: columns [label_area, blup], sorted ascending by blup."""
    opt = _opt(fit)
    des = opt.get("des", {}) or {}
    blocks = _blocks(des)
    b = _bhat(opt)
    idx = [i for i in _area_idx(blocks) if 0 <= i < b.size]
    if not idx:
        return pd.DataFrame({"label_area": [], "blup": []})
    labels = _labels(fit, "_area_labels", des.get("area_labels", []))
    lab = [labels[i] if i < len(labels) else i for i in idx]
    out = pd.DataFrame({"label_area": lab, "blup": b[idx]})
    return out.sort_values("blup", kind="stable").reset_index(drop=True)


def forest_fe(fit) -> pd.DataFrame:
    """Fixed effects on the original predictor scale.

    Columns [param, est, lo, hi]; lo/hi = est +/- 1.96*se. Structural raw_*
    parameters are excluded by construction (fixed block only).
    """
    opt = _opt(fit)
    inf = getattr(fit, "_inf", None) or {}
    p = int(opt.get("p", 0) or 0)
    labels = list(inf.get("param_labels", []) or [])
    try:
        est = np.asarray(fit.beta_original, dtype=float).ravel()
        se = np.asarray(fit.se_original, dtype=float).ravel()
    except Exception:
        est = np.asarray(inf.get("estimate", []), dtype=float).ravel()
        se = np.asarray(inf.get("se_fixed", inf.get("se", [])), dtype=float).ravel()
    p = min(p if p > 0 else min(est.size, se.size), est.size, se.size)
    if p <= 0:
        return pd.DataFrame({"param": [], "est": [], "lo": [], "hi": []})
    if len(labels) < p:
        labels = labels + [f"beta{i}" for i in range(len(labels), p)]
    est, se = est[:p], se[:p]
    return pd.DataFrame({
        "param": labels[:p],
        "est": est,
        "lo": est - 1.96 * se,
        "hi": est + 1.96 * se,
    })


def blup_time(fit) -> pd.DataFrame:
    """Mean time BLUP per period: columns [label_tahun, blup_mean]."""
    opt = _opt(fit)
    des = opt.get("des", {}) or {}
    vals = _time_mean(des, _blocks(des), _bhat(opt))
    if vals is None or vals.size == 0:
        return pd.DataFrame({"label_tahun": [], "blup_mean": []})
    labels = _labels(fit, "_time_labels", des.get("time_labels", []))
    if len(labels) != vals.size:
        labels = [str(i) for i in range(vals.size)]
    return pd.DataFrame({"label_tahun": labels, "blup_mean": vals})


def moran_scatter(fit, W) -> dict:
    """Moran scatter inputs and permutation test.

    Returns dict(resid_area, lag, I, p_value) with resid_area = y - mu_hat,
    lag = W (resid - mean), and I/p from stmmlib.beta.report._moran (999
    permutations, fixed seed 42). NaN I/p when W is missing or mismatched.
    """
    opt = _opt(fit)
    y = np.asarray(opt.get("y", []), dtype=float).ravel()
    if y.size == 0:
        return {"resid_area": np.array([]), "lag": np.array([]),
                "I": float("nan"), "p_value": float("nan")}
    resid = y - _mu_hat(fit, opt, y.size)
    nan_out = {"resid_area": resid, "lag": np.full(y.size, np.nan),
               "I": float("nan"), "p_value": float("nan")}
    if W is None or resid.size < 3:
        return nan_out
    try:
        Wt = W.toarray() if hasattr(W, "toarray") else np.asarray(W, dtype=float)
        Wt = np.asarray(Wt, dtype=float)
        if Wt.ndim != 2 or Wt.shape[0] != resid.size or Wt.shape[1] != resid.size:
            return nan_out
        from stmmlib.beta.report import _moran
        lag = Wt @ (resid - np.mean(resid))
        I, p = _moran(resid, Wt)
        return {"resid_area": resid, "lag": lag,
                "I": float(I), "p_value": float(p)}
    except Exception:
        return nan_out


def resid_normality(fit) -> np.ndarray:
    """Dunn-Smyth conditional residuals, same definition as
    _diag_beta_advanced.ds_residuals_from_fit: Phi^-1(F(y)) with the fitted
    conditional mu and phi, clipped to [-8, 8]."""
    opt = _opt(fit)
    y = np.asarray(opt.get("y", []), dtype=float).ravel()
    if y.size == 0:
        return np.array([])
    mu = _mu_hat(fit, opt, y.size)
    phi = _phi(opt)
    if not np.isfinite(phi) or phi <= 0:
        return np.full(y.size, np.nan)
    from stmmlib.beta.report import _dunn_smyth
    return np.clip(np.asarray(_dunn_smyth(y, mu, phi), dtype=float), -8.0, 8.0)


def resid_fit(fit) -> pd.DataFrame:
    """Fitted vs Pearson residuals: [fitted, resid_pearson].

    Pearson = (y - mu) / sqrt(mu(1-mu)/(1+phi)) with the conditional mu.
    """
    opt = _opt(fit)
    y = np.asarray(opt.get("y", []), dtype=float).ravel()
    if y.size == 0:
        return pd.DataFrame({"fitted": [], "resid_pearson": []})
    mu = _mu_hat(fit, opt, y.size)
    phi = _phi(opt)
    var = mu * (1.0 - mu) / (1.0 + phi) if (np.isfinite(phi) and phi > 0) \
        else np.full(y.size, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = (y - mu) / np.sqrt(np.clip(var, 1e-12, None))
    return pd.DataFrame({"fitted": mu, "resid_pearson": resid})


def re_area(fit) -> np.ndarray:
    """Area BLUP values in block order (ndarray)."""
    opt = _opt(fit)
    des = opt.get("des", {}) or {}
    b = _bhat(opt)
    idx = [i for i in _area_idx(_blocks(des)) if 0 <= i < b.size]
    if not idx:
        return np.array([], dtype=float)
    return b[idx].copy()


def re_time(fit) -> np.ndarray:
    """All time BLUP values: full area_time block when area-specific,
    time block when global (ndarray; empty when no time block)."""
    opt = _opt(fit)
    des = opt.get("des", {}) or {}
    blocks = _blocks(des)
    b = _bhat(opt)
    for key in ("area_time", "cell"):
        block = blocks.get(key)
        if block:
            idx = [int(i) for i in block if 0 <= int(i) < b.size]
            return b[idx].copy()
    block = blocks.get("time")
    if block:
        idx = [int(i) for i in block if 0 <= int(i) < b.size]
        return b[idx].copy()
    return np.array([], dtype=float)
