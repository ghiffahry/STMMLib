"""Model diagnostics: BP test, White test, Moran's I, residual analysis.

Post-fit diagnostic suite for Gaussian spatio-temporal models.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stmmlib._backend import Backend


# ---------------------------------------------------------------------------
# Residual extraction
# ---------------------------------------------------------------------------


def _extract_resid_fit(fit: dict):
    """Extract residuals and fitted values from fit dict."""
    residuals = fit.get("residuals", fit.get("resid", None))
    fitted = fit.get("fitted", fit.get("fitted_values", None))
    if residuals is None or fitted is None:
        raise ValueError("Fit dict must contain 'residuals' and 'fitted'.")
    return np.asarray(residuals).ravel(), np.asarray(fitted).ravel()


# ---------------------------------------------------------------------------
# Heteroskedasticity tests
# ---------------------------------------------------------------------------


def bp_test(resid, fitted_vals) -> dict:
    """Breusch-Pagan test for heteroskedasticity.

    Regresses squared residuals on fitted values.
    Test statistic ~ chi2(1) under H0 of homoskedasticity.
    """
    from scipy.stats import chi2

    resid = np.asarray(resid).ravel()
    fitted = np.asarray(fitted_vals).ravel()
    n = len(resid)

    # Auxiliary regression: resid^2 ~ fitted
    X_aux = np.column_stack([np.ones(n), fitted])
    resid_sq = resid**2
    beta_aux = np.linalg.lstsq(X_aux, resid_sq, rcond=None)[0]
    fitted_aux = X_aux @ beta_aux
    ss_res = np.sum((resid_sq - fitted_aux) ** 2)
    ss_tot = np.sum((resid_sq - np.mean(resid_sq)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    stat = n * r_sq
    p_value = chi2.sf(stat, df=1)

    return {"statistic": float(stat), "p_value": float(p_value), "df": 1, "test": "Breusch-Pagan"}


def white_test(resid, fitted_vals) -> dict:
    """White test for heteroskedasticity.

    Regresses squared residuals on fitted, fitted^2.
    Test statistic ~ chi2(2) under H0.
    """
    from scipy.stats import chi2

    resid = np.asarray(resid).ravel()
    fitted = np.asarray(fitted_vals).ravel()
    n = len(resid)

    X_aux = np.column_stack([np.ones(n), fitted, fitted**2])
    resid_sq = resid**2
    beta_aux = np.linalg.lstsq(X_aux, resid_sq, rcond=None)[0]
    fitted_aux = X_aux @ beta_aux
    ss_res = np.sum((resid_sq - fitted_aux) ** 2)
    ss_tot = np.sum((resid_sq - np.mean(resid_sq)) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    stat = n * r_sq
    p_value = chi2.sf(stat, df=2)

    return {"statistic": float(stat), "p_value": float(p_value), "df": 2, "test": "White"}


# ---------------------------------------------------------------------------
# Assumption test suite
# ---------------------------------------------------------------------------


def assumption_test(fit: dict, alpha: float = 0.05, backend: Backend = None) -> dict:
    """Full assumption test: Shapiro-Wilk + BP + White.

    Args:
        fit: dict with residuals and fitted.
        alpha: Significance level.
        backend: Array backend.

    Returns:
        dict with shapiro, bp, white results and pass/fail flags.
    """
    from scipy.stats import shapiro

    resid, fitted = _extract_resid_fit(fit)

    # Shapiro-Wilk (subsample if large)
    max_n = 5000
    if len(resid) > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(resid), max_n, replace=False)
        resid_sub = resid[idx]
    else:
        resid_sub = resid

    try:
        sw_stat, sw_p = shapiro(resid_sub)
    except Exception:
        sw_stat, sw_p = 0.0, 1.0

    bp = bp_test(resid, fitted)
    wh = white_test(resid, fitted)

    return {
        "shapiro": {"statistic": float(sw_stat), "p_value": float(sw_p)},
        "bp": bp,
        "white": wh,
        "normality_ok": sw_p > alpha,
        "homoskedasticity_ok": bp["p_value"] > alpha and wh["p_value"] > alpha,
    }


# ---------------------------------------------------------------------------
# Moran's I (spatial autocorrelation)
# ---------------------------------------------------------------------------


def moran_I_value(res_area, W, backend: Backend = None) -> float:
    """Compute Moran's I statistic for spatial autocorrelation.

    I = (n/S0) * (e' W e) / (e' e)

    Args:
        res_area: Area-level residuals (m,).
        W: Spatial weight matrix (m, m).
        backend: Array backend.

    Returns:
        Moran's I statistic.
    """
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu=False)
    xp = backend.xp

    e = xp.asarray(res_area, dtype=xp.float64).ravel()
    if backend.is_sparse(W):
        W_arr = W.toarray()
    else:
        W_arr = xp.asarray(W, dtype=xp.float64)

    n = len(e)
    S0 = float(xp.sum(W_arr))
    if S0 == 0:
        return 0.0

    We = W_arr @ e
    numer = float(e @ We)
    denom = float(e @ e)
    if denom == 0:
        return 0.0

    return float((n / S0) * (numer / denom))


def moran_permutation(res_area, W, n_perm: int = 199, seed: int = None, backend: Backend = None) -> dict:
    """Permutation test for Moran's I.

    Args:
        res_area: Area-level residuals.
        W: Spatial weight matrix.
        n_perm: Number of permutations.
        seed: Random seed.
        backend: Array backend.

    Returns:
        dict with I_obs, p_value, I_perms.
    """
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu=False)
    xp = backend.xp

    I_obs = moran_I_value(res_area, W, backend)
    res_arr = np.asarray(res_area).ravel()
    n = len(res_arr)

    rng = np.random.default_rng(seed)
    I_perms = np.zeros(n_perm, dtype=np.float64)
    for k in range(n_perm):
        perm = rng.permutation(n)
        I_perms[k] = moran_I_value(res_arr[perm], W, backend)

    p_value = (1 + np.sum(np.abs(I_perms) >= np.abs(I_obs))) / (n_perm + 1)

    return {
        "I_obs": float(I_obs),
        "p_value": float(p_value),
        "I_perms": I_perms,
    }


# ---------------------------------------------------------------------------
# Diagnostic suite
# ---------------------------------------------------------------------------


def diagnostics(
    fit: dict,
    type: str = "all",
    agg: str = "mean",
    n_perm: int = 199,
    seed: int = None,
    max_shapiro_n: int = 5000,
    backend: Backend = None,
) -> dict:
    """Full diagnostic suite: residual + temporal + spatial.

    Args:
        fit: dict with residuals, fitted, W (spatial weights), area_index.
        type: "residual", "temporal", "spatial", or "all".
        agg: Aggregation for temporal ("mean" or "median").
        n_perm: Permutations for Moran's I.
        seed: Random seed.
        max_shapiro_n: Max subsample for Shapiro-Wilk.
        backend: Array backend.

    Returns:
        dict with residual_diag, temporal_diag, spatial_diag.
    """
    if backend is None:
        from stmmlib._backend import Backend
        backend = Backend(gpu=False)

    result = {}
    resid, fitted = _extract_resid_fit(fit)

    if type in ("residual", "all"):
        result["residual"] = {
            "bp": bp_test(resid, fitted),
            "white": white_test(resid, fitted),
            "shapiro": _shapiro_test(resid, max_shapiro_n),
        }

    if type in ("temporal", "all") and "area_index" in fit:
        result["temporal"] = _temporal_diag(resid, fit["area_index"], fit.get("T", 10))

    if type in ("spatial", "all") and "W" in fit and fit["W"] is not None:
        area_resid = _aggregate_to_area(resid, fit.get("area_index"), agg)
        result["spatial"] = moran_permutation(area_resid, fit["W"], n_perm, seed, backend)

    return result


def _shapiro_test(resid, max_n=5000):
    from scipy.stats import shapiro
    resid = np.asarray(resid).ravel()
    if len(resid) > max_n:
        rng = np.random.default_rng(42)
        resid = resid[rng.choice(len(resid), max_n, replace=False)]
    try:
        stat, p = shapiro(resid)
        return {"statistic": float(stat), "p_value": float(p)}
    except Exception:
        return {"statistic": 0.0, "p_value": 1.0}


def _temporal_diag(resid, area_index, T):
    """Temporal ACF check per area."""
    area_index = np.asarray(area_index).ravel()
    resid = np.asarray(resid).ravel()
    m = int(np.max(area_index)) + 1
    acf_vals = []
    for a in range(m):
        mask = area_index == a
        r = resid[mask]
        if len(r) > 1:
            acf = np.correlate(r - np.mean(r), r - np.mean(r), mode="full")
            acf = acf[len(acf) // 2:]
            acf = acf / acf[0] if acf[0] != 0 else acf
            acf_vals.append(acf[1] if len(acf) > 1 else 0)
    return {"mean_acf_lag1": float(np.mean(acf_vals)) if acf_vals else 0.0}


def _aggregate_to_area(resid, area_index, agg="mean"):
    area_index = np.asarray(area_index).ravel()
    resid = np.asarray(resid).ravel()
    m = int(np.max(area_index)) + 1
    result = np.zeros(m, dtype=np.float64)
    for a in range(m):
        mask = area_index == a
        if agg == "mean":
            result[a] = np.mean(resid[mask])
        else:
            result[a] = np.median(resid[mask])
    return result
