"""Residual diagnostic suite (P9) for Beta and Gaussian GLMM fits.

One entry point per family, returning a plain dict of formal tests:
    - normality of (randomized) quantile residuals: Shapiro-Wilk,
      Anderson-Darling, KS vs N(0,1), skew, excess kurtosis
    - simulated envelope for the quantile residuals (Espinheira et al. 2008
      style, conditional on fitted mu/phi)
    - heteroskedasticity: Spearman of |r| vs fitted + BP-style OLS of r^2
    - influence: approximate Cook's distance via IRLS working weights
      (beta only; gaussian result dicts do not carry X)
    - residual-vs-predictor structure: Spearman per predictor column

All tests are designed to be reported directly in a thesis diagnostic table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from scipy.special import ndtri


def _normality_tests(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    out = {"n": n, "skew": float(stats.skew(r)),
           "excess_kurtosis": float(stats.kurtosis(r))}
    if n < 8:
        return {**out, "shapiro_p": np.nan, "ad_stat": np.nan,
                "ks_p": np.nan}
    sw = stats.shapiro(r if n <= 5000 else r[:5000])
    ad = stats.anderson(r, dist="norm")
    # scale AD stat by the documented small-sample correction factor
    _ad_p = float(ad.statistic / (1.0 + 0.75 / n + 2.25 / n**2))
    ks = stats.kstest(r, "norm")
    out.update({"shapiro_p": float(sw.pvalue), "ad_stat_scaled": _ad_p,
                "ks_p": float(ks.pvalue)})
    return out


def _heteroskedasticity(r: np.ndarray, fitted: np.ndarray) -> dict:
    r = np.asarray(r, float)
    fitted = np.asarray(fitted, float)
    ok = np.isfinite(r) & np.isfinite(fitted)
    rho_abs = stats.spearmanr(np.abs(r[ok]), fitted[ok]).statistic
    # BP-style: OLS of r^2 on fitted (z-scored); t-test on slope
    x = (fitted[ok] - fitted[ok].mean()) / (fitted[ok].std() + 1e-12)
    y2 = r[ok] ** 2
    slope = float(np.dot(x, y2 - y2.mean()) / np.dot(x, x))
    resid2 = y2 - y2.mean() - slope * x
    dof = max(len(x) - 2, 1)
    se = float(np.sqrt(np.dot(resid2, resid2) / dof / np.dot(x, x)))
    t = slope / se if se > 0 else 0.0
    return {"spearman_abs_r_fitted": float(rho_abs),
            "bp_slope": slope, "bp_t": float(t), "bp_p": float(2 * stats.t.sf(abs(t), dof))}


def _cook_leverage_beta(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate hat matrix diag and Cook's D via IRLS working weights."""
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    Q, _ = np.linalg.qr(Xw)
    h = np.clip(np.sum(Q * Q, axis=1), 0.0, 1.0 - 1e-12)
    p = X.shape[1]
    return h, h / (p * (1.0 - h) ** 2)  # (leverage, cooks D, unitless)


def residual_diagnostics_beta(fit, y_override=None) -> dict:
    """Full residual diagnostics for a BetastmmResult.

    Uses Dunn-Smyth randomized quantile residuals (conditional on b_hat),
    a conditional simulated envelope, and IRLS-weight Cook's distance.
    ``y_override`` evaluates residuals against a different response vector
    than the fitted one (out-of-sample misfit detection).
    """
    o = fit._opt
    y = np.asarray(o["y"], float) if y_override is None \
        else np.asarray(y_override, float)
    X = np.asarray(o["X"], float)
    beta = np.asarray(o["beta_hat"], float)
    phi = float(np.exp(o["logphi"]))
    mu = np.asarray(fit.mu_hat, float)
    r = np.asarray(fit.residuals("dunn_smyth"), float)

    out = {"family": "beta"}
    out["normality"] = _normality_tests(r)
    out["heteroskedasticity"] = _heteroskedasticity(r, mu)

    # Simulated envelope: draw k response vectors from the fitted conditional
    # model, compute their (non-randomized) quantile residuals, take per-obs
    # 2.5/97.5 quantiles as the band.
    rng = np.random.default_rng(42)
    k = 500
    a = mu * phi
    b = (1.0 - mu) * phi
    # Data residual: r = ndtri(F_beta(y))
    r = ndtri(np.clip(stats.beta.cdf(y, a, b), 1e-12, 1 - 1e-12))
    # Simulated residuals: same transform applied to drawn responses
    sims = np.stack([
        ndtri(np.clip(stats.beta.cdf(rng.beta(a, b), a, b), 1e-12, 1 - 1e-12))
        for _ in range(k)
    ])
    r_lo = np.quantile(sims, 0.025, axis=0)
    r_hi = np.quantile(sims, 0.975, axis=0)
    outside = (r < r_lo) | (r > r_hi)
    out["envelope"] = {
        "k": k, "n_outside": int(outside.sum()),
        "pct_outside": float(100.0 * outside.mean()),
        "max_abs_r": float(np.nanmax(np.abs(r))),
    }

    # Influence (beta curvature as working weights)
    try:
        from stmmlib.beta.solvers import beta_nll_curvature
        eta = X @ beta
        w = np.asarray(beta_nll_curvature(eta, y, phi), float)
        h, cook = _cook_leverage_beta(X, np.clip(w, 1e-12, None))
        top = np.argsort(cook)[::-1][:5]
        out["influence"] = {
            "cooks_d_max": float(cook.max()),
            "cooks_d_mean": float(cook.mean()),
            "leverage_max": float(h.max()),
            "top5_idx": [int(i) for i in top],
            "top5_cooks": [float(cook[i]) for i in top],
        }
    except Exception as e:  # noqa: BLE001
        out["influence"] = {"error": str(e)}

    # Residual vs each predictor (structure left in residuals?)
    x_cols = o.get("des", {}).get("x_cols", [])
    preds = {}
    for j, c in enumerate(x_cols):
        xv = X[:, j + 1]
        ok = np.isfinite(xv) & np.isfinite(r)
        if np.std(xv[ok]) < 1e-12:
            continue
        preds[c] = float(stats.spearmanr(r[ok], xv[ok]).statistic)
    out["vs_predictors_spearman"] = preds
    return out


def residual_diagnostics_gaussian(result: dict) -> dict:
    """Residual diagnostics for a gaussian stmm_normal_core result dict.

    No influence stats (X is not carried in the result dict).
    """
    r = np.asarray(result["residuals"], float).ravel()
    fitted = np.asarray(result["fitted"], float).ravel()
    se = float(np.std(r, ddof=1)) or 1.0
    rs = r / se  # standardized
    out = {"family": "gaussian"}
    out["normality"] = _normality_tests(rs)
    out["heteroskedasticity"] = _heteroskedasticity(rs, fitted)
    return out


def random_effects_diagnostics_beta(fit) -> dict:
    """P10: random-effects diagnostics for a BetastmmResult.

    Per block (area / area_time / interaction):
      - normality of the BLUPs (Shapiro, skew, excess kurtosis)
      - Moran's I on area-level BLUPs (spatial structure of the effects)
      - lag-1 ACF of year-averaged cell BLUPs (temporal structure)
    Plus: shrinkage (correlation of area BLUPs with the raw per-area signal)
    and Wald log-scale CIs for variance components.
    """
    o = fit._opt
    des = o.get("des", {})
    blocks = des.get("blocks", {})
    b = np.asarray(o["b_hat"], float)
    W = o.get("W")
    out = {"blocks": {}}

    def _norm(v):
        v = v[np.isfinite(v)]
        res = {"shapiro_p": float(stats.shapiro(v).pvalue) if len(v) >= 8 else np.nan,
               "skew": float(stats.skew(v)), "excess_kurtosis": float(stats.kurtosis(v)),
               "n": int(len(v))}
        # qq kuantil: korelasi kuantil sampel vs normal teoretis (ndtri sudah terimport)
        if len(v) >= 8:
            probs = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            theo = ndtri(probs)
            obs = np.quantile(v, probs)
            res["qq_corr"] = float(np.corrcoef(theo, obs)[0, 1])
            res["qq_points"] = {"theo": theo.tolist(), "obs": obs.tolist()}
        else:
            res["qq_corr"] = np.nan
            res["qq_points"] = {"theo": [], "obs": []}
        return res

    def _moran_i(v, Wm):
        # Pembanding E[I] netral sisi (tanpa uji p).
        z = v - v.mean()
        n = len(z)
        if Wm is None or n < 3:
            return np.nan
        if hasattr(Wm, "toarray"):
            Wm = Wm.toarray()
        Wm = np.asarray(Wm, float)[:n, :n]
        s0 = Wm.sum()
        if s0 <= 0:
            return np.nan
        return float(n / s0 * (z @ Wm @ z) / (z @ z))

    # Area block: normality + Moran
    if "area" in blocks:
        idx = np.asarray(blocks["area"], int)
        v = b[idx]
        info = _norm(v)
        info["moran_I"] = _moran_i(v, W)
        info["moran_EI"] = -1.0 / (len(v) - 1) if len(v) > 1 else np.nan
        out["blocks"]["area"] = info
    if "prov" in blocks:
        idx = np.asarray(blocks["prov"], int)
        out["blocks"]["prov"] = _norm(b[idx])

    # Cell / time block: normality + year-averaged ACF(1)
    cell_key = "area_time" if "area_time" in blocks else ("cell" if "cell" in blocks else None)
    if cell_key:
        idx = np.asarray(blocks[cell_key], int)
        v = b[idx]
        info = _norm(v)
        m = des.get("m")
        T = des.get("T")
        if m and T and len(v) == m * T:
            vmat = v.reshape(m, T)
            yr = vmat.mean(axis=0)  # year-averaged effect
            yc = yr - yr.mean()
            info["acf1"] = float(np.dot(yc[:-1], yc[1:]) /
                                 max(np.dot(yc, yc), 1e-12))
            # ljung-box per area pada deret BLUP area_time
            L = int(min(3, T // 2))
            if L >= 1 and T >= 8:
                pv, cnt = [], 0
                for j in range(m):
                    x = vmat[j]
                    xc = x - x.mean()
                    den = float(np.dot(xc, xc))
                    if den <= 0:
                        continue
                    cnt += 1
                    rks = [float(np.dot(xc[:-k], xc[k:]) / den) for k in range(1, L + 1)]
                    Q = T * (T + 2) * sum(rk ** 2 / (T - k) for k, rk in zip(range(1, L + 1), rks))
                    pv.append(float(stats.chi2.sf(Q, L)))
                if cnt:
                    info["lb_reject_prop"] = float(np.mean([p < 0.05 for p in pv]))
                    info["lb_p_median"] = float(np.median(pv))
                    info["n_areas_tested"] = int(cnt)
        out["blocks"][cell_key] = info

    if "interaction" in blocks:
        idx = np.asarray(blocks["interaction"], int)
        out["blocks"]["interaction"] = _norm(b[idx])

    # Shrinkage: corr(area BLUP, raw per-area logit signal)
    if "area" in blocks:
        y = np.asarray(o["y"], float)
        X = np.asarray(o["X"], float)
        beta = np.asarray(o["beta_hat"], float)
        area_idx = np.asarray(des.get("area_idx"), int)
        ylogit = np.log(np.clip(y, 1e-6, 1 - 1e-6) /
                        (1 - np.clip(y, 1e-6, 1 - 1e-6)))
        raw = ylogit - X @ beta
        m = len(blocks["area"])
        cnt = np.bincount(area_idx, minlength=m).astype(float)
        s = np.bincount(area_idx, weights=raw, minlength=m) / np.maximum(cnt, 1)
        vb = b[np.asarray(blocks["area"], int)]
        ok = np.isfinite(s) & (cnt > 0)
        out["shrinkage_corr_area"] = float(stats.spearmanr(vb[ok], s[ok]).statistic)

    # Wald log-scale CIs for variance components
    inf = getattr(fit, "inference", None) or {}
    est = np.asarray(inf.get("estimate", []), float)
    se = np.asarray(inf.get("se", []), float)
    labels = list(inf.get("param_labels", []))
    cis = {}
    z = 1.959963985
    for i, lab in enumerate(labels):
        if lab.startswith("log_sigma") and i < len(est):
            lo, hi = est[i] - z * se[i], est[i] + z * se[i]
            cis[lab] = {"sigma": float(np.exp(est[i])),
                        "ci": [float(np.exp(lo)), float(np.exp(hi))]}
    out["variance_cis"] = cis
    return out


def vif_fixed_effects(X: np.ndarray, x_cols: list[str]) -> dict:
    """P11: variance inflation factors for the (standardized) design columns.

    Column 0 is the intercept (skipped). VIF_j = 1/(1-R_j^2) where R_j^2 is
    from regressing column j on all other columns (intercept included).
    Rule of thumb: VIF > 5 (R^2 > 0.8) warrants attention, > 10 serious.
    """
    X = np.asarray(X, float)
    p = X.shape[1]
    out = {}
    if p < 3:
        return out
    Zfull = np.column_stack([np.ones(len(X)), X[:, 1:]])
    for j in range(1, p):
        yj = X[:, j]
        others = np.delete(Zfull, j, axis=1)
        beta_j, *_ = np.linalg.lstsq(others, yj, rcond=None)
        resid = yj - others @ beta_j
        tss = np.sum((yj - yj.mean()) ** 2)
        r2 = 1.0 - np.sum(resid**2) / tss if tss > 0 else 0.0
        vif = 1.0 / max(1.0 - r2, 1e-12)
        out[x_cols[j - 1] if j - 1 < len(x_cols) else f"X{j}"] = {
            "vif": round(float(vif), 2), "r2": round(float(r2), 3),
            "flag": "HIGH" if vif > 10 else ("watch" if vif > 5 else "ok"),
        }
    return out


def nested_lrt(loglik_full: float, loglik_null: float) -> dict:
    """Interior (non-boundary) nested LRT with the plain chi2_df reference.

    Use ONLY when the tested parameter is NOT at a boundary under H0
    (e.g. dropping a whole AR(1)/SAR structure whose params are interior).
    For variance components tested at zero use stmmlib.lrt.boundary_lrt.
    """
    D = 2.0 * (float(loglik_full) - float(loglik_null))
    df = None  # caller supplies df via nested_lrt_with_df
    return {"D": D, "D_clamped": max(D, 0.0)}


def nested_lrt_with_df(loglik_full: float, loglik_null: float, df: int) -> dict:
    """Interior nested LRT with explicit df (chi2_df reference)."""
    from scipy.stats import chi2 as _chi2
    D = max(2.0 * (float(loglik_full) - float(loglik_null)), 0.0)
    return {"D": D, "df": int(df),
            "p_value": float(_chi2.sf(D, df)),
            "reference": f"chi2_{df} (interior H0)",
            "verdict": ("signifikan" if _chi2.sf(D, df) < 0.05
                        else "tidak signifikan")}


def wald_table_beta(fit) -> "list[dict]":
    """P11: tidy Wald table (estimate, SE, z, p) from the inference block.

    Estimates are on the FITTED scale (standardized predictors for beta);
    use fit.coef() for original-scale coefficients.
    """
    inf = getattr(fit, "inference", None) or {}
    est = np.asarray(inf.get("estimate", []), float)
    se = np.asarray(inf.get("se", []), float)
    pv = np.asarray(inf.get("pvalue", []), float)
    labels = list(inf.get("param_labels", []))
    rows = []
    for i, lab in enumerate(labels):
        if i >= len(est):
            break
        rows.append({"param": lab, "estimate": float(est[i]),
                     "se": float(se[i]) if i < len(se) else np.nan,
                     "z": float(est[i] / se[i]) if i < len(se) and se[i] > 0 else np.nan,
                     "p_value": float(pv[i]) if i < len(pv) else np.nan})
    return rows


def validate_assumptions(data: "pd.DataFrame", area_col: str = "area",
                         time_col: str = "time", W=None,
                         fit=None) -> dict:
    """P12: structural assumption checks, one entry point.

    Checks (independent of any fit):
      - balanced panel: every area observed at every time, 1 obs per cell
      - index contiguity (0..m-1 / 0..T-1 after internal recoding)
      - W validity: square, non-negative, zero diagonal, row sums
        (row-standardized or not), connectivity (spectral radius of I-W < 1
        requires W irreducible; here we check W is fully connected via
        symmetric reachability), and SAR stability bound max|eig| < 1/0.95

    With ``fit`` (BetastmmResult) additionally:
      - boundary mu: count of fitted mu within eps of 0/1
      - at_bound / degenerate flags from the convergence certificate
    """
    d = data
    a = np.asarray(d[area_col].values)
    t = np.asarray(d[time_col].values)
    areas = np.unique(a)
    times = np.unique(t)
    n, m, T = len(d), len(areas), len(times)
    ai = pd.Categorical(a, categories=areas).codes
    ti = pd.Categorical(t, categories=times).codes
    cells = ai * T + ti
    n_cells = len(np.unique(cells))

    out = {
        "panel": {
            "n": int(n), "m": int(m), "T": int(T),
            "n_cells": int(n_cells), "balanced": bool(n_cells == m * T),
            "obs_per_cell": round(n / max(n_cells, 1), 2),
            "balanced_note": ("1 obs/sel" if n_cells == m * T == n else
                              ("replikasi" if n_cells < n else "UNBALANCED (sel kosong)")),
        },
    }

    if W is not None:
        Wm = W.toarray() if hasattr(W, "toarray") else np.asarray(W, float)
        sym = Wm + Wm.T
        # connectivity via BFS on the symmetric graph
        seen = np.zeros(m, dtype=bool)
        stack = [0]
        seen[0] = True
        while stack:
            i = stack.pop()
            for j in np.where(sym[i] > 0)[0]:
                if not seen[j]:
                    seen[j] = True
                    stack.append(int(j))
        eigs = np.abs(np.linalg.eigvalsh(Wm))
        out["W"] = {
            "shape_ok": bool(Wm.shape == (m, m)),
            "nonneg": bool((Wm >= -1e-12).all()),
            "zero_diag": bool(np.allclose(np.diag(Wm), 0)),
            "row_sum_min": float(Wm.sum(1).min()),
            "row_sum_max": float(Wm.sum(1).max()),
            "connected": bool(seen.all()),
            "max_abs_eig": float(eigs.max()),
            "sar_stable_0.95": bool(eigs.max() < 1 / 0.95),
        }

    if fit is not None:
        o = fit._opt
        mu = np.asarray(fit.mu_hat, float)
        out["fit"] = {
            "mu_near_0": int(np.sum(mu < 1e-6)),
            "mu_near_1": int(np.sum(mu > 1 - 1e-6)),
            "at_bound": bool(o.get("at_bound", False)),
            "degenerate": bool(o.get("degenerate", False)),
            "status": o.get("status"),
        }
    return out


def prediction_intervals_gaussian(result: dict, level: float = 0.95,
                                  include_obs: bool = True) -> dict:
    """P5: prediction intervals for a gaussian stmm_normal_core result.

    y_new | data ~ N(X beta_hat, Z G Z' + sigma2_e I) approximated with
    beta uncertainty added on the mean: Var(pred) = diag(X Vbb X') +
    diag(V) - sigma2_e (+ sigma2_e when include_obs), where V = ZGZ' +
    sigma2_e I is the fitted marginal covariance carried in the result.
    """
    from scipy.stats import norm
    X = np.asarray(result["X"], float)
    V = np.asarray(result["V"], float)
    fitted = np.asarray(result["fitted"], float).ravel()
    n = X.shape[0]
    Vb = np.asarray(result["vcov_beta"], float)
    theta = result.get("theta", {}) or {}
    s2e = float(theta.get("sigma2_e", 0.0))
    se_pred = np.sqrt(np.maximum(
        np.einsum("ij,jk,ik->i", X, Vb, X) + np.diag(V) - s2e
        + (s2e if include_obs else 0.0), 0.0))
    z = float(norm.ppf(0.5 + level / 2))
    return {
        "mean": fitted,
        "lower": fitted - z * se_pred,
        "upper": fitted + z * se_pred,
        "se_pred": se_pred,
        "width_mean": float(np.mean(2 * z * se_pred)),
        "include_obs": bool(include_obs), "level": level,
    }


def variance_ci_beta(fit, method: str = "wald") -> dict:
    """CI utk komponen varian (log_sigma_*) dari fit beta.

    method="wald"    : est +/- 1.96 SE pada skala log (dari inference).
    method="profile" : scan likelihood 1-D per param (inference method="profile",
                       opt-in lambat); CI = est +/- 1.96 SE_profile.
    Mengembalikan dict label -> {sigma, lo, hi, method}.
    """
    from scipy.stats import norm as _norm
    inf = getattr(fit, "inference", None) or {}
    o = fit._opt
    est = np.asarray(inf.get("estimate", []), float)
    labels = list(inf.get("param_labels", []))
    z = float(_norm.ppf(0.975))
    out = {}
    for i, lab in enumerate(labels):
        if not lab.startswith("log_sigma") or i >= len(est):
            continue
        if method == "profile":
            # recompute inference with profile SEs (cached attr to avoid repeat)
            key = "_inference_profile"
            ip = getattr(fit, key, None)
            if ip is None:
                from stmmlib.beta.inference import compute_stbl_inference
                ip = compute_stbl_inference(o, method="profile")
                try:
                    object.__setattr__(fit, key, ip)
                except Exception:
                    fit.__dict__[key] = ip
            se_map = dict(zip(ip.get("param_labels", []), np.asarray(ip.get("se", []), float)))
            se_i = float(se_map.get(lab, np.nan))
        else:
            se_map = dict(zip(labels, np.asarray(inf.get("se", []), float)))
            se_i = float(se_map.get(lab, np.nan))
        sigma = float(np.exp(est[i]))
        out[lab] = {
            "sigma": sigma,
            "lo": float(np.exp(est[i] - z * se_i)) if np.isfinite(se_i) else np.nan,
            "hi": float(np.exp(est[i] + z * se_i)) if np.isfinite(se_i) else np.nan,
            "method": method,
        }
    return out


def bootstrap_variance_ci_beta(fit, n_boot: int = 50, level: float = 0.95,
                               seed: int = 42, max_iter: int = 800,
                               inner_max_iter: int = 20) -> dict:
    """Parametric-bootstrap CI utk komponen varian beta.

    Menggambar y ~ Beta(mu_hat, phi_hat) kondisional pada b_hat, refit model
    sama (n_starts=1), lalu CI = kuantil sigma lintas bootstrap. Lebih jujur
    dari Wald di panel kecil (lihat P4) tapi mahal (n_boot refit).
    """
    from scipy.stats import norm as _norm
    o = fit._opt
    y = np.asarray(o["y"], float)
    mu = np.asarray(fit.mu_hat, float)
    phi = float(np.exp(o["logphi"]))
    rng = np.random.default_rng(seed)
    labels = list((getattr(fit, "inference", None) or {}).get("param_labels", []))
    sig_idx = {lab: i for i, lab in enumerate(labels) if lab.startswith("log_sigma")}
    if not sig_idx:
        return {}
    draws = {lab: [] for lab in sig_idx}
    n_ok = 0
    data = o.get("des", {}).get("_data_ref")  # not stored; refit via stmm below
    for b in range(n_boot):
        yb = rng.beta(mu * phi, (1 - mu) * phi)
        yb = np.clip(yb, 1e-6, 1 - 1e-6)
        try:
            # refit via the low-level structural solver on identical design
            from stmmlib.beta.solvers import beta_laplace_structural
            d = o["_boot_frame"] if "_boot_frame" in o else None
            if d is None:
                return {}
            r = beta_laplace_structural(None, d, yb, "Laplace", o["model"],
                                        W=o.get("W"), n_starts=1,
                                        max_iter=max_iter,
                                        inner_max_iter=inner_max_iter,
                                        start_theta=o.get("_boot_theta"),
                                        backend=None)
            th = r["logs"]
            for lab, i in sig_idx.items():
                if i - (len(o["beta_hat"]) + 1) < len(th):
                    draws[lab].append(float(np.exp(th[i - (len(o["beta_hat"]) + 1)])))
            n_ok += 1
        except Exception:
            continue
    a = (1 - level) / 2
    out = {}
    for lab, vals in draws.items():
        if len(vals) < max(5, n_boot // 4):
            out[lab] = {"sigma": np.nan, "lo": np.nan, "hi": np.nan,
                        "n_boot_ok": n_ok, "method": "parametric-bootstrap"}
            continue
        v = np.asarray(vals, float)
        out[lab] = {
            "sigma": float(np.median(v)),
            "lo": float(np.quantile(v, a)),
            "hi": float(np.quantile(v, 1 - a)),
            "n_boot_ok": n_ok, "method": "parametric-bootstrap",
        }
    return out


def format_random_effects_report(d: dict) -> str:
    L = ["Random-effects diagnostics"]
    for blk, info in d.get("blocks", {}).items():
        L.append(f"  {blk:<12}: n={info['n']} skew={info['skew']:+.2f} "
                 f"exc.kurt={info['excess_kurtosis']:+.2f} "
                 f"qq={info['qq_corr']:+.3f} "
                 f"Shapiro p={info['shapiro_p']:.3g}")
        if "moran_I" in info:
            L.append(f"                 Moran I={info['moran_I']:.3f} "
                     f"(E[I]={info['moran_EI']:.3f})")
        if "acf1" in info:
            L.append(f"                 ACF1 (year-avg)={info['acf1']:+.3f}")
        if "lb_reject_prop" in info:
            L.append(f"                 Ljung-Box: {info['n_areas_tested']} area, "
                     f"{info['lb_reject_prop']*100:.1f}% tolak(0.05), p median={info['lb_p_median']:.3f}")
    if "shrinkage_corr_area" in d:
        L.append(f"  shrinkage corr(area BLUP, raw signal) = "
                 f"{d['shrinkage_corr_area']:+.3f}")
    for lab, ci in d.get("variance_cis", {}).items():
        L.append(f"  {lab:<16} sigma={ci['sigma']:.4f} "
                 f"95% CI [{ci['ci'][0]:.4f}, {ci['ci'][1]:.4f}]")
    return "\n".join(L)


def format_residual_report(d: dict) -> str:
    """Compact text report for thesis notebooks/logs."""
    L = []
    L.append(f"Residual diagnostics ({d['family']})")
    n = d["normality"]
    L.append(f"  normality : n={n['n']} skew={n['skew']:+.3f} "
             f"exc.kurt={n['excess_kurtosis']:+.3f} "
             f"Shapiro p={n['shapiro_p']:.4g} KS p={n['ks_p']:.4g} "
             f"AD*={n.get('ad_stat_scaled', float('nan')):.3f}")
    h = d["heteroskedasticity"]
    L.append(f"  heterosk. : Spearman(|r|,fit)={h['spearman_abs_r_fitted']:+.3f} "
             f"BP slope t={h['bp_t']:+.2f} p={h['bp_p']:.4g}")
    if "envelope" in d:
        e = d["envelope"]
        L.append(f"  envelope  : {e['n_outside']}/{e['k'] * 0 + d['normality']['n']} obs "
                 f"({e['pct_outside']:.1f}%) outside 95% band; max|r|={e['max_abs_r']:.2f}")
    if "influence" in d and "cooks_d_max" in d.get("influence", {}):
        inf = d["influence"]
        L.append(f"  influence : Cook max={inf['cooks_d_max']:.3f} "
                 f"lev max={inf['leverage_max']:.3f} "
                 f"top5 idx={inf['top5_idx']}")
    if d.get("vs_predictors_spearman"):
        worst = sorted(d["vs_predictors_spearman"].items(),
                       key=lambda kv: -abs(kv[1]))[:3]
        L.append("  vs pred.  : " + ", ".join(f"{c}:{v:+.2f}" for c, v in worst))
    return "\n".join(L)
