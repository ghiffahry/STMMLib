"""Evaluation + diagnostics reporting for Beta STMM fits.

Provides `stmm_beta_evaluation` and `stmm_beta_diagnostics` operating on a
`BetastmmResult`, returning a dict and a printable report. Designed to be
called from the notebook in place of bare `print(fit)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.special import betainc, betaln
from scipy.stats import norm, chi2


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def _beta_log_score(y, mu, phi):
    """Log score: -1/n * sum log f_Beta(y | mu, phi)."""
    y = np.clip(np.asarray(y, dtype=float), 1e-15, 1 - 1e-15)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-15, 1 - 1e-15)
    a = mu * phi
    b = (1 - mu) * phi
    lf = -betaln(a, b) + (a - 1) * np.log(y) + (b - 1) * np.log(1 - y)
    lf = np.where(np.isfinite(lf), lf, np.nan)
    return float(-np.nanmean(lf)) if np.isfinite(lf).any() else np.nan


def _rmse_mae(y, mu):
    y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
    ok = np.isfinite(y) & np.isfinite(mu)
    if not ok.any():
        return np.nan, np.nan
    err = y[ok] - mu[ok]
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def _r2(criteria_m, criteria_c):
    return criteria_m, criteria_c


def _beta_criteria(fit, use_W=None):
    """cAIC + cR2 via the single library implementation (criteria.py).

    BUG-B2 fix: this was a third copy of the notebook helper still carrying
    the betaln sign bug (ll_cond = betaln + ... instead of - betaln) and the
    stale-mu path. Delegate; keep only the criterion keys reports use.
    Nested models return NaN fields (documented criteria.py scope).
    """
    from stmmlib.beta.criteria import beta_criteria
    cr = beta_criteria(fit, W=use_W)
    if cr is None:
        return None
    return {k: cr.get(k) for k in
            ("cAIC", "df_hat", "ll_cond", "cR2_marginal", "cR2_conditional")}


def evaluate(fit, criteria=None, holdout=None, use_W=None) -> dict:
    """Compute full evaluation table for one fit.

    Args:
        fit: BetastmmResult.
        criteria: optional dict from the notebook's beta_criteria (reused).
        holdout: optional dict {"y_true": arr, "pred": arr, "label": str}.
        use_W: spatial weight matrix (for R2 random-variance term).

    Returns dict of scalars.
    """
    opt = fit._opt
    y = np.asarray(opt.get("y"), dtype=float)
    mu = np.asarray(opt.get("mu_hat", np.full(len(y), np.nan)), dtype=float)
    phi = float(fit.phi)

    cr = criteria or _beta_criteria(fit, use_W=use_W) or {}
    train_rmse, train_mae = _rmse_mae(y, mu)
    logsc_tr = _beta_log_score(y, mu, phi)

    out = {
        "n_train": int(len(y)),
        "logLik": float(fit.log_lik),
        "AIC": float(fit.AIC),
        "BIC": float(fit.BIC),
        "cAIC": cr.get("cAIC", np.nan),
        "df_hat": cr.get("df_hat", np.nan),
        "R2_marginal": cr.get("cR2_marginal", np.nan),
        "R2_conditional": cr.get("cR2_conditional", np.nan),
        "train_RMSE": train_rmse, "train_MAE": train_mae,
        "train_log_score": logsc_tr,
        "calib_obs_mean": float(np.nanmean(y)), "calib_pred_mean": float(np.nanmean(mu)),
        "converged": bool(opt.get("converged", False)),
        "degenerate": bool(opt.get("degenerate", False)),
    }
    if holdout is not None:
        hy = np.asarray(holdout["y_true"], dtype=float)
        hp = np.asarray(holdout["pred"], dtype=float)
        h_rmse, h_mae = _rmse_mae(hy, hp)
        out.update({
            "holdout_label": holdout.get("label", "Pred"),
            "holdout_n": int(len(hy)),
            "holdout_RMSE": h_rmse, "holdout_MAE": h_mae,
            "holdout_log_score": _beta_log_score(hy, hp, phi),
        })
    return out


def _fmt_report(fit, ev) -> str:
    L = []
    bar = "-" * 60
    L.append("Model evaluation (%s)" % fit.model)
    L.append(bar)
    r2m = ev.get("R2_marginal"); r2c = ev.get("R2_conditional")
    L.append(f"  logLik (Laplace)   : {ev['logLik']:.3f}")
    L.append(f"  AIC / BIC          : {ev['AIC']:.3f} / {ev['BIC']:.3f}")
    L.append(f"  cAIC (cond)        : {ev['cAIC']:.3f}   (df_hat={ev['df_hat']:.2f})")
    L.append(f"  R2 marginal        : {r2m:.4f}" if r2m is not None else "  R2 marginal        : nan")
    L.append(f"  R2 conditional     : {r2c:.4f}" if r2c is not None else "  R2 conditional     : nan")
    L.append(bar)
    L.append(f"  Train (n={ev['n_train']})  RMSE={ev['train_RMSE']:.4f}  MAE={ev['train_MAE']:.4f}  log-score={ev['train_log_score']:.4f}")
    L.append(f"  Calibration: obs-mean={ev['calib_obs_mean']:.4f}  pred-mean={ev['calib_pred_mean']:.4f}")
    if "holdout_n" in ev:
        L.append(f"  {ev.get('holdout_label','Holdout')} (n={ev['holdout_n']})  RMSE={ev['holdout_RMSE']:.4f}  MAE={ev['holdout_MAE']:.4f}  log-score={ev['holdout_log_score']:.4f}")
    if ev.get("degenerate"):
        L.append("  >> MODEL DEGENERATE (sigma-collapse) - do not use.")
    L.append(bar)
    return "\n".join(L)


def stmm_beta_evaluation(fit, criteria=None, holdout=None, use_W=None, as_str: bool = True):
    """Evaluation table dict (and optional printable string)."""
    ev = evaluate(fit, criteria=criteria, holdout=holdout, use_W=use_W)
    if as_str:
        return ev, _fmt_report(fit, ev)
    return ev


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _dunn_smyth(y, mu, phi):
    """Randomized quantile (Dunn-Smyth) residual. For the continuous Beta model
    this is simply Phi^-1(F(y)) with F the Beta CDF -> ~N(0,1) under H0."""
    y = np.clip(np.asarray(y, dtype=float), 1e-15, 1 - 1e-15)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-15, 1 - 1e-15)
    a = mu * phi
    b = (1 - mu) * phi
    u = betainc(a, b, y)
    u = np.clip(u, 1e-15, 1 - 1e-15)
    return norm.ppf(u)


def _moran(resid_area, Wt, n_perm=999, sisi="dua", center=None):
    """Moran I dgn p permutasi dua sisi terhadap pusat distribusi permutasi.
    Pusat empiris c = mean(Ip) bila center=None."""
    z = resid_area - np.mean(resid_area)
    S0 = np.sum(Wt)
    if S0 == 0 or len(z) < 3:
        return np.nan, np.nan
    lag = Wt @ z
    I = (len(z) / S0) * (z @ lag) / (z @ z) if (z @ z) > 0 else 0.0
    rng = np.random.default_rng(42)
    Ips = []
    for _ in range(n_perm):
        zp = rng.permutation(z)
        Ip = (len(z) / S0) * (zp @ (Wt @ zp)) / (zp @ zp) if (zp @ zp) > 0 else 0.0
        Ips.append(Ip)
    Ips = np.asarray(Ips, dtype=float)
    Ips = Ips[np.isfinite(Ips)]
    if Ips.size == 0:
        return float(I), float("nan")
    if sisi == "besar":
        cnt = int(np.sum(Ips >= I))
    elif sisi == "kecil":
        cnt = int(np.sum(Ips <= I))
    else:
        c = float(np.mean(Ips)) if center is None else float(center)
        cnt = int(np.sum(np.abs(Ips - c) >= abs(float(I) - c)))
    return float(I), float((cnt + 1) / (Ips.size + 1))


def _ljung_box(resid, nlags=8):
    n = len(resid)
    if n < nlags + 2:
        return np.nan, np.nan
    x = resid - np.mean(resid)
    denom = np.sum(x ** 2)
    if denom == 0:
        return np.nan, np.nan
    acf = np.array([np.sum(x[:n - k] * x[k:]) / denom for k in range(1, nlags + 1)])
    stat = n * (n + 2) * np.sum(acf ** 2 / np.arange(1, nlags + 1))
    return float(stat), float(1 - chi2.cdf(stat, nlags))


def stmm_beta_diagnostics(fit, W=None, as_str: bool = True):
    """Dunn-Smyth, Moran (per-time, summarized), Ljung-Box (summarized),
    optimizer status."""
    opt = fit._opt
    y = np.asarray(opt.get("y"), dtype=float)
    mu = opt.get("mu_hat")
    if mu is None:
        mu = 1 / (1 + np.exp(-(opt["X"] @ fit.beta)))
    mu = np.clip(np.asarray(mu, dtype=float), 1e-15, 1 - 1e-15)
    phi = float(fit.phi)
    des = opt.get("des", {})

    ds = _dunn_smyth(y, mu, phi)
    ds_mean = float(np.nanmean(ds)); ds_sd = float(np.nanstd(ds))

    # Moran per time
    moran_res = {}
    if W is not None:
        na = des.get("m", 1)
        Tt = des.get("T", 1)
        resid = y - mu
        time_idx = des.get("time_idx", np.arange(len(y)))
        if len(np.unique(time_idx)) > 1:
            for t in np.unique(time_idx):
                m = time_idx == t
                if m.sum() == na:
                    I, p = _moran(resid[m], W)
                    moran_res[int(t)] = (I, p)
    moran_summary = None
    if moran_res:
        ps = [p for _, p in moran_res.values()]
        moran_summary = {
            "n_times": len(moran_res),
            "median_p": float(np.median(ps)),
            "min_p": float(np.min(ps)),
            "n_reject": int(sum(p < 0.05 for p in ps)),
        }

    # Ljung-Box summarized over areas
    area_idx = des.get("area_idx", np.arange(len(y)))
    resid = y - mu
    from collections import defaultdict
    by_area = defaultdict(list)
    for r, a in zip(resid, area_idx):
        by_area[a].append(r)
    lb_ps = []
    for rv in by_area.values():
        if len(rv) > 10:
            _, p = _ljung_box(np.array(rv))
            lb_ps.append(p)
    lb_summary = None
    if lb_ps:
        lb_summary = {
            "n_areas": len(lb_ps),
            "median_p": float(np.median(lb_ps)),
            "min_p": float(np.min(lb_ps)),
            "n_reject": int(sum(p < 0.05 for p in lb_ps)),
        }

    om = opt.get("optim", {})
    diag = {
        "DunnSmyth_mean": ds_mean, "DunnSmyth_sd": ds_sd,
        "DunnSmyth_pass": bool(abs(ds_mean) < 0.1 and abs(ds_sd - 1) < 0.2),
        "Moran": moran_summary,
        "LjungBox": lb_summary,
        "optimizer": {"success": om.get("success"), "nit": om.get("nit"),
                      "iters": om.get("iters", om.get("nit", -1)),
                      "max_iter": om.get("max_iter"),
                      "start": om.get("start"), "message": om.get("message"),
                      "status": opt.get("status", "konvergen" if opt.get("converged") else "invalid")},
    }
    if not as_str:
        return diag

    L = []
    bar = "-" * 60
    L.append("Diagnostics (%s)" % fit.model)
    L.append(bar)
    L.append(f"  Dunn-Smyth : mean={ds_mean:.4f}  sd={ds_sd:.4f}   {'PASS' if diag['DunnSmyth_pass'] else 'FAIL'}  (target N(0,1))")
    if moran_summary:
        ms = moran_summary
        L.append(f"  Moran (per-period, {ms['n_times']} periods): median p={ms['median_p']:.4f}  min p={ms['min_p']:.4f}  reject={ms['n_reject']}/{ms['n_times']}")
    else:
        L.append("  Moran : W not available / insufficient data.")
    if lb_summary:
        lb = lb_summary
        L.append(f"  Ljung-Box (per-area, {lb['n_areas']} areas): median p={lb['median_p']:.4f}  min p={lb['min_p']:.4f}  reject={lb['n_reject']}/{lb['n_areas']}")
    else:
        L.append("  Ljung-Box : insufficient per-area data.")
    st = opt.get("status", "konvergen" if opt.get("converged") else "invalid")
    iters = om.get("iters", om.get("nit", "?"))
    max_iter = om.get("max_iter", "?")
    _gn = om.get("grad_norm")
    _gn_txt = f"  grad={float(_gn):.1e}" if isinstance(_gn, (int, float)) and np.isfinite(_gn) else ""
    _msg = om.get("message", "")
    _msg_txt = f" ({_msg})" if _msg else ""
    L.append(f"  Optimizer  : status={st}  success={om.get('success')}  it={iters}/{max_iter}  start={om.get('start')}{_gn_txt}{_msg_txt}")
    L.append(bar)
    return diag, "\n".join(L)


def convergence_certificate(fit, as_str: bool = True):
    """Single-source-of-truth convergence certificate for a Beta fit.

    Mirrors the lme4/glmmTMB diagnostic philosophy: reported convergence is a
    *verified* optimum, judged by gradient + curvature + boundary + degeneracy
    + optimizer termination -- NOT by iteration count. maxfun is a ceiling, so
    the certificate reports actual function evaluations (n_evals), not the
    budget. `converged` here is exactly `status == "konvergen"`.
    """
    opt = fit._opt
    om = opt.get("optim", {})
    inf = getattr(fit, "inference", None) or {}

    st = opt.get("status", "konvergen" if opt.get("converged") else "invalid")

    import numpy as np
    gn = om.get("grad_norm")
    grad_norm = float(gn) if isinstance(gn, (int, float)) and np.isfinite(gn) else float("nan")

    n_evals = om.get("n_evals", -1)
    objective = om.get("fun")  # negative marginal log-lik at optimum
    hessian_pd = bool(inf.get("hessian_pd"))
    lambda_min = inf.get("lambda_min")
    lam_txt = f"{float(lambda_min):.2e}" if isinstance(lambda_min, (int, float)) and np.isfinite(lambda_min) else "NA"

    cert = {
        "model": opt.get("model"),
        "optimizer": "L-BFGS-B",
        "n_starts": om.get("n_starts", 1),
        "best_start": om.get("start", -1),
        "outer_iterations": om.get("iters", om.get("nit", -1)),
        "function_evals_actual": n_evals,
        "maxfun_ceiling": max(200_000, int(om.get("max_iter", 0)) * 8),
        "max_iter": om.get("max_iter"),
        "objective_negloglik": objective if np.isfinite(objective) else float("nan"),
        "logLik": opt.get("log_lik"),
        "grad_norm": grad_norm,
        "success": bool(om.get("success", False)),
        "hessian_pd": hessian_pd,
        "lambda_min": lambda_min,
        "at_bound": bool(opt.get("at_bound", False)),
        "degenerate": bool(opt.get("degenerate", False)),
        "status": st,
        "converged": st == "konvergen",
        "message": om.get("message", ""),
    }

    if not as_str:
        return cert

    L = []
    bar = "-" * 60
    L.append(f"Convergence Certificate -- {cert['model']}")
    L.append(bar)
    L.append(f"  Optimizer       : {cert['optimizer']}   starts={cert['n_starts']} (best start {cert['best_start']})")
    L.append(f"  Outer iters     : {cert['outer_iterations']}   |   max_iter ceiling={cert['max_iter']}")
    L.append(f"  Function evals  : {cert['function_evals_actual']} actual  (ceil {cert['maxfun_ceiling']})")
    L.append(f"  Objective(-ll)  : {cert['objective_negloglik']:.6f}   |   logLik={cert['logLik']:.4f}")
    L.append(f"  Gradient norm   : {cert['grad_norm']:.3e}   |   success={cert['success']}")
    L.append(f"  Hessian PD      : {'Yes' if cert['hessian_pd'] else 'No'}   lambda_min={lam_txt}")
    L.append(f"  Boundary hit    : {'Yes' if cert['at_bound'] else 'No'}   |   Degenerate: {'Yes' if cert['degenerate'] else 'No'}")
    L.append(f"  Status          : {st}   |   converged={cert['converged']}")
    if cert["message"]:
        L.append(f"  Optimizer msg   : {cert['message']}")
    L.append(bar)
    return cert, "\n".join(L)
