"""Gaussian/normal-family output formatter (statsmodels summary style, English).

Formats a fitted normal (gaussian) result dict into a readable
regression-style report: header, fixed-effects table, variance components,
shared model-fit statistics, and convergence note.
"""

from __future__ import annotations

import numpy as np

from stmmlib.output_format import _fmt, _sig, _footer, fmt_model_fit


def _t_pvalues(t_vals, df):
    """Two-sided p-values for a vector of t-values."""
    t_vals = np.asarray(t_vals, dtype=float)
    df = float(df)
    if np.isinf(df) or df > 1e7:
        from scipy.stats import norm
        return 2 * norm.sf(np.abs(t_vals))
    from scipy.stats import t as t_dist
    return 2 * t_dist.sf(np.abs(t_vals), df=df)


def format_normal(obj: dict, width: int = 72) -> str:
    """Format a normal (gaussian) fit dict as an English statsmodels report."""
    n = float(obj.get("n", np.nan))
    p = int(obj.get("p", 0))
    beta_hat = np.asarray(obj.get("beta_hat", np.nan)).ravel()
    se_beta = np.asarray(obj.get("se_beta", np.nan)).ravel()

    coef_names = list(obj.get("coef_names", []) or [])
    if not coef_names:
        coef_names = [f"beta_{i}" for i in range(len(beta_hat))]

    ln = []
    bar = "-" * width
    ln.append(bar)
    ln.append("Spatio-Temporal Linear Mixed Model".center(width))
    ln.append(bar)
    ln.append(f"Model  : {obj.get('model_name', '')}")
    ln.append(f"Family : normal, Link: identity")
    ln.append(f"Method : {obj.get('method', 'REML')}")
    ln.append(f"Data   : n = {_fmt(n, 0)}, groups = {_fmt(obj.get('m'), 0)}, "
              f"periods = {_fmt(obj.get('T'), 0)}")
    ln.append("")

    df = n - p
    if len(beta_hat) and len(se_beta):
        t_vals = beta_hat / np.where(se_beta > 0, se_beta, 1.0)
        p_vals = _t_pvalues(t_vals, df)
        ln.append("Fixed Effects:")
        ln.append(f"  {'':<2}{'':<24}{'coef':>11}{'std err':>11}{'t':>9}"
                  f"{'P>|t|':>11}{'':>6}")
        for i in range(len(beta_hat)):
            if i >= len(coef_names):
                break
            star = _sig(p_vals[i])
            ln.append(
                f"  {str(coef_names[i]):<26}{_fmt(beta_hat[i]):>11}"
                f"{_fmt(se_beta[i]):>11}{_fmt(t_vals[i], 3):>9}"
                f"{_fmt(p_vals[i]):>11} {star:>2}"
            )
        ln.append(f"  (df = {_fmt(df, 0)})")
        ln.extend(_footer(width))
    else:
        ln.append("(Fixed-effect estimates not computed for this model)")

    # Variance components
    theta = obj.get("theta", {}) or {}
    vd = obj.get("variance_decomp", {}) or {}
    ln.append("")
    ln.append("Variance components:")
    if isinstance(theta, dict) and theta:
        for k, v in theta.items():
            ln.append(f"  {k:<26}: {_fmt(np.real(v))}")
    elif theta is not None:
        names = ["theta" + str(i) for i in range(len(theta))]
        for nm, v in zip(names, np.asarray(theta).ravel()):
            ln.append(f"  {nm:<26}: {_fmt(np.real(v))}")
    for key in ("sigma_sq", "sigma2", "tau2", "R2"):
        if key in vd and vd[key] is not None:
            ln.append(f"  {key:<26}: {_fmt(vd[key])}")

    r2 = obj.get("R2")
    if isinstance(r2, dict):
        for k, v in r2.items():
            if isinstance(v, (int, float)):
                ln.append(f"  R2 ({k}){'':<18}: {_fmt(v)}")

    ln.append("")
    ln.extend(fmt_model_fit(
        {"logLik": obj.get("log_lik"), "AIC": obj.get("AIC"),
         "BIC": obj.get("BIC"), "k": obj.get("k"), "n": n}, width=width))

    conv = obj.get("converged")
    ln.append("")
    ln.append(f"Optimization: {'Converged' if conv else 'Not converged'}"
              if conv is not None else "Optimization: n/a")
    ln.append(bar)
    return "\n".join(ln)
