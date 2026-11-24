"""Response-scale marginal effects for Beta fixed effects.

The Beta model uses a logit link: ``mu = expit(X beta)``. A coefficient on
the logit scale is not a response-scale effect. This module reports the
average marginal effect (AME) over the fitted rows or the marginal effect at
the mean covariate profile (MEM), with delta-method standard errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

_COLUMNS = ["term", "estimate", "se", "z", "p", "ci_low", "ci_high"]


def _fixed_vcov(fit, p: int):
    """The fixed-effect block of the fit's vcov, or None when unusable."""
    inf = getattr(fit, "_inf", None)
    if not isinstance(inf, dict):
        return None
    V = inf.get("vcov")
    if V is None:
        return None
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] < p or V.shape[1] < p:
        return None
    V = V[:p, :p]
    if not np.all(np.isfinite(V)):
        return None
    return V


def marginal_effects(fit, at: str = "average", scale: float = 1.0,
                     level: float = 0.95) -> pd.DataFrame:
    """Marginal effects of the fixed effects on the response scale.

    Args:
        fit: a fitted Beta model with a design matrix and fixed effects
            (``BetastmmResult``).
        at: ``"average"`` for the average marginal effect over the fitted
            rows, ``"mean"`` for the effect at the mean covariate profile.
        scale: multiplier for the reported effects. ``100`` reports
            percentage points. z and p are invariant to this scale.
        level: confidence level for the interval.

    Returns:
        DataFrame with ``term``, ``estimate``, ``se``, ``z``, ``p``,
        ``ci_low``, and ``ci_high``. The intercept is excluded. ``se``,
        ``z``, ``p``, and the interval are NaN when the fit carries no
        usable fixed-effect vcov.
    """
    at = str(at).lower()
    if at not in ("average", "mean"):
        raise ValueError(f"at must be 'average' or 'mean', got {at!r}.")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}.")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and > 0, got {scale}.")

    opt = getattr(fit, "_opt", None)
    if not isinstance(opt, dict) or "X" not in opt:
        raise TypeError(
            "marginal_effects needs a fitted Beta model with a design matrix.")
    if not hasattr(fit, "beta") or not hasattr(fit, "beta_original"):
        raise TypeError("marginal_effects needs a Beta model result.")

    p = int(opt["p"])
    X = np.asarray(opt["X"], dtype=float)
    b = np.asarray(fit.beta, dtype=float)[:p]
    b_orig = np.asarray(fit.beta_original, dtype=float)[:p]
    if X.ndim != 2 or X.shape[1] != p or b.size != p or b_orig.size != p:
        raise ValueError("the fit's design matrix and fixed effects do not agree.")

    des = opt.get("des", {}) or {}
    raw_cols = list(des.get("x_cols", []) or [])
    terms = [str(c) for c in raw_cols]
    if len(terms) != p - 1:
        raise ValueError("the fit does not carry one original name per predictor.")
    x_scale = des.get("x_scale", {}) or {}
    sds = np.empty(p - 1, dtype=float)
    for j, c in enumerate(raw_cols):
        sd = float(x_scale.get(c, (0.0, 1.0))[1])
        if not np.isfinite(sd) or sd <= 1e-12:
            raise ValueError(
                f"predictor {c!r} has zero scale; its marginal effect is not identified.")
        sds[j] = sd

    mu = expit(X @ b)
    w = mu * (1.0 - mu)
    if at == "average":
        wbar = float(np.mean(w))
        est = b_orig[1:] * wbar
        base = (X.T @ (w * (1.0 - 2.0 * mu))) / X.shape[0]
    else:
        mu0 = float(expit(b[0]))
        w0 = mu0 * (1.0 - mu0)
        est = b_orig[1:] * w0

    n = len(terms)
    se = np.full(n, np.nan)
    V = _fixed_vcov(fit, p)
    if V is not None:
        for j in range(n):
            if at == "average":
                g = b_orig[j + 1] * base
                g[j + 1] += wbar / sds[j]
            else:
                g = np.zeros(p)
                g[0] = b_orig[j + 1] * w0 * (1.0 - 2.0 * mu0)
                g[j + 1] = w0 / sds[j]
            se[j] = float(np.sqrt(max(g @ V @ g, 0.0)))

    z = np.divide(est, se, out=np.full(n, np.nan),
                  where=np.isfinite(se) & (se > 0.0))
    pval = 2.0 * norm.sf(np.abs(z))
    zcrit = float(norm.ppf(0.5 + level / 2.0))
    return pd.DataFrame({
        "term": terms,
        "estimate": est * scale,
        "se": se * scale,
        "z": z,
        "p": pval,
        "ci_low": (est - zcrit * se) * scale,
        "ci_high": (est + zcrit * se) * scale,
    }, columns=_COLUMNS)
