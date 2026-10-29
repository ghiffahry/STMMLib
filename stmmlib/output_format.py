"""Output formatter for stmmlib results.

Single dispatch entry (`format_output`) that routes to the family-specific
formatter (`output_format_beta` / `output_format_normal`).  The model-fit
statistics block (`fmt_model_fit`) is shared by every family so logLik /
AIC / BIC always render identically regardless of family.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _fmt(x, digits: int = 4) -> str:
    """Format a number for display; NA-safe."""
    if x is None or (hasattr(x, "__len__")):
        return "NA"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(x) or np.isinf(x):
        return "NA"
    if abs(x) < 2e-16:
        return "0.0000"
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def _sig(p) -> str:
    """Significance code for a p-value (statsmodels / R convention)."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return " "


def _ci(est, se, level: float = 0.95):
    """Wald confidence interval without external stats dependency."""
    z = 1.959963984540054  # qnorm(0.975)
    est, se = float(est), float(se)
    return est - z * se, est + z * se


def _footer(width: int = 72) -> list:
    ln = []
    ln.append("-" * width)
    ln.append(f"Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    return ln


def _assess(obj: Any) -> str:
    """Return the formatter name ('beta' | 'normal') for an object."""
    if isinstance(obj, dict):
        fam = str(obj.get("family", "normal")).lower()
        return "normal" if fam in ("normal", "gaussian") else "normal"
    cls = type(obj).__name__
    if cls in ("BetastmmResult", "BetaResult"):
        return "beta"
    fam = getattr(obj, "family", None)
    if fam is not None:
        return "beta" if str(fam).lower() in ("beta",) else "normal"
    if cls == "StMM":
        return "beta" if str(getattr(obj, "_family", "gaussian")).lower() == "beta" else "normal"
    return "normal"


def format_output(obj: Any, width: int = 72) -> str:
    """Format a fitted stmmlib object for display.

    Dispatches automatically on the fitted object's family.

    Args:
        obj: A BetastmmResult (beta) or a normal fit dict.
        width: Line width for the report.

    Returns:
        str: English, statsmodels-style summary report.
    """
    kind = _assess(obj)
    if kind != "beta":
        from stmmlib.output_format_normal import format_normal
        return format_normal(obj, width=width)

    from stmmlib.output_format_beta import format_beta
    head = format_beta(obj, width=width)

    # Append evaluation + diagnostics so a single `print(fit)` shows the full
    # English report, exactly as stmm_beta_evaluation / stmm_beta_diagnostics.
    from stmmlib.beta.report import (
        stmm_beta_evaluation,
        stmm_beta_diagnostics,
    )

    _W = getattr(obj, "_W", None)
    try:
        _ev, _evtxt = stmm_beta_evaluation(obj, use_W=_W)
    except Exception:
        _evtxt = None
    try:
        _dg, _dgtxt = stmm_beta_diagnostics(obj, W=_W, as_str=True)
    except Exception:
        _dgtxt = None

    parts = [head]
    if _evtxt:
        parts.append("")
        parts.append(_evtxt)
    if _dgtxt:
        parts.append("")
        parts.append(_dgtxt)
    return "\n".join(parts)


def fmt_model_fit(stats: dict, width: int = 72) -> list:
    """Shared model-fit statistics block (logLik / AIC / BIC / cAIC).

    Identical for every family so the fit-statistics section reads the same
    regardless of whether a beta or a gaussian model produced it.

    Args:
        stats: dict with keys logLik, AIC, BIC (float) and optional
            cAIC, k, n.
        width: Line width.

    Returns:
        list[str]: Lines for the model-fit block.
    """
    ln = []
    ln.append("Model fit statistics:")
    ln.append(f"  log-likelihood : {_fmt(stats.get('logLik'), 4)}")
    ln.append(f"  AIC            : {_fmt(stats.get('AIC'), 4)}")
    ln.append(f"  BIC            : {_fmt(stats.get('BIC'), 4)}")
    if stats.get("cAIC") is not None:
        ln.append(f"  cAIC           : {_fmt(stats.get('cAIC'), 4)}")
    if stats.get("k") is not None and stats.get("n") is not None:
        ln.append(f"  (k = {_fmt(stats.get('k'), 0)}, n = {_fmt(stats.get('n'), 0)})")
    return ln
