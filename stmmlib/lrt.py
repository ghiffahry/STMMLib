"""Boundary LRT (Self & Liang 1987) for variance components at zero.

H0: sigma2 = 0 places the null on the PARAMETER BOUNDARY, so the naive
chi2_1 reference overstates significance (roughly by a factor 2 in the
upper tail). The correct asymptotic reference for ONE variance component
at the boundary is the 50:50 mixture

    0.5 * chi2_0 + 0.5 * chi2_1                     (chi2_0 = point mass at 0)

so p = 0.5 * (1 - F_chi1(D)) for any D > 0 (Self & Liang 1987; Stram & Lee
1994 for variance components). For k components at the boundary the weights
shift; only the single-component case is implemented (that is the case that
occurs for one interaction / one random block).

Usage (any family):
    from stmmlib.lrt import boundary_lrt
    res = boundary_lrt(loglik_full, loglik_null, label="sigma2_gamma")
"""
from __future__ import annotations

from scipy.stats import chi2


def mixture_chi2_pvalue(D: float) -> float:
    """p-value for D under 0.5*chi2_0 + 0.5*chi2_1 (single boundary component).

    chi2_0 contributes point mass at 0: for D > 0 its tail probability is 0,
    hence p = 0.5 * (1 - F_chi1(D)) = p_naive / 2. The naive chi2_1 reference
    is therefore CONSERVATIVE for one boundary component (type-I error 2.5%
    instead of 5%); the correct rejection threshold is D > 2.706, not 3.841.
    """
    D = float(D)
    if D < 0:
        return 1.0
    if D == 0:
        return 1.0  # point mass at 0 under H0
    return 0.5 * (1.0 - float(chi2.cdf(D, 1)))


def boundary_lrt(loglik_full: float, loglik_null: float,
                 label: str = "sigma2_gamma") -> dict:
    """Boundary likelihood-ratio test for one variance component.

    Args:
        loglik_full: maximised log-likelihood of the model WITH the component.
        loglik_null: maximised log-likelihood of the model WITHOUT it
            (a separate fit on the reduced model, not a fixed parameter).
        label: human-readable name of the tested component.

    Returns dict with D, p_value (mixture), naive p (chi2_1, for contrast),
    and the verdict strings. D is clamped at 0 (numerical slack can make
    loglik_full marginally below loglik_null; the models are nested so the
    true D >= 0). The mixture p is exactly half the naive chi2_1 p for D > 0
    (the naive reference is conservative for one boundary component).
    """
    D = 2.0 * (float(loglik_full) - float(loglik_null))
    D_clamped = max(D, 0.0)
    p_mix = mixture_chi2_pvalue(D_clamped)
    p_naive = float(chi2.sf(D_clamped, 1)) if D_clamped > 0 else 1.0
    if p_mix < 0.01:
        verdict = "signifikan (p<0.01, mixture)"
    elif p_mix < 0.05:
        verdict = "signifikan (p<0.05, mixture)"
    else:
        verdict = "tidak signifikan (p>=0.05, mixture)"
    return {
        "label": label,
        "D": D,
        "D_clamped": D_clamped,
        "p_value": p_mix,
        "p_value_naive_chi2": p_naive,
        "reference": "0.5*chi2_0 + 0.5*chi2_1 (Self & Liang 1987)",
        "verdict": verdict,
    }
