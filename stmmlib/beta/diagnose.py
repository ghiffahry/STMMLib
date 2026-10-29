"""Boundary diagnosis for AR/precision parameters (pure decision logic).

Maps the manual decision tree in note/PENANGANAN_BOUNDARY_AR_PHI.md into a
pure function: no fitting, no simulation, no data access. It only classifies
evidence the caller already collected (seed endpoints, bound sensitivity,
gradient, panel length T).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BoundaryDiagnosis:
    """Structured boundary verdict (not free text)."""

    classification: str
    evidence: list = field(default_factory=list)
    recommended_action: str = ""
    citations: list = field(default_factory=list)


_CITATIONS = [
    "Nickell (1981) panel pile-up on short T",
    "Chung et al. (2013) penalized likelihood for variance parameters",
]

_SEED_TOL = 1e-6
_SHIFT_TOL = 0.01
_GRAD_TOL = 1.0


def _as_floats(values) -> list:
    out = []
    for v in values:
        try:
            if isinstance(v, (tuple, list)) and len(v) > 0:
                v = v[0]
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def diagnose_boundary(
    fit_result=None,
    *,
    n_seeds_tested: int = 0,
    seed_endpoints=(),
    bound_sensitivity=None,
    gradient_at_solution=None,
    T=None,
    t_warn: int = 15,
) -> BoundaryDiagnosis:
    """Classify a boundary solution from collected evidence (pure function).

    Rules: non-identical seed endpoints -> MULTI_BASIN; identical endpoints
    with a significant bound shift and a small gradient -> ARTIFICIAL_BOUND;
    identical endpoints with no shift and a large gradient ->
    STRUCTURAL_BOUNDARY (dual ML + penalized report required, no cosmetic
    fix). T < t_warn appends a Nickell/pile-up warning to evidence.
    """
    evidence: list = []
    grad = gradient_at_solution
    tt = T
    if grad is None and fit_result is not None:
        try:
            om = (fit_result._opt or {}).get("optim", {}) or {}
            g = om.get("grad_norm")
            grad = float(g) if g is not None else None
        except (AttributeError, TypeError, ValueError):
            grad = None
    if tt is None and fit_result is not None:
        try:
            des = (fit_result._opt or {}).get("des", {}) or {}
            tt = des.get("T")
        except (AttributeError, TypeError):
            tt = None

    endpoints = _as_floats(seed_endpoints or ())
    n_seed_info = len(endpoints)
    if n_seed_info >= 2:
        spread = max(endpoints) - min(endpoints)
        identical = spread <= _SEED_TOL
        evidence.append(
            f"{n_seed_info} seed endpoints, spread {spread:.3g} "
            f"({'identical' if identical else 'not identical'})."
        )
    else:
        identical = True
        evidence.append(
            f"Only {n_seed_info} seed endpoint(s) "
            f"(n_seeds_tested={n_seeds_tested}); multi-basin not ruled out."
        )

    shift = None
    shifted = False
    if isinstance(bound_sensitivity, dict) and len(bound_sensitivity) >= 2:
        vals = _as_floats(bound_sensitivity.values())
        if vals:
            shift = max(vals) - min(vals)
            shifted = shift > _SHIFT_TOL
            evidence.append(
                f"Bound sensitivity over {len(vals)} settings, "
                f"shift {shift:.3g} "
                f"({'significant' if shifted else 'small'})."
            )
    elif bound_sensitivity is not None:
        evidence.append("Bound sensitivity incomplete (need 2+ settings).")

    if grad is not None:
        try:
            evidence.append(f"Gradient at solution {float(grad):.3g}.")
        except (TypeError, ValueError):
            pass
    grad_large = grad is not None and float(grad) >= _GRAD_TOL
    grad_small = grad is not None and float(grad) < _GRAD_TOL

    if tt is not None:
        try:
            evidence.append(f"Panel length T={tt}.")
        except (TypeError, ValueError):
            pass

    if not identical:
        classification = "MULTI_BASIN"
        action = "Perbanyak seed dan perbaiki starting value."
    elif shifted and grad_small:
        classification = "ARTIFICIAL_BOUND"
        action = "Longgarkan bound berdasar bukti sensitivitas."
    elif (not shifted) and grad_large and shift is not None:
        classification = "STRUCTURAL_BOUNDARY"
        action = (
            "Wajib lapor ganda ML dan penalized/Bayes modal; "
            "larang cosmetic fix."
        )
    elif shift is None and grad_large:
        classification = "STRUCTURAL_BOUNDARY"
        action = (
            "Wajib lapor ganda ML dan penalized/Bayes modal; "
            "larang cosmetic fix (sensitivitas bound disarankan)."
        )
    elif grad_large and shifted:
        classification = "STRUCTURAL_BOUNDARY"
        action = (
            "Wajib lapor ganda ML dan penalized/Bayes modal; "
            "larang cosmetic fix."
        )
    else:
        classification = "INDETERMINATE"
        action = "Lengkapi bukti (seed, sensitivitas bound, gradien)."

    try:
        if tt is not None and float(tt) < float(t_warn):
            evidence.append(
                f"T={tt} < {t_warn}: warning Nickell bias and pile-up "
                "pada panel pendek."
            )
    except (TypeError, ValueError):
        pass

    return BoundaryDiagnosis(
        classification=classification,
        evidence=evidence,
        recommended_action=action,
        citations=list(_CITATIONS),
    )
