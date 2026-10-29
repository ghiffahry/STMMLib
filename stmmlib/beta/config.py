"""Central numeric constants for Beta STMM fitting.

Single source of truth for bounds, tolerances, and accuracy presets
previously scattered as literals across solvers/precision/design/criteria.
All defaults reproduce the historical behavior exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

# Auto weak-prior constants (blme-style, Chung et al. 2013): center 6.9 ~
# phi 1000. These do NOT move the box bounds, only the prior center/spread.
# M5 queen: tarikan ganda (phi+dep) jatuh ke lembah degenerate ll -276832, maka dep off.
PHI_PRIOR_AUTO_MEAN = 6.9
PHI_PRIOR_AUTO_SD = 2.0
DEP_PRIOR_AUTO_SD = None


@dataclass(frozen=True)
class BetaFitConfig:
    """Numeric config for Beta STMM fitting (all defaults = legacy behavior).

    Weak-prior penalty (opt-in MAP, default OFF):
        phi_prior_mean / phi_prior_sd: Gaussian prior on logphi, penalty
            0.5*((logphi-mu)/sd)^2. dep_prior_sd: Gaussian prior N(0, sd^2)
            on raw_dep, penalty 0.5*sum(raw_dep^2)/sd^2. All default None
            (= unpenalized ML; behavior identical to before).
        MAP with weak Gaussian prior; precedent Chung et al. 2013 blme
        (Psychometrika) and brglm2 MPL Jeffreys. Always report penalized
        alongside the unpenalized solution.
        Note: passing an explicit config (even a default BetaFitConfig())
        disables the automatic boundary refit in stmm_beta(); leaving
        config=None lets stmm_beta auto-refit with a weak prior
        (PHI_PRIOR_AUTO_MEAN/SD, DEP_PRIOR_AUTO_SD) when the unpenalized
        fit hits the log_phi/raw_dep boundary.
        hessian_time_budget 3600.0 = cap 1 jam untuk Hessian (default; cegah
        hang). None = tanpa batas waktu. Set angka detik lain untuk mengubah cap.
    """

    phi_bound: float = 0.95
    # Half-life rationale for 0.95: rho=0.95 gives half-life
    # ln(0.5)/ln(0.95) ~= 13.5 periods, about the panel length T=11.
    # Do not raise the bound just to escape the boundary.
    beta_bounds: tuple = (-20.0, 20.0)
    logphi_bounds: tuple = (-10.0, 10.0)
    logs_bounds: tuple = (-10.0, 3.0)
    dep_margin: float = 0.95
    grad_tol: float = 1.0
    grad_slack: float = 100.0
    inner_probe_cap: int = 15
    # Interim dataset-calibrated prior; always report alongside unpenalized ML (Chung et al. 2013).
    phi_prior_mean: float | None = None
    phi_prior_sd: float | None = None
    dep_prior_sd: float | None = None
    hessian_time_budget: float | None = 3600.0
    # Laplace posterior curvature of the random effects at the final mode
    # (H_b = Z'diag(w)Z + Q), stored as its Cholesky factor for EBP draws.
    # Opt-in: dense q x q (q ~ 1.7e3 for M5, ~23 MB per fit). Default off.
    compute_posterior: bool = False

    def __post_init__(self):
        if (self.phi_prior_mean is None) != (self.phi_prior_sd is None):
            raise ValueError(
                "phi_prior_mean and phi_prior_sd must both be set or both None.")
        if self.phi_prior_sd is not None and self.phi_prior_sd <= 0:
            raise ValueError("phi_prior_sd must be > 0.")
        if self.dep_prior_sd is not None and self.dep_prior_sd <= 0:
            raise ValueError("dep_prior_sd must be > 0.")
        if self.hessian_time_budget is not None and self.hessian_time_budget <= 0:
            raise ValueError("hessian_time_budget must be > 0 or None.")


PRESET_ACCURACY: dict = {
    "fast": (1e-5, 1e-5, False, 500),
    "balanced": (1e-8, 1e-6, True, 500),
    "medium": (1e-8, 1e-6, False, None),
    "full": (1e-12, 1e-6, True, None),
}


def resolve_preset(preset: str | None) -> tuple:
    """Resolve accuracy preset name to (ftol, gtol, hess, cap)."""
    if preset is None:
        preset = "full"
    try:
        return PRESET_ACCURACY[preset]
    except KeyError:
        raise ValueError(
            f"Unknown preset: {preset!r}. Use 'fast', 'balanced', 'medium' or 'full'.")
