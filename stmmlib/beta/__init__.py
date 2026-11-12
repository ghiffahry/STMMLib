"""Beta family backend for spatio-temporal mixed models.

Modules:
    structure: M1-M14 model registry
    precision: AR(1) precision, rho/phi bounds, effective phi
    design: Z matrices, contrast matrices, W_gamma
    solvers: Newton/BFGS mode solver, Laplace marginal likelihood engine
    inference: Hessian-based SEs, Wald tests, confidence intervals
    result: BetastmmResult container
    fit: stmm_beta(), betaml() top-level fitting
    diagnostics: Residuals, ACF, Dunn-Smyth, Moran panel, Ljung-Box
    criteria: conditional AIC / conditional R-squared for flat models
    report: diagnostics report and convergence certificate
    lrt_fixed: restricted-refit likelihood ratio test for fixed effects
    config: numerical bounds and tolerances (single source of truth)
    moran_panel, plot_data, gpu_solvers, simulation, diagnose: helpers
"""
