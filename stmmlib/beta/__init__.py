"""Beta family backend for spatio-temporal mixed models.

Modules:
    structure: M1-M14 model registry, nraw_interaction
    precision: AR(1) precision, SAR precision, block-diagonal assembly
    design: Z matrices, contrast matrices, W_gamma
    solvers: Newton/BFGS mode solver, Laplace marginal likelihood engine
    inference: Hessian-based SEs, Wald tests, confidence intervals
    result: BetastmmResult, BetaResult containers
    fit: stmm_beta(), betaml() top-level fitting
    diagnostics: Residuals (marginal, conditional, randomized), ACF
    stability: variance inflation factor
    metrics: MSE, RMSE, MAE, logLik, AIC, BIC for Beta
    evaluation: MAPE, MAD, MSE, coverage, width
    criteria: conditional AIC (Huber) + conditional R2 (Nakagawa) for Beta
    simulation: simulate_beta(), simulate_beta_prediction()
    validation: response/model validation
"""
