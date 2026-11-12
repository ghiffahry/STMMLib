# Changelog

All notable changes to `stmmlib`. Format follows Keep-a-Changelog; versions
follow Semantic Versioning.

## [0.2.2] - 2026-11-13

### Fixed

- README claims now match the code:
  - M7/M14 area effect is nested IID, not nested SAR. The Beta path says so
    explicitly, and the Gaussian nested covariance does not use `W`.
  - GPU acceleration is PyTorch-only and applies to the Beta inner solves;
    the Gaussian path stays on CPU.
  - Removed the spatial block cross-validation claim: the package does not
    implement it.
  - Holdout prediction is Beta-only (`ebp_newdata`), and that method rejects
    nested and interaction models.
  - The quick-start report call uses `summary(strict=True)` and describes the
    certificate warning accurately.
  - Citation version and LICENSE link now point at the current release and
    absolute GitHub URLs, so they work on PyPI.

### Changed

- The `gpu` extra no longer pulls `cupy-cuda12x`; only the PyTorch path is
  used for acceleration.
- The sdist now ships `CHANGELOG.md` and `docs/API.md` through `MANIFEST.in`,
  because the README references the API reference.
- Added Documentation and Changelog project URLs to the package metadata.
- The package docstring now describes the actual GPU path (PyTorch/CUDA for
  the Beta inner solves).
- License metadata now uses the SPDX expression `MIT` with
  `setuptools>=77` and no deprecated license classifier.

## [0.2.1] - 2026-11-12

### Fixed

- The Gaussian optimiser's default iteration cap went from 200 to 500. On
  scipy 1.11 the AR(1) profile likelihood needs 223 iterations (scipy 1.18
  needs 196), so the old default reported not-converged for a fit that does
  converge. An explicit `max_iter` is still honoured exactly: the fit then
  reports not-converged rather than running past the cap.

### Changed

- Installation section now states that `pip install stmmlib` pulls NumPy,
  SciPy and pandas automatically, notes that the `dev` extra needs a source
  checkout because the tests are not part of the wheel, and gives a
  one-line verification command.
- Version moved to 0.2.1 so a rebuild from this branch cannot collide with
  the published 0.2.0 filenames on PyPI.

## [0.2.0] - 2026-11-11

### Added

- Fixed-effect inference in both families. Wald is the default; a
  restricted-refit likelihood ratio test is available with
  `inference="lrt"` and `lrt_terms=[...]`.
  - Beta: `stmm(..., family="beta", inference="lrt")` or the universal
    entry point `stmmlib.beta.lrt_fixed.lrt_fixed_effects(fit, data, terms=...)`;
    supports `lrt_deadline_s` and `method="refine"` (Newton ascent on the
    restricted model) or `method="plain"` (re-runs the solver's own
    optimiser). A refit that does not reach the restricted maximum does not
    report a p-value.
  - Gaussian: same API. Likelihoods are compared with ML (REML
    likelihoods are not comparable across fixed-effects designs); the
    caller's REML fit is untouched. The certificate is the likelihood:
    two independent starts must agree, and the restricted likelihood may
    not exceed the full one. The fixed-effects table shows variable
    names (not `beta_0`) and, in LRT mode, p (LRT) with the Wald p-value
    kept at `fit["inference"]["pvalue_wald"]`.
- Gaussian fits now carry `coef_names` (real predictor names), `x_cols`,
  `X`, `y`, `model`, `lrt_family` — the Gaussian tables are readable.
- Convergence certificate for the Beta family: `fit.certificate`
  (`valid`, `optimum_valid`, `grad_norm`, `hessian_pd`, `near_pd_applied`).
  `fit.summary(strict=True)` hides Wald inference when the certificate is
  not inference-grade.
- Input guards: Beta response outside (0,1) or NaN predictors raise at the
  door; unbalanced panels warn; non-contiguous area/time codes raise; W
  ordering is warned about.
- Suite doubles to 150+ tests (inference switch, LRT both families, input
  guards, universal API, Satterthwaite honesty).

### Fixed

- The restricted refit could not find the restricted maximum on the thesis
  panel (step dominated by bound-active coordinates, then clipped to an
  empty move). The Newton ascent is now projected onto the KKT active
  set and the Armijo test uses the realised displacement; all six terms
  certify `ok` and D agrees with the Wald z\u00b2 statistic.
- A silently wrong Gaussian model: the seblup and nested paths skipped
  the literal `"y"` while the response is read from `response_col`, so any
  other response name made the response a predictor of itself. The skip
  list now comes from the names actually in use.
- A predictor literally named `prov` or `kab` was silently dropped by the
  grouping-column convention in both families; the paths now warn and say
  how to rename it.
- The Beta summary printed "efek-waktu 11 level" in every report (a
  literal from one panel). It now prints the fitted number of periods.
- `df_method="satterthwaite"` is honest: it warns that it uses residual
  degrees of freedom and documents the `df_method="normal"` alternative.
- A predictor named `y` now raises a clear collision error instead of a
  misleading "response must be in (0,1)" message.

### Changed

- `stmm()` validates the family against the model registry:
  `beta_regression` cannot be fitted as if it were a Gaussian model, and
  unknown `area`/`time` types raise instead of producing
  `logLik=nan` / `status=invalid` silently.
- The registry's `time_scope` is honored by the Gaussian family
  (previously re-derived from sar+ar1); the seblup and nested families
  carry it too.
- Nested Beta models print cAIC / R\u00b2 as `n/a` with the reason
  (conditional criteria are flat-model only), not a bare `nan`.
- Comments and docstrings no longer quote thesis measurements; `stmmlib/
  note/` moved to `docs/notes/`; the package builds clean sdist/wheel.
- Default LRT behaviour: the untested terms carry NaN p-values under
  the LRT header; Wald p-values stay available.
- `beta/__init__.py` documents the actual modules; README is a general
  reference (not a thesis-bound note), lists both families' LRT and the
  test-suite workflow.

## [0.1.0] - 2026-11-10

Initial public reference implementation: Gaussian and Beta families, one
API, 14 model structures (m1-m14) + `beta_regression`, spatial (IID, SAR,
nested, SEBLUP Rao-Yu) and temporal (IID, AR(1)) random effects, optional
interactions, Gaussian ML/REML, Beta Laplace-approximated ML, EBLUP/EBP
prediction, GPU backend via CuPy/PyTorch.

[0.1.0]: initial public reference implementation (2026-11-10).
