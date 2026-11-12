# stmmlib

Spatio-temporal mixed models for areal panel data, with Gaussian and Beta
families behind one API.

## Overview

`stmmlib` fits area-level panel models that combine spatial random effects
(IID, SAR, nested, SEBLUP Rao-Yu), temporal random effects (IID, AR(1)), and
optional area-time interactions. Two likelihood families are available, so the
same model structure can be fitted to continuous responses or to proportions.

Developed as a reference implementation for a study of Indonesian district
poverty data. It is a general tool for area-level panel data, does not
depend on that dataset, and is published as a software reference.

## Features

- 14 model structures (M1 to M14) plus a Beta regression baseline
- Gaussian family: ML and REML by marginal likelihood
- Beta family: Laplace-approximated ML with convergence certificates (gradient
  norm, Hessian positive definiteness, iteration count)
- Prediction: EBLUP for Gaussian, EBP from posterior draws for Beta, both
  in-sample and for same-area holdout
- Configurable column names, so data does not need renaming
- Fixed-effect inference in both families: Wald tests and restricted-refit
  likelihood-ratio tests (`inference="lrt"`, `lrt_terms=[...]`)
- Optional GPU backend through CuPy or PyTorch

## Installation

The package is not on PyPI yet. Install from source:

```bash
git clone https://github.com/ghiffahry/STMMLib.git
cd STMMLib
pip install -e .
```

For the optional GPU backend:

```bash
pip install -e ".[gpu]"
```

Run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

## Quick start

Runs end to end on synthetic data, nothing external needed:

```python
import numpy as np
import pandas as pd
from stmmlib.stmm import stmm

m, T = 12, 6
rng = np.random.default_rng(7)
area = np.repeat(np.arange(m), T)
time = np.tile(np.arange(T), m)

df = pd.DataFrame({
    "y": np.clip(rng.beta(6, 4, m * T), 1e-4, 1 - 1e-4),   # in (0, 1), as required
    "area": area,
    "time": time,
    "x1": rng.normal(size=m * T),
    "x2": rng.normal(size=m * T),
})

fit = stmm(data=df, model="m1_iid_iid", family="beta",
           method="Laplace", gpu="no", n_starts=1)
print(fit.summary())             # reads the certificate, refuses to call an invalid fit inference-grade
print(fit.certificate["valid"])  # True/False: may the SEs be quoted?
```

Gaussian family, same call surface:

```python
df_gauss = df.assign(y=df["x1"] + rng.normal(0, 1, len(df)))
W = np.ones((m, m)); np.fill_diagonal(W, 0); W /= W.sum(1, keepdims=True)

fit = stmm(data=df_gauss, model="m3_sar_iid", family="gaussian",
           W=W, method="REML", area_col="area", time_col="time", gpu="no")
fit["beta_hat"]; fit["se_beta"]; fit["blup"]["v_hat"]
```

Fixed-effect inference in both families, Wald or likelihood-ratio test,
and the requirement that the LRT compares ML likelihoods, are documented
in `docs/API.md`.

## Models

| ID | Area effect | Time effect | Interaction |
|---|---|---|---|
| `m1_iid_iid` | IID | IID | none |
| `m2_iid_ar1` | IID | AR(1) | none |
| `m3_sar_iid` | SAR | IID | none |
| `m4_sar_ar1` | SAR | AR(1) | none |
| `m5_seblup_rao_yu_it` | SEBLUP Rao-Yu | AR(1) | none |
| `m6_nested_iid_iid` | nested IID | IID | none |
| `m7_nested_ar1` | nested SAR | AR(1) | none |
| `m8_iid_iid_int` | IID | IID | IID |
| `m9_sar_iid_int` | SAR | IID | spatial |
| `m10_iid_ar1_int` | IID | AR(1) | temporal |
| `m11_sar_ar1_int` | SAR | AR(1) | spatio-temporal |
| `m12_seblup_rao_yu_it_int` | SEBLUP Rao-Yu | AR(1) | spatio-temporal |
| `m13_nested_iid_iid_int` | nested IID | IID | temporal |
| `m14_nested_ar1_int` | nested SAR | AR(1) | IID |
| `beta_regression` | none | none | none |

Models ending in `_int` add an interaction term between area and time effects.
Models with SAR or SEBLUP area effects require `W`.

## Estimation and prediction

- Gaussian: `method="ML"` or `method="REML"`. The fit is returned as a dict
  with `beta_hat`, `se_beta`, `log_lik`, `AIC`, `BIC`, and `blup`
  (`v_hat`, `u_hat`, `se_v`).
- Beta: `method="Laplace"`. The fit is an object with `summary()`,
  `certificate`, `beta_original`, `se_original`, `ebp()`, and `ebp_newdata()`.
- Fixed effects in the Beta family are standardised internally. Use
  `beta_original` and `se_original` for reporting, not the standardised values.
- `ebp_newdata()` predicts for held-out periods of areas seen during fitting.
  Areas never seen during fitting are rejected.
- Accuracy presets: `fast`, `balanced`, `medium`, `full`.

## Requirements

- Python 3.10 or newer
- NumPy 1.24+, SciPy 1.11+, pandas 2.0+
- CuPy and PyTorch only for the `gpu` extra

## Documentation

`docs/API.md` holds a compact API reference covering fitting, diagnostics, LRT,
and prediction intervals.

## Limitations

- EBP here means `E[mu | y]`, computed from posterior draws. It is not the
  Molina-Rao EBP, and its mean squared error is not implemented.
- Spatial block cross-validation supports fixed effects only.
- The Gaussian path is CPU only. Run it with `gpu="no"`.
- Gaussian fits return a dict, Beta fits return an object. The two return types
  are not unified yet.
- Nested models with interactions are dense and slow. Raise `max_iter` for them.
- `simulate_beta` supports the flat models M1-M5 and the interaction models
  M8-M12. Nested models M6/M7/M13/M14 raise `ValueError`.
- `beta_criteria` (cAIC, conditional R2) returns NaN for nested models; their
  province/district/cell precision builder is not implemented.
- When the automatic prior refit is used, the reported `logLik`, `AIC`, and
  `BIC` are the penalized objective. The fit's `optim` record
  (`fit._opt["optim"]`) flags this with `penalty_auto` and reports the
  unpenalized value at the same estimate in `penalty_refit_msg`.

## Citation

```bibtex
@software{ghiffahry_stmmlib,
  author  = {Ghiffary, Ghardapaty},
  title   = {STMMLib: spatio-temporal mixed models for areal panel data},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/ghiffahry/STMMLib},
  license = {MIT}
}
```

## License

MIT. See [LICENSE](LICENSE).
