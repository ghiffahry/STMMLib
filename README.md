# stmmlib

Spatio-temporal mixed models for areal panel data, with Gaussian and Beta
families behind one API.

## Overview

`stmmlib` fits area-level panel models that combine spatial random effects
(IID, SAR, nested, SEBLUP Rao-Yu), temporal random effects (IID, AR(1)), and
optional area-time interactions. Two likelihood families are available, so the
same model structure can be fitted to continuous responses or to proportions.

The package was developed for small area estimation on Indonesian district
poverty data and is published as a reference implementation for that work.

## Features

- 14 model structures (M1 to M14) plus a Beta regression baseline
- Gaussian family: ML and REML by marginal likelihood
- Beta family: Laplace-approximated ML with convergence certificates (gradient
  norm, Hessian positive definiteness, iteration count)
- Prediction: EBLUP for Gaussian, EBP from posterior draws for Beta, both
  in-sample and for same-area holdout
- Configurable column names, so data does not need renaming
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

## Quick start

Gaussian fit:

```python
import pandas as pd
from stmmlib.stmm import stmm

df = pd.read_csv("panel.csv")  # needs y, area, time, and predictor columns

# W must be m x m, row standardised, with rows ordered like sorted(area)
fit = stmm(
    data=df,
    model="m3_sar_iid",
    family="gaussian",
    W=W,
    method="REML",
    area_col="area",
    time_col="time",
    gpu="no",
)

fit["beta_hat"]
fit["se_beta"]
fit["blup"]["v_hat"]
```

Beta fit:

```python
fit = stmm(
    data=df,
    model="m5_seblup_rao_yu_it",
    family="beta",
    W=W,
    method="Laplace",
    area_col="area",
    time_col="time",
    gpu="no",
)

print(fit.summary())
fit.beta_original   # fixed effects on the original predictor scale
fit.se_original
mu, se_mu = fit.ebp(n_draws=200, seed=0, return_se=True)
```

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
prediction intervals, and the thesis helper scripts.

## Limitations

- EBP here means `E[mu | y]`, computed from posterior draws. It is not the
  Molina-Rao EBP, and its mean squared error is not implemented.
- Spatial block cross-validation supports fixed effects only.
- The Gaussian path is CPU only. Run it with `gpu="no"`.
- Gaussian fits return a dict, Beta fits return an object. The two return types
  are not unified yet.
- Nested models with interactions are dense and slow. Raise `max_iter` for them.
- `simulate_beta` does not generate interaction effects, so Monte Carlo
  recovery for M8 and above is not available.

## Citation

```bibtex
@mastersthesis{ghiffahry_stmmlib,
  author = {Ghiffary, Ghardapaty},
  title  = {<thesis title>},
  school = {IPB University},
  year   = {<year>},
  note   = {Software: \url{https://github.com/ghiffahry/STMMLib}}
}
```

## License

MIT. See [LICENSE](LICENSE).
