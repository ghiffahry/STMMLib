# stmmlib API Reference (ringkas)

## Fitting

```python
from stmmlib.stmm import stmm

# Satu pintu untuk semua model (M1-M14, gaussian & beta)
fit = stmm(data=df, model="m4_sar_ar1", family="beta", W=W,
           response_col="Miskin", area_col="Kabupaten", time_col="Tahun",
           prov_col="Provinsi", kab_col="Kabupaten",
           gpu="no", n_starts=1, max_iter=3000, inner_max_iter=15)
```

- `family="beta"` → `BetastmmResult`; `family="gaussian"` → dict hasil core.
- Model ids: lihat `stmmlib/registry.py` (m1_iid_iid ... m14_nested_ar1_int,
  plus `beta_regression` = ML fixed-effects multi-prediktor).
- `gpu`: `"no"` (disarankan; gaussian path tidak GPU-portable), `"yes"`, `"auto"`.

### BetastmmResult (beta)
| Atribut/metode | Isi |
|---|---|
| `beta`, `phi`, `sigma_time`, `sigma_gamma`, `VarCorr()` | estimasi |
| `log_lik`, `logLik`, `AIC`, `BIC` | kriteria |
| `convergence` (dict), `status` | konvergen \| max_iter \| invalid; `at_bound`, `degenerate` |
| `inference` | est/se/z/p per param + `vcov`, `hessian_pd`, `lambda_min`, label |
| `residuals("dunn_smyth" \| "response" \| "conditional" \| "marginal")` | residual |
| `ranef()`, `coef()`, `predict(X_new)` | efek acak, koefisien, prediksi |
| `predict_interval(level=0.95, include_obs=True)` | PI logit-delta (P5) |
| `stmmlib.beta.criteria.beta_criteria(fit)` | cAIC, df_hat, cR2 marginal/conditional, komponen varian |

## Diagnostik (satu pintu)

```python
from stmmlib.diagnostics_suite import (
    residual_diagnostics_beta, residual_diagnostics_gaussian,   # P9
    random_effects_diagnostics_beta,                            # P10
    vif_fixed_effects, nested_lrt_with_df, wald_table_beta,     # P11
    validate_assumptions,                                       # P12
    prediction_intervals_gaussian,                              # P5
)
```

- Residual: Shapiro/AD/KS, simulated envelope k=500, heteroskedastisitas
  (Spearman + BP), Cook/leverage, residual-vs-prediktor.
  `y_override=` untuk deteksi misfit out-of-sample.
- Efek acak: normalitas BLUP per blok, Moran I (W fit), ACF1 tahunan,
  shrinkage, CI Wald-log komponen varian.
- Struktur: balanced panel, validitas W (row-std, konektivitas, spectral),
  boundary mu, flag at_bound/degenerate.

## Uji komponen varian

```python
from stmmlib.lrt import boundary_lrt          # H0 di BATAS (Self & Liang)
from stmmlib.diagnostics_suite import nested_lrt_with_df  # H0 interior

boundary_lrt(ll_full, ll_null, label="sigma2_gamma")   # p = 0.5*(1-F1(D))
nested_lrt_with_df(ll_full, ll_null, df=1)             # chi2_df biasa
```

Aturan: varian diuji ke 0 → WAJIB `boundary_lrt` (naif chi2_1 konservatif,
ambang benar D>2.706). Melepas struktur utuh (rho/phi interior) → `nested_lrt_with_df`.
