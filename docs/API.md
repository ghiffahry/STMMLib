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
| `beta`, `phi`, `sigma_area`, `sigma_time`, `sigma_gamma` | estimasi |
| `log_lik`, `AIC`, `BIC`, `k` | kriteria |
| `convergence` (dict), `status` | konvergen \| max_iter \| invalid; `at_bound`, `degenerate` |
| `inference` | est/se/z/p per param + `vcov`, `hessian_pd`, `lambda_min`, label |
| `residuals("dunn_smyth" \| "response" \| "conditional" \| "marginal")` | residual |
| `ranef()`, `coef()`, `predict(X_new)` | efek acak, koefisien, prediksi |
| `predict_interval(level=0.95, include_obs=True)` | PI logit-delta (P5) |
| `variance_decomp` / via notebook | R2_m, R2_c (\u00a75.9/5.10 note) |

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

## Validasi silang R

`script/tesis/crossval_r/run_crossval_r.py` — 5 kasus, 14/14 metrik (lme4
REML identik 0.0000; glmmTMB ar1 identik 0.0000; beta family diff <=0.014).
Detail: note \u00a714.15, \u00a714.21.

## Skrip tesis (script/tesis/)

| Skrip | Fungsi |
|---|---|
| `refit_fast.py` | refit M1-M5 paralel data asli -> fit_results/*.pkl |
| `run_coverage_M5.py`, `run_coverage_variance.py` | G6 coverage (efek tetap / varian) |
| `run_residual_diagnostics_M1_M5.py` | tabel P9 |
| `run_random_effects_diagnostics_M1_M5.py` | tabel P10 |
| `run_lrt_beta.py` | demo LRT batas 2 arah |
| `smoke_stmm_m1_m14.py`, `smoke_beta_m8m12.py`, `smoke_beta_nested.py` | smoke 14 model |
| `block_cv.py` | spatial block CV OOS |

Dokumentasi metodologi lengkap (gate G1-G7, audit, bug history):
`stmmlib/note/Kerangka_BetaGLMM_M1-M14_Referensi_Akademis.md` (\u00a714.1-14.23).
