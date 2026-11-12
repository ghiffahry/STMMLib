# stmmlib API Reference (ringkas)

## Fitting

```python
from stmmlib.stmm import stmm

# Satu pintu untuk semua model (M1-M14, gaussian & beta)
fit = stmm(data=df, model="m4_sar_ar1", family="beta", W=W,
           response_col="proporsi", area_col="wilayah", time_col="periode",
           prov_col="provinsi", kab_col="wilayah",
           gpu="no", n_starts=1, max_iter=3000, inner_max_iter=15)
```

- `family="beta"` → `BetastmmResult`; `family="gaussian"` → dict hasil core.
- Model ids: lihat `stmmlib/registry.py` (m1_iid_iid ... m14_nested_ar1_int,
  plus `beta_regression` = ML fixed-effects multi-prediktor).
- `gpu`: `"no"` (disarankan; gaussian path tidak GPU-portable), `"yes"`, `"auto"`.

### Uji efek tetap: Wald atau LRT

```python
# Wald (default): z dari Hessian, p dari N(0,1)
fit = stmm(data=df, model="m5_seblup_rao_yu_it", family="beta", W=W, ...)

# LRT: refit model tanpa satu peubah, D = 2*(ll_full - ll_red), df = 1
fit = stmm(data=df, model="m5_seblup_rao_yu_it", family="beta", W=W,
           inference="lrt", lrt_terms=["pendidikan", "pendapatan"])

# universal: LRT untuk fit yang sudah ada, data mentah sebagai argumen
from stmmlib.beta.lrt_fixed import lrt_fixed_effects
lrt = lrt_fixed_effects(fit, df, terms=["pendidikan"])
```

- `inference`: `"wald"` (default) atau `"lrt"`. P-value yang dilaporkan
  mengikuti metode yang dipilih; p Wald selalu tersimpan di
  `fit.inference["pvalue_wald"]`.
- `lrt_terms`: peubah yang diuji LRT. `None` = semua peubah (satu refit per
  peubah, mahal).
- `lrt_deadline_s`: batas waktu; peubah yang tersisa ditandai
  `skipped_deadline`.
- Refit LRT memakai model yang sama dengan fit utama (config yang sama
  menentukan batas rho dan phi, jadi config berbeda berarti likelihood
  berbeda) dan mode `b` dari fit utama sebagai warm start solve dalam.
  Ada dua jalur: `method="plain"` menjalankan ulang optimizer solver dengan
  prosedur yang sama seperti fit utama, `method="refine"` (default)
  mengevaluasi likelihood secara langsung dan melangkah dengan arah Newton,
  sehingga refit yang tidak mencapai titik diam tidak menghasilkan p-value.

#### Gaussian: LRT efek tetap selain Wald

```python
fit = stmm(data=df, model="m3_sar_iid", family="gaussian", W=W,
           method="REML",                       # fit utama tetap REML
           inference="lrt", lrt_terms=["x1"])
```

- Perbandingan likelihood **harus ML**: likelihood REML tidak bisa
  dibandingkan antar desain X yang berbeda. Karena fit utama boleh REML,
  LRT menjalankan refit kedua model dengan ML secara terpisah dan blok
  `fit["lrt"]` melaporkan logLik ML itu sendiri; fit utama tidak berubah
  byte-per-byte (logLik, beta, se identik dengan `inference="wald"`).
- Sertifikatnya berbasis likelihood, bukan flag optimizer: dua start
  independen (warm start model penuh + start otomatis keluarga) harus
  mencapai likelihood yang sama, dan likelihood reduksi tidak boleh
  melampaui likelihood penuh (nested check). Kalau salah satu gagal,
  p-value = None dan status menjadi `failed` / `not_converged`.
- Tabel fixed effects menampilkan nama variabel asli (`x_cols`), bukan
  `beta_0`; kolom statistik/p (LRT) berbeda header, p Wald tersimpan di
  `fit["inference"]["pvalue_wald"]` untuk peubah yang tidak diuji.
- Di model campuran, LRT dan Wald bukan uji yang sama: LRT adalah uji
  profil (komponen varian di-refit), Wald uji kondisional (varian di-fiks).
  Pada panel sintetis: D = 46,70 vs z^2 = 66,88; dengan varian di-fiks,
  D = z^2 persis. Untuk pemeriksaan silang, pakai fit `method="ML"`.
- Keluarga yang didukung: core (m1-m4, m8-m11), SEBLUP (m5/m12),
  nested (m6/m13), nested-ar1 (m7/m14).

### Sertifikat konvergensi dan laporan

```python
from stmmlib.output_format_beta import format_beta

print(format_beta(fit))              # banner SERTIFIKAT bila valid=False
print(format_beta(fit, strict=True)) # z/p/CI efek tetap disembunyikan
```

- `fit.inference_ok`: True/False/None. False berarti titik berada di batas
  atau gradiennya terlalu besar: estimasi titik dan prediksi boleh dipakai,
  Wald z/p/CI tidak.
- `fit.status` hanya soal optimizer (konvergen / max_iter / invalid);
  kelayakan inferensi dibaca dari `fit.inference_ok` atau
  `fit.certificate["valid"]`.

### Batas korelasi dan refit penalti otomatis

```python
from stmmlib.beta.config import BetaFitConfig
cfg = BetaFitConfig(rho_bound=0.90, phi_bound=0.95, auto_prior=False)
```

- `rho_bound`: batas korelasi spasial (rho SAR, rho gamma).
- `phi_bound`: batas koefisien AR(1) (phi_T, phi_gamma). Keduanya terpisah;
  sebelumnya satu knob mengatur keduanya.
- Presisi Beta bukan salah satu dari keduanya; itu `logphi_bounds`.
- `auto_prior`: None (default; refit penalti hanya jalan bila config tidak
  eksplisit), True (selalu), False (tidak pernah).
- `fit.lrt`: blok bukti mentah (`D`, `df`, `p`, `status`, `converged`,
  `seconds`, `phi_at_bound`, `penalized`). Refit yang gagal tidak pernah
  menghasilkan angka p.

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
