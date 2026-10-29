# STMMLib

**Pustaka model campuran spatio-temporal untuk data panel area, dengan keluarga
sebaran Gaussian dan Beta.**

## Abstrak

STMMLib menyediakan satu antarmuka pemodelan untuk data panel area yang
menggabungkan efek acak wilayah, efek acak waktu, dan struktur korelasi spasial
melalui matriks pembobot `W`. Dua keluarga sebaran didukung: Gaussian dengan
pendugaan kemungkinan maksimum (ML/REML), dan Beta dengan pendugaan kemungkinan
maksimum beraproksimasi Laplace disertai sertifikat kelengkungan Hessian. Setiap
model menghasilkan penduga efek tetap beserta galat bakunya, komponen varians,
serta prediksi area melalui EBLUP (Gaussian) atau EBP (Beta). Pustaka ini
dikembangkan sebagai perangkat penelitian tesis dan diuji dengan uji invariansi
yang memastikan penambahan modul prediksi tidak mengubah hasil pendugaan.

## 1. Spesifikasi model

Untuk unit area `i = 1, ..., m` dan waktu `t = 1, ..., T`:

```
y_it = x_it' beta + b_i + u_t + e_it
```

dengan `b` efek acak area (IID, SAR, nested, atau SEBLUP Rao-Yu), `u` efek acak
waktu (IID atau AR(1)), dan `e` galat residual. Struktur interaksi area-waktu
(Tipe I sampai IV) tersedia untuk memisahkan komponen utama dari komponen
interaksi.

Kontrak pengurutan yang harus dipenuhi: baris dan kolom ke-`i` matriks `W`
bersesuaian dengan area ke-`i` dalam urutan terurut (`pd.Categorical`
mengurutkan kategori sebelum memberi kode). Pelanggaran kontrak ini tidak
menghasilkan galat, tetapi menghasilkan penduga yang salah.

## 2. Struktur model

Registry memuat 14 model struktural dan satu model regresi Beta tanpa efek acak.
`n_var` adalah jumlah parameter varians, tidak termasuk efek tetap.

| ID | Efek area | Efek waktu | Cakupan waktu | Interaksi | n_var (Gauss/Beta) |
|---|---|---|---|---|---|
| `m1_iid_iid` | IID | IID | per area | - | 3 / 3 |
| `m2_iid_ar1` | IID | AR(1) | per area | - | 4 / 4 |
| `m3_sar_iid` | SAR | IID | per area | - | 4 / 4 |
| `m4_sar_ar1` | SAR | AR(1) | global | - | 5 / 5 |
| `m5_seblup_rao_yu_it` | SEBLUP Rao-Yu | AR(1) | per area | - | 5 / 5 |
| `m6_nested_iid_iid` | nested IID | IID | per area | - | 4 / 4 |
| `m7_nested_ar1` | nested SAR | AR(1) | per area | - | 5 / 5 |
| `m8_iid_iid_int` | IID | IID | per area | IID | 4 / 4 |
| `m9_sar_iid_int` | SAR | IID | per area | spasial | 6 / 6 |
| `m10_iid_ar1_int` | IID | AR(1) | per area | temporal | 6 / 6 |
| `m11_sar_ar1_int` | SAR | AR(1) | global | spatio-temporal | 8 / 8 |
| `m12_seblup_rao_yu_it_int` | SEBLUP Rao-Yu | AR(1) | per area | spatio-temporal | 8 / 8 |
| `m13_nested_iid_iid_int` | nested IID | IID | per area | temporal | 6 / 6 |
| `m14_nested_ar1_int` | nested SAR | AR(1) | per area | IID | 6 / 6 |
| `beta_regression` | - | - | - | - | 0 / 2 |

## 3. Pendugaan dan prediksi

**Gaussian.** Kemungkinan marginal dimaksimumkan dengan ML atau REML; ragam
efek acak diperoleh dari optimasi kuasi-Newton dengan batas. Prediksi area
memakai EBLUP, yaitu `E[b | y]` pada parameter varians hasil pendugaan.

**Beta.** Kemungkinan maksimum didekati dengan aproksimasi Laplace, karena
integral marginal tidak tertutup. Setiap hasil fit menyertakan sertifikat
kelengkungan: norma gradien, ketermenentuan positif Hessian, dan jumlah
iterasi, sehingga status `konvergen` dapat dibedakan dari `invalid`.

**Skala koefisien.** Pada keluarga Beta, prediktor distandardisasi secara
internal sebelum optimasi. Nilai yang disimpan sebagai `beta_std` berada pada
skala terstandardisasi; tabel dan interpretasi harus memakai `beta_original`
disertai `se_original` agar satuan kembali ke satuan prediktor asli.

**Prediksi.** `ebp()` menghasilkan prediksi in-sample, `ebp_newdata()`
menghasilkan prediksi holdout untuk area yang sama, sedangkan area yang tidak
pernah teramati ditolak secara eksplisit. Validasi silang blok ruang pada
versi ini hanya mendukung prediksi efek tetap.

Catatan metodologis: EBP di sini berarti `E[mu | y]` yang dihitung dari
undian posterior. Ini bukan EBP Molina-Rao, dan galat perkiraan kuadrat tengah
(prinsip Prasad-Rao atau bootstrap) belum diimplementasikan untuk kedua
keluarga sebaran.

## 4. Tumpukan teknologi

| Lapisan | Komponen | Peran |
|---|---|---|
| Bahasa | Python >= 3.10 | implementasi tunggal untuk kedua keluarga |
| Komputasi | NumPy >= 1.24 | aljabar matriks, kolokasi Cholesky, undian normal |
| Optimasi | SciPy >= 1.11 | L-BFGS-B, matriks jarang, fungsi khusus |
| Data | pandas >= 2.0 | panel, kategori area, penanganan data hilang |
| Akselerasi | CuPy / PyTorch (opsional, ekstra `[gpu]`) | operasi array di GPU, diimpor dengan pengaman |
| Pengujian | pytest >= 7, pytest-cov | uji invariansi, uji regresi, uji kecocokan solusi |
| Paket | setuptools + wheel | backend build, distribusi wheel murni Python |
| Lingkungan | uv | kunci dependensi (`uv.lock`) |

Seluruh dependensi inti murni CPU. `cupy` dan `torch` hanya dibutuhkan bila
ekstra `[gpu]` dipasang.

## 5. Instalasi dan penggunaan

```powershell
pip install -e .
```

```python
from stmmlib.stmm import stmm

fit = stmm(
    data=df,
    model="m5_seblup_rao_yu_it",
    family="beta",
    W=W,
    method="Laplace",
    area_col="Kabupaten",
    time_col="Tahun",
    gpu="no",
)

print(fit.summary())
beta = fit.beta_original          # skala asli
se = fit.se_original              # galat baku, skala asli
mu, se_mu = fit.ebp(n_draws=200, seed=0, return_se=True)
```

Nama kolom dapat disesuaikan (`response_col`, `area_col`, `time_col`,
`prov_col`, `kab_col`) sehingga data tidak perlu diganti nama. Preset akurasi
`fast`, `balanced`, `medium`, dan `full` mengatur toleransi, batas iterasi, dan
perhitungan Hessian.

## 6. Validasi

Verifikasi dilakukan pada lingkungan pengembangan internal. Rangkaian uji tidak
disertakan dalam repositori ini, sehingga angka di bawah tidak dapat dijalankan
ulang dari salinan publik.

- Invariansi: menyalakan atau mematikan `compute_posterior` tidak mengubah
  `beta`, `logLik`, `b_hat`, maupun komponen varians (selisih eksak 0.0).
- Kecocokan: EBLUP model `m5_seblup_rao_yu_it` dibandingkan dengan solusi
  langsung persamaan campuran Henderson pada toleransi `1e-7`.
- Regresi: rangkaian uji dijalankan penuh sebelum setiap hasil dipakai untuk
  pelaporan.

Rincian API ringkas tersedia di `docs/API.md`.

## 7. Keterbatasan

- Jalur Gaussian belum portabel ke GPU; gunakan `gpu="no"`.
- EBP belum disertai galat perkiraan kuadrat tengah.
- Validasi silang blok hanya untuk efek tetap, bukan efek acak.
- Model nested dengan interaksi berukuran padat dan lambat; naikkan `max_iter`.
- `simulate_beta` belum membangkitkan efek interaksi, sehingga pemulihan
  Monte Carlo untuk M8 dan seterusnya belum tersedia.

## 8. Sitasi

```bibtex
@mastersthesis{ghiffary_stmmlib,
  author = {Ghiffary, Ghardapaty},
  title  = {<judul tesis>},
  school = {IPB University},
  year   = {<tahun>},
  note   = {Pustaka penelitian: \url{https://github.com/ghiffahry/STMMLib}}
}
```

Isi `<judul tesis>` dan `<tahun>` sebelum repositori dipublikasikan.

## 9. Lisensi

MIT. Lihat berkas `LICENSE`.
