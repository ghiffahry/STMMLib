"""Uji Moran panel tiga pendekatan untuk residual model STMM Beta.

P1 (moran_current)       : Moran per tahun pada residual respons (y - mu)
                           kondisional, p-value via permutasi.
P2 (moran_literature)    : Moran per tahun pada residual Dunn-Smyth
                           kondisional, ekspektasi/varians analitik
                           Cliff-Ord (S0, S1, S2), kombinasi Fisher +
                           Stouffer + Bonferroni.
P3 (moran_panel_derived) : Moran panel pooled (W_panel = kron(I_T, W)),
                           p-value via block permutation, Stouffer-Hartung
                           dengan koreksi korelasi antar tahun, bootstrap
                           CI dengan resample area.

Semua fungsi publik menerima W=None. Bila efek acak area IID dan W tidak
diberikan, Moran tidak terdefinisi dan dict skipped dikembalikan.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from stmmlib.beta.structure import stmm_beta_structure
from stmmlib.output_format import _sig

# ---------------------------------------------------------------------------
# BAGIAN 0: konstanta default
# ---------------------------------------------------------------------------

DEFAULT_N_PERM = 999
DEFAULT_N_BOOT = 999
DEFAULT_SEED = 42

_BAR = "-" * 60

_SKIPPED_REASON = (
    "efek acak area IID: Moran tidak terdefinisi tanpa W; "
    "berikan W eksternal untuk uji pembanding, atau gunakan Ljung-Box"
)


# ---------------------------------------------------------------------------
# BAGIAN 1: kernel Moran
# ---------------------------------------------------------------------------

def _center(v):
    """Pusatkan vektor residual (kurangi rata-rata)."""
    v = np.asarray(v, dtype=float)
    return v - np.nanmean(v)


def _moran_I(v_c, W):
    """Moran's I dari vektor yang sudah terpusat dan matriks bobot W.

    Guard: bila S0 = sum(W) = 0 (semua baris W nol) raise ValueError.
    """
    v_c = np.asarray(v_c, dtype=float)
    m = v_c.size
    S0 = float(np.sum(W))
    if S0 <= 0:
        raise ValueError("semua baris W nol: Moran tidak terdefinisi")
    den = float(v_c @ v_c)
    if not np.isfinite(den) or den <= 0:
        return float("nan")
    lag = np.asarray(W, dtype=float) @ v_c
    return float((m / S0) * float(v_c @ lag) / den)


def _EI_var(W, k):
    """E[I] dan Var[I] rumus Cliff-Ord (asumsi normalitas, randomisasi).

    E[I] = -1/(m-k) (koreksi derajat bebas efek tetap; k=0 memberi
    -1/(m-1) standar). Var[I] = (m^2 S1 - m S2 + 3 S0^2)/(S0^2 (m^2-1))
    - E[I]^2, dengan S1 = 1/2 sum_ij (w_ij + w_ji)^2 dan
    S2 = sum_k (w_k. + w_.k)^2. Baris W nol tetap ikut dihitung.
    """
    W = np.asarray(W, dtype=float)
    m = W.shape[0]
    S0 = float(np.sum(W))
    S1 = 0.5 * float(np.sum((W + W.T) ** 2))
    S2 = float(np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2))
    if m - k <= 1 or S0 <= 0:
        return float("nan"), float("nan")
    e_i = -1.0 / (m - k)
    num = m * m * S1 - m * S2 + 3.0 * S0 * S0
    den = S0 * S0 * (m * m - 1.0)
    return float(e_i), float(num / den - e_i ** 2)


def _perm_p(stat, stats_perm, sisi="dua", center=None):
    """P-value permutasi dua sisi terhadap pusat distribusi permutasi.
    Pusat empiris c = mean(stats_perm) bila center=None."""
    stats_perm = np.asarray(stats_perm, dtype=float)
    stats_perm = stats_perm[np.isfinite(stats_perm)]
    if stats_perm.size == 0:
        return float("nan")
    if sisi == "besar":
        cnt = int(np.sum(stats_perm >= stat))
    elif sisi == "kecil":
        cnt = int(np.sum(stats_perm <= stat))
    else:
        c = float(np.mean(stats_perm)) if center is None else float(center)
        cnt = int(np.sum(np.abs(stats_perm - c) >= abs(float(stat) - c)))
    return float((cnt + 1) / (stats_perm.size + 1))


# ---------------------------------------------------------------------------
# BAGIAN 2 (internal): ekstraksi fit, guard IID, agregasi residual
# ---------------------------------------------------------------------------

def _iid_area(fit) -> bool:
    """True bila efek acak area model IID (termasuk nested IID)."""
    model = (getattr(fit, "_opt", None) or {}).get("model", "")
    try:
        s = stmm_beta_structure(model)
    except KeyError:
        return False
    return s["area"] in ("iid", "nested_iid")


def _resolve_W(fit, W):
    """W efektif: argumen lebih dulu, lalu W tersimpan di fit._opt."""
    if W is None:
        W = (getattr(fit, "_opt", None) or {}).get("W")
    if W is None:
        return None
    if hasattr(W, "toarray"):
        W = W.toarray()
    return np.asarray(W, dtype=float)


def _setup(fit, W):
    """Kembalikan (y, mu, resid_respons, des, k, m, T, area_idx, time_idx, Wd)
    atau dict skipped bila guard IID terpicu."""
    if _iid_area(fit) and W is None:
        return {"skipped": True, "reason": _SKIPPED_REASON}
    Wd = _resolve_W(fit, W)
    if Wd is None:
        raise ValueError("W tidak tersedia: berikan W atau fit dengan W tersimpan")
    opt = fit._opt
    y = np.asarray(opt["y"], dtype=float)
    mu = opt.get("mu_hat")
    if mu is None:
        mu = 1.0 / (1.0 + np.exp(-(np.asarray(opt["X"]) @ fit.beta)))
    mu = np.clip(np.asarray(mu, dtype=float), 1e-15, 1.0 - 1e-15)
    des = opt.get("des", {}) or {}
    m = int(des.get("m", 1))
    T = int(des.get("T", 1))
    k = int(np.asarray(opt["X"]).shape[1]) if opt.get("X") is not None else 0
    area_idx = np.asarray(
        des.get("area_idx", np.zeros(y.size, dtype=int)), dtype=int)
    time_idx = np.asarray(
        des.get("time_idx", np.zeros(y.size, dtype=int)), dtype=int)
    resid = y - mu
    return {"y": y, "mu": mu, "resid": resid, "des": des, "k": k, "m": m,
            "T": T, "area_idx": area_idx, "time_idx": time_idx, "W": Wd}


def _resid_area_t(resid_t, area_t, m):
    """Agregasi residual per area dalam satu tahun: rata-rata bincount."""
    cnt = np.bincount(area_t, minlength=m).astype(float)
    cnt[cnt == 0] = 1.0
    return np.bincount(area_t, weights=np.nan_to_num(resid_t),
                       minlength=m) / cnt


def _resid_matrix(resid, area_idx, time_idx, m, T):
    """Matriks residual m x T: R[a, t] = rata-rata residual area a tahun t."""
    R = np.zeros((m, T))
    cnt = np.zeros((m, T))
    for r, a, t in zip(resid, area_idx, time_idx):
        if 0 <= a < m and 0 <= t < T:
            R[a, t] += np.nan_to_num(r)
            cnt[a, t] += 1.0
    cnt[cnt == 0] = 1.0
    return R / cnt


def _year_labels(des, T):
    labels = des.get("time_labels", []) or []
    if len(labels) == T:
        return [str(l) for l in labels]
    return [str(t) for t in range(T)]


# ---------------------------------------------------------------------------
# BAGIAN 3: P1 - Moran per tahun, residual respons, permutasi
# ---------------------------------------------------------------------------

def moran_current(fit, W=None, n_perm=DEFAULT_N_PERM, seed=DEFAULT_SEED):
    """Moran per tahun pada residual respons (y - mu) kondisional.

    Per tahun: residual diagregasi rata-rata per area, I_t dihitung,
    p-value via permutasi posisi area (dua sisi terhadap pusat
    distribusi permutasi, p=(1+#ekstrem)/(B+1)).
    """
    st = _setup(fit, W)
    if "skipped" in st:
        return st
    m, T, Wd = st["m"], st["T"], st["W"]
    rng = np.random.default_rng(seed)
    notes = [
        "Residual respons kondisional (y - mu_hat), diagregasi rata-rata per area per tahun",
        "p-value permutasi dua sisi terhadap pusat distribusi permutasi, p=(1+#ekstrem)/(B+1)",
        f"n_perm={n_perm}, seed={seed}",
    ]
    if _iid_area(fit):
        notes.append("W eksternal, bukan struktur model")
    per_year = []
    for t in range(T):
        mask = st["time_idx"] == t
        if mask.sum() < 3:
            per_year.append({"year": _year_labels(st["des"], T)[t],
                             "I": float("nan"), "p_value": float("nan")})
            continue
        res_t = _resid_area_t(st["resid"][mask], st["area_idx"][mask], m)
        I_t = _moran_I(_center(res_t), Wd)
        stats_perm = []
        for _ in range(n_perm):
            zp = _center(res_t[rng.permutation(m)])
            stats_perm.append(_moran_I(zp, Wd))
        p_t = _perm_p(I_t, stats_perm)
        per_year.append({"year": _year_labels(st["des"], T)[t],
                         "I": float(I_t), "p_value": float(p_t)})
    is_valid = [d["I"] for d in per_year if np.isfinite(d["I"])]
    ps_valid = [d["p_value"] for d in per_year if np.isfinite(d["p_value"])]
    summary = {
        "I_mean": float(np.mean(is_valid)) if is_valid else float("nan"),
        "I_median": float(np.median(is_valid)) if is_valid else float("nan"),
        "p_median": float(np.median(ps_valid)) if ps_valid else float("nan"),
        "n_reject_05": int(sum(p < 0.05 for p in ps_valid)),
    }
    return {
        "approach": "P1",
        "per_year": per_year,
        "summary": summary,
        "notes": notes,
        "meta": {"n_perm": int(n_perm), "seed": int(seed)},
    }


# ---------------------------------------------------------------------------
# BAGIAN 4: P2 - Moran per tahun, residual DS, Anselin/Cliff-Ord
# ---------------------------------------------------------------------------

def moran_literature(fit, W=None):
    """Moran per tahun pada residual Dunn-Smyth kondisional.

    Sumber residual: stmmlib.beta.report._dunn_smyth (definisi tunggal yang
    sama dipakai plot_data.resid_normality dan ds_residuals_from_fit).
    E[I] per tahun = tr(M_t W)/(m-k) dengan M_t = I - X_t(X_t'X_t)^-1 X_t'
    (aproksimasi OLS); Var[I] Cliff-Ord (S0, S1, S2).
    """
    st = _setup(fit, W)
    if "skipped" in st:
        return st
    from stmmlib.beta.report import _dunn_smyth
    resid_ds = np.asarray(_dunn_smyth(st["y"], st["mu"], float(fit.phi)),
                          dtype=float)
    m, T, k, Wd = st["m"], st["T"], st["k"], st["W"]
    X = st["des"].get("X")
    if X is None:
        opt = fit._opt
        X = np.asarray(opt["X"], dtype=float) if opt.get("X") is not None else None
    _, v_cliff = _EI_var(Wd, k)
    notes = [
        "Residual Dunn-Smyth kondisional: Phi^-1(F_Beta(y)), sumber tunggal stmmlib.beta.report._dunn_smyth",
        "E[I] per tahun: tr(M_t W)/(m-k), M_t = I - X_t(X_t'X_t)^-1 X_t' (aproksimasi OLS)",
        "Var[I]: rumus Cliff-Ord (S0, S1, S2; asumsi normalitas, randomisasi)",
        "p/Z dua sisi (H1: autokorelasi positif atau negatif)",
        "Kombinasi: Fisher (df=2T), Stouffer, Bonferroni (alpha/T)",
    ]
    if _iid_area(fit):
        notes.append("W eksternal, bukan struktur model")
    per_year = []
    z_list = []
    p_list = []
    labels = _year_labels(st["des"], T)
    for t in range(T):
        mask = st["time_idx"] == t
        if mask.sum() < 3 or not np.isfinite(v_cliff) or v_cliff <= 0:
            per_year.append({"year": labels[t], "I": float("nan"),
                             "E_I": float("nan"), "Z": float("nan"),
                             "p_value": float("nan")})
            continue
        res_t = _resid_area_t(resid_ds[mask], st["area_idx"][mask], m)
        I_t = _moran_I(_center(res_t), Wd)
        # M_t aproksimasi OLS: proyeksi komplement per tahun
        e_t = float("nan")
        if X is not None:
            X_t = np.asarray(X[mask], dtype=float)
            if X_t.shape[0] == m and 0 < k < m:
                try:
                    H = X_t @ np.linalg.pinv(X_t.T @ X_t) @ X_t.T
                    M_t = np.eye(m) - H
                    e_t = float(np.trace(M_t @ Wd)) / (m - k)
                except Exception:
                    e_t = float("nan")
        z_t = (I_t - e_t) / np.sqrt(v_cliff) if np.isfinite(e_t) else float("nan")
        p_t = float(2 * stats.norm.sf(abs(z_t))) if np.isfinite(z_t) else float("nan")
        per_year.append({"year": labels[t], "I": float(I_t), "E_I": e_t,
                         "Z": float(z_t), "p_value": p_t})
        if np.isfinite(z_t):
            z_list.append(z_t)
        if np.isfinite(p_t):
            p_list.append(p_t)
    # Kombinasi uji per tahun
    fisher_stat = fisher_p = stouffer_z = stouffer_p = float("nan")
    if p_list:
        p_clip = np.clip(p_list, 1e-16, 1.0)
        fisher_stat = float(-2.0 * np.sum(np.log(p_clip)))
        fisher_p = float(stats.chi2.sf(fisher_stat, 2 * len(p_list)))
    if z_list:
        stouffer_z = float(np.sum(z_list) / np.sqrt(len(z_list)))
        stouffer_p = float(2 * stats.norm.sf(abs(stouffer_z)))
    n_reject_bonf = int(sum(p < 0.05 / T for p in p_list)) if p_list else 0
    return {
        "approach": "P2",
        "per_year": per_year,
        "combined": {
            "fisher_stat": float(fisher_stat), "fisher_p": float(fisher_p),
            "stouffer_Z": stouffer_z, "stouffer_p": stouffer_p,
            "n_reject_bonf": n_reject_bonf,
        },
        "notes": notes,
        "meta": {"k": k, "m": m, "T": T},
    }


# ---------------------------------------------------------------------------
# BAGIAN 5: P3 - Moran panel pooled, block permutation, Stouffer-Hartung,
#            bootstrap CI resample area
# ---------------------------------------------------------------------------

def moran_panel_derived(fit, W=None, n_perm=DEFAULT_N_PERM,
                        n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED):
    """Moran panel pooled + Stouffer-Hartung + bootstrap CI.

    W_panel = kron(I_T, W) (urutan time-major: indeks t*m + a).
    I_panel dengan pemusatan global; p via block permutation (permutasi
    residual dalam tiap tahun dengan posisi antar tahun dipertahankan).
    Stouffer-Hartung: Z_t per tahun dari distribusi permutasi + koreksi
    korelasi antar tahun. Bootstrap CI: resample area dengan replacement.
    """
    st = _setup(fit, W)
    if "skipped" in st:
        return st
    m, T, Wd = st["m"], st["T"], st["W"]
    R = _resid_matrix(st["resid"], st["area_idx"], st["time_idx"], m, T)
    v = R.T.ravel()  # time-major: indeks = t*m + a
    Wp = np.kron(np.eye(T), Wd)
    S0p = float(np.sum(Wp))
    rng = np.random.default_rng(seed)
    notes = [
        "Pendekatan pooled panel: residual respons (y - mu_hat) dalam satu vektor m*T",
        "W_panel = kron(I_T, W); I_panel dengan pemusatan global",
        "p-value block permutation dua sisi terhadap pusat distribusi permutasi, p=(1+#ekstrem)/(B+1): posisi area dipermutasi sama di semua tahun (ketergantungan antar tahun dipertahankan)",
        f"Stouffer-Hartung dengan df={T - 1}" if T > 1 else "Stouffer-Hartung: hanya 1 tahun, pakai normal",
        f"Bootstrap CI resample area dengan replacement, B={n_boot}, seed={seed}",
    ]
    if _iid_area(fit):
        notes.append("W eksternal, bukan struktur model")

    # I panel
    I_panel = _moran_I(_center(v), Wp)

    # Block permutation + distribusi I_t per tahun
    stats_p = []
    per_t = np.zeros((n_perm, T))
    for b in range(n_perm):
        perm = rng.permutation(m)
        vb = v.reshape(T, m)[:, perm]
        stats_p.append(_moran_I(_center(vb.ravel()), Wp))
        for t in range(T):
            per_t[b, t] = _moran_I(_center(vb[t]), Wd)
    p_block = _perm_p(I_panel, stats_p)

    # Z_t per tahun dari distribusi permutasi
    mu_t = per_t.mean(axis=0)
    sd_t = per_t.std(axis=0, ddof=1)
    I_t_obs = np.array([_moran_I(_center(R[:, t]), Wd) for t in range(T)])
    Z_t = np.where(sd_t > 1e-12, (I_t_obs - mu_t) / np.where(sd_t > 1e-12, sd_t, 1.0),
                   np.nan)

    # Matriks korelasi antar tahun dari replika permutasi
    corr = np.zeros((T, T))
    if T > 1 and n_perm > 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(per_t.T)
        corr = np.where(np.isfinite(corr), corr, 0.0)
    corr_list = [[float(x) for x in row] for row in corr]

    # Stouffer-Hartung
    off = corr[~np.eye(T, dtype=bool)]
    rho_bar = max(float(np.mean(off)), 0.0) if off.size else 0.0
    if T > 1 and np.isfinite(Z_t).sum() > 0:
        denom = np.sqrt(T + T * (T - 1) * rho_bar)
        z_sh = float(np.nansum(Z_t) / denom)
        p_sh = float(2 * stats.t.sf(abs(z_sh), T - 1))
    else:
        z_sh = float(np.nanmean(Z_t)) if np.isfinite(Z_t).any() else float("nan")
        p_sh = float(2 * stats.norm.sf(abs(z_sh))) if np.isfinite(z_sh) else float("nan")

    # Bootstrap CI: resample area dengan replacement
    ib_boot = []
    for _ in range(n_boot):
        idx = rng.choice(m, m, replace=True)
        Wb = Wd[np.ix_(idx, idx)]
        vb = R[idx, :].T.ravel()
        Wpb = np.kron(np.eye(T), Wb)
        if float(np.sum(Wpb)) <= 0:
            continue  # replika degenerate (area unik terlalu sedikit)
        ib_boot.append(_moran_I(_center(vb), Wpb))
    ib_boot = np.asarray([x for x in ib_boot if np.isfinite(x)], dtype=float)
    if ib_boot.size >= 2:
        ci = [float(np.percentile(ib_boot, 2.5)),
              float(np.percentile(ib_boot, 97.5))]
    else:
        ci = [float("nan"), float("nan")]

    labels = _year_labels(st["des"], T)
    per_year = []
    for t in range(T):
        stats_t = per_t[:, t]
        stats_t = stats_t[np.isfinite(stats_t)]
        p_t = _perm_p(I_t_obs[t], stats_t) if stats_t.size else float("nan")
        per_year.append({"year": labels[t], "I": float(I_t_obs[t]),
                         "p_value": float(p_t)})

    perm_center = float(np.mean(stats_p)) if stats_p else float("nan")

    return {
        "approach": "P3",
        "I_panel": float(I_panel),
        "p_block": float(p_block),
        "stouffer_Z": float(z_sh),
        "p_stouffer": float(p_sh),
        "ci95": ci,
        "per_year": per_year,
        "perm_center": perm_center,
        "resid_panel": R.T.ravel().copy(),
        "corr_matrix": corr_list,
        "notes": notes,
        "meta": {"n_perm": int(n_perm), "n_boot": int(n_boot),
                 "seed": int(seed), "k": st["k"], "m": m, "T": T},
    }


# ---------------------------------------------------------------------------
# BAGIAN 5b: caption + data scatter
# ---------------------------------------------------------------------------

def describe_moran(I, p, konteks="") -> str:
    """Satu kalimat deskripsi Moran's I dengan p dua sisi."""
    return f"{konteks}: I={I:+.4f}, p={p:.4f} (dua sisi)."


def build_caption_s4(I_obs, c, p_two, m, S0, T, tol=5e-4) -> str:
    """Caption plot Moran S4 dengan tafsir arah terhadap pusat permutasi."""
    delta = I_obs - c
    if abs(delta) < tol or p_two >= 0.05:
        arah = "tidak berbeda nyata dari pusat permutasi"
        tafsir = (
            "Residual area tidak menunjukkan pola spasial sistematis: "
            "area dengan residual tinggi tidak mengelompok di dekat area "
            "berresidual tinggi maupun rendah. Tafsir ini TIDAK "
            "membandingkan langsung dengan P1/P2/P3."
        )
    elif delta < 0:
        arah = "negatif"
        tafsir = (
            "Kecenderungan spasial negatif: area berresidual tinggi cenderung "
            "dikelilingi area berresidual rendah (pola HL/LH). Tafsir ini "
            "TIDAK membandingkan langsung dengan P1/P2/P3."
        )
    else:
        arah = "positif"
        tafsir = (
            "Kecenderungan spasial positif: area berresidual tinggi cenderung "
            "dikelilingi area berresidual tinggi, area berresidual rendah "
            "dikelilingi area berresidual rendah (pola HH/LL). Tafsir ini "
            "TIDAK membandingkan langsung dengan P1/P2/P3."
        )
    return (
        f"Sumbu X: residual area (rata-rata {T} tahun); sumbu Y: lag spasial "
        f"(W x residual). Arah {arah} (I={I_obs:+.4f}; pusat permutasi "
        f"c={c:+.4f}; p dua sisi={p_two:.4f}; m={m}; S0={S0:.1f}). {tafsir}"
    )


def build_caption_p3(I_panel, c, p_block, ci95, I_t, T_, m, S0_panel) -> str:
    """Caption plot Moran panel P3 (Panel A pooled, Panel B per tahun)."""
    return (
        f"Sumbu X: residual panel (y - mu); sumbu Y: lag spasial W_panel x "
        f"residual. Panel A: seluruh T tahun (I_panel={I_panel:+.4f}, "
        f"p_block={p_block:.4f}, CI95 bootstrap=[{ci95[0]:+.4f}, "
        f"{ci95[1]:+.4f}]). Panel B: per tahun, I_t berkisar "
        f"{min(I_t):+.3f} sampai {max(I_t):+.3f}. Garis pada plot adalah "
        f"OLS; slope OLS = {I_panel * S0_panel / (m * T_):+.4f} = I_panel x "
        f"S0_panel/(mT) karena S0_panel={S0_panel:.0f} dan mT={m * T_}; "
        f"bukan I_panel. Block permutation mempertahankan ketergantungan "
        f"antar tahun."
    )


def moran_panel_scatter_data(fit, W=None, n_perm=DEFAULT_N_PERM,
                             n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED):
    """Data plot Moran panel P3 + caption, tanpa menghitung ulang statistik.

    Memanggil moran_panel_derived SEKALI dan menurunkan koordinat scatter
    (z_flat, lag_flat) serta slope dari hasilnya. Pemanggil wajib memakai
    seed yang sama dengan pemanggilan P3 lain agar residual DS dan
    distribusi permutasi identik.

    Return dict: z_flat, lag_flat, colors, I_panel, c (perm_center),
    p_block, stouffer_Z, p_stouffer, ci95, I_t (list per tahun),
    slope_ols, slope_I, S0_panel, m, T, caption. Bila skipped, dict
    skipped dikembalikan apa adanya.
    """
    st = moran_panel_derived(fit, W, n_perm, n_boot, seed)
    if st.get("skipped"):
        return st
    des = (getattr(fit, "_opt", None) or {}).get("des", {}) or {}
    m = int(des.get("m", 1))
    T = int(des.get("T", 1))
    area_idx = np.asarray(
        des.get("area_idx", np.zeros(1, dtype=int)), dtype=int)
    Wd = _resolve_W(fit, W)
    if Wd is None:
        raise ValueError("W tidak tersedia: berikan W atau fit dengan W tersimpan")

    z_flat = np.asarray(st["resid_panel"], dtype=float)
    z_flat = z_flat - np.mean(z_flat)
    Wp = np.kron(np.eye(T), Wd)
    lag_flat = Wp @ z_flat
    colors = np.repeat(np.arange(T), m)
    S0_panel = float(np.sum(Wp))
    slope_ols = float(z_flat @ lag_flat / (z_flat @ z_flat))
    slope_I = float(st["I_panel"]) * S0_panel / (m * T)
    I_t = [float(d["I"]) for d in st["per_year"]]
    caption = build_caption_p3(float(st["I_panel"]), float(st["perm_center"]),
                               float(st["p_block"]), st["ci95"], I_t, T, m,
                               S0_panel)
    return {
        "z_flat": z_flat,
        "lag_flat": lag_flat,
        "colors": colors,
        "I_panel": float(st["I_panel"]),
        "c": float(st["perm_center"]),
        "p_block": float(st["p_block"]),
        "stouffer_Z": float(st["stouffer_Z"]),
        "p_stouffer": float(st["p_stouffer"]),
        "ci95": st["ci95"],
        "I_t": I_t,
        "slope_ols": slope_ols,
        "slope_I": slope_I,
        "S0_panel": S0_panel,
        "m": m,
        "T": T,
        "caption": caption,
    }


# ---------------------------------------------------------------------------
# BAGIAN 6: format output
# ---------------------------------------------------------------------------
def format_P1(result) -> str:
    """Format teks hasil P1 (moran_current)."""
    if result.get("skipped"):
        return f"DILEWATI: {result['reason']}"
    L = ["Moran per tahun (P1, residual respons)", _BAR]
    L.append(f"  {'Tahun':<10} {'I':>10} {'p':>10}  sig")
    for d in result["per_year"]:
        star = _sig(d["p_value"]) if np.isfinite(d.get("p_value", np.nan)) else ""
        L.append(f"  {d['year']:<10} {d['I']:>10.4f} {d['p_value']:>10.4f}  {star}")
    s = result["summary"]
    n_year = len(result["per_year"])
    L.append(f"  Ringkasan: I_mean={s['I_mean']:.4f}  I_median={s['I_median']:.4f}"
             f"  p_median={s['p_median']:.4f}")
    L.append(f"  Tahun signifikan (p<0.05): {s['n_reject_05']}/{n_year}")
    L.append("  CATATAN:")
    for n in result["notes"]:
        L.append(f"   - {n}")
    meta = result["meta"]
    L.append(f"  Meta: n_perm={meta['n_perm']}, seed={meta['seed']}")
    L.append(_BAR)
    return "\n".join(L)


def format_P2(result) -> str:
    """Format teks hasil P2 (moran_literature)."""
    if result.get("skipped"):
        return f"DILEWATI: {result['reason']}"
    L = ["Moran per tahun (P2, residual Dunn-Smyth, Anselin/Cliff-Ord)", _BAR]
    L.append(f"  {'Tahun':<10} {'I':>10} {'E[I]':>10} {'Z':>10} {'p':>10}  sig")
    for d in result["per_year"]:
        star = _sig(d["p_value"]) if np.isfinite(d.get("p_value", np.nan)) else ""
        L.append(f"  {d['year']:<10} {d['I']:>10.4f} {d['E_I']:>10.4f} "
                 f"{d['Z']:>10.4f} {d['p_value']:>10.4f}  {star}")
    c = result["combined"]
    L.append(f"  Fisher   : X2={c['fisher_stat']:.4f}, p={c['fisher_p']:.4f} (df=2T)")
    L.append(f"  Stouffer : Z={c['stouffer_Z']:.4f}, p={c['stouffer_p']:.4f}")
    L.append(f"  Bonferroni (alpha/T): {c['n_reject_bonf']}/{len(result['per_year'])} tahun")
    L.append("  CATATAN:")
    for n in result["notes"]:
        L.append(f"   - {n}")
    meta = result["meta"]
    L.append(f"  Meta: k={meta['k']}, m={meta['m']}, T={meta['T']}")
    L.append(_BAR)
    return "\n".join(L)


def format_P3(result) -> str:
    """Format teks hasil P3 (moran_panel_derived)."""
    if result.get("skipped"):
        return f"DILEWATI: {result['reason']}"
    L = ["Moran panel pooled (P3, W_panel = kron(I_T, W))", _BAR]
    L.append(f"  I_panel      : {result['I_panel']:+.4f}")
    L.append(f"  p (block)    : {result['p_block']:.4f}")
    L.append(f"  Stouffer-Hartung : Z={result['stouffer_Z']:+.4f}, "
             f"p={result['p_stouffer']:.4f} (df={result['meta']['T'] - 1})")
    L.append(f"  CI95 bootstrap: [{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}] (B={result['meta']['n_boot']})")
    L.append(f"  {'Tahun':<10} {'I':>10} {'p':>10}  sig")
    for d in result["per_year"]:
        star = _sig(d["p_value"]) if np.isfinite(d.get("p_value", np.nan)) else ""
        L.append(f"  {d['year']:<10} {d['I']:>10.4f} {d['p_value']:>10.4f}  {star}")
    L.append("  CATATAN:")
    for n in result["notes"]:
        L.append(f"   - {n}")
    meta = result["meta"]
    L.append(f"  Meta: n_perm={meta['n_perm']}, n_boot={meta['n_boot']}, seed={meta['seed']}, "
             f"k={meta['k']}, m={meta['m']}, T={meta['T']}")
    L.append(_BAR)
    return "\n".join(L)


def format_compare(p1, p2, p3) -> str:
    """Format teks perbandingan tiga pendekatan."""
    lines = []
    for r in (p1, p2, p3):
        if r.get("skipped"):
            lines.append(f"DILEWATI: {r['reason']}")
        elif r.get("error"):
            lines.append(f"GAGAL ({r.get('approach', '?')}): {r['error']}")
    if lines:
        return "\n".join(lines)
    L = ["Perbandingan uji Moran panel (3 pendekatan)", _BAR]
    s1, c2 = p1["summary"], p2["combined"]
    L.append(f"  P1 respons (per tahun)     : I_median={s1['I_median']:+.4f}  "
             f"p_median={s1['p_median']:.4f}  reject={s1['n_reject_05']}/{len(p1['per_year'])}")
    L.append(f"  P2 Anselin (residual DS)   : Stouffer Z={c2['stouffer_Z']:+.4f}  "
             f"p={c2['stouffer_p']:.4f}  bonf_reject={c2['n_reject_bonf']}/{len(p2['per_year'])}")
    L.append(f"  P3 pooled panel            : I_panel={p3['I_panel']:+.4f}  "
             f"p_block={p3['p_block']:.4f}  p_Stouffer-Hartung={p3['p_stouffer']:.4f}")
    L.append("  Semua p/Z dua sisi (H1: positif atau negatif).")
    L.append("  CATATAN:")
    for n in p3["notes"][:2]:
        L.append(f"   - {n}")
    L.append(f"  df Stouffer-Hartung = {p3['meta']['T'] - 1}, B bootstrap = {p3['meta']['n_boot']}")
    L.append(_BAR)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# BAGIAN 7: fasade
# ---------------------------------------------------------------------------

def compare_moran(fit, W=None, n_perm=DEFAULT_N_PERM, n_boot=DEFAULT_N_BOOT,
                  seed=DEFAULT_SEED):
    """Jalankan ketiga pendekatan, kembalikan dict {P1, P2, P3, compare}."""
    out = {}
    for name, fn in (("P1", moran_current), ("P2", moran_literature),
                     ("P3", moran_panel_derived)):
        try:
            if name == "P1":
                out[name] = fn(fit, W, n_perm=n_perm, seed=seed)
            elif name == "P3":
                out[name] = fn(fit, W, n_perm=n_perm, n_boot=n_boot, seed=seed)
            else:
                out[name] = fn(fit, W)
        except Exception as e:
            out[name] = {"approach": name, "error": str(e)}
    out["compare"] = format_compare(out["P1"], out["P2"], out["P3"])
    return out


# ---------------------------------------------------------------------------
# BAGIAN 8: self-check
# ---------------------------------------------------------------------------

class _FakeFit:
    """Fit tiruan minimal untuk self-check (tanpa refit)."""

    def __init__(self, model, m, T, seed):
        rng = np.random.default_rng(seed)
        n = m * T
        X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
        mu = 1.0 / (1.0 + np.exp(-(X @ np.array([0.0, 0.3, -0.2]))))
        area_idx = np.repeat(np.arange(m), T)
        time_idx = np.tile(np.arange(T), m)
        W = np.zeros((m, m))
        for i in range(m - 1):
            W[i, i + 1] = W[i + 1, i] = 1.0
        self.phi = 25.0
        self.beta = np.zeros(3)
        self.model = model
        self._opt = {
            "y": np.clip(mu + rng.normal(0, 0.05, n), 0.01, 0.99),
            "mu_hat": mu, "X": X, "des": {
                "m": m, "T": T, "area_idx": area_idx, "time_idx": time_idx,
                "area_labels": [f"A{i}" for i in range(m)],
                "time_labels": [str(2018 + t) for t in range(T)],
            },
            "W": W, "model": model,
        }


def _self_check():
    m, T = 8, 3
    print(f"Self-check moran_panel: panel sintetis m={m}, T={T}, W rantai")
    fit = _FakeFit("m3_sar_iid", m, T, DEFAULT_SEED)
    r1 = moran_current(fit)
    r2 = moran_literature(fit)
    r3 = moran_panel_derived(fit)
    print(format_compare(r1, r2, r3))
    # guard IID: M1 tanpa W harus skipped
    fit_iid = _FakeFit("m1_iid_iid", m, T, DEFAULT_SEED)
    sk = moran_current(fit_iid)
    print("Guard IID:", sk["reason"] if sk.get("skipped") else "GAGAL: guard tidak terpicu")


if __name__ == "__main__":
    _self_check()
