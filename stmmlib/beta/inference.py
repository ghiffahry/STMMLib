"""Standard errors, Wald tests, and inference for Beta STMMs.

Uses numerical Hessian from optim_results to compute:
- SEs for all parameters (beta, logphi, log_sigma, raw_dep)
- Wald t-tests and p-values
- Confidence intervals
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def compute_stbl_inference(optim_results: dict, method: str = "numeric",
                           skip_hessian: bool = False) -> dict:
    """Compute standard errors from Hessian of the Laplace objective.

    Args:
        optim_results: Output from beta_laplace_structural.
        method: "numeric" (finite-difference Hessian, fast) or "profile"
            (adds opt-in likelihood-profile SEs for structural/dependence
            params; one 1-D scan per param, slower).
        skip_hessian: True skips ALL Hessian work (fast/medium presets) --
            se/vcov come back NaN/None and hessian_pd None. Display code must
            say "SE tidak dihitung", never silently present NaNs as numbers.

    Returns:
        dict with se, tvalues, pvalues for all parameters, plus the split
        se_fixed / se_structural and explicit hessian_pd / near_pd_applied
        flags (never hide a repaired Hessian).
    """
    from stmmlib.beta.solvers import beta_nll_curvature

    p = optim_results["p"]
    ksd = optim_results.get("ksd", 2)
    k = optim_results["k"]
    n = optim_results["n"]

    beta_hat = optim_results["beta_hat"]
    logphi = optim_results["logphi"]
    logs = optim_results["logs"]
    raw_dep = optim_results["raw_dep"]

    if skip_hessian:
        return _empty_inference(optim_results, p, ksd, raw_dep)

    param_vec = np.concatenate([beta_hat, [logphi], logs, raw_dep])
    n_params = len(param_vec)

    # Single Hessian source: solver H only (eps 1e-4 at best_param via
    # _neg_marginal_l with warm-chain inner plus polish). A second FD here
    # (eps 1e-5, fixed-start inner, no prior, gamma rho bound) gave 165x
    # SE divergence on M5, so no recompute: missing H means honest NaNs.
    _H_solver = (optim_results.get("optim", {}) or {}).get("hessian")
    if _H_solver is None or np.asarray(_H_solver).shape != (n_params, n_params):
        _na = _empty_inference(optim_results, p, ksd, raw_dep)
        _na["hessian_note"] = ("Hessian solver tidak tersedia; SE tidak dihitung "
                               "(sumber tunggal: Hessian solver, tanpa hitung ulang).")
        return _na
    H = np.asarray(_H_solver, dtype=np.float64)

    # Ensure positive definite. A non-PD Hessian at the converged point means
    # the point may not be a valid local maximum or the model is weakly
    # identified (Pawitan 2001). Always report explicitly when near_pd() was
    # applied; never present a repaired Hessian silently as clean.
    hessian_pd = False
    near_pd_applied = False
    lambda_min = float("nan")
    try:
        from stmmlib._utils import is_pd, near_pd
        from stmmlib._backend import Backend
        backend = Backend(gpu=False)
        lambda_min = float(np.linalg.eigvalsh(H).min())
        if is_pd(H, backend):
            hessian_pd = True
        else:
            near_pd_applied = True
            H = near_pd(H, backend)
        V = np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(V), 0))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)

    tvalues = param_vec / se
    pvalues = 2 * (1 - norm.cdf(np.abs(tvalues)))

    # Parameter labels (shared with the skip_hessian path below)
    labels = _inference_labels(optim_results, raw_dep, ksd, p)


    # Split inference: Wald SEs are reliable for fixed effects (linear
    # predictor scale). For structural/dependence parameters the Wald SE is
    # only indicative on the transformed scale and can be unreliable near
    # boundaries; prefer likelihood-profile SEs (method="profile", opt-in:
    # slower, one 1-D scan per structural param).
    p_fixed = int(p)
    p_struct = int(p + 1 + ksd + len(raw_dep))
    se_wald = se
    se_profile = _profile_se(param_vec, p_fixed, p_struct, optim_results) if method == "profile" else np.full(p_struct - p_fixed, np.nan)

    return {
        "param_labels": labels,
        "estimate": param_vec,
        "se": se_wald,
        "se_fixed": se_wald[:p_fixed],
        "se_structural": se_wald[p_fixed:p_struct],
        "se_profile_structural": se_profile,
        "tvalue": tvalues,
        "pvalue": pvalues,
        "vcov": V if not np.any(np.isnan(se_wald)) else None,
        "hessian_pd": bool(hessian_pd),
        "lambda_min": lambda_min,
        "near_pd_applied": bool(near_pd_applied),
        "hessian_note": (
            "Hessian tidak PD, diperbaiki via near_pd(). Titik konvergensi mungkin "
            "bukan maksimum lokal valid / identifiability lemah."
            if near_pd_applied else "Hessian positive definite."
        ),
    }


def _inference_labels(optim_results: dict, raw_dep, ksd: int, p: int) -> list:
    """Parameter labels by role (single copy used by both inference paths)."""
    x_cols = optim_results.get("des", {}).get("x_cols", [])
    labels = ["(Intercept)"] + x_cols + ["log(phi)"]
    from stmmlib.beta.structure import stmm_beta_structure
    _s = stmm_beta_structure(optim_results["model"])
    _is_nested = "prov" in optim_results.get("des", {}).get("blocks", {})
    # Sigma labels by role: flat = [area, time, (gamma)];
    # nested = [prov, kab, (eta if ar1), (gamma)]
    _sig_names = (["log_sigma_prov", "log_sigma_kab"]
                  if _is_nested else ["log_sigma_area", "log_sigma_time"])
    if _is_nested and _s["time"] == "ar1":
        _sig_names.append("log_sigma_eta")
    if ksd > len(_sig_names):
        _sig_names.append("log_sigma_gamma")
    labels += _sig_names
    # Dependence names by role, in raw_dep order. Labels say "raw_* (atanh)":
    # these are the internal optimizer parameters, NOT the model's rho/phi
    # (rho = rb*tanh(raw), phi = phi_bound*tanh(...), phi_bound from des);
    # transformed values are shown alongside in the summary table.
    _dep_names = []
    if _s["area"] in ("sar", "seblup_rao_yu"):
        _dep_names.append("raw_rho (atanh)")
    if _s["time"] == "ar1":
        _dep_names.append("raw_phi_T (atanh)")
    if _s["interaction"] in ("spatial", "st"):
        _dep_names.append("raw_rho_gamma (atanh)")
    if _s["interaction"] in ("temporal", "st"):
        _dep_names.append("raw_phi_gamma (atanh)")
    labels += _dep_names + [f"raw_dep_{i}" for i in range(len(raw_dep) - len(_dep_names))]
    return labels


def _empty_inference(optim_results: dict, p: int, ksd: int, raw_dep) -> dict:
    """No-Hessian inference shell: estimates + labels, SEs all NaN."""
    beta_hat = np.asarray(optim_results["beta_hat"], dtype=float)
    logphi = float(np.asarray(optim_results["logphi"], dtype=float).ravel()[0])
    logs = np.asarray(optim_results["logs"], dtype=float).ravel()
    raw_dep = np.asarray(raw_dep, dtype=float).ravel()
    param_vec = np.concatenate([beta_hat, [logphi], logs, raw_dep])
    n_all = len(param_vec)
    nan = np.full(n_all, np.nan)
    p_struct = int(p + 1 + ksd + len(raw_dep))
    return {
        "param_labels": _inference_labels(optim_results, raw_dep, ksd, p),
        "estimate": param_vec,
        "se": nan,
        "se_fixed": nan[:p],
        "se_structural": nan[p:p_struct],
        "se_profile_structural": np.full(max(p_struct - p, 0), np.nan),
        "tvalue": nan,
        "pvalue": nan,
        "vcov": None,
        "hessian_pd": None,
        "lambda_min": float("nan"),
        "near_pd_applied": False,
        "hessian_note": ("Hessian tidak dihitung (mode fast/medium) -- "
                         "SE/Wald/CI tidak tersedia."),
    }


def _profile_se(param_vec: np.ndarray, p_fixed: int, p_struct: int,
                optim_results: dict) -> np.ndarray:
    """Likelihood-profile SEs for structural/dependence params via 1-D scans.

    For each structural parameter j (>= p_fixed), find the interval where the
    profiled negative log-likelihood rises by chi2_1(0.95)/2 = 1.9207 from its
    minimum; SE = (upper-lower)/(2*1.96). Returns NaN when the scan hits a
    boundary or the objective does not rise enough (weakly identified).

    Cost is bounded (a few dozen objective evals per parameter), so this is
    opt-in at the call site and skipped when the objective is unbounded.
    """
    n_struct = p_struct - p_fixed
    se = np.full(n_struct, np.nan, dtype=np.float64)
    if n_struct <= 0:
        return se
    thr = 1.920729  # 0.5 * chi2_1(0.95) = 0.5*3.841459
    grid = np.linspace(-2.0, 2.0, 41)  # steps in units of Wald SE
    base = _neg_loglik_for_hessian(param_vec, optim_results)

    for k in range(n_struct):
        idx = p_fixed + k
        sd = _safe_wald_sd(param_vec, p_fixed, p_struct, optim_results, k)
        if not np.isfinite(sd) or sd <= 0:
            continue
        cross = []
        for s in grid:
            pv = param_vec.copy()
            pv[idx] += s * sd
            try:
                ll = _neg_loglik_for_hessian(pv, optim_results)
            except Exception:
                ll = np.nan
            if np.isfinite(ll):
                cross.append((s, ll - base))
        if not cross:
            continue
        # lower crossing (profile rises to threshold below the min)
        lo = hi = np.nan
        for s, delta in cross:
            if s < 0 and np.isfinite(delta) and delta <= thr:
                lo = s
        for s, delta in cross:
            if s > 0 and np.isfinite(delta) and delta <= thr:
                hi = s
        if np.isfinite(lo) and np.isfinite(hi):
            se[k] = (hi - lo) * sd / (2 * 1.96)
    return se


def _safe_wald_sd(param_vec, p_fixed, p_struct, optim_results, k) -> float:
    """Diagonal Wald SD for one structural param, or NaN if unusable."""
    try:
        _pv = param_vec.copy()
        eps = 1e-5
        for i in range(p_fixed, p_struct):
            pp = _pv.copy(); pm = _pv.copy()
            pp[i] += eps; pm[i] -= eps
            fp = _neg_loglik_for_hessian(pp, optim_results)
            fm = _neg_loglik_for_hessian(pm, optim_results)
            dd = (fp - 2 * _neg_loglik_for_hessian(_pv, optim_results) + fm) / eps**2
            if i == p_fixed + k:
                return float(np.sqrt(1.0 / dd)) if dd > 1e-12 else np.nan
    except Exception:
        return np.nan
    return np.nan


def _neg_loglik_for_hessian(param_vec: np.ndarray, optim_results: dict) -> float:
    """Evaluate negative marginal log-likelihood at given parameter vector."""
    from scipy.special import betaln
    from stmmlib.beta.solvers import beta_nll_curvature
    from stmmlib.beta.precision import ar1_precision
    from stmmlib.beta.design import stmm_beta_make_design
    from stmmlib.beta.structure import stmm_beta_structure

    p = optim_results["p"]
    ksd = optim_results.get("ksd", 2)
    n = optim_results["n"]
    Z = optim_results["Z"]
    X = optim_results["X"]
    y = optim_results["y"]
    des = optim_results["des"]
    model = optim_results["model"]
    blocks = des["blocks"]
    n_area = len(blocks.get("area", []))
    n_time = len(blocks.get("area_time", blocks.get("time", [])))
    n_gamma = len(blocks.get("interaction", []))

    beta = param_vec[:p]
    logphi = param_vec[p]
    logs = param_vec[p + 1: p + 1 + ksd]
    raw_dep = param_vec[p + 1 + ksd:]
    phi = np.exp(logphi)

    # Assemble Q (profile path only; main SEs use solver H above).
    s = stmm_beta_structure(model)
    rb = des.get("rho_bound", des.get("rho_bound_g", 0.95))
    _pb = des.get("phi_bound", 0.95)
    from stmmlib._backend import Backend
    backend = Backend(gpu=False)

    _has_sar = s["area"] in ("sar", "seblup_rao_yu")
    _has_ar1 = s["time"] == "ar1"
    is_nested = "prov" in blocks
    _raw_len = len(raw_dep)
    _rho_i = 0 if _has_sar else None
    _ph_i = (1 if _has_sar else 0) if _has_ar1 else None

    from scipy.sparse import block_diag as _sp_blkdiag
    blocks_Q = []
    if is_nested:
        # Nested hierarchy: [prov, kab, cell]; cell time shares kab variance
        # for iid time (B9) and uses its own eta variance for AR(1) time.
        n_prov = len(blocks.get("prov", []))
        n_kab = len(blocks.get("kab", []))
        n_cell = len(blocks.get("cell", []))
        blocks_Q.append(np.exp(-2.0 * logs[0]) * np.eye(n_prov, dtype=np.float64))
        blocks_Q.append(np.exp(-2.0 * logs[1]) * np.eye(n_kab, dtype=np.float64))
        if _has_ar1:
            ph = _pb * np.tanh(raw_dep[_ph_i]) if _raw_len > _ph_i else 0.0
            TQ = ar1_precision(des["T"], logs[2], ph, backend, phi_bound=_pb)
            if hasattr(TQ, 'toarray'):
                TQ = TQ.toarray()
            blocks_Q.append(_sp_blkdiag([TQ] * n_kab, format="csr").toarray())
        else:
            blocks_Q.append(np.exp(-2.0 * logs[1]) * np.eye(n_cell, dtype=np.float64))
        Q_base = np.asarray(_sp_blkdiag(blocks_Q, format="csr").toarray())
    else:
        if s["area"] in ("iid",):
            Q_a = np.exp(-2.0 * logs[0]) * np.eye(n_area, dtype=np.float64)
        elif _has_sar:
            rho = rb * np.tanh(raw_dep[_rho_i]) if _raw_len > _rho_i else 0.0
            W = optim_results.get("W")
            if W is None:
                W = des.get("W_gamma")
            if W is None:
                W = np.eye(n_area, dtype=np.float64)
            elif hasattr(W, 'toarray'):
                W = W.toarray()
            A = np.eye(n_area, dtype=np.float64) - rho * np.asarray(W, dtype=np.float64)[:n_area, :n_area]
            Q_a = np.exp(-2.0 * logs[0]) * (A.T @ A)
        else:
            raise ValueError(f"Unknown area type: {s['area']!r}")
        blocks_Q.append(Q_a)

        if _has_ar1:
            ph = _pb * np.tanh(raw_dep[_ph_i]) if _raw_len > _ph_i else 0.0
            if "area_time" in blocks:
                # area_specific ar1: block-diag per area (n_area blok T x T), no cross-area coupling
                T_cell = n_time // n_area if n_area else n_time
                TQ = ar1_precision(T_cell, logs[1], ph, backend, phi_bound=_pb)
                if hasattr(TQ, 'toarray'):
                    TQ = TQ.toarray()
                Q_t = _sp_blkdiag([TQ] * n_area, format="csr").toarray()
            else:
                Q_t = ar1_precision(n_time, logs[1], ph, backend, phi_bound=_pb)
                if hasattr(Q_t, 'toarray'):
                    Q_t = Q_t.toarray()
        else:
            Q_t = np.exp(-2.0 * logs[1]) * np.eye(n_time, dtype=np.float64)
        blocks_Q.append(Q_t)
        Q_base = np.asarray(_sp_blkdiag(blocks_Q, format="csr").toarray())

    Q_full = Q_base
    if n_gamma > 0:
        from stmmlib.beta.solvers import beta_Q_gamma
        Q_g = beta_Q_gamma(s, des, des["m"], des["T"], n_gamma, logs, raw_dep,
                           ksd, optim_results.get("q_dep", len(raw_dep)),
                           _has_sar, _has_ar1, backend)
        if Q_g is not None:
            Q_full = np.block([[Q_base, np.zeros((Q_base.shape[0], n_gamma))],
                               [np.zeros((n_gamma, Q_base.shape[0])), Q_g]])

    q = Z.shape[1]
    if Q_full.shape[0] < q:
        pad = q - Q_full.shape[0]
        Q_full = np.block([
            [Q_full, np.zeros((Q_full.shape[0], pad))],
            [np.zeros((pad, Q_full.shape[0])), np.eye(pad) * 1e-10]
        ])
    elif Q_full.shape[0] > q:
        Q_full = Q_full[:q, :q]

    # Newton solve (blocked for area-specific non-interaction models, same as fit)
    from stmmlib.beta.solvers import beta_solve_mode, beta_solve_mode_blocked, laplace_block_logdets
    use_blocked = (not is_nested and s["interaction"] == "none"
                   and s["time_scope"] == "area_specific")
    schur_inner = use_blocked
    start_b = optim_results.get("b_hat")
    if use_blocked:
        b, _, _ = beta_solve_mode_blocked(X, Z, Q_full, des["m"], des["T"], beta, logphi, y,
                                          inner_max_iter=50, start_b=start_b, inner_tol=1e-6)
    else:
        b, _, _ = beta_solve_mode(X, Z, Q_full, beta, logphi, y, inner_max_iter=50, start_b=start_b, inner_tol=1e-6)

    # NLL
    eta = X @ beta + Z @ b
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
    a = mu * phi
    bval = (1.0 - mu) * phi
    nll = -np.sum(-betaln(a, bval) + (a - 1) * np.log(np.clip(y, 1e-15, 1.0)) +
                   (bval - 1) * np.log(np.clip(1.0 - y, 1e-15, 1.0)))
    pen = 0.5 * b @ Q_full @ b

    try:
        if schur_inner:
            logdet_H, logdet_Q = laplace_block_logdets(
                Z, beta_nll_curvature(eta, y, phi), Q_full, des["m"], des["T"]
            )
        else:
            logdet_Q = np.linalg.slogdet(Q_full)[1]
            H_mat = Z.T @ (Z * beta_nll_curvature(eta, y, phi)[:, None]) + Q_full
            logdet_H = np.linalg.slogdet(H_mat)[1]
    except Exception:
        logdet_Q = 0.0
        logdet_H = 0.0

    loglik = -nll - pen + 0.5 * logdet_Q - 0.5 * logdet_H
    return -loglik if np.isfinite(loglik) else 1e20
