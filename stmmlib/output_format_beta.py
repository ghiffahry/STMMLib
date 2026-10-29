"""Beta-family output formatter (statsmodels summary style, English).

Formats a fitted `BetastmmResult` into a readable regression-style report:
header, fixed-effects table, precision/variance parameters, random-effects
components, shared model-fit statistics, and an honest convergence note.
"""

from __future__ import annotations

import numpy as np

from stmmlib.output_format import _fmt, _sig, _ci, _footer, fmt_model_fit


def _status_label(status: str) -> str:
    return {
        "konvergen": "Converged",
        "max_iter": "Max iterations reached",
        "invalid": "Invalid / degenerate",
    }.get(str(status), str(status))


def _bound_rows(opt, labels) -> set:
    """Map certificate bound_params (internal names) to table row indices.

    Layout: [beta_0..beta_{p-1}, logphi, logs_ksd, raw_dep].
    Solvers use ["log_phi"] + [f"log_sigma_{i}"] + [f"raw_dep_{i}"].
    Unknown names are ignored (show too much, never hide).
    """
    try:
        cert = opt.get("certificate") or {}
    except AttributeError:
        return set()
    bps = cert.get("bound_params") or []
    if not cert.get("at_bound") or not bps:
        return set()
    try:
        p = int(opt.get("p", 0))
    except (TypeError, ValueError):
        return set()
    n = len(labels)
    try:
        ksd = opt.get("ksd", None)
        ksd = int(ksd) if ksd is not None else None
    except (TypeError, ValueError):
        ksd = None
    if ksd is None:
        ksd = sum(1 for _l in labels if str(_l).startswith("log_sigma"))
    out = set()
    for _b in bps:
        _name = str(_b)
        _idx = None
        if _name == "log_phi":
            _idx = p
        elif _name.startswith("log_sigma_"):
            try:
                _idx = p + 1 + int(_name.split("log_sigma_")[1])
            except (TypeError, ValueError, IndexError):
                continue
        elif _name.startswith("logs_"):
            try:
                _idx = p + 1 + int(_name.split("logs_")[1])
            except (TypeError, ValueError, IndexError):
                continue
        elif _name.startswith("raw_dep_"):
            try:
                _idx = p + 1 + ksd + int(_name.split("raw_dep_")[1])
            except (TypeError, ValueError, IndexError):
                continue
        else:
            continue
        if _idx is not None and 0 <= _idx < n:
            out.add(_idx)
    return out


def _effect_row(i, labels, est, se, tval, pval, bound_idx=None, _pd_ok=True) -> str:
    """Single copy of the fixed/random effect row renderer (identical output).

    Jalan B: always render SE/z/p/CI/star as is; non-PD honesty lives in the
    banner above, not in forced NA cells. Missing SE (nan) still prints NA
    via _fmt naturally.
    """
    lo, hi = _ci(est[i], se[i])
    star = _sig(pval[i]) if i < len(pval) else ""
    return (
        f"  {str(labels[i]):<26}{_fmt(est[i]):>11}{_fmt(se[i]):>11}"
        f"{_fmt(tval[i], 3):>9}{_fmt(pval[i]):>11} {star:>2}"
        f"   [{_fmt(lo)},{_fmt(hi)}]"
    )


def format_beta(obj, width: int = 72) -> str:
    """Format a `BetastmmResult` as an English statsmodels-style report."""
    opt = obj._opt
    inf = obj._inf
    p = int(opt.get("p", 0))
    des = opt.get("des", {}) or {}
    n = opt.get("n", np.nan)
    m = des.get("m", np.nan)
    Tt = des.get("T", np.nan)

    labels = list(inf.get("param_labels", []) or [])
    est = np.asarray(inf.get("estimate", []), dtype=float)
    se = np.asarray(inf.get("se", []), dtype=float)
    tval = np.asarray(inf.get("tvalue", []), dtype=float)
    pval = np.asarray(inf.get("pvalue", []), dtype=float)
    bound_idx = _bound_rows(opt, labels)

    ln = []
    bar = "-" * width
    ln.append(bar)
    ln.append("Beta Spatio-Temporal Generalized Linear Mixed Model".center(width))
    ln.append(bar)
    ln.append(f"Model  : {opt.get('model', '')}")
    st = obj._structure()
    ln.append(f"Structure : {st.get('label', '')}")
    ln.append(f"Data   : n = {_fmt(n, 0)}, areas = {_fmt(m, 0)}, periods = {_fmt(Tt, 0)}")
    ln.append(f"Method : Laplace (INLA-style)")
    _pd_ok = ((opt.get("certificate") or {}).get("hessian_pd") is True)
    if not _pd_ok:
        ln.append("SE tidak didukung sertifikat Hessian PD; dilarang dikutip untuk inferensi")
    ln.append("")

    # Fixed effects (Jalan B: always show SE/z/p/CI; non-PD honesty via banner)
    ln.append("Fixed Effects:")
    ln.append(f"  {'':<2}{'':<24}{'coef':>11}{'std err':>11}{'z':>9}"
              f"{'P>|z|':>11}{'':>6}")
    nfe = min(len(labels), p)
    for i in range(nfe):
        ln.append(_effect_row(i, labels, est, se, tval, pval, bound_idx, _pd_ok))
    ln.extend(_footer(width))
    ln.append("Uji Wald acuan N(0,1); baris NA = parameter di boundary, tidak diinterpretasikan; efek-waktu 11 level lemah struktural.")

    # Random-effects / variance parameters (params beyond the p fixed coefs)
    if len(labels) > nfe:
        ln.append("")
        ln.append("Random effects / variance structure:")
        ln.append(f"  {'':<2}{'':<24}{'coef':>11}{'std err':>11}{'z':>9}"
                  f"{'P>|z|':>11}{'':>6}")
        for i in range(nfe, len(labels)):
            ln.append(_effect_row(i, labels, est, se, tval, pval, bound_idx, _pd_ok))
            # Two-scale display: raw optimizer params get their transformed
            # model-scale value right beneath (rho = rb*tanh(raw); the AR(1)
            # phi follows the model's frozen double squash).
            _lbl = str(labels[i])
            _v = float(est[i])
            if _lbl.startswith("raw_rho (atanh)"):
                _rb = des.get("rho_bound", 0.95)
                ln.append(f"  {'-> rho_S = rb*tanh(raw)':<26}"
                          f"{_fmt(_rb * np.tanh(_v)):>11}")
            elif _lbl.startswith("raw_phi_T (atanh)"):
                from stmmlib.beta.precision import effective_phi as _ephi
                _phiT = _ephi(_v, phi_bound=des.get("phi_bound", 0.95))
                ln.append(f"  {'-> phi_T (effective)':<26}{_fmt(_phiT):>11}")
            elif _lbl.startswith("raw_rho_gamma (atanh)"):
                _rg = des.get("rho_bound_g", 0.95)
                ln.append(f"  {'-> rho_gamma = rg*tanh(raw)':<26}"
                          f"{_fmt(_rg * np.tanh(_v)):>11}")
            elif _lbl.startswith("raw_phi_gamma (atanh)"):
                _pbg = des.get("phi_bound", 0.95)
                ln.append(f"  {'-> phi_gamma = pb*tanh(raw)':<26}"
                          f"{_fmt(_pbg * np.tanh(_v)):>11}")

    # Random-effects components (VarCorr)
    vc = obj.VarCorr()
    if vc:
        ln.append("")
        ln.append("Random effects components:")
        for k, v in vc.items():
            if isinstance(v, dict):
                detail = ", ".join(
                    f"{kk}={_fmt(vv) if isinstance(vv, float) else vv}"
                    for kk, vv in v.items()
                )
                ln.append(f"  {k}: {detail}")
            else:
                ln.append(f"  {k}: {_fmt(v)}")

    ln.append("")
    ln.extend(fmt_model_fit(
        {"logLik": obj.log_lik, "AIC": obj.AIC, "BIC": obj.BIC,
         "k": opt.get("k"), "n": n}, width=width))

    # Honest convergence
    om = opt.get("optim", {}) or {}
    iters = om.get("iters", om.get("nit", "?"))
    max_iter = om.get("max_iter", "?")
    degen = opt.get("degenerate", False)
    ln.append("")
    ln.append(f"Optimization: {_status_label(obj.status)}"
              + ("  [degenerate sigma-collapse!]" if degen else ""))
    ln.append(f"  iterations : {iters} / {max_iter}"
              f" (ftol = {om.get('ftol', '?')}, gtol = {om.get('gtol', '?')})")
    if om.get("message"):
        ln.append(f"  message    : {om['message'][:52]}")
    if len(se) and not np.any(np.isfinite(se)):
        ln.append("  SE/Wald/CI   : tidak dihitung (mode fast/medium tanpa Hessian)")

    # Convergence certificate
    cert = opt.get("certificate") or {}
    if cert:
        ln.append("")
        _valid = bool(cert.get("valid"))
        _opt_ok = bool(cert.get("optimum_valid"))
        if _valid:
            _clab = "VALID"
        elif _opt_ok:
            _clab = "OPTIMUM-OK (inferensi terbatas)"
        else:
            _clab = "INVALID"
        _gn = cert.get("grad_norm")
        _gn = float("nan") if _gn is None else _gn
        _pg = cert.get("projected_grad")
        _pg = float("nan") if _pg is None else _pg
        _ll = cert.get("ll_per_obs")
        _ll = float("nan") if _ll is None else _ll
        ln.append(f"Certificate: {_clab}"
                  f"  |  grad={_fmt(_gn, 3)}"
                  f"  |  projected_grad={_fmt(_pg, 3)}"
                  f"  |  hessian_PD={cert.get('hessian_pd')}"
                  f"  |  at_bound={cert.get('at_bound')}"
                  f"  |  ll/obs={_fmt(_ll, 4)}")
        for _w in cert.get("warnings", []):
            ln.append(f"  WARNING: {_w}")
    if om.get("penalty_refit_failed"):
        ln.append("  refit penalized ditolak, hasil unpenalized dipakai")
    if om.get("hessian_timeout"):
        ln.append("  Hessian dibatalkan budget waktu")
    if om.get("interrupted"):
        ln.append("  dihentikan user, hasil parsial")
    try:
        _n_fail = int(om.get("n_failed_starts", 0) or 0)
    except (TypeError, ValueError):
        _n_fail = 0
    if om.get("all_starts_abort") or _n_fail > 0:
        _fails = om.get("failed_starts", []) or []
        _detail = "; ".join(str(_f) for _f in _fails[:3])
        if _n_fail > len(_fails):
            _detail = f"{_n_fail} start gagal" + (f": {_detail}" if _detail else "")
        elif _detail:
            _detail = f"start gagal: {_detail}"
        else:
            _detail = f"start gagal: {_n_fail}"
        ln.append(f"  {_detail[:120]}")
    ln.append(bar)
    return "\n".join(ln)
