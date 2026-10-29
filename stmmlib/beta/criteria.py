"""Conditional selection criteria for Beta STMM: cAIC + conditional R2.

Single source of truth for the `beta_criteria` helper previously inlined
in the thesis notebooks (5 copies). Import it instead of copying::

    from stmmlib.beta.criteria import beta_criteria

Formulas (validated):
    cAIC (Huber et al. 2016):  -2 * ll_cond + 2 * df_hat
        df_hat = tr(H_pen^-1 @ H_unpen), the Liang-Wu-Zou (2008) bias
        correction. H_pen/_unpen use the Fisher weight w = phi*mu*(1-mu).
    cR2 (Nakagawa & Schielzeth 2013, latent-scale working version):
        vf = var(X beta); vre = mean(diag(Z G Z')), G = Q^-1;
        vresid = mean(mu(1-mu)/(1+phi)).

Fixes vs the old notebook copy:
    - beta log-density SUBTRACTS ln B(a,b) (the old copy added it, which
      produced M1 cAIC = +4.8jt instead of ~-11.1k).
    - mu is always recomputed from the fitted (beta, b_hat); a stored
      mu_hat is never trusted.
    - area SAR uses des["rho_bound"] (spectral bound), not the gamma bound.

Guards (refuse garbage instead of printing it):
    - df_hat must lie in [0, p+q] (theorem: H_pen = H_unpen + blkdiag(0,Q)).
    - ll_cond sanity vs the Laplace marginal (loose heuristic tolerance).
    - finite vre.

Scope: flat (non-nested) models M1-M5, M8-M12. Nested fits (M6/M7/M13/M14)
return all-NaN fields -- their [prov, kab, cell] precision needs its own
builder (documented follow-up, not silent garbage).
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaln


def _as_opt(fit):
    return fit._opt if hasattr(fit, "_opt") else fit


def beta_criteria(fit, W=None):
    """Compute cAIC, df_hat, ll_cond, cR2 for a fitted Beta STMM.

    Args:
        fit: BetastmmResult (or raw opt dict from beta_laplace_structural).
        W: spatial weight matrix override (default: opt/des stored W).

    Returns:
        dict with cAIC, df_hat, ll_cond, cR2_marginal, cR2_conditional,
        var_fixed, var_random_latent, var_resid_beta; NaN fields when the
        inputs are inconsistent (never a garbage number). None only when
        required arrays are missing or an unexpected error occurs.
    """
    if fit is None:
        return None
    try:
        opt = _as_opt(fit)
        X = opt.get("X")
        Z = opt.get("Z")
        y = opt.get("y")
        des = opt.get("des", {}) or {}
        model = opt.get("model", "")
        if X is None or Z is None or y is None:
            return None

        beta = fit.beta if hasattr(fit, "_opt") else opt.get("beta_hat")
        try:
            phi = float(fit.phi) if hasattr(fit, "_opt") else float(opt.get("phi", 20.0))
        except Exception:
            phi = float(np.exp(np.asarray(opt.get("logphi", [np.log(20.0)]))).ravel()[0])

        logs = np.asarray(opt.get("logs", np.zeros(2)), dtype=float).ravel()
        raw_dep = np.asarray(opt.get("raw_dep", np.array([])), dtype=float).ravel()
        bh = opt.get("b_hat")
        if beta is None or bh is None:
            return None
        beta = np.asarray(beta, dtype=float).ravel()
        bh = np.asarray(bh, dtype=float).ravel()

        from stmmlib.beta.structure import stmm_beta_structure
        s = stmm_beta_structure(model)
        if s["area"] in ("nested_iid", "nested_sar"):
            return _nan_result()  # nested precision: dedicated builder needed

        blocks = des.get("blocks", {}) or {}
        n_area = len(blocks.get("area", []))
        n_time = len(blocks.get("area_time", blocks.get("time", [])))
        # Area spectral bound (NOT the gamma bound) for the SAR display mapping
        rb = des.get("rho_bound", des.get("rho_bound_g", 0.95))

        blocks_Q = []
        if s["area"] in ("iid",):
            Q_a = np.exp(-2.0 * logs[0]) * np.eye(n_area)
        elif s["area"] in ("sar", "nested_sar", "seblup_rao_yu"):
            rho = rb * np.tanh(raw_dep[0]) if len(raw_dep) > 0 else 0.0
            # BUG-B4 fix: explicit W argument wins; opt["W"] may be None even
            # when the key exists (old expression silently ignored use_W).
            W_arr = W if W is not None else opt.get("W")
            if W_arr is None:
                W_arr = des.get("W_gamma")
            if W_arr is None:
                W_arr = np.eye(n_area)
            elif hasattr(W_arr, "toarray"):
                W_arr = W_arr.toarray()
            W_arr = np.asarray(W_arr, dtype=float)[:n_area, :n_area]
            A = np.eye(n_area) - rho * W_arr
            Q_a = np.exp(-2.0 * logs[0]) * (A.T @ A)
        else:
            Q_a = np.eye(n_area)
        blocks_Q.append(Q_a)

        area_specific = s.get("time_scope") == "area_specific"
        if s["time"] == "ar1" and n_time > 0:
            # Frozen model mapping: ar1_precision squashes phi_t again
            # (pb*tanh(pb*tanh(raw))); keep the tridiagonal form here.
            pb = des.get("phi_bound", 0.95)
            phi_t = pb * np.tanh(raw_dep[1]) if len(raw_dep) > 1 else 0.0
            sigma2_inv = np.exp(-2.0 * (logs[1] if len(logs) > 1 else 0.0))
            if area_specific:
                from stmmlib.beta.precision import ar1_precision as _ar1p
                from stmmlib._backend import Backend as _B
                from scipy.sparse import block_diag as _spbd
                T_loc = int(des.get("T", len(blocks.get("time", []))))
                TQ = _ar1p(max(T_loc, 1), logs[1] if len(logs) > 1 else 0.0, phi_t, _B(gpu=False),
                           phi_bound=pb)
                if hasattr(TQ, "toarray"):
                    TQ = TQ.toarray()
                Q_t = _spbd([TQ] * n_area, format="csr").toarray()
            else:
                di = np.full(n_time, 1.0 + phi_t ** 2)
                di[0] = di[-1] = 1.0
                Q_t = sigma2_inv * (np.diag(di)
                                    + np.diag(np.full(n_time - 1, -phi_t), 1)
                                    + np.diag(np.full(n_time - 1, -phi_t), -1))
            blocks_Q.append(Q_t)
        elif n_time > 0:
            Q_t = np.exp(-2.0 * logs[1]) * np.eye(n_time) if len(logs) > 1 else np.eye(n_time)
            blocks_Q.append(Q_t)

        from scipy.sparse import block_diag as _spblockdiag
        Q_base = _spblockdiag(blocks_Q)
        if hasattr(Q_base, "toarray"):
            Q_base = Q_base.toarray()

        n_gamma = len(blocks.get("interaction", []))
        if n_gamma > 0 and len(logs) > 2:
            # Working simplification (as in the validated notebook): diagonal
            # gamma block; gamma dependence params not propagated here.
            Q_g = np.exp(-2.0 * logs[2]) * np.eye(n_gamma)
            Q_full = np.block([[Q_base, np.zeros((Q_base.shape[0], n_gamma))],
                               [np.zeros((n_gamma, Q_base.shape[0])), Q_g]])
        else:
            Q_full = Q_base

        q = Z.shape[1]
        if Q_full.shape[0] < q:
            pad = q - Q_full.shape[0]
            Q_full = np.block([[Q_full, np.zeros((Q_full.shape[0], pad))],
                               [np.zeros((pad, Q_full.shape[0])), np.eye(pad) * 1e-10]])
        elif Q_full.shape[0] > q:
            Q_full = Q_full[:q, :q]

        # Conditional mean from the fitted (beta, b_hat) -- never stored mu_hat
        mu = np.clip(1 / (1 + np.exp(-(X @ beta + Z @ bh))), 1e-10, 1 - 1e-10)
        w = phi * mu * (1 - mu)
        A11 = X.T @ (X * w[:, None])
        A12 = X.T @ (Z * w[:, None])
        A22 = Z.T @ (Z * w[:, None]) + Q_full
        Hm = np.block([[A11, A12], [A12.T, A22]])
        M22 = Z.T @ (Z * w[:, None])
        Mm = np.block([[A11, A12], [A12.T, M22]])

        # df guard: nested cells-in-areas make Z'WZ rank-deficient; without
        # enough Q shrinkage the raw solve explodes. df in [0, p+q] or NaN.
        p_dof = X.shape[1] + Z.shape[1]
        try:
            df_hat = float(np.trace(np.linalg.solve(Hm, Mm)))
            if not np.isfinite(df_hat) or df_hat < 0 or df_hat > p_dof:
                df_hat = np.nan
        except np.linalg.LinAlgError:
            df_hat = np.nan

        # Conditional log-lik: (a-1)ln y + (b-1)ln(1-y) MINUS ln B(a,b)
        a = mu * phi
        b = (1 - mu) * phi
        ll_cond = float(np.sum((a - 1) * np.log(np.clip(y, 1e-15, 1.0)) +
                               (b - 1) * np.log(np.clip(1 - y, 1e-15, 1.0)) -
                               betaln(a, b)))
        # ll guard (loose heuristic): far below the Laplace marginal means
        # stale inputs -- refuse garbage instead of printing it. Threshold
        # max(10, 1%|ll|): tight enough to catch miscalibrated mu (measured
        # stale M1 case: ll_cond 16.5 vs ll 47.9 -> gap 31), loose enough for
        # Laplace-vs-mode wiggle on heavy-shrink fits.
        ll_marg = float(opt.get("log_lik", np.nan))
        if (not np.isfinite(ll_cond) or not np.isfinite(df_hat)
                or (np.isfinite(ll_marg)
                    and ll_cond < ll_marg - max(10.0, 0.01 * abs(ll_marg)))):
            out = _nan_result()
            out["df_hat"], out["ll_cond"] = df_hat, ll_cond
            return out
        caic = -2 * ll_cond + 2 * df_hat

        # cR2 (Nakagawa working version)
        try:
            G = np.linalg.inv(Q_full)
            vre = np.sum((Z @ G) * Z) / len(y)
            if not np.isfinite(vre):
                vre = np.nan
        except np.linalg.LinAlgError:
            vre = np.nan
        vf = float(np.var(X @ beta))
        vresid = float(np.mean(mu * (1 - mu) / (1 + phi)))
        tot = vf + vre + vresid
        r2m = vf / tot if tot > 0 else np.nan
        r2c = (vf + vre) / tot if tot > 0 else np.nan

        return {
            "cAIC": caic, "df_hat": df_hat, "ll_cond": ll_cond,
            "cR2_marginal": r2m, "cR2_conditional": r2c,
            "var_fixed": vf, "var_random_latent": vre, "var_resid_beta": vresid,
        }
    except Exception:
        return None


def _nan_result():
    nan = float("nan")
    return {"cAIC": nan, "df_hat": nan, "ll_cond": nan,
            "cR2_marginal": nan, "cR2_conditional": nan,
            "var_fixed": nan, "var_random_latent": nan,
            "var_resid_beta": nan}
