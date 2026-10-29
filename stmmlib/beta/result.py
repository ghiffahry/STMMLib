"""Result containers for Beta mixed model objects.

Provides BetastmmResult (random effects) and BetaResult (fixed only)
wrapping optim_results and inference results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class BetastmmResult:
    """Beta STMM result with random effects and inference."""

    def __init__(self, optim_results: dict, inference_results: dict):
        self._opt = optim_results
        self._inf = inference_results
        self._convergence = {
            "log_lik": optim_results.get("log_lik"),
            "AIC": optim_results.get("AIC"),
            "BIC": optim_results.get("BIC"),
            "optim": {"code": 0 if optim_results.get("converged") else 1},
        }

    @property
    def model(self) -> str:
        return self._opt.get("model", "")

    @property
    def coefficients(self) -> dict:
        """Named fixed-effect coefficients on the ORIGINAL (un-scaled) predictor
        scale. Internal optimization uses standardized predictors; this property
        back-transforms so tables read naturally."""
        return dict(zip(self._inf["param_labels"][:self._opt["p"]],
                        self.beta_original))

    @property
    def beta(self) -> np.ndarray:
        """Fixed-effect coefficients on the STANDARDIZED predictor scale (as
        used internally by opt["X"] @ beta). Use beta_original for the natural
        predictor scale."""
        return self._inf["estimate"][:self._opt["p"]]

    def _backscale(self) -> tuple:
        """Back-transform standardized beta/se to original predictor scale.

        Column 0 is the intercept (not scaled). For predictor j (= x_cols[j],
        standardized as (col-mean)/sd):
            beta_orig[j+1] = beta_std[j+1] / sd
            beta_orig[0]   = beta_std[0] - sum_j beta_std[j+1] * (mean/sd)
            se_orig[j+1]   = se_std[j+1] / sd   (intercept SE left as-is)
        Returns (beta_orig, se_orig).
        """
        p = self._opt["p"]
        beta_std = self._inf["estimate"][:p]
        se_std = self._inf["se_fixed"] if "se_fixed" in self._inf else self._inf["se"][:p]
        se_std = np.asarray(se_std, dtype=float)[:p]
        x_scale = self._opt.get("des", {}).get("x_scale", {}) or {}
        x_cols = self._opt.get("des", {}).get("x_cols", [])
        if not x_scale:
            return beta_std.copy(), se_std.copy()
        beta_orig = beta_std.copy()
        se_orig = se_std.copy()
        for j, c in enumerate(x_cols):
            mn, sd = x_scale.get(c, (0.0, 1.0))
            if sd <= 1e-12:
                continue
            beta_orig[j + 1] = beta_std[j + 1] / sd
            se_orig[j + 1] = se_std[j + 1] / sd
            beta_orig[0] -= beta_std[j + 1] * (mn / sd)
        return beta_orig, se_orig

    @property
    def beta_original(self) -> np.ndarray:
        """Fixed-effect coefficients on the original predictor scale."""
        b, _ = self._backscale()
        return b

    @property
    def se_original(self) -> np.ndarray:
        """Fixed-effect standard errors on the original predictor scale."""
        _, s = self._backscale()
        return s

    @property
    def phi(self) -> float:
        return self._opt.get("phi", np.exp(self._opt.get("logphi", 0.0)))

    @property
    def log_lik(self) -> float:
        return self._opt.get("log_lik", np.nan)

    # ponytail: camelCase alias; notebooks/scripts use logLik, lib stores log_lik
    @property
    def logLik(self) -> float:
        return self.log_lik

    @property
    def sigma_time(self) -> float:
        return self._opt.get("sigma_time", 1.0)

    @property
    def sigma_gamma(self) -> float | None:
        """Interaction SD (None for models without interaction)."""
        return self._opt.get("sigma_gamma", None)

    @property
    def AIC(self) -> float:
        return self._opt.get("AIC", np.nan)

    @property
    def BIC(self) -> float:
        return self._opt.get("BIC", np.nan)

    @property
    def convergence(self) -> dict:
        return self._convergence

    @property
    def status(self) -> str:
        """Honest 3-state fitness label: konvergen | max_iter | invalid."""
        return self._opt.get("status", "konvergen" if self._opt.get("converged") else "invalid")

    @property
    def certificate(self) -> dict:
        """Convergence certificate: valid/grad/boundary/hessian/ll-per-obs."""
        cert = self._opt.get("certificate")
        if cert is None:
            return {"valid": None, "warnings": []}
        return cert

    @property
    def inference(self) -> dict:
        return self._inf

    @property
    def design(self) -> dict:
        return self._opt.get("des", {})

    @property
    def mu_hat(self) -> np.ndarray:
        """Fitted conditional mean on response scale:
        mu = sigmoid(X beta + Z b_hat)."""
        if "X" not in self._opt:
            return np.full(self._opt.get("n", 0), np.nan)
        eta = self._opt["X"] @ self.beta
        if "Z" in self._opt and self._opt.get("b_hat") is not None:
            eta = eta + self._opt["Z"] @ self._opt["b_hat"]
        return 1.0 / (1.0 + np.exp(-eta))

    def predict(self, X=None, include_random=True, Z=None) -> np.ndarray:
        """Return fitted/predicted mean on response scale.

        Args:
            X: Optional fixed-effects design (n, p); defaults to fitted X.
            include_random: Add Z b_hat term when True.
            Z: Optional random-effects design for new data (n, p_r); if None
                and X rows match training Z, use training Z, else fixed only.
        """
        if X is None:
            return self.mu_hat
        eta = X @ self.beta
        Zr = Z if Z is not None else self._opt.get("Z")
        b = self._opt.get("b_hat")
        if include_random and Zr is not None and b is not None and np.asarray(Zr).shape[0] == X.shape[0]:
            eta = eta + np.asarray(Zr) @ np.asarray(b)
        return 1.0 / (1.0 + np.exp(-eta))

    def predict_interval(self, X=None, level: float = 0.95,
                         include_obs: bool = True, Z=None) -> dict:
        """P5: prediction interval on the response scale (logit-scale delta).

        Variance on the logit scale (approximate, documented):
          fixed  : x' V_bb x  (Wald vcov of beta from the Hessian)
          random : z' G z     (PRIOR block variances: sigma2_area for area
                   columns, marginal sigma2_time/(1-phi^2) for ar1 cell
                   columns else sigma2_time, sigma2_gamma for interaction;
                   prior-based = conservative, ignores shrinkage)
          obs    : 1 / (phi * mu * (1 - mu))  when include_obs
        Interval: sigmoid(logit(mu_hat) +/- z * se_logit), clipped to (0,1).

        Returns dict with mean, lower, upper, se_logit, width_mean.
        Z: optional random-effects design for new data (n, p_r); if None
            and X rows match training Z, use training Z, else fixed only.
        """
        from stmmlib.beta.structure import stmm_beta_structure
        o = self._opt
        des = o.get("des", {})
        s = stmm_beta_structure(o["model"])
        in_sample = X is None
        X = np.asarray(o["X"], float) if X is None else np.asarray(X, float)
        Zr = Z if Z is not None else o.get("Z")
        Zr = np.asarray(Zr, float) if Zr is not None else None
        b_hat = o.get("b_hat")
        if b_hat is not None:
            b_hat = np.asarray(b_hat, float)
        beta = np.asarray(o["beta_hat"], float)
        phi = float(np.exp(o["logphi"]))
        logs = np.asarray(o["logs"], float)
        raw = np.asarray(o.get("raw_dep", []), float)

        use_random = Zr is not None and b_hat is not None and Zr.shape[0] == X.shape[0]
        eta = X @ beta + (Zr @ b_hat if use_random else 0.0)
        mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1 - 1e-12)

        # fixed part
        inf = getattr(self, "inference", None) or {}
        V = inf.get("vcov")
        p = len(beta)
        Vb = np.asarray(V[:p, :p], float) if V is not None \
            else np.zeros((p, p))
        var_fixed = np.einsum("ij,jk,ik->i", X, Vb, X)

        # random part: prior variance per column of Z, by block
        var_random = np.zeros(X.shape[0])
        if use_random:
            blocks = des.get("blocks", {})
            s2a = float(np.exp(2 * logs[0])) if len(logs) > 0 else 0.0
            # BUG-B3 fix: use the model's EFFECTIVE phi (double squash); the
            # single-squash value overstated cell_var by ~37% at raw=1.
            from stmmlib.beta.precision import effective_phi
            _pb = des.get("phi_bound", 0.95)
            phi_t = effective_phi(float(raw[1 if s["area"] in ("sar", "seblup_rao_yu") else 0]),
                                  phi_bound=_pb) \
                if (s["time"] == "ar1" and len(raw) > (1 if s["area"] in ("sar", "seblup_rao_yu") else 0)) else 0.0
            s2t = float(np.exp(2 * logs[1])) if len(logs) > 1 else 0.0
            s2g = float(np.exp(2 * logs[-1])) if (len(blocks.get("interaction", [])) and len(logs) >= 3) else 0.0
            cell_var = s2t / max(1 - phi_t ** 2, 1e-6) if s["time"] == "ar1" else s2t
            per_col = []
            for key in ("area", "prov"):
                per_col += [s2a] * len(blocks.get(key, []))
            if "kab" in blocks:
                per_col += [float(np.exp(2 * logs[1]))] * len(blocks["kab"])
            for key in ("area_time", "cell", "time"):
                per_col += [cell_var] * len(blocks.get(key, []))
            per_col += [s2g] * len(blocks.get("interaction", []))
            per_col = np.asarray(per_col, float)
            if len(per_col) == Zr.shape[1]:
                var_random = (Zr ** 2) @ per_col

        se_logit = np.sqrt(var_fixed + var_random +
                           (1.0 / (phi * mu * (1 - mu)) if include_obs else 0.0))
        z = 1.959963985 if level == 0.95 else float(
            __import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.5 + level / 2))
        lg = np.log(mu / (1 - mu))
        lo = 1.0 / (1.0 + np.exp(-(lg - z * se_logit)))
        hi = 1.0 / (1.0 + np.exp(-(lg + z * se_logit)))
        out = {
            "mean": mu, "lower": np.clip(lo, 0.0, 1.0),
            "upper": np.clip(hi, 0.0, 1.0),
            "se_logit": se_logit,
            "width_mean": float(np.mean(hi - lo)),
            "include_obs": bool(include_obs), "level": level,
        }
        if not in_sample and not use_random:
            out["note"] = "tanpa Z baru: interval efek-tetap saja"
        return out

    # ── EBP: Empirical Best Predictor (Beta) ────────────────────────────────
    def posterior_draws(self, n_draws: int = 200, seed: int = 0,
                        chunk: int = 100):
        """Yield blocks of random-effect draws from N(b_hat, H_b^-1).

        H_b = Z'diag(w)Z + Q at the final fit (Laplace approximation), stored
        as a Cholesky factor when the fit ran with
        BetaFitConfig(compute_posterior=True).
        """
        opt = self._opt
        L = opt.get("H_b_chol")
        if L is None:
            raise ValueError(
                "Posterior curvature not stored: refit with "
                "BetaFitConfig(compute_posterior=True).")
        from scipy.linalg import solve_triangular
        L = np.asarray(L, dtype=float)
        b = np.asarray(opt["b_hat"], dtype=float)
        rng = np.random.default_rng(seed)
        done = 0
        while done < n_draws:
            m = min(chunk, n_draws - done)
            z = rng.standard_normal((m, b.size))
            yield b + solve_triangular(L.T, z.T, lower=False).T
            done += m

    def ebp(self, n_draws: int = 200, seed: int = 0, chunk: int = 100,
            return_se: bool = False):
        """Empirical Best Predictor of the fitted conditional means.

        EBP_i = E[mu_i | y] = (1/B) sum_b sigmoid(x_i'beta + z_i'b),
        b ~ N(b_hat, H_b^-1) under the Laplace approximation, with b_hat as
        the posterior mean. Differs from mu_hat (plug-in at the posterior
        mode) by O(posterior variance) on the logit scale.

        Returns the EBP vector, or (ebp, MC standard error) if return_se.
        """
        opt = self._opt
        if "X" not in opt or opt.get("b_hat") is None:
            raise ValueError("EBP needs a fitted mixed model.")
        from scipy.special import expit
        X = np.asarray(opt["X"], dtype=float)
        Z = np.asarray(opt["Z"], dtype=float)
        eta = (X @ np.asarray(opt["beta_hat"], dtype=float)
               + Z @ np.asarray(opt["b_hat"], dtype=float))
        s1 = np.zeros(len(eta))
        s2 = np.zeros(len(eta))
        for bd in self.posterior_draws(n_draws, seed, chunk):
            mu = expit(eta[:, None] + Z @ bd.T)
            s1 += mu.sum(1)
            s2 += (mu ** 2).sum(1)
        mean = s1 / n_draws
        if not return_se:
            return mean
        var = np.maximum(s2 / n_draws - mean ** 2, 0.0)
        return mean, np.sqrt(var / n_draws)

    def ebp_newdata(self, newdata, area_col: str = "area",
                    time_col: str = "time", n_draws: int = 200,
                    seed: int = 0, chunk: int = 100):
        """EBP for new observations on areas present in the training data.

        The area effect is drawn from the fitted posterior. A time period not
        seen in training is drawn from the fitted time prior: AR(1) one-step
        recursion from the last training period, or the iid prior
        N(0, sigma_time^2). Areas absent from training have no posterior and
        are rejected; for those the fixed part (predict) is the only
        available predictor.

        Returns (ebp, MC standard error).
        """
        opt = self._opt
        des = opt.get("des", {}) or {}
        blocks = des.get("blocks", {}) or {}
        if "prov" in blocks:
            raise NotImplementedError(
                "EBP for newdata: nested models unsupported.")
        s = self._structure()
        if s.get("interaction", "none") != "none":
            raise NotImplementedError(
                "EBP for newdata: interaction models unsupported.")
        if not hasattr(self, "_area_labels") or not self._area_labels:
            raise ValueError("EBP for newdata: area labels missing on the fit.")
        if not hasattr(self, "_time_labels") or not self._time_labels:
            raise ValueError("EBP for newdata: time labels missing on the fit.")

        x_cols = list(des.get("x_cols", []))
        missing = [c for c in x_cols if c not in newdata.columns]
        if missing:
            raise ValueError(f"newdata missing predictors: {missing}")
        n_new = len(newdata)
        X = np.column_stack([np.ones(n_new)]
                            + [newdata[c].to_numpy(dtype=float) for c in x_cols])
        scale = des.get("x_scale", {}) or {}
        for j, c in enumerate(x_cols):
            mn, sd = scale.get(c, (0.0, 1.0))
            if sd > 1e-12:
                X[:, j + 1] = (X[:, j + 1] - mn) / sd
        beta = np.asarray(opt["beta_hat"], dtype=float)

        m, T = int(des["m"]), int(des["T"])
        name_to_code = {v: k for k, v in self._area_labels.items()}
        a = newdata[area_col].map(name_to_code)
        if a.isna().any():
            bad = sorted(newdata.loc[a.isna(), area_col].astype(str).unique())[:5]
            raise ValueError(
                f"EBP newdata: areas not in training data: {bad}")
        a = a.to_numpy(dtype=int)

        code_of_time = {v: k for k, v in self._time_labels.items()}
        t_raw = newdata[time_col]
        t = t_raw.map(code_of_time)
        for k, v in enumerate(sorted(set(t_raw[t.isna()]))):
            t.loc[t_raw == v] = T + k
        t = t.to_numpy(dtype=int)

        from scipy.special import expit
        eta_fix = X @ beta
        global_time = s.get("time_scope") == "global"
        ar1 = s.get("time") == "ar1"
        logs = np.asarray(opt.get("logs", []), dtype=float)
        sigma_t = float(np.exp(logs[1])) if logs.size > 1 else 0.0
        phi_t = 0.0
        if ar1:
            from stmmlib.beta.precision import effective_phi
            raw = np.asarray(opt.get("raw_dep", []), dtype=float)
            i_phi = 1 if s.get("area") in ("sar", "seblup_rao_yu") else 0
            phi_t = float(effective_phi(
                float(raw[i_phi]), phi_bound=des.get("phi_bound", 0.95)))

        idx_known = np.where(t < T)[0]
        idx_fut = np.where(t >= T)[0]
        rng_eps = np.random.default_rng(seed + 101)
        s1 = np.zeros(n_new)
        s2 = np.zeros(n_new)
        for bd in self.posterior_draws(n_draws, seed, chunk):
            B = bd.shape[0]
            eta = np.repeat(eta_fix[:, None], B, axis=1)
            eta += bd[:, a].T                      # area effect (posterior draw)
            if idx_known.size:
                cols = (m + t[idx_known] if global_time
                        else m + a[idx_known] * T + t[idx_known])
                eta[idx_known] += bd[:, cols].T    # observed time effect
            for code in np.unique(t[idx_fut]):
                sel = idx_fut[t[idx_fut] == code]
                if ar1:
                    prev = ((m + T - 1) if global_time
                            else m + a[sel] * T + (T - 1))
                    u = bd[:, prev].T.copy()
                    for _ in range(T, int(code) + 1):
                        u = phi_t * u + rng_eps.normal(0.0, sigma_t, u.shape)
                else:
                    u = rng_eps.normal(0.0, sigma_t, (sel.size, B))
                eta[sel] += u
            mu = expit(eta)
            s1 += mu.sum(1)
            s2 += (mu ** 2).sum(1)
        mean = s1 / n_draws
        var = np.maximum(s2 / n_draws - mean ** 2, 0.0)
        return mean, np.sqrt(var / n_draws)


    # ── Accessor API (coef/fixef/ranef/VarCorr/...) ─────────────────────────
    def coef(self) -> dict:
        """Named fixed-effect coefficients (intercept + predictors)."""
        return dict(self.coefficients)

    def fixef(self) -> dict:
        """Alias for fixed effects (R-style)."""
        return self.coef()

    def ranef(self) -> dict:
        """Predicted random effects by level, split as area / cell."""
        opt = self._opt
        b = opt.get("b_hat")
        if b is None:
            return {}
        des = opt.get("des", {})
        blocks = des.get("blocks", {})
        na = len(blocks.get("area", []))
        nc = len(blocks.get("area_time", blocks.get("time", [])))
        out = {"area": None if na == 0 else self._rename_level(b[:na], des.get("area_labels"))}
        if nc > 0:
            out["area_time"] = b[na:na + nc] if len(b) >= na + nc else b[na:]
        return out

    @staticmethod
    def _rename_level(v, labels):
        if not labels:
            return v
        names = list(labels)
        if len(names) > len(v):
            names = names[:len(v)]
        return dict(zip(names, v))

    def VarCorr(self) -> dict:
        """Random-effect (co)variance components as R-style dict.

        Uses the fitted precision blocks: G = Q^{-1}. The spatial SAR block is
        reported as its formula + dimensions (never the full dense matrix).
        """
        opt = self._opt
        des = opt.get("des", {})
        blocks = des.get("blocks", {})
        na = len(blocks.get("area", []))
        logs = opt.get("logs", np.zeros(2))
        raw = opt.get("raw_dep", np.array([]))
        s = self._structure()
        out = {}

        is_nested = "prov" in blocks
        if is_nested:
            n_prov = len(blocks.get("prov", []))
            n_kab = len(blocks.get("kab", []))
            sig_p2 = np.exp(2.0 * logs[0]) if len(logs) > 0 else 1.0
            sig_k2 = np.exp(2.0 * logs[1]) if len(logs) > 1 else 1.0
            out["prov"] = {"type": "IID (nested)", "sigma_prov^2": float(sig_p2),
                           "n_prov": int(n_prov)}
            out["kab"] = {"type": "IID (nested)", "sigma_kab^2": float(sig_k2),
                          "n_kab": int(n_kab)}
        elif s["area"] in ("sar", "nested_sar", "seblup_rao_yu"):
            # Display rho with the ACTUAL spectral bound used by the model
            # (des["rho_bound"] = 0.95/max|eig(W)|), not the gamma bound.
            rb_disp = des.get("rho_bound", 0.95)
            rho = rb_disp * np.tanh(raw[0]) if len(raw) > 0 else 0.0
            sig_v2 = np.exp(2.0 * logs[0]) if len(logs) > 0 else 1.0
            _Wm = opt.get("W", getattr(self, "_W", None))
            try:
                _Wa = _Wm.toarray() if hasattr(_Wm, "toarray") else np.asarray(_Wm, dtype=float)
                _nr, _nc = _Wa.shape
                _sym = bool(np.allclose(_Wa, _Wa.T, atol=1e-8))
                _isl = int(np.sum(_Wa.sum(axis=1) == 0))
                _wd = f"{_nr}x{_nc} {'sym' if _sym else 'asym'}"
                if _isl:
                    _wd += f", islands={_isl}"
            except Exception:
                _wd = f"{na}x{na}"
            out["area"] = {
                "type": "SAR",
                "raw_rho (atanh)": float(raw[0]) if len(raw) > 0 else 0.0,
                "rho": float(rho),
                "sigma_v^2": float(sig_v2),
                "G": f"sigma_v^2 (I - rho W)^-1 (I - rho W)^-T",
                "W_dim": _wd,
            }
        elif s["area"] == "iid":
            sig_a2 = np.exp(2.0 * logs[0]) if len(logs) > 0 else 1.0
            out["area"] = {"type": "IID", "sigma_v^2": float(sig_a2), "n_area": int(na)}

        if s["time"] == "ar1":
            _ph_i = 0 if is_nested else (1 if s["area"] in ("sar", "nested_sar", "seblup_rao_yu") else 0)
            # Display the EFFECTIVE phi via the single helper (double squash).
            from stmmlib.beta.precision import effective_phi
            _pb = des.get("phi_bound", 0.95)
            phi = effective_phi(float(raw[_ph_i]), phi_bound=_pb) if len(raw) > _ph_i else 0.0
            _eta_i = 2 if is_nested else 1
            sig_eta2 = np.exp(2.0 * logs[_eta_i]) if len(logs) > _eta_i else 1.0
            out["time"] = {
                "type": "AR(1)",
                "raw_phi_T (atanh)": float(raw[_ph_i]) if len(raw) > _ph_i else 0.0,
                "phi": float(phi),
                "sigma_eta^2 (innovation)": float(sig_eta2),
                "sigma_marginal^2": float(sig_eta2 / max(1 - phi ** 2, 1e-12)),
                "scope": s.get("time_scope", ""),
            }
        else:
            sig_t2 = np.exp(2.0 * logs[1]) if len(logs) > 1 else 1.0
            out["time"] = {"type": "IID", "sigma^2": float(sig_t2)}
        out["precision (kappa = phi)"] = float(self.phi)
        return out

    def residuals(self, type="response"):
        """Residuals. type: 'response' | 'dunn_smyth' | 'conditional' | 'marginal'."""
        opt = self._opt
        y = opt.get("y")
        if y is None:
            raise ValueError("Response not found in result.")
        if type == "response":
            return y - self.mu_hat
        if type == "dunn_smyth":
            from stmmlib.beta.diagnostics import compute_beta_random_resid
            return compute_beta_random_resid(self).get("resid_random")
        if type == "conditional":
            from stmmlib.beta.diagnostics import compute_beta_cond_resid
            return compute_beta_cond_resid(self).get("resid_cond")
        if type == "marginal":
            from stmmlib.beta.diagnostics import compute_beta_marginal_resid
            return compute_beta_marginal_resid(self).get("resid_marginal")
        raise ValueError(f"Unknown residual type {type!r}")

    def _structure(self) -> dict:
        from stmmlib.beta.structure import stmm_beta_structure
        try:
            return stmm_beta_structure(self.model)
        except Exception:
            return {}

    # ── Rich printable summary (English, statsmodels style) ────────────────
    def summary(self, width: int = 72) -> str:
        """Format an English, statsmodels-style summary report."""
        from stmmlib.output_format import format_output
        return format_output(self, width=width)

    def summary_dict(self) -> dict:
        """Programmatic summary (meta + effects + fit), sans printed strings."""
        return {
            "model": self.model,
            "structure": self._structure(),
            "n": self._opt.get("n"), "p": self._opt.get("p"), "k": self._opt.get("k"),
            "converged": self._opt.get("converged"), "degenerate": self._opt.get("degenerate"),
            "optim": self._opt.get("optim", {}),
            "logLik": self.log_lik, "AIC": self.AIC, "BIC": self.BIC,
            "VarCorr": self.VarCorr(),
            "fixed": list(zip(self._inf.get("param_labels", []),
                              self._inf.get("estimate", []))),
        }

    def __str__(self) -> str:
        try:
            return self.summary()
        except Exception:
            return self.__repr__()

    def get_selected(self, design_matrix: dict) -> np.ndarray:
        """Compute eta = X @ beta using design matrix from data."""
        if design_matrix is not None and "X" in design_matrix:
            return design_matrix["X"] @ self.beta
        return np.full(self._opt["n"], np.nan)

    def __repr__(self) -> str:
        return (
            f"BetastmmResult(model={self.model!r}, "
            f"logLik={self.log_lik:.2f}, "
            f"AIC={self.AIC:.2f}, "
            f"BIC={self.BIC:.2f})"
        )


class BetaResult:
    """Beta fixed-effects-only result."""

    def __init__(self, optim_results: dict):
        self._opt = optim_results

    @property
    def coefficients(self) -> dict:
        beta_hat = self._opt.get("beta_hat", np.array([]))
        return {"(Intercept)": beta_hat[0] if len(beta_hat) > 0 else np.nan}

    @property
    def phi(self) -> float:
        return self._opt.get("phi", 1.0)

    @property
    def logLik(self) -> float:
        return self._opt.get("log_lik", np.nan)

    @property
    def AIC(self) -> float:
        return self._opt.get("AIC", np.nan)

    @property
    def BIC(self) -> float:
        return self._opt.get("BIC", np.nan)

    def __repr__(self) -> str:
        return f"BetaResult(phi={self.phi:.2f}, logLik={self.logLik:.2f})"
