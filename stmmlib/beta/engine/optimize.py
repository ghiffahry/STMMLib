"""Outer L-BFGS-B start loop for the Beta Laplace engine."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from stmmlib.beta.precision import PrecisionShapeError
from stmmlib.beta.beta_nll import (
    beta_nll_curvature,
    _beta_nll_score,
    _beta_grad_pieces,
)
from stmmlib.beta.schur import _blocked_hinv_terms, _blocked_pen_terms
from stmmlib.beta.mode import (
    beta_mode_newton_blocked,
    beta_solve_mode,
    beta_solve_mode_blocked,
)
from stmmlib.beta.numerics import (
    _num_hessian,
    _num_hessian_grad,
    _projected_grad_norm,
    _os_env_debug,
)


def optimize_starts(ctx):
    W = ctx["W"]
    X = ctx["X"]
    Z = ctx["Z"]
    _eval_count = ctx["_eval_count"]
    _get_start = ctx["_get_start"]
    _jac_ok = ctx["_jac_ok"]
    _laplace_grad = ctx["_laplace_grad"]
    _neg_marginal_l = ctx["_neg_marginal_l"]
    _state = ctx["_state"]
    bounds = ctx["bounds"]
    data = ctx["data"]
    des = ctx["des"]
    ftol = ctx["ftol"]
    gtol = ctx["gtol"]
    ksd = ctx["ksd"]
    max_iter = ctx["max_iter"]
    model = ctx["model"]
    n = ctx["n"]
    n_starts = ctx["n_starts"]
    p = ctx["p"]
    param_dim = ctx["param_dim"]
    q_dep = ctx["q_dep"]
    seed = ctx["seed"]
    start_b = ctx["start_b"]
    start_theta = ctx["start_theta"]
    y = ctx["y"]
    best_val = np.inf
    best_param = None
    best_meta = {}

    # Warm start (F2): caller supplies a full parameter vector (from a previous
    # cheaper/warm-start model map), bypassing OLS + random perturbed starts.
    if start_theta is not None:
        starts = [np.asarray(start_theta, dtype=np.float64)]
    else:
        # Perturb random restarts; seeded for reproducibility (F3).
        rng = np.random.default_rng(seed) if seed is not None else np.random
        starts = [_get_start()]
        for _ in range(max(n_starts - 1, 0)):
            s0 = starts[0].copy()
            s0[p] += rng.normal(0, 0.5)  # perturb logphi
            s0[p + 1: p + 1 + ksd] += rng.normal(0, 0.15, ksd)
            s0[:p] += rng.normal(0, 0.05, p)  # perturb beta (explore basins)
            if q_dep > 0:
                s0[p + 1 + ksd:] += rng.normal(0, 0.15, q_dep)  # perturb raw dep
            starts.append(s0)

    failed_starts = []
    _interrupted = False
    _abort_val = np.inf
    _abort_param = None
    _abort_meta = None
    try:
        for i, param0 in enumerate(starts):
            param0 = np.asarray(param0, dtype=np.float64)
            if param0.shape != (param_dim,) or not np.all(np.isfinite(param0)):
                failed_starts.append(f"start {i} ditolak: non-finite")
                continue
            _state["b"] = np.zeros(Z.shape[1], dtype=np.float64)
            if start_b is not None:
                _b0 = np.asarray(start_b, dtype=np.float64).ravel()
                if _b0.size != Z.shape[1]:
                    raise ValueError(
                        f"start_b has {_b0.size} entries but the model has "
                        f"{Z.shape[1]} random-effect modes.")
                _state["b"] = _b0.copy()
            _state.pop("cache", None)
            try:
                v0 = _neg_marginal_l(param0)
            except Exception as e:  # noqa: BLE001
                failed_starts.append(f"start {i} ditolak: {type(e).__name__}: {e}")
                continue
            if not np.isfinite(v0) or v0 >= 1e19:
                failed_starts.append(f"start {i} ditolak: objective non-finite ({v0})")
                continue
            # Pre-check gradien di titik start: tolak start kasar sebelum optimasi.
            try:
                if _jac_ok:
                    _gn_pre = float(np.linalg.norm(_laplace_grad(param0)))
                else:
                    _eps_pre = 1e-6
                    _gp = []
                    for _k in range(param_dim):
                        _dp = param0.copy(); _dm = param0.copy()
                        _dp[_k] += _eps_pre; _dm[_k] -= _eps_pre
                        _vp = _neg_marginal_l(_dp)
                        _vm = _neg_marginal_l(_dm)
                        if (not np.isfinite(_vp) or not np.isfinite(_vm)
                                or _vp >= 1e19 or _vm >= 1e19):
                            raise FloatingPointError("FD start non-finite")
                        _gp.append((_vp - _vm) / (2.0 * _eps_pre))
                    _gn_pre = float(np.linalg.norm(np.asarray(_gp, dtype=np.float64)))
            except Exception:
                _gn_pre = float("inf")
            if not np.isfinite(_gn_pre) or _gn_pre > 1e6:
                failed_starts.append(
                    f"start {i} ditolak: grad start non-finite/ledak ({_gn_pre:.3e})")
                continue
            _state["b"] = np.zeros(Z.shape[1], dtype=np.float64)
            if start_b is not None:
                _b0 = np.asarray(start_b, dtype=np.float64).ravel()
                if _b0.size != Z.shape[1]:
                    raise ValueError(
                        f"start_b has {_b0.size} entries but the model has "
                        f"{Z.shape[1]} random-effect modes.")
                _state["b"] = _b0.copy()
            _state.pop("cache", None)
            try:
                # maxfun guards total F,G evals (default 15000). Stiff log-scale
                # params eat many evals per line search; cap it generously so the
                # run is limited by max_iter, not an early eval cutoff.
                # NOTE: gradient stop is `gtol`, not `pgtol` -- pgtol is not a valid
                # L-BFGS-B option for scipy.minimize and was silently ignored before.
                res = minimize(_neg_marginal_l, param0, method="L-BFGS-B",
                               jac=_laplace_grad if _jac_ok else None,
                               bounds=bounds,
                               options={"maxiter": max_iter,
                                        "maxfun": max(200_000, max_iter * 8),
                                        "ftol": ftol, "gtol": gtol})
                if not np.isfinite(res.fun):
                    failed_starts.append(f"start {i} ditolak: objective non-finite")
                    continue
                _jac = getattr(res, "jac", None)
                try:
                    _gn0 = float(np.linalg.norm(_jac)) if _jac is not None else np.inf
                except Exception:
                    _gn0 = np.inf
                try:
                    _gvec0 = (np.asarray(_jac, dtype=np.float64).ravel().copy()
                              if _jac is not None else None)
                    if _gvec0 is not None and not np.all(np.isfinite(_gvec0)):
                        pass
                except Exception:
                    _gvec0 = None
                _nit0 = int(getattr(res, "nit", -1))
                _msg0 = str(getattr(res, "message", ""))
                if _nit0 == 0 and ((not np.isfinite(_gn0)) or _gn0 > 1e6
                                   or "abnormal" in _msg0.lower()):
                    failed_starts.append(
                        f"start {i} ditolak: abort iterasi 0 "
                        f"({_msg0.strip()} grad {_gn0:.3e})")
                    if float(res.fun) < _abort_val:
                        _abort_val = float(res.fun)
                        _abort_param = res.x.copy()
                        _abort_meta = {
                            "success": False,
                            "nit": _nit0,
                            "n_evals": int(getattr(res, "nfev", -1)),
                            "fun": float(res.fun),
                            "start": int(i),
                            "message": _msg0.strip(),
                            "grad_norm": _gn0,
                            "grad_vec": _gvec0,
                            "ftol": float(ftol),
                            "gtol": float(gtol),
                        }
                    continue
                if res.fun < best_val:
                    best_val = float(res.fun)
                    best_param = res.x
                    best_meta = {
                        "success": bool(getattr(res, "success", False)),
                        "nit": _nit0,
                        "n_evals": int(getattr(res, "nfev", -1)),
                        "fun": float(res.fun),
                        "start": int(i),
                        "message": _msg0.strip(),
                        "grad_norm": _gn0,
                        "grad_vec": _gvec0,
                        "ftol": float(ftol),
                        "gtol": float(gtol),
                    }
            except KeyboardInterrupt:
                print("interrupted: start dihentikan user, pakai titik terbaik sejauh ini")
                _interrupted = True
                break
            except Exception as e:  # noqa: BLE001
                failed_starts.append(f"start {i} gagal: {type(e).__name__}: {e}")
                continue
    except KeyboardInterrupt:
        print("interrupted: start dihentikan user, pakai titik terbaik sejauh ini")
        _interrupted = True

    # Semua start abort iterasi 0 dgn fun finite: pakai titik abort terbaik sbg
    # hasil INVALID (angka sama dgn perilaku lama, status jujur bukan max_iter).
    _from_abort = False
    if best_param is None and _abort_param is not None:
        best_param, best_val, best_meta = _abort_param, _abort_val, _abort_meta
        _from_abort = True

    if best_param is None:
        msg = ("semua start gagal/non-finite objective; tidak ada hasil optimasi. "
               + (" | ".join(failed_starts) if failed_starts else "tidak ada start valid."))
        nan_param = np.full(param_dim, np.nan, dtype=np.float64)
        nan_hess = np.full((param_dim, param_dim), np.nan, dtype=np.float64)
        return {"abort_payload": {
            "status": "invalid",
            "at_bound": False,
            "certificate": {
                "valid": False,
                "optimum_valid": False,
                "grad_norm": None,
                "projected_grad": None,
                "at_bound": False,
                "bound_params": [],
                "hessian_pd": None,
                "lambda_min": None,
                "ll_per_obs": None,
                "warnings": [msg],
            },
            "beta_hat": np.array(nan_param[:p], dtype=np.float64, copy=True),
            "logphi": float("nan"),
            "phi": float("nan"),
            "logs": np.array(nan_param[p + 1: p + 1 + ksd], dtype=np.float64, copy=True),
            "raw_dep": np.array(nan_param[p + 1 + ksd:], dtype=np.float64, copy=True),
            "b_hat": np.zeros(Z.shape[1], dtype=np.float64),
            "H_b_chol": None,
            "sigma_area": float("nan"),
            "sigma_time": float("nan"),
            "sigma_gamma": None,
            "ksd": ksd,
            "q_dep": q_dep,
            "log_lik": float("nan"),
            "AIC": float("nan"),
            "BIC": float("nan"),
            "k": param_dim,
            "n": n,
            "p": p,
            "converged": False,
            "degenerate": False,
            "optim": {
                "success": False,
                "nit": 0,
                "iters": 0,
                "n_evals": 0,
                "max_iter": int(max_iter),
                "fun": float("nan"),
                "start": -1,
                "n_starts": len(starts),
                "n_failed_starts": len(failed_starts),
                "failed_starts": list(failed_starts),
                "n_evals_total": _eval_count["n"],
                "n_dense_fallbacks": _eval_count["fb"],
                "jac": "analytic" if _jac_ok else "finite-diff",
                "message": msg,
                "grad_norm": float("nan"),
                "ftol": float(ftol),
                "gtol": float(gtol),
                "hessian": nan_hess,
                "lambda_min": float("nan"),
                "hessian_pd": None,
                "hessian_timeout": False,
                "interrupted": bool(_interrupted),
                "all_starts_abort": False,
            },
            "W": W,
            "X": X,
            "Z": Z,
            "des": des,
            "model": model,
            "y": y,
            "_boot_frame": data,
            "_boot_theta": np.array(nan_param, dtype=np.float64, copy=True),
        }
    }

    return (best_param, best_val, best_meta, starts, failed_starts,
            _interrupted, _from_abort)
