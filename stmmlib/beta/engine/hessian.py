"""Finite-difference Hessian of the marginal negative log-likelihood."""

from __future__ import annotations

import numpy as np

from stmmlib.beta.numerics import _num_hessian, _num_hessian_grad


def final_hessian(cfg, compute_hessian, hess_eps, best_param, b_hat,
                  _jac_ok, _state, _neg_marginal_l, _laplace_grad):
    """Hessian at the optimum plus its curvature certificate.

    Analytic-jac models take the cheap path (central FD of the analytic
    gradient, 2k evals); the others keep objective FD (~2k^2). The returned
    hessian_pd is None when the Hessian was not computed or failed.
    """
    hessian = None
    lambda_min = None
    hessian_pd = None
    _hess_timeout = False
    _interrupted = False
    if compute_hessian:
        import time as _time
        _hess_t0 = _time.monotonic()
        _budget = cfg.hessian_time_budget
        _orig_neg = _neg_marginal_l

        def _hess_obj(v):
            if _budget is not None and _time.monotonic() - _hess_t0 > _budget:
                raise TimeoutError("hessian budget")
            return _orig_neg(v)

        def _hess_grad(v):
            # Cheap Hessian path (analytic-jac models): refresh the inner mode
            # and cache at the perturbed point through the same mechanism as
            # _neg_marginal_l, then evaluate the existing analytic gradient of
            # the negative objective. Same time budget as the objective path.
            va = np.asarray(v, dtype=np.float64)
            if _budget is not None and _time.monotonic() - _hess_t0 > _budget:
                raise TimeoutError("hessian budget")
            _orig_neg(va)
            return _laplace_grad(va)

        _state["hessian_probe"] = True
        try:
            if _jac_ok:
                hessian = _num_hessian_grad(_hess_grad, best_param, eps=hess_eps)
            else:
                hessian = _num_hessian(_hess_obj, best_param, eps=hess_eps)
            lam = np.linalg.eigvalsh(hessian)
            lambda_min = float(lam.min())
            hessian_pd = bool(lambda_min > 1e-6)
        except KeyboardInterrupt:
            print("interrupted: Hessian dibatalkan user; SE tidak tersedia")
            hessian, lambda_min, hessian_pd = None, None, None
            _interrupted = True
        except TimeoutError:
            hessian, lambda_min, hessian_pd = None, None, None
            _hess_timeout = True
        except Exception:  # noqa: BLE001
            hessian, lambda_min, hessian_pd = None, None, None
        finally:
            _state.pop("hessian_probe", None)
            # Restore final mode: FD Hessian probes mutate _state b and cache
            # across perturbed points; put back the best-param mode so the
            # returned b_hat always comes from this call own state.
            _state["b"] = b_hat.copy()
            _state["cache"] = {"x": best_param.copy(), "b": b_hat.copy()}
    return hessian, lambda_min, hessian_pd, _hess_timeout, _interrupted
