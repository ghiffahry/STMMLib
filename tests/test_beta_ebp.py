"""EBP (Empirical Best Predictor) for the Beta mixed model.

Two properties are load-bearing:
  1. Computing H_b after convergence must not move any fit quantity
     (beta, logLik, phi, rho) by even one bit.
  2. EBP must be near the plug-in mode (small posterior variance) and its
     Monte Carlo error must shrink with the number of draws.
"""
import numpy as np
import pytest

from stmmlib.beta.config import BetaFitConfig
from tests.conftest import fit_beta


def _fit(df, model, W=None, posterior=True, **kw):
    # compute_posterior is orthogonal to config: the automatic weak-prior
    # refit still works (passing an explicit config would disable it).
    kw.setdefault("seed", 0)  # deterministic starts: no flaky aborts
    return fit_beta(df, model, W=W, compute_posterior=posterior,
                    compute_hessian=False, **kw)


def test_posterior_does_not_move_fit(df_flat, W_flat):
    plain = _fit(df_flat, "m3_sar_iid", W=W_flat, posterior=False)
    draw = _fit(df_flat, "m3_sar_iid", W=W_flat, posterior=True)

    assert draw._opt.get("H_b_chol") is not None
    assert plain._opt.get("H_b_chol") is None
    assert draw.logLik == plain.logLik
    assert np.array_equal(draw.beta, plain.beta)
    assert np.array_equal(draw._opt["b_hat"], plain._opt["b_hat"])
    assert draw.phi == plain.phi
    assert np.array_equal(draw._opt["logs"], plain._opt["logs"])
    assert np.array_equal(draw._opt["raw_dep"], plain._opt["raw_dep"])


def test_ebp_close_to_plugin_and_mc_error_shrinks(df_flat, W_flat):
    fit = _fit(df_flat, "m3_sar_iid", W=W_flat)
    mu = fit.mu_hat
    ebp, se = fit.ebp(n_draws=200, seed=1, return_se=True)
    assert ebp.shape == mu.shape
    assert np.all((ebp > 0) & (ebp < 1))
    assert np.max(np.abs(ebp - mu)) < 0.10      # mode is the posterior mean
    ebp2, se2 = fit.ebp(n_draws=2000, seed=1, return_se=True)
    assert np.mean(se2) < 0.6 * np.mean(se)     # ~sqrt(10) expected
    assert np.max(np.abs(ebp2 - ebp)) < 0.01


def test_ebp_requires_posterior_flag(df_flat):
    fit = _fit(df_flat, "m1_iid_iid", posterior=False)
    with pytest.raises(ValueError, match="compute_posterior"):
        fit.ebp()


def test_ebp_newdata_same_area_future_year(df_flat):
    train = df_flat[df_flat["time"] < 3].copy()
    test = df_flat[df_flat["time"] == 3].copy()
    fit = _fit(train, "m2_iid_ar1")

    mu, se = fit.ebp_newdata(test, n_draws=200, seed=2)
    assert mu.shape == (len(test),)
    assert np.all(np.isfinite(mu)) and np.all((mu > 0) & (mu < 1))
    assert np.all(np.isfinite(se)) and np.all(se > 0)

    # Fixed part only is the same latent location without the random effects;
    # the two must stay on the same scale (no silent unit/scale mistake).
    from scipy.special import expit
    X_raw = np.column_stack([np.ones(len(test)),
                             test["x1"].to_numpy(dtype=float)])
    fixed = expit(X_raw @ fit.beta_original)
    assert np.all(np.abs(mu - fixed) < 0.25)


def test_ebp_newdata_rejects_unseen_area(df_flat):
    train = df_flat[df_flat["area"] < 11].copy()
    fit = _fit(train, "m1_iid_iid")
    test = df_flat[df_flat["area"] == 11].copy()
    with pytest.raises(ValueError, match="not in training"):
        fit.ebp_newdata(test)
