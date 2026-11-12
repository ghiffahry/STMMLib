"""Bug-fix regression tests: B1 eig-bound, B2 delegation, B3 phi, B4 W-priority."""
import numpy as np

from conftest import fit_beta
from stmmlib.beta.criteria import beta_criteria
from stmmlib.beta.precision import effective_phi, rho_bound
from stmmlib._utils import row_standardize


def test_b1_rho_bound_asymmetric_W():
    """eigvals (spectral radius), NOT eigvalsh: KNN row-std W is asymmetric."""
    from stmmlib._backend import Backend
    m = 20
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, (m, 2))
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    W = np.zeros((m, m))
    for i in range(m):
        W[i, np.argsort(D[i])[:4]] = 1.0
    W = row_standardize(W, safe=True)
    assert not np.allclose(W, W.T)          # asymmetric input confirmed
    rb = rho_bound(W, Backend(gpu=False))    # cpu backend
    # row-stochastic => spectral radius exactly 1 => bound exactly 0.95
    assert abs(rb - 0.95) < 1e-12, rb


def test_b3_effective_phi_double_squash():
    raw = 1.0
    assert abs(effective_phi(raw) - 0.95 * np.tanh(0.95 * np.tanh(raw))) < 1e-15
    assert effective_phi(0.0) == 0.0


def test_b2_report_delegates_to_criteria(df_flat):
    from stmmlib.beta.report import evaluate
    r = fit_beta(df_flat, "m1_iid_iid", max_iter=200)
    ev = evaluate(r, criteria=None)          # force the library fallback path
    cr = beta_criteria(r)
    # fallback must agree with the library implementation (no garbage sign)
    assert np.isfinite(ev["cAIC"]) and np.isfinite(cr["cAIC"])
    assert abs(ev["cAIC"] - cr["cAIC"]) < 1e-6
    assert cr["ll_cond"] > r._opt["log_lik"]


def test_b4_criteria_explicit_W_wins(df_flat, W_flat):
    r = fit_beta(df_flat, "m3_sar_iid", W_flat, max_iter=200)
    cr1 = beta_criteria(r, W=W_flat)          # explicit
    cr2 = beta_criteria(r, W=None)            # fall back to opt["W"]
    a, b = cr1["cAIC"], cr2["cAIC"]
    assert np.isfinite(a) and np.isfinite(b)
    assert abs(a - b) < 1e-6                  # same W -> same number
