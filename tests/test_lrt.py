"""Boundary LRT (Self & Liang 1987) for variance components."""
import numpy as np
import pandas as pd
import pytest

from conftest import fit_beta
from stmmlib.beta.config import BetaFitConfig
from stmmlib.lrt import boundary_lrt, mixture_chi2_pvalue


def _panel(signal: float, seed: int = 7):
    m, T, reps = 12, 4, 2
    n = m * T * reps
    rng = np.random.default_rng(seed)
    area = np.repeat(np.arange(m), T * reps)
    time = np.tile(np.repeat(np.arange(T), reps), m)
    aef = rng.normal(0, 0.4, m)[area]
    sl = rng.normal(0, signal, m)[area]
    eta = -1.2 + 0.5 * rng.normal(size=n) + aef + sl * (time - T / 2)
    y = rng.beta(20 * np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98), 20)
    return pd.DataFrame({"y": y, "x1": rng.normal(size=n), "area": area,
                         "time": time, "prov": area // 4, "kab": area})


def test_mixture_pvalue_properties():
    # p in [0, 0.5]; mixture threshold D=2.706 (naive chi2_1 is conservative)
    assert mixture_chi2_pvalue(0.0) == 1.0
    assert mixture_chi2_pvalue(-1.0) == 1.0
    assert mixture_chi2_pvalue(2.706) == pytest.approx(0.05, abs=1e-3)
    assert mixture_chi2_pvalue(3.841) == pytest.approx(0.025, abs=1e-3)
    assert 0 <= mixture_chi2_pvalue(10.0) <= 0.5


def test_lrt_not_significant_without_signal():
    """No interaction signal -> LRT must NOT reject (D small, p_mix >= 0.05)."""
    df = _panel(0.0)
    f10 = fit_beta(df, "m10_iid_ar1_int", max_iter=1500, n_starts=2,
                   inner_max_iter=50, seed=42, config=BetaFitConfig())
    f2 = fit_beta(df, "m2_iid_ar1", max_iter=800, seed=42, config=BetaFitConfig())
    r = boundary_lrt(f10._opt["log_lik"], f2._opt["log_lik"])
    assert r["D_clamped"] >= 0
    assert r["p_value"] >= 0.05, r


def test_lrt_significant_with_strong_signal():
    """Strong temporal-interaction signal -> LRT rejects at p<0.01."""
    df = _panel(0.9)
    f10 = fit_beta(df, "m10_iid_ar1_int", max_iter=1500, n_starts=2,
                   inner_max_iter=50, seed=42, config=BetaFitConfig())
    f2 = fit_beta(df, "m2_iid_ar1", max_iter=800, seed=42, config=BetaFitConfig())
    r = boundary_lrt(f10._opt["log_lik"], f2._opt["log_lik"])
    assert r["p_value"] < 0.01, r
    # mixture p = naive/2 for D>0 (naive chi2_1 is the conservative reference)
    assert r["p_value"] == pytest.approx(r["p_value_naive_chi2"] / 2, rel=1e-9)
