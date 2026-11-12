"""E2: unbalanced/replication coverage across models (M3-M5, M8) + gaussian."""
import numpy as np
import pandas as pd

from conftest import fit_beta, fit_gauss


def _panel(seed=7, reps=2, drop_frac=0.0):
    """Balanced-with-replication panel; optionally drop random rows."""
    m, T = 12, 4
    n = m * T * reps
    rng = np.random.default_rng(seed)
    area = np.repeat(np.arange(m), T * reps)
    time = np.tile(np.repeat(np.arange(T), reps), m)
    eta = -1.2 + 0.5 * rng.normal(size=n) + 0.4 * rng.normal(0, 1, m)[area]
    y = rng.beta(20 * np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98), 20)
    df = pd.DataFrame({"y": y, "x1": rng.normal(size=n), "area": area, "time": time})
    if drop_frac > 0:
        idx = rng.choice(n, size=int(n * drop_frac), replace=False)
        df = df.drop(index=idx).reset_index(drop=True)
    return df


def test_replicated_panel_m3_m5_m8(W_flat):
    """2 obs per cell (n=96, 48 cells) across SAR/AR1/SEBLUP/interaction."""
    df = _panel(reps=2)
    for mid, w in [("m3_sar_iid", W_flat), ("m4_sar_ar1", W_flat),
                   ("m5_seblup_rao_yu_it", W_flat), ("m8_iid_iid_int", None)]:
        r = fit_beta(df, mid, W_flat if w is not None else None, max_iter=800)
        assert np.isfinite(r._opt["log_lik"]), mid
        assert r._opt["sigma_area"] > 0, mid


def test_unbalanced_replicated_m3_m8(W_flat):
    """Drop 20% rows from a replicated panel -> unbalanced, still fits."""
    df = _panel(reps=2, drop_frac=0.2)
    assert len(df) < 96
    for mid, w in [("m3_sar_iid", W_flat), ("m8_iid_iid_int", None)]:
        r = fit_beta(df, mid, W_flat if w is not None else None, max_iter=800)
        assert np.isfinite(r._opt["log_lik"]), mid


def test_gaussian_unbalanced_replicated(W_flat):
    """Gaussian M3/M4/M5 on replicated + unbalanced panel."""
    df = _panel(reps=2, drop_frac=0.2)
    for mid, w in [("m3_sar_iid", W_flat), ("m4_sar_ar1", W_flat),
                   ("m5_seblup_rao_yu_it", W_flat)]:
        r = fit_gauss(df, mid, W_flat if w is not None else None)
        assert np.isfinite(r["log_lik"]), mid
        assert r.get("converged", False), mid
