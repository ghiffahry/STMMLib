"""Shared fixtures: small panels for fast, deterministic tests.

All fits use gpu="no", n_starts=1 to keep the suite fast and CPU-only
(gaussian path is not GPU-portable).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def W_flat():
    m = 12
    W = np.ones((m, m))
    np.fill_diagonal(W, 0)
    return W / W.sum(1, keepdims=True)


@pytest.fixture(scope="session")
def df_flat(W_flat):
    """Flat balanced panel m=12, T=4 with a real area effect (beta response)."""
    m, T, n = 12, 4, 48
    rng = np.random.default_rng(7)
    area = np.repeat(np.arange(m), T)
    aef = rng.normal(0, 0.4, m)
    eta = -1.2 + 0.5 * rng.normal(size=n) + aef[area]
    y = rng.beta(20 * np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98), 20)
    return pd.DataFrame({
        "y": y, "x1": rng.normal(size=n),
        "area": area, "time": np.tile(np.arange(T), m),
    })


@pytest.fixture(scope="session")
def df_gauss():
    """Flat balanced panel with gaussian response."""
    m, T, n = 12, 4, 48
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "y": rng.normal(size=n), "x1": rng.normal(size=n),
        "area": np.repeat(np.arange(m), T), "time": np.tile(np.arange(T), m),
    })


@pytest.fixture(scope="session")
def df_nested():
    """Nested panel: 3 provinces x 4 kab each, T=4, prov+kab effects."""
    m, T, n = 12, 4, 48
    rng = np.random.default_rng(7)
    area = np.repeat(np.arange(m), T)
    prov = np.repeat(np.arange(m // 4), 4 * T)
    pef = rng.normal(0, 0.3, m // 4)[prov // 4]
    kef = rng.normal(0, 0.4, m)[area]
    eta = -1.2 + 0.5 * rng.normal(size=n) + pef + kef
    y = rng.beta(20 * np.clip(1 / (1 + np.exp(-eta)), 0.02, 0.98), 20)
    return pd.DataFrame({
        "y": y, "x1": rng.normal(size=n),
        "area": area, "time": np.tile(np.arange(T), m),
        "prov": prov, "kab": area,
    })


def _col_kwargs(data):
    kw = {"area_col": "area", "time_col": "time"}
    if "prov" in data.columns:
        kw["prov_col"] = "prov"
    if "kab" in data.columns:
        kw["kab_col"] = "kab"
    return kw


def fit_gauss(data, model, W=None, **kw):
    from stmmlib.stmm import stmm
    cols = _col_kwargs(data)
    cols.setdefault("n_starts", 1)
    cols.update(kw)
    return stmm(data=data, model=model, family="gaussian", W=W, gpu="no", **cols)


def fit_beta(data, model, W=None, **kw):
    from stmmlib.stmm import stmm
    cols = _col_kwargs(data)
    cols.setdefault("n_starts", 1)
    cols["max_iter"] = kw.pop("max_iter", 400)
    cols["inner_max_iter"] = 30
    cols.update(kw)
    return stmm(data=data, model=model, family="beta", W=W, gpu="no", **cols)
