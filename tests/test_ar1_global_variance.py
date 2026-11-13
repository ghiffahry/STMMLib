"""Every free variance parameter must move the likelihood.

The global AR(1) layout used to expose sigma2_u while build_V and
build_G_time only read sigma2_eta. The optimizer therefore had an inert
parameter: logLik_REML did not change when sigma2_u changed, so M4 and M11
reported a variance that was never estimated. These tests pin the layout
and prove that each free parameter is identified.
"""

import numpy as np
import pandas as pd
import pytest

from stmmlib._backend import Backend
from stmmlib.gaussian.design import build_design_matrices
from stmmlib.gaussian.gls import logLik_REML
from stmmlib.gaussian.transforms import (
    get_theta_layout,
    make_model,
    prepare_panel_data,
)


def _panel():
    m, T = 6, 4
    n = m * T
    rng = np.random.RandomState(11)
    area = np.repeat(np.arange(m), T)
    time = np.tile(np.arange(T), m)
    area_effect = rng.normal(0, 0.4, m)[area]
    time_effect = np.zeros(n)
    for i in range(m):
        path = np.zeros(T)
        for j in range(1, T):
            path[j] = 0.4 * path[j - 1] + rng.normal(0, 0.3)
        time_effect[i * T:(i + 1) * T] = path
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + area_effect + time_effect + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x1": x, "area": area, "time": time})
    W = np.zeros((m, m))
    for i in range(m):
        W[i, (i + 1) % m] = W[(i + 1) % m, i] = 1.0
    W = W / W.sum(1, keepdims=True)
    return df, W


def _prepared(df, W):
    backend = Backend(gpu=False)
    prep = prepare_panel_data(df, "area", "time")
    prep["W"] = W
    prep.update(build_design_matrices(prep, backend))
    return prep, backend


@pytest.mark.parametrize("time_scope", ["global", "area_specific"])
def test_ar1_layout_uses_sigma2_eta(time_scope):
    model = make_model("sar", "ar1", "none", time_scope)
    layout = get_theta_layout(model)
    assert "sigma2_eta" in layout
    assert "sigma2_u" not in layout


def test_every_free_parameter_moves_the_likelihood():
    df, W = _panel()
    prep, backend = _prepared(df, W)
    model = make_model("sar", "ar1", "none", "global")
    y = df["y"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), df["x1"].to_numpy(dtype=float)])

    theta = {"sigma2_e": 0.3, "sigma2_v": 0.4, "rho": 0.2,
             "sigma2_eta": 0.2, "phi": 0.3}
    base = float(logLik_REML(theta, y, X, prep, model, backend))

    inert = []
    for name in get_theta_layout(model):
        bumped = dict(theta)
        if name.startswith("sigma2_"):
            bumped[name] *= 1.05
        else:
            bumped[name] = min(bumped[name] + 0.05, 0.9)
        moved = float(logLik_REML(bumped, y, X, prep, model, backend))
        if abs(moved - base) < 1e-8:
            inert.append(name)
    assert inert == [], f"free parameters with no effect on the likelihood: {inert}"
