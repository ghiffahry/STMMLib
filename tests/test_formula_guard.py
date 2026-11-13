"""Contract: ``formula`` is not implemented and must be rejected.

Silently ignoring a formula is a correctness trap: the user believes the
model contains only the terms in ``formula``, while the design matrix is
built from every non-skipped column.
"""

import numpy as np
import pandas as pd
import pytest

from stmmlib.stmm import stmm, stmm_beta, stmm_normal


@pytest.fixture
def panel():
    rng = np.random.default_rng(3)
    n = 9
    return pd.DataFrame({
        "y": rng.uniform(0.1, 0.9, n),
        "x1": rng.normal(size=n),
        "area": np.repeat(np.arange(3), 3),
        "time": np.tile(np.arange(3), 3),
    })


def test_stmm_gaussian_rejects_formula(panel):
    with pytest.raises(NotImplementedError, match="formula"):
        stmm(formula="y ~ x1", data=panel, model="m1_iid_iid",
             family="gaussian", gpu="no")


def test_stmm_beta_rejects_formula(panel):
    with pytest.raises(NotImplementedError, match="formula"):
        stmm(formula="y ~ x1", data=panel, model="m1_iid_iid",
             family="beta", gpu="no")


def test_stmm_normal_rejects_formula(panel):
    with pytest.raises(NotImplementedError, match="formula"):
        stmm_normal(formula="y ~ x1", data=panel, model="m1_iid_iid",
                    gpu="no")


def test_stmm_beta_entry_rejects_formula(panel):
    with pytest.raises(NotImplementedError, match="formula"):
        stmm_beta(formula="y ~ x1", data=panel, model="m1_iid_iid",
                  gpu="no")
