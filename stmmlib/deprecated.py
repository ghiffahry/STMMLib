"""Deprecated compatibility functions.

Provides stLMM(), stlmm5_fit(), and other legacy entry points
that map to the new API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import numpy as np


def stLMM(
    formula=None,
    data: pd.DataFrame = None,
    model: str = "m4_sar_ar1",
    family: str = "gaussian",
    W=None,
    method: str = "REML",
    gpu: str = "auto",
    **kwargs,
):
    """Deprecated: Use stmm() instead.

    Fits a spatio-temporal mixed model (Gaussian or Beta).
    """
    import warnings
    warnings.warn(
        "stLMM is deprecated. Use stmm() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from stmmlib.stmm import stmm
    return stmm(formula, data, model, family=family, W=W, method=method, gpu=gpu, **kwargs)


def stlmm5_fit(
    data: pd.DataFrame = None,
    y: np.ndarray = None,
    X: np.ndarray = None,
    model: str = "m4_sar_ar1",
    W=None,
    method: str = "REML",
    gpu: str = "auto",
    **kwargs,
):
    """Deprecated: Direct fit with pre-built matrices.

    Use stmm() for the standard interface.
    """
    import warnings
    warnings.warn(
        "stlmm5_fit is deprecated. Use stmm() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if data is None and y is not None:
        data = pd.DataFrame({"y": y})

    from stmmlib.stmm import stmm
    return stmm(data=data, model=model, W=W, method=method, gpu=gpu, **kwargs)
