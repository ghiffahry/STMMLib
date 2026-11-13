"""Parametric bootstrap MSE of the area-level predictor.

Functions: ``mse_ebp(fit, ...)`` for Beta, ``mse_eblup(fit, ...)`` for
Gaussian. The flat ``stmmlib.mse_bootstrap`` module is a compatibility shim.
"""
from stmmlib.mse.bootstrap import mse_ebp, mse_eblup

__all__ = [
    "mse_ebp",
    "mse_eblup",
]
