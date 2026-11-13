"""Compatibility shim for ``stmmlib.mse_bootstrap``.

Use ``stmmlib.mse`` instead. ``beta_laplace_structural`` and
``refit_gaussian`` are re-exported for backwards compatibility with code
that monkeypatched through the old module path.
"""
from stmmlib.mse.bootstrap import (
    beta_laplace_structural,
    mse_ebp,
    mse_eblup,
    refit_gaussian,
)

__all__ = [
    "mse_ebp",
    "mse_eblup",
]
