"""Laplace engine split into single-purpose modules.

solvers.beta_laplace_structural builds the fit context and calls these in
order: run_hooks -> optimize_starts -> build_payload.
"""

from stmmlib.beta.engine.finalize import build_payload
from stmmlib.beta.engine.gradient import laplace_grad
from stmmlib.beta.engine.hooks import run_hooks
from stmmlib.beta.engine.likelihood import neg_marginal_l
from stmmlib.beta.engine.optimize import optimize_starts

__all__ = [
    "neg_marginal_l",
    "laplace_grad",
    "run_hooks",
    "optimize_starts",
    "build_payload",
]
