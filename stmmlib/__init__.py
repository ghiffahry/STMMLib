"""stmmlib — Family-aware spatio-temporal mixed models.

Supports 14 model structures (M1-M14) for Gaussian and Beta families,
with optional CUDA/GPU acceleration via CuPy.
"""

from stmmlib.registry import (
    stmm_available_models,
    stmm_model_label,
    stmm_parse_model,
)
from stmmlib.stmm import stmm, stmm_normal, stmm_beta
from stmmlib.output_format import format_output
from stmmlib._backend import Backend

try:  # single source of truth for the version: the installed package metadata
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("stmmlib")
except Exception:  # source checkout without an install
    __version__ = "0.2.0"
__all__ = [
    "stmm",
    "stmm_normal",
    "stmm_beta",
    "format_output",
    "stmm_available_models",
    "stmm_model_label",
    "stmm_parse_model",
    "Backend",
]
