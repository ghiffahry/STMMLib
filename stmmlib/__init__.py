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

__version__ = "0.1.0"
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
