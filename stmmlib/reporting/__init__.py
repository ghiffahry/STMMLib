"""Output formatting for fitted models.

Public API: ``format_output``, ``format_beta``, ``format_normal``.
Implementation modules: ``stmmlib.reporting._common``, ``.beta``, ``.normal``.
The flat ``stmmlib.output_format*`` modules are compatibility shims and will
not grow.
"""
from stmmlib.reporting._common import (
    _ci,
    _fmt,
    _footer,
    _sig,
    fmt_model_fit,
    format_output,
)
from stmmlib.reporting.beta import format_beta
from stmmlib.reporting.normal import format_normal

__all__ = [
    "format_output",
    "format_beta",
    "format_normal",
]
