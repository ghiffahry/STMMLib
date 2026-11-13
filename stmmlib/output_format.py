"""Compatibility shim for ``stmmlib.output_format``.

Use ``stmmlib.reporting`` instead. Kept so existing imports keep working.
"""
from stmmlib.reporting._common import (
    _assess,
    _ci,
    _fmt,
    _footer,
    _sig,
    fmt_model_fit,
    format_output,
)

__all__ = [
    "format_output",
    "fmt_model_fit",
]
