"""Compatibility shim for ``stmmlib.output_format_normal``.

Use ``stmmlib.reporting.normal`` instead. Kept so existing imports keep working.
"""
from stmmlib.reporting.normal import (
    _t_pvalues,
    format_normal,
)

__all__ = [
    "format_normal",
]
