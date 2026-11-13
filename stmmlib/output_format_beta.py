"""Compatibility shim for ``stmmlib.output_format_beta``.

Use ``stmmlib.reporting.beta`` instead. Kept so existing imports keep working.
"""
from stmmlib.reporting.beta import (
    _bound_rows,
    _effect_row,
    _status_label,
    format_beta,
)

__all__ = [
    "format_beta",
]
