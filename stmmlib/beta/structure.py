"""Beta model structure: one source of truth for the model list.

The model list lives in `stmmlib/registry.py`. This module only adds the
Beta-specific reading of it, so the fitter and the public API can never
disagree about which models exist.
"""

from __future__ import annotations

from stmmlib.registry import MODELS

KNOWN_AREA = ("iid", "sar", "nested_iid", "nested_sar", "seblup_rao_yu")
KNOWN_TIME = ("iid", "ar1")


def stmm_beta_structure(model: str) -> dict:
    """Get structural components for a Beta model.

    Args:
        model: Model ID string (e.g. "m1_iid_iid").

    Returns:
        dict with area, time, time_scope, interaction, label.
    """
    spec = MODELS.get(model)
    if spec is None:
        raise KeyError(
            f"Unknown Beta model: {model!r}. Available: {list(MODELS.keys())}")
    if spec.area not in KNOWN_AREA:
        raise ValueError(
            f"model {model!r}: area type {spec.area!r} tidak didukung "
            f"untuk family Beta; tipe yang dikenal: {list(KNOWN_AREA)}")
    if spec.time not in KNOWN_TIME:
        raise ValueError(
            f"model {model!r}: time type {spec.time!r} tidak didukung "
            f"untuk family Beta; tipe yang dikenal: {list(KNOWN_TIME)}")
    return {
        "area": spec.area,
        "time": spec.time,
        "time_scope": spec.time_scope,
        "interaction": spec.interaction,
        "label": spec.label,
    }


def stmm_beta_nraw_interaction(interaction: str) -> int:
    """Count of independent raw dependence parameters for interaction type.

    Args:
        interaction: Interaction type string.

    Returns:
        Number of raw params: iid→0, spatial→1, temporal→1, st→2.
    """
    mapping = {"none": 0, "iid": 0, "spatial": 1, "temporal": 1, "st": 2}
    return mapping.get(interaction, 0)
