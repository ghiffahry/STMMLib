"""Beta model structure registry: M1-M14 mapping.

Maps model IDs to their structural components: area type, time type,
time_scope, and interaction type for the Beta family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BetaModelSpec:
    """Structural specification for one Beta model."""

    model_id: str
    area: str
    time: str
    time_scope: str  # "area_specific" or "global"
    interaction: str  # "none", "iid", "spatial", "temporal", "st"
    label: str


BETA_MODELS: dict[str, BetaModelSpec] = {
    "m1_iid_iid": BetaModelSpec("m1_iid_iid", "iid", "iid", "area_specific", "none", "IID + IID"),
    "m2_iid_ar1": BetaModelSpec("m2_iid_ar1", "iid", "ar1", "area_specific", "none", "IID + AR(1)"),
    "m3_sar_iid": BetaModelSpec("m3_sar_iid", "sar", "iid", "area_specific", "none", "SAR + IID"),
    "m4_sar_ar1": BetaModelSpec("m4_sar_ar1", "sar", "ar1", "global", "none", "SAR + AR(1) [global]"),
    "m5_seblup_rao_yu_it": BetaModelSpec("m5_seblup_rao_yu_it", "seblup_rao_yu", "ar1", "area_specific", "none", "SEBLUP Rao-Yu IT"),
    "m6_nested_iid_iid": BetaModelSpec("m6_nested_iid_iid", "nested_iid", "iid", "area_specific", "none", "Nested IID + IID"),
    "m7_nested_ar1": BetaModelSpec("m7_nested_ar1", "nested_sar", "ar1", "area_specific", "none", "Nested IID + AR(1)"),
    "m8_iid_iid_int": BetaModelSpec("m8_iid_iid_int", "iid", "iid", "area_specific", "iid", "IID + IID + Type I"),
    "m9_sar_iid_int": BetaModelSpec("m9_sar_iid_int", "sar", "iid", "area_specific", "spatial", "SAR + IID + Type II"),
    "m10_iid_ar1_int": BetaModelSpec("m10_iid_ar1_int", "iid", "ar1", "area_specific", "temporal", "IID + AR(1) + Type III"),
    "m11_sar_ar1_int": BetaModelSpec("m11_sar_ar1_int", "sar", "ar1", "global", "st", "SAR + AR(1) + Type IV [global]"),
    "m12_seblup_rao_yu_it_int": BetaModelSpec("m12_seblup_rao_yu_it_int", "seblup_rao_yu", "ar1", "area_specific", "st", "SEBLUP Rao-Yu IT + Type IV"),
    "m13_nested_iid_iid_int": BetaModelSpec("m13_nested_iid_iid_int", "nested_iid", "iid", "area_specific", "temporal", "Nested IID + IID + Type III"),
    "m14_nested_ar1_int": BetaModelSpec("m14_nested_ar1_int", "nested_sar", "ar1", "area_specific", "iid", "Nested IID + AR(1) + Type I"),
    "beta_regression": BetaModelSpec("beta_regression", "iid", "iid", "area_specific", "none", "Beta Regression (fixed only)"),
}


def stmm_beta_structure(model: str) -> dict:
    """Get structural components for a Beta model.

    Args:
        model: Model ID string (e.g. "m1_iid_iid").

    Returns:
        dict with area, time, time_scope, interaction, label.
    """
    spec = BETA_MODELS.get(model)
    if spec is None:
        raise KeyError(f"Unknown Beta model: {model!r}. Available: {list(BETA_MODELS.keys())}")
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
